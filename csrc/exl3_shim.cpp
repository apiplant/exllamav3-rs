// C-ABI bridge over the ExLlamaV3 kernels. Each `void*` is a `tch` tensor handle,
// which is an `at::Tensor*` (torch-sys: `typedef at::Tensor* tensor`). We never
// own them; callers keep the tch::Tensor alive for the duration of the call.

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>

#include <cstdio>
#include <exception>

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "norm.cuh"
#include "rope.cuh"
#include "hgemm.cuh"
#include "attention.cuh"
#include "activation.cuh"
#include "softcap.cuh"
#include "add.cuh"
#include "quant/exl3_gemm.cuh"
#include "quant/hadamard.cuh"
#include "quant/reconstruct.cuh"
#include "generator/cache.cuh"
#include "generator/rep_pen.cuh"
#include "generator/sampling_basic.cuh"
#include "generator/gumbel.cuh"
#include "cache/q_cache.cuh"
#include "gdn.cuh"

static inline at::Tensor& T(void* p) { return *reinterpret_cast<at::Tensor*>(p); }
static inline c10::optional<at::Tensor> OT(void* p) {
    return p ? c10::optional<at::Tensor>(T(p)) : c10::nullopt;
}

extern "C" {

// LinearEXL3 forward: C = had(A)·trellis(B) with input/output scale-flips folded in.
// A: (m,k) fp16, B: (k/16,n/16,16*K) int16, C: (m,n) fp16/fp32, A_had: scratch like A.
int exl3_shim_gemm(void* A, void* B, void* C, void* suh, void* A_had, void* svh,
                   bool mcg, bool mul1) {
    // force_num_sms MUST be 0 (= "use the device SM count"), not -1: the kernel
    // does `num_sms = force_num_sms ? force_num_sms : get_num_sms()`, so any
    // non-zero value is taken literally. -1 then collapses the cooperative-GEMM
    // grid to a single SM (`MAX(MIN(max_slices, num_sms), 1)`), running m=1
    // decode on 1/128th of the GPU (~14x slower). Matches upstream's `run_gr`.
    return exl3_gemm(T(A), T(B), T(C), OT(suh), OT(A_had), OT(svh),
                     /*force_shape_idx*/ -1, mcg, mul1, /*force_num_sms*/ 0);
}

// Multi-GEMM over a pointer list of EXL3 weights — `quant/exl3_gemm.cu`.
// One launch covers every expert a token selected, with the selection read from
// the device (`indices`), so a MoE decode step needs no routing readback and
// stays capturable. `weights`, when given, scales each expert's contribution and
// the kernel reduces them into row 0 of C.
//
// A: (bszm, m, k) fp16. B/suh/svh: (num_matrices,) int64 data pointers.
// C: (bszm, m, n) fp16 or fp32. A_had: scratch holding bszm * m * k halves —
// the kernel writes one transformed slab per matrix, and an undersized buffer is
// silent corruption rather than an error.
int exl3_shim_mgemm(void* A, void* B, void* C, void* suh, void* A_had, void* svh,
                    void* indices, void* weights, int K, bool mcg, bool mul1,
                    int num_tokens) {
    return exl3_mgemm(T(A), T(B), T(C), T(suh), T(A_had), T(svh),
                      OT(indices), OT(weights), K,
                      /*force_shape_idx*/ -1, mcg, mul1,
                      /*min_index*/ -1, /*max_index*/ -1,
                      /*force_num_sms*/ 0, num_tokens,
                      /*size_n_list*/ {}, /*c_ptrs*/ {});
}

void exl3_shim_rms_norm(void* x, void* w, void* y, float eps,
                        float constant_bias, float constant_scale) {
    rms_norm(T(x), OT(w), T(y), eps, constant_bias, constant_scale,
             /*span_heads*/ false, /*add_residual*/ false);
}

// Fused Q/K RMSNorm (optional) + RoPE, matching modules/attn.py's rope.apply().
// If `positions` (int32 [bsz], device) is non-null it overrides the scalar
// `position`: token t of seq b gets rotary position `t + positions[b]`. One
// captured graph then serves any decode offset (the pointer is stable, the
// value on device changes between replays).
void exl3_shim_rope(void* q, void* out_q, void* k, void* out_k, void* inv_freq,
                    uint32_t position, void* positions, void* q_norm, void* k_norm,
                    float norm_eps, float norm_constant_bias,
                    int rope_mode, float attn_factor) {
    c10::optional<at::Tensor> ok = OT(out_k);
    rope(T(q), T(out_q), OT(k), ok, T(inv_freq), position,
         /*positions*/ OT(positions), /*position_ids*/ c10::nullopt,
         rope_mode, attn_factor, OT(q_norm), OT(k_norm),
         norm_eps, norm_constant_bias,
         /*llama_4_scaling_beta*/ 0.0f, /*llama_4_scaling_original*/ 1,
         /*rotate_dims*/ 1, /*rotate_offset*/ 0);
}

void exl3_shim_hgemm(void* a, void* b, void* c) { hgemm(T(a), T(b), T(c)); }

// Non-cached GQA attention (modules/attn.py flash_attn_nc path, dispatch fallback).
void exl3_shim_bighead_attn(void* q, void* k, void* v, void* o,
                            int kv_chunk_size, bool causal, float sm_scale) {
    bighead_attn(T(q), T(k), T(v), T(o), kv_chunk_size, causal, sm_scale);
}

// Append new K/V (bsz,S,H,D fp16) into the paged cache at offset cache_seqlens[b].
void exl3_shim_paged_kv_cache_update(void* k, void* v, void* k_cache, void* v_cache,
                                     void* block_table, void* cache_seqlens) {
    paged_kv_cache_update(T(k), T(v), T(k_cache), T(v_cache),
                          T(block_table), T(cache_seqlens));
}

// Paged GQA attention. total_k_len = cache_seqlens[b] + q_len, so cache_seqlens
// is the pre-append length. Bottom-right causal masking as in bighead_attn.
void exl3_shim_bighead_attn_paged(void* q, void* k, void* v, void* k_cache, void* v_cache,
                                  void* block_table, void* cache_seqlens, void* o,
                                  int kv_chunk_size, bool causal, float sm_scale) {
    bighead_attn_paged(T(q), T(k), T(v), T(k_cache), T(v_cache),
                       T(block_table), T(cache_seqlens), T(o),
                       kv_chunk_size, causal, sm_scale);
}
// Paged GQA attention against a QUANTIZED KV cache (qk/sk/qv/sv). Fresh fp16 rows
// k/v are quantized in first; short-q decode dequantizes on the fly inside the
// kernel (no fp16 cache-sized scratch), long-q prefill via a compact window.
void exl3_shim_bighead_attn_paged_q(void* q, void* k, void* v,
                                    void* qk, void* sk, void* qv, void* sv,
                                    void* block_table, void* cache_seqlens, void* o,
                                    int kv_chunk_size, bool causal, float sm_scale,
                                    float compand_a) {
    bighead_attn_paged_q(T(q), T(k), T(v), T(qk), T(sk), T(qv), T(sv),
                         T(block_table), T(cache_seqlens), T(o),
                         kv_chunk_size, causal, sm_scale, compand_a);
}

// --- elementwise / norm / activation ------------------------------------------

void exl3_shim_rms_norm_res_in(void* x, void* w, void* y, void* r, float eps,
                               float constant_bias, float constant_scale) {
    rms_norm_res_in(T(x), OT(w), T(y), T(r), eps, constant_bias, constant_scale);
}
void exl3_shim_gated_rms_norm(void* x, void* w, void* y, void* g, float eps,
                              float constant_bias, int w_groups, bool gate_first,
                              int gate_act) {
    gated_rms_norm(T(x), T(w), T(y), T(g), eps, constant_bias, w_groups, gate_first, gate_act);
}
void exl3_shim_softcap(void* x, void* y, float factor) { softcap(T(x), T(y), factor); }
void exl3_shim_add(void* x, void* y, void* z) { add(T(x), T(y), T(z)); }

void exl3_shim_silu_mul(void* x, void* y, void* z, float act_limit) { silu_mul(T(x), T(y), T(z), act_limit); }
void exl3_shim_gelu_mul(void* x, void* y, void* z, float act_limit) { gelu_mul(T(x), T(y), T(z), act_limit); }
void exl3_shim_relu2_mul(void* x, void* y, void* z, float act_limit) { relu2_mul(T(x), T(y), T(z), act_limit); }

// Random-Hadamard transform of a [rows, k] (k % 128 == 0) half tensor, with an
// optional per-column pre-scale (folded before) or post-scale (folded after).
// Used for the reconstruct+hgemm prefill path (EXL3 Linear at rows > 144).
void exl3_shim_had_r_128(void* x, void* out, void* pre_scale, void* post_scale, float scale) {
    had_r_128(T(x), T(out), OT(pre_scale), OT(post_scale), scale);
}
// Dequantize a packed EXL3 trellis tensor into a dense [k, n] half weight.
void exl3_shim_reconstruct(void* unpacked, void* packed, int k, bool mcg, bool mul1) {
    reconstruct(T(unpacked), T(packed), k, mcg, mul1);
}

// x *= sigmoid(y), in place (interleaved / full output gate for attention).
void exl3_shim_mul_sigmoid_(void* x, void* y) { mul_sigmoid_(T(x), T(y)); }
// Split an interleaved [.., Nq, 2*hd] projection into q [.., Nq, hd] and g [.., Nq*hd].
void exl3_shim_deinterleave_qg(void* qg, void* q, void* g, int head_dim) {
    deinterleave_qg(T(qg), T(q), T(g), head_dim);
}

// --- gated-delta-net (Qwen3.5 linear attention) ------------------------------

// Split projections path: mixed_qkv[b,f,s] bf16 <- bf16(qkv[b,s,f]); beta/g from b,a.
void exl3_shim_gdn_fused_op_2(void* b, void* a, void* dt_bias, void* a_log,
                              void* beta, void* g, float beta_scale) {
    gated_delta_net_fused_op_2(T(b), T(a), T(dt_bias), T(a_log), T(beta), T(g), beta_scale);
}
// Sequential gated delta rule over the whole seqlen (prefill and decode both);
// folds new K/V into `recurrent_state[slot, 0]`. slots optional ([bsz] int32).
void exl3_shim_recurrent_gated_delta_rule(void* mixed_qkv, void* g, void* beta,
                                          void* recurrent_state, void* core_attn_out,
                                          int num_k_heads, int num_v_heads,
                                          int k_head_dim, int v_head_dim,
                                          void* slots, bool history) {
    cuda_recurrent_gated_delta_rule(T(mixed_qkv), T(g), T(beta), T(recurrent_state),
                                    T(core_attn_out), num_k_heads, num_v_heads,
                                    k_head_dim, v_head_dim, OT(slots), history);
}
extern "C" void gdn_chunk_fused(const at::Tensor& w, const at::Tensor& q, const at::Tensor& k,
                                const at::Tensor& qkt, const at::Tensor& uv,
                                const at::Tensor& alog, const at::Tensor& h0,
                                at::Tensor& out, at::Tensor& ht, double scale);
void exl3_shim_gdn_chunk_fused(void* w, void* q, void* k, void* qkt, void* uv, void* alog,
                               void* h0, void* out, void* ht, double scale) {
    at::Tensor o = T(out), h = T(ht);
    gdn_chunk_fused(T(w), T(q), T(k), T(qkt), T(uv), T(alog), T(h0), o, h, scale);
}

extern "C" void gdn_chunk_wy(const at::Tensor& k, const at::Tensor& v, const at::Tensor& beta,
                             const at::Tensor& a_log, at::Tensor& w, at::Tensor& uv);
void exl3_shim_gdn_chunk_wy(void* k, void* v, void* beta, void* a_log, void* w, void* uv) {
    at::Tensor ww = T(w), uu = T(uv);
    gdn_chunk_wy(T(k), T(v), T(beta), T(a_log), ww, uu);
}

// Causal depthwise conv1d + SiLU with rolling conv_state. x/out [bsz,dim,S] bf16.
void exl3_shim_causal_conv1d_update(void* x, void* conv_state, void* slots,
                                    void* weight, void* bias, void* out,
                                    bool activation, bool history) {
    cuda_causal_conv1d_update(T(x), T(conv_state), OT(slots), T(weight), OT(bias),
                              T(out), activation, history);
}

// --- sampler kernels ----------------------------------------------------------
// in-place-safe: in_logits and out_logits may alias.

void exl3_shim_apply_rep_pens(void* in_logits, void* out_logits, void* past_ids,
                              float rep_p, int sustain_range, int decay_range) {
    apply_rep_pens(T(in_logits), T(out_logits), T(past_ids), rep_p, sustain_range, decay_range);
}
void exl3_shim_apply_pres_freq_pens(void* in_logits, void* out_logits, void* past_ids,
                                    float pres_p, float freq_p, int sustain_range, int decay_range) {
    apply_pres_freq_pens(T(in_logits), T(out_logits), T(past_ids), pres_p, freq_p, sustain_range, decay_range);
}
// argmax / Gumbel-noise categorical draw. ids: int32/int64 [rows]. max_logit != 0
// clamps the search range. random is the per-call PRNG seed (upstream: uint32).
void exl3_shim_argmax_sample(void* logits, void* ids, int max_logit) {
    argmax_sample(T(logits), T(ids), max_logit);
}
void exl3_shim_gumbel_sample(void* logits, void* ids, int max_logit, uint32_t random) {
    gumbel_sample(T(logits), T(ids), max_logit, random);
}
void exl3_shim_gumbel_noise_f32(void* logits_in, void* logits_out, uint32_t random) {
    gumbel_noise_f32(T(logits_in), T(logits_out), random);
}
void exl3_shim_cache_rotate(void* cache, void* order, void* temp) {
    cache_rotate(T(cache), T(order), T(temp));
}

// --- KV cache quantization --------------------------------------------------
// Quantize fresh K/V rows (bsz,S,n_kv,hd fp16) straight into the paged quantized
// cache (qk/qv int32 [pages,256,dim/32*bits], sk/sv fp16 [pages,256,dim/32]) at
// cache_seqlens..+seq_len. `in_contiguous` = the K/V rows are packed (bsz,S,·).
void exl3_shim_quant_cache_paged(void* k_in, void* k_out, void* k_out_scales,
                                 void* v_in, void* v_out, void* v_out_scales,
                                 void* cache_seqlens, void* block_table,
                                 int page_size, int seq_len, float compand_a,
                                 bool in_contiguous) {
    quant_cache_paged(T(k_in), T(k_out), T(k_out_scales),
                      T(v_in), T(v_out), T(v_out_scales),
                      T(cache_seqlens), T(block_table),
                      page_size, seq_len, compand_a, in_contiguous);
}
// Dequantize the stored prefix [0, cache_seqlens) into fp16 scratch pools
// (k_out/v_out, same [pages,256,n_kv,hd] layout as the plain fp16 cache).
void exl3_shim_dequant_cache_paged(void* k_in, void* k_in_scales, void* k_out,
                                   void* v_in, void* v_in_scales, void* v_out,
                                   void* cache_seqlens, void* block_table,
                                   int page_size, int sliding_window, float compand_a) {
    dequant_cache_paged(T(k_in), T(k_in_scales), T(k_out),
                        T(v_in), T(v_in_scales), T(v_out),
                        T(cache_seqlens), T(block_table),
                        page_size, sliding_window, compand_a);
}
// As above, but writes the referenced window COMPACTLY: block-table page p of
// row b lands at scratch page (b*pages_per_seq + p), so the scratch only needs
// to span the block table (bsz*pages_per_seq pages) not the whole pool.
// Attention then runs with an identity block table. `bonus_len` rows past
// cache_seqlens are included (a pre-appended prefill chunk).
void exl3_shim_dequant_cache_paged_window(void* k_in, void* k_in_scales, void* k_out,
                                          void* v_in, void* v_in_scales, void* v_out,
                                          void* cache_seqlens, void* block_table,
                                          int page_size, int bonus_len, float compand_a) {
    dequant_cache_paged_window(T(k_in), T(k_in_scales), T(k_out),
                               T(v_in), T(v_in_scales), T(v_out),
                               T(cache_seqlens), T(block_table),
                               page_size, bonus_len, compand_a);
}

// --- CUDA graph capture/replay of the decode step -------------------------------
// Raw at::cuda::CUDAGraph. Capture must run on a non-default stream; we switch tch
// onto a pooled side stream at capture_begin and stay there (the caller runs the
// whole generation loop on it), so replay + surrounding ops share one stream.

static c10::optional<at::cuda::CUDAStream> g_side_stream;

void* exl3_graph_new() {
    try { return new at::cuda::CUDAGraph(); } catch (...) { return nullptr; }
}
void  exl3_graph_free(void* g) { delete reinterpret_cast<at::cuda::CUDAGraph*>(g); }

// Return CUDA-graph private-mempool blocks (released by ~CUDAGraph but still
// reserved by the caching allocator) to the driver. Call after freeing a graph
// that will not be recaptured at the same shape, or memory grows unbounded as
// decode graphs are recaptured for longer contexts.
// Defined in kernels/attention.cu — force the tensor-core attention path on or
// off for this process, overriding EXL3_MMA_ATTN. Used by `bin/attn_check`.
extern "C" void exl3_set_mma_attn(int on);
extern "C" void exl3_shim_set_mma_attn(int on) { exl3_set_mma_attn(on); }

void exl3_empty_cache() {
    try { c10::cuda::CUDACachingAllocator::emptyCache(); } catch (...) {}
}

// Free VRAM in MiB (driver view), for leak debugging.
long exl3_cuda_free_mib() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return -1;
    return (long)(f / (1024 * 1024));
}

int exl3_graph_capture_begin(void* g) {
    try {
        int dev = c10::cuda::current_device();
        if (!g_side_stream)
            g_side_stream = at::cuda::getStreamFromPool(/*isHighPriority*/ false, dev);
        at::cuda::setCurrentCUDAStream(*g_side_stream);
        reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_begin();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "cuda graph capture_begin failed: %s\n", e.what());
        return 1;
    } catch (...) { return 1; }
}
int exl3_graph_capture_end(void* g) {
    try {
        reinterpret_cast<at::cuda::CUDAGraph*>(g)->capture_end();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "cuda graph capture_end failed: %s\n", e.what());
        return 1;
    } catch (...) { return 1; }
}
int exl3_graph_replay(void* g) {
    try {
        reinterpret_cast<at::cuda::CUDAGraph*>(g)->replay();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "cuda graph replay failed: %s\n", e.what());
        return 1;
    } catch (...) { return 1; }
}
// Move tch onto the side stream without capturing (used for warmup so the caching
// allocator / autotuner state is primed on the stream capture will use).
void exl3_use_side_stream() {
    int dev = c10::cuda::current_device();
    if (!g_side_stream)
        g_side_stream = at::cuda::getStreamFromPool(false, dev);
    at::cuda::setCurrentCUDAStream(*g_side_stream);
}
void exl3_sync_side_stream() {
    if (g_side_stream) g_side_stream->synchronize();
}

} // extern "C"
