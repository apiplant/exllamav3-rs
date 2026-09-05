//! Raw bindings to `csrc/exl3_shim.cpp` and thin safe wrappers.
//!
//! Every wrapper takes already-contiguous CUDA tensors of the dtype the kernel
//! expects (fp16 unless noted). Shapes follow the upstream kernel doc-comments.

use std::os::raw::c_void;
use tch::Tensor;

fn p(t: &Tensor) -> *mut c_void {
    t.as_ptr() as *mut c_void
}
fn op(t: Option<&Tensor>) -> *mut c_void {
    t.map_or(std::ptr::null_mut(), p)
}

#[allow(non_snake_case)]
extern "C" {
    fn exl3_shim_gemm(
        A: *mut c_void,
        B: *mut c_void,
        C: *mut c_void,
        suh: *mut c_void,
        A_had: *mut c_void,
        svh: *mut c_void,
        mcg: bool,
        mul1: bool,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn exl3_shim_mgemm(
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
        suh: *mut c_void,
        a_had: *mut c_void,
        svh: *mut c_void,
        indices: *mut c_void,
        weights: *mut c_void,
        k: i32,
        mcg: bool,
        mul1: bool,
        num_tokens: i32,
    ) -> i32;
    fn exl3_shim_rms_norm(
        x: *mut c_void,
        w: *mut c_void,
        y: *mut c_void,
        eps: f32,
        constant_bias: f32,
        constant_scale: f32,
    );
    fn exl3_shim_rope(
        q: *mut c_void,
        out_q: *mut c_void,
        k: *mut c_void,
        out_k: *mut c_void,
        inv_freq: *mut c_void,
        position: u32,
        positions: *mut c_void,
        q_norm: *mut c_void,
        k_norm: *mut c_void,
        norm_eps: f32,
        norm_constant_bias: f32,
        rope_mode: i32,
        attn_factor: f32,
    );
    fn exl3_shim_hgemm(a: *mut c_void, b: *mut c_void, c: *mut c_void);
    fn exl3_shim_bighead_attn(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        o: *mut c_void,
        kv_chunk_size: i32,
        causal: bool,
        sm_scale: f32,
    );
    fn exl3_shim_paged_kv_cache_update(
        k: *mut c_void,
        v: *mut c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        block_table: *mut c_void,
        cache_seqlens: *mut c_void,
    );
    fn exl3_shim_bighead_attn_paged(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        block_table: *mut c_void,
        cache_seqlens: *mut c_void,
        o: *mut c_void,
        kv_chunk_size: i32,
        causal: bool,
        sm_scale: f32,
    );
    #[allow(clippy::too_many_arguments)]
    fn exl3_shim_bighead_attn_paged_q(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        qk: *mut c_void,
        sk: *mut c_void,
        qv: *mut c_void,
        sv: *mut c_void,
        block_table: *mut c_void,
        cache_seqlens: *mut c_void,
        o: *mut c_void,
        kv_chunk_size: i32,
        causal: bool,
        sm_scale: f32,
        compand_a: f32,
    );
    fn exl3_shim_rms_norm_res_in(
        x: *mut c_void, w: *mut c_void, y: *mut c_void, r: *mut c_void,
        eps: f32, constant_bias: f32, constant_scale: f32,
    );
    fn exl3_shim_gated_rms_norm(
        x: *mut c_void, w: *mut c_void, y: *mut c_void, g: *mut c_void,
        eps: f32, constant_bias: f32, w_groups: i32, gate_first: bool,
        gate_act: i32,
    );
    fn exl3_shim_softcap(x: *mut c_void, y: *mut c_void, factor: f32);
    fn exl3_shim_add(x: *mut c_void, y: *mut c_void, z: *mut c_void);
    fn exl3_shim_silu_mul(x: *mut c_void, y: *mut c_void, z: *mut c_void, act_limit: f32);
    fn exl3_shim_gelu_mul(x: *mut c_void, y: *mut c_void, z: *mut c_void, act_limit: f32);
    fn exl3_shim_relu2_mul(x: *mut c_void, y: *mut c_void, z: *mut c_void, act_limit: f32);
    fn exl3_shim_apply_rep_pens(
        in_logits: *mut c_void, out_logits: *mut c_void, past_ids: *mut c_void,
        rep_p: f32, sustain_range: i32, decay_range: i32,
    );
    fn exl3_shim_apply_pres_freq_pens(
        in_logits: *mut c_void, out_logits: *mut c_void, past_ids: *mut c_void,
        pres_p: f32, freq_p: f32, sustain_range: i32, decay_range: i32,
    );
    fn exl3_shim_argmax_sample(logits: *mut c_void, ids: *mut c_void, max_logit: i32);
    fn exl3_shim_gumbel_sample(logits: *mut c_void, ids: *mut c_void, max_logit: i32, random: u32);
    fn exl3_shim_gumbel_noise_f32(logits_in: *mut c_void, logits_out: *mut c_void, random: u32);
    fn exl3_shim_cache_rotate(cache: *mut c_void, order: *mut c_void, temp: *mut c_void);
    #[allow(clippy::too_many_arguments)]
    fn exl3_shim_quant_cache_paged(
        k_in: *mut c_void, k_out: *mut c_void, k_out_scales: *mut c_void,
        v_in: *mut c_void, v_out: *mut c_void, v_out_scales: *mut c_void,
        cache_seqlens: *mut c_void, block_table: *mut c_void,
        page_size: i32, seq_len: i32, compand_a: f32, in_contiguous: bool,
    );
    #[allow(clippy::too_many_arguments)]
    fn exl3_shim_dequant_cache_paged(
        k_in: *mut c_void, k_in_scales: *mut c_void, k_out: *mut c_void,
        v_in: *mut c_void, v_in_scales: *mut c_void, v_out: *mut c_void,
        cache_seqlens: *mut c_void, block_table: *mut c_void,
        page_size: i32, sliding_window: i32, compand_a: f32,
    );
    #[allow(clippy::too_many_arguments)]
    fn exl3_shim_dequant_cache_paged_window(
        k_in: *mut c_void, k_in_scales: *mut c_void, k_out: *mut c_void,
        v_in: *mut c_void, v_in_scales: *mut c_void, v_out: *mut c_void,
        cache_seqlens: *mut c_void, block_table: *mut c_void,
        page_size: i32, bonus_len: i32, compand_a: f32,
    );

    fn exl3_shim_had_r_128(
        x: *mut c_void, out: *mut c_void, pre_scale: *mut c_void, post_scale: *mut c_void, scale: f32,
    );
    fn exl3_shim_reconstruct(unpacked: *mut c_void, packed: *mut c_void, k: i32, mcg: bool, mul1: bool);
    fn exl3_shim_mul_sigmoid_(x: *mut c_void, y: *mut c_void);
    fn exl3_shim_deinterleave_qg(qg: *mut c_void, q: *mut c_void, g: *mut c_void, head_dim: i32);
    fn exl3_shim_gdn_fused_op_2(
        b: *mut c_void, a: *mut c_void, dt_bias: *mut c_void, a_log: *mut c_void,
        beta: *mut c_void, g: *mut c_void, beta_scale: f32,
    );
    #[allow(clippy::too_many_arguments)]
    fn exl3_shim_gdn_chunk_wy(
        k: *mut c_void, v: *mut c_void, beta: *mut c_void, a_log: *mut c_void,
        w: *mut c_void, uv: *mut c_void,
    );
    fn exl3_shim_gdn_chunk_fused(
        w: *mut c_void, q: *mut c_void, k: *mut c_void, qkt: *mut c_void, uv: *mut c_void,
        alog: *mut c_void, h0: *mut c_void, out: *mut c_void, ht: *mut c_void, scale: f64,
    );
    fn exl3_shim_recurrent_gated_delta_rule(
        mixed_qkv: *mut c_void, g: *mut c_void, beta: *mut c_void,
        recurrent_state: *mut c_void, core_attn_out: *mut c_void,
        num_k_heads: i32, num_v_heads: i32, k_head_dim: i32, v_head_dim: i32,
        slots: *mut c_void, history: bool,
    );
    fn exl3_shim_causal_conv1d_update(
        x: *mut c_void, conv_state: *mut c_void, slots: *mut c_void,
        weight: *mut c_void, bias: *mut c_void, out: *mut c_void,
        activation: bool, history: bool,
    );

    fn exl3_graph_new() -> *mut c_void;
    fn exl3_graph_free(g: *mut c_void);
    fn exl3_graph_capture_begin(g: *mut c_void) -> i32;
    fn exl3_graph_capture_end(g: *mut c_void) -> i32;
    fn exl3_graph_replay(g: *mut c_void) -> i32;
    fn exl3_use_side_stream();
    fn exl3_sync_side_stream();
    fn exl3_empty_cache();
    fn exl3_shim_set_mma_attn(on: i32);
    fn exl3_cuda_free_mib() -> i64;
}

/// Free VRAM in MiB (driver view). For leak debugging.
/// Force the tensor-core attention kernels on/off for this process, overriding
/// `EXL3_MMA_ATTN`. Only `bin/attn_check` uses it, to diff both kernels over the
/// same inputs in one process.
pub fn set_mma_attn(on: bool) {
    unsafe { exl3_shim_set_mma_attn(on as i32) }
}

pub fn cuda_free_mib() -> i64 {
    unsafe { exl3_cuda_free_mib() }
}

pub const ROPE_NEOX: i32 = 2;
pub const ROPE_GPTJ: i32 = 1;
pub const ROPE_NONE: i32 = 0;

/// `C(m,n) = hadamard(A(m,k)) · trellis(B) · scaleflips`. `a_had` is scratch shaped like `a`.
/// Per-expert weight pointer tables for [`exl3_mgemm`]: `[num_experts]` int64
/// tensors of device data pointers, one set per projection.
pub struct MultiLinear {
    pub trellis: Tensor,
    pub suh: Tensor,
    pub svh: Tensor,
    pub k: i64,
    pub mcg: bool,
    pub mul1: bool,
}

/// Multi-GEMM over a pointer list of EXL3 weights: one launch for every expert a
/// token selected, with the selection taken from the device.
///
/// `a` `[bszm, m, k]` fp16, `c` `[bszm, m, n]` fp16 or fp32, `a_had` scratch of
/// at least `bszm * m * k` halves (the kernel writes one transformed slab per
/// matrix; too small is silent corruption). `indices` `[num_tokens, matrices]`
/// int64 picks the weight per output row. `weights` `[num_tokens, matrices]` fp16,
/// when given, scales each row and the kernel reduces them into row 0 of `c` —
/// which is how the routed mixture is summed without a separate pass.
///
/// Unlike the fully fused MoE kernel this places no restriction on the codebook,
/// so it works on plain 3INST checkpoints as well as `mcg`/`mul1` ones.
#[allow(clippy::too_many_arguments)]
pub fn exl3_mgemm(
    a: &Tensor,
    w: &MultiLinear,
    c: &Tensor,
    a_had: &Tensor,
    indices: Option<&Tensor>,
    weights: Option<&Tensor>,
    num_tokens: i32,
) {
    let opt = |t: Option<&Tensor>| t.map(p).unwrap_or(std::ptr::null_mut());
    let _rc = unsafe {
        exl3_shim_mgemm(
            p(a),
            p(&w.trellis),
            p(c),
            p(&w.suh),
            p(a_had),
            p(&w.svh),
            opt(indices),
            opt(weights),
            w.k as i32,
            w.mcg,
            w.mul1,
            num_tokens,
        )
    };
}

pub fn exl3_gemm(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    suh: &Tensor,
    a_had: &Tensor,
    svh: &Tensor,
    mcg: bool,
    mul1: bool,
) {
    // Return value is the chosen shape/kernel tag: >0 for the cooperative GEMM,
    // 90 for the QTIP GEMV, and **0 for the int8-GEMV fast path** (default-on for
    // `mul1` tensors) — so 0 is success, not failure. A genuinely unsupported
    // shape aborts inside the kernel via TORCH_CHECK, not here.
    let _rc = unsafe {
        exl3_shim_gemm(p(a), p(b), p(c), p(suh), p(a_had), p(svh), mcg, mul1)
    };
}

/// RMSNorm over the last dim of a 2-D `x`. `y` pre-allocated, same shape.
pub fn rms_norm(x: &Tensor, w: &Tensor, y: &Tensor, eps: f32, constant_bias: f32, constant_scale: f32) {
    unsafe { exl3_shim_rms_norm(p(x), p(w), p(y), eps, constant_bias, constant_scale) }
}

/// Fused (optional) Q/K RMSNorm + RoPE. `q`,`k` are `(bsz, qlen, heads, head_dim)` fp16.
/// `positions` (int32 `[bsz]`, device), if given, overrides the scalar `position`.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    q: &Tensor,
    out_q: &Tensor,
    k: Option<&Tensor>,
    out_k: Option<&Tensor>,
    inv_freq: &Tensor,
    position: i64,
    positions: Option<&Tensor>,
    q_norm: Option<&Tensor>,
    k_norm: Option<&Tensor>,
    norm_eps: f32,
    norm_constant_bias: f32,
    rope_mode: i32,
    attn_factor: f32,
) {
    unsafe {
        exl3_shim_rope(
            p(q),
            p(out_q),
            op(k),
            op(out_k),
            p(inv_freq),
            position as u32,
            op(positions),
            op(q_norm),
            op(k_norm),
            norm_eps,
            norm_constant_bias,
            rope_mode,
            attn_factor,
        )
    }
}

/// Append new K/V `(bsz, s, n_kv, head_dim)` fp16 into the paged cache at
/// `cache_seqlens[b]`. `block_table` int32 `[bsz, num_pages]`, `cache_seqlens` int32 `[bsz]`.
pub fn paged_kv_cache_update(
    k: &Tensor,
    v: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_table: &Tensor,
    cache_seqlens: &Tensor,
) {
    unsafe {
        exl3_shim_paged_kv_cache_update(
            p(k), p(v), p(k_cache), p(v_cache), p(block_table), p(cache_seqlens),
        )
    }
}

/// Paged causal GQA attention. `cache_seqlens` is the pre-append length per seq;
/// the kernel attends over `cache_seqlens[b] + q_len` keys.
#[allow(clippy::too_many_arguments)]
pub fn bighead_attn_paged(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_table: &Tensor,
    cache_seqlens: &Tensor,
    o: &Tensor,
    sm_scale: f32,
) {
    unsafe {
        exl3_shim_bighead_attn_paged(
            p(q), p(k), p(v), p(k_cache), p(v_cache), p(block_table),
            p(cache_seqlens), p(o), 256, true, sm_scale,
        )
    }
}

/// Paged GQA attention against a **quantized** KV cache: `k`/`v` are the fresh
/// fp16 rows (quantized into `qk/sk/qv/sv` in-kernel), then attention reads the
/// packed store directly (short-q decode, online dequant, no fp16 scratch) or
/// via a compact window (long-q prefill). Writes `o` in place.
#[allow(clippy::too_many_arguments)]
pub fn bighead_attn_paged_q(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    qk: &Tensor,
    sk: &Tensor,
    qv: &Tensor,
    sv: &Tensor,
    block_table: &Tensor,
    cache_seqlens: &Tensor,
    o: &Tensor,
    sm_scale: f32,
    compand_a: f32,
) {
    unsafe {
        exl3_shim_bighead_attn_paged_q(
            p(q), p(k), p(v), p(qk), p(sk), p(qv), p(sv),
            p(block_table), p(cache_seqlens), p(o), 256, true, sm_scale, compand_a,
        )
    }
}

/// RAII wrapper over `at::cuda::CUDAGraph`. Capture switches tch onto a pooled
/// side stream and stays there; run the whole generation loop after `new`.
pub struct CudaGraph(*mut c_void);
unsafe impl Send for CudaGraph {}

impl CudaGraph {
    pub fn new() -> Self {
        CudaGraph(unsafe { exl3_graph_new() })
    }
    /// Prime allocator/autotuner state on the stream capture will use.
    pub fn use_side_stream() {
        unsafe { exl3_use_side_stream() }
    }
    pub fn sync_side_stream() {
        unsafe { exl3_sync_side_stream() }
    }
    /// Return freed CUDA-graph mempool blocks to the driver. Call after dropping
    /// a graph that won't be recaptured at the same shape.
    pub fn empty_cache() {
        unsafe { exl3_empty_cache() }
    }
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    /// Record `body`'s kernels into the graph (they are captured, not executed —
    /// call `replay` afterwards to run them). Returns `false` if capture failed.
    #[must_use]
    pub fn capture(&self, body: impl FnOnce()) -> bool {
        if unsafe { exl3_graph_capture_begin(self.0) } != 0 {
            return false;
        }
        body();
        unsafe { exl3_graph_capture_end(self.0) == 0 }
    }
    #[must_use]
    pub fn replay(&self) -> bool {
        unsafe { exl3_graph_replay(self.0) == 0 }
    }
}

impl Default for CudaGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { exl3_graph_free(self.0) }
    }
}

pub fn hgemm(a: &Tensor, b: &Tensor, c: &Tensor) {
    unsafe { exl3_shim_hgemm(p(a), p(b), p(c)) }
}

/// Non-cached causal GQA attention. q `(b,ql,nqh,d)`, k/v `(b,kl,nkvh,d)`, o `(b,ql,nqh,d)`.
pub fn bighead_attn(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, causal: bool, sm_scale: f32) {
    unsafe { exl3_shim_bighead_attn(p(q), p(k), p(v), p(o), 256, causal, sm_scale) }
}

// --- elementwise / norm / activation -----------------------------------------

/// RMSNorm of `x` fused with a residual add into `r` (`r += x` before norm),
/// writing the normed result to `y`. `w` optional weight.
pub fn rms_norm_res_in(x: &Tensor, w: Option<&Tensor>, y: &Tensor, r: &Tensor, eps: f32, bias: f32, scale: f32) {
    unsafe { exl3_shim_rms_norm_res_in(p(x), op(w), p(y), p(r), eps, bias, scale) }
}
/// Which nonlinearity the gate goes through. GDN and Mamba2 use `Silu`;
/// qwen4_exp's `output_gate_type` and KDA select `Sigmoid`. The distinction is
/// invisible in shapes and only mildly changes the output scale, so it has to be
/// carried explicitly rather than inferred.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GateAct {
    Silu = 0,
    Sigmoid = 1,
}

/// Gated RMSNorm: `y = rmsnorm(x, w) * act(g)` (gate applied per upstream `gated_rms_norm`).
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm(x: &Tensor, w: &Tensor, y: &Tensor, g: &Tensor, eps: f32, bias: f32, w_groups: i32, gate_first: bool, gate_act: GateAct) {
    unsafe { exl3_shim_gated_rms_norm(p(x), p(w), p(y), p(g), eps, bias, w_groups, gate_first, gate_act as i32) }
}
/// `y = softcap_factor * tanh(x / softcap_factor)`, elementwise, in fp16.
pub fn softcap(x: &Tensor, y: &Tensor, factor: f32) {
    unsafe { exl3_shim_softcap(p(x), p(y), factor) }
}
/// `z = x + y`, elementwise fp16.
pub fn add(x: &Tensor, y: &Tensor, z: &Tensor) {
    unsafe { exl3_shim_add(p(x), p(y), p(z)) }
}
/// `z = silu(x) * y` (SwiGLU). `act_limit <= 0` disables clamping.
pub fn silu_mul(x: &Tensor, y: &Tensor, z: &Tensor, act_limit: f32) {
    unsafe { exl3_shim_silu_mul(p(x), p(y), p(z), act_limit) }
}
/// `z = gelu(x) * y`.
pub fn gelu_mul(x: &Tensor, y: &Tensor, z: &Tensor, act_limit: f32) {
    unsafe { exl3_shim_gelu_mul(p(x), p(y), p(z), act_limit) }
}
/// `z = relu(x)^2 * y`.
pub fn relu2_mul(x: &Tensor, y: &Tensor, z: &Tensor, act_limit: f32) {
    unsafe { exl3_shim_relu2_mul(p(x), p(y), p(z), act_limit) }
}

// --- sampler kernels --------------------------------------------------------

/// Multiplicative repetition penalty (`SS_RepP`) over a sustain+decay window of
/// the token history. `in_logits` fp16 or fp32 `[vocab]`, `out_logits` fp32
/// `[vocab]` (may alias `in_logits`), `past_ids` int64 `[past_len]` on device.
/// `sustain_range` tokens back get the full penalty; the next `decay_range`
/// linearly fade it out. Pass `sustain_range = past_len, decay_range = 0` for the
/// classic full-history penalty.
pub fn apply_rep_pens(in_logits: &Tensor, out_logits: &Tensor, past_ids: &Tensor, rep_p: f32, sustain_range: i32, decay_range: i32) {
    unsafe { exl3_shim_apply_rep_pens(p(in_logits), p(out_logits), p(past_ids), rep_p, sustain_range, decay_range) }
}
/// Additive presence + frequency penalty (`SS_PresFreqP`), same windowing/shapes
/// as [`apply_rep_pens`].
pub fn apply_pres_freq_pens(in_logits: &Tensor, out_logits: &Tensor, past_ids: &Tensor, pres_p: f32, freq_p: f32, sustain_range: i32, decay_range: i32) {
    unsafe { exl3_shim_apply_pres_freq_pens(p(in_logits), p(out_logits), p(past_ids), pres_p, freq_p, sustain_range, decay_range) }
}
/// Block-wise argmax over fp16 `logits` `[rows, vocab]`, writing int64 token ids
/// to `ids` `[rows]`. `max_logit` (0 = vocab) clamps the search range.
pub fn argmax_sample(logits: &Tensor, ids: &Tensor, max_logit: i32) {
    unsafe { exl3_shim_argmax_sample(p(logits), p(ids), max_logit) }
}
/// Exact Gumbel-max categorical draw: adds Gumbel(0,1) noise (Philox seeded by
/// `random`) to fp16 `logits` `[rows, vocab]` then argmax → `ids` `[rows]` int64.
pub fn gumbel_sample(logits: &Tensor, ids: &Tensor, max_logit: i32, random: u32) {
    unsafe { exl3_shim_gumbel_sample(p(logits), p(ids), max_logit, random) }
}
/// Add Gumbel(0,1) noise to fp32 `logits_in` → `logits_out` (for a manual
/// argmax afterwards). Same Philox stream as [`gumbel_sample`].
pub fn gumbel_noise_f32(logits_in: &Tensor, logits_out: &Tensor, random: u32) {
    unsafe { exl3_shim_gumbel_noise_f32(p(logits_in), p(logits_out), random) }
}
/// Permute cache pages/rows in place along dim 0 by `order` (int32), using
/// `temp` as a one-slot scratch buffer shaped like a single slice of `cache`.
pub fn cache_rotate(cache: &Tensor, order: &Tensor, temp: &Tensor) {
    unsafe { exl3_shim_cache_rotate(p(cache), p(order), p(temp)) }
}

/// Quantize fresh K/V rows `(bsz, seq_len, n_kv, hd)` fp16 straight into the
/// paged quantized cache at `cache_seqlens[b] .. + seq_len`.
/// `qk`/`qv`: int32 `[pages, 256, n_kv*hd/32*bits]`; `sk`/`sv`: fp16 `[pages, 256, n_kv*hd/32]`.
/// `compand_a = 0.0` for linear quantization. `in_contiguous` = K/V rows are packed.
#[allow(clippy::too_many_arguments)]
pub fn quant_cache_paged(
    k: &Tensor, qk: &Tensor, sk: &Tensor,
    v: &Tensor, qv: &Tensor, sv: &Tensor,
    cache_seqlens: &Tensor, block_table: &Tensor,
    seq_len: i64, compand_a: f32, in_contiguous: bool,
) {
    unsafe {
        exl3_shim_quant_cache_paged(
            p(k), p(qk), p(sk), p(v), p(qv), p(sv),
            p(cache_seqlens), p(block_table),
            256, seq_len as i32, compand_a, in_contiguous,
        )
    }
}

/// Dequantize the stored prefix `[0, cache_seqlens[b])` of the quantized paged
/// cache into fp16 scratch pools `k_out`/`v_out` (`[pages, 256, n_kv, hd]`, the
/// plain-fp16-cache layout). `sliding_window < 0` dequantizes the whole prefix.
#[allow(clippy::too_many_arguments)]
pub fn dequant_cache_paged(
    qk: &Tensor, sk: &Tensor, k_out: &Tensor,
    qv: &Tensor, sv: &Tensor, v_out: &Tensor,
    cache_seqlens: &Tensor, block_table: &Tensor,
    sliding_window: i64, compand_a: f32,
) {
    unsafe {
        exl3_shim_dequant_cache_paged(
            p(qk), p(sk), p(k_out), p(qv), p(sv), p(v_out),
            p(cache_seqlens), p(block_table),
            256, sliding_window as i32, compand_a,
        )
    }
}

/// As `dequant_cache_paged`, but writes the referenced prefix COMPACTLY: page `p`
/// of block-table row `b` lands at output page `b*pages_per_seq + p`, so `k_out`/
/// `v_out` only need `bsz*pages_per_seq` pages, not the whole pool. Attention then
/// runs with an identity block table (`arange(bsz*pages_per_seq)`). `bonus_len`
/// rows past `cache_seqlens[b]` are also dequantized (a pre-appended chunk).
#[allow(clippy::too_many_arguments)]
pub fn dequant_cache_paged_window(
    qk: &Tensor, sk: &Tensor, k_out: &Tensor,
    qv: &Tensor, sv: &Tensor, v_out: &Tensor,
    cache_seqlens: &Tensor, block_table: &Tensor,
    bonus_len: i64, compand_a: f32,
) {
    unsafe {
        exl3_shim_dequant_cache_paged_window(
            p(qk), p(sk), p(k_out), p(qv), p(sv), p(v_out),
            p(cache_seqlens), p(block_table),
            256, bonus_len as i32, compand_a,
        )
    }
}

// --- gated-delta-net (Qwen3.5 linear attention) ------------------------------

/// Random-Hadamard transform of `[rows, k]` half `x` → `out` (k % 128 == 0),
/// with an optional per-column `pre_scale` (before) / `post_scale` (after).
pub fn had_r_128(x: &Tensor, out: &Tensor, pre_scale: Option<&Tensor>, post_scale: Option<&Tensor>, scale: f32) {
    unsafe { exl3_shim_had_r_128(p(x), p(out), op(pre_scale), op(post_scale), scale) }
}

/// Dequantize a packed EXL3 trellis tensor `packed` into dense `[k, n]` half `unpacked`.
pub fn reconstruct(unpacked: &Tensor, packed: &Tensor, k: i64, mcg: bool, mul1: bool) {
    unsafe { exl3_shim_reconstruct(p(unpacked), p(packed), k as i32, mcg, mul1) }
}

/// `x *= sigmoid(y)` in place (attention output gate).
pub fn mul_sigmoid_(x: &Tensor, y: &Tensor) {
    unsafe { exl3_shim_mul_sigmoid_(p(x), p(y)) }
}

/// Split an interleaved q/gate projection `qg [b,s,Nq,2*hd]` (fp16) into
/// `q [b,s,Nq,hd]` and `g [b,s,Nq*hd]`.
pub fn deinterleave_qg(qg: &Tensor, q: &Tensor, g: &Tensor, head_dim: i64) {
    unsafe { exl3_shim_deinterleave_qg(p(qg), p(q), p(g), head_dim as i32) }
}

/// Split-projection GDN input op: `beta = sigmoid(b)*beta_scale` (bf16),
/// `g = -softplus(a + dt_bias) * exp(a_log)` (f32). `b`/`a` are `[B,S,H]` f32.
pub fn gdn_fused_op_2(b: &Tensor, a: &Tensor, dt_bias: &Tensor, a_log: &Tensor,
                      beta: &Tensor, g: &Tensor, beta_scale: f32) {
    unsafe { exl3_shim_gdn_fused_op_2(p(b), p(a), p(dt_bias), p(a_log), p(beta), p(g), beta_scale) }
}

/// Sequential gated delta rule over the full seqlen (prefill + decode).
/// `mixed_qkv [b,s,2*k_dim+v_dim]` bf16, `g`/`beta [b,s,Nv]` (f32 / bf16),
/// `recurrent_state [b,H+1,Nv,k_hd,v_hd]` f32 (updated in place at index 0),
/// `core_attn_out [b,s,Nv,v_hd]` bf16. `slots` optional `[b]` int32.
#[allow(clippy::too_many_arguments)]
/// WY-chunked gated delta rule, stages 1+2 fused: K K^T, the decay ratios, M,
/// the triangular solve and W/Uv in one kernel per (V head, chunk). `k
/// [nk,nc,C,khd]` half; `v [nv,nc,C,vhd]`, `beta`/`a_log [nv,nc,C]`,
/// `w [nv,nc,C,khd]`, `uv [nv,nc,C,vhd]` f32. All contiguous.
pub fn gdn_chunk_wy(k: &Tensor, v: &Tensor, beta: &Tensor, a_log: &Tensor, w: &Tensor, uv: &Tensor) {
    unsafe { exl3_shim_gdn_chunk_wy(p(k), p(v), p(beta), p(a_log), p(w), p(uv)) }
}

pub fn gdn_chunk_fused(
    w: &Tensor, q: &Tensor, k: &Tensor, qkt: &Tensor, uv: &Tensor,
    alog: &Tensor, h0: &Tensor, out: &Tensor, ht: &Tensor, scale: f64,
) {
    unsafe {
        exl3_shim_gdn_chunk_fused(p(w), p(q), p(k), p(qkt), p(uv), p(alog), p(h0), p(out), p(ht), scale)
    }
}

pub fn recurrent_gated_delta_rule(
    mixed_qkv: &Tensor, g: &Tensor, beta: &Tensor,
    recurrent_state: &Tensor, core_attn_out: &Tensor,
    num_k_heads: i64, num_v_heads: i64, k_head_dim: i64, v_head_dim: i64,
    slots: Option<&Tensor>, history: bool,
) {
    unsafe {
        exl3_shim_recurrent_gated_delta_rule(
            p(mixed_qkv), p(g), p(beta), p(recurrent_state), p(core_attn_out),
            num_k_heads as i32, num_v_heads as i32, k_head_dim as i32, v_head_dim as i32,
            op(slots), history,
        )
    }
}

/// Causal depthwise conv1d + SiLU with rolling `conv_state`.
/// `x`/`out [b,dim,S]` bf16 (out is actually written `[b,S,dim]` by the kernel —
/// caller allocates accordingly), `conv_state [num_slots,dim,k]` bf16,
/// `weight [dim,k]` bf16, `slots` optional `[b]` int32.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_update(
    x: &Tensor, conv_state: &Tensor, slots: Option<&Tensor>,
    weight: &Tensor, bias: Option<&Tensor>, out: &Tensor,
    activation: bool, history: bool,
) {
    unsafe {
        exl3_shim_causal_conv1d_update(
            p(x), p(conv_state), op(slots), p(weight), op(bias), p(out),
            activation, history,
        )
    }
}
