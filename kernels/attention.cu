#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include "add.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include <algorithm>
#include "quant/exl3_devctx.cuh"
#include "attention_mma.cuh"
#include "cache/q_cache_kernels.cuh"   // dequant_block_x4<BITS>
#include "cache/q_cache.cuh"           // quant_cache_paged, dequant_cache_paged_window

#define G_MAX 16
#define PAGE_SIZE 256

__device__ __forceinline__ uint64_t paged_cache_offset
(
    const int32_t* __restrict__ block_row,  // [num_pages_per_seq]
    int64_t logical_pos,
    int64_t kv_head,
    int64_t n_kv_heads,
    int64_t dim,
    int d
)
{
    const int64_t logical_page   = logical_pos / PAGE_SIZE;
    const int64_t offset_in_page = logical_pos % PAGE_SIZE;
    const int32_t physical_page  = block_row[logical_page];

    return
        ((((uint64_t)physical_page * (uint64_t)PAGE_SIZE +
           (uint64_t)offset_in_page) * (uint64_t)n_kv_heads +
          (uint64_t)kv_head) * (uint64_t)dim) + (uint64_t)d;
}

template<int D>
__global__ void kv_cache_update_kernel_paged
(
    const half*    __restrict__ k,             // [bsz, kv_append_len, n_kv_heads, D]
    const half*    __restrict__ v,             // [bsz, kv_append_len, n_kv_heads, D]
    half*          __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    half*          __restrict__ v_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    int64_t bsz,
    int64_t kv_append_len,
    int64_t n_kv_heads,
    int64_t num_pages_per_seq
)
{
    constexpr int THREADS = D / 2;

    const int64_t bt_idx  = (int64_t) blockIdx.x;  // batch-token flattened
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int tid         = threadIdx.x;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch  = bt_idx / kv_append_len;
    const int64_t kv_pos = bt_idx % kv_append_len;

    const int64_t logical_pos = (int64_t)cache_seqlens[batch] + kv_pos;
    const int32_t* block_row  = block_table + batch * num_pages_per_seq;

    const uint64_t src_off =
        ((((uint64_t)batch * (uint64_t)kv_append_len + (uint64_t)kv_pos) * (uint64_t)n_kv_heads +
          (uint64_t)kv_head) * (uint64_t)D) + (uint64_t)d0;

    const uint64_t dst_off =
        paged_cache_offset(block_row, logical_pos, kv_head, n_kv_heads, D, d0);

    *((half2*)(k_cache + dst_off)) = *((const half2*)(k + src_off));
    *((half2*)(v_cache + dst_off)) = *((const half2*)(v + src_off));
}


// ---------------------------------------------------------------------------
// Flash-decoding paged attention (short q) — CUDA port of
// `attention_fn/triton_paged.py::_paged_attn_decode_split_kernel` (+ the
// `_paged_attn_decode_combine_kernel`, for which the existing
// `attn_reduce_kernel` is reused). Phase 1: one CTA per (batch, kv_head, q_pos,
// kv-split). The KV range of a split is walked in DEC_BLOCK_N-key tiles staged
// through shared memory (vectorised half2 loads, one __syncthreads per tile
// instead of the two-per-key of the scalar `attn_chunked_paged_kernel`), with an
// online-softmax accumulator per GQA sibling head. `num_splits` is sized on the
// host to ~2x the SM count so long-context decode keeps every SM busy.
// ---------------------------------------------------------------------------

// Per-head-dim KV tile width: keeps the double-buffered K+V staging under 48 KB
// of static shared memory for every supported head_dim.
template<int D> struct DecTile { static constexpr int N = 32; };
template<>      struct DecTile<256> { static constexpr int N = 16; };

template<int D, int G>
__global__ void attn_decode_split_kernel
(
    const half*    __restrict__ q,             // [bsz, q_len, n_q_heads, D]
    const half*    __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const half*    __restrict__ v_cache,
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    float*         __restrict__ workspace,     // [bsz, q_len, n_q_heads, num_splits, D+2]
    half*          __restrict__ o_single,      // [bsz, q_len, n_q_heads, D] (num_splits==1)
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t num_splits,
    int64_t split_len,
    int64_t num_pages_per_seq,
    bool    causal,
    float   scale,
    bool    final_single
)
{
    constexpr int TH    = D / 2;              // half2 lanes: 32 (d64) / 64 (d128) / 128 (d256)
    constexpr int WARPS = (TH + 31) / 32;
    constexpr int BN    = DecTile<D>::N;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int64_t split = (int64_t) blockIdx.y;

    // grid.x = batch * (n_kv_heads * q_len) + kv_head * q_len + q_pos
    const int64_t bx      = (int64_t) blockIdx.x;
    const int64_t q_pos   = bx % q_len;
    const int64_t tmp     = bx / q_len;
    const int64_t kv_head = tmp % n_kv_heads;
    const int64_t batch   = tmp / n_kv_heads;
    const int64_t q_head0 = kv_head * G;

    const int64_t total_k = (int64_t) cache_seqlens[batch] + kv_append_len;
    const int64_t q_abs   = total_k - q_len + q_pos;              // bottom-right causal
    const int64_t n_lo    = split * split_len;
    int64_t n_hi          = min(n_lo + split_len, total_k);
    if (causal) n_hi = min(n_hi, q_abs + 1);

    const int32_t* block_row = block_table + batch * num_pages_per_seq;

    // Double-buffered KV staging: tile t+1 is cp.async-loaded while tile t computes.
    __shared__ half2 k_smem[2][BN * TH];
    __shared__ half2 v_smem[2][BN * TH];
    __shared__ float red_smem[BN * G * WARPS];
    __shared__ float sc_smem[BN * G];

    half2 q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head0 + g)) * (uint64_t)D) + (uint64_t)(2 * tid);
        q_reg[g] = *((const half2*)(q + off));
    }

    float m_reg[G], l_reg[G], o0[G], o1[G];
    #pragma unroll
    for (int g = 0; g < G; g++) { m_reg[g] = -INFINITY; l_reg[g] = 0.f; o0[g] = 0.f; o1[g] = 0.f; }

    const int n_tiles = (int) ((n_hi - n_lo + BN - 1) / BN);

    // Issue the cp.async loads for one KV tile into shared buffer `buf`. Positions past
    // `n_hi` are clamped onto the last valid token (masked out later in the score step),
    // so every async copy targets an in-range, page-mapped address.
    auto load_tile = [&] (int t, int buf)
    {
        const int64_t base = n_lo + (int64_t) t * BN;
        #pragma unroll
        for (int r = 0; r < BN; r++)
        {
            const int64_t pos = min(base + r, n_hi - 1);
            const uint64_t off = paged_cache_offset(block_row, pos, kv_head, n_kv_heads, D, 2 * tid);
            __pipeline_memcpy_async(&k_smem[buf][r * TH + tid], k_cache + off, sizeof(half2));
            __pipeline_memcpy_async(&v_smem[buf][r * TH + tid], v_cache + off, sizeof(half2));
        }
        __pipeline_commit();
    };

    if (n_tiles > 0)
    {
        load_tile(0, 0);
        for (int t = 0; t < n_tiles; t++)
        {
            if (t + 1 < n_tiles) load_tile(t + 1, (t + 1) & 1);
            __pipeline_wait_prior(t + 1 < n_tiles ? 1 : 0);
            __syncthreads();

            const int buf     = t & 1;
            const int64_t base = n_lo + (int64_t) t * BN;
            const int tile     = (int) min((int64_t) BN, n_hi - base);

            for (int r = 0; r < tile; r++)
            {
                const float2 kf = __half22float2(k_smem[buf][r * TH + tid]);
                #pragma unroll
                for (int g = 0; g < G; g++)
                {
                    const float2 qf = __half22float2(q_reg[g]);
                    float part = qf.x * kf.x + qf.y * kf.y;
                    #pragma unroll
                    for (int msk = 16; msk > 0; msk >>= 1)
                        part += __shfl_xor_sync(0xffffffff, part, msk);
                    if (lane == 0) red_smem[(r * G + g) * WARPS + warp] = part;
                }
            }
            __syncthreads();

            for (int i = tid; i < tile * G; i += TH)
            {
                float s = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS; w++) s += red_smem[i * WARPS + w];
                sc_smem[i] = s * scale;
            }
            __syncthreads();

            for (int r = 0; r < tile; r++)
            {
                const float2 vf = __half22float2(v_smem[buf][r * TH + tid]);
                #pragma unroll
                for (int g = 0; g < G; g++)
                {
                    const float s     = sc_smem[r * G + g];
                    const float m_new = fmaxf(m_reg[g], s);
                    const float alpha = __expf(m_reg[g] - m_new);
                    const float e     = __expf(s - m_new);
                    o0[g]    = alpha * o0[g]    + e * vf.x;
                    o1[g]    = alpha * o1[g]    + e * vf.y;
                    l_reg[g] = alpha * l_reg[g] + e;
                    m_reg[g] = m_new;
                }
            }
            __syncthreads();  // buffer `buf` is reusable once every thread has read it
        }
    }

    if (final_single)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float inv = l_reg[g] > 0.f ? (1.f / l_reg[g]) : 0.f;
            half* op = o_single +
                ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                  (uint64_t)(q_head0 + g)) * (uint64_t)D) + (uint64_t)(2 * tid);
            op[0] = __float2half(o0[g] * inv);
            op[1] = __float2half(o1[g] * inv);
        }
        return;
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head0 + g)) * (uint64_t)num_splits +
              (uint64_t)split) * (uint64_t)(D + 2));
        if (tid == 0) { ws[0] = m_reg[g]; ws[1] = l_reg[g]; }
        ws[2 + 2 * tid]     = o0[g];
        ws[2 + 2 * tid + 1] = o1[g];
    }
}


// ---------------------------------------------------------------------------
// Online-dequant flash-decode: identical to `attn_decode_split_kernel` except
// K/V are read from the PACKED quant cache (qk/sk/qv/sv). Each KV tile's rows are
// dequantized warp-collectively via `dequant_block_x4<BITS>` straight into the
// fp16 staging buffers — no fp16 cache-sized scratch anywhere (this is the
// zero-scratch path Python takes via its Triton `_fns_qc` backend). Workspace
// layout is unchanged, so `attn_reduce_kernel` combines the splits as-is.
//
// Packing (see cache/q_cache_kernels.cuh): a token's K vector for all kv heads is
// `groups_per_token = n_kv_heads*D/32` groups of 32; group g's codes occupy
// `BITS` int32 words at `codes[(token_pos*groups_per_token + g)*BITS]`, its scale
// one half at `scales[token_pos*groups_per_token + g]`. `dequant_block_x4` does 4
// groups (128 values) per warp; a head is `D/32` groups = `ceil(D/128)` quads.
// ---------------------------------------------------------------------------
template<int D, int G, int BITS>
__global__ void attn_decode_split_kernel_q
(
    const half*     __restrict__ q,           // [bsz, q_len, n_q_heads, D]
    const uint32_t* __restrict__ qk_cache,    // [num_pages, 256, groups_per_token*BITS]
    const half*     __restrict__ sk_cache,    // [num_pages, 256, groups_per_token]
    const uint32_t* __restrict__ qv_cache,
    const half*     __restrict__ sv_cache,
    const int32_t*  __restrict__ block_table, // [bsz, num_pages_per_seq]
    const int32_t*  __restrict__ cache_seqlens,
    float*          __restrict__ workspace,   // [bsz, q_len, n_q_heads, num_splits, D+2]
    half*           __restrict__ o_single,    // [bsz, q_len, n_q_heads, D] (num_splits==1)
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t num_splits,
    int64_t split_len,
    int64_t num_pages_per_seq,
    int64_t groups_per_token,
    float   compand_a,
    bool    causal,
    float   scale,
    bool    final_single
)
{
    constexpr int TH    = D / 2;
    constexpr int NWARP = (TH + 31) / 32;
    constexpr int BN    = DecTile<D>::N;
    constexpr int GPH   = D / 32;              // groups per head
    constexpr int QPH   = (GPH + 3) / 4;       // 4-group quads per head
    constexpr int KUNITS = BN * QPH;           // dequant units for K in one tile
    constexpr int UNITS  = 2 * KUNITS;         // + V

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int64_t split = (int64_t) blockIdx.y;
    const int64_t bx      = (int64_t) blockIdx.x;
    const int64_t q_pos   = bx % q_len;
    const int64_t tmp     = bx / q_len;
    const int64_t kv_head = tmp % n_kv_heads;
    const int64_t batch   = tmp / n_kv_heads;
    const int64_t q_head0 = kv_head * G;

    const int64_t total_k = (int64_t) cache_seqlens[batch] + kv_append_len;
    const int64_t q_abs   = total_k - q_len + q_pos;              // bottom-right causal
    const int64_t n_lo    = split * split_len;
    int64_t n_hi          = min(n_lo + split_len, total_k);
    if (causal) n_hi = min(n_hi, q_abs + 1);

    const int32_t* block_row = block_table + batch * num_pages_per_seq;

    __shared__ half2 k_smem[BN * TH];
    __shared__ half2 v_smem[BN * TH];
    __shared__ float red_smem[BN * G * NWARP];
    __shared__ float sc_smem[BN * G];

    half2 q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head0 + g)) * (uint64_t)D) + (uint64_t)(2 * tid);
        q_reg[g] = *((const half2*)(q + off));
    }

    float m_reg[G], l_reg[G], o0[G], o1[G];
    #pragma unroll
    for (int g = 0; g < G; g++) { m_reg[g] = -INFINITY; l_reg[g] = 0.f; o0[g] = 0.f; o1[g] = 0.f; }

    const int n_tiles = (int) ((n_hi - n_lo + BN - 1) / BN);

    for (int t = 0; t < n_tiles; t++)
    {
        const int64_t base = n_lo + (int64_t) t * BN;
        const int tile     = (int) min((int64_t) BN, n_hi - base);

        // --- dequantize this tile's K and V rows into the fp16 staging buffers.
        // Every warp runs the same UNITS/NWARP iterations (UNITS is a multiple of
        // NWARP for D in {64,128,256}), so `dequant_block_x4`'s warp shuffles are
        // always full-mask.
        for (int u = warp; u < UNITS; u += NWARP)
        {
            const bool is_v  = u >= KUNITS;
            const int  uu    = is_v ? (u - KUNITS) : u;
            const int  r     = uu / QPH;
            const int  qd    = uu % QPH;
            const int  active = min(4, GPH - qd * 4);
            const int64_t pos = min(base + r, n_hi - 1);
            const int64_t lp  = pos >> 8;               // PAGE_SIZE 256
            const int64_t op  = pos & 255;
            const int64_t pp  = block_row[lp];
            const int64_t rb  = ((pp << 8) + op) * groups_per_token + kv_head * GPH + qd * 4;
            const uint32_t* codes = (is_v ? qv_cache : qk_cache) + rb * BITS;
            const half*     sc    = (is_v ? sv_cache : sk_cache) + rb;
            half* out = (half*) (&(is_v ? v_smem : k_smem)[r * TH]) + qd * 128;
            dequant_block_x4<BITS>(codes, sc, out, active, compand_a);
        }
        __syncthreads();

        for (int r = 0; r < tile; r++)
        {
            const float2 kf = __half22float2(k_smem[r * TH + tid]);
            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                const float2 qf = __half22float2(q_reg[g]);
                float part = qf.x * kf.x + qf.y * kf.y;
                #pragma unroll
                for (int msk = 16; msk > 0; msk >>= 1)
                    part += __shfl_xor_sync(0xffffffff, part, msk);
                if (lane == 0) red_smem[(r * G + g) * NWARP + warp] = part;
            }
        }
        __syncthreads();

        for (int i = tid; i < tile * G; i += TH)
        {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < NWARP; w++) s += red_smem[i * NWARP + w];
            sc_smem[i] = s * scale;
        }
        __syncthreads();

        for (int r = 0; r < tile; r++)
        {
            const float2 vf = __half22float2(v_smem[r * TH + tid]);
            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                const float sv    = sc_smem[r * G + g];
                const float m_new = fmaxf(m_reg[g], sv);
                const float alpha = __expf(m_reg[g] - m_new);
                const float e     = __expf(sv - m_new);
                o0[g]    = alpha * o0[g]    + e * vf.x;
                o1[g]    = alpha * o1[g]    + e * vf.y;
                l_reg[g] = alpha * l_reg[g] + e;
                m_reg[g] = m_new;
            }
        }
        __syncthreads();
    }

    if (final_single)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float inv = l_reg[g] > 0.f ? (1.f / l_reg[g]) : 0.f;
            half* op = o_single +
                ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                  (uint64_t)(q_head0 + g)) * (uint64_t)D) + (uint64_t)(2 * tid);
            op[0] = __float2half(o0[g] * inv);
            op[1] = __float2half(o1[g] * inv);
        }
        return;
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head0 + g)) * (uint64_t)num_splits +
              (uint64_t)split) * (uint64_t)(D + 2));
        if (tid == 0) { ws[0] = m_reg[g]; ws[1] = l_reg[g]; }
        ws[2 + 2 * tid]     = o0[g];
        ws[2 + 2 * tid + 1] = o1[g];
    }
}


template<int G>
__global__ void attn_chunked_paged_kernel_512x256
(
    const half*    __restrict__ q,             // [bsz, q_len, n_q_heads, 512]
    const half*    __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, 512]
    const half*    __restrict__ v_cache,       // [num_cache_pages, 256, n_kv_heads, 512]
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    float*         __restrict__ workspace,     // [bsz, q_len, n_q_heads, n_chunks, 514]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    int64_t num_pages_per_seq,
    bool causal,
    float scale,
    half*  __restrict__ o_single_unused,
    bool   single_unused
)
{
    constexpr int D       = 512;
    constexpr int THREADS = 256;
    constexpr int WARPS   = THREADS / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t total_k_len = (int64_t)cache_seqlens[batch] + kv_append_len;
    const int64_t kv_start    = chunk * kv_chunk_size;
    const int64_t kv_end      = min(kv_start + kv_chunk_size, total_k_len);

    // Bottom-right aligned causal masking:
    // causal_limit = q_pos + seqlen_k - seqlen_q
    const int64_t causal_limit = causal ? (total_k_len - q_len + q_pos) : (total_k_len - 1);

    const int32_t* block_row = block_table + batch * num_pages_per_seq;

    __shared__ float partial_smem[G * WARPS];
    __shared__ float score_smem[G];

    half2 q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)d0;

        q_reg[g] = *((half2*)(q + q_off));
    }

    float m_reg[G], l_reg[G], o0_reg[G], o1_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g]  = -INFINITY;
        l_reg[g]  = 0.f;
        o0_reg[g] = 0.f;
        o1_reg[g] = 0.f;
    }

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        const uint64_t k_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, d0);
        const half2 k_reg = *((half2*)(k_cache + k_off));
        const float2 kf   = __half22float2(k_reg);

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float2 qf = __half22float2(q_reg[g]);
            partial[g] = qf.x * kf.x + qf.y * kf.y;
        }

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                partial_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        if (warp_id == 0 && lane_id < G)
        {
            const int g = lane_id;
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; w++)
                s += partial_smem[g * WARPS + w];
            score_smem[g] = s * scale;
        }
        __syncthreads();

        const uint64_t v_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, d0);
        const half2 v_reg = *((half2*)(v_cache + v_off));
        const float2 vf   = __half22float2(v_reg);

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = score_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o0_reg[g] = alpha * o0_reg[g] + exp_s * vf.x;
            o1_reg[g] = alpha * o1_reg[g] + exp_s * vf.y;
            l_reg[g]  = alpha * l_reg[g]  + exp_s;
            m_reg[g]  = m_new;
        }
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks +
              (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }

        ws[2 + d0] = o0_reg[g];
        ws[2 + d1] = o1_reg[g];
    }
}


template<int G>
__global__ void attn_chunked_kernel_512x256
(
    const half* __restrict__ q,         // [bsz, q_len,  n_q_heads, 512]
    const half* __restrict__ k,         // [bsz, kv_len, n_kv_heads, 512]
    const half* __restrict__ v,         // [bsz, kv_len, n_kv_heads, 512]
    float*      __restrict__ workspace, // [bsz, q_len, n_q_heads, n_chunks, 514]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    bool causal,
    float scale,
    half*  __restrict__ o_single_unused,
    bool   single_unused
)
{
    constexpr int D       = 512;
    constexpr int THREADS = 256;
    constexpr int WARPS   = THREADS / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t kv_start = chunk * kv_chunk_size;
    const int64_t kv_end   = min(kv_start + kv_chunk_size, kv_len);

    // Lower-right aligned causal masking:
    // q[q_pos] corresponds to absolute position (kv_len - q_len + q_pos).
    const int64_t causal_limit = causal ? (kv_len - q_len + q_pos) : (kv_len - 1);

    __shared__ float partial_smem[G * WARPS];
    __shared__ float score_smem[G];

    // Load q once into per-thread registers.
    half2 q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)d0;

        q_reg[g] = *((half2*)(q + q_off));
    }

    float m_reg[G], l_reg[G], o0_reg[G], o1_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g]  = -INFINITY;
        l_reg[g]  = 0.f;
        o0_reg[g] = 0.f;
        o1_reg[g] = 0.f;
    }

    const half* k_base = k + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);
    const half* v_base = v + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        const uint64_t kv_off = ((uint64_t)kv_pos * (uint64_t)n_kv_heads * (uint64_t)D) + (uint64_t)d0;

        // Load K directly to registers.
        const half2 k_reg = *((const half2*)(k_base + kv_off));
        const float2 kf   = __half22float2(k_reg);

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float2 qf = __half22float2(q_reg[g]);
            partial[g] = qf.x * kf.x + qf.y * kf.y;
        }

        // Intra-warp reduction.
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        // One partial per warp/head.
        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                partial_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        // Warp 0 reduces across warps. Lane g handles head g.
        if (warp_id == 0 && lane_id < G)
        {
            const int g = lane_id;
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; w++)
                s += partial_smem[g * WARPS + w];
            score_smem[g] = s * scale;
        }
        __syncthreads();

        // Load V directly to registers.
        const half2 v_reg = *((half2*)(v_base + kv_off));
        const float2 vf   = __half22float2(v_reg);

        // Online softmax update.
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = score_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o0_reg[g] = alpha * o0_reg[g] + exp_s * vf.x;
            o1_reg[g] = alpha * o1_reg[g] + exp_s * vf.y;
            l_reg[g]  = alpha * l_reg[g]  + exp_s;
            m_reg[g]  = m_new;
        }
    }

    // Write chunk partials to workspace.
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks +
              (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }

        ws[2 + d0] = o0_reg[g];
        ws[2 + d1] = o1_reg[g];
    }
}


__global__ void attn_reduce_kernel_512x256
(
    const float* __restrict__ workspace,  // [bsz, q_len, n_q_heads, n_chunks, 514]
    half*        __restrict__ output,     // [bsz, q_len, n_q_heads, 512]
    int64_t bsz,
    int64_t q_len,
    int64_t n_q_heads,
    int64_t n_chunks
)
{
    constexpr int D = 512;

    const int64_t bq_idx = (int64_t)(blockIdx.x);
    const int64_t q_head = (int64_t)(blockIdx.y);
    const int tid        = threadIdx.x;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch = bq_idx / q_len;
    const int64_t q_pos = bq_idx % q_len;

    float m_acc  = -INFINITY;
    float l_acc  = 0.f;
    float o0_acc = 0.f;
    float o1_acc = 0.f;

    const float* ws_base = workspace +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
          (uint64_t)q_head) * (uint64_t)n_chunks * (uint64_t)(D + 2));

    for (int64_t c = 0; c < n_chunks; c++)
    {
        const float* ws_c = ws_base + (uint64_t)c * (uint64_t)(D + 2);
        const float m_c   = ws_c[0];
        const float l_c   = ws_c[1];

        if (l_c == 0.f) continue;

        const float o0_c  = ws_c[2 + d0];
        const float o1_c  = ws_c[2 + d1];

        const float m_new = fmaxf(m_acc, m_c);
        const float alpha = __expf(m_acc - m_new);
        const float beta  = __expf(m_c  - m_new);

        o0_acc = alpha * o0_acc + beta * o0_c;
        o1_acc = alpha * o1_acc + beta * o1_c;
        l_acc  = alpha * l_acc  + beta * l_c;
        m_acc  = m_new;
    }

    half* out = output +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
          (uint64_t)q_head) * (uint64_t)D);

    out[d0] = __float2half(l_acc > 0.f ? (o0_acc / l_acc) : 0.f);
    out[d1] = __float2half(l_acc > 0.f ? (o1_acc / l_acc) : 0.f);
}


#include "attention_mma_kernel.cuh"

// ---------------------------------------------------------------------------
// Query-tiled paged prefill attention.
//
// `attn_chunked_paged_kernel` below is a decode kernel: one CTA per (query,
// kv_head), one KV position per iteration, four `__syncthreads()` each. Run over
// a prefill chunk it re-reads the whole KV prefix once per query token and
// spends most of its time in barriers — measured ~44 ms for a single
// 2048q x 2048k layer on a 4090 (~2.3 TFLOP/s), which is what made prompt
// ingestion O(n^2) with a huge constant (~450 tok/s at 24k context).
//
// This kernel keeps the same online-softmax math and paged layout but:
//   * one CTA covers WARPS consecutive query positions, one warp each, so a KV
//     tile is fetched once per WARPS queries instead of once per query;
//   * KV is staged BN positions at a time in shared memory — two barriers per
//     tile instead of four per KV position;
//   * each lane owns E2 `half2` of the head dim (dims 2l, 2l+1, 2l+64, ...),
//     making every smem read conflict-free and the output write coalesced.
//
// Runs the full KV range in one pass (no split-K / workspace / reduce pass):
// with q_len/WARPS * n_kv_heads * bsz CTAs there is already ample parallelism.
// ---------------------------------------------------------------------------

template<int D, int G>
__global__ __launch_bounds__(256) void attn_prefill_paged_kernel
(
    const half*    __restrict__ q,             // [bsz, q_len, n_q_heads, D]
    const half*    __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const half*    __restrict__ v_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t num_pages_per_seq,
    bool causal,
    float scale,
    half*  __restrict__ o,                     // [bsz, q_len, n_q_heads, D]
    float* __restrict__ workspace,             // [bsz, q_len, n_q_heads, num_splits, D+2]
    int64_t num_splits,                        // split-K over the KV range
    int64_t split_len,
    bool fin                                   // num_splits == 1: normalize, skip workspace
)
{
    constexpr int WARPS = 8;         // queries per CTA
    constexpr int BN    = 32;        // KV positions staged per tile (divides PAGE_SIZE)
    constexpr int E2    = D / 64;    // half2 per lane per head-dim vector

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    const int64_t batch   = (int64_t) blockIdx.z % bsz;
    const int64_t split   = (int64_t) blockIdx.z / bsz;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t q_base  = (int64_t) blockIdx.x * WARPS;
    const int64_t q_pos   = q_base + warp_id;

    const int64_t total_k_len = (int64_t) cache_seqlens[batch] + kv_append_len;
    const int32_t* block_row  = block_table + batch * num_pages_per_seq;
    const int64_t q_head_start = kv_head * G;

    // this warp's last visible KV position (-1 if the warp has no query)
    const int64_t my_limit =
        (q_pos >= q_len) ? -1
                         : (causal ? (total_k_len - q_len + q_pos) : (total_k_len - 1));
    // CTA-wide bound: the last warp's limit governs how far the tile loop runs
    const int64_t last_q  = min(q_base + WARPS - 1, q_len - 1);
    const int64_t vis_end =
        min(total_k_len, (causal ? (total_k_len - q_len + last_q) : (total_k_len - 1)) + 1);
    // this CTA's slice of the KV range
    const int64_t kv_lo   = split * split_len;
    const int64_t cta_end = min(vis_end, kv_lo + split_len);

    __shared__ __align__(16) half k_tile[BN * D];
    __shared__ __align__(16) half v_tile[BN * D];

    // q in registers, distributed as half2 across the warp
    half2 q_reg[G][E2];
    if (q_pos < q_len)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const half2* qp = (const half2*)
                (q + ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                       (uint64_t)(q_head_start + g)) * (uint64_t)D));
            #pragma unroll
            for (int e = 0; e < E2; e++) q_reg[g][e] = qp[lane + 32 * e];
        }
    }

    float m_reg[G], l_reg[G], o_reg[G][E2 * 2];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g] = -INFINITY;
        l_reg[g] = 0.f;
        #pragma unroll
        for (int e = 0; e < E2 * 2; e++) o_reg[g][e] = 0.f;
    }

    constexpr int VEC   = 8;              // halfs per 128-bit load
    constexpr int VPT   = BN * D / VEC;   // uint4 loads per tile
    for (int64_t kv0 = kv_lo; kv0 < cta_end; kv0 += BN)
    {
        __syncthreads();  // previous tile fully consumed
        {
            // a BN-aligned tile never straddles a page (BN divides PAGE_SIZE)
            const int64_t page_base =
                (int64_t) block_row[kv0 / PAGE_SIZE] * PAGE_SIZE + (kv0 % PAGE_SIZE);
            const int64_t rows = min((int64_t) BN, total_k_len - kv0);
            for (int idx = tid; idx < VPT; idx += 256)
            {
                const int j  = idx / (D / VEC);
                const int d8 = idx % (D / VEC);
                uint4 val = make_uint4(0, 0, 0, 0);
                if (j < rows)
                {
                    const uint64_t off =
                        (((uint64_t)(page_base + j) * (uint64_t)n_kv_heads + (uint64_t)kv_head)
                         * (uint64_t)D) + (uint64_t)(d8 * VEC);
                    val = *(const uint4*)(k_cache + off);
                    *(uint4*)(v_tile + (uint64_t)j * D + d8 * VEC) = *(const uint4*)(v_cache + off);
                }
                *(uint4*)(k_tile + (uint64_t)j * D + d8 * VEC) = val;
            }
        }
        __syncthreads();

        if (my_limit < 0) continue;

        const int64_t jmax = min((int64_t) BN, min(total_k_len, my_limit + 1) - kv0);
        for (int64_t j = 0; j < jmax; j++)
        {
            const half2* kp = (const half2*) (k_tile + j * D);
            const half2* vp = (const half2*) (v_tile + j * D);
            half2 kv2[E2], vv2[E2];
            #pragma unroll
            for (int e = 0; e < E2; e++) { kv2[e] = kp[lane + 32 * e]; vv2[e] = vp[lane + 32 * e]; }

            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                // half2 MACs for the per-lane partial: one instruction per two
                // dims instead of a convert-to-float pair plus two FMAs. The
                // cross-lane reduction below is still fp32, so only the 8-16
                // element per-lane partial is accumulated in fp16.
                half2 acc = __float2half2_rn(0.f);
                #pragma unroll
                for (int e = 0; e < E2; e++) acc = __hfma2(q_reg[g][e], kv2[e], acc);
                const float2 af = __half22float2(acc);
                float s = af.x + af.y;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1)
                    s += __shfl_xor_sync(0xffffffff, s, mask);
                s *= scale;

                const float m_new = fmaxf(m_reg[g], s);
                const float alpha = __expf(m_reg[g] - m_new);
                const float p     = __expf(s - m_new);
                #pragma unroll
                for (int e = 0; e < E2; e++)
                {
                    const float2 vf = __half22float2(vv2[e]);
                    o_reg[g][2 * e]     = alpha * o_reg[g][2 * e]     + p * vf.x;
                    o_reg[g][2 * e + 1] = alpha * o_reg[g][2 * e + 1] + p * vf.y;
                }
                l_reg[g] = alpha * l_reg[g] + p;
                m_reg[g] = m_new;
            }
        }
    }

    if (q_pos >= q_len) return;

    if (fin)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float inv = l_reg[g] > 0.f ? (1.f / l_reg[g]) : 0.f;
            half2* op = (half2*)
                (o + ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                       (uint64_t)(q_head_start + g)) * (uint64_t)D));
            #pragma unroll
            for (int e = 0; e < E2; e++)
                op[lane + 32 * e] =
                    __floats2half2_rn(o_reg[g][2 * e] * inv, o_reg[g][2 * e + 1] * inv);
        }
        return;
    }

    // split-K partials for `attn_reduce_kernel`: [m, l, unnormalized o[D]]
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)num_splits +
              (uint64_t)split) * (uint64_t)(D + 2));
        if (lane == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }
        #pragma unroll
        for (int e = 0; e < E2; e++)
        {
            ws[2 + lane * 2 + 64 * e]     = o_reg[g][2 * e];
            ws[2 + lane * 2 + 64 * e + 1] = o_reg[g][2 * e + 1];
        }
    }
}


// Query-tiled attention straight off the PACKED quant cache: identical to
// `attn_prefill_paged_kernel` except each KV tile is dequantized into the shared
// staging buffers by `dequant_block_x4<BITS>` instead of being read from an fp16
// cache. Nothing materializes the fp16 KV window, so the per-step
// `dequant_cache_paged_window` pass disappears — that pass was writing and
// re-reading ~3.3 GB per decode step at 50k context (~21 ms) purely to feed the
// fp16 kernel. Because a tile is dequantized once per CTA and consumed by all 8
// query warps and G query heads, the dequant cost amortizes the way the
// one-CTA-per-query online decode kernel never could.
template<int D, int G, int BITS>
__global__ __launch_bounds__(256) void attn_prefill_paged_kernel_q
(
    const half*     __restrict__ q,            // [bsz, q_len, n_q_heads, D]
    const uint32_t* __restrict__ qk_cache,     // [num_pages, 256, groups_per_token*BITS]
    const half*     __restrict__ sk_cache,     // [num_pages, 256, groups_per_token]
    const uint32_t* __restrict__ qv_cache,
    const half*     __restrict__ sv_cache,
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t num_pages_per_seq,
    int64_t groups_per_token,
    float compand_a,
    bool causal,
    float scale,
    half*  __restrict__ o,                     // [bsz, q_len, n_q_heads, D]
    float* __restrict__ workspace,             // [bsz, q_len, n_q_heads, num_splits, D+2]
    int64_t num_splits,                        // split-K over the KV range
    int64_t split_len,
    bool fin                                   // num_splits == 1: normalize, skip workspace
)
{
    constexpr int WARPS = 8;         // queries per CTA
    constexpr int BN    = 16;        // KV positions staged per tile (divides PAGE_SIZE)
    constexpr int E2    = D / 64;    // half2 per lane per head-dim vector

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    const int64_t batch   = (int64_t) blockIdx.z % bsz;
    const int64_t split   = (int64_t) blockIdx.z / bsz;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t q_base  = (int64_t) blockIdx.x * WARPS;
    const int64_t q_pos   = q_base + warp_id;

    const int64_t total_k_len = (int64_t) cache_seqlens[batch] + kv_append_len;
    const int32_t* block_row  = block_table + batch * num_pages_per_seq;
    const int64_t q_head_start = kv_head * G;

    // this warp's last visible KV position (-1 if the warp has no query)
    const int64_t my_limit =
        (q_pos >= q_len) ? -1
                         : (causal ? (total_k_len - q_len + q_pos) : (total_k_len - 1));
    // CTA-wide bound: the last warp's limit governs how far the tile loop runs
    const int64_t last_q  = min(q_base + WARPS - 1, q_len - 1);
    const int64_t vis_end =
        min(total_k_len, (causal ? (total_k_len - q_len + last_q) : (total_k_len - 1)) + 1);
    // this CTA's slice of the KV range
    const int64_t kv_lo   = split * split_len;
    const int64_t cta_end = min(vis_end, kv_lo + split_len);

    __shared__ __align__(16) half k_tile[BN * D];
    __shared__ __align__(16) half v_tile[BN * D];

    // q in registers, distributed as half2 across the warp
    half2 q_reg[G][E2];
    if (q_pos < q_len)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const half2* qp = (const half2*)
                (q + ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                       (uint64_t)(q_head_start + g)) * (uint64_t)D));
            #pragma unroll
            for (int e = 0; e < E2; e++) q_reg[g][e] = qp[lane + 32 * e];
        }
    }

    float m_reg[G], l_reg[G], o_reg[G][E2 * 2];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g] = -INFINITY;
        l_reg[g] = 0.f;
        #pragma unroll
        for (int e = 0; e < E2 * 2; e++) o_reg[g][e] = 0.f;
    }

    for (int64_t kv0 = kv_lo; kv0 < cta_end; kv0 += BN)
    {
        __syncthreads();  // previous tile fully consumed
        {
            // dequantize this tile's K and V rows straight into the staging
            // buffers. UNITS is a multiple of the warp count for D in {128,256},
            // so `dequant_block_x4`'s warp shuffles stay full-mask.
            constexpr int GPH    = D / 32;          // quant groups per head
            constexpr int QPH    = (GPH + 3) / 4;   // 4-group quads per head
            constexpr int KUNITS = BN * QPH;
            constexpr int UNITS  = 2 * KUNITS;
            const int64_t rows = min((int64_t) BN, total_k_len - kv0);
            for (int u = warp_id; u < UNITS; u += WARPS)
            {
                const bool is_v = u >= KUNITS;
                const int  uu   = is_v ? (u - KUNITS) : u;
                const int  r    = uu / QPH;
                const int  qd   = uu % QPH;
                const int  active = min(4, GPH - qd * 4);
                const int64_t pos = kv0 + min((int64_t) r, rows - 1);
                const int64_t pp  = block_row[pos >> 8];   // PAGE_SIZE 256
                const int64_t rb  = ((pp << 8) + (pos & 255)) * groups_per_token
                                  + kv_head * GPH + qd * 4;
                const uint32_t* codes = (is_v ? qv_cache : qk_cache) + rb * BITS;
                const half*     sc    = (is_v ? sv_cache : sk_cache) + rb;
                half* out = (is_v ? v_tile : k_tile) + (int64_t) r * D + qd * 128;
                dequant_block_x4<BITS>(codes, sc, out, active, compand_a);
            }
        }
        __syncthreads();

        if (my_limit < 0) continue;

        const int64_t jmax = min((int64_t) BN, min(total_k_len, my_limit + 1) - kv0);
        for (int64_t j = 0; j < jmax; j++)
        {
            const half2* kp = (const half2*) (k_tile + j * D);
            const half2* vp = (const half2*) (v_tile + j * D);
            half2 kv2[E2], vv2[E2];
            #pragma unroll
            for (int e = 0; e < E2; e++) { kv2[e] = kp[lane + 32 * e]; vv2[e] = vp[lane + 32 * e]; }

            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                // half2 MACs for the per-lane partial: one instruction per two
                // dims instead of a convert-to-float pair plus two FMAs. The
                // cross-lane reduction below is still fp32, so only the 8-16
                // element per-lane partial is accumulated in fp16.
                half2 acc = __float2half2_rn(0.f);
                #pragma unroll
                for (int e = 0; e < E2; e++) acc = __hfma2(q_reg[g][e], kv2[e], acc);
                const float2 af = __half22float2(acc);
                float s = af.x + af.y;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1)
                    s += __shfl_xor_sync(0xffffffff, s, mask);
                s *= scale;

                const float m_new = fmaxf(m_reg[g], s);
                const float alpha = __expf(m_reg[g] - m_new);
                const float p     = __expf(s - m_new);
                #pragma unroll
                for (int e = 0; e < E2; e++)
                {
                    const float2 vf = __half22float2(vv2[e]);
                    o_reg[g][2 * e]     = alpha * o_reg[g][2 * e]     + p * vf.x;
                    o_reg[g][2 * e + 1] = alpha * o_reg[g][2 * e + 1] + p * vf.y;
                }
                l_reg[g] = alpha * l_reg[g] + p;
                m_reg[g] = m_new;
            }
        }
    }

    if (q_pos >= q_len) return;

    if (fin)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float inv = l_reg[g] > 0.f ? (1.f / l_reg[g]) : 0.f;
            half2* op = (half2*)
                (o + ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                       (uint64_t)(q_head_start + g)) * (uint64_t)D));
            #pragma unroll
            for (int e = 0; e < E2; e++)
                op[lane + 32 * e] =
                    __floats2half2_rn(o_reg[g][2 * e] * inv, o_reg[g][2 * e + 1] * inv);
        }
        return;
    }

    // split-K partials for `attn_reduce_kernel`: [m, l, unnormalized o[D]]
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)num_splits +
              (uint64_t)split) * (uint64_t)(D + 2));
        if (lane == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }
        #pragma unroll
        for (int e = 0; e < E2; e++)
        {
            ws[2 + lane * 2 + 64 * e]     = o_reg[g][2 * e];
            ws[2 + lane * 2 + 64 * e + 1] = o_reg[g][2 * e + 1];
        }
    }
}


template<int D, int G>
__global__ void attn_chunked_paged_kernel
(
    const half*    __restrict__ q,             // [bsz, q_len, n_q_heads, D]
    const half*    __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const half*    __restrict__ v_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    float*         __restrict__ workspace,     // [bsz, q_len, n_q_heads, n_chunks, D+2]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    int64_t num_pages_per_seq,
    bool causal,
    float scale,
    half*  __restrict__ o_single,   // [bsz, q_len, n_q_heads, D] — written iff `single`
    bool   single                   // n_chunks == 1: normalize in-kernel, skip workspace + reduce
)
{
    constexpr int WARPS = D / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t total_k_len = (int64_t)cache_seqlens[batch] + kv_append_len;
    const int64_t kv_start    = chunk * kv_chunk_size;
    const int64_t kv_end      = min(kv_start + kv_chunk_size, total_k_len);

    const int64_t causal_limit = causal ? (total_k_len - q_len + q_pos) : (total_k_len - 1);

    const int32_t* block_row = block_table + batch * num_pages_per_seq;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half*  kv_smem     = (half*) smem_raw;
    float* reduce_smem = (float*) (kv_smem + D);

    register half q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)tid;
        q_reg[g] = q[q_off];
    }

    register float m_reg[G], l_reg[G], o_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g] = -INFINITY;
        l_reg[g] = 0.f;
        o_reg[g] = 0.f;
    }

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        const uint64_t k_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, tid);
        kv_smem[tid] = k_cache[k_off];
        __syncthreads();

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
            partial[g] = __half2float(q_reg[g]) * __half2float(kv_smem[tid]);

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                reduce_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        if (tid == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                float s = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS; w++)
                    s += reduce_smem[g * WARPS + w];
                reduce_smem[g] = s * scale;
            }
        }
        __syncthreads();

        const uint64_t v_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, tid);
        kv_smem[tid] = v_cache[v_off];
        __syncthreads();

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = reduce_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o_reg[g] = alpha * o_reg[g] + exp_s * __half2float(kv_smem[tid]);
            l_reg[g] = alpha * l_reg[g] + exp_s;
            m_reg[g] = m_new;
        }
        __syncthreads();
    }

    if (single)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float inv = l_reg[g] > 0.f ? (1.f / l_reg[g]) : 0.f;
            half* op = o_single +
                ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                  (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)tid;
            *op = __float2half(o_reg[g] * inv);
        }
        return;
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks +
              (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }
        ws[2 + tid] = o_reg[g];
    }
}


template<int D, int G>
__global__ void attn_chunked_kernel
(
    const half* __restrict__ q,         // [bsz, q_len,  n_q_heads,  D]
    const half* __restrict__ k,         // [bsz, kv_len, n_kv_heads, D]
    const half* __restrict__ v,         // [bsz, kv_len, n_kv_heads, D]
    float*      __restrict__ workspace, // [bsz, q_len, n_q_heads, n_chunks, D+2]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    bool causal,
    float scale,
    half*  __restrict__ o_single,
    bool   single
)
{
    constexpr int WARPS = D / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t kv_start = chunk * kv_chunk_size;
    const int64_t kv_end   = min(kv_start + kv_chunk_size, kv_len);

    // Standard causal semantics: query position q_pos may attend to keys 0..q_pos.
    // If you later want decode-style bottom-right alignment, make q_start explicit.
    const int64_t seq_pos      = q_pos;
    const int64_t causal_limit = causal ? ((kv_len - q_len) + seq_pos) : (kv_len - 1);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half*  kv_smem     = (half*) smem_raw;
    float* reduce_smem = (float*) (kv_smem + D);
    // reduce_smem has WARPS*G_MAX entries:
    //   reduce_smem[g*WARPS + w] is warp w's partial for head g.

    register half q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)tid;
        q_reg[g] = q[q_off];
    }

    register float m_reg[G], l_reg[G], o_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g] = -INFINITY;
        l_reg[g] = 0.f;
        o_reg[g] = 0.f;
    }

    const half* k_base = k + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);
    const half* v_base = v + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        kv_smem[tid] = k_base[((uint64_t)kv_pos * (uint64_t)n_kv_heads * (uint64_t)D) + (uint64_t)tid];
        __syncthreads();

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
            partial[g] = __half2float(q_reg[g]) * __half2float(kv_smem[tid]);

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                reduce_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        if (tid == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                float s = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS; w++)
                    s += reduce_smem[g * WARPS + w];
                reduce_smem[g] = s * scale;
            }
        }

        kv_smem[tid] = v_base[((uint64_t)kv_pos * (uint64_t)n_kv_heads * (uint64_t)D) + (uint64_t)tid];
        __syncthreads();

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = reduce_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o_reg[g] = alpha * o_reg[g] + exp_s * __half2float(kv_smem[tid]);
            l_reg[g] = alpha * l_reg[g] + exp_s;
            m_reg[g] = m_new;
        }
        __syncthreads();
    }

    if (single)
    {
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float inv = l_reg[g] > 0.f ? (1.f / l_reg[g]) : 0.f;
            half* op = o_single +
                ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
                  (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)tid;
            *op = __float2half(o_reg[g] * inv);
        }
        return;
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks + (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }
        ws[2 + tid] = o_reg[g];
    }
}

template<int D, int G>
__global__ void attn_reduce_kernel
(
    const float* __restrict__ workspace,  // [bsz, q_len, n_q_heads, n_chunks, D+2]
    half*        __restrict__ output,     // [bsz, q_len, n_q_heads, D]
    int64_t bsz,
    int64_t q_len,
    int64_t n_q_heads,
    int64_t n_chunks
)
{
    const int64_t bq_idx = (int64_t)(blockIdx.x);
    const int64_t q_head = (int64_t)(blockIdx.y);
    const int tid        = threadIdx.x;

    const int64_t batch = bq_idx / q_len;
    const int64_t q_pos = bq_idx % q_len;

    float m_acc = -INFINITY;
    float l_acc = 0.f;
    float o_acc = 0.f;

    const float* ws_base = workspace +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
          (uint64_t)q_head) * (uint64_t)n_chunks * (uint64_t)(D + 2));

    for (int64_t c = 0; c < n_chunks; c++)
    {
        const float* ws_c = ws_base + (uint64_t)c * (uint64_t)(D + 2);
        const float m_c   = ws_c[0];
        const float l_c   = ws_c[1];
        const float o_c   = ws_c[2 + tid];

        if (l_c == 0.f) continue;

        const float m_new = fmaxf(m_acc, m_c);
        const float alpha = __expf(m_acc - m_new);
        const float beta  = __expf(m_c  - m_new);

        o_acc = alpha * o_acc + beta * o_c;
        l_acc = alpha * l_acc + beta * l_c;
        m_acc = m_new;
    }

    half* out = output +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads + (uint64_t)q_head) * (uint64_t)D);

    out[tid] = __float2half(l_acc > 0.f ? (o_acc / l_acc) : 0.f);
}


// ---------------------------------------------------------------------------
// Tensor-core attention dispatch (see attention_mma.cuh).
//
// `EXL3_MMA_ATTN=1` enables it; it is opt-in until validated on the target card
// against the scalar kernels, which stay the default and the reference.
// ---------------------------------------------------------------------------
static int g_mma_attn = -1;   // -1 = not resolved, else 0/1

// Also settable at runtime (`exl3_set_mma_attn`) so a single process can run both
// kernels over the same inputs and diff them — see `bin/attn_check.rs`.
extern "C" void exl3_set_mma_attn(int on) { g_mma_attn = on ? 1 : 0; }

static bool mma_attn_enabled()
{
    if (g_mma_attn < 0)
    {
        const char* e = getenv("EXL3_MMA_ATTN");
        // on by default; EXL3_MMA_ATTN=0 falls back to the scalar kernels
        g_mma_attn = (e && e[0] == '0') ? 0 : 1;
    }
    return g_mma_attn != 0;
}

// The CTA shape is fixed (RT row tiles x PAIR warps) and independent of G, so
// every supported head dim is worth running on the tensor cores.
static bool mma_shape_ok(int64_t dim, int64_t) { return dim == 128 || dim == 256; }

// Split-K / launch geometry shared by both mma entry points.
struct MmaLaunch
{
    int64_t num_splits;
    int64_t split_len;
    bool    fin;
    dim3    grid;
    dim3    red_grid;
    size_t  smem;
    int     threads;
};

template<int D, int G>
static MmaLaunch mma_plan(int64_t bsz, int64_t q_len, int64_t n_q_heads, int64_t n_kv_heads,
                          int64_t max_total_k_len, uint64_t ws_numel, int device)
{
    using Cfg = MmaCfg<D>;
    constexpr int STRIDE = Cfg::STRIDE;
    MmaLaunch p{};
    // rows are (query, query-head) pairs; a CTA covers RT tiles of 16 rows
    const int64_t rows    = q_len * G;
    const int64_t q_tiles = (rows + 16 * Cfg::RT - 1) / (16 * Cfg::RT);
    int num_sms = 84;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    const int64_t programs = q_tiles * n_kv_heads * bsz;
    int64_t ns = (int64_t)(2 * num_sms) / (programs > 0 ? programs : 1);
    const int64_t cap = (max_total_k_len + 4 * MMA_BN - 1) / (4 * MMA_BN);
    if (ns > cap) ns = cap;
    if (ns > 128) ns = 128;
    if (ns < 1)   ns = 1;
    while (ns > 1 && (uint64_t)bsz * q_len * n_q_heads * ns * (D + 2) > ws_numel) ns >>= 1;
    int64_t sl = (((max_total_k_len + ns - 1) / ns) + MMA_BN - 1) / MMA_BN * MMA_BN;
    if (sl < MMA_BN) sl = MMA_BN;
    p.num_splits = ns;
    p.split_len  = sl;
    p.fin        = (ns == 1);
    p.grid       = dim3((uint32_t) q_tiles, (uint32_t) n_kv_heads, (uint32_t)(bsz * ns));
    p.red_grid   = dim3((uint32_t)(bsz * q_len), (uint32_t) n_q_heads);
    p.smem       = (size_t) 2 * MMA_BN * STRIDE * sizeof(half)
                 + (size_t) Cfg::RT * 256 * sizeof(float);
    p.threads    = Cfg::THREADS;
    return p;
}

void bighead_attn_paged
(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    const at::Tensor& o,
    // const at::Tensor& workspace,
    int kv_chunk_size,
    bool causal,
    float sm_scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k, kHalf);
    TORCH_CHECK_DTYPE(v, kHalf);
    TORCH_CHECK_DTYPE(k_cache, kHalf);
    TORCH_CHECK_DTYPE(v_cache, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    // TORCH_CHECK_DTYPE(workspace, kFloat);
    TORCH_CHECK_DTYPE(block_table, kInt);
    TORCH_CHECK_DTYPE(cache_seqlens, kInt);

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
    // TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");

    TORCH_CHECK(q.dim() == 4, "q must be rank-4");
    TORCH_CHECK(k.dim() == 4, "k must be rank-4");
    TORCH_CHECK(v.dim() == 4, "v must be rank-4");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be rank-4");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be rank-4");
    TORCH_CHECK(o.dim() == 4, "o must be rank-4");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be rank-2");
    TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must be rank-1");

    const int64_t bsz            = q.size(0);
    const int64_t q_len          = q.size(1);
    const int64_t n_q_heads      = q.size(2);
    const int64_t dim            = q.size(3);

    const int64_t kv_append_len  = k.size(1);
    const int64_t n_kv_heads     = k.size(2);

    const int64_t num_cache_pages   = k_cache.size(0);
    const int64_t page_size         = k_cache.size(1);
    const int64_t num_pages_per_seq = block_table.size(1);

    TORCH_CHECK(k.size(0) == bsz, "k batch mismatch");
    TORCH_CHECK(v.size(0) == bsz, "v batch mismatch");
    TORCH_CHECK(v.size(1) == kv_append_len, "v kv_len mismatch");
    TORCH_CHECK(v.size(2) == n_kv_heads, "v n_kv_heads mismatch");
    TORCH_CHECK(k.size(3) == dim, "k head_dim mismatch");
    TORCH_CHECK(v.size(3) == dim, "v head_dim mismatch");

    TORCH_CHECK(k_cache.size(1) == PAGE_SIZE, "k_cache page size must be 256");
    TORCH_CHECK(v_cache.size(1) == PAGE_SIZE, "v_cache page size must be 256");
    TORCH_CHECK(k_cache.size(0) == num_cache_pages, "internal k_cache shape error");
    TORCH_CHECK(v_cache.size(0) == num_cache_pages, "v_cache num_pages mismatch");
    TORCH_CHECK(k_cache.size(2) == n_kv_heads, "k_cache n_kv_heads mismatch");
    TORCH_CHECK(v_cache.size(2) == n_kv_heads, "v_cache n_kv_heads mismatch");
    TORCH_CHECK(k_cache.size(3) == dim, "k_cache head_dim mismatch");
    TORCH_CHECK(v_cache.size(3) == dim, "v_cache head_dim mismatch");

    TORCH_CHECK(o.size(0) == bsz, "o batch mismatch");
    TORCH_CHECK(o.size(1) == q_len, "o q_len mismatch");
    TORCH_CHECK(o.size(2) == n_q_heads, "o n_q_heads mismatch");
    TORCH_CHECK(o.size(3) == dim, "o head_dim mismatch");

    TORCH_CHECK(block_table.size(0) == bsz, "block_table batch mismatch");
    TORCH_CHECK(cache_seqlens.size(0) == bsz, "cache_seqlens batch mismatch");

    TORCH_CHECK(n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads");
    const int64_t G = n_q_heads / n_kv_heads;
    TORCH_CHECK(G <= G_MAX, "GQA ratio ", G, " exceeds G_MAX=", G_MAX);
    TORCH_CHECK(kv_chunk_size > 0, "kv_chunk_size must be positive");

    const int64_t max_total_k_len = num_pages_per_seq * PAGE_SIZE;

    // Split-K over KV chunks needs an fp32 partials buffer of
    // bsz*q_len*n_q_heads*n_chunks*(dim+2) in the fixed device workspace. Grow the
    // chunk size until it fits; if even a single chunk doesn't fit (large-q prefill
    // through this fallback), take the `single` path: one CTA does the whole causal
    // reduction in registers and writes `o` directly — no workspace, no reduce pass.
    const uint64_t ws_numel      = WORKSPACE_SIZE / sizeof(float);
    int64_t n_chunks = (max_total_k_len + kv_chunk_size - 1) / kv_chunk_size;
    while (n_chunks > 1)
    {
        const uint64_t ws_needed = (uint64_t)bsz * (uint64_t)q_len * (uint64_t)n_q_heads * (uint64_t)n_chunks * (uint64_t)(dim + 2);
        if (ws_needed <= ws_numel) break;
        kv_chunk_size *= 2;
        n_chunks = (max_total_k_len + kv_chunk_size - 1) / kv_chunk_size;
    }
    const bool single = (n_chunks == 1) && (dim != 512);
    if (n_chunks == 1 && dim == 512)
    {
        const uint64_t ws_needed = (uint64_t)bsz * (uint64_t)q_len * (uint64_t)n_q_heads * (uint64_t)(dim + 2);
        TORCH_CHECK(ws_needed <= ws_numel, "head_dim 512 long-q attention exceeds workspace");
    }
    float* ws_ptr                = (float*) DevCtx::instance().get_ws(q.get_device());
    // DBGI2(kv_chunk_size, n_chunks);

    const half* q_ptr            = (const half*) q.data_ptr();
    const half* k_ptr            = (const half*) k.data_ptr();
    const half* v_ptr            = (const half*) v.data_ptr();
    half* k_cache_ptr            = (half*) k_cache.data_ptr();
    half* v_cache_ptr            = (half*) v_cache.data_ptr();
    half* o_ptr                  = (half*) o.data_ptr();
    const int32_t* block_ptr     = block_table.data_ptr<int32_t>();
    const int32_t* seqlens_ptr   = cache_seqlens.data_ptr<int32_t>();

    dim3 grid_up((uint32_t)std::max<int64_t>(1, bsz * kv_append_len), (uint32_t)n_kv_heads);
    dim3 grid1((uint32_t)(bsz * q_len), (uint32_t)n_kv_heads, (uint32_t)n_chunks);
    dim3 grid2((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

    const float scale = sm_scale == 0.0f ? rsqrtf((float) dim) : sm_scale;

    #define PAGED_UPDATE_ARGS \
        k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, block_ptr, seqlens_ptr, \
        bsz, kv_append_len, n_kv_heads, num_pages_per_seq

    #define PAGED_ARGS1 \
        q_ptr, k_cache_ptr, v_cache_ptr, block_ptr, seqlens_ptr, ws_ptr, \
        bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, n_chunks, \
        (int64_t)kv_chunk_size, num_pages_per_seq, causal, scale, o_ptr, single

    #define ARGS2 \
        ws_ptr, o_ptr, bsz, q_len, n_q_heads, n_chunks

    // --- Tensor-core path (opt-in via EXL3_MMA_ATTN, see attention_mma.cuh).
    if (q_len > 1 && (dim == 128 || dim == 256) && mma_attn_enabled() && mma_shape_ok(dim, G))
    {
        #define LAUNCH_MMA(DIM, GVAL) \
            if (dim == DIM && G == GVAL) { \
                if (kv_append_len > 0) { \
                    kv_cache_update_kernel_paged<DIM><<<grid_up, DIM / 2, 0, stream>>>(PAGED_UPDATE_ARGS); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                auto pl = mma_plan<DIM, GVAL>(bsz, q_len, n_q_heads, n_kv_heads, \
                                              max_total_k_len, ws_numel, q.get_device()); \
                attn_flash_mma_impl<DIM, GVAL, false, 0> \
                    <<<pl.grid, pl.threads, pl.smem, stream>>>( \
                    q_ptr, k_cache_ptr, v_cache_ptr, nullptr, nullptr, nullptr, nullptr, \
                    block_ptr, seqlens_ptr, bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, \
                    num_pages_per_seq, 0, 0.0f, causal, scale, o_ptr, \
                    ws_ptr, pl.num_splits, pl.split_len, pl.fin); \
                cuda_check(cudaPeekAtLastError()); \
                if (!pl.fin) { \
                    attn_reduce_kernel<DIM, GVAL><<<pl.red_grid, DIM, 0, stream>>>( \
                        ws_ptr, o_ptr, bsz, q_len, n_q_heads, pl.num_splits); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                return; \
            }
        LAUNCH_MMA(128, 1) LAUNCH_MMA(128, 2) LAUNCH_MMA(128, 3) LAUNCH_MMA(128, 4)
        LAUNCH_MMA(128, 5) LAUNCH_MMA(128, 6) LAUNCH_MMA(128, 7) LAUNCH_MMA(128, 8)
        LAUNCH_MMA(256, 1) LAUNCH_MMA(256, 2) LAUNCH_MMA(256, 3) LAUNCH_MMA(256, 4)
        LAUNCH_MMA(256, 5) LAUNCH_MMA(256, 6) LAUNCH_MMA(256, 7) LAUNCH_MMA(256, 8)
        #undef LAUNCH_MMA
    }

    // --- Query-tiled path: every q_len > 1 case (prompt ingestion AND the
    // speculative-verify step). One CTA covers 8 query positions with the KV tile
    // staged in shared memory, so the KV range is read once per 8 queries instead
    // of once per query — the scalar path below re-read the whole prefix for
    // every query token, and the flash-decode path above does the same across its
    // q_len programs (a 5-token MTP verify at 50k context read the KV five times
    // over, ~23 ms of the ~50 ms step).
    //
    // Split-K over the KV range keeps the GPU busy when q_len is short: with only
    // ceil(q_len/8) * n_kv_heads * bsz CTAs a verify step would occupy 4 SMs.
    // Partials go to the shared fp32 workspace and `attn_reduce_kernel` combines
    // them, exactly as the flash-decode path does.
    if (q_len > 1 && (dim == 128 || dim == 256))
    {
        constexpr int PF_WARPS = 8;
        constexpr int PF_BN    = 32;
        const int64_t q_tiles = (q_len + PF_WARPS - 1) / PF_WARPS;
        int num_sms = 84;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, q.get_device());
        const int64_t programs = q_tiles * n_kv_heads * bsz;
        int64_t num_splits = (int64_t)(4 * num_sms) / (programs > 0 ? programs : 1);
        const int64_t scap = (max_total_k_len + 4 * PF_BN - 1) / (4 * PF_BN);
        if (num_splits > scap) num_splits = scap;
        if (num_splits > 128) num_splits = 128;
        if (num_splits < 1)   num_splits = 1;
        while (num_splits > 1 &&
               (uint64_t)bsz * q_len * n_q_heads * num_splits * (dim + 2) > ws_numel)
            num_splits >>= 1;
        int64_t split_len =
            (((max_total_k_len + num_splits - 1) / num_splits) + PF_BN - 1) / PF_BN * PF_BN;
        if (split_len < PF_BN) split_len = PF_BN;
        const bool pf_fin = (num_splits == 1);
        dim3 gpf((uint32_t)q_tiles, (uint32_t)n_kv_heads, (uint32_t)(bsz * num_splits));
        dim3 gpfred((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

        #define LAUNCH_PREFILL(DIM, GVAL) \
            if (dim == DIM && G == GVAL) { \
                if (kv_append_len > 0) { \
                    kv_cache_update_kernel_paged<DIM><<<grid_up, DIM / 2, 0, stream>>>(PAGED_UPDATE_ARGS); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                attn_prefill_paged_kernel<DIM, GVAL><<<gpf, 256, 0, stream>>>( \
                    q_ptr, k_cache_ptr, v_cache_ptr, block_ptr, seqlens_ptr, \
                    bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, \
                    num_pages_per_seq, causal, scale, o_ptr, \
                    ws_ptr, num_splits, split_len, pf_fin); \
                cuda_check(cudaPeekAtLastError()); \
                if (!pf_fin) { \
                    attn_reduce_kernel<DIM, GVAL><<<gpfred, DIM, 0, stream>>>( \
                        ws_ptr, o_ptr, bsz, q_len, n_q_heads, num_splits); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                return; \
            }
        LAUNCH_PREFILL(128, 1) LAUNCH_PREFILL(128, 2) LAUNCH_PREFILL(128, 3) LAUNCH_PREFILL(128, 4)
        LAUNCH_PREFILL(128, 5) LAUNCH_PREFILL(128, 6) LAUNCH_PREFILL(128, 7) LAUNCH_PREFILL(128, 8)
        LAUNCH_PREFILL(256, 1) LAUNCH_PREFILL(256, 2) LAUNCH_PREFILL(256, 3) LAUNCH_PREFILL(256, 4)
        LAUNCH_PREFILL(256, 5) LAUNCH_PREFILL(256, 6) LAUNCH_PREFILL(256, 7) LAUNCH_PREFILL(256, 8)
        #undef LAUNCH_PREFILL
    }

    // --- Flash-decoding path: short q (decode + n-gram verify). Split-K over KV;
    // partials land in the same fp32 workspace and are combined by
    // `attn_reduce_kernel`. Falls through to the scalar chunked path for
    // (dim, G) it has no instantiation for.
    //
    // Splits used to target ~2 CTAs per SM. Each CTA is only D/2 threads (4 warps
    // at head_dim 256), so 2 CTAs/SM leaves an SM running 8 of its 48 resident
    // warps and the whole kernel latency-bound: a speculative verify step
    // (q_len 5, 4 KV heads => 20 programs) got 12 splits, i.e. 240 CTAs for 128
    // SMs, each serially walking ~5.5k KV positions. Targeting SM_TARGET CTAs
    // per SM instead keeps long-context decode close to memory bandwidth — this
    // is what made decode throughput fall off with context length while the
    // Python port's stayed flat. The workspace/cap clamps below still apply.
    if (q_len <= 16 && (dim == 64 || dim == 128 || dim == 256))
    {
        const int64_t DEC_BN = 32;   // split length granularity (multiple of every DecTile<D>::N)
        const int64_t SM_TARGET = 8; // CTAs resident per SM to aim for
        int num_sms = 84;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, q.get_device());
        const int64_t programs = bsz * n_kv_heads * q_len;
        int64_t num_splits = (int64_t)(SM_TARGET * num_sms) / (programs > 0 ? programs : 1);
        const int64_t cap = (max_total_k_len + 4 * DEC_BN - 1) / (4 * DEC_BN);
        if (num_splits > cap)  num_splits = cap;
        if (num_splits > 128)  num_splits = 128;
        if (num_splits < 1)    num_splits = 1;
        while (num_splits > 1 &&
               (uint64_t)bsz * q_len * n_q_heads * num_splits * (dim + 2) > ws_numel)
            num_splits >>= 1;
        int64_t split_len =
            (((max_total_k_len + num_splits - 1) / num_splits) + DEC_BN - 1)
            / DEC_BN * DEC_BN;
        if (split_len < DEC_BN) split_len = DEC_BN;
        const bool fin = (num_splits == 1);
        dim3 gdec((uint32_t)programs, (uint32_t)num_splits);
        dim3 gred((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

        #define LAUNCH_DECODE(DIM, GVAL) \
            if (dim == DIM && G == GVAL) { \
                if (kv_append_len > 0) { \
                    kv_cache_update_kernel_paged<DIM><<<grid_up, DIM / 2, 0, stream>>>(PAGED_UPDATE_ARGS); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                attn_decode_split_kernel<DIM, GVAL><<<gdec, DIM / 2, 0, stream>>>( \
                    q_ptr, k_cache_ptr, v_cache_ptr, block_ptr, seqlens_ptr, ws_ptr, o_ptr, \
                    bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, num_splits, split_len, \
                    num_pages_per_seq, causal, scale, fin); \
                cuda_check(cudaPeekAtLastError()); \
                if (!fin) { \
                    attn_reduce_kernel<DIM, GVAL><<<gred, DIM, 0, stream>>>( \
                        ws_ptr, o_ptr, bsz, q_len, n_q_heads, num_splits); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                return; \
            }
        LAUNCH_DECODE(64, 1)  LAUNCH_DECODE(64, 2)  LAUNCH_DECODE(64, 4)  LAUNCH_DECODE(64, 8)
        LAUNCH_DECODE(128, 1) LAUNCH_DECODE(128, 2) LAUNCH_DECODE(128, 3) LAUNCH_DECODE(128, 4)
        LAUNCH_DECODE(128, 5) LAUNCH_DECODE(128, 6) LAUNCH_DECODE(128, 7) LAUNCH_DECODE(128, 8)
        LAUNCH_DECODE(256, 1) LAUNCH_DECODE(256, 2) LAUNCH_DECODE(256, 3) LAUNCH_DECODE(256, 4)
        LAUNCH_DECODE(256, 5) LAUNCH_DECODE(256, 6) LAUNCH_DECODE(256, 7) LAUNCH_DECODE(256, 8)
        #undef LAUNCH_DECODE
    }

    #define LAUNCH_PAGED(DIM, GVAL) \
        if (dim == DIM && G == GVAL) { \
            if (kv_append_len > 0) { \
                kv_cache_update_kernel_paged<DIM><<<grid_up, DIM / 2, 0, stream>>>(PAGED_UPDATE_ARGS); \
                cuda_check(cudaPeekAtLastError()); \
            } \
            const size_t smem_bytes = \
                (size_t)DIM * sizeof(half) + (size_t)(DIM / 32) * G * sizeof(float); \
            attn_chunked_paged_kernel<DIM, GVAL><<<grid1, DIM, smem_bytes, stream>>>(PAGED_ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            if (!single) { \
                attn_reduce_kernel<DIM, GVAL><<<grid2, DIM, 0, stream>>>(ARGS2); \
                cuda_check(cudaPeekAtLastError()); \
            } \
        }

    #define LAUNCH_PAGED_512(GVAL) \
        if (dim == 512 && G == GVAL) { \
            if (kv_append_len > 0) { \
                kv_cache_update_kernel_paged<512><<<grid_up, 256, 0, stream>>>(PAGED_UPDATE_ARGS); \
                cuda_check(cudaPeekAtLastError()); \
            } \
            attn_chunked_paged_kernel_512x256<GVAL><<<grid1, 256, 0, stream>>>(PAGED_ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            attn_reduce_kernel_512x256<<<grid2, 256, 0, stream>>>(ARGS2); \
            cuda_check(cudaPeekAtLastError()); \
        }

    LAUNCH_PAGED_512(1)
    else LAUNCH_PAGED_512(2)
    else LAUNCH_PAGED_512(4)
    else LAUNCH_PAGED_512(8)
    else LAUNCH_PAGED_512(16)
    else LAUNCH_PAGED(64, 1)
    else LAUNCH_PAGED(64, 2)
    else LAUNCH_PAGED(64, 4)
    else LAUNCH_PAGED(64, 8)
    else LAUNCH_PAGED(128, 1)
    else LAUNCH_PAGED(128, 2)
    else LAUNCH_PAGED(128, 4)
    else LAUNCH_PAGED(128, 8)
    else LAUNCH_PAGED(256, 1)
    else LAUNCH_PAGED(256, 2)
    else LAUNCH_PAGED(256, 4)
    else LAUNCH_PAGED(256, 8)
    // exl3-rs: odd GQA ratios (Qwen3.5 is 24 q / 4 kv = 6 @ head_dim 256). The
    // kernel is generic over G; these are just extra template instantiations.
    else LAUNCH_PAGED(128, 3)
    else LAUNCH_PAGED(128, 5)
    else LAUNCH_PAGED(128, 6)
    else LAUNCH_PAGED(128, 7)
    else LAUNCH_PAGED(256, 3)
    else LAUNCH_PAGED(256, 5)
    else LAUNCH_PAGED(256, 6)
    else LAUNCH_PAGED(256, 7)
    else TORCH_CHECK(false,
        "head_dim must be 64, 128, 256, or 512, num_kv_groups must be 1, 2, 4 or 8 (or 16 for head_dim 512)");

    #undef LAUNCH_PAGED_512
    #undef LAUNCH_PAGED
    #undef ARGS2
    #undef PAGED_ARGS1
    #undef PAGED_UPDATE_ARGS
}


// ---------------------------------------------------------------------------
// Online-dequant paged attention against a QUANTIZED KV cache (qk/sk/qv/sv) —
// Python's zero-scratch `_fns_qc` path. Fresh rows `k`/`v` are quantized into the
// packed store first, then `attn_decode_split_kernel_q` dequantizes each KV tile
// warp-collectively into shared memory as it goes (no fp16 cache-sized scratch).
// Short q only (decode / spec-verify), dim in {128, 256}, bits in {4, 6, 8};
// callers handle the long-q prefill case with a compact window instead.
// ---------------------------------------------------------------------------
void bighead_attn_paged_q
(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& qk,
    const at::Tensor& sk,
    const at::Tensor& qv,
    const at::Tensor& sv,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    const at::Tensor& o,
    int   kv_chunk_size,
    bool  causal,
    float sm_scale,
    float compand_a
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k, kHalf);
    TORCH_CHECK_DTYPE(v, kHalf);
    TORCH_CHECK_DTYPE(qk, kInt);
    TORCH_CHECK_DTYPE(qv, kInt);
    TORCH_CHECK_DTYPE(sk, kHalf);
    TORCH_CHECK_DTYPE(sv, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    TORCH_CHECK_DTYPE(block_table, kInt);
    TORCH_CHECK_DTYPE(cache_seqlens, kInt);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && o.dim() == 4, "q/k/v/o must be rank-4");
    TORCH_CHECK(qk.dim() == 3 && qv.dim() == 3 && sk.dim() == 3 && sv.dim() == 3,
               "packed cache tensors must be [num_pages, 256, ...]");

    const int64_t bsz         = q.size(0);
    const int64_t q_len       = q.size(1);
    const int64_t n_q_heads   = q.size(2);
    const int64_t dim         = q.size(3);
    const int64_t kv_append_len = k.size(1);
    const int64_t n_kv_heads  = k.size(2);
    const int64_t num_pages_per_seq = block_table.size(1);
    const int64_t groups_per_token  = sk.size(2);
    const int64_t k_bits = qk.size(2) / groups_per_token;
    const int64_t v_bits = qv.size(2) / groups_per_token;

    TORCH_CHECK(groups_per_token == n_kv_heads * dim / 32, "packed cache group count mismatch");
    TORCH_CHECK(k_bits == v_bits, "k_bits/v_bits must match for the online-dequant path");
    TORCH_CHECK(n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads");
    const int64_t G = n_q_heads / n_kv_heads;
    TORCH_CHECK(G <= G_MAX, "GQA ratio ", G, " exceeds G_MAX=", G_MAX);

    // 1. quantize the fresh K/V rows into the packed store (contiguous input).
    if (kv_append_len > 0)
        quant_cache_paged(k, qk, sk, v, qv, sv, cache_seqlens, block_table,
                          PAGE_SIZE, (int) kv_append_len, compand_a, true);

    const int64_t max_total_k_len = num_pages_per_seq * PAGE_SIZE;
    const uint64_t ws_numel = WORKSPACE_SIZE / sizeof(float);
    float* ws_ptr = (float*) DevCtx::instance().get_ws(q.get_device());
    const float scale = sm_scale == 0.0f ? rsqrtf((float) dim) : sm_scale;

    const half*     q_ptr  = (const half*) q.data_ptr();
    const uint32_t* qk_ptr = (const uint32_t*) qk.data_ptr();
    const half*     sk_ptr = (const half*) sk.data_ptr();
    const uint32_t* qv_ptr = (const uint32_t*) qv.data_ptr();
    const half*     sv_ptr = (const half*) sv.data_ptr();
    const int32_t*  blk    = block_table.data_ptr<int32_t>();
    const int32_t*  sql    = cache_seqlens.data_ptr<int32_t>();
    half*           o_ptr  = (half*) o.data_ptr();

    TORCH_CHECK((dim == 128 || dim == 256) && (k_bits == 4 || k_bits == 6 || k_bits == 8),
                "bighead_attn_paged_q: online path needs head_dim 128/256, bits 4/6/8; "
                "caller must use the compact-window path otherwise");
    TORCH_CHECK(k_bits == v_bits, "bighead_attn_paged_q: k_bits must equal v_bits");

    // --- Tensor-core path (opt-in via EXL3_MMA_ATTN, see attention_mma.cuh).
    if (q_len > 1 && mma_attn_enabled() && mma_shape_ok(dim, G))
    {
        #define LAUNCH_MMA_Q(DIM, GVAL, BV) \
            if (dim == DIM && G == GVAL && k_bits == BV) { \
                auto pl = mma_plan<DIM, GVAL>(bsz, q_len, n_q_heads, n_kv_heads, \
                                              max_total_k_len, ws_numel, q.get_device()); \
                attn_flash_mma_impl<DIM, GVAL, true, BV> \
                    <<<pl.grid, pl.threads, pl.smem, stream>>>( \
                    q_ptr, nullptr, nullptr, qk_ptr, sk_ptr, qv_ptr, sv_ptr, \
                    blk, sql, bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, \
                    num_pages_per_seq, groups_per_token, compand_a, causal, scale, o_ptr, \
                    ws_ptr, pl.num_splits, pl.split_len, pl.fin); \
                cuda_check(cudaPeekAtLastError()); \
                if (!pl.fin) { \
                    attn_reduce_kernel<DIM, GVAL><<<pl.red_grid, DIM, 0, stream>>>( \
                        ws_ptr, o_ptr, bsz, q_len, n_q_heads, pl.num_splits); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                return; \
            }
        #define LAUNCH_MMA_Q_G(DIM, BV) \
            LAUNCH_MMA_Q(DIM, 1, BV) LAUNCH_MMA_Q(DIM, 2, BV) LAUNCH_MMA_Q(DIM, 3, BV) \
            LAUNCH_MMA_Q(DIM, 4, BV) LAUNCH_MMA_Q(DIM, 5, BV) LAUNCH_MMA_Q(DIM, 6, BV) \
            LAUNCH_MMA_Q(DIM, 7, BV) LAUNCH_MMA_Q(DIM, 8, BV)
        LAUNCH_MMA_Q_G(128, 4) LAUNCH_MMA_Q_G(128, 6) LAUNCH_MMA_Q_G(128, 8)
        LAUNCH_MMA_Q_G(256, 4) LAUNCH_MMA_Q_G(256, 6) LAUNCH_MMA_Q_G(256, 8)
        #undef LAUNCH_MMA_Q_G
        #undef LAUNCH_MMA_Q
    }

    // --- Query-tiled online-dequant path (q_len > 1): prompt ingestion and the
    // speculative-verify step. One CTA dequantizes a KV tile once and shares it
    // across 8 query warps x G heads, so neither the fp16 window nor the
    // per-query re-dequant of the decode kernel below is needed.
    if (q_len > 1)
    {
        constexpr int PF_WARPS = 8;
        constexpr int PF_BN    = 32;
        const int64_t q_tiles = (q_len + PF_WARPS - 1) / PF_WARPS;
        int num_sms = 84;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, q.get_device());
        const int64_t programs = q_tiles * n_kv_heads * bsz;
        int64_t num_splits = (int64_t)(4 * num_sms) / (programs > 0 ? programs : 1);
        const int64_t scap = (max_total_k_len + 4 * PF_BN - 1) / (4 * PF_BN);
        if (num_splits > scap) num_splits = scap;
        if (num_splits > 128) num_splits = 128;
        if (num_splits < 1)   num_splits = 1;
        while (num_splits > 1 &&
               (uint64_t)bsz * q_len * n_q_heads * num_splits * (dim + 2) > ws_numel)
            num_splits >>= 1;
        int64_t split_len =
            (((max_total_k_len + num_splits - 1) / num_splits) + PF_BN - 1) / PF_BN * PF_BN;
        if (split_len < PF_BN) split_len = PF_BN;
        const bool pf_fin = (num_splits == 1);
        dim3 gpf((uint32_t) q_tiles, (uint32_t) n_kv_heads, (uint32_t)(bsz * num_splits));
        dim3 gpfred((uint32_t)(bsz * q_len), (uint32_t) n_q_heads);

        #define LAUNCH_PREFILL_Q(DIM, GVAL, BV) \
            if (dim == DIM && G == GVAL && k_bits == BV) { \
                attn_prefill_paged_kernel_q<DIM, GVAL, BV><<<gpf, 256, 0, stream>>>( \
                    q_ptr, qk_ptr, sk_ptr, qv_ptr, sv_ptr, blk, sql, \
                    bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, \
                    num_pages_per_seq, groups_per_token, compand_a, causal, scale, o_ptr, \
                    ws_ptr, num_splits, split_len, pf_fin); \
                cuda_check(cudaPeekAtLastError()); \
                if (!pf_fin) { \
                    attn_reduce_kernel<DIM, GVAL><<<gpfred, DIM, 0, stream>>>( \
                        ws_ptr, o_ptr, bsz, q_len, n_q_heads, num_splits); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                return; \
            }
        #define LAUNCH_PREFILL_Q_G(DIM, BV) \
            LAUNCH_PREFILL_Q(DIM, 1, BV) LAUNCH_PREFILL_Q(DIM, 2, BV) LAUNCH_PREFILL_Q(DIM, 3, BV) \
            LAUNCH_PREFILL_Q(DIM, 4, BV) LAUNCH_PREFILL_Q(DIM, 5, BV) LAUNCH_PREFILL_Q(DIM, 6, BV) \
            LAUNCH_PREFILL_Q(DIM, 7, BV) LAUNCH_PREFILL_Q(DIM, 8, BV)
        LAUNCH_PREFILL_Q_G(128, 4) LAUNCH_PREFILL_Q_G(128, 6) LAUNCH_PREFILL_Q_G(128, 8)
        LAUNCH_PREFILL_Q_G(256, 4) LAUNCH_PREFILL_Q_G(256, 6) LAUNCH_PREFILL_Q_G(256, 8)
        #undef LAUNCH_PREFILL_Q_G
        #undef LAUNCH_PREFILL_Q
    }

    {
        const int64_t DEC_BN = 32;
        int num_sms = 84;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, q.get_device());
        const int64_t programs = bsz * n_kv_heads * q_len;
        int64_t num_splits = (int64_t)(2 * num_sms) / (programs > 0 ? programs : 1);
        const int64_t cap = (max_total_k_len + 4 * DEC_BN - 1) / (4 * DEC_BN);
        if (num_splits > cap)  num_splits = cap;
        if (num_splits > 128)  num_splits = 128;
        if (num_splits < 1)    num_splits = 1;
        while (num_splits > 1 &&
               (uint64_t)bsz * q_len * n_q_heads * num_splits * (dim + 2) > ws_numel)
            num_splits >>= 1;
        int64_t split_len =
            (((max_total_k_len + num_splits - 1) / num_splits) + DEC_BN - 1) / DEC_BN * DEC_BN;
        if (split_len < DEC_BN) split_len = DEC_BN;
        const bool fin = (num_splits == 1);
        dim3 gdec((uint32_t) programs, (uint32_t) num_splits);
        dim3 gred((uint32_t)(bsz * q_len), (uint32_t) n_q_heads);

        #define LAUNCH_DECODE_Q(DIM, GVAL, BV) \
            if (dim == DIM && G == GVAL && k_bits == BV) { \
                attn_decode_split_kernel_q<DIM, GVAL, BV><<<gdec, DIM / 2, 0, stream>>>( \
                    q_ptr, qk_ptr, sk_ptr, qv_ptr, sv_ptr, blk, sql, ws_ptr, o_ptr, \
                    bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, num_splits, split_len, \
                    num_pages_per_seq, groups_per_token, compand_a, causal, scale, fin); \
                cuda_check(cudaPeekAtLastError()); \
                if (!fin) { \
                    attn_reduce_kernel<DIM, GVAL><<<gred, DIM, 0, stream>>>( \
                        ws_ptr, o_ptr, bsz, q_len, n_q_heads, num_splits); \
                    cuda_check(cudaPeekAtLastError()); \
                } \
                return; \
            }
        #define LAUNCH_DECODE_Q_G(DIM, BV) \
            LAUNCH_DECODE_Q(DIM, 1, BV) LAUNCH_DECODE_Q(DIM, 2, BV) LAUNCH_DECODE_Q(DIM, 3, BV) \
            LAUNCH_DECODE_Q(DIM, 4, BV) LAUNCH_DECODE_Q(DIM, 5, BV) LAUNCH_DECODE_Q(DIM, 6, BV) \
            LAUNCH_DECODE_Q(DIM, 7, BV) LAUNCH_DECODE_Q(DIM, 8, BV)
        LAUNCH_DECODE_Q_G(128, 4) LAUNCH_DECODE_Q_G(128, 6) LAUNCH_DECODE_Q_G(128, 8)
        LAUNCH_DECODE_Q_G(256, 4) LAUNCH_DECODE_Q_G(256, 6) LAUNCH_DECODE_Q_G(256, 8)
        #undef LAUNCH_DECODE_Q_G
        #undef LAUNCH_DECODE_Q
        TORCH_CHECK(false, "bighead_attn_paged_q: no kernel for (head_dim ", dim,
                   ", GQA ", G, ", bits ", k_bits, ")");
    }
}

void bighead_attn
(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& o,
    // const at::Tensor& workspace,
    int kv_chunk_size,
    bool causal,
    float sm_scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k, kHalf);
    TORCH_CHECK_DTYPE(v, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    // TORCH_CHECK_DTYPE(workspace, kFloat);

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
    // TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");

    TORCH_CHECK(q.dim() == 4, "q must be rank-4");
    TORCH_CHECK(k.dim() == 4, "k must be rank-4");
    TORCH_CHECK(v.dim() == 4, "v must be rank-4");
    TORCH_CHECK(o.dim() == 4, "o must be rank-4");

    const int64_t bsz        = q.size(0);
    const int64_t q_len      = q.size(1);
    const int64_t n_q_heads  = q.size(2);
    const int64_t dim        = q.size(3);
    const int64_t kv_len     = k.size(1);
    const int64_t n_kv_heads = k.size(2);
    const int64_t G          = n_q_heads / n_kv_heads;

    TORCH_CHECK(k.size(0) == bsz, "k batch mismatch");
    TORCH_CHECK(v.size(0) == bsz, "v batch mismatch");
    TORCH_CHECK(v.size(1) == kv_len, "v kv_len mismatch");
    TORCH_CHECK(v.size(2) == n_kv_heads, "v n_kv_heads mismatch");
    TORCH_CHECK(k.size(3) == dim, "k head_dim mismatch");
    TORCH_CHECK(v.size(3) == dim, "v head_dim mismatch");
    TORCH_CHECK(o.size(0) == bsz, "o batch mismatch");
    TORCH_CHECK(o.size(1) == q_len, "o q_len mismatch");
    TORCH_CHECK(o.size(2) == n_q_heads, "o n_q_heads mismatch");
    TORCH_CHECK(o.size(3) == dim, "o head_dim mismatch");

    TORCH_CHECK(n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads");
    TORCH_CHECK(G <= G_MAX, "GQA ratio ", G, " exceeds G_MAX=", G_MAX);
    TORCH_CHECK(kv_chunk_size > 0, "kv_chunk_size must be positive");

    const uint64_t ws_numel      = WORKSPACE_SIZE / sizeof(float);
    int64_t n_chunks = (kv_len + kv_chunk_size - 1) / kv_chunk_size;
    while (n_chunks > 1)
    {
        const uint64_t ws_needed = (uint64_t)bsz * (uint64_t)q_len * (uint64_t)n_q_heads * (uint64_t)n_chunks * (uint64_t)(dim + 2);
        if (ws_needed <= ws_numel) break;
        kv_chunk_size *= 2;
        n_chunks = (kv_len + kv_chunk_size - 1) / kv_chunk_size;
    }
    const bool single = (n_chunks == 1) && (dim != 512);
    if (n_chunks == 1 && dim == 512)
    {
        const uint64_t ws_needed = (uint64_t)bsz * (uint64_t)q_len * (uint64_t)n_q_heads * (uint64_t)(dim + 2);
        TORCH_CHECK(ws_needed <= ws_numel, "head_dim 512 long-q attention exceeds workspace");
    }
    float* ws_ptr                = (float*) DevCtx::instance().get_ws(q.get_device());

    const half* q_ptr = (const half*) q.data_ptr();
    const half* k_ptr = (const half*) k.data_ptr();
    const half* v_ptr = (const half*) v.data_ptr();
    half* o_ptr       = (half*) o.data_ptr();

    dim3 grid1((uint32_t)(bsz * q_len), (uint32_t)n_kv_heads, (uint32_t)n_chunks);
    dim3 grid2((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

    dim3 grid1_512((uint32_t)(bsz * q_len), (uint32_t)n_kv_heads, (uint32_t)n_chunks);
    dim3 grid2_512((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

    const float scale = sm_scale == 0.0f ? rsqrtf((float) dim) : sm_scale;

    #define ARGS1 \
        q_ptr, k_ptr, v_ptr, ws_ptr, \
        bsz, q_len, kv_len, n_q_heads, n_kv_heads, n_chunks, \
        (int64_t)kv_chunk_size, causal, scale, o_ptr, single

    #define ARGS2 \
        ws_ptr, o_ptr, bsz, q_len, n_q_heads, n_chunks

    #define LAUNCH(DIM, GVAL) \
        if (dim == DIM && G == GVAL) { \
            const size_t smem_bytes = \
                (size_t)DIM * sizeof(half) + (size_t)(DIM / 32) * G * sizeof(float); \
            attn_chunked_kernel<DIM, GVAL><<<grid1, DIM, smem_bytes, stream>>>(ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            if (!single) { \
                attn_reduce_kernel<DIM, GVAL><<<grid2, DIM, 0, stream>>>(ARGS2); \
                cuda_check(cudaPeekAtLastError()); \
            } \
        }

    #define LAUNCH_512(GVAL) \
        if (dim == 512 && G == GVAL) { \
            attn_chunked_kernel_512x256<GVAL><<<grid1_512, 256, 0, stream>>>(ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            attn_reduce_kernel_512x256<<<grid2_512, 256, 0, stream>>>(ARGS2); \
            cuda_check(cudaPeekAtLastError()); \
        }

    LAUNCH_512(1)
    else LAUNCH_512(2)
    else LAUNCH_512(4)
    else LAUNCH_512(8)
    else LAUNCH_512(16)
    else LAUNCH(64, 1)
    else LAUNCH(64, 2)
    else LAUNCH(64, 4)
    else LAUNCH(64, 8)
    else LAUNCH(128, 1)
    else LAUNCH(128, 2)
    else LAUNCH(128, 4)
    else LAUNCH(128, 8)
    else LAUNCH(256, 1)
    else LAUNCH(256, 2)
    else LAUNCH(256, 4)
    else LAUNCH(256, 8)
    // exl3-rs: odd GQA ratios (see the paged dispatch above)
    else LAUNCH(128, 3)
    else LAUNCH(128, 5)
    else LAUNCH(128, 6)
    else LAUNCH(128, 7)
    else LAUNCH(256, 3)
    else LAUNCH(256, 5)
    else LAUNCH(256, 6)
    else LAUNCH(256, 7)
    else TORCH_CHECK(false, "head_dim must be 64, 128, 256, or 512, num_kv_groups must be 1, 2, 4 or 8 (or 16 for head_dim 512)");

    #undef LAUNCH_512
    #undef LAUNCH
    #undef ARGS2
    #undef ARGS1
}

size_t bighead_attn_workspace_size
(
    int bsz,
    int q_len,
    int n_q_heads,
    int max_kv_len,  // or PAGE_SIZE * max_pages_per_seq
    int kv_chunk_size,
    int dim
)
{
    const int64_t n_chunks = (max_kv_len + kv_chunk_size - 1) / kv_chunk_size;
    return
        (size_t)bsz *
        (size_t)q_len *
        (size_t)n_q_heads *
        (size_t)n_chunks *
        (size_t)(dim + 2);
}
