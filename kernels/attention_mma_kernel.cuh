// Tensor-core paged flash attention kernel — see attention_mma.cuh for the
// fragment layouts this file assumes.
//
// Rows of the mma M dimension are (query position, query head) PAIRS, flattened
// as `row = q_idx * G + head`. Every query head that shares a kv head attends
// over the same K/V, so they can all be stacked into one 16-row mma tile. That
// matters most for the speculative-verify step: with one head per tile a 5-token
// verify filled 5 of every 16 rows, wasting ~3x of the tensor cores; stacked,
// G*q_len = 30 rows fill two tiles almost exactly.
//
// A CTA covers RT row tiles for one kv head and one KV split, with PAIR warps
// per tile each owning D/PAIR of the head dim (so the O accumulator is always 64
// registers per thread). The CTA shape no longer depends on G.
#pragma once

#include "attention_mma.cuh"

template<int D>
struct MmaCfg
{
    static constexpr int PAIR   = (D + 127) / 128;  // warps cooperating per tile
    // Row tiles per CTA. 6 keeps the CTA count (and therefore how many times the
    // KV range is staged) the same as one-head-per-tile did for a G=6 prefill,
    // while still letting a short speculative verify pack all its heads into one
    // or two live tiles.
    static constexpr int RT     = 6;
    static constexpr int DW     = D / PAIR;         // head dims owned by one warp
    static constexpr int WARPS  = RT * PAIR;
    static constexpr int THREADS = 32 * WARPS;
    static constexpr int KSTEPS = DW / 16;          // QK k-steps over this warp's dims
    static constexpr int NTILE  = DW / 8;           // PV n-tiles over this warp's dims
    static constexpr int STRIDE = D + MMA_SPAD;     // shared K/V row stride, halfs
};

// grid.x = ceil(q_len * G / (16 * RT)),  grid.y = n_kv_heads,
// grid.z = bsz * num_splits
template<int D, int G, bool QUANT, int BITS>
__global__ __launch_bounds__(MmaCfg<D>::THREADS) void attn_flash_mma_impl
(
    const half*     __restrict__ q,            // [bsz, q_len, n_q_heads, D]
    const half*     __restrict__ k_cache,      // fp16 path: [pages, 256, n_kv, D]
    const half*     __restrict__ v_cache,
    const uint32_t* __restrict__ qk_cache,     // quant path: packed codes
    const half*     __restrict__ sk_cache,
    const uint32_t* __restrict__ qv_cache,
    const half*     __restrict__ sv_cache,
    const int32_t*  __restrict__ block_table,  // [bsz, num_pages_per_seq]
    const int32_t*  __restrict__ cache_seqlens,
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t num_pages_per_seq,
    int64_t groups_per_token,
    float   compand_a,
    bool    causal,
    float   scale,
    half*   __restrict__ o,                    // [bsz, q_len, n_q_heads, D]
    float*  __restrict__ workspace,            // [bsz, q_len, n_q_heads, splits, D+2]
    int64_t num_splits,
    int64_t split_len,
    bool    fin
)
{
    using C = MmaCfg<D>;
    constexpr int PAIR = C::PAIR;
    constexpr int DW   = C::DW;

    const int tid     = threadIdx.x;
    const int warp    = tid / 32;
    const int lane    = tid % 32;
    const int unit    = warp / PAIR;    // which row tile
    const int half_id = warp % PAIR;    // which half of the head dim
    const int wd_base = half_id * DW;

    const int64_t batch   = (int64_t) blockIdx.z % bsz;
    const int64_t split   = (int64_t) blockIdx.z / bsz;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t rows_total = q_len * G;
    const int64_t row_base = ((int64_t) blockIdx.x * C::RT + unit) * 16;
    // A CTA's last row tiles can be entirely past the end (a 5-token verify fills
    // 30 of 96 rows). Those warps still stage K/V and hit every barrier, but skip
    // the mma work — otherwise the dead tiles cost as much as the live ones.
    const bool tile_live = (row_base < rows_total);

    const int64_t total_k = (int64_t) cache_seqlens[batch] + kv_append_len;
    const int32_t* block_row = block_table + batch * num_pages_per_seq;

    // Widest causal reach in the CTA = the last valid query position across ALL
    // of its row tiles. This bound MUST be CTA-uniform: the KV loop below
    // contains __syncthreads(), so every warp has to run the same trip count.
    // (Taking it per row tile is a barrier mismatch — rows are (query, head)
    // pairs now, so different tiles cover different queries.)
    const int64_t last_row =
        min((int64_t) (blockIdx.x + 1) * C::RT * 16 - 1, rows_total - 1);
    const int64_t last_q   = last_row / G;
    const int64_t vis_end =
        min(total_k, (causal ? (total_k - q_len + last_q) : (total_k - 1)) + 1);
    const int64_t kv_lo = split * split_len;
    const int64_t kv_hi = min(vis_end, kv_lo + split_len);

    // Single-buffered KV tile. Double buffering was tried (stage tile i+1 while
    // the mma consumes tile i, one barrier per tile instead of two) and was a
    // clear LOSS — 52.8 -> 84.1 ms on 2048q x 48000ctx Q4 — because doubling the
    // shared-memory footprint halves CTAs/SM. This kernel is occupancy limited,
    // not barrier limited; don't re-try it without shrinking the tile first.
    extern __shared__ __align__(16) char mma_smem[];
    half* k_tile_buf = (half*) mma_smem;
    half* v_tile_buf = k_tile_buf + MMA_BN * C::STRIDE;
    float* s_ex = (float*) (v_tile_buf + MMA_BN * C::STRIDE);   // [RT][32][8]

    // the two M rows this lane owns, resolved to (query position, query head)
    const int r0 = lane / 4;
    int64_t qpos[2], qhd[2];
    bool    rvalid[2];
    #pragma unroll
    for (int rr = 0; rr < 2; rr++)
    {
        const int64_t m = row_base + r0 + rr * 8;
        rvalid[rr] = (m < rows_total);
        const int64_t mm = rvalid[rr] ? m : 0;
        qpos[rr] = mm / G;
        qhd[rr]  = kv_head * G + (mm % G);
    }

    // ---- Q as A fragments, one set per k-step, held for the whole KV loop ----
    uint32_t qa[C::KSTEPS][4];
    {
        const int kc = (lane % 4) * 2;
        #pragma unroll
        for (int s = 0; s < C::KSTEPS; s++)
        {
            #pragma unroll
            for (int hs = 0; hs < 2; hs++)          // k offset 0 / 8
            {
                const int d0 = wd_base + s * 16 + kc + hs * 8;
                #pragma unroll
                for (int rr = 0; rr < 2; rr++)
                {
                    half h0 = __float2half(0.f), h1 = __float2half(0.f);
                    if (rvalid[rr])
                    {
                        const half* qs = q +
                            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)qpos[rr]) *
                              (uint64_t)n_q_heads + (uint64_t)qhd[rr]) * (uint64_t)D);
                        h0 = qs[d0];
                        h1 = qs[d0 + 1];
                    }
                    qa[s][hs * 2 + rr] = pack2(h0, h1);
                }
            }
        }
    }

    float o_acc[C::NTILE][4];
    #pragma unroll
    for (int n = 0; n < C::NTILE; n++)
        #pragma unroll
        for (int i = 0; i < 4; i++) o_acc[n][i] = 0.f;
    float m_run[2] = { -INFINITY, -INFINITY };
    float l_run[2] = { 0.f, 0.f };

    for (int64_t kv0 = kv_lo; kv0 < kv_hi; kv0 += MMA_BN)
    {
        half* k_tile = k_tile_buf;
        half* v_tile = v_tile_buf;
        __syncthreads();
        {
            const int64_t rows = min((int64_t) MMA_BN, total_k - kv0);
            if constexpr (QUANT)
            {
                constexpr int GPH    = D / 32;
                constexpr int QPH    = (GPH + 3) / 4;
                constexpr int KUNITS = MMA_BN * QPH;
                constexpr int UNITS  = 2 * KUNITS;
                for (int u = warp; u < UNITS; u += C::WARPS)
                {
                    const bool is_v = u >= KUNITS;
                    const int  uu   = is_v ? (u - KUNITS) : u;
                    const int  r    = uu / QPH;
                    const int  qd   = uu % QPH;
                    const int  active = min(4, GPH - qd * 4);
                    const int64_t pos = kv0 + min((int64_t) r, rows - 1);
                    const int64_t pp  = block_row[pos >> 8];
                    const int64_t rb  = ((pp << 8) + (pos & 255)) * groups_per_token
                                      + kv_head * GPH + qd * 4;
                    const uint32_t* codes = (is_v ? qv_cache : qk_cache) + rb * BITS;
                    const half*     sc    = (is_v ? sv_cache : sk_cache) + rb;
                    half* out = (is_v ? v_tile : k_tile) + (int64_t) r * C::STRIDE + qd * 128;
                    dequant_block_x4<BITS>(codes, sc, out, active, compand_a);
                }
            }
            else
            {
                constexpr int VEC = 8;                     // halfs per 128-bit load
                constexpr int VPT = MMA_BN * (D / VEC);
                for (int idx = tid; idx < VPT; idx += C::THREADS)
                {
                    const int r  = idx / (D / VEC);
                    const int d8 = idx % (D / VEC);
                    const int64_t pos = kv0 + min((int64_t) r, rows - 1);
                    const int64_t pp  = block_row[pos >> 8];
                    const uint64_t off =
                        ((((uint64_t)((pp << 8) + (pos & 255))) * (uint64_t)n_kv_heads +
                          (uint64_t)kv_head) * (uint64_t)D) + (uint64_t)(d8 * VEC);
                    *(uint4*)(k_tile + (int64_t) r * C::STRIDE + d8 * VEC) =
                        *(const uint4*)(k_cache + off);
                    *(uint4*)(v_tile + (int64_t) r * C::STRIDE + d8 * VEC) =
                        *(const uint4*)(v_cache + off);
                }
            }
        }
        __syncthreads();

        // ---------------- S = Q @ K^T over this warp's dim slice ----------------
        float s_acc[2][4];
        #pragma unroll
        for (int n = 0; n < 2; n++)
            #pragma unroll
            for (int i = 0; i < 4; i++) s_acc[n][i] = 0.f;

        #pragma unroll
        for (int n = 0; n < 2 && tile_live; n++)   // two n8 halves of the 16 KV columns
        {
            #pragma unroll
            for (int s = 0; s < C::KSTEPS; s++)
            {
                // B[n = kv][k = dim] is k_tile row-major: no transpose needed
                const int kvrow = n * 8 + lane / 4;
                const int dcol  = wd_base + s * 16 + (lane % 4) * 2;
                const half* kp  = k_tile + (int64_t) kvrow * C::STRIDE;
                uint32_t b[2] = { pack2(kp[dcol], kp[dcol + 1]),
                                  pack2(kp[dcol + 8], kp[dcol + 9]) };
                mma_m16n8k16(s_acc[n], qa[s], b, s_acc[n]);
            }
        }

        // ---------------- combine the PAIR partial sums ----------------
        if constexpr (PAIR > 1)
        {
            float* ex = s_ex + (int64_t)(unit * 32 + lane) * 8;
            if (half_id == 1)
            {
                #pragma unroll
                for (int n = 0; n < 2; n++)
                    #pragma unroll
                    for (int i = 0; i < 4; i++) ex[n * 4 + i] = s_acc[n][i];
            }
            __syncthreads();
            if (half_id == 0)
            {
                #pragma unroll
                for (int n = 0; n < 2; n++)
                    #pragma unroll
                    for (int i = 0; i < 4; i++) ex[n * 4 + i] += s_acc[n][i];
            }
            __syncthreads();
            #pragma unroll
            for (int n = 0; n < 2; n++)
                #pragma unroll
                for (int i = 0; i < 4; i++) s_acc[n][i] = ex[n * 4 + i];
        }

        if (!tile_live) continue;

        // ---------------- scale + causal mask + online softmax ----------------
        #pragma unroll
        for (int n = 0; n < 2; n++)
        {
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                const int rr  = i / 2;                        // 0 -> row r0, 1 -> row r0+8
                const int col = n * 8 + (lane % 4) * 2 + (i % 2);
                const int64_t kvp = kv0 + col;
                const int64_t lim = causal ? (total_k - q_len + qpos[rr]) : (total_k - 1);
                s_acc[n][i] = (rvalid[rr] && kvp < total_k && kvp <= lim)
                                ? s_acc[n][i] * scale
                                : -INFINITY;
            }
        }

        float p[2][4];
        #pragma unroll
        for (int rr = 0; rr < 2; rr++)
        {
            float mx = fmaxf(fmaxf(s_acc[0][rr * 2], s_acc[0][rr * 2 + 1]),
                             fmaxf(s_acc[1][rr * 2], s_acc[1][rr * 2 + 1]));
            mx = row_reduce_max(mx);
            const float m_new = fmaxf(m_run[rr], mx);
            const float alpha = (m_new == -INFINITY) ? 0.f : __expf(m_run[rr] - m_new);
            float sum = 0.f;
            #pragma unroll
            for (int n = 0; n < 2; n++)
                #pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    const float e = (s_acc[n][rr * 2 + j] == -INFINITY)
                                        ? 0.f
                                        : __expf(s_acc[n][rr * 2 + j] - m_new);
                    p[n][rr * 2 + j] = e;
                    sum += e;
                }
            sum = row_reduce_sum(sum);
            l_run[rr] = alpha * l_run[rr] + sum;
            m_run[rr] = m_new;
            #pragma unroll
            for (int n = 0; n < C::NTILE; n++)
            {
                o_acc[n][rr * 2]     *= alpha;
                o_acc[n][rr * 2 + 1] *= alpha;
            }
        }

        // ---------------- O += P @ V ----------------
        // The S accumulator's two n8 halves are exactly the A fragment for this
        // product: tile 0 -> a0,a1 (k 0..7), tile 1 -> a2,a3 (k 8..15).
        uint32_t pa[4] = {
            pack2(__float2half(p[0][0]), __float2half(p[0][1])),
            pack2(__float2half(p[0][2]), __float2half(p[0][3])),
            pack2(__float2half(p[1][0]), __float2half(p[1][1])),
            pack2(__float2half(p[1][2]), __float2half(p[1][3])),
        };
        #pragma unroll
        for (int n = 0; n < C::NTILE; n++)
        {
            // B[n = dim][k = kv]: transposed read of v_tile[kv][dim]
            const int dcol = wd_base + n * 8 + lane / 4;
            const int kv_a = (lane % 4) * 2;
            uint32_t b[2] = {
                pack2(v_tile[(int64_t)(kv_a) * C::STRIDE + dcol],
                      v_tile[(int64_t)(kv_a + 1) * C::STRIDE + dcol]),
                pack2(v_tile[(int64_t)(kv_a + 8) * C::STRIDE + dcol],
                      v_tile[(int64_t)(kv_a + 9) * C::STRIDE + dcol]),
            };
            mma_m16n8k16(o_acc[n], pa, b, o_acc[n]);
        }
    }

    // ---------------- epilogue ----------------
    #pragma unroll
    for (int rr = 0; rr < 2; rr++)
    {
        if (!rvalid[rr]) continue;
        const uint64_t row_off =
            (((uint64_t)batch * (uint64_t)q_len + (uint64_t)qpos[rr]) * (uint64_t)n_q_heads +
             (uint64_t)qhd[rr]);

        if (fin)
        {
            const float inv = l_run[rr] > 0.f ? (1.f / l_run[rr]) : 0.f;
            half* op = o + row_off * (uint64_t)D;
            #pragma unroll
            for (int n = 0; n < C::NTILE; n++)
            {
                const int d = wd_base + n * 8 + (lane % 4) * 2;
                op[d]     = __float2half(o_acc[n][rr * 2]     * inv);
                op[d + 1] = __float2half(o_acc[n][rr * 2 + 1] * inv);
            }
        }
        else
        {
            float* ws = workspace +
                (row_off * (uint64_t)num_splits + (uint64_t)split) * (uint64_t)(D + 2);
            if (lane % 4 == 0 && half_id == 0)
            {
                ws[0] = m_run[rr];
                ws[1] = l_run[rr];
            }
            #pragma unroll
            for (int n = 0; n < C::NTILE; n++)
            {
                const int d = wd_base + n * 8 + (lane % 4) * 2;
                ws[2 + d]     = o_acc[n][rr * 2];
                ws[2 + d + 1] = o_acc[n][rr * 2 + 1];
            }
        }
    }
}
