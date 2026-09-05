// Fused kernels for the WY-chunked gated delta rule. See src/gdn_chunk.rs for
// the derivation; this file is the fused implementation of the stages that
// version expresses as tensor ops (which measured 5.4 ms against the sequential
// kernel's 3.7 ms — the math was right, the packaging was the problem).
//
// Structure follows fla's `chunk_gated_delta_rule_fwd`: a state scan that keeps
// the recurrent state in REGISTERS across every chunk, then the outputs.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "attention_mma.cuh"   // mma_m16n8k16, pack2

#define GC_C 64          // chunk length (must match gdn_chunk.rs CHUNK)
#define GC_BV 64         // v-dim slice per CTA
// 512 threads = 16 warps per CTA. The scan's parallelism is only
// (vhd/BV) * nv = 96 CTAs, so warps-per-CTA is the only occupancy knob: at 128
// threads that is 3 warps/SM and ~2% of FMA peak, which is exactly the ~3.4 ms
// the scan measured. Each thread then owns khd/TPC state floats.
#define GC_THREADS 512

// NOTE: a scalar fp32 state-scan kernel used to live here (state in registers,
// chunk loop in-kernel, no tensor cores). It was correct and was tuned from
// 4.2 ms to 2.9 ms — 128 -> 512 threads/CTA, then strided rather than blocked
// state ownership to clear shared-memory bank conflicts — but never beat cuBLAS,
// because scalar FMA tops out near 4 TFLOP/s on these shapes. It is superseded
// by the fused tensor-core kernel below; see the notes for the full history.

// ---------------------------------------------------------------------------
// Stages 3+4 fused, on the tensor cores.
//
// The scalar version above is correct but only reaches parity with cuBLAS: at
// ~4 TFLOP/s it cannot pay for itself, and fusing the outputs into it would add
// 1.8x the compute for no gain. Both facts point the same way — the products
// have to run on the tensor cores, and once they do it costs nothing to keep the
// outputs here too, which removes the [nv, nc, khd, vhd] state array (201 MB at
// seq 4096) and the whole separate output pass.
//
// Per chunk this computes
//     U   = Uv_c - W_c S                       (1)
//     O_c = (Q_c S + QK_c U) * a_c * scale     (2),(3)
//     S   = a_last_c * (S + K_c^T U)           (4)
// A warp owns a 16-column slice of the BV v-columns for all of them, so its S
// accumulator is 16 m16n8 tiles = 64 registers per thread — the state stays
// resident exactly as in the scalar kernel.
//
// S is the mma *accumulator* in (4) but the *B operand* in (1) and (2), and U is
// an accumulator in (1) but a B operand in (3) and (4); neither pair shares a
// fragment layout, so both round-trip through shared memory once per chunk
// (16 KB + 8 KB per CTA per chunk, which is cheap next to what it saves).
// 8 warps: the grid is only (vhd/BV) * nv = 96 CTAs, so warps per CTA is again
// the only occupancy knob. Each warp then owns GC_WCOLS = 8 v columns, i.e. one
// m16n8 column tile, and its S accumulator is 8 tiles = 32 registers.
#define GC_WARPS 8
#define GC_WCOLS (GC_BV / GC_WARPS)   // v columns per warp (16)

__global__ __launch_bounds__(32 * GC_WARPS)
void gdn_chunk_fused_kernel
(
    const half* __restrict__ w,        // [nv, nc, C, khd]  (per V head: beta scaling)
    const half* __restrict__ q,        // [nk, nc, C, khd]  (per K head)
    const half* __restrict__ kk,       // [nk, nc, C, khd]  (per K head)
    const half* __restrict__ qkt,      // [nk, nc, C, C]    (per K head, no decay)
    const float* __restrict__ uv,      // [nv, nc, C, vhd]
    const float* __restrict__ alog,    // [nv, nc, C]      cumulative log decay in chunk
    const float* __restrict__ h0,      // [nv, khd, vhd]
    float* __restrict__ out,           // [nv, nc, C, vhd]
    float* __restrict__ ht,            // [nv, khd, vhd]
    int nc,
    int khd,
    int vhd,
    int group,                         // V heads per K head
    float scale
)
{
    const int vblk = blockIdx.x;
    const int head = blockIdx.y;
    const int v0 = vblk * GC_BV;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wc0 = warp * GC_WCOLS;      // this warp's first v column within the slice

    constexpr int NCC = GC_WCOLS / 8;     // m16n8 column tiles per warp
    const int KT = khd / 16;              // k-tiles of 16 along the head dim (8)
    const int CT = GC_C / 16;             // row tiles of 16 along C (4)

    extern __shared__ char gcf_smem[];
    half* sh_s = (half*) gcf_smem;                       // [khd][BV]  S as B operand
    half* sh_u = sh_s + (size_t) 128 * GC_BV;            // [C][BV]    U as B operand
    half* sh_t = sh_u + (size_t) GC_C * GC_BV;           // [C][khd]   W / Q / K tile
    half* sh_qk = sh_t + (size_t) GC_C * 128;            // [C][C]     QK tile
    half* sh_ud = sh_qk + (size_t) GC_C * GC_C;         // [C][BV]    U scaled by D[last]
    __shared__ float sh_al[GC_C];                       // this chunk's log decay

    // ---- S accumulator: 8 row tiles x 2 col tiles per warp = 16 tiles --------
    float s_acc[8][NCC][4];
    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int cc = 0; cc < NCC; cc++)
            #pragma unroll
            for (int i = 0; i < 4; i++) s_acc[r][cc][i] = 0.f;

    // load h0 into the accumulator layout: tile (r, cc) covers rows 16r.., cols 8cc..
    {
        const int r0 = lane / 4, c0 = (lane % 4) * 2;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            #pragma unroll
            for (int cc = 0; cc < NCC; cc++)
            {
                const int col = v0 + wc0 + cc * 8 + c0;
                const float* p = h0 + (size_t) head * khd * vhd + col;
                s_acc[r][cc][0] = p[(size_t)(r * 16 + r0) * vhd];
                s_acc[r][cc][1] = p[(size_t)(r * 16 + r0) * vhd + 1];
                s_acc[r][cc][2] = p[(size_t)(r * 16 + r0 + 8) * vhd];
                s_acc[r][cc][3] = p[(size_t)(r * 16 + r0 + 8) * vhd + 1];
            }
    }

    for (int c = 0; c < nc; ++c)
    {
        const size_t hb = ((size_t) head * nc + c);
        // q / k / QK^T are shared by the `group` V heads of one K head
        const size_t kb = ((size_t)(head / group) * nc + c);
        // Every decay this kernel needs — A, D[t][j] and D[last] — is derived
        // from the cumulative log decay right here. Materializing D as an
        // [nv, nc, C, C] array to fold into QK^T on the host cost 1.5 ms of pure
        // write-then-read-back, and forced QK^T to broaden from nk to nv heads.
        if (tid < GC_C) sh_al[tid] = alog[hb * GC_C + tid];

        // ---- publish S in B-operand layout: sh_s[k][n] --------------------
        __syncthreads();
        {
            const int r0 = lane / 4, c0 = (lane % 4) * 2;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    const int col = wc0 + cc * 8 + c0;
                    half* p = sh_s + (size_t)(r * 16 + r0) * GC_BV + col;
                    p[0] = __float2half(s_acc[r][cc][0]);
                    p[1] = __float2half(s_acc[r][cc][1]);
                    half* p2 = sh_s + (size_t)(r * 16 + r0 + 8) * GC_BV + col;
                    p2[0] = __float2half(s_acc[r][cc][2]);
                    p2[1] = __float2half(s_acc[r][cc][3]);
                }
        }
        // W tile for (1)
        for (int i = tid; i < GC_C * khd; i += 32 * GC_WARPS)
            sh_t[i] = w[hb * GC_C * khd + i];
        __syncthreads();

        // ---- (1) U = Uv - W S ; accumulate in C layout --------------------
        float u_acc[4][NCC][4];
        #pragma unroll
        for (int r = 0; r < CT; r++)
            #pragma unroll
            for (int cc = 0; cc < NCC; cc++)
                #pragma unroll
                for (int i = 0; i < 4; i++) u_acc[r][cc][i] = 0.f;

        #pragma unroll
        for (int r = 0; r < CT; r++)
            for (int kt = 0; kt < KT; kt++)
            {
                uint32_t a[4];
                {
                    const int row = r * 16 + lane / 4, kk0 = kt * 16 + (lane % 4) * 2;
                    const half* t0 = sh_t + (size_t) row * khd + kk0;
                    const half* t1 = sh_t + (size_t)(row + 8) * khd + kk0;
                    a[0] = pack2(t0[0], t0[1]);
                    a[1] = pack2(t1[0], t1[1]);
                    a[2] = pack2(t0[8], t0[9]);
                    a[3] = pack2(t1[8], t1[9]);
                }
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    const int n = wc0 + cc * 8 + lane / 4;
                    const int kk0 = kt * 16 + (lane % 4) * 2;
                    uint32_t b[2] = {
                        pack2(sh_s[(size_t) kk0 * GC_BV + n], sh_s[(size_t)(kk0 + 1) * GC_BV + n]),
                        pack2(sh_s[(size_t)(kk0 + 8) * GC_BV + n], sh_s[(size_t)(kk0 + 9) * GC_BV + n]),
                    };
                    mma_m16n8k16(u_acc[r][cc], a, b, u_acc[r][cc]);
                }
            }
        // U = Uv - WS, and publish it as a B operand for (3) and (4)
        {
            const int r0 = lane / 4, c0 = (lane % 4) * 2;
            #pragma unroll
            for (int r = 0; r < CT; r++)
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    const int col = wc0 + cc * 8 + c0;
                    const float* pu = uv + hb * GC_C * vhd + v0 + col;
                    const int rowa = r * 16 + r0, rowb = rowa + 8;
                    const float u0 = pu[(size_t) rowa * vhd] - u_acc[r][cc][0];
                    const float u1 = pu[(size_t) rowa * vhd + 1] - u_acc[r][cc][1];
                    const float u2 = pu[(size_t) rowb * vhd] - u_acc[r][cc][2];
                    const float u3 = pu[(size_t) rowb * vhd + 1] - u_acc[r][cc][3];
                    u_acc[r][cc][0] = u0; u_acc[r][cc][1] = u1;
                    u_acc[r][cc][2] = u2; u_acc[r][cc][3] = u3;
                    sh_u[(size_t) rowa * GC_BV + col] = __float2half(u0);
                    sh_u[(size_t) rowa * GC_BV + col + 1] = __float2half(u1);
                    sh_u[(size_t) rowb * GC_BV + col] = __float2half(u2);
                    sh_u[(size_t) rowb * GC_BV + col + 1] = __float2half(u3);
                    // second copy scaled by D[last], for the state update: the
                    // output term needs Ũ, the state term needs D[last] ⊙ Ũ
                    const float alL = sh_al[GC_C - 1];
                    const float da = __expf(alL - sh_al[rowa]), db = __expf(alL - sh_al[rowb]);
                    sh_ud[(size_t) rowa * GC_BV + col] = __float2half(u0 * da);
                    sh_ud[(size_t) rowa * GC_BV + col + 1] = __float2half(u1 * da);
                    sh_ud[(size_t) rowb * GC_BV + col] = __float2half(u2 * db);
                    sh_ud[(size_t) rowb * GC_BV + col + 1] = __float2half(u3 * db);
                }
        }

        // ---- (2) O = Q S  (reuse sh_t for the Q tile) ---------------------
        __syncthreads();
        for (int i = tid; i < GC_C * khd; i += 32 * GC_WARPS)
            sh_t[i] = q[kb * GC_C * khd + i];
        __syncthreads();
        for (int i = tid; i < GC_C * GC_C; i += 32 * GC_WARPS)
        {
            const int t = i / GC_C, j = i - t * GC_C;
            // tril(Q K^T) . D, with D[t][j] = exp(A_t - A_j) <= 1 for j <= t
            const float qv = __half2float(qkt[kb * GC_C * GC_C + i]);
            sh_qk[i] = __float2half(j <= t ? qv * __expf(sh_al[t] - sh_al[j]) : 0.f);
        }
        __syncthreads();

        float o_acc[4][NCC][4];
        #pragma unroll
        for (int r = 0; r < CT; r++)
            #pragma unroll
            for (int cc = 0; cc < NCC; cc++)
                #pragma unroll
                for (int i = 0; i < 4; i++) o_acc[r][cc][i] = 0.f;

        #pragma unroll
        for (int r = 0; r < CT; r++)
        {
            for (int kt = 0; kt < KT; kt++)
            {
                uint32_t a[4];
                {
                    const int row = r * 16 + lane / 4, kk0 = kt * 16 + (lane % 4) * 2;
                    const half* t0 = sh_t + (size_t) row * khd + kk0;
                    const half* t1 = sh_t + (size_t)(row + 8) * khd + kk0;
                    a[0] = pack2(t0[0], t0[1]);
                    a[1] = pack2(t1[0], t1[1]);
                    a[2] = pack2(t0[8], t0[9]);
                    a[3] = pack2(t1[8], t1[9]);
                }
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    const int n = wc0 + cc * 8 + lane / 4;
                    const int kk0 = kt * 16 + (lane % 4) * 2;
                    uint32_t b[2] = {
                        pack2(sh_s[(size_t) kk0 * GC_BV + n], sh_s[(size_t)(kk0 + 1) * GC_BV + n]),
                        pack2(sh_s[(size_t)(kk0 + 8) * GC_BV + n], sh_s[(size_t)(kk0 + 9) * GC_BV + n]),
                    };
                    mma_m16n8k16(o_acc[r][cc], a, b, o_acc[r][cc]);
                }
            }
            // O = A * (Q S) so far; the QK.U term already carries D inside qkt
            {
                const int rr0 = r * 16 + lane / 4;
                const float sa = __expf(sh_al[rr0]), sb = __expf(sh_al[rr0 + 8]);
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    o_acc[r][cc][0] *= sa; o_acc[r][cc][1] *= sa;
                    o_acc[r][cc][2] *= sb; o_acc[r][cc][3] *= sb;
                }
            }
            // ---- (3) O += QK U  (k runs over C) --------------------------
            for (int kt = 0; kt < CT; kt++)
            {
                uint32_t a[4];
                {
                    const int row = r * 16 + lane / 4, kk0 = kt * 16 + (lane % 4) * 2;
                    const half* t0 = sh_qk + (size_t) row * GC_C + kk0;
                    const half* t1 = sh_qk + (size_t)(row + 8) * GC_C + kk0;
                    a[0] = pack2(t0[0], t0[1]);
                    a[1] = pack2(t1[0], t1[1]);
                    a[2] = pack2(t0[8], t0[9]);
                    a[3] = pack2(t1[8], t1[9]);
                }
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    const int n = wc0 + cc * 8 + lane / 4;
                    const int kk0 = kt * 16 + (lane % 4) * 2;
                    uint32_t b[2] = {
                        pack2(sh_u[(size_t) kk0 * GC_BV + n], sh_u[(size_t)(kk0 + 1) * GC_BV + n]),
                        pack2(sh_u[(size_t)(kk0 + 8) * GC_BV + n], sh_u[(size_t)(kk0 + 9) * GC_BV + n]),
                    };
                    mma_m16n8k16(o_acc[r][cc], a, b, o_acc[r][cc]);
                }
            }
        }
        // scale by the per-position decay and write out
        {
            const int r0 = lane / 4, c0 = (lane % 4) * 2;
            #pragma unroll
            for (int r = 0; r < CT; r++)
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    const int col = v0 + wc0 + cc * 8 + c0;
                    const int rowa = r * 16 + r0, rowb = rowa + 8;
                    float* po = out + hb * GC_C * vhd + col;
                    po[(size_t) rowa * vhd] = o_acc[r][cc][0] * scale;
                    po[(size_t) rowa * vhd + 1] = o_acc[r][cc][1] * scale;
                    po[(size_t) rowb * vhd] = o_acc[r][cc][2] * scale;
                    po[(size_t) rowb * vhd + 1] = o_acc[r][cc][3] * scale;
                }
        }

        // ---- (4) S = a_last * (S + K^T U) --------------------------------
        __syncthreads();
        for (int i = tid; i < GC_C * khd; i += 32 * GC_WARPS)
            sh_t[i] = kk[kb * GC_C * khd + i];
        __syncthreads();

        // S <- A_last S  BEFORE accumulating, since the update is now
        // S = A_last S + K^T (D[last] . Ũ) rather than A_last (S + K^T U)
        const float al = __expf(sh_al[GC_C - 1]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            #pragma unroll
            for (int cc = 0; cc < NCC; cc++)
                #pragma unroll
                for (int i = 0; i < 4; i++) s_acc[r][cc][i] *= al;

        #pragma unroll
        for (int r = 0; r < 8; r++)
            for (int kt = 0; kt < CT; kt++)
            {
                // A = K^T: A[m = khd row][k = C] = K[k][m], read transposed
                uint32_t a[4];
                {
                    const int m0 = r * 16 + lane / 4, kk0 = kt * 16 + (lane % 4) * 2;
                    a[0] = pack2(sh_t[(size_t) kk0 * khd + m0], sh_t[(size_t)(kk0 + 1) * khd + m0]);
                    a[1] = pack2(sh_t[(size_t) kk0 * khd + m0 + 8], sh_t[(size_t)(kk0 + 1) * khd + m0 + 8]);
                    a[2] = pack2(sh_t[(size_t)(kk0 + 8) * khd + m0], sh_t[(size_t)(kk0 + 9) * khd + m0]);
                    a[3] = pack2(sh_t[(size_t)(kk0 + 8) * khd + m0 + 8], sh_t[(size_t)(kk0 + 9) * khd + m0 + 8]);
                }
                #pragma unroll
                for (int cc = 0; cc < NCC; cc++)
                {
                    const int n = wc0 + cc * 8 + lane / 4;
                    const int kk0 = kt * 16 + (lane % 4) * 2;
                    uint32_t b[2] = {
                        pack2(sh_ud[(size_t) kk0 * GC_BV + n], sh_ud[(size_t)(kk0 + 1) * GC_BV + n]),
                        pack2(sh_ud[(size_t)(kk0 + 8) * GC_BV + n], sh_ud[(size_t)(kk0 + 9) * GC_BV + n]),
                    };
                    mma_m16n8k16(s_acc[r][cc], a, b, s_acc[r][cc]);
                }
            }
    }

    // ---- final state ------------------------------------------------------
    {
        const int r0 = lane / 4, c0 = (lane % 4) * 2;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            #pragma unroll
            for (int cc = 0; cc < NCC; cc++)
            {
                const int col = v0 + wc0 + cc * 8 + c0;
                float* p = ht + (size_t) head * khd * vhd + col;
                p[(size_t)(r * 16 + r0) * vhd] = s_acc[r][cc][0];
                p[(size_t)(r * 16 + r0) * vhd + 1] = s_acc[r][cc][1];
                p[(size_t)(r * 16 + r0 + 8) * vhd] = s_acc[r][cc][2];
                p[(size_t)(r * 16 + r0 + 8) * vhd + 1] = s_acc[r][cc][3];
            }
    }
}

extern "C" void gdn_chunk_fused
(
    const at::Tensor& w, const at::Tensor& q, const at::Tensor& k, const at::Tensor& qkt,
    const at::Tensor& uv, const at::Tensor& alog, const at::Tensor& h0,
    at::Tensor& out, at::Tensor& ht, double scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int nv = w.size(0), nc = w.size(1), khd = w.size(3);
    const int vhd = uv.size(3);
    const int group = nv / (int) q.size(0);

    TORCH_CHECK(khd == 128 && vhd % GC_BV == 0, "gdn_chunk_fused: needs khd 128");
    TORCH_CHECK(w.size(2) == GC_C, "chunk length must be ", GC_C);

    const size_t smem = ((size_t) 128 * GC_BV + (size_t) 2 * GC_C * GC_BV
                         + (size_t) GC_C * 128 + (size_t) GC_C * GC_C) * sizeof(half);
    cudaFuncSetAttribute(gdn_chunk_fused_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    dim3 blocks(vhd / GC_BV, nv);
    gdn_chunk_fused_kernel<<<blocks, 32 * GC_WARPS, smem, stream>>>(
        (const half*) w.data_ptr(), (const half*) q.data_ptr(), (const half*) k.data_ptr(),
        (const half*) qkt.data_ptr(), (const float*) uv.data_ptr(),
        (const float*) alog.data_ptr(),
        (const float*) h0.data_ptr(), (float*) out.data_ptr(), (float*) ht.data_ptr(),
        nc, khd, vhd, group, (float) scale);
    cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// Stages 1+2 fused: K K^T, the decay ratios, M, the triangular solve, and W/Uv,
// in one kernel per (V head, chunk).
//
// As tensor ops these were 4.7 of the 7.4 ms — not because the arithmetic is
// large (it is ~2 GFLOP) but because each step materialized and re-read a
// [nv, nc, C, C] or [nv, nc, C, khd] array: the decay ratios alone were 1.5 ms
// to write and read back 50 MB. Nothing here reaches global memory except the
// inputs and the two outputs; D is recomputed from the cumulative log decay
// where it is used, exactly as fla does.
//
// The solve is a plain forward substitution over the C rows. It is sequential in
// t, but each of the 256 right-hand-side columns is independent, so one thread
// owns one column and the whole CTA advances a row at a time.
#define GC_WYT 256   // threads = khd + vhd right-hand-side columns

__global__ __launch_bounds__(GC_WYT)
void gdn_chunk_wy_kernel
(
    const half*  __restrict__ kin,    // [nk, nc, C, khd]
    const float* __restrict__ vin,    // [nv, nc, C, vhd]
    const float* __restrict__ beta,   // [nv, nc, C]
    const float* __restrict__ a_log,  // [nv, nc, C]  cumulative log decay in chunk
    float* __restrict__ wout,         // [nv, nc, C, khd]
    float* __restrict__ uvout,        // [nv, nc, C, vhd]
    int nc,
    int khd,
    int vhd,
    int group
)
{
    const int c = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t hb = (size_t) head * nc + c;
    const size_t kb = (size_t)(head / group) * nc + c;

    extern __shared__ char wy_smem[];
    half* sh_k = (half*) wy_smem;                            // [C][khd]
    float* sh_m = (float*) (sh_k + (size_t) GC_C * 128);     // [C][C]
    __shared__ float sh_al[GC_C], sh_b[GC_C];

    for (int i = tid; i < GC_C * khd; i += GC_WYT) sh_k[i] = kin[kb * GC_C * khd + i];
    if (tid < GC_C)
    {
        sh_al[tid] = a_log[hb * GC_C + tid];
        sh_b[tid] = beta[hb * GC_C + tid];
    }
    __syncthreads();

    // ---- M[t][j] = beta_t (k_t·k_j) exp(A_t - A_j),  j < t --------------------
    // D is formed here from the cumulative log decay rather than read back from a
    // [nv, nc, C, C] array; with j < t the exponent is <= 0, so it cannot overflow.
    for (int p = tid; p < GC_C * GC_C; p += GC_WYT)
    {
        const int t = p / GC_C, j = p % GC_C;
        float m = 0.f;
        if (j < t)
        {
            // half2 loads: one shared-memory transaction per two elements. This
            // dot product is the bulk of the kernel (C*C/2 pairs x khd each), so
            // the instruction count here is what it costs.
            const half2* kt = (const half2*) (sh_k + (size_t) t * khd);
            const half2* kj = (const half2*) (sh_k + (size_t) j * khd);
            float2 dot = make_float2(0.f, 0.f);
            #pragma unroll 8
            for (int d = 0; d < khd / 2; ++d)
            {
                const float2 a = __half22float2(kt[d]);
                const float2 b = __half22float2(kj[d]);
                dot.x = fmaf(a.x, b.x, dot.x);
                dot.y = fmaf(a.y, b.y, dot.y);
            }
            m = sh_b[t] * (dot.x + dot.y) * __expf(sh_al[t] - sh_al[j]);
        }
        sh_m[p] = m;
    }
    __syncthreads();

    // ---- forward substitution: X[t] = RHS[t] - sum_{j<t} M[t][j] X[j] ---------
    // One thread per right-hand-side column, and the substitution only ever
    // touches its OWN column: X[t][col] depends on X[j][col], never on another.
    // So there is no cross-thread dependency, no barrier, and no reason for X to
    // be in shared memory at all — it lives in registers (C floats per thread,
    // 64 here). Putting it in smem cost 64 KB, which pinned the kernel to one
    // CTA per SM; both loops have compile-time bounds so the array stays in
    // registers rather than spilling to local memory.
    //
    // Right-looking order: once x[j] is final, propagate it forward immediately.
    const int col = tid;
    const bool is_k = col < khd;
    float acc[GC_C];
    #pragma unroll
    for (int t = 0; t < GC_C; ++t)
        acc[t] = is_k
            ? __half2float(sh_k[(size_t) t * khd + col]) * sh_b[t] * __expf(sh_al[t])
            : vin[hb * GC_C * vhd + (size_t) t * vhd + (col - khd)] * sh_b[t];

    #pragma unroll
    for (int j = 0; j < GC_C; ++j)
    {
        const float xj = acc[j];
        if (is_k) wout[hb * GC_C * khd + (size_t) j * khd + col] = xj;
        else uvout[hb * GC_C * vhd + (size_t) j * vhd + (col - khd)] = xj;
        #pragma unroll
        for (int t = j + 1; t < GC_C; ++t)
            acc[t] = fmaf(-sh_m[(size_t) t * GC_C + j], xj, acc[t]);
    }
}

extern "C" void gdn_chunk_wy
(
    const at::Tensor& k, const at::Tensor& v, const at::Tensor& beta,
    const at::Tensor& a_log, at::Tensor& w, at::Tensor& uv
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int nv = v.size(0), nc = v.size(1), vhd = v.size(3);
    const int khd = k.size(3);
    TORCH_CHECK(khd + vhd == GC_WYT, "gdn_chunk_wy: khd + vhd must be ", GC_WYT);
    TORCH_CHECK(k.size(2) == GC_C, "chunk length must be ", GC_C);
    const int group = nv / (int) k.size(0);

    const size_t smem = (size_t) GC_C * 128 * sizeof(half)
                      + (size_t) GC_C * GC_C * sizeof(float);
    cudaFuncSetAttribute(gdn_chunk_wy_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    dim3 blocks(nc, nv);
    gdn_chunk_wy_kernel<<<blocks, GC_WYT, smem, stream>>>(
        (const half*) k.data_ptr(), (const float*) v.data_ptr(),
        (const float*) beta.data_ptr(), (const float*) a_log.data_ptr(),
        (float*) w.data_ptr(), (float*) uv.data_ptr(),
        nc, khd, vhd, group);
    cuda_check(cudaPeekAtLastError());
}
