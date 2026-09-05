// ---------------------------------------------------------------------------
// Tensor-core (mma.sync) paged flash attention.
//
// The scalar `attn_prefill_paged_kernel` in attention.cu runs the QK and PV
// products as per-lane FMAs plus a 5-step warp shuffle reduction: ~1.2 FLOP per
// issued instruction, measured at 10-15 TFLOP/s on an RTX 4090 where the tensor
// cores can do ~165. This kernel keeps the same paged layout, online softmax and
// split-K workspace protocol, but does both products with
// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
//
// Decomposition, per CTA:
//   * one CTA covers 16 query positions for ONE kv head, all G query heads that
//     share it, and one split of the KV range;
//   * PAIR warps cooperate on each query head, each owning DW = D/PAIR of the
//     head dim, so the O accumulator is always 16*DW/32 = 64 registers per
//     thread whatever the head dim. With D=256 that is PAIR=2.
//   * the CTA therefore runs G*PAIR warps (<= 16, i.e. <= 512 threads).
//   * K and V for a 16-position tile live in shared memory and are consumed by
//     every warp, so the KV range is read once per CTA rather than once per
//     query as in the scalar kernel.
//
// Fragment layouts (standard PTX m16n8k16, `t` = lane):
//   A (16x16 row-major, 4 regs / 8 halfs):
//     a0 = (row t/4,   k (t%4)*2+0), (row t/4,   k (t%4)*2+1)
//     a1 = (row t/4+8, k (t%4)*2+0), (row t/4+8, k (t%4)*2+1)
//     a2 = (row t/4,   k (t%4)*2+8), (row t/4,   k (t%4)*2+9)
//     a3 = (row t/4+8, k (t%4)*2+8), (row t/4+8, k (t%4)*2+9)
//   B (16x8, "col" = B[n][k], 2 regs / 4 halfs):
//     b0 = (k (t%4)*2+0, n t/4), (k (t%4)*2+1, n t/4)
//     b1 = (k (t%4)*2+8, n t/4), (k (t%4)*2+9, n t/4)
//   C/D (16x8 f32, 4 regs):
//     d0 = (row t/4,   col (t%4)*2+0)   d1 = (row t/4,   col (t%4)*2+1)
//     d2 = (row t/4+8, col (t%4)*2+0)   d3 = (row t/4+8, col (t%4)*2+1)
//
// Two consequences the code relies on:
//   * S = Q @ K^T wants B[n=kv][k=dim], which is exactly the row-major
//     k_tile[kv][dim] already in shared memory — no transpose needed for QK.
//   * the two n8 halves of the 16-wide S accumulator hold precisely the four
//     A-fragment registers needed by the following PV product, so P never round
//     trips through shared memory.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#define MMA_BN 16   // KV positions staged per tile (also the PV k-dim)
#define MMA_SPAD 8  // shared-memory row padding, in halfs

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ uint32_t pack2(half lo, half hi)
{
    const half2 h = __halves2half2(lo, hi);
    uint32_t r;
    memcpy(&r, &h, sizeof(r));
    return r;
}

// Rows of a 16x16 accumulator are split across the 4 lanes sharing `t/4`; reduce
// over them with two butterfly steps.
__device__ __forceinline__ float row_reduce_max(float v)
{
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}
__device__ __forceinline__ float row_reduce_sum(float v)
{
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}
