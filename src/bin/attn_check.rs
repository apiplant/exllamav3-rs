//! Numerical check for the tensor-core attention kernel (`EXL3_MMA_ATTN`).
//!
//! Runs `bighead_attn_paged` / `bighead_attn_paged_q` twice over *identical*
//! inputs — once with the scalar query-tiled kernel (the verified reference) and
//! once with the mma.sync kernel — and reports the max / RMS difference per
//! shape, plus a timing ratio. Both accumulate in fp32 and round to fp16, so
//! agreement to ~1e-2 relative is expected; anything larger is a layout bug.
//!
//! Needs only a few MB of VRAM, but a CUDA context is ~0.5 GB — run it with the
//! server stopped:  ./run-attn-check.sh [fp16|q4|q8|all]

use exl3::ffi;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

const PAGE: i64 = 256;

#[derive(Clone, Copy)]
struct Case {
    q_len: i64,
    ctx: i64,
    head_dim: i64,
    n_q: i64,
    n_kv: i64,
    bsz: i64,
}

fn rand_h(shape: &[i64], dev: Device) -> Tensor {
    ((Tensor::rand(shape, (Kind::Float, dev)) - 0.5) * 2.0).to_kind(Kind::Half)
}

/// Inputs for one case, generated once and shared by both kernels.
struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kc: Tensor,
    vc: Tensor,
    qk: Tensor,
    sk: Tensor,
    qv: Tensor,
    sv: Tensor,
    bt: Tensor,
    sl: Tensor,
}

fn make(c: &Case, bits: i64, dev: Device) -> Inputs {
    let pages = (c.ctx + c.q_len + PAGE - 1) / PAGE;
    let np = pages * c.bsz;
    let groups = c.n_kv * c.head_dim / 32;
    Inputs {
        q: rand_h(&[c.bsz, c.q_len, c.n_q, c.head_dim], dev),
        k: rand_h(&[c.bsz, c.q_len, c.n_kv, c.head_dim], dev),
        v: rand_h(&[c.bsz, c.q_len, c.n_kv, c.head_dim], dev),
        kc: rand_h(&[np, PAGE, c.n_kv, c.head_dim], dev),
        vc: rand_h(&[np, PAGE, c.n_kv, c.head_dim], dev),
        // random codes, random positive scales: both kernels dequantize the same
        // store, so any disagreement is the attention math, not the quantizer
        qk: Tensor::randint(1 << 20, [np, PAGE, groups * bits.max(1)], (Kind::Int, dev)),
        qv: Tensor::randint(1 << 20, [np, PAGE, groups * bits.max(1)], (Kind::Int, dev)),
        sk: (Tensor::rand([np, PAGE, groups], (Kind::Float, dev)) * 0.5 + 0.25)
            .to_kind(Kind::Half),
        sv: (Tensor::rand([np, PAGE, groups], (Kind::Float, dev)) * 0.5 + 0.25)
            .to_kind(Kind::Half),
        bt: Tensor::arange(np, (Kind::Int, dev)).reshape([c.bsz, pages]),
        sl: Tensor::full([c.bsz], c.ctx, (Kind::Int, dev)),
    }
}

/// `quant_cache_paged` inside the quant entry point mutates the store with this
/// call's fresh rows, so each kernel must start from an identical copy.
fn run(c: &Case, i: &Inputs, mode: &str, mma: bool, dev: Device) -> (Tensor, f64) {
    ffi::set_mma_attn(mma);
    let o = Tensor::zeros([c.bsz, c.q_len, c.n_q, c.head_dim], (Kind::Half, dev));
    let (qk, sk, qv, sv) = (i.qk.copy(), i.sk.copy(), i.qv.copy(), i.sv.copy());
    let go = || {
        if mode == "fp16" {
            ffi::bighead_attn_paged(&i.q, &i.k, &i.v, &i.kc, &i.vc, &i.bt, &i.sl, &o, 0.0);
        } else {
            ffi::bighead_attn_paged_q(
                &i.q, &i.k, &i.v, &qk, &sk, &qv, &sv, &i.bt, &i.sl, &o, 0.0, 0.0,
            );
        }
    };
    go();
    tch::Cuda::synchronize(0);
    let t = Instant::now();
    const REP: usize = 5;
    for _ in 0..REP {
        go();
    }
    tch::Cuda::synchronize(0);
    (o.to_kind(Kind::Float), t.elapsed().as_secs_f64() * 1000.0 / REP as f64)
}

/// Time the gated-delta-rule recurrence in isolation at prefill shapes. Its
/// launch geometry is (bsz, num_v_heads, v_split) with a sequential loop over
/// the sequence inside the kernel, so this is the thing to point `ncu` at:
///   ncu --set none --metrics \
///     sm__throughput.avg.pct_of_peak_sustained_elapsed,\
///     launch__occupancy_limit_warps,sm__warps_active.avg.pct_of_peak_sustained_active \
///     ./target/release/attn_check gdn
fn bench_gdn(dev: Device) {
    // A timing-only harness will happily "optimize" into a configuration that
    // computes the wrong thing — v_split=2 measured 12% faster and produced
    // garbage. Every split is checked against v_split=1 before its time counts.
    // Qwen3.8-27B linear-attention layer: 16 k heads, 48 v heads, 128/128 dims
    let (nk, nv, khd, vhd) = (16i64, 48i64, 128i64, 128i64);
    let fdim = 2 * nk * khd + nv * vhd;
    for &s in &[512i64, 1024, 2048, 4096] {
        let qkv = Tensor::rand([1, s, fdim], (Kind::BFloat16, dev));
        let g = (Tensor::rand([1, s, nv], (Kind::Float, dev)) * -0.01) - 0.001;
        let beta = Tensor::rand([1, s, nv], (Kind::BFloat16, dev));
        let mut state = Tensor::zeros([1, 1, nv, khd, vhd], (Kind::Float, dev));
        let out = Tensor::empty([1, s, nv, vhd], (Kind::BFloat16, dev));
        let mut run_split = |v: &str, noreg: bool| -> (Tensor, f64) {
            std::env::set_var("EXL3_GDN_VSPLIT", v);
            std::env::set_var("EXL3_GDN_NOREG", if noreg { "1" } else { "0" });
            let o = Tensor::zeros([1, s, nv, vhd], (Kind::BFloat16, dev));
            let mut go = || {
                let _ = state.zero_();
                ffi::recurrent_gated_delta_rule(
                    &qkv, &g, &beta, &state, &o, nk, nv, khd, vhd, None, false,
                );
            };
            go();
            tch::Cuda::synchronize(0);
            let t = Instant::now();
            const REP: usize = 10;
            for _ in 0..REP {
                go();
            }
            tch::Cuda::synchronize(0);
            (o.to_kind(Kind::Float), t.elapsed().as_secs_f64() * 1000.0 / REP as f64)
        };
        // ground truth: the original global-memory kernel
        let (reference, ref_ms) = run_split("4", true);
        let mut ok = format!("global-mem {ref_ms:.3}ms |");
        for v in ["1", "2", "4", "8"] {
            let (o, ms) = run_split(v, false);
            let d = (&o - &reference).abs().max().double_value(&[]);
            let sc = reference.abs().mean(Kind::Float).double_value(&[]).max(1e-9);
            ok.push_str(&format!(" reg-v{v} {ms:.3}{}", if d / sc < 0.02 { "" } else { "=WRONG" }));
        }
        std::env::remove_var("EXL3_GDN_VSPLIT");
        std::env::remove_var("EXL3_GDN_NOREG");

        let go = || {
            ffi::recurrent_gated_delta_rule(
                &qkv, &g, &beta, &state, &out, nk, nv, khd, vhd, None, false,
            )
        };
        go();
        tch::Cuda::synchronize(0);
        let t = Instant::now();
        const REP: usize = 10;
        for _ in 0..REP {
            go();
        }
        tch::Cuda::synchronize(0);
        let ms = t.elapsed().as_secs_f64() * 1000.0 / REP as f64;
        // 2 * s * nv * khd * vhd MACs for the state update + the same for the read-out
        let gflop = 4.0 * s as f64 * nv as f64 * khd as f64 * vhd as f64 / 1e9;
        println!(
            "  gdn seq={s:<5} {ms:7.3} ms  ({:.2} TFLOP/s)  -> x48 layers = {:.0} ms/chunk \
\n      {ok}",
            gflop / (ms / 1000.0) / 1000.0,
            ms * 48.0
        );
    }
}

/// Time the prefill linear path at the 27B's MLP shapes: the raw cuBLAS GEMM,
/// and the `reconstruct` (trellis -> dense fp16) that precedes it. The MLPs are
/// ~48% of a prefill chunk, so this says whether there is any headroom left in
/// them or whether the GEMM is already at the card's fp32-accumulate roof
/// (~82.6 TFLOP/s on a 4090 — fp16 tensor cores are half rate with fp32 accum).
/// The depthwise causal conv1d in the GDN layer. It touches
/// `seq * fdim_qkv * 2 bytes` in and out with a 4-tap kernel, so it should be
/// pure bandwidth (~0.1 ms at seq 2048); the trunk profile says ~1.05 ms/layer.
/// Decode-shape trellis GEMV (`exl3_gemm`), the kernel every 4-bit linear uses
/// when rows <= 144. Decode is weight-bandwidth bound, so the figure that
/// matters is GB/s of trellis read against the ~1008 GB/s the card can do.
/// Includes the lm_head shape, which the MTP path invokes 5x per decode step
/// (4 draft + 1 verify).
/// Verify + time the WY-chunked delta rule against the sequential kernel that
/// is the ground truth for the recurrence.
/// Warm up once, then average `REPS` runs. Both sides of every comparison in
/// this file go through here: timing a candidate as an average while timing the
/// reference with a single cold call inflated the reference ~20% and produced a
/// "win" the model contradicted.
fn timed<F: FnMut()>(mut f: F) -> f64 {
    const REPS: usize = 5;
    f();
    tch::Cuda::synchronize(0);
    let t = Instant::now();
    for _ in 0..REPS {
        f();
    }
    tch::Cuda::synchronize(0);
    t.elapsed().as_secs_f64() * 1000.0 / REPS as f64
}

fn bench_gdn_chunk(dev: Device) {
    let (nk, nv, khd, vhd) = (16i64, 48i64, 128i64, 128i64);
    let fdim = 2 * nk * khd + nv * vhd;
    for &s in &[128i64, 512, 2048, 4096] {
        let qkv = ((Tensor::rand([1, s, fdim], (Kind::Float, dev)) - 0.5) * 2.0)
            .to_kind(Kind::BFloat16);
        // g_raw is a log decay: small negative, as `gdn_fused_op_2` produces
        // Decay range matters: with a mild g the cumulative decay A stays near 1
        // and any formulation works. EXL3_GDN_TEST_G sets the magnitude so the
        // numerically hard regime (A underflowing within a chunk) is testable.
        let gmag: f64 = std::env::var("EXL3_GDN_TEST_G")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.05);
        let g = (Tensor::rand([1, s, nv], (Kind::Float, dev)) * -gmag) - 0.001;
        let beta = Tensor::rand([1, s, nv], (Kind::Float, dev)).to_kind(Kind::BFloat16);

        // A NONZERO initial state is essential: with h0 = 0 any bug in the
        // carry-in path is invisible, and the model carries state across every
        // prefill chunk. (This is exactly what let a broken h0 load pass here
        // while the model emitted garbage.)
        let h0 = (Tensor::rand([nv, khd, vhd], (Kind::Float, dev)) - 0.5) * 0.1;
        let st_ref = h0.reshape([1, 1, nv, khd, vhd]).copy();
        let o_ref = Tensor::zeros([1, s, nv, vhd], (Kind::BFloat16, dev));
        // Warm up and average, exactly as the chunked path below — timing the
        // reference with a single cold call inflated it by ~20% and made the
        // chunked path look like a win when the model said otherwise.
        let seq_ms = timed(|| {
            let _ = st_ref.shallow_clone().copy_(&h0.reshape([1, 1, nv, khd, vhd]));
            ffi::recurrent_gated_delta_rule(
                &qkv, &g, &beta, &st_ref, &o_ref, nk, nv, khd, vhd, None, false,
            );
        });

        // chunked
        let mut st_ch = h0.copy();
        let o_ch = Tensor::zeros([1, s, nv, vhd], (Kind::BFloat16, dev));
        let ch_ms = timed(|| {
            let _ = st_ch.copy_(&h0);
            exl3::gdn_chunk::chunked_delta_rule(
                &qkv, &g, &beta, &st_ch, &o_ch, nk, nv, khd, vhd,
            )
        });

        let a = o_ref.to_kind(Kind::Float);
        let b = o_ch.to_kind(Kind::Float);
        // Both sides write bf16, so compare max error against the signal RANGE,
        // not its mean — bf16 carries ~3 decimal digits, and dividing a max error
        // by a mean magnitude reports several % for what is just rounding.
        let d = (&a - &b).abs().max().double_value(&[]);
        let sc = a.abs().max().double_value(&[]).max(1e-9);
        let rms = (&a - &b).square().mean(Kind::Float).double_value(&[]).sqrt()
            / a.square().mean(Kind::Float).double_value(&[]).sqrt().max(1e-9);
        let sd = (st_ref.select(0, 0).select(0, 0) - &st_ch).abs().max().double_value(&[]);
        let ss = st_ref.abs().max().double_value(&[]).max(1e-9);
        let ok = d / sc < 0.02 && rms < 0.01 && sd / ss < 0.02;
        println!(
            "  seq={s:<5} sequential {seq_ms:7.3} ms  chunked {ch_ms:7.3} ms ({:.2}x)  \
             out max {:.1e} rms {:.1e} state {:.1e}  {}",
            seq_ms / ch_ms.max(1e-9),
            d / sc,
            rms,
            sd / ss,
            if ok { "OK" } else { "**WRONG**" }
        );
    }
}

fn bench_exl3(dev: Device) {
    for &(k, n, tag) in &[
        (5120i64, 17408i64, "mlp gate/up "),
        (17408i64, 5120i64, "mlp down    "),
        (5120i64, 248320i64, "lm_head     "),
    ] {
        // EXL3 trellis: [k/16, n/16, 16*bits] int16, bits = 4
        let bits = 4i64;
        let trellis = Tensor::zeros([k / 16, n / 16, 16 * bits], (Kind::Int16, dev));
        let suh = Tensor::ones([k], (Kind::Half, dev));
        let svh = Tensor::ones([n], (Kind::Half, dev));
        let bytes = (k / 16) * (n / 16) * 16 * bits * 2;
        for &m in &[1i64, 5] {
            let a = Tensor::rand([m, k], (Kind::Half, dev));
            let c = Tensor::empty([m, n], (Kind::Half, dev));
            let ah = Tensor::empty([m, k], (Kind::Half, dev));
            let go = || ffi::exl3_gemm(&a, &trellis, &c, &suh, &ah, &svh, false, false);
            go();
            tch::Cuda::synchronize(0);
            let t = Instant::now();
            const REP: usize = 30;
            for _ in 0..REP {
                go();
            }
            tch::Cuda::synchronize(0);
            let ms = t.elapsed().as_secs_f64() * 1000.0 / REP as f64;
            println!(
                "  {tag} m={m} [{k}x{n}] {ms:7.3} ms  ({:.0} GB/s of ~1008, {:.0}% of roof)",
                bytes as f64 / (ms / 1000.0) / 1e9,
                bytes as f64 / (ms / 1000.0) / 1e9 / 1008.0 * 100.0
            );
        }
    }
}

fn bench_conv(dev: Device) {
    let (nk, nv, khd, vhd) = (16i64, 48i64, 128i64, 128i64);
    let fdim = 2 * nk * khd + nv * vhd;
    let k = 4i64;
    for &s in &[512i64, 2048, 4096] {
        let x = Tensor::rand([1, fdim, s], (Kind::BFloat16, dev));
        let mut state = Tensor::zeros([1, fdim, k + 8], (Kind::BFloat16, dev));
        let w = Tensor::rand([fdim, k], (Kind::BFloat16, dev));
        let out = Tensor::empty([1, s, fdim], (Kind::BFloat16, dev));
        // Correctness first: the tiled prefill path only engages at seq >= 64, so
        // compare it against the per-channel kernel run one short slice at a time.
        let _ = state.zero_();
        ffi::causal_conv1d_update(&x, &state, None, &w, None, &out, true, false);
        tch::Cuda::synchronize(0);
        let fast = out.to_kind(Kind::Float).copy();
        let fast_state = state.to_kind(Kind::Float).copy();
        // reference: feed the same input in 32-token slices (below the tiled
        // threshold), carrying conv_state forward exactly as generation does
        let _ = state.zero_();
        let refout = Tensor::zeros([1, s, fdim], (Kind::BFloat16, dev));
        let step = 32i64;
        for off in (0..s).step_by(step as usize) {
            let n = step.min(s - off);
            let xs = x.narrow(2, off, n).contiguous();
            let os = Tensor::empty([1, n, fdim], (Kind::BFloat16, dev));
            ffi::causal_conv1d_update(&xs, &state, None, &w, None, &os, true, false);
            let _ = refout.narrow(1, off, n).copy_(&os);
        }
        tch::Cuda::synchronize(0);
        let d = (&fast - &refout.to_kind(Kind::Float)).abs().max().double_value(&[]);
        let ds = (&fast_state - &state.to_kind(Kind::Float)).abs().max().double_value(&[]);
        let verdict = if d < 2e-2 && ds < 2e-2 { "ok" } else { "WRONG" };

        let go = || ffi::causal_conv1d_update(&x, &state, None, &w, None, &out, true, false);
        go();
        tch::Cuda::synchronize(0);
        let t = Instant::now();
        const REP: usize = 20;
        for _ in 0..REP {
            go();
        }
        tch::Cuda::synchronize(0);
        let ms = t.elapsed().as_secs_f64() * 1000.0 / REP as f64;
        let gb = 2.0 * (s * fdim * 2) as f64 / 1e9; // read x + write out
        println!(
            "  conv seq={s:<5} {ms:7.3} ms  ({:.0} GB/s of ~1008 peak)  -> x48 = {:.0} ms/chunk \
             | vs per-channel kernel: {verdict} (out {d:.1e} state {ds:.1e})",
            gb / (ms / 1000.0),
            ms * 48.0
        );
    }
}

fn bench_mlp(dev: Device) {
    let m = 2048i64;
    for &(k, n, tag) in &[
        (5120i64, 17408i64, "mlp gate/up  "),
        (17408i64, 5120i64, "mlp down     "),
        (5120i64, 10240i64, "gdn qkv_proj "),
        (5120i64, 6144i64, "gdn z_proj   "),
        (5120i64, 128i64, "gdn a/b_proj "),
    ] {
        let a = Tensor::rand([m, k], (Kind::Half, dev));
        let w = Tensor::rand([k, n], (Kind::Half, dev));
        let c = Tensor::empty([m, n], (Kind::Half, dev));
        let time = |f: &dyn Fn()| -> f64 {
            f();
            tch::Cuda::synchronize(0);
            let t = Instant::now();
            const REP: usize = 20;
            for _ in 0..REP {
                f();
            }
            tch::Cuda::synchronize(0);
            t.elapsed().as_secs_f64() * 1000.0 / REP as f64
        };
        let gemm_ms = time(&|| ffi::hgemm(&a, &w, &c));
        let tflops = 2.0 * m as f64 * k as f64 * n as f64 / (gemm_ms / 1000.0) / 1e12;
        println!(
            "  {tag} [{m}x{k}]x[{k}x{n}]  gemm {gemm_ms:7.3} ms  {tflops:6.1} TFLOP/s              ({:.0}% of 82.6 fp32-acc roof)",
            tflops / 82.6 * 100.0
        );
    }
}

fn main() {
    let dev = Device::Cuda(0);
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if which == "gdnchunk" {
        println!("=== WY-chunked delta rule vs sequential kernel ===");
        bench_gdn_chunk(dev);
        return;
    }
    if which == "exl3" {
        println!("=== EXL3 trellis GEMV (decode shapes) ===");
        bench_exl3(dev);
        return;
    }
    if which == "conv" {
        println!("=== GDN causal conv1d ===");
        bench_conv(dev);
        return;
    }
    if which == "mlp" {
        println!("=== prefill linear path (m=2048) ===");
        bench_mlp(dev);
        return;
    }
    if which == "gdn" {
        println!("=== gated delta rule (sequential recurrence) ===");
        bench_gdn(dev);
        return;
    }
    let modes: Vec<&str> = match which.as_str() {
        "fp16" => vec!["fp16"],
        "q4" => vec!["q4"],
        "q8" => vec!["q8"],
        _ => vec!["fp16", "q4", "q8"],
    };

    // Qwen3.5 hybrid full-attention shape is head_dim 256, 24q/4kv (G=6); the
    // rest cover the other instantiations the dispatcher can select, including
    // q_len below/at/above the 16-row tile and a non-page-aligned context.
    let cases = [
        Case { q_len: 2,    ctx: 300,  head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 5,    ctx: 1024, head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 16,   ctx: 3000, head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 33,   ctx: 777,  head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 512,  ctx: 2048, head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 2048, ctx: 4096, head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 7,    ctx: 900,  head_dim: 256, n_q: 24, n_kv: 4, bsz: 2 },
        Case { q_len: 64,   ctx: 1500, head_dim: 128, n_q: 16, n_kv: 2, bsz: 1 },
        Case { q_len: 9,    ctx: 600,  head_dim: 128, n_q: 8,  n_kv: 8, bsz: 1 },
        // the real speculative-verify shape at long context, and a prefill chunk
        // against a long prefix — these are what dominate the server's time
        Case { q_len: 5,    ctx: 50000, head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 1,    ctx: 50000, head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
        Case { q_len: 2048, ctx: 48000, head_dim: 256, n_q: 24, n_kv: 4, bsz: 1 },
    ];

    let mut bad = 0;
    for mode in modes {
        let bits = if mode == "q4" { 4 } else { 8 };
        println!("\n=== {mode} ===");
        for c in &cases {
            let inp = make(c, bits, dev);
            let (a, ta) = run(c, &inp, mode, false, dev);
            let (b, tb) = run(c, &inp, mode, true, dev);
            let diff = (&a - &b).abs();
            let max = diff.max().double_value(&[]);
            let rms = diff.square().mean(Kind::Float).double_value(&[]).sqrt();
            let scale = a.abs().mean(Kind::Float).double_value(&[]).max(1e-6);
            let rel = max / scale;
            let ok = rel < 0.05 && a.isnan().any().int64_value(&[]) == 0
                && b.isnan().any().int64_value(&[]) == 0;
            if !ok {
                bad += 1;
                // localize: which (query, head, dim) disagree
                let d = diff.reshape([c.bsz, c.q_len, c.n_q, c.head_dim]);
                let per_q = d.amax(&[2i64, 3], false).squeeze_dim(0);
                let per_h = d.amax(&[1i64, 3], false).squeeze_dim(0);
                let per_d = d.amax(&[1i64, 2], false).squeeze_dim(0);
                let top = |t: &Tensor, n: i64| -> Vec<i64> {
                    let k = n.min(t.size()[0]);
                    let (_, idx) = t.topk(k, 0, true, false);
                    (0..k).map(|i| idx.int64_value(&[i])).collect()
                };
                let nz = |t: &Tensor| t.gt(1e-4).sum(Kind::Float).double_value(&[]) as i64;
                println!(
                    "      bad queries {}/{} first {:?} | bad heads {}/{} {:?} | bad dims {}/{} {:?}",
                    nz(&per_q), c.q_len, top(&per_q, 6),
                    nz(&per_h), c.n_q, top(&per_h, 6),
                    nz(&per_d), c.head_dim, top(&per_d, 6),
                );
            }
            println!(
                "  q={:<5} ctx={:<5} d={} G={} bsz={} | max {:.2e} rms {:.2e} rel {:.2e} | \
                 scalar {:7.3}ms  mma {:7.3}ms  ({:.2}x) {}",
                c.q_len, c.ctx, c.head_dim, c.n_q / c.n_kv, c.bsz,
                max, rms, rel, ta, tb, ta / tb.max(1e-9),
                if ok { "OK" } else { "**FAIL**" }
            );
        }
    }
    if bad > 0 {
        eprintln!("\n{bad} case(s) FAILED");
        std::process::exit(1);
    }
    println!("\nall cases match");
}
