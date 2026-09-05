//! Chunked (WY-representation) gated delta rule — the matmul form of the
//! recurrence that `cuda_recurrent_gated_delta_rule_kernel_128` walks one token
//! at a time.
//!
//! # Why
//!
//! The sequential kernel is right for decode but wrong for prefill: it runs
//! `seqlen` dependent steps with only `num_v_heads * v_split` CTAs, measured at
//! 1.3–2.3 TFLOP/s on ~150 GFLOP. Upstream takes the chunked path (`fla`'s
//! `chunk_gated_delta_rule`) whenever `seqlen >= num_v_heads and not history`.
//! This does the same: `seqlen/C` dependent steps of dense GEMMs instead of
//! `seqlen` dependent steps of rank-1 updates.
//!
//! # Derivation
//!
//! The kernel implements, per v-head, with state `S` (`dk x dv`), scalars
//! `g_t = exp(g_raw_t)` and `beta_t`, and L2-normalized `q_t`, `k_t`:
//!
//! ```text
//!   S_t = g_t * S_{t-1} + beta_t * k_t (v_t - g_t * S_{t-1}^T k_t)^T
//!       = g_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
//!   o_t = scale * S_t^T q_t                      (note: S AFTER the update)
//! ```
//!
//! Let `A_t = prod_{i<=t} g_i` be the cumulative decay from the chunk start and
//! `Ŝ_t = S_t / A_t`. Since `A_t = g_t A_{t-1}`, the gate cancels:
//!
//! ```text
//!   Ŝ_t = (I - beta_t k_t k_t^T) Ŝ_{t-1} + (beta_t / A_t) v_t k_t^T
//!       = Ŝ_{t-1} + k_t u_t^T,   u_t = b̂_t v_t - beta_t Ŝ_{t-1}^T k_t
//! ```
//!
//! so `Ŝ_t = Ŝ_in + Σ_{j<=t} k_j u_j^T` — the WY form. Substituting gives a
//! triangular system for `U = [u_0 … u_{C-1}]`:
//!
//! ```text
//!   (I + M) U = diag(b̂) V - diag(beta) K Ŝ_in,   M = tril(diag(beta) K K^T, -1)
//! ```
//!
//! `Ŝ_in` is only known sequentially, so the two right-hand sides are solved
//! separately and combined per chunk:
//!
//! ```text
//!   W  = (I+M)^{-1} diag(beta) K          U_c = Uv_c - W_c S
//!   Uv = (I+M)^{-1} diag(b̂) V             O_c = scale * diag(A_c) (Q_c S + tril(Q_c K_c^T) U_c)
//!                                          S   = A_c[last] * (S + K_c^T U_c)
//! ```
//!
//! Everything but the last three lines is independent of the state and batches
//! over (head, chunk); only those run in chunk order.
//!
//! # Numerics — why nothing divides by the cumulative decay
//!
//! The naive form of the above carries `b̂_t = beta_t / A_t`, and `A_t` is a
//! product of `t` gates. That is fine for a mild decay and catastrophic for a
//! real one: at `|g| ~ 0.5` per step, `A` underflows inside a single 64-token
//! chunk and `b̂` overflows to inf, which reaches the model as NaN. (It passed a
//! harness built with `g in [-0.05, 0]` and destroyed the model's output.)
//!
//! So the implementation rescales the unknowns by `A`: it solves for
//! `Ũ_t = A_t u_t` instead of `u_t`, and every decay that survives appears only
//! as a RATIO `exp(A_log_t - A_log_j)` with `j <= t`, which is bounded by 1:
//!
//! ```text
//!   D[t][j] = exp(A_log_t - A_log_j)          (j <= t, so D <= 1)
//!   (I + M̃) Ũ = diag(beta) V - diag(beta) diag(A) K S_in,
//!        M̃[t][j] = beta_t (k_t·k_j) D[t][j]           (j < t)
//!   O   = scale * ( diag(A) Q S_in + (tril(Q K^T) ⊙ D) Ũ )
//!   S   = A_last S_in + K^T diag(D[last]) Ũ
//! ```
//!
//! `A` itself may still underflow to 0, which is correct and harmless — it just
//! means the incoming state has decayed away.

use std::sync::OnceLock;
use tch::{Kind, Tensor};

/// `EXL3_GDN_SCAN_REF=1`: run the chunk scan as tensor ops instead of the fused
/// kernel — the reference the kernel is validated against.
static SCAN_REF: OnceLock<bool> = OnceLock::new();
/// `EXL3_GDN_WY_REF=1`: likewise for the WY stage (M + solve + W/Uv).
static WY_REF: OnceLock<bool> = OnceLock::new();

/// Chunk length. 64 matches upstream; it bounds how far `A` can decay inside a
/// chunk (and therefore how large `b̂` gets) as well as the C×C work per token.
pub const CHUNK: i64 = 64;

/// Minimum sequence length worth taking this path — below it the sequential
/// kernel wins on launch overhead alone.
pub const MIN_SEQ: i64 = 2 * CHUNK;

/// L2-normalize over the last dim, matching the kernel's `rsqrt(sum + 1e-6)`.
fn l2norm(x: &Tensor) -> Tensor {
    let s = (x * x).sum_dim_intlist([-1i64].as_slice(), true, Kind::Float);
    x * (s + 1e-6).rsqrt()
}

/// Chunked gated delta rule for one batch row.
///
/// * `mixed_qkv` `[1, s, 2*nk*khd + nv*vhd]` bf16 — `q | k | v`, post-conv
/// * `g_raw` `[1, s, nv]` f32 (log decay), `beta` `[1, s, nv]` bf16
/// * `state` `[nv, khd, vhd]` f32, updated in place
/// * `core_out` `[1, s, nv, vhd]` bf16, written in place
#[allow(clippy::too_many_arguments)]
pub fn chunked_delta_rule(
    mixed_qkv: &Tensor,
    g_raw: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    core_out: &Tensor,
    nk: i64,
    nv: i64,
    khd: i64,
    vhd: i64,
) {
    // EXL3_GDN_CHUNK_PROF=1: stage timings with syncs, to see which phase costs
    let prof = std::env::var("EXL3_GDN_CHUNK_PROF").is_ok();
    let mark = |t: &mut std::time::Instant, name: &str| {
        if prof {
            tch::Cuda::synchronize(0);
            eprintln!("    [gdn_chunk] {name}: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);
            *t = std::time::Instant::now();
        }
    };
    let mut tm = std::time::Instant::now();

    let dev = mixed_qkv.device();
    let s = mixed_qkv.size()[1];
    let group = nv / nk;
    let scale = 1.0 / (khd as f64).sqrt();
    let nc = (s + CHUNK - 1) / CHUNK;
    let pad = nc * CHUNK - s;

    // ---- split q | k | v, normalize, expand k-heads to v-heads (GQA) --------
    let qk_dim = nk * khd;
    let q = mixed_qkv.narrow(2, 0, qk_dim).reshape([s, nk, khd]).to_kind(Kind::Float);
    let k = mixed_qkv.narrow(2, qk_dim, qk_dim).reshape([s, nk, khd]).to_kind(Kind::Float);
    let v = mixed_qkv.narrow(2, 2 * qk_dim, nv * vhd).reshape([s, nv, vhd]).to_kind(Kind::Float);
    // q and k are per K head; only V (and beta) are per V head. Expanding them to
    // nv here tripled every downstream tensor — the fused kernel indexes the K
    // head instead, and the few places that genuinely need per-V-head q/k
    // broadcast over a [nk, group, ...] view.
    let q = l2norm(&q); // [s, nk, khd]
    let k = l2norm(&k);

    // ---- [s, nv, d] -> [nv, NC, C, d], zero-padding the tail ----------------
    let to_chunks = |x: &Tensor, h: i64, d: i64| -> Tensor {
        let x = x.transpose(0, 1).contiguous(); // [h, s, d]
        let x = if pad > 0 { x.constant_pad_nd([0, 0, 0, pad]) } else { x };
        x.reshape([h, nc, CHUNK, d])
    };
    let q = to_chunks(&q, nk, khd); // [nk, NC, C, khd]
    let k = to_chunks(&k, nk, khd);
    let v = to_chunks(&v, nv, vhd);

    // gates: pad with 0 (decay 1) and beta with 0 (no update) so the tail is inert
    let gc = {
        let x = g_raw.reshape([s, nv]).to_kind(Kind::Float).transpose(0, 1).contiguous();
        let x = if pad > 0 { x.constant_pad_nd([0, pad]) } else { x };
        x.reshape([nv, nc, CHUNK])
    };
    let bc = {
        let x = beta.reshape([s, nv]).to_kind(Kind::Float).transpose(0, 1).contiguous();
        let x = if pad > 0 { x.constant_pad_nd([0, pad]) } else { x };
        x.reshape([nv, nc, CHUNK])
    };

    // ---- cumulative decay within each chunk (log space) --------------------
    let a_log = gc.cumsum(-1, Kind::Float); // [nv, NC, C]
    let a = a_log.exp(); // may underflow to 0; that is fine
    mark(&mut tm, "  decay (log only)");

    // ---- state-independent per-chunk factors ------------------------------
    // K K^T and Q K^T are per K HEAD, so they cost a third of what they did when
    // q/k were expanded to nv. Only the beta scaling (and V) is per V head.
    let bk3 = nk * nc;
    let bv3 = nv * nc;
    let qk3 = q.reshape([bk3, CHUNK, khd]);
    let kk3 = k.reshape([bk3, CHUNK, khd]);
    let vf = v.reshape([bv3, CHUNK, vhd]);
    let bf = bc.reshape([bv3, CHUNK, 1]);
    mark(&mut tm, "  split/norm/chunk");

    let qkt0 = qk3.matmul(&kk3.transpose(1, 2)); // [nk*NC, C, C], no decay, no mask
    mark(&mut tm, "  Q K^T (per K head)");

    // Stages 1+2 (K K^T, D, M, the triangular solve, W/Uv). `EXL3_GDN_WY_REF=1`
    // runs them as tensor ops — the reference the fused kernel is checked
    // against — but they cost 4.7 ms there against ~2 GFLOP of actual work,
    // because every step writes and re-reads a [nv, nc, C, C] or [nv, nc, C, khd]
    // array. The kernel keeps all of it on chip.
    let (w, uv);
    if *WY_REF.get_or_init(|| std::env::var("EXL3_GDN_WY_REF").is_ok()) || !dev.is_cuda() {
        // D is materialized only on the reference path; the kernels derive it
        let d5 = (a_log.unsqueeze(-1) - a_log.unsqueeze(-2))
            .exp()
            .tril(0)
            .reshape([nk, group, nc, CHUNK, CHUNK]);
        let kkt = kk3.matmul(&kk3.transpose(1, 2)); // [nk*NC, C, C]
        let kk5 = kkt.reshape([nk, 1, nc, CHUNK, CHUNK]);
        let m = ((kk5 * &d5).reshape([bv3, CHUNK, CHUNK]) * &bf).tril(-1);
        let eye = Tensor::eye(CHUNK, (Kind::Float, dev)).unsqueeze(0);
        let ipm = eye + m;
        let ba = &bf * a.reshape([bv3, CHUNK, 1]);
        let bkexp = (kk3.reshape([nk, 1, nc, CHUNK, khd])
            * ba.reshape([nk, group, nc, CHUNK, 1]))
        .reshape([bv3, CHUNK, khd]);
        let rhs = Tensor::cat(&[bkexp, &vf * &bf], 2);
        let sol = ipm.linalg_solve_triangular(&rhs, false, true, true);
        w = sol.narrow(2, 0, khd).contiguous();
        uv = sol.narrow(2, khd, vhd).contiguous();
    } else {
        let ww = Tensor::empty([nv, nc, CHUNK, khd], (Kind::Float, dev));
        let uu = Tensor::empty([nv, nc, CHUNK, vhd], (Kind::Float, dev));
        crate::ffi::gdn_chunk_wy(
            &k.to_kind(Kind::Half).contiguous(),
            &v.contiguous(),
            &bc.contiguous(),
            &a_log.contiguous(),
            &ww,
            &uu,
        );
        w = ww.reshape([bv3, CHUNK, khd]);
        uv = uu.reshape([bv3, CHUNK, vhd]);
    }
    mark(&mut tm, "  WY (M + solve + W/Uv)");

    let w4 = w.reshape([nv, nc, CHUNK, khd]);
    let q4 = q; // [nk, NC, C, khd]
    let uv4 = uv.reshape([nv, nc, CHUNK, vhd]);
    let k4 = k; // [nk, NC, C, khd]
    let qk4 = qkt0.reshape([nk, nc, CHUNK, CHUNK]); // [nk, NC, C, C], decay applied on chip
    let a4 = a.reshape([nv, nc, CHUNK, 1]);
    let d_last = (a_log.narrow(2, CHUNK - 1, 1) - &a_log).exp(); // [nv, NC, C]
    let dl4 = d_last.reshape([nv, nc, CHUNK, 1]);
    // A at each chunk's last position, precomputed so the loop has no slicing
    let a_last = a.narrow(2, CHUNK - 1, 1).unsqueeze(-1); // [nv, NC, 1, 1]

    // Scan + outputs. `EXL3_GDN_SCAN_REF=1` runs them as tensor ops (the
    // reference the fused kernel is validated against); by default one CUDA
    // kernel does both on the tensor cores, keeping the state in registers
    // across every chunk so neither the per-chunk state array nor a second pass
    // over W/Q/QK/U is ever materialized.
    let o;
    let use_ref = !dev.is_cuda()
        || *SCAN_REF.get_or_init(|| std::env::var("EXL3_GDN_SCAN_REF").is_ok());
    if use_ref {
        let mut st = state.shallow_clone(); // [nv, khd, vhd]
        let mut s_in: Vec<Tensor> = Vec::with_capacity(nc as usize);
        for c in 0..nc {
            s_in.push(st.shallow_clone()); // state ENTERING chunk c
            let kc = k4
                .select(1, c)
                .reshape([nk, 1, CHUNK, khd])
                .expand([nk, group, CHUNK, khd], true)
                .contiguous()
                .reshape([nv, CHUNK, khd]);
            let u = uv4.select(1, c) - w4.select(1, c).matmul(&st);
            // S = A_last S + K^T diag(D[last]) Ũ
            st = &st * a_last.select(1, c)
                + kc.transpose(1, 2).matmul(&(&u * dl4.select(1, c)));
        }
        let s_all = Tensor::stack(&s_in, 1); // [nv, NC, khd, vhd]
        let u_all = &uv4 - w4.matmul(&s_all);
        let qe = q4.reshape([nk, 1, nc, CHUNK, khd]);
        let sg = s_all.reshape([nk, group, nc, khd, vhd]);
        // O = scale (diag(A) Q S_in + (tril(QK^T) . D) Ũ); only the state term
        // carries the per-position decay now, the rest is inside qk4.
        let qs = qe.matmul(&sg).reshape([nv, nc, CHUNK, vhd]) * &a4;
        let dref = (a_log.unsqueeze(-1) - a_log.unsqueeze(-2)).exp().tril(0);
        // keep this 4-D: u_all is [nv, NC, C, vhd], so a [nv*NC, C, C] operand
        // would not broadcast against it
        let qkd = (qk4.reshape([nk, 1, nc, CHUNK, CHUNK])
            * dref.reshape([nk, group, nc, CHUNK, CHUNK]))
        .reshape([nv, nc, CHUNK, CHUNK]);
        let oo = (qs + qkd.matmul(&u_all)) * scale;
        o = oo.reshape([nv, nc * CHUNK, vhd]);
        let _ = state.shallow_clone().copy_(&st);
    } else {
        let h = Kind::Half;
        let oo = Tensor::empty([nv, nc, CHUNK, vhd], (Kind::Float, dev));
        let ht = Tensor::empty([nv, khd, vhd], (Kind::Float, dev));
        crate::ffi::gdn_chunk_fused(
            &w4.to_kind(h).contiguous(),
            &q4.to_kind(h).contiguous(),
            &k4.to_kind(h).contiguous(),
            &qk4.to_kind(h).contiguous(),
            &uv4.contiguous(),
            &a_log.contiguous(),
            &state.contiguous(),
            &oo,
            &ht,
            scale,
        );
        o = oo.reshape([nv, nc * CHUNK, vhd]);
        let _ = state.shallow_clone().copy_(&ht);
    }
    mark(&mut tm, "outputs");

    let o = o.narrow(1, 0, s).transpose(0, 1).contiguous(); // [s, nv, vhd]
    // cast to whatever the caller's buffer is (bf16 in the model, fp32 in tests)
    let _ = core_out
        .shallow_clone()
        .copy_(&o.reshape(core_out.size()).to_kind(core_out.kind()));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight transcription of the recurrence
    /// `cuda_recurrent_gated_delta_rule_kernel_128` implements, in scalar Rust.
    /// This is the ground truth the chunked form has to reproduce.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        qkv: &[f32],
        g_raw: &[f32],
        beta: &[f32],
        s: usize,
        nk: usize,
        nv: usize,
        khd: usize,
        vhd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let group = nv / nk;
        let scale = 1.0f32 / (khd as f32).sqrt();
        let fdim = 2 * nk * khd + nv * vhd;
        let mut st = vec![0.0f32; nv * khd * vhd]; // [h][d][j]
        let mut out = vec![0.0f32; s * nv * vhd];
        let l2 = |v: &[f32]| -> Vec<f32> {
            let n: f32 = v.iter().map(|x| x * x).sum();
            let r = (n + 1e-6).sqrt().recip();
            v.iter().map(|x| x * r).collect()
        };
        for t in 0..s {
            let row = &qkv[t * fdim..(t + 1) * fdim];
            for h in 0..nv {
                let kh = h / group;
                let q = l2(&row[kh * khd..(kh + 1) * khd]);
                let k = l2(&row[nk * khd + kh * khd..nk * khd + (kh + 1) * khd]);
                let v = &row[2 * nk * khd + h * vhd..2 * nk * khd + (h + 1) * vhd];
                let g_h = g_raw[t * nv + h].exp();
                let b_h = beta[t * nv + h];
                let sb = h * khd * vhd;
                // dot1 = S^T k
                let mut dot1 = vec![0.0f32; vhd];
                for (d, &kd) in k.iter().enumerate() {
                    for (j, dj) in dot1.iter_mut().enumerate() {
                        *dj += kd * st[sb + d * vhd + j];
                    }
                }
                for d in 0..khd {
                    for j in 0..vhd {
                        let v_eff = v[j] - dot1[j] * g_h;
                        st[sb + d * vhd + j] = st[sb + d * vhd + j] * g_h + k[d] * v_eff * b_h;
                    }
                }
                for (d, &qd) in q.iter().enumerate() {
                    for j in 0..vhd {
                        out[(t * nv + h) * vhd + j] += qd * st[sb + d * vhd + j];
                    }
                }
            }
        }
        for o in out.iter_mut() {
            *o *= scale;
        }
        (out, st)
    }

    /// Regression for the failure that reached the model as NaN: the naive WY
    /// form carries `b̂ = beta / A`, and `A` underflows inside one chunk once the
    /// gate is strong, so `b̂` overflows. A mild decay hides it completely — the
    /// original test used `g in [-0.05, 0]` and passed while the model emitted
    /// garbage. These magnitudes put `A` far below fp32 denormals within a chunk.
    #[test]
    fn survives_strong_decay() {
        for gmag in [0.5f64, 2.0, 8.0] {
            let (nk, nv, khd, vhd) = (2i64, 4i64, 16i64, 16i64);
            let s = 2 * CHUNK + 5;
            let fdim = 2 * nk * khd + nv * vhd;
            let dev = tch::Device::Cpu;
            let _ = tch::manual_seed(3);
            let qkv = (Tensor::rand([1, s, fdim], (Kind::Float, dev)) - 0.5) * 2.0;
            let g = (Tensor::rand([1, s, nv], (Kind::Float, dev)) * -gmag) - 0.001;
            let beta = Tensor::rand([1, s, nv], (Kind::Float, dev));

            let qv: Vec<f32> = Vec::<f32>::try_from(qkv.reshape([-1])).unwrap();
            let gv: Vec<f32> = Vec::<f32>::try_from(g.reshape([-1])).unwrap();
            let bv: Vec<f32> = Vec::<f32>::try_from(beta.reshape([-1])).unwrap();
            let (ref_out, _) = reference(
                &qv, &gv, &bv, s as usize, nk as usize, nv as usize, khd as usize, vhd as usize,
            );

            let st = Tensor::zeros([nv, khd, vhd], (Kind::Float, dev));
            let out = Tensor::zeros([1, s, nv, vhd], (Kind::Float, dev));
            chunked_delta_rule(&qkv, &g, &beta, &st, &out, nk, nv, khd, vhd);

            assert!(
                !bool::try_from(out.isnan().any()).unwrap()
                    && !bool::try_from(out.isinf().any()).unwrap(),
                "|g| <= {gmag}: output is not finite"
            );
            let got: Vec<f32> = Vec::<f32>::try_from(out.reshape([-1])).unwrap();
            let scale = ref_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
            let rel = ref_out
                .iter()
                .zip(&got)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
                / scale;
            assert!(rel < 1e-3, "|g| <= {gmag}: rel {rel}");
        }
    }

    /// The chunked form must reproduce the sequential recurrence, including the
    /// carried state, across a non-multiple-of-CHUNK length and with GQA.
    #[test]
    fn matches_sequential_recurrence() {
        let (nk, nv, khd, vhd) = (2i64, 4i64, 16i64, 16i64);
        let s = 3 * CHUNK + 17; // deliberately not a multiple of CHUNK
        let fdim = 2 * nk * khd + nv * vhd;
        let dev = tch::Device::Cpu;
        let _ = tch::manual_seed(7);

        let qkv = (Tensor::rand([1, s, fdim], (Kind::Float, dev)) - 0.5) * 2.0;
        // g_raw is a small negative log-decay, as `gdn_fused_op_2` produces
        let g = (Tensor::rand([1, s, nv], (Kind::Float, dev)) * -0.05) - 0.001;
        let beta = Tensor::rand([1, s, nv], (Kind::Float, dev));

        let qv: Vec<f32> = Vec::<f32>::try_from(qkv.reshape([-1])).unwrap();
        let gv: Vec<f32> = Vec::<f32>::try_from(g.reshape([-1])).unwrap();
        let bv: Vec<f32> = Vec::<f32>::try_from(beta.reshape([-1])).unwrap();
        let (ref_out, ref_st) = reference(
            &qv, &gv, &bv, s as usize, nk as usize, nv as usize, khd as usize, vhd as usize,
        );

        let st = Tensor::zeros([nv, khd, vhd], (Kind::Float, dev));
        let out = Tensor::zeros([1, s, nv, vhd], (Kind::Float, dev));
        chunked_delta_rule(&qkv, &g, &beta, &st, &out, nk, nv, khd, vhd);

        let got_out: Vec<f32> = Vec::<f32>::try_from(out.reshape([-1])).unwrap();
        let got_st: Vec<f32> = Vec::<f32>::try_from(st.reshape([-1])).unwrap();

        let rel = |a: &[f32], b: &[f32]| -> f32 {
            let scale = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / scale
        };
        let ro = rel(&ref_out, &got_out);
        let rs = rel(&ref_st, &got_st);
        assert!(ro < 1e-3, "output mismatch: rel {ro}");
        assert!(rs < 1e-3, "carried state mismatch: rel {rs}");
    }

    /// A chunk boundary must carry state exactly: running the whole sequence in
    /// one call has to equal running it in two calls that hand the state over.
    #[test]
    fn state_carries_across_calls() {
        let (nk, nv, khd, vhd) = (2i64, 4i64, 16i64, 16i64);
        let s = 2 * CHUNK;
        let fdim = 2 * nk * khd + nv * vhd;
        let dev = tch::Device::Cpu;
        let _ = tch::manual_seed(11);
        let qkv = (Tensor::rand([1, s, fdim], (Kind::Float, dev)) - 0.5) * 2.0;
        let g = (Tensor::rand([1, s, nv], (Kind::Float, dev)) * -0.05) - 0.001;
        let beta = Tensor::rand([1, s, nv], (Kind::Float, dev));

        let st1 = Tensor::zeros([nv, khd, vhd], (Kind::Float, dev));
        let o1 = Tensor::zeros([1, s, nv, vhd], (Kind::Float, dev));
        chunked_delta_rule(&qkv, &g, &beta, &st1, &o1, nk, nv, khd, vhd);

        // same input, two halves, state handed over
        let st2 = Tensor::zeros([nv, khd, vhd], (Kind::Float, dev));
        let mut halves = Vec::new();
        for h in 0..2 {
            let o = Tensor::zeros([1, CHUNK, nv, vhd], (Kind::Float, dev));
            chunked_delta_rule(
                &qkv.narrow(1, h * CHUNK, CHUNK).contiguous(),
                &g.narrow(1, h * CHUNK, CHUNK).contiguous(),
                &beta.narrow(1, h * CHUNK, CHUNK).contiguous(),
                &st2, &o, nk, nv, khd, vhd,
            );
            halves.push(o);
        }
        let o2 = Tensor::cat(&halves, 1);
        let d = (&o1 - &o2).abs().max().double_value(&[]);
        let sc = o1.abs().max().double_value(&[]).max(1e-6);
        assert!(d / sc < 1e-4, "split-call mismatch: rel {}", d / sc);
    }
}
