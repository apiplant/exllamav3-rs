//! Block-sparse (mixture-of-experts) MLP — port of `modules/block_sparse_mlp.py`
//! (grade C: same math, different execution strategy).
//!
//! Two execution paths, identical in arithmetic:
//!
//! - **Multi-GEMM** ([`Multi`]) for single-row decode: `exl3_mgemm` takes a
//!   pointer table of expert weights and the routing *as a device tensor*, so
//!   three launches cover every selected expert and nothing round-trips through
//!   the host. That is what makes a decode step capturable, which is worth far
//!   more than the launch count — see ARCHITECTURES.md for the measurement.
//! - **Per-expert loop** for everything else, and whenever the fast path declines
//!   (fp16 experts, mixed bit rates or codebooks, padded projections): one
//!   grouped GEMM per active expert over the rows routed to it, which needs the
//!   routing table on the host to bucket rows.
//!
//! The loop is also the reference the fast path is checked against
//! (`EXL3_NO_MGEMM_MOE=1` forces it). Upstream additionally has a fully fused
//! `exl3_moe` kernel for batched decode; it supports only the `mcg`/`mul1`
//! codebooks, which no checkpoint here uses, so it is deliberately not wired.

use crate::config::{Config, MoeParams, RouterKind};
use crate::ffi;
use crate::modules::{GatedMlp, Linear};
use crate::safetensors::SafeTensors;
use anyhow::Result;
use tch::{Device, Kind, Tensor};

/// One expert's gate/up/down projections.
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub struct BlockSparseMlp {
    /// Router projection, `[hidden, num_experts]`. Dense fp16 — it is tiny and
    /// upstream leaves it unquantized.
    gate_w: Tensor,
    experts: Vec<Expert>,
    num_experts_per_tok: i64,
    router: RouterKind,
    /// `Dots` only: per-expert selection bias, `[num_experts]` fp32. Added to
    /// the scores for ranking, never to the weights that are actually applied.
    e_score_bias: Option<Tensor>,
    /// `Dots` only: routed weights are multiplied by this after normalization.
    routed_scaling_factor: f64,
    /// Always-on expert whose output is added to the routed mixture.
    /// `None` on architectures without one (Qwen3-MoE).
    shared: Option<GatedMlp>,
    /// The fused-kernel fast path, when every expert qualifies for it.
    multi: Option<Multi>,
    /// `hidden -> 1` projection whose sigmoid weights the shared expert's output
    /// (Qwen2-MoE lineage, and qwen4_exp). `None` means add it unweighted, which
    /// is what GLM4-MoE does.
    shared_gate: Option<Linear>,
}

/// The multi-GEMM decode path: per-expert weight pointer tables plus the scratch
/// the kernel needs, prepared once at load.
///
/// The point is not only that three launches replace `3 * top_k` of them. The
/// per-expert loop has to know *which* rows went to each expert, and getting the
/// routing table to the host costs a device sync per layer — 48 of them per token
/// on a Qwen3-30B-A3B decode, and an outright error under CUDA graph capture.
/// `exl3_mgemm` reads the selection from the device instead.
struct Multi {
    gate: crate::ffi::MultiLinear,
    up: crate::ffi::MultiLinear,
    down: crate::ffi::MultiLinear,
    /// `[top_k, 1, hidden]` fp16 — one hadamard-transformed input slab per expert.
    a_had: Tensor,
    /// `[top_k, 1, interm]` fp16 gate/up outputs and the activated product.
    interm_g: Tensor,
    interm_u: Tensor,
    interm_a: Tensor,
    /// `[top_k, 1, hidden]` fp32; the kernel reduces the weighted experts into row 0.
    out_d: Tensor,
    /// Scratch for the down projection's own transformed slabs.
    d_had: Tensor,
}

impl Multi {
    /// Prepare the decode fast path, or decline it. EXL3 experts only, with
    /// uniform bit rate and codebook across the experts of each projection, and
    /// no padding between the config's dimensions and the loaded weights — the
    /// scratch is cut to those dimensions, so a padded projection would read
    /// past its row.
    fn build(experts: &[Expert], top_k: i64, hidden: i64, interm: i64, device: Device) -> Option<Self> {
        if experts.is_empty() || !device.is_cuda() {
            return None;
        }
        if experts[0].gate.in_features != hidden
            || experts[0].up.in_features != hidden
            || experts[0].up.out_features != interm
            || experts[0].gate.out_features != interm
            || experts[0].down.in_features != interm
            || experts[0].down.out_features != hidden
        {
            return None;
        }

        let table = |sel: fn(&Expert) -> &Linear| -> Option<crate::ffi::MultiLinear> {
            let parts: Vec<_> = experts
                .iter()
                .map(|e| sel(e).exl3_parts())
                .collect::<Option<_>>()?;
            let (k, mcg, mul1) = (parts[0].3, parts[0].4, parts[0].5);
            if !parts.iter().all(|p| p.3 == k && p.4 == mcg && p.5 == mul1) {
                return None;
            }
            // 0 = trellis, 1 = suh, 2 = svh — a closure returning a borrow out of
            // the tuple needs a higher-ranked lifetime for no benefit.
            let ptrs = |sel: usize| {
                let raw: Vec<i64> = parts
                    .iter()
                    .map(|p| match sel {
                        0 => p.0,
                        1 => p.1,
                        _ => p.2,
                    }
                    .data_ptr() as i64)
                    .collect();
                Tensor::from_slice(&raw).to_device(device)
            };
            Some(crate::ffi::MultiLinear {
                trellis: ptrs(0),
                suh: ptrs(1),
                svh: ptrs(2),
                k,
                mcg,
                mul1,
            })
        };

        let half = |w: i64| Tensor::zeros([top_k, 1, w], (Kind::Half, device));
        Some(Self {
            gate: table(|e| &e.gate)?,
            up: table(|e| &e.up)?,
            down: table(|e| &e.down)?,
            a_had: half(hidden),
            interm_g: half(interm),
            interm_u: half(interm),
            interm_a: half(interm),
            out_d: Tensor::zeros([top_k, 1, hidden], (Kind::Float, device)),
            d_had: half(interm),
        })
    }
}

impl BlockSparseMlp {
    pub fn load(stc: &SafeTensors, key: &str, cfg: &Config, device: Device) -> Result<Self> {
        let p: &MoeParams = cfg
            .moe
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("BlockSparseMlp::load on a non-MoE config"))?;
        let h = cfg.hidden_size;
        let i = p.moe_intermediate_size;

        // `[num_experts, hidden]` in the checkpoint; transposed once at load so
        // routing is a plain `y @ gate_w`.
        let gate_w = stc
            .get(&format!("{key}.gate.weight"), device, true, false)?
            .transpose(0, 1)
            .contiguous();

        let mut experts = Vec::with_capacity(p.num_experts as usize);
        for e in 0..p.num_experts {
            let k = format!("{key}.experts.{e}");
            experts.push(Expert {
                gate: Linear::load(stc, &format!("{k}.gate_proj"), None, h, i, device, false, 0.0)?,
                up: Linear::load(stc, &format!("{k}.up_proj"), None, h, i, device, false, 0.0)?,
                down: Linear::load(stc, &format!("{k}.down_proj"), None, i, h, device, true, 0.0)?,
            });
        }

        // Upstream's `_esb_h` recenters the bias when the checkpoint stores it
        // in anything but fp16 — the ranking is invariant to a constant shift,
        // and the shift keeps the fp16 cast in range. Replicated exactly so a
        // recentered and a non-recentered checkpoint rank identically.
        let e_score_bias = match p.router {
            RouterKind::Dots => stc
                .get_opt(&format!("{key}.gate.e_score_correction_bias"), device)
                .map(|b| {
                    let b = b.to_kind(Kind::Float);
                    let m = b.mean(Kind::Float);
                    b - m
                }),
            RouterKind::Std => None,
        };

        // The two families that have a shared expert spell it differently:
        // GLM4-MoE `shared_experts`, qwen4_exp `shared_expert`.
        let shared = match p.shared_expert_intermediate_size {
            0 => None,
            n => {
                let sk = ["shared_experts", "shared_expert"]
                    .into_iter()
                    .map(|sfx| format!("{key}.{sfx}"))
                    .find(|k| stc.has_group(k, &["down_proj.trellis"]) || stc.has_group(k, &["down_proj.weight"]))
                    .ok_or_else(|| {
                        anyhow::anyhow!("{key}: shared expert width is {n} but no shared_expert(s) tensors found")
                    })?;
                Some(GatedMlp::load_sized(stc, &sk, h, n, device)?)
            }
        };
        let shared_gate = match p.shared_gate {
            true => Some(Linear::load(
                stc,
                &format!("{key}.shared_expert_gate"),
                None,
                h,
                1,
                device,
                false,
                0.0,
            )?),
            false => None,
        };

        // `EXL3_NO_MGEMM_MOE=1` forces the reference per-expert loop, for A/B
        // against the multi-GEMM path.
        let multi = match std::env::var("EXL3_NO_MGEMM_MOE").is_ok() {
            true => None,
            false => Multi::build(&experts, p.num_experts_per_tok, h, i, device),
        };

        Ok(Self {
            gate_w,
            experts,
            num_experts_per_tok: p.num_experts_per_tok,
            router: p.router,
            e_score_bias,
            routed_scaling_factor: p.routed_scaling_factor,
            shared,
            shared_gate,
            multi,
        })
    }

    /// Top-k routing. Returns `(indices [rows, k] int64, weights [rows, k] f32)`.
    ///
    /// Top-k over the raw logits, then softmax over the selected k only. That is
    /// what `routing_std_kernel` computes, and it equals a full softmax followed
    /// by renormalizing over the top-k set (`norm_topk_prob = true`) — the same
    /// value, without materializing the full distribution.
    fn route(&self, y: &Tensor) -> (Tensor, Tensor) {
        let logits = y.matmul(&self.gate_w).to_kind(Kind::Float);
        match self.router {
            RouterKind::Std => {
                let (top_v, top_i) = logits.topk(self.num_experts_per_tok, -1, true, true);
                let w = top_v.softmax(-1, Kind::Float);
                (top_i, w)
            }
            // `routing_ds3_nogroup` with ACT = sigmoid: rank by
            // `sigmoid(logit) + bias`, but weight by the *unbiased* sigmoid,
            // normalized over the selected set and scaled. The bias steers
            // which experts are picked (load balancing) without touching how
            // much each contributes, so it must not leak into the weights.
            RouterKind::Dots => {
                let scores = logits.sigmoid();
                let ranked = match &self.e_score_bias {
                    Some(b) => &scores + b.unsqueeze(0),
                    None => scores.shallow_clone(),
                };
                let top_i = ranked.topk(self.num_experts_per_tok, -1, true, true).1;
                let sel = scores.gather(1, &top_i, false);
                let denom = sel.sum_dim_intlist(vec![1i64].as_slice(), true, Kind::Float) + 1e-20;
                (top_i, sel * (self.routed_scaling_factor / denom))
            }
        }
    }

    /// `x`: `[1, seq, hidden]`. Returns the same shape.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.size();
        let h = *shape.last().unwrap();
        let y = x.reshape([-1, h]);
        let rows = y.size()[0];

        let (top_i, top_w) = self.route(&y);
        let mut out = Tensor::zeros([rows, h], (Kind::Float, y.device()));

        // Multi-GEMM decode path: three launches for the whole expert loop, with
        // the routing read from the device rather than the host. Single-row only —
        // that is the shape where the per-layer sync dominates, and where the
        // kernel's reduction across `top_k` slots lands the mixture in one place.
        if let Some(m) = &self.multi {
            if rows == 1 {
                let idx = top_i.reshape([1, -1]).to_kind(Kind::Int64).contiguous();
                let w = top_w.reshape([1, -1]).to_kind(Kind::Half).contiguous();
                let a = y.to_kind(Kind::Half).reshape([1, 1, h]).contiguous();

                ffi::exl3_mgemm(&a, &m.gate, &m.interm_g, &m.a_had, Some(&idx), None, 1);
                ffi::exl3_mgemm(&a, &m.up, &m.interm_u, &m.a_had, Some(&idx), None, 1);
                // silu(gate) * up, in place into the activation slab.
                let _ = m.interm_a.shallow_clone().copy_(&(m.interm_g.silu() * &m.interm_u));
                // `weights` makes the kernel scale each expert and reduce them
                // into row 0, so the routed mixture comes out of the same call.
                let _ = m.out_d.shallow_clone().zero_();
                ffi::exl3_mgemm(
                    &m.interm_a,
                    &m.down,
                    &m.out_d,
                    &m.d_had,
                    Some(&idx),
                    Some(&w),
                    1,
                );
                let _ = out.f_add_(&m.out_d.narrow(0, 0, 1).reshape([1, h])).unwrap();
                return self.finish(&out, &y, shape, x.kind());
            }
        }

        // Which rows chose each expert. One host round-trip for the whole
        // routing table, rather than one per expert.
        let flat = Vec::<i64>::try_from(top_i.reshape([-1]).to_device(Device::Cpu))
            .expect("routing indices");
        let k = self.num_experts_per_tok as usize;
        let mut buckets: Vec<Vec<i64>> = vec![Vec::new(); self.experts.len()];
        for (slot, &e) in flat.iter().enumerate() {
            buckets[e as usize].push(slot as i64 / k as i64);
        }

        for (e, rows_for_e) in buckets.iter().enumerate() {
            if rows_for_e.is_empty() {
                continue;
            }
            let idx = Tensor::from_slice(rows_for_e).to_device(y.device());
            let xe = y.index_select(0, &idx);

            let ex = &self.experts[e];
            let g = ex.gate.forward(&xe);
            let u = ex.up.forward(&xe);
            let a = Tensor::empty_like(&g);
            ffi::silu_mul(&g, &u, &a, 0.0);
            let d = ex.down.forward(&a).to_kind(Kind::Float);

            // Each routed row carries this expert's own weight from its slot in
            // the top-k table; gather them in the same order as `rows_for_e`.
            let w = top_w
                .reshape([-1])
                .index_select(0, &expert_slots(&flat, e, k, y.device()));
            // in-place accumulate; a row routed to k experts lands here k times
            let _ = out.index_add_(0, &idx, &(d * w.unsqueeze(1)));
        }

        self.finish(&out, &y, shape, x.kind())
    }

    /// Whether this layer has the fused kernel available. A decode step is only
    /// free of a host round-trip — and so only capturable — if every sparse layer
    /// does.
    pub fn is_fused(&self) -> bool {
        self.multi.is_some()
    }

    /// Add the shared expert, if any, and restore the caller's shape and dtype.
    /// Shared by both routed paths — it is the same tail either way.
    fn finish(&self, out: &Tensor, y: &Tensor, shape: Vec<i64>, kind: Kind) -> Tensor {
        // The shared expert sees every row and runs in parallel with the routed
        // mixture, weighted by its own sigmoid gate where the architecture has one.
        if let Some(sh) = &self.shared {
            let mut z = sh.forward(y).to_kind(Kind::Float);
            if let Some(g) = &self.shared_gate {
                z *= g.forward(y).to_kind(Kind::Float).sigmoid();
            }
            let _ = out.shallow_clone().f_add_(&z).unwrap();
        }
        out.reshape(shape).to_kind(kind)
    }
}

/// Positions in the flattened `[rows * k]` routing table that selected expert `e`,
/// in row order — the index into `top_w` matching each entry of that expert's bucket.
fn expert_slots(flat: &[i64], e: usize, _k: usize, device: Device) -> Tensor {
    let slots: Vec<i64> = flat
        .iter()
        .enumerate()
        .filter(|(_, &x)| x as usize == e)
        .map(|(s, _)| s as i64)
        .collect();
    Tensor::from_slice(&slots).to_device(device)
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    /// The kernel (and `route`) take the top-k of the raw logits and softmax over
    /// just those. Upstream's config knob is described as "softmax over all
    /// experts, then renormalize the top-k so they sum to 1". Those are the same
    /// number; this pins that, because if they ever diverged every MoE token
    /// would be subtly misweighted with nothing to catch it.
    #[test]
    fn topk_softmax_equals_full_softmax_renormalized() {
        let device = Device::Cpu;
        let (rows, num_experts, k) = (16i64, 32i64, 4i64);
        let logits = Tensor::randn([rows, num_experts], (Kind::Float, device)) * 3.0;

        // what `route` does
        let (top_v, top_i) = logits.topk(k, -1, true, true);
        let ours = top_v.softmax(-1, Kind::Float);

        // full softmax, gather the same k, renormalize
        let full = logits.softmax(-1, Kind::Float);
        let gathered = full.gather(1, &top_i, false);
        let theirs = &gathered / gathered.sum_dim_intlist(vec![1i64].as_slice(), true, Kind::Float);

        let max_diff = f64::try_from((ours - theirs).abs().max()).unwrap();
        assert!(max_diff < 1e-5, "top-k softmax diverged from renormalized: {max_diff}");
    }

    /// The `dots` router ranks by `sigmoid(logit) + bias` but weights by the
    /// unbiased `sigmoid(logit)`. Conflating the two is the easy mistake, and it
    /// is invisible: the model still runs, just with load-balancing pressure
    /// leaking into the expert mixture. This pins the two apart.
    #[test]
    fn dots_bias_steers_selection_but_not_weights() {
        let device = Device::Cpu;
        let (rows, num_experts, k) = (8i64, 32i64, 4i64);
        let logits = Tensor::randn([rows, num_experts], (Kind::Float, device)) * 3.0;
        // A bias big enough on expert 0 to force it into every row's top-k.
        let bias = Tensor::zeros([num_experts], (Kind::Float, device));
        let _ = bias.narrow(0, 0, 1).fill_(10.0);

        let scores = logits.sigmoid();
        let ranked = &scores + bias.unsqueeze(0);
        let top_i = ranked.topk(k, -1, true, true).1;
        let sel = scores.gather(1, &top_i, false);

        // expert 0 is selected for every row...
        let first = Vec::<i64>::try_from(top_i.narrow(1, 0, 1).reshape([-1])).unwrap();
        assert!(first.iter().all(|&e| e == 0), "bias did not force selection: {first:?}");
        // ...but its weight is still just sigmoid(logit), well under 1.
        let w0 = f64::try_from(sel.narrow(1, 0, 1).max()).unwrap();
        assert!(w0 < 1.0, "bias leaked into the routing weight: {w0}");
    }

    /// `dots` weights normalize over the selected set and then scale, so they sum
    /// to exactly `routed_scaling_factor` — not to 1 like the `std` router.
    #[test]
    fn dots_weights_sum_to_the_scaling_factor() {
        let device = Device::Cpu;
        let (rows, num_experts, k) = (8i64, 64i64, 8i64);
        let rsf = 2.5f64;
        let scores = (Tensor::randn([rows, num_experts], (Kind::Float, device)) * 3.0).sigmoid();
        let top_i = scores.topk(k, -1, true, true).1;
        let sel = scores.gather(1, &top_i, false);
        let denom = sel.sum_dim_intlist(vec![1i64].as_slice(), true, Kind::Float) + 1e-20;
        let w = sel * (rsf / denom);

        let sums = w.sum_dim_intlist(vec![1i64].as_slice(), false, Kind::Float);
        let err = f64::try_from((sums - rsf).abs().max()).unwrap();
        assert!(err < 1e-5, "dots weights do not sum to routed_scaling_factor: {err}");
    }

    /// Weights must sum to 1 per row, or the residual stream silently changes scale.
    #[test]
    fn routing_weights_are_normalized() {
        let device = Device::Cpu;
        let logits = Tensor::randn([8, 64], (Kind::Float, device)) * 5.0;
        let (top_v, _) = logits.topk(8, -1, true, true);
        let w = top_v.softmax(-1, Kind::Float);
        let sums = w.sum_dim_intlist(vec![1i64].as_slice(), false, Kind::Float);
        let err = f64::try_from((sums - 1.0).abs().max()).unwrap();
        assert!(err < 1e-6, "routing weights do not sum to 1: {err}");
    }
}
