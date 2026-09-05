//! Qwen3.5 hybrid architecture — `architecture/qwen3_5.py` + `modules/gated_delta_net.py`
//! (grade C: text-only path, split projections, sequential recurrent kernel for
//! both prefill and decode, no fla/triton chunked rule, no MTP/vision).
//!
//! A Qwen3.5 decoder layer is either:
//!   * `full_attention`  — the ordinary gated GQA block (`modules::Attention` with
//!     `output_gate = true`, partial RoPE), paged KV cache; or
//!   * `linear_attention` — [`GatedDeltaNet`]: split `in_proj_{qkv,z,b,a}`, a
//!     depthwise causal conv1d + SiLU, the gated delta rule recurrence, a gated
//!     RMSNorm and `out_proj`. Its "cache" is the recurrent [`GdnState`]
//!     (rolling conv window + `[k_head_dim, v_head_dim]` recurrent state).

use crate::config::{Config, GdnParams};
use crate::ffi;
use crate::modules::Linear;
use crate::safetensors::SafeTensors;
use anyhow::Result;
use std::sync::OnceLock;
use tch::{Device, Kind, Tensor};

/// `EXL3_GDN_CHUNK=0` falls back to the sequential delta-rule kernel for long
/// prefill chunks (see `attn_check gdnchunk`).
static GDN_CHUNK: OnceLock<bool> = OnceLock::new();

/// Per-layer recurrent state for one `linear_attention` layer, single sequence.
///
/// `conv_state`: `[1, fdim_qkv, conv_kernel_size]` bf16 — the last `K` conv inputs.
/// `recurrent_state`: `[1, 1, num_v_heads, k_head_dim, v_head_dim]` f32 — the
/// running `S = Σ βₖ kₖ vₖᵀ · gate` matrix the delta rule carries forward.
pub struct GdnState {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
}

impl GdnState {
    pub fn new(p: &GdnParams, device: Device) -> Self {
        Self {
            conv_state: Tensor::zeros(
                [1, p.fdim_qkv(), p.conv_kernel_size],
                (Kind::BFloat16, device),
            ),
            recurrent_state: Tensor::zeros(
                [1, 1, p.num_v_heads, p.k_head_dim, p.v_head_dim],
                (Kind::Float, device),
            ),
        }
    }

    pub fn reset(&mut self) {
        let _ = self.conv_state.zero_();
        let _ = self.recurrent_state.zero_();
    }
}

/// `modules/gated_delta_net.py` `GatedDeltaNet`, split-projection torch path.
pub struct GatedDeltaNet {
    qkv_proj: Linear, // -> [.., 2*k_dim + v_dim]  (q | k | v)
    z_proj: Linear,   // -> [.., v_dim]            (output-gate pre-activation)
    b_proj: Linear,   // -> [.., num_v_heads]
    a_proj: Linear,   // -> [.., num_v_heads]
    o_proj: Linear,   // [v_dim] -> hidden
    conv1d_weight: Tensor, // [fdim_qkv, K] bf16
    conv1d_bias: Option<Tensor>,
    a_log: Tensor,   // [num_v_heads] bf16
    dt_bias: Tensor, // [num_v_heads] bf16
    norm_w: Tensor,  // [v_head_dim] f32 — gated RMSNorm weight
    p: GdnParams,
    norm_eps: f32,
    /// Nonlinearity on the output gate of the GDN norm. Qwen3.5 uses silu;
    /// qwen4_exp selects it with `output_gate_type`.
    gate_act: crate::ffi::GateAct,
}

impl GatedDeltaNet {
    pub fn load(stc: &SafeTensors, key: &str, cfg: &Config, device: Device) -> Result<Self> {
        let p = cfg.gdn.expect("GatedDeltaNet::load on a non-Qwen3.5 config");
        let h = cfg.hidden_size;
        let (k_dim, v_dim) = (p.k_dim(), p.v_dim());
        let bf16 = |t: Tensor| t.to_kind(Kind::BFloat16);

        let conv_w = stc
            .get(&format!("{key}.conv1d.weight"), device, false, true)?
            .squeeze_dim(1)
            .contiguous();
        Ok(Self {
            qkv_proj: Linear::load(stc, &format!("{key}.in_proj_qkv"), None, h, 2 * k_dim + v_dim, device, true, 0.0)?,
            z_proj: Linear::load(stc, &format!("{key}.in_proj_z"), None, h, v_dim, device, true, 0.0)?,
            b_proj: Linear::load(stc, &format!("{key}.in_proj_b"), None, h, p.num_v_heads, device, true, 0.0)?,
            a_proj: Linear::load(stc, &format!("{key}.in_proj_a"), None, h, p.num_v_heads, device, true, 0.0)?,
            o_proj: Linear::load(stc, &format!("{key}.out_proj"), None, v_dim, h, device, true, 0.0)?,
            conv1d_weight: bf16(conv_w),
            conv1d_bias: stc
                .get_opt(&format!("{key}.conv1d.bias"), device)
                .map(bf16),
            a_log: bf16(stc.get(&format!("{key}.A_log"), device, false, true)?),
            dt_bias: bf16(stc.get(&format!("{key}.dt_bias"), device, false, true)?),
            norm_w: stc
                .get(&format!("{key}.norm.weight"), device, false, true)?
                .to_kind(Kind::Float),
            p,
            norm_eps: cfg.rms_norm_eps,
            gate_act: cfg.gdn_gate_act,
        })
    }

    /// `x`: `[bsz, s, hidden]` (fp32 residual slice, post input-layernorm).
    /// Advances the recurrent state in place and returns `[bsz, s, hidden]` fp16.
    ///
    /// `conv_state` `[slots, fdim_qkv, K (+ max_history)]` bf16 and
    /// `recurrent_state` `[slots, 1 (+ max_history), Nv, k_hd, v_hd]` f32 are the
    /// (possibly shared) state pools; `slots` `[bsz]` int32 maps each batch row to
    /// its pool slot (`None` = identity, single sequence).
    ///
    /// `history`: also write per-token snapshots into the extra planes so a
    /// speculative forward can be rewound to its accepted prefix
    /// (`Qwen35PagedCache::gdn_rewind`). Requires the pools to have been built
    /// with `max_history >= s`.
    pub fn forward(
        &self,
        x: &Tensor,
        conv_state: &Tensor,
        recurrent_state: &Tensor,
        slots: Option<&Tensor>,
        history: bool,
    ) -> Tensor {
        let (b, s) = (x.size()[0], x.size()[1]);
        let p = &self.p;
        let dev = x.device();
        let (nk, nv) = (p.num_k_heads, p.num_v_heads);
        let (khd, vhd) = (p.k_head_dim, p.v_head_dim);
        let f = p.fdim_qkv();

        // `EXL3_TRUNK_PROF=1` sub-attribution for the GDN layer: the recurrence
        // is sequential over the sequence, so it is worth knowing how much of
        // the layer is actually the recurrence vs the projections around it.
        let gprof = *crate::model::trunk_prof_on();
        let gt = || {
            if gprof {
                tch::Cuda::synchronize(0);
            }
            std::time::Instant::now()
        };
        let g0 = gt();

        // --- projections ---------------------------------------------------
        let qkv = self.qkv_proj.forward(x); // [b,s,f] half
        let z = self.z_proj.forward(x).to_kind(Kind::Float); // [b,s,v_dim]
        let bb = self.b_proj.forward(x).to_kind(Kind::Float).contiguous(); // [b,s,nv]
        let aa = self.a_proj.forward(x).to_kind(Kind::Float).contiguous();

        // beta = sigmoid(b) * beta_scale (bf16);  g = -softplus(a + dt_bias) * exp(a_log) (f32)
        let beta = Tensor::empty([b, s, nv], (Kind::BFloat16, dev));
        let g = Tensor::empty([b, s, nv], (Kind::Float, dev));
        ffi::gdn_fused_op_2(&bb, &aa, &self.dt_bias, &self.a_log, &beta, &g, p.beta_scale);

        let g1 = gt();
        // --- causal conv1d + SiLU (rolling conv_state) --------------------
        // kernel wants x as [b, f, s] bf16; writes out as [b, s, f] bf16
        let mixed_in = qkv
            .transpose(1, 2)
            .to_kind(Kind::BFloat16)
            .contiguous(); // [b,f,s]
        let conv_out = Tensor::empty([b, s, f], (Kind::BFloat16, dev));
        ffi::causal_conv1d_update(
            &mixed_in,
            conv_state,
            slots,
            &self.conv1d_weight,
            self.conv1d_bias.as_ref(),
            &conv_out,
            true,    // SiLU activation
            history, // per-token conv-window snapshots for speculative rewind
        );

        let g2 = gt();
        // --- gated delta rule recurrence --------------------------------
        // mixed_qkv: [b, s, 2*k_dim + v_dim] bf16  (q | k | v), already that layout
        //
        // Long prefill chunks can take the WY-chunked matmul form (see
        // `gdn_chunk`): seqlen/64 dependent steps of dense GEMMs instead of
        // `seqlen` dependent rank-1 steps. Decode, speculative history and
        // multi-row batches always stay on the sequential kernel.
        //
        // On by default; `EXL3_GDN_CHUNK=0` restores the sequential kernel,
        // which remains the reference `attn_check gdnchunk` validates against.
        let core = Tensor::empty([b, s, nv, vhd], (Kind::BFloat16, dev));
        let use_chunk = !history
            && b == 1
            && s >= crate::gdn_chunk::MIN_SEQ
            && *GDN_CHUNK.get_or_init(|| {
                std::env::var("EXL3_GDN_CHUNK").map_or(true, |v| v != "0")
            });
        if use_chunk {
            let slot = slots.map_or(0, |t| t.int64_value(&[0]));
            // [slots, hist+1, nv, khd, vhd] -> this row's live state [nv, khd, vhd]
            let st = recurrent_state.select(0, slot).select(0, 0);
            crate::gdn_chunk::chunked_delta_rule(
                &conv_out, &g, &beta, &st, &core, nk, nv, khd, vhd,
            );
        } else {
            ffi::recurrent_gated_delta_rule(
                &conv_out, &g, &beta, recurrent_state, &core,
                nk, nv, khd, vhd, slots, history,
            );
        }

        let g3 = gt();
        // --- gated RMSNorm (gate = z), then out_proj --------------------
        let core = core.reshape([-1, vhd]); // bf16
        let zg = z.reshape([-1, vhd]).contiguous(); // f32 gate
        let y = Tensor::empty([b * s * nv, vhd], (Kind::Half, dev));
        ffi::gated_rms_norm(&core, &self.norm_w, &y, &zg, self.norm_eps, 0.0, 1, false, self.gate_act);
        let y = y.reshape([b, s, nv * vhd]);
        let out = self.o_proj.forward(&y);
        if gprof {
            use std::sync::atomic::Ordering::Relaxed;
            let g4 = gt();
            crate::model::TP_G_PROJ.fetch_add((g1 - g0).as_nanos() as u64, Relaxed);
            crate::model::TP_G_CONV.fetch_add((g2 - g1).as_nanos() as u64, Relaxed);
            crate::model::TP_G_RULE.fetch_add((g3 - g2).as_nanos() as u64, Relaxed);
            crate::model::TP_G_OUT.fetch_add((g4 - g3).as_nanos() as u64, Relaxed);
        }
        out
    }
}
