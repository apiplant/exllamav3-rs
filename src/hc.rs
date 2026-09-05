//! Hyper-connections — port of `modules/hyperconnections.py`, the `GatedResidual`
//! and `ExpandStreams` halves (grade B: the fp32 reference path).
//!
//! Instead of one residual stream, the block stack carries `hc_mult` parallel
//! streams. Each sublayer site reads a single collapsed vector out of the stack
//! and writes its output back into every stream through a per-stream gate, so
//! the streams stay distinct across depth. `Qwen4ExpForConditionalGeneration`
//! uses the gated-residual (low-rank, elementwise) flavour in place of its
//! input/post layernorms; `DeepseekV4ForCausalLM` and `Glm5NextForConditionalGeneration`
//! use the full mHC mixer, which is not implemented here.
//!
//! Upstream has two compute paths: a fused `gr_mix` kernel pair for small row
//! counts (decode) and half-precision GEMMs for large ones (prefill). Both are
//! checked against `_mix_ref`, the fp32 torch reference — which is what this is.
//! Replacing it with the fused form is a pure performance change.

use crate::safetensors::SafeTensors;
use anyhow::Result;
use tch::{Device, Kind, Tensor};

/// Broadcast a `(b, s, hidden)` embedding into `hc_mult` parallel fp32 streams,
/// `(b, s, hc_mult, hidden)`. Stateless — upstream carries it as a module only
/// so it has a place in the graph.
pub fn expand_streams(x: &Tensor, hc_mult: i64) -> Tensor {
    let sz = x.size();
    let (b, s, d) = (sz[0], sz[1], sz[2]);
    x.to_kind(Kind::Float)
        .unsqueeze(2)
        .expand([b, s, hc_mult, d], false)
        .contiguous()
}

/// One gated-residual site (`use_combine = true`) or the final stream collapse
/// (`use_combine = false`, HF `hyper_connection_mixer`).
pub struct GatedResidual {
    /// `(hc_mult, hidden)` fp32, already `+ 1.0` — the checkpoint stores the
    /// norm weight zero-initialized and it is applied as `1 + w`.
    norm_w: Tensor,
    /// `(rank, hc_mult * hidden)` fp32.
    down: Tensor,
    /// `(hc_mult * hidden, rank)` fp32, checkpoint orientation.
    up: Tensor,
    /// `(hc_mult, hc_mult * hidden)` fp32. `None` for the final mixer, which has
    /// nothing to inject back.
    inject: Option<Tensor>,
    hc_mult: i64,
    hidden_size: i64,
    eps: f64,
}

impl GatedResidual {
    pub fn load(
        stc: &SafeTensors,
        key: &str,
        hc_mult: i64,
        hidden_size: i64,
        eps: f32,
        use_combine: bool,
        device: Device,
    ) -> Result<Self> {
        let f = |k: &str| -> Result<Tensor> {
            Ok(stc.get(&format!("{key}.{k}"), device, false, false)?.to_kind(Kind::Float))
        };
        Ok(Self {
            norm_w: (f("hc_norm.weight")? + 1.0).view([hc_mult, hidden_size]).contiguous(),
            down: f("input_mix_weight_down.weight")?,
            up: f("input_mix_weight_up.weight")?,
            inject: match use_combine {
                true => Some(f("block_inject_weight.weight")?),
                false => None,
            },
            hc_mult,
            hidden_size,
            eps: eps as f64,
        })
    }

    /// `streams` `(b, s, H, D)` fp32 → `(post, mixed)`.
    ///
    /// `mixed` `(b, s, D)` is what the sublayer consumes; `post` `(b, s, H)` is
    /// the per-stream injection gate for `apply_`, `None` on the final mixer.
    ///
    /// The `/ hc_mult` divisions before both nonlinearities are not a
    /// normalization of the *streams* — they compensate for the flattened
    /// `H * D` input to the low-rank projections, and dropping them silently
    /// saturates both the silu and the sigmoid.
    pub fn mix(&self, streams: &Tensor) -> (Option<Tensor>, Tensor) {
        let (h, d) = (self.hc_mult, self.hidden_size);
        let x = streams.to_kind(Kind::Float);

        // Per-stream RMSNorm over the hidden axis, weighted by `1 + hc_norm`.
        let ms = x.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float);
        let normed = &x * (ms + self.eps).rsqrt() * &self.norm_w;
        let flat = normed.flatten(-2, -1);

        // Low-rank elementwise gate: which channels of which stream feed the
        // mean that the sublayer sees.
        let t = (flat.matmul(&self.down.tr()) / h as f64).silu();
        let g = t.matmul(&self.up.tr()).sigmoid();
        let mixed = (g.unflatten(-1, [h, d]) * &normed).mean_dim(-2, false, Kind::Float);

        // Per-stream scalar gate in [0, 2] for writing the sublayer output back.
        let post = self
            .inject
            .as_ref()
            .map(|inj| (flat.matmul(&inj.tr()) / h as f64).sigmoid() * 2.0);

        (post, mixed)
    }

    /// Final-mixer form: collapse the stack to `(b, s, D)`.
    pub fn forward(&self, streams: &Tensor) -> Tensor {
        debug_assert!(self.inject.is_none(), "site-form GatedResidual is used via mix/apply");
        self.mix(streams).1
    }

    /// Residual update for one sublayer site, in place:
    /// `streams[..., h, :] += post[..., h] * y`.
    pub fn apply_(&self, streams: &mut Tensor, y: &Tensor, post: &Tensor) {
        let _ = streams.f_add_(&(post.unsqueeze(-1) * y.to_kind(Kind::Float).unsqueeze(-2)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    fn dummy(h: i64, d: i64, rank: i64, use_combine: bool) -> GatedResidual {
        let dev = Device::Cpu;
        GatedResidual {
            norm_w: Tensor::randn([h, d], (Kind::Float, dev)) * 0.1 + 1.0,
            down: Tensor::randn([rank, h * d], (Kind::Float, dev)) * 0.05,
            up: Tensor::randn([h * d, rank], (Kind::Float, dev)) * 0.05,
            inject: use_combine.then(|| Tensor::randn([h, h * d], (Kind::Float, dev)) * 0.05),
            hc_mult: h,
            hidden_size: d,
            eps: 1e-6,
        }
    }

    /// `expand_streams` must broadcast, not split: every stream starts as a copy
    /// of the embedding. If it ever sliced the hidden dim instead, the first
    /// block would still run and produce plausible-looking garbage.
    #[test]
    fn expand_streams_replicates() {
        let x = Tensor::randn([1, 5, 8], (Kind::Half, Device::Cpu));
        let s = expand_streams(&x, 4);
        assert_eq!(s.size(), vec![1, 5, 4, 8]);
        for h in 0..4 {
            let diff = f64::try_from(
                (s.select(2, h) - x.to_kind(Kind::Float)).abs().max(),
            )
            .unwrap();
            assert!(diff == 0.0, "stream {h} is not a copy of the embedding");
        }
    }

    /// The injection gate is `2 * sigmoid(...)`, so it lives in (0, 2) — a
    /// stream can be written to at up to double weight or held nearly frozen.
    /// A plain sigmoid here would halve every residual write.
    #[test]
    fn post_gate_spans_zero_to_two() {
        let gr = dummy(4, 32, 8, true);
        // Large inputs so the sigmoid actually saturates both ways.
        let streams = Tensor::randn([1, 64, 4, 32], (Kind::Float, Device::Cpu)) * 50.0;
        let (post, _) = gr.mix(&streams);
        let post = post.unwrap();
        assert_eq!(post.size(), vec![1, 64, 4]);
        let (lo, hi) = (
            f64::try_from(post.min()).unwrap(),
            f64::try_from(post.max()).unwrap(),
        );
        assert!(lo > 0.0 && hi < 2.0, "post gate out of (0, 2): {lo}..{hi}");
    }

    /// `apply_` writes the sublayer output into each stream scaled by that
    /// stream's own gate. Getting the broadcast axes backwards would scale by
    /// the wrong gate without changing any shape.
    #[test]
    fn apply_scales_each_stream_by_its_own_gate() {
        let dev = Device::Cpu;
        let (h, d) = (4i64, 16i64);
        let gr = dummy(h, d, 8, true);
        let mut streams = Tensor::zeros([1, 1, h, d], (Kind::Float, dev));
        let y = Tensor::ones([1, 1, d], (Kind::Float, dev));
        let post = Tensor::from_slice(&[0.25f32, 0.5, 1.0, 1.75]).view([1, 1, h]);

        gr.apply_(&mut streams, &y, &post);
        for (i, want) in [0.25f64, 0.5, 1.0, 1.75].iter().enumerate() {
            let got = f64::try_from(streams.select(2, i as i64).mean(Kind::Float)).unwrap();
            assert!((got - want).abs() < 1e-6, "stream {i}: {got} != {want}");
        }
    }

    /// The final mixer collapses `(b, s, H, D)` to `(b, s, D)` and has no
    /// injection gate — it is the last thing before the head.
    #[test]
    fn final_mixer_collapses_the_stack() {
        let gr = dummy(4, 32, 8, false);
        let streams = Tensor::randn([1, 7, 4, 32], (Kind::Float, Device::Cpu));
        let (post, mixed) = gr.mix(&streams);
        assert!(post.is_none());
        assert_eq!(mixed.size(), vec![1, 7, 32]);
        assert_eq!(gr.forward(&streams).size(), vec![1, 7, 32]);
    }
}
