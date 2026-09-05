//! PLE (per-layer embedding) injection — port of `modules/ple.py`'s
//! `forward_streams_reference` (grade B: the op-by-op reference form).
//!
//! `Qwen4ExpForConditionalGeneration` puts one of these ahead of an early
//! decoder block. It feeds hashed n-gram features into *every* stream of the
//! hyper-connection stack (see `crate::hc`): the n-gram embedding projects to
//! one key per stream and one shared value, each stream's own normed activation
//! gates that value through a signed-sqrt dot product, and a depthwise dilated
//! causal conv over the gated values adds local lexical context. The caller adds
//! the result to the raw stack: `streams += ple(streams, ngram_embedding)`.
//!
//! **This is the arithmetic only.** Two things it deliberately does not do, both
//! required before the layer can run in the generator:
//!
//! - The n-gram embedding itself (`modules/ngram_embedding.py`) — token-history
//!   hashing, eos segmentation and dedup, against a table far too large to hold
//!   resident. `emb` is an input here.
//! - The recurrent state. Upstream carries a per-slot conv window *and* the
//!   trailing `ngram_size - 1` token ids, with the same alloc/rewind/stash
//!   contract as the GDN state, so incremental decoding hooks into the cache and
//!   the generator's rollback paths. `conv_state` is passed in and the updated
//!   column stream handed back, but nothing owns it yet.
//!
//! Upstream fuses the whole thing into one `ple_forward_streams` call; this is
//! the reference that fusion is checked against.

use crate::safetensors::SafeTensors;
use anyhow::Result;
use tch::{Device, Kind, Tensor};

/// Grouped RMSNorm over the stream stack: normalize each `(.., h, :)` row over
/// the hidden axis, scaled by `1 + w[h]`. The weight is stored zero-init and
/// applied with a constant bias of 1, exactly as in `crate::hc`.
fn grouped_rms(x: &Tensor, w: &Tensor, eps: f64) -> Tensor {
    let x = x.to_kind(Kind::Float);
    let ms = x.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float);
    &x * (ms + eps).rsqrt() * w
}

pub struct PleLayer {
    /// `(ple_embed_dim, hc_mult * hidden)` — one key per stream.
    key_proj: crate::modules::Linear,
    /// `(ple_embed_dim, hidden)` — one value shared across streams.
    value_proj: crate::modules::Linear,
    /// Each `(hc_mult, hidden)` fp32, already `+ 1.0`.
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// Depthwise conv kernel, `(hc_mult * hidden, 1, conv_kernel_size)` fp32.
    conv_w: Tensor,
    hc_mult: i64,
    hidden_size: i64,
    /// Conv dilation — this is `ngram_size`, not a separate knob: the conv strides
    /// over one position per n-gram order rather than over adjacent tokens.
    dilation: i64,
    /// `(conv_kernel_size - 1) * dilation` — how many trailing columns the next
    /// chunk needs as its conv state.
    conv_state_len: i64,
    /// `1 / sqrt(hidden)`, applied to the raw key/query dot before the gate.
    gate_scale: f64,
    eps: f64,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        stc: &SafeTensors,
        key: &str,
        hc_mult: i64,
        hidden_size: i64,
        ple_embed_dim: i64,
        ngram_size: i64,
        conv_kernel_size: i64,
        eps: f32,
        device: Device,
    ) -> Result<Self> {
        let hc_hidden = hc_mult * hidden_size;
        let norm = |name: &str| -> Result<Tensor> {
            Ok((stc.get(&format!("{key}.{name}.weight"), device, false, false)?
                .to_kind(Kind::Float)
                + 1.0)
                .view([hc_mult, hidden_size])
                .contiguous())
        };
        Ok(Self {
            key_proj: crate::modules::Linear::load(
                stc, &format!("{key}.key_proj"), None, ple_embed_dim, hc_hidden, device, true, 0.0,
            )?,
            value_proj: crate::modules::Linear::load(
                stc, &format!("{key}.value_proj"), None, ple_embed_dim, hidden_size, device, true, 0.0,
            )?,
            norm_key: norm("norm_key")?,
            norm_query: norm("norm_query")?,
            norm_conv: norm("norm_conv")?,
            conv_w: stc
                .get(&format!("{key}.conv1d.weight"), device, false, false)?
                .to_kind(Kind::Float)
                .view([hc_hidden, 1, conv_kernel_size])
                .contiguous(),
            hc_mult,
            hidden_size,
            dilation: ngram_size,
            conv_state_len: (conv_kernel_size - 1) * ngram_size,
            gate_scale: 1.0 / (hidden_size as f64).sqrt(),
            eps: eps as f64,
        })
    }

    /// `sigmoid(signed_sqrt(dot * gate_scale))`, broadcast over the hidden axis
    /// and applied to the shared value row.
    ///
    /// The square root is the point: the raw dot product of two `hidden`-wide
    /// normed vectors has a range that saturates a plain sigmoid almost
    /// everywhere, so the gate would be all-or-nothing. `sign(g) * sqrt(|g|)`
    /// compresses it back into the sigmoid's usable band while keeping the sign.
    fn gate(&self, dot: &Tensor, value: &Tensor) -> Tensor {
        let g = dot * self.gate_scale;
        // `max(|g|, 1e-6)` guards the derivative at zero; `sign` keeps sign(0) = 0.
        let ss = g.sign() * g.abs().clamp_min(1e-6).sqrt();
        ss.sigmoid().unsqueeze(-1) * value.to_kind(Kind::Float).unsqueeze(-2)
    }

    /// Depthwise **dilated causal** conv over the normed gated values.
    ///
    /// `x` `(b, seq, hc_mult * hidden)`; `conv_state` `(b, hc_mult * hidden,
    /// conv_state_len)` trailing columns from the previous chunk, or `None` at
    /// sequence start. Causality comes from prepending exactly
    /// `conv_state_len` columns and using no padding, so the output is `seq`
    /// wide and position `t` never sees past itself.
    ///
    /// Returns `(silu(conv(x)), the full column stream)` — the stream's trailing
    /// `conv_state_len` columns are the next chunk's state.
    fn short_conv(&self, x: &Tensor, conv_state: Option<&Tensor>) -> (Tensor, Tensor) {
        let (b, seq, ch) = {
            let s = x.size();
            (s[0], s[1], s[2])
        };
        let xt = x.transpose(1, 2);
        let state = match conv_state {
            Some(s) => s.to_kind(Kind::Float),
            None => Tensor::zeros([b, ch, self.conv_state_len], (Kind::Float, x.device())),
        };
        let stream = Tensor::cat(&[state, xt], -1);
        let y = stream.conv1d::<&Tensor>(&self.conv_w, None, 1, 0, self.dilation, ch);
        let _ = seq;
        (y.silu().transpose(1, 2), stream)
    }

    /// `streams` `(b, seq, hc_mult, hidden)` fp32, `emb` `(b, seq, ple_embed_dim)`
    /// from the n-gram embedding. Returns `(delta, conv column stream)`; `delta`
    /// is added to the raw stream stack by the caller.
    pub fn forward_streams(
        &self,
        streams: &Tensor,
        emb: &Tensor,
        conv_state: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let (h, d) = (self.hc_mult, self.hidden_size);
        let (b, seq) = {
            let s = streams.size();
            (s[0], s[1])
        };

        let key = grouped_rms(
            &self.key_proj.forward(emb).view([b, seq, h, d]),
            &self.norm_key,
            self.eps,
        );
        let value = self.value_proj.forward(emb);
        let query = grouped_rms(streams, &self.norm_query, self.eps);

        // Per-stream key/query dot: a batched (1, D) x (D, 1) matmul over the
        // flattened (b * seq * h) rows.
        let dot = query
            .reshape([-1, 1, d])
            .bmm(&key.reshape([-1, d, 1]))
            .view([b, seq, h]);

        let gated = self.gate(&dot, &value);
        let normed = grouped_rms(&gated, &self.norm_conv, self.eps).flatten(-2, -1);
        let (conv_out, stream) = self.short_conv(&normed, conv_state);
        (&gated + conv_out.view([b, seq, h, d]), stream)
    }

    /// How many trailing columns of the conv stream the next chunk needs.
    pub fn conv_state_len(&self) -> i64 {
        self.conv_state_len
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    /// The gate is `sigmoid(sign(g) * sqrt(|g|))`, not `sigmoid(g)`. Dropping the
    /// signed sqrt leaves a gate that saturates to 0 or 1 for essentially every
    /// real dot product, which still runs and still produces text.
    #[test]
    fn signed_sqrt_keeps_the_gate_off_the_rails() {
        let g = Tensor::from_slice(&[-25.0f32, -1.0, 0.0, 1.0, 25.0]);
        let ss = g.sign() * g.abs().clamp_min(1e-6).sqrt();
        let v = Vec::<f64>::try_from(ss.sigmoid()).unwrap();
        let plain = Vec::<f64>::try_from(g.sigmoid()).unwrap();

        // sign is preserved around zero, and 0 maps to exactly 0.5
        assert!((v[2] - 0.5).abs() < 1e-6, "sign(0) != 0: {}", v[2]);
        assert!(v[1] < 0.5 && v[3] > 0.5);
        // a dot of 25 pins a plain sigmoid at the rails; through the sqrt it is 5,
        // which still leaves the gate a usable gradient
        assert!(
            plain[4] > 1.0 - 1e-9 && plain[0] < 1e-9,
            "test premise: a plain sigmoid did not saturate at +/-25"
        );
        assert!(v[4] < 0.995, "gate saturated at the top: {}", v[4]);
        assert!(v[0] > 0.005, "gate saturated at the bottom: {}", v[0]);
    }

    /// The value row is shared across streams; only the scalar gate differs. If
    /// the broadcast axes were swapped the shapes would still line up whenever
    /// `hc_mult` happened to divide `hidden`.
    #[test]
    fn value_is_shared_across_streams() {
        let (b, s, h, d) = (1i64, 3i64, 4i64, 8i64);
        let dot = Tensor::randn([b, s, h], (Kind::Float, Device::Cpu));
        let value = Tensor::randn([b, s, d], (Kind::Float, Device::Cpu));

        let g = &dot * 0.5;
        let ss = g.sign() * g.abs().clamp_min(1e-6).sqrt();
        let out = ss.sigmoid().unsqueeze(-1) * value.unsqueeze(-2);
        assert_eq!(out.size(), vec![b, s, h, d]);

        // every stream is the same value row up to one scalar
        for i in 0..h {
            let ratio = out.select(2, i) / &value;
            let spread = f64::try_from(
                (ratio.max_dim(-1, false).0 - ratio.min_dim(-1, false).0).abs().max(),
            )
            .unwrap();
            assert!(spread < 1e-5, "stream {i} is not a scaled copy of value: {spread}");
        }
    }

    /// The conv is causal by construction: `conv_state_len` prepended columns and
    /// no padding. Perturbing a future position must not move an earlier output.
    #[test]
    fn dilated_conv_is_causal() {
        let (ch, k, dil) = (6i64, 4i64, 3i64);
        let state_len = (k - 1) * dil;
        let seq = 12i64;
        let w = Tensor::randn([ch, 1, k], (Kind::Float, Device::Cpu));

        let run = |x: &Tensor| -> Tensor {
            let state = Tensor::zeros([1, ch, state_len], (Kind::Float, Device::Cpu));
            let stream = Tensor::cat(&[state, x.shallow_clone()], -1);
            stream.conv1d::<&Tensor>(&w, None, 1, 0, dil, ch)
        };

        let x = Tensor::randn([1, ch, seq], (Kind::Float, Device::Cpu));
        let y0 = run(&x);
        assert_eq!(y0.size(), vec![1, ch, seq], "conv changed the sequence length");

        // clobber the last position only
        let x2 = x.copy();
        let _ = x2.narrow(2, seq - 1, 1).fill_(99.0);
        let y1 = run(&x2);

        let head_diff =
            f64::try_from((y0.narrow(2, 0, seq - 1) - y1.narrow(2, 0, seq - 1)).abs().max())
                .unwrap();
        assert!(head_diff == 0.0, "future token leaked into earlier outputs: {head_diff}");
    }

    /// The conv state is the trailing `(k-1)*dilation` columns of the stream, so
    /// running a sequence in two chunks must match running it in one. This is the
    /// property the (unwritten) recurrent-state plumbing has to preserve.
    #[test]
    fn chunked_conv_matches_whole_sequence() {
        let (ch, k, dil) = (6i64, 4i64, 3i64);
        let state_len = (k - 1) * dil;
        let seq = 16i64;
        let w = Tensor::randn([ch, 1, k], (Kind::Float, Device::Cpu));
        let x = Tensor::randn([1, ch, seq], (Kind::Float, Device::Cpu));

        let conv = |stream: &Tensor| stream.conv1d::<&Tensor>(&w, None, 1, 0, dil, ch);
        let zeros = Tensor::zeros([1, ch, state_len], (Kind::Float, Device::Cpu));

        let whole = conv(&Tensor::cat(&[zeros.shallow_clone(), x.shallow_clone()], -1));

        let split = 10i64;
        let s1 = Tensor::cat(&[zeros, x.narrow(2, 0, split)], -1);
        let y1 = conv(&s1);
        let state = s1.narrow(2, s1.size()[2] - state_len, state_len);
        let s2 = Tensor::cat(&[state, x.narrow(2, split, seq - split)], -1);
        let y2 = conv(&s2);

        let joined = Tensor::cat(&[y1, y2], -1);
        let diff = f64::try_from((whole - joined).abs().max()).unwrap();
        assert!(diff < 1e-5, "chunked conv diverged from the whole sequence: {diff}");
    }
}
