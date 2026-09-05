//! QSA (Qwen sparse attention) indexer — port of `modules/qsa_indexer.py`'s
//! eager reference path.
//!
//! The full-attention layers of `Qwen4ExpForConditionalGeneration` do not attend
//! densely: a small indexer head decides, per query, which 4-token blocks of the
//! history are visible, and that selection is ANDed into the causal mask.
//!
//! Per token the indexer projects one raw 128-D key (cached **unnormed and
//! unroped** — that is what the indexer key cache stores) and `n_heads` query
//! heads (RMS-normed, then partially roped at the query's position). Every
//! *complete* `compress_ratio` block of raw keys is mean-pooled in fp32,
//! RMS-normed, and roped at the block's **start** position — deterministic once
//! the block closes, which is what makes pooled keys cacheable and incremental.
//! Block scores are `relu(q · k)` summed over the index heads and scaled by
//! `1/sqrt(head_dim)`; each query keeps the top `token_budget / compress_ratio`
//! complete blocks, plus its own incomplete tail block unconditionally.
//!
//! Two forms of the selection are produced, matching upstream:
//!
//! - [`QsaIndexer::token_mask`] — a dense `(bsz, seq, total)` boolean mask that
//!   already includes causality, for a masked-SDPA attention call.
//! - [`QsaIndexer::select_indices`] — the same selection as per-row index lists
//!   for a gathered-attention kernel.
//!
//! **Reference arithmetic, unpaged.** Upstream's fast path runs the scorer,
//! `dsa_topk` and a block-expand kernel per row slab, taking every bound from
//! host-side cache lengths, and supports a paged pooled-key plane. This is the
//! eager batch form (no padding, contiguous positions) those kernels are checked
//! against; `kernels/dsa_topk.cu` is already vendored for when the fast path
//! lands. Nothing here is wired into the model's attention yet.

use crate::modules::{Linear, RmsNorm};
use crate::safetensors::SafeTensors;
use anyhow::Result;
use tch::{Device, Kind, Tensor};

/// Partial rotary over the first `cos.size(-1)` dims of `x`; `cos`/`sin`
/// broadcast over the head axes. Dims past the rotary width pass through.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let r = *cos.size().last().unwrap();
    let d = *x.size().last().unwrap();
    let x_rope = x.narrow(-1, 0, r);
    let h = r / 2;
    let rot = Tensor::cat(&[-x_rope.narrow(-1, h, r - h), x_rope.narrow(-1, 0, h)], -1);
    let rotated = &x_rope * cos + rot * sin;
    if d == r {
        rotated
    } else {
        Tensor::cat(&[rotated, x.narrow(-1, r, d - r)], -1)
    }
}

/// The shape parameters the scoring and selection math needs. Split out from
/// the weights so the selection can be exercised — and reasoned about — without
/// a checkpoint behind it.
#[derive(Clone, Copy, Debug)]
pub struct QsaParams {
    pub n_heads: i64,
    pub head_dim: i64,
    pub token_budget: i64,
    pub compress_ratio: i64,
    /// `token_budget / compress_ratio`: complete blocks kept per query.
    pub block_topk: i64,
    pub scale: f64,
}

impl QsaParams {
    pub fn new(n_heads: i64, head_dim: i64, token_budget: i64, compress_ratio: i64) -> Self {
        Self {
            n_heads,
            head_dim,
            token_budget,
            compress_ratio,
            block_topk: token_budget / compress_ratio,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }
}

pub struct QsaIndexer {
    index_qk_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    /// The main attention's rotary frequencies — the indexer ropes at the same
    /// partial width, not one of its own.
    inv_freq: Tensor,
    attn_factor: f64,
    pub p: QsaParams,
}

impl QsaIndexer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        stc: &SafeTensors,
        key: &str,
        hidden_size: i64,
        n_heads: i64,
        head_dim: i64,
        token_budget: i64,
        compress_ratio: i64,
        rms_norm_eps: f32,
        rope: &crate::rope::RoPE,
        device: Device,
    ) -> Result<Self> {
        // kv_heads is 1: the indexer scores against a single raw key head.
        let index_qk_proj = Linear::load(
            stc,
            &format!("{key}.index_qk_proj"),
            None,
            hidden_size,
            (n_heads + 1) * head_dim,
            device,
            true,
            0.0,
        )?;
        Ok(Self {
            index_qk_proj,
            q_layernorm: RmsNorm::load_biased(stc, &format!("{key}.q_layernorm"), rms_norm_eps, 1.0, device)?,
            k_layernorm: RmsNorm::load_biased(stc, &format!("{key}.k_layernorm"), rms_norm_eps, 1.0, device)?,
            inv_freq: rope.inv_freq.shallow_clone(),
            attn_factor: rope.attn_factor,
            p: QsaParams::new(n_heads, head_dim, token_budget, compress_ratio),
        })
    }
}

impl QsaParams {
    /// Width of a `select_indices` row: `block_topk` blocks plus a tail block,
    /// rounded up to 32 to match the selection kernel's tiling.
    pub fn k_pad(&self) -> i64 {
        let cr = self.compress_ratio;
        let w = self.block_topk * cr + cr - 1;
        (w + 31) / 32 * 32
    }

    /// `q` `(bsz, seq, H, dk)`, `pooled` `(bsz, nb, dk)` -> `(bsz, seq, nb)` fp32.
    /// The relu is per head *before* the sum, so a head that dislikes a block
    /// contributes nothing rather than vetoing the heads that like it.
    pub fn block_scores(&self, q: &Tensor, pooled: &Tensor) -> Tensor {
        let s = Tensor::einsum(
            "bshd,bnd->bshn",
            &[q.to_kind(Kind::Float), pooled.to_kind(Kind::Float)],
            None::<i64>,
        );
        s.relu().sum_dim_intlist(vec![2i64].as_slice(), false, Kind::Float) * self.scale
    }

    /// Per-query count of complete blocks visible at absolute position `p`.
    fn nb_visible(&self, abs_pos: &Tensor) -> Tensor {
        (abs_pos + 1).floor_divide(&Tensor::from(self.compress_ratio).to_device(abs_pos.device()))
    }

    /// Selection as a mask for causal, unpadded attention: `(bsz, seq, total_len)`
    /// bool, true = may attend, causality already folded in. `past_len` is the
    /// absolute position of the first query.
    pub fn token_mask(&self, q: &Tensor, pooled: &Tensor, past_len: i64, total_len: i64) -> Tensor {
        let sz = q.size();
        let (bsz, seq) = (sz[0], sz[1]);
        let cr = self.compress_ratio;
        let dev = q.device();
        let abs_pos = Tensor::arange(seq, (Kind::Int64, dev)) + past_len;
        let kv_pos = Tensor::arange(total_len, (Kind::Int64, dev));
        let nb_q = self.nb_visible(&abs_pos);
        let nb = pooled.size()[1];

        // The tail block (everything past the last complete block) is always
        // visible, bounded by causality.
        let causal = kv_pos.unsqueeze(0).le_tensor(&abs_pos.unsqueeze(1));
        let mut mask = kv_pos
            .unsqueeze(0)
            .ge_tensor(&(&nb_q.unsqueeze(1) * cr))
            .logical_and(&causal)
            .unsqueeze(0)
            .expand([bsz, seq, total_len], false)
            .contiguous();

        if nb > 0 {
            let scores = self.block_scores(q, pooled);
            let block_j = Tensor::arange(nb, (Kind::Int64, dev));
            let out_of_range = block_j.unsqueeze(0).ge_tensor(&nb_q.unsqueeze(1)).unsqueeze(0);
            let scores = scores.masked_fill(&out_of_range, f64::NEG_INFINITY);
            let k = self.block_topk.min(nb);
            let sel = scores.topk(k, -1, true, true).1;
            let block_mask = Tensor::zeros([bsz, seq, nb], (Kind::Int, dev))
                .scatter_value(-1, &sel, 1i64)
                .to_kind(Kind::Bool);
            let mut token_sel = block_mask.repeat_interleave_self_int(cr, -1, None);
            let have = token_sel.size()[2];
            if have < total_len {
                token_sel = Tensor::cat(
                    &[token_sel, Tensor::zeros([bsz, seq, total_len - have], (Kind::Bool, dev))],
                    -1,
                );
            } else if have > total_len {
                token_sel = token_sel.narrow(2, 0, total_len);
            }
            // A selected block may run past the query's own position; causality
            // trims it here rather than in the top-k.
            mask = mask.logical_or(&token_sel.logical_and(&causal.unsqueeze(0)));
        }
        mask
    }

    /// The same selection as flat row index lists: per query row, the top
    /// `block_topk` complete visible blocks expanded to tokens plus the tail
    /// block, as `b * batch_stride + t`, `-1` padded to [`Self::k_pad`].
    pub fn select_indices(
        &self,
        q: &Tensor,
        pooled: &Tensor,
        past_len: i64,
        batch_stride: i64,
    ) -> Tensor {
        let sz = q.size();
        let (bsz, seq) = (sz[0], sz[1]);
        let cr = self.compress_ratio;
        let dev = q.device();
        let nb = pooled.size()[1];
        let k_pad = self.k_pad();
        let out = Tensor::full([bsz, seq, k_pad], -1i64, (Kind::Int, dev));
        let boffs = (Tensor::arange(bsz, (Kind::Int64, dev)) * batch_stride).view([bsz, 1, 1]);

        let qpos = Tensor::arange(seq, (Kind::Int64, dev)) + past_len;
        let nbq = self.nb_visible(&qpos);
        let within = Tensor::arange(cr, (Kind::Int64, dev));

        let (sel_tok, sel_ok) = if nb > 0 {
            let scores = self.block_scores(q, pooled);
            let out_of_range = Tensor::arange(nb, (Kind::Int64, dev))
                .view([1, 1, -1])
                .ge_tensor(&nbq.view([1, -1, 1]));
            let scores = scores.masked_fill(&out_of_range, f64::NEG_INFINITY);
            let ksel = self.block_topk.min(nb);
            let (vals, idx) = scores.topk(ksel, -1, true, true);
            // A query early in the sequence may have fewer than ksel complete
            // blocks; those slots come back as -inf and must not be emitted.
            let ok = vals.gt(f64::NEG_INFINITY);
            let tok = (idx * cr).unsqueeze(-1) + &within;
            (
                tok.flatten(2, 3),
                ok.unsqueeze(-1).expand([bsz, seq, ksel, cr], false).flatten(2, 3),
            )
        } else {
            (
                Tensor::zeros([bsz, seq, 0], (Kind::Int64, dev)),
                Tensor::zeros([bsz, seq, 0], (Kind::Bool, dev)),
            )
        };

        let tail_tok = (&nbq * cr).view([1, -1, 1]) + &within;
        let tail_tok = tail_tok.expand([bsz, seq, cr], false);
        let tok = Tensor::cat(&[sel_tok, tail_tok.shallow_clone()], 2);
        let ok = Tensor::cat(
            &[sel_ok, Tensor::ones([bsz, seq, cr], (Kind::Bool, dev))],
            2,
        )
        .logical_and(&tok.le_tensor(&qpos.view([1, -1, 1])));
        let l = tok.size()[2];
        let vals = (tok + &boffs)
            .where_self(&ok, &Tensor::from(-1i64).to_device(dev))
            .to_kind(Kind::Int);
        let _ = out.narrow(2, 0, l).copy_(&vals);
        out.view([bsz * seq, k_pad])
    }
}

impl QsaIndexer {
    /// NEOX-style rope tables for positions `pos0 .. pos0 + n`, shaped
    /// `(1, n, rope_dim)` — each half-frequency duplicated, matching the
    /// `cat(freqs, freqs)` layout [`rope`] expects.
    pub fn rope_tables(&self, pos0: i64, n: i64, device: Device) -> (Tensor, Tensor) {
        let pos = (Tensor::arange(n, (Kind::Float, device)) + pos0 as f64).unsqueeze(-1);
        let f = pos * self.inv_freq.to_device(device).to_kind(Kind::Float).unsqueeze(0);
        let emb = Tensor::cat(&[f.shallow_clone(), f], -1).unsqueeze(0);
        (emb.cos() * self.attn_factor, emb.sin() * self.attn_factor)
    }

    /// `x` `(bsz, seq, hidden)`; `cos_q`/`sin_q` the rope tables at the query
    /// positions, broadcastable to `(bsz, seq, 1, rope_dim)`.
    ///
    /// Returns `q` `(bsz, seq, n_heads, head_dim)` normed and roped, and `raw_k`
    /// `(bsz, seq, head_dim)` — unnormed and unroped, because the pooling that
    /// consumes it happens in fp32 *before* the norm and the rope position of a
    /// pooled key is its block's start, not the token's own.
    pub fn project(&self, x: &Tensor, cos_q: &Tensor, sin_q: &Tensor) -> (Tensor, Tensor) {
        let sz = x.size();
        let (bsz, seq) = (sz[0], sz[1]);
        let qk = self.index_qk_proj.forward(&x.contiguous());
        let qw = self.p.n_heads * self.p.head_dim;
        let q = qk.narrow(-1, 0, qw);
        let raw_k = qk.narrow(-1, qw, self.p.head_dim).contiguous();
        let q = self
            .q_layernorm
            .forward(&q.reshape([bsz, seq, self.p.n_heads, self.p.head_dim]).contiguous());
        (rope(&q, &cos_q.unsqueeze(-2), &sin_q.unsqueeze(-2)), raw_k)
    }

    /// `raw_k` `(bsz, total, head_dim)` full raw key history -> pooled block keys
    /// `(bsz, nb, head_dim)`. The trailing incomplete block is not pooled at all;
    /// it is force-included by the selection instead.
    pub fn pool_keys(&self, raw_k: &Tensor, cos_full: &Tensor, sin_full: &Tensor) -> Tensor {
        let sz = raw_k.size();
        let (bsz, total, dk) = (sz[0], sz[1], sz[2]);
        let cr = self.p.compress_ratio;
        let nb = total / cr;
        if nb == 0 {
            return Tensor::zeros([bsz, 0, dk], (raw_k.kind(), raw_k.device()));
        }
        let pooled = raw_k
            .narrow(1, 0, nb * cr)
            .view([bsz, nb, cr, dk])
            .to_kind(Kind::Float)
            .mean_dim(2, false, Kind::Float)
            .to_kind(raw_k.kind());
        let pooled = self.k_layernorm.forward(&pooled);
        let starts = Tensor::arange_start_step(0, nb * cr, cr, (Kind::Int64, raw_k.device()));
        rope(
            &pooled,
            &cos_full.index_select(-2, &starts),
            &sin_full.index_select(-2, &starts),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident_rope(seq: i64, r: i64) -> (Tensor, Tensor) {
        // cos = 1, sin = 0: rope becomes the identity, so tests can isolate the
        // selection logic from the rotary.
        (
            Tensor::ones([1, seq, r], (Kind::Float, Device::Cpu)),
            Tensor::zeros([1, seq, r], (Kind::Float, Device::Cpu)),
        )
    }


    /// Deterministic pseudo-random q/pooled: the selection must be exercised
    /// with scores that actually differ between blocks.
    fn synth(seq: i64, nb: i64, h: i64, dk: i64) -> (Tensor, Tensor) {
        let n = seq * h * dk;
        let q = (Tensor::arange(n, (Kind::Float, Device::Cpu)) * 0.7).sin().view([1, seq, h, dk]);
        let k = (Tensor::arange(nb * dk, (Kind::Float, Device::Cpu)) * 1.3).cos().view([1, nb, dk]);
        (q, k)
    }

    #[test]
    fn mask_is_causal_and_keeps_the_tail_block() {
        let p = QsaParams::new(2, 8, 8, 4); // budget 8 -> 2 blocks of 4
        let (total, past) = (24, 0);
        let nb = total / p.compress_ratio;
        let (q, pooled) = synth(total, nb, p.n_heads, p.head_dim);
        let mask = p.token_mask(&q, &pooled, past, total);
        for s in 0..total {
            let row = mask.select(0, 0).select(0, s);
            // nothing past the query position
            assert!(!bool::try_from(row.narrow(0, s + 1, total - s - 1).any()).unwrap());
            // the whole visible part of the tail block
            let nbq = (s + 1) / p.compress_ratio;
            for t in nbq * p.compress_ratio..=s {
                assert!(bool::try_from(row.get(t)).unwrap(), "pos {s} lost tail token {t}");
            }
        }
    }

    /// A query may only ever see `token_budget` tokens plus its tail block. If
    /// this slips the layer silently stops being sparse and the KV traffic it
    /// exists to avoid comes right back.
    #[test]
    fn selection_never_exceeds_the_budget() {
        let p = QsaParams::new(2, 8, 8, 4);
        let total = 40;
        let nb = total / p.compress_ratio;
        let (q, pooled) = synth(total, nb, p.n_heads, p.head_dim);
        let mask = p.token_mask(&q, &pooled, 0, total);
        let counts = mask.to_kind(Kind::Int64).sum_dim_intlist(vec![-1i64].as_slice(), false, Kind::Int64);
        let cap = p.token_budget + p.compress_ratio;
        assert!(f64::try_from(counts.max()).unwrap() <= cap as f64);
    }

    /// The two selection forms are the same selection. The mask feeds masked
    /// SDPA and the index lists feed the gathered kernel, so a divergence would
    /// show up only as a quality gap between two paths that should be identical.
    #[test]
    fn index_lists_agree_with_the_mask() {
        let p = QsaParams::new(2, 8, 8, 4);
        let (total, past) = (32, 0);
        let nb = total / p.compress_ratio;
        let (q, pooled) = synth(total, nb, p.n_heads, p.head_dim);
        let mask = p.token_mask(&q, &pooled, past, total);
        let idx = p.select_indices(&q, &pooled, past, total);
        for s in 0..total {
            let mut from_idx = vec![false; total as usize];
            let row = idx.select(0, s);
            for j in 0..row.size()[0] {
                let v = i64::try_from(row.get(j)).unwrap();
                if v >= 0 {
                    from_idx[v as usize] = true;
                }
            }
            for t in 0..total {
                let want = bool::try_from(mask.select(0, 0).select(0, s).get(t)).unwrap();
                assert_eq!(want, from_idx[t as usize], "row {s} token {t}");
            }
        }
    }

    /// Batch rows are addressed as `b * batch_stride + t`, and each row's
    /// selection is its own — a stride bug would silently make sequence 1 attend
    /// into sequence 0's keys.
    #[test]
    fn index_lists_offset_each_sequence() {
        let p = QsaParams::new(2, 8, 8, 4);
        let total = 16;
        let nb = total / p.compress_ratio;
        let (q0, pooled0) = synth(total, nb, p.n_heads, p.head_dim);
        let q = Tensor::cat(&[q0.shallow_clone(), q0.shallow_clone()], 0);
        let pooled = Tensor::cat(&[pooled0.shallow_clone(), pooled0], 0);
        let idx = p.select_indices(&q, &pooled, 0, 100);
        let a = idx.narrow(0, 0, total);
        let b = idx.narrow(0, total, total);
        let valid = a.ge(0);
        let shifted = (&b - &a).masked_select(&valid);
        assert!(f64::try_from(shifted.to_kind(Kind::Float).min()).unwrap() == 100.0);
        assert!(f64::try_from(shifted.to_kind(Kind::Float).max()).unwrap() == 100.0);
    }

    /// Scores relu per head *then* sum: one enthusiastic head must be able to
    /// carry a block that every other head scores negative.
    #[test]
    fn a_single_head_can_carry_a_block() {
        let p = QsaParams::new(3, 4, 4, 4); // top-1 block
        let dk = p.head_dim;
        let q = Tensor::zeros([1, 1, 3, dk], (Kind::Float, Device::Cpu));
        let pooled = Tensor::zeros([1, 2, dk], (Kind::Float, Device::Cpu));
        // head 0 loves block 1; heads 1 and 2 hate it and mildly like block 0.
        let _ = q.select(0, 0).select(0, 0).select(0, 0).get(0).fill_(1.0);
        let _ = q.select(0, 0).select(0, 0).select(0, 1).get(1).fill_(-1.0);
        let _ = q.select(0, 0).select(0, 0).select(0, 2).get(1).fill_(-1.0);
        let _ = pooled.select(0, 0).select(0, 1).get(0).fill_(4.0);
        let _ = pooled.select(0, 0).select(0, 1).get(1).fill_(1.0);
        let _ = pooled.select(0, 0).select(0, 0).get(1).fill_(-1.0);
        let scores = p.block_scores(&q, &pooled);
        assert!(
            f64::try_from(scores.get(0).get(0).get(1)).unwrap()
                > f64::try_from(scores.get(0).get(0).get(0)).unwrap()
        );
    }

    #[test]
    fn rope_leaves_the_pass_through_dims_alone() {
        let x = Tensor::arange(8, (Kind::Float, Device::Cpu)).view([1, 1, 1, 8]);
        let (cos, sin) = ident_rope(1, 4);
        let y = rope(&x, &cos.unsqueeze(-2), &sin.unsqueeze(-2));
        assert!(f64::try_from((y - x).abs().max()).unwrap() < 1e-6);
    }
}
