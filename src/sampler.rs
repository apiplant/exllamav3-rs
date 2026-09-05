//! Sampler chain — partial port of `generator/sampler/custom.py` (grade C).
//!
//! Ported stages, applied in the upstream order (penalties first, then the
//! temperature/truncation/sample tail): repetition penalty, presence/frequency
//! penalty, temperature, top-k, top-p, min-p, then argmax or categorical sample.
//!
//! Differences from upstream (all documented in PLAN.md): sampling uses
//! `torch.multinomial` rather than the exact Gumbel-noise kernel, so the RNG is
//! not bit-identical (upstream's own docstring: "Doesn't guarantee perfect
//! determinism"). The repetition / presence / frequency penalties now use the
//! `apply_rep_pens` / `apply_pres_freq_pens` CUDA kernels with the same
//! sustain/decay windowing as `custom.py` (`sustain_range` / `decay_range`;
//! defaults reproduce full-history behaviour). DRY, XTC, mirostat, typical-p,
//! quadratic, adaptive-p, skew and grammar filters are not ported.

use std::collections::HashMap;
use tch::{Kind, Tensor};

#[derive(Clone, Debug)]
pub struct SamplerSettings {
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    pub min_p: f64,
    /// multiplicative repetition penalty (`SS_RepP`), 1.0 = off
    pub rep_penalty: f64,
    /// additive presence penalty (`SS_PresFreqP`)
    pub pres_penalty: f64,
    /// additive frequency penalty (`SS_PresFreqP`)
    pub freq_penalty: f64,
    /// penalty window: tokens within `sustain_range` back get the full penalty
    /// (0 = whole history), the next `decay_range` fade it out linearly
    pub sustain_range: i64,
    pub decay_range: i64,
}

impl Default for SamplerSettings {
    /// Matches `presets.DefaultSampler` (min_p 0.08, temperature 0.8) but greedy
    /// unless the caller opts into randomness.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            rep_penalty: 1.0,
            pres_penalty: 0.0,
            freq_penalty: 0.0,
            sustain_range: 0,
            decay_range: 0,
        }
    }
}

impl SamplerSettings {
    pub fn greedy() -> Self {
        Self::default()
    }

    pub(crate) fn needs_past_ids(&self) -> bool {
        self.rep_penalty != 1.0 || self.pres_penalty != 0.0 || self.freq_penalty != 0.0
    }

    /// Sample one token id from `logits` (shape `[vocab]`, f32, any device).
    /// `past_ids` is the full token history (prompt + generated) for the penalties.
    pub fn sample(&self, logits: &Tensor, past_ids: &[i64]) -> i64 {
        // avoid a redundant full-vocab copy when the caller already handed us f32
        let mut l = if logits.kind() == Kind::Float {
            logits.shallow_clone()
        } else {
            logits.to_kind(Kind::Float)
        };

        if self.needs_past_ids() && !past_ids.is_empty() {
            l = self.apply_penalties(&l, past_ids);
        }

        if self.temperature <= 0.0 {
            return l.argmax(0, false).int64_value(&[]);
        }
        l = l / self.temperature;

        if self.top_k > 0 {
            let k = self.top_k.min(l.size()[0]);
            let (vals, _) = l.topk(k, 0, true, true);
            let kth = vals.get(k - 1).double_value(&[]);
            l = l.where_scalarother(&l.ge(kth), f64::NEG_INFINITY);
        }

        let mut probs = l.softmax(0, Kind::Float);

        if self.min_p > 0.0 {
            let top = probs.max().double_value(&[]);
            let keep = probs.ge(top * self.min_p);
            probs = probs.where_scalarother(&keep, 0.0);
        }

        if self.top_p < 1.0 {
            let (sorted, idx) = probs.sort(0, true);
            let cum = sorted.cumsum(0, Kind::Float);
            // keep everything up to and including the crossing element
            let mask = cum.le(self.top_p).logical_or(
                &Tensor::arange(sorted.size()[0], (Kind::Int64, sorted.device())).eq(0),
            );
            let kept = sorted.where_scalarother(&mask, 0.0);
            probs = probs.zeros_like().scatter(0, &idx, &kept);
        }

        let s = probs.sum(Kind::Float).double_value(&[]);
        if s <= 0.0 {
            return probs.argmax(0, false).int64_value(&[]);
        }
        probs = &probs / s;
        probs.multinomial(1, true).int64_value(&[0])
    }

    /// `SS_RepP` (multiplicative) then `SS_PresFreqP` (additive), in upstream
    /// order. On CUDA this uses the `apply_rep_pens` / `apply_pres_freq_pens`
    /// kernels (sustain/decay windowed, exactly as `generator/sampler/custom.py`);
    /// on CPU it falls back to the equivalent full-history computation.
    fn apply_penalties(&self, logits: &Tensor, past_ids: &[i64]) -> Tensor {
        if logits.device().is_cuda() {
            let dev = logits.device();
            // kernels want [1, vocab] f32 logits and [1, past_len] i64 ids
            let past = Tensor::from_slice(past_ids).to_device(dev).reshape([1, past_ids.len() as i64]);
            let sustain = if self.sustain_range > 0 {
                self.sustain_range as i32
            } else {
                past_ids.len() as i32
            };
            let decay = self.decay_range.max(0) as i32;
            let out = logits.to_kind(Kind::Float).reshape([1, -1]);
            if self.rep_penalty != 1.0 {
                crate::ffi::apply_rep_pens(&out, &out, &past, self.rep_penalty as f32, sustain, decay);
            }
            if self.pres_penalty != 0.0 || self.freq_penalty != 0.0 {
                crate::ffi::apply_pres_freq_pens(
                    &out, &out, &past,
                    self.pres_penalty as f32, self.freq_penalty as f32, sustain, decay,
                );
            }
            return out.reshape([-1]);
        }
        self.apply_penalties_cpu(logits, past_ids)
    }

    fn apply_penalties_cpu(&self, logits: &Tensor, past_ids: &[i64]) -> Tensor {
        let device = logits.device();
        let mut counts: HashMap<i64, i64> = HashMap::new();
        for &t in past_ids {
            *counts.entry(t).or_insert(0) += 1;
        }
        let ids: Vec<i64> = counts.keys().copied().collect();
        let idx = Tensor::from_slice(&ids).to_device(device);
        let mut l = logits.shallow_clone();

        if self.rep_penalty != 1.0 {
            let sel = l.index_select(0, &idx);
            // positive logits divided by rep_p, negative ones multiplied by it
            let pos = sel.gt(0.0);
            let negs = sel.where_scalarother(&pos.logical_not(), 0.0) * self.rep_penalty;
            let poss = sel.where_scalarother(&pos, 0.0) * (1.0 / self.rep_penalty);
            let penalised = negs + poss;
            l = l.index_copy(0, &idx, &penalised);
        }

        if self.pres_penalty != 0.0 || self.freq_penalty != 0.0 {
            let cvec: Vec<f64> = ids.iter().map(|t| counts[t] as f64).collect();
            let c = Tensor::from_slice(&cvec).to_device(device);
            let delta = c * self.freq_penalty + self.pres_penalty;
            let sel = l.index_select(0, &idx) - delta;
            l = l.index_copy(0, &idx, &sel);
        }

        l
    }
}
