//! Online drafter-confidence → acceptance calibration, used to cut a speculative
//! draft block short at the first position that is unlikely to be accepted.
//!
//! Port of upstream's `generator/draft_confidence.py`. The drafter reports a
//! per-position confidence (the argmax **logit** value); this maps that score to
//! an observed acceptance probability so the generator can stop drafting once
//! the estimated probability of the whole block being accepted falls below the
//! target. Every skipped draft position saves a full `lm_head` pass — 0.686 ms
//! on this model, ~1.7% of a decode step each — plus a narrower verify.
//!
//! Scores go into fixed-width bins of exponentially decayed `(tested, accepted)`
//! counts. Labels come only from positions the verifier actually tested: the
//! accepted ones, and the first mismatch.

use std::collections::HashMap;

pub struct DraftConfidence {
    /// target probability that a drafted block is accepted in full
    pub confidence: f32,
    bin_width: f32,
    decay: f64,
    min_count: f64,
    burn_in: f64,
    /// bin index -> (tested, accepted), decayed
    bins: HashMap<i64, (f64, f64)>,
    total: f64,
    cached_threshold: Option<f32>,
}

impl DraftConfidence {
    pub fn new(confidence: f32) -> Self {
        assert!(
            confidence > 0.0 && confidence < 1.0,
            "draft_confidence must be in (0, 1)"
        );
        Self {
            confidence,
            bin_width: 1.0,
            decay: 0.995,
            min_count: 8.0,
            burn_in: 64.0,
            bins: HashMap::new(),
            total: 0.0,
            cached_threshold: None,
        }
    }

    fn bin(&self, score: f32) -> i64 {
        (score / self.bin_width).floor() as i64
    }

    pub fn add_label(&mut self, score: f32, accepted: bool) {
        let idx = self.bin(score);
        let e = self.bins.entry(idx).or_insert((0.0, 0.0));
        e.0 += 1.0;
        if accepted {
            e.1 += 1.0;
        }
        self.total += 1.0;
        self.cached_threshold = None;
    }

    /// Age the statistics once per verification round, so the mapping tracks
    /// drift in output style (prose vs code) over a few hundred rounds.
    pub fn decay_step(&mut self) {
        for b in self.bins.values_mut() {
            b.0 *= self.decay;
            b.1 *= self.decay;
        }
        self.total *= self.decay;
        self.cached_threshold = None;
    }

    /// Estimated conditional acceptance probability for a drafted position with
    /// this score, from the nearest populated bin at or below it. Optimistic 1.0
    /// while no statistics exist, so drafting keeps producing full windows (and
    /// therefore labels) during the learning phase.
    pub fn estimate(&self, score: f32) -> f32 {
        if self.total < self.burn_in || self.bins.is_empty() {
            return 1.0;
        }
        let idx = self.bin(score);
        let mut populated: Vec<i64> = self
            .bins
            .iter()
            .filter(|(_, v)| v.0 >= self.min_count)
            .map(|(k, _)| *k)
            .collect();
        if populated.is_empty() {
            return 1.0;
        }
        populated.sort_unstable();
        let k = match populated.iter().rev().find(|&&k| k <= idx) {
            Some(&k) => k,
            None => populated[0],
        };
        let (tested, accepted) = self.bins[&k];
        (accepted / tested) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimistic_before_burn_in() {
        let mut c = DraftConfidence::new(0.4);
        assert_eq!(c.estimate(0.0), 1.0);
        for _ in 0..10 {
            c.add_label(5.0, true);
        }
        // still under burn_in -> optimistic, so full windows keep producing labels
        assert_eq!(c.estimate(5.0), 1.0);
    }

    #[test]
    fn learns_that_low_scores_are_rejected() {
        let mut c = DraftConfidence::new(0.4);
        for _ in 0..100 {
            c.add_label(20.0, true); // confident positions get accepted
            c.add_label(1.0, false); // unconfident ones do not
        }
        assert!(c.estimate(20.0) > 0.9, "{}", c.estimate(20.0));
        assert!(c.estimate(1.0) < 0.1, "{}", c.estimate(1.0));
        // a score between the populated bins falls back to the nearest below
        assert!(c.estimate(2.0) < 0.1);
    }

    #[test]
    fn decay_ages_statistics() {
        let mut c = DraftConfidence::new(0.4);
        for _ in 0..100 {
            c.add_label(20.0, true);
        }
        let before = c.total;
        c.decay_step();
        assert!(c.total < before);
    }
}
