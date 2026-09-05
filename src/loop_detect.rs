//! Streaming loop detector — 1:1 port of `generator/loop_detect.py` `LoopDetector`
//! (grade A; pure-Python logic file, no CUDA involved).
//!
//! Flat-latency: observes streamed tokens and fires when the *entire* observed
//! window of `window_size` tokens is made up of a repeating sequence of some
//! period `p` (`1 ..= max_period`). The heap schedules per-period checks so the
//! common (no-loop) case stays O(1) amortised.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct LoopDetector {
    w: i64,
    max_period: usize,
    buf: Vec<Option<i64>>,
    total: i64,
    /// consecutive matches for each period (index `0` unused)
    streak: Vec<i64>,
    /// scheduling heap of `(wake_time, period, generation)`
    heap: BinaryHeap<Reverse<(i64, usize, u64)>>,
    /// current generation counter per period (stale heap entries are skipped)
    gen: Vec<u64>,
    detected_period: Option<usize>,
}

impl LoopDetector {
    /// `window_size` in tokens; `max_period` defaults to `window_size / 3` and is
    /// capped at `window_size / 2` (no longer period can tile the window).
    pub fn new(window_size: i64, max_period: Option<usize>) -> Self {
        let w = window_size.max(2);
        let max_period = max_period
            .unwrap_or((w / 3) as usize)
            .min((w / 2) as usize)
            .max(1);
        let mut heap = BinaryHeap::new();
        for p in 1..=max_period {
            heap.push(Reverse((w, p, 0u64)));
        }
        Self {
            w,
            max_period,
            buf: vec![None; w as usize],
            total: 0,
            streak: vec![0; max_period + 1],
            heap,
            gen: vec![0; max_period + 1],
            detected_period: None,
        }
    }

    pub fn detected(&self) -> bool {
        self.detected_period.is_some()
    }
    pub fn period(&self) -> Option<usize> {
        self.detected_period
    }

    fn schedule(&mut self, p: usize, wake_time: i64) {
        self.gen[p] += 1;
        self.heap.push(Reverse((wake_time, p, self.gen[p])));
    }

    /// Count consecutive positions back from the newest satisfying `s[i] == s[i-p]`.
    fn backlog_scan(&self, p: usize) -> i64 {
        let t = self.total;
        let mut streak = 0;
        let max_check = self.w - p as i64;
        for k in 0..max_check {
            let ia = (t - 1 - k).rem_euclid(self.w) as usize;
            let ib = (t - 1 - k - p as i64).rem_euclid(self.w) as usize;
            if self.buf[ia] == self.buf[ib] {
                streak += 1;
            } else {
                break;
            }
        }
        streak
    }

    /// Feed one token. Returns `true` while a loop is currently detected.
    pub fn feed(&mut self, token: i64) -> bool {
        let pos = self.total.rem_euclid(self.w) as usize;
        self.buf[pos] = Some(token);
        self.total += 1;

        if self.total < self.w {
            return false;
        }

        let t = self.total;
        let mut detected = false;

        while let Some(&Reverse((wt, _, _))) = self.heap.peek() {
            if wt > t {
                break;
            }
            let Reverse((_, p, g)) = self.heap.pop().unwrap();
            if g != self.gen[p] {
                continue; // stale entry
            }

            if self.streak[p] > 0 {
                // active detector: check the newest token
                let a = self.buf[(t - 1).rem_euclid(self.w) as usize];
                let b = self.buf[(t - 1 - p as i64).rem_euclid(self.w) as usize];
                if a == b {
                    self.streak[p] += 1;
                    if self.streak[p] >= self.w - p as i64 {
                        self.detected_period = Some(p);
                        detected = true;
                    }
                    self.schedule(p, t + 1);
                } else {
                    self.streak[p] = 0;
                    if self.detected_period == Some(p) {
                        self.detected_period = None;
                    }
                    self.schedule(p, t + self.w);
                }
            } else {
                // waking from sleep: scan the backlog
                let streak = self.backlog_scan(p);
                self.streak[p] = streak;
                if streak >= self.w - p as i64 {
                    self.detected_period = Some(p);
                    detected = true;
                    self.schedule(p, t + 1);
                } else if streak > 0 {
                    self.schedule(p, t + 1);
                } else {
                    self.schedule(p, t + self.w);
                }
            }
        }

        detected || self.detected_period.is_some()
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        *self = LoopDetector::new(self.w, Some(self.max_period));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_period_4_loop() {
        // A pure [0,1,2,3] tile fills the window at t == W (i == 99); it also has
        // period 8,12,…,32 — the detector reports the largest, matching the
        // Python reference exactly (fires at i=99, period=32).
        let mut d = LoopDetector::new(100, None);
        let mut fired = None;
        for i in 0..400 {
            if d.feed((i % 4) as i64) {
                fired = Some(i);
                break;
            }
        }
        assert_eq!(fired, Some(99));
        let p = d.period().unwrap();
        assert!(p % 4 == 0 && p <= 33, "period {p}");
    }

    #[test]
    fn no_false_positive_on_varied_text() {
        // deterministic non-periodic stream (LCG), longer than any window
        let mut d = LoopDetector::new(64, None);
        let mut x: u64 = 12345;
        let mut any = false;
        for _ in 0..5000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            any |= d.feed((x >> 40) as i64);
        }
        assert!(!any);
    }
}
