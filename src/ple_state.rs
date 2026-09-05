//! Per-slot recurrent state for the PLE layer (`Qwen4ExpForConditionalGeneration`).
//!
//! [`crate::ple`] is a pure function of `(streams, embedding, conv_state)`, and
//! [`crate::ngram`] is a pure function of a token *history* that reaches
//! `ngram_size - 1` tokens before the chunk. Incremental decoding therefore needs
//! two things carried across forwards, per sequence:
//!
//! * the trailing `conv_state_len = (conv_kernel_size - 1) * ngram_size` columns
//!   of the PLE conv input, and
//! * the trailing `ngram_size - 1` token ids, so the next chunk's n-grams hash
//!   the same way they would in a single-shot prefill.
//!
//! Both follow the same alloc/rewind/stash contract as the GDN state in
//! [`crate::paged`], and for the same reason: a speculative forward runs ahead of
//! what the sampler accepts, so every buffer keeps `max_history` extra trailing
//! entries and [`PleState::rewind`] re-derives the working state from the tail
//! once the accepted count is known. The layouts mirror `Qwen35PagedCache`'s
//! GDN planes deliberately — this is meant to slot into that cache when the
//! Qwen4Exp block exists, not to be a parallel mechanism.
//!
//! A fresh slot's token history is filled with eos, not zeros: eos is what
//! upstream's hashing reads across a segment boundary, so an eos-filled history
//! is exactly "no preceding context" rather than "the token whose id is 0".

use tch::{Device, Kind, Tensor};

pub struct PleState {
    /// `[max_slots, width, conv_state_len + max_history]` fp32. `[.., :conv_len]`
    /// is the working window; the rest is the rewind tail.
    conv: Tensor,
    /// `[max_slots, context_len + max_history]` int64, on the CPU where the
    /// hashing runs. Same split as `conv`.
    tokens: Tensor,
    conv_len: i64,
    context_len: i64,
    max_history: i64,
    eos_token_id: i64,
}

impl PleState {
    /// `width` is the PLE conv's channel count (`hc_mult * hidden_size`).
    /// `max_history` is the longest speculative forward that can be rewound; 0
    /// disables rewind and keeps only the working state.
    pub fn new(
        max_slots: i64,
        width: i64,
        conv_state_len: i64,
        context_len: i64,
        max_history: i64,
        eos_token_id: i64,
        device: Device,
    ) -> Self {
        let s = Self {
            conv: Tensor::zeros(
                [max_slots, width, conv_state_len + max_history],
                (Kind::Float, device),
            ),
            tokens: Tensor::zeros([max_slots, context_len + max_history], (Kind::Int64, Device::Cpu)),
            conv_len: conv_state_len,
            context_len,
            max_history,
            eos_token_id,
        };
        for slot in 0..max_slots {
            s.reset_slot(slot);
        }
        s
    }

    pub fn conv_state_len(&self) -> i64 {
        self.conv_len
    }

    pub fn context_len(&self) -> i64 {
        self.context_len
    }

    /// Clear one slot to "start of sequence" — call when a job is admitted.
    pub fn reset_slot(&self, slot: i64) {
        let _ = self.conv.narrow(0, slot, 1).zero_();
        let _ = self.tokens.narrow(0, slot, 1).fill_(self.eos_token_id);
    }

    /// The working conv state for the next forward: `(width, conv_state_len)`.
    pub fn window(&self, slot: i64) -> Tensor {
        self.conv.select(0, slot).narrow(1, 0, self.conv_len)
    }

    /// The token history to prepend to `new_tokens` before hashing:
    /// `(1, context_len + new_tokens.len())` int64 on the CPU, ready for
    /// [`crate::ngram::NGramEmbedding::forward`].
    pub fn token_input(&self, slot: i64, new_tokens: &Tensor) -> Tensor {
        let hist = self.tokens.select(0, slot).narrow(0, 0, self.context_len);
        Tensor::cat(&[hist, new_tokens.to_device(Device::Cpu).to_kind(Kind::Int64).reshape([-1])], 0)
            .unsqueeze(0)
    }

    /// Commit a forward of `consumed` tokens. `cols` is the conv input for those
    /// tokens, `(width, consumed)`; `new_tokens` their ids.
    ///
    /// Old window and new columns are concatenated and the tail of that is what
    /// the buffers keep, so `rewind` can re-cut the working window at any
    /// accepted length without re-running the conv.
    pub fn push(&self, slot: i64, cols: &Tensor, new_tokens: &Tensor) {
        let consumed = cols.size()[1];
        let full = Tensor::cat(&[self.window(slot), cols.to_kind(Kind::Float)], 1);
        let buf = self.conv.select(0, slot);
        let l = buf.size()[1];
        let keep = l.min(full.size()[1]);
        let _ = buf
            .narrow(1, l - keep, keep)
            .copy_(&full.narrow(1, full.size()[1] - keep, keep));

        let new_tokens = new_tokens.to_device(Device::Cpu).to_kind(Kind::Int64).reshape([-1]);
        let tfull = Tensor::cat(
            &[self.tokens.select(0, slot).narrow(0, 0, self.context_len), new_tokens],
            0,
        );
        let tbuf = self.tokens.select(0, slot);
        let tl = tbuf.size()[0];
        let tkeep = tl.min(tfull.size()[0]);
        let _ = tbuf
            .narrow(0, tl - tkeep, tkeep)
            .copy_(&tfull.narrow(0, tfull.size()[0] - tkeep, tkeep));

        self.rewind(slot, consumed, consumed);
    }

    /// After a [`push`](Self::push) of `consumed` tokens, commit exactly the
    /// first `keep` of them (drop `consumed - keep` rejected draft tokens).
    ///
    /// The working window is re-cut from the tail at `p = L - (consumed - keep)`,
    /// the same arithmetic as `Qwen35PagedCache::gdn_rewind`. `keep == consumed`
    /// is the ordinary non-speculative case and picks the very last entries.
    pub fn rewind(&self, slot: i64, keep: i64, consumed: i64) {
        debug_assert!(keep >= 1 && keep <= consumed);
        debug_assert!(
            keep == consumed || self.max_history >= consumed,
            "rewinding {consumed} tokens needs max_history >= {consumed}"
        );
        let drop = consumed - keep;

        let buf = self.conv.select(0, slot);
        let p = buf.size()[1] - drop;
        let win = buf.narrow(1, p - self.conv_len, self.conv_len).contiguous();
        let _ = buf.narrow(1, 0, self.conv_len).copy_(&win);

        let tbuf = self.tokens.select(0, slot);
        let tp = tbuf.size()[0] - drop;
        let hist = tbuf.narrow(0, tp - self.context_len, self.context_len).contiguous();
        let _ = tbuf.narrow(0, 0, self.context_len).copy_(&hist);
    }

    /// Copy a slot's working state out, for the prefix-cache LRU. Conv columns
    /// go to host RAM alongside the token history; restoring is an H2D copy.
    pub fn snapshot_cpu(&self, slot: i64) -> PleCheckpoint {
        // `copy` (not `contiguous`): a narrow of an already-contiguous row
        // aliases the live buffer, and the next reset_slot would erase the
        // checkpoint out from under its owner.
        PleCheckpoint {
            conv: self.window(slot).to_device(Device::Cpu).copy(),
            tokens: self.tokens.select(0, slot).narrow(0, 0, self.context_len).copy(),
        }
    }

    /// Restore a checkpoint into `slot`. Call after [`reset_slot`](Self::reset_slot)
    /// and before the tail prefill: only the working state is written, and the
    /// history tail stays cleared until the next forward repopulates it.
    pub fn restore(&self, slot: i64, cp: &PleCheckpoint) {
        let buf = self.conv.select(0, slot);
        let _ = buf
            .narrow(1, 0, self.conv_len)
            .copy_(&cp.conv.to_device(buf.device()));
        let _ = self
            .tokens
            .select(0, slot)
            .narrow(0, 0, self.context_len)
            .copy_(&cp.tokens);
    }
}

/// Prefix-cache snapshot of one slot's PLE state, taken at a page boundary.
/// Tiny next to [`crate::paged::GdnCheckpoint`] — one conv window and a handful
/// of token ids — so it rides along with the GDN checkpoint rather than needing
/// its own budget.
pub struct PleCheckpoint {
    pub conv: Tensor,
    pub tokens: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conv this state feeds: depthwise, dilated, causal, no padding —
    /// mirrors `PleLayer::short_conv` so the chunking test below has something
    /// to be equal to.
    fn conv(x: &Tensor, w: &Tensor, dilation: i64) -> Tensor {
        let ch = x.size()[0];
        Tensor::conv1d(
            &x.unsqueeze(0),
            w,
            None::<Tensor>,
            [1],
            [0],
            [dilation],
            ch,
        )
        .squeeze_dim(0)
    }

    fn state(max_history: i64) -> PleState {
        PleState::new(2, 6, 4, 3, max_history, 7, Device::Cpu)
    }

    #[test]
    fn a_fresh_slot_reads_as_start_of_sequence() {
        let s = state(0);
        let toks = Tensor::from_slice(&[11i64, 12]);
        let inp = s.token_input(0, &toks);
        assert_eq!(inp.size(), vec![1, 5]);
        assert_eq!(Vec::<i64>::try_from(inp.select(0, 0)).unwrap(), vec![7, 7, 7, 11, 12]);
        assert!(f64::try_from(s.window(0).abs().max()).unwrap() == 0.0);
    }

    /// Decoding chunk by chunk through the state must give the same conv output
    /// as one pass over the whole sequence. This is the invariant the whole
    /// buffer exists to preserve; break it and the model degrades only slightly,
    /// only in long generations.
    #[test]
    fn chunked_decode_matches_a_single_pass() {
        let (width, k, dil) = (6i64, 2i64, 2i64); // conv_state_len = (k - 1) * dil
        let s = PleState::new(1, width, (k - 1) * dil, 3, 8, 7, Device::Cpu);
        let w = Tensor::randn([width, 1, k], (Kind::Float, Device::Cpu));
        let total = 9;
        let x = Tensor::randn([width, total], (Kind::Float, Device::Cpu));
        let pad = Tensor::zeros([width, (k - 1) * dil], (Kind::Float, Device::Cpu));
        let want = conv(&Tensor::cat(&[pad, x.shallow_clone()], 1), &w, dil);

        let mut got = vec![];
        let mut at = 0;
        for n in [4i64, 1, 1, 3] {
            let cols = x.narrow(1, at, n);
            let inp = Tensor::cat(&[s.window(0), cols.shallow_clone()], 1);
            got.push(conv(&inp, &w, dil));
            let ids = Tensor::zeros([n], (Kind::Int64, Device::Cpu));
            s.push(0, &cols, &ids);
            at += n;
        }
        let got = Tensor::cat(&got, 1);
        assert_eq!(got.size(), want.size());
        assert!(f64::try_from((got - want).abs().max()).unwrap() < 1e-5);
    }

    /// Rewinding a rejected speculative tail must land on exactly the state that
    /// would have been reached by never running those tokens at all.
    #[test]
    fn rewind_lands_on_the_accepted_prefix() {
        let width = 6;
        let cols = Tensor::randn([width, 5], (Kind::Float, Device::Cpu));
        let ids = Tensor::from_slice(&[21i64, 22, 23, 24, 25]);

        let spec = state(8);
        spec.push(0, &cols, &ids);
        spec.rewind(0, 2, 5);

        let plain = state(8);
        plain.push(0, &cols.narrow(1, 0, 2), &ids.narrow(0, 0, 2));

        assert!(f64::try_from((spec.window(0) - plain.window(0)).abs().max()).unwrap() == 0.0);
        assert_eq!(
            Vec::<i64>::try_from(spec.token_input(0, &Tensor::from_slice(&[0i64])).select(0, 0)).unwrap(),
            Vec::<i64>::try_from(plain.token_input(0, &Tensor::from_slice(&[0i64])).select(0, 0)).unwrap()
        );
    }

    /// Slots must not bleed into each other — the failure mode is one request
    /// conditioning on another request's tokens.
    #[test]
    fn slots_are_independent() {
        let s = state(0);
        let cols = Tensor::ones([6, 3], (Kind::Float, Device::Cpu));
        s.push(0, &cols, &Tensor::from_slice(&[1i64, 2, 3]));
        assert!(f64::try_from(s.window(1).abs().max()).unwrap() == 0.0);
        assert_eq!(
            Vec::<i64>::try_from(s.token_input(1, &Tensor::from_slice(&[9i64])).select(0, 0)).unwrap(),
            vec![7, 7, 7, 9]
        );
    }

    /// A checkpoint must round-trip both halves of the state.
    #[test]
    fn checkpoint_round_trips() {
        let s = state(0);
        let cols = Tensor::randn([6, 4], (Kind::Float, Device::Cpu));
        s.push(0, &cols, &Tensor::from_slice(&[31i64, 32, 33, 34]));
        let cp = s.snapshot_cpu(0);
        let want_w = s.window(0).contiguous();
        let want_t = Vec::<i64>::try_from(s.token_input(0, &Tensor::from_slice(&[0i64])).select(0, 0)).unwrap();

        s.reset_slot(0);
        s.restore(0, &cp);
        assert!(f64::try_from((s.window(0) - want_w).abs().max()).unwrap() == 0.0);
        assert_eq!(
            Vec::<i64>::try_from(s.token_input(0, &Tensor::from_slice(&[0i64])).select(0, 0)).unwrap(),
            want_t
        );
    }
}
