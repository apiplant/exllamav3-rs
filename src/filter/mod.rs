//! Constrained-decoding filters — hand-rolled port of `generator/filter/*.py`.
//!
//! **No `kbnf` / `llguidance` / `formatron` dependency is vendored.** A small
//! byte-level grammar engine ([`gbnf`]) drives a push-down automaton over a
//! GBNF-subset grammar; [`json_schema`] compiles a JSON-Schema subset down to the
//! same grammar IR. Each generation step the active filters intersect their
//! allowed-token sets into one additive logit mask.
//!
//! Ported: the [`Filter`] state machine + journal/rewind (`filter.py`),
//! trigger-token activation, `eos_after_completed`. Not ported: the background
//! worker pool (`use_background_worker`), banned-strings, formatron's
//! `get_original_characters` vocab remap.

pub mod gbnf;
pub mod json_schema;

use crate::tokenizer::Tok;
use std::sync::Arc;
use tch::{Kind, Tensor};

/// Dense allowed-token set over the (unpadded) vocab, one bit per id.
#[derive(Clone)]
pub struct TokenMask {
    words: Vec<u64>,
    vocab: usize,
}

impl TokenMask {
    pub fn all_denied(vocab: usize) -> Self {
        Self { words: vec![0u64; vocab.div_ceil(64)], vocab }
    }
    pub fn all_allowed(vocab: usize) -> Self {
        let mut m = Self { words: vec![u64::MAX; vocab.div_ceil(64)], vocab };
        // clear the tail bits past `vocab`
        let tail = vocab % 64;
        if tail != 0 {
            *m.words.last_mut().unwrap() = (1u64 << tail) - 1;
        }
        m
    }
    #[inline]
    pub fn allow(&mut self, id: usize) {
        if id < self.vocab {
            self.words[id >> 6] |= 1u64 << (id & 63);
        }
    }
    #[inline]
    pub fn is_allowed(&self, id: usize) -> bool {
        id < self.vocab && (self.words[id >> 6] >> (id & 63)) & 1 == 1
    }
    /// In-place intersection (`self &= other`).
    pub fn intersect(&mut self, other: &TokenMask) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
    }
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
    /// Additive `[vocab]` f32 mask: `0` on allowed ids, `-inf` elsewhere, on `dev`.
    pub fn to_additive(&self, dev: tch::Device) -> Tensor {
        let mut v = vec![f32::NEG_INFINITY; self.vocab];
        for (wi, &w) in self.words.iter().enumerate() {
            if w == 0 {
                continue;
            }
            let base = wi * 64;
            for bit in 0..64 {
                if (w >> bit) & 1 == 1 && base + bit < self.vocab {
                    v[base + bit] = 0.0;
                }
            }
        }
        Tensor::from_slice(&v).to_device(dev)
    }
}

/// Byte-prefix trie over the token vocab. Walking it against a grammar's
/// byte-acceptor yields the allowed-token mask in `O(live nodes)` rather than
/// `O(vocab)` per step.
pub struct VocabTrie {
    /// `nodes[i]` = (children: sorted Vec<(byte, node_idx)>, token_id ending here)
    children: Vec<Vec<(u8, u32)>>,
    terminal: Vec<i32>, // token id that ends exactly at this node, or -1
    pieces: Vec<String>,
    vocab: usize,
}

impl VocabTrie {
    /// `logit_vocab` is the model's (padded) output width — masks are built at
    /// that width so the additive `-inf` tensor lines up with the logits row.
    /// Ids past the tokenizer's piece table (special / pad slots) stay denied.
    pub fn build(tok: &Tok, logit_vocab: usize) -> Self {
        let table = tok.piece_table().to_vec();
        let vocab = logit_vocab.max(table.len());
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut terminal: Vec<i32> = vec![-1];
        for (id, piece) in table.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            let mut node = 0u32;
            for &b in piece.as_bytes() {
                let next = match children[node as usize].binary_search_by_key(&b, |&(bb, _)| bb) {
                    Ok(pos) => children[node as usize][pos].1,
                    Err(pos) => {
                        let idx = children.len() as u32;
                        children.push(Vec::new());
                        terminal.push(-1);
                        children[node as usize].insert(pos, (b, idx));
                        idx
                    }
                };
                node = next;
            }
            terminal[node as usize] = id as i32;
        }
        Self { children, terminal, pieces: table, vocab }
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn piece(&self, id: i64) -> &str {
        self.pieces.get(id as usize).map(String::as_str).unwrap_or("")
    }

    /// Depth-first walk: for every vocab piece whose bytes are all accepted by
    /// `acc` (a fresh clone of the grammar byte-acceptor per branch), set its bit.
    pub fn collect_allowed<A: ByteAcceptor>(&self, acc: &A, out: &mut TokenMask) {
        self.walk(0, acc, out);
    }

    fn walk<A: ByteAcceptor>(&self, node: u32, acc: &A, out: &mut TokenMask) {
        // Reaching this node means every byte of some piece was accepted by the
        // grammar (a rejected byte returns `None` from `step` and prunes the
        // branch), so the piece is a valid continuation — allow it. `can_end`
        // gates only the EOS token, which the filter handles separately.
        let t = self.terminal[node as usize];
        if t >= 0 {
            out.allow(t as usize);
        }
        for &(b, child) in &self.children[node as usize] {
            if let Some(next) = acc.step(b) {
                self.walk(child, &next, out);
            }
        }
    }
}

/// A cloneable byte-level acceptor: `step` consumes one byte (returns the
/// resulting state or `None` if rejected), `can_end` reports whether the current
/// state is a valid stopping point.
pub trait ByteAcceptor: Clone {
    fn step(&self, byte: u8) -> Option<Self>;
    fn can_end(&self) -> bool;
}

// --- Filter journal / state machine (port of filter.py) -------------------

#[derive(Clone, Copy, PartialEq)]
enum Fj {
    Pass,
    Trigger,
    Accept,
    Complete,
}

/// One constrained-decoding filter attached to a job. Wraps a grammar matcher
/// with trigger-token activation and a rewindable journal.
pub struct FilterState {
    matcher: gbnf::Matcher,
    trie: Arc<VocabTrie>,
    trigger_token: Option<i64>,
    eos_after_completed: bool,
    active: bool,
    journal: Vec<(Fj, i64)>,
    /// masks + additive `-inf` tensors keyed by matcher signature. Bounded LRU —
    /// grammar states recur heavily (e.g. every char inside a string value has
    /// the same signature), so this turns the O(vocab) trie walk into a one-time
    /// cost per distinct state.
    mask_cache: std::collections::HashMap<u64, TokenMask>,
    add_cache: std::collections::HashMap<u64, tch::Tensor>,
    cache_order: std::collections::VecDeque<u64>,
}

impl FilterState {
    pub fn new(
        matcher: gbnf::Matcher,
        trie: Arc<VocabTrie>,
        trigger_token: Option<i64>,
        eos_after_completed: bool,
    ) -> Self {
        Self {
            active: trigger_token.is_none(),
            matcher,
            trie,
            trigger_token,
            eos_after_completed,
            journal: Vec::new(),
            mask_cache: Default::default(),
            add_cache: Default::default(),
            cache_order: Default::default(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.active && !self.matcher.is_complete()
    }

    const CACHE_CAP: usize = 128;

    fn touch(&mut self, sig: u64) {
        self.cache_order.push_back(sig);
        while self.cache_order.len() > Self::CACHE_CAP {
            if let Some(old) = self.cache_order.pop_front() {
                if !self.cache_order.contains(&old) {
                    self.mask_cache.remove(&old);
                    self.add_cache.remove(&old);
                }
            }
        }
    }

    /// Allowed-token mask for the current state (all-allowed when inactive).
    pub fn mask(&mut self) -> TokenMask {
        if !self.is_active() {
            return TokenMask::all_allowed(self.trie.vocab());
        }
        let sig = self.matcher.signature();
        if let Some(m) = self.mask_cache.get(&sig) {
            return m.clone();
        }
        let mut m = TokenMask::all_denied(self.trie.vocab());
        self.trie.collect_allowed(&self.matcher.acceptor(), &mut m);
        self.mask_cache.insert(sig, m.clone());
        self.touch(sig);
        m
    }

    /// Additive `-inf` logit tensor for the current state at `width`, on `dev`.
    /// A cheap `shallow_clone` of a cached tensor once the state has been seen.
    pub fn additive(&mut self, dev: tch::Device, width: i64) -> Option<tch::Tensor> {
        if !self.is_active() {
            return None;
        }
        let sig = self.matcher.signature();
        if let Some(t) = self.add_cache.get(&sig) {
            return Some(t.shallow_clone());
        }
        let m = self.mask();
        let mut add = m.to_additive(dev).to_kind(tch::Kind::Float);
        let aw = add.size()[0];
        if aw > width {
            add = add.narrow(0, 0, width);
        } else if aw < width {
            let pad =
                tch::Tensor::full([width - aw], f64::NEG_INFINITY, (tch::Kind::Float, dev));
            add = tch::Tensor::cat(&[add, pad], 0);
        }
        self.add_cache.insert(sig, add.shallow_clone());
        self.touch(sig);
        Some(add)
    }

    /// Advance on an emitted token. Returns `true` if the filter completed and
    /// `eos_after_completed` is set (the job should stop).
    pub fn feed(&mut self, tok: i64) -> bool {
        if !self.active {
            if Some(tok) == self.trigger_token {
                self.active = true;
                self.matcher.reset();
                self.journal.push((Fj::Trigger, tok));
            } else {
                self.journal.push((Fj::Pass, tok));
            }
            return false;
        }
        let piece = self.trie.piece(tok).to_string();
        self.matcher.accept_bytes(piece.as_bytes());
        if self.matcher.is_complete() {
            self.active = false;
            self.journal.push((Fj::Complete, tok));
            return self.eos_after_completed;
        }
        self.journal.push((Fj::Accept, tok));
        false
    }

    /// Roll the filter back over the last `n` fed tokens (banned-string / loop
    /// rewind). Rebuilds by replay when a native rollback isn't available.
    pub fn rewind(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let n = n.min(self.journal.len());
        let popped: Vec<(Fj, i64)> = self.journal.split_off(self.journal.len() - n);
        if popped.iter().any(|&(e, _)| e == Fj::Trigger) {
            self.rebuild();
            return;
        }
        if popped.iter().any(|&(e, _)| e == Fj::Complete) {
            self.active = true;
        }
        self.rebuild();
    }

    fn rebuild(&mut self) {
        self.active = self.trigger_token.is_none();
        self.matcher.reset();
        let journal = std::mem::take(&mut self.journal);
        for &(e, tok) in &journal {
            match e {
                Fj::Trigger => {
                    self.active = true;
                    self.matcher.reset();
                }
                Fj::Accept | Fj::Complete => {
                    let p = self.trie.piece(tok).to_string();
                    self.matcher.accept_bytes(p.as_bytes());
                    if e == Fj::Complete {
                        self.active = false;
                    }
                }
                Fj::Pass => {}
            }
        }
        self.journal = journal;
    }
}

/// Apply a set of filters' combined mask to a `[vocab]` f32 logit row in place.
pub fn apply_filters(filters: &mut [FilterState], logits: &mut Tensor) {
    let active: Vec<usize> = (0..filters.len()).filter(|&i| filters[i].is_active()).collect();
    let w = logits.size().last().copied().unwrap_or(0);
    let dev = logits.device();
    let add = match active.as_slice() {
        [] => return,
        // single filter (the common case) — reuse its per-state cached tensor
        [i] => match filters[*i].additive(dev, w) {
            Some(a) => a,
            None => return,
        },
        many => {
            let vocab = filters[0].trie.vocab();
            let mut combined = TokenMask::all_allowed(vocab);
            for &i in many {
                combined.intersect(&filters[i].mask());
            }
            let mut a = combined.to_additive(dev).to_kind(Kind::Float);
            let aw = a.size()[0];
            if aw > w {
                a = a.narrow(0, 0, w);
            } else if aw < w {
                a = Tensor::cat(&[a, Tensor::full([w - aw], f64::NEG_INFINITY, (Kind::Float, dev))], 0);
            }
            a
        }
    };
    let _ = logits
        .f_add_(&add)
        .expect("filter mask add failed (logit/mask width mismatch)");
}
