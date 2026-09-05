//! Dynamic batching generator — behavioural port of `generator/generator.py`
//! (`Generator`) + `generator/job.py` (`Job`) (grade —, equivalence only).
//!
//! ## What this does
//! Continuous batching: many jobs share one paged KV cache. Each `iterate()`
//! runs one prefill chunk for jobs still ingesting their prompt, then one batched
//! decode step for every job that has finished prefill, samples per-job, checks
//! stop conditions, and streams decoded text. New jobs join the batch as others
//! finish and free their pages.
//!
//! ## What upstream has and this does not (see PLAN.md)
//! Ported: n-gram speculative decode (`enable_ngram`, `src/sam.rs`), token
//! healing (`JobSpec::token_healing`), streaming loop detection
//! (`JobSpec::stop_on_loop`, `src/loop_detect.rs`). Still missing: prefix-cache
//! dedup / partial-page K/V reuse (`pagetable.py`), draft-model / MTP / DFlash
//! speculative decode, dynamic draft length + confidence calibration, recurrent
//! state & checkpoints, MRoPE offsets, CFG, grammar filters (formatron /
//! llguidance), the async wrapper, CPU cache offload, the fair-scheduling
//! requeue mechanism, per-token top-k probability reporting, and the exact
//! Gumbel-noise sampling RNG.

use crate::loop_detect::LoopDetector;
use crate::model::Model;
use crate::config::ArchKind;
use crate::draft::DraftModel;
use crate::filter::{FilterState, VocabTrie};
use crate::mtp::{MtpBatched, MtpBatchedCache, MtpModel};
use crate::cpu_cache::CpuPageCache;
use crate::paged::{
    chain_hash, pages_for, GdnCheckpoint, PageHash, PageTable, PagedCache, QuantPagedCache,
    Qwen35PagedCache, PAGE_SIZE,
};
use crate::sam::BcSam;
use crate::sampler::SamplerSettings;
use crate::tokenizer::Tok;
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;

/// `EXL3_DRAFT_COST_RATIO`: per-draft-step cost as a fraction of the verify's,
/// the only parameter the draft-window model needs (see `draft_window`).
static DRAFT_COST_RATIO: OnceLock<Option<f32>> = OnceLock::new();
use tch::{Device, Kind, Tensor};

/// Outcome of [`Generator::mm_generate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmFinish {
    /// hit an EOS / stop token or a stop string
    Stop,
    /// hit `max_new`
    Length,
    /// the request's cancel flag was set
    Cancelled,
}

/// Per-request configuration handed to `Generator::enqueue`.
pub struct JobSpec {
    pub prompt: Vec<i64>,
    pub sampler: SamplerSettings,
    pub max_new: usize,
    pub min_new: usize,
    pub stop_tokens: HashSet<i64>,
    pub stop_strings: Vec<String>,
    /// regenerate the last prompt token constrained to tokens sharing its text
    /// prefix (`job.py` `token_healing`). No-op if the prompt has ≤ 1 token.
    pub token_healing: bool,
    /// `(window_size, min_reps)` — stop the job once the last `window_size`
    /// sampled tokens are entirely a sequence repeating ≥ `min_reps` times
    /// (`job.py` `stop_on_loop`). `None` disables loop detection.
    pub stop_on_loop: Option<(i64, usize)>,
    /// constrained-decoding filters (`Generator::compile_grammar` /
    /// `compile_json_schema`). Each step the active filters intersect their
    /// allowed-token sets into an additive logit mask; speculation is suppressed
    /// while any filter is active.
    pub filters: Vec<FilterState>,
    /// classifier-free guidance: `(negative_prompt_ids, scale)`. Both the
    /// positive and negative contexts are run each step and the logits mixed
    /// (`uncond + scale*(cond-uncond)`) before sampling. Non-hybrid targets only;
    /// mutually exclusive with speculative decoding.
    pub cfg: Option<(Vec<i64>, f64)>,
}

impl JobSpec {
    pub fn new(prompt: Vec<i64>, max_new: usize) -> Self {
        Self {
            prompt,
            sampler: SamplerSettings::greedy(),
            max_new,
            min_new: 0,
            stop_tokens: HashSet::new(),
            stop_strings: Vec::new(),
            token_healing: false,
            stop_on_loop: None,
            filters: Vec::new(),
            cfg: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    Started,
    Prefill { done: i64, total: i64 },
    Streaming,
}

/// One `iterate()` result row, mirroring the `generator.py` `iterate()` dicts.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    pub serial: u64,
    pub stage: Stage,
    /// newly emittable text since the last event (Streaming only)
    pub text: String,
    pub token: Option<i64>,
    pub eos: bool,
    pub eos_reason: Option<String>,
    /// set on the final (eos) event
    pub full_text: Option<String>,
    pub new_tokens: usize,
}

struct Job {
    serial: u64,
    seq: Vec<i64>,
    /// current KV/prefill prompt boundary — grows on each fair-scheduling requeue
    /// as the generated text-so-far is folded into the next prompt.
    prompt_len: i64,
    /// original prompt boundary, fixed at enqueue. Streaming / detok always slice
    /// `seq[stream_prompt_len..]` so requeue is invisible to the caller.
    stream_prompt_len: i64,
    /// fair-scheduling: bounded per-physical-job token budget hit → reap this job
    /// and re-enqueue it with `seq` as the new prompt (`generator.py` requeue).
    requeue_pending: bool,
    kv_pos: i64,
    block: Vec<i32>,
    /// recurrent-state slot for the Qwen3.5 GDN layers (-1 = homogeneous cache)
    slot: i64,
    sampler: SamplerSettings,
    max_new: usize,
    min_new: usize,
    new_tokens: usize,
    stop_tokens: HashSet<i64>,
    stop_strings: Vec<String>,
    started_emitted: bool,
    emitted_text: String,
    /// incremental detokenisation of `seq[prompt_len..]` — `gen_text` is the
    /// committed decoded string, `detok_prefix`/`detok_read` are the rolling
    /// (gen-relative) token offsets of the vLLM-style two-window decode. Keeps
    /// per-step detok work bounded instead of re-decoding the whole output.
    gen_text: String,
    detok_prefix: usize,
    detok_read: usize,
    done: bool,
    eos_reason: Option<String>,
    /// suffix automaton for n-gram drafting (present iff n-gram decode is on)
    sam: Option<BcSam>,
    /// draft-model KV pages for this job (empty unless `Speculator::Draft`)
    draft_block: Vec<i32>,
    /// MTP: trunk post-final-norm hidden at position `kv_pos - 1` (`q-1`), the
    /// input the MTP head consumes to draft the token at `kv_pos`.
    mtp_h_prev: Option<Tensor>,
    /// MTP: the trunk hidden `[1, 1, h]` for the token just before the next
    /// prefill chunk. The MTP head is primed **incrementally, one prefill chunk
    /// at a time** (`mtp_prime_chunk`) rather than by concatenating every chunk's
    /// hiddens and doing one `prompt_len`-wide MTP forward at the end — that
    /// peaked at ~4 GB on a 19k prompt and grows linearly with context. Only this
    /// one-token carry crosses a chunk boundary, so the MTP prime's working set
    /// is O(chunk_size). `None` = start of the prime range (zero hidden, matching
    /// upstream's `carry_hidden` fallback).
    mtp_carry_h: Option<Tensor>,
    /// speculator per-job state (MTP KV prime / draft KV prime) is done
    spec_primed: bool,
    /// prefix-cache: token position the trunk KV resumed from (0 = fresh). The
    /// MTP head only needs priming over `[spec_prime_from, prompt_len)` — its KV
    /// for the shared prefix is already resident in the reused pages.
    spec_prime_from: i64,
    /// prefix-cache: number of this job's complete pages already content-hashed
    /// and registered for sharing (`PageTable::register_page_hash`).
    hashed_pages: usize,
    /// CFG (classifier-free guidance): the negative-prompt context. `neg_seq`
    /// grows with the same sampled tokens as `seq`; `neg_block`/`neg_kv_pos` are
    /// its own KV pages in the shared pool. `cfg_scale > 1` sharpens toward the
    /// positive prompt (`logits = uncond + scale*(cond - uncond)`).
    neg_seq: Option<Vec<i64>>,
    neg_block: Vec<i32>,
    neg_kv_pos: i64,
    cfg_scale: f64,
    cfg_primed: bool,
    #[allow(dead_code)]
    accepted_draft_tokens: u64,
    #[allow(dead_code)]
    rejected_draft_tokens: u64,
    /// token healing: the dropped last prompt token, its surface piece, and an
    /// additive `[vocab]` mask (0 on allowed tokens, -inf elsewhere) applied to
    /// the first decode step only. `heal_offset` is 1 while a healed token is in
    /// the sequence so it doesn't count against min/max-new.
    #[allow(dead_code)] // kept for parity / future requeue; healing uses the fields below
    prefix_token: Option<i64>,
    unhealed_piece: String,
    heal_mask: Option<Tensor>,
    heal_pending: bool,
    heal_offset: usize,
    /// streaming loop detector (present iff `stop_on_loop` was set)
    loop_detector: Option<LoopDetector>,
    /// constrained-decoding filters (empty unless a grammar was attached)
    filters: Vec<FilterState>,
}

impl Job {
    fn prefill_done(&self) -> bool {
        self.kv_pos >= self.prompt_len - 1
    }
    fn pages_held(&self) -> usize {
        self.block.len()
    }
    /// physical page count this job will ever need (remaining new tokens only —
    /// after a requeue the prompt already contains the generation so far).
    fn gen_reserve(&self) -> i64 {
        (self.max_new as i64 - self.new_tokens as i64).max(0) + 1
    }
    fn pages_needed(&self) -> i64 {
        pages_for(self.prompt_len + self.gen_reserve())
    }

    /// tokens generated for the caller's accounting (the healed token doesn't count)
    fn gen_count(&self) -> usize {
        self.new_tokens.saturating_sub(self.heal_offset)
    }

    /// Advance the incremental detokeniser over any newly appended generated
    /// tokens, growing `gen_text`. Two-window trick (à la vLLM): decode the
    /// pending suffix both with and without the last committed window, and
    /// commit the byte delta only when it isn't a partial trailing char. Each
    /// call decodes at most a handful of tokens.
    fn extend_detok(&mut self, tok: &Tok) {
        let gen = &self.seq[self.stream_prompt_len as usize..];
        let g = gen.len();
        if self.detok_read >= g {
            return;
        }
        let prefix_text = tok.decode(&gen[self.detok_prefix..self.detok_read]).unwrap_or_default();
        let new_text = tok.decode(&gen[self.detok_prefix..g]).unwrap_or_default();
        if let Some(delta) = new_text.strip_prefix(prefix_text.as_str()) {
            if !new_text.ends_with('\u{FFFD}') && !delta.is_empty() {
                self.gen_text.push_str(delta);
                self.detok_prefix = self.detok_read;
                self.detok_read = g;
            }
        }
    }

    /// Speculative draft tokens from the suffix-array n-gram matcher — 1:1 with
    /// `job.py` `get_ngram_draft`. Returns the tokens following the earliest
    /// occurrence of the longest current suffix, capped at `draft_length`.
    fn get_ngram_draft(&mut self, draft_length: i64, match_min: usize) -> Vec<i64> {
        if self.heal_pending {
            return Vec::new(); // the first (healed) step is logit-masked; don't speculate
        }
        let seq = self.seq.clone();
        let sam = self.sam.as_mut().expect("sam");
        let (beg, end) = sam.accept_tensor(&seq);
        if end - beg >= match_min as i64 {
            let n = self.seq.len() as i64;
            let a = end.clamp(0, n) as usize;
            let b = (end + draft_length).clamp(0, n) as usize;
            self.seq[a..b].to_vec()
        } else {
            Vec::new()
        }
    }
}

/// The generator's shared KV cache — fp16 pool or a quantized pool.
enum GenCache {
    Plain(PagedCache),
    Quant(QuantPagedCache),
    Qwen35(Qwen35PagedCache),
}

impl GenCache {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        model: &Model,
        ids: &Tensor,
        block_table: &Tensor,
        seqlens: &Tensor,
        slots: &Tensor,
        last_only: bool,
        gdn_history: bool,
    ) -> Tensor {
        match self {
            GenCache::Plain(c) => {
                model.forward_paged_batched(ids, &c.k, &c.v, block_table, seqlens, last_only)
            }
            GenCache::Quant(c) => {
                model.forward_paged_batched_quant(ids, c, block_table, seqlens, last_only)
            }
            GenCache::Qwen35(c) => {
                model.forward_qwen35_batched(ids, c, block_table, seqlens, slots, last_only, gdn_history)
            }
        }
    }

    fn is_qwen35(&self) -> bool {
        matches!(self, GenCache::Qwen35(_))
    }
}

/// Model-based speculator for the batched decode loop. `None` / `Ngram` keep the
/// existing paths (n-gram is gated by `ngram_match_min`); `Draft` / `Mtp` run a
/// draft model / the Qwen3.5 MTP head and share `spec_verify_round`.
enum Speculator {
    None,
    Draft(DraftModel),
    Mtp(MtpBatched),
    /// DFlash2 block drafter: one forward per round proposes a whole block per
    /// row, and one shared `lm_head` pass covers the entire batch.
    DFlash2(crate::dflash2::DFlash2Batched),
}

pub struct Generator {
    model: Model,
    tok: Tok,
    cache: GenCache,
    pages: PageTable,
    device: Device,
    max_chunk: i64,
    max_batch: usize,
    /// minimum suffix match length for n-gram drafting (0 = disabled)
    ngram_match_min: usize,
    /// number of future tokens to draft per round
    num_draft_tokens: i64,
    /// how many slots have a DFlash2 window cache (rows above this don't draft)
    dflash2_slots: usize,
    /// fair-scheduling requeue: once a physical job has generated more than this
    /// many tokens it is reaped and re-enqueued with its output-so-far as the new
    /// prompt, bounding per-job cache growth (0 = disabled). `generator.py`
    /// `max_rq_tokens`.
    rq_budget: usize,
    /// KV-cache quant bit widths for a Qwen3.5 hybrid cache (`(0, 0)` = fp16).
    /// Preserved across the `new_hist` rebuilds done by the speculative paths.
    q35_kv_bits: (i64, i64),
    /// prefix-cache GDN checkpoints (Qwen3.5 only): SSM + conv state at a shared
    /// page boundary, keyed by that page's chained content hash. Small LRU —
    /// lets a follow-up turn reuse the KV pages AND resume the GDN recurrence
    /// instead of re-prefilling the whole conversation.
    /// Online drafter-confidence calibration (upstream's `draft_confidence`).
    /// **Off by default**: it cuts the block at the first position unlikely to be
    /// accepted, but reading the per-position confidence needs a device->host
    /// copy on every draft step, and that sync costs more than the `lm_head`
    /// pass it saves here (measured 76.5 vs 82.0 tok/s at 16k). Upstream pays a
    /// host round-trip per step regardless, so the trade differs there. Set
    /// `EXL3_DRAFT_CONFIDENCE=0.4` to enable. The sync-free adaptation of the
    /// same idea — sizing the whole window from observed acceptance — is
    /// same idea — sizing the window from observed acceptance — was also tried
    /// and was ALSO a loss: the verify forward re-reads all 8.5 GB of weights
    /// whatever `q_len` is, so a shorter window just means more verify rounds.
    /// The window wants to be as LONG as acceptance repays the extra `lm_head`
    /// per draft step, which for this model measured at 3 (2: 66-68 tok/s,
    /// 3: 75-82, 4: 64-75, 7: ~25% worse). Hence a fixed `draft_num_tokens`.
    draft_cal: Option<crate::draft_conf::DraftConfidence>,

    /// `(rows, drafts, per-position confidences)` from the round's draft, held
    /// until the verify reports how many were accepted so the calibrator can be
    /// labelled.
    pending_conf: Option<(Vec<usize>, Vec<Vec<i64>>, Vec<Vec<f32>>)>,
    /// Online estimate of the drafter's PER-TOKEN acceptance probability, used to
    /// size the next round's draft window (see `draft_window`). Note this is NOT
    /// the "acceptance %" a metrics line reports — that is `E[accepted]/n`, which
    /// is lower. Estimated by maximum likelihood from the verify's own outcome,
    /// which is already on the host: a round that drafts `n` and accepts `a`
    /// contributes `a` successes and, if `a < n`, one failure.
    accept_p: f32,
    /// rounds observed, for the warm-up before the estimate is trusted
    accept_obs: u32,
    /// Measured cost of one draft step relative to one verify (`d/V`). This must
    /// be measured, not assumed: the draft step is a fixed cost while the verify
    /// grows with context, so `r` spans ~0.25 at short context down to ~0.076 at
    /// 10k — which is precisely why the best window is 3 on short prompts and 4
    /// on long ones.
    cost_r: f32,
    gdn_ckpts: HashMap<PageHash, GdnCheckpoint>,
    gdn_ckpt_lru: VecDeque<PageHash>,
    /// pinned host-RAM second tier for evicted hashed KV pages (`enable_cpu_cache`)
    cpu_cache: Option<CpuPageCache>,
    /// model-based speculator (draft model / MTP head)
    spec: Speculator,
    /// aggregate speculative-decode acceptance stats
    ngram_accepted: u64,
    ngram_drafted: u64,
    next_serial: u64,
    queue: VecDeque<Job>,
    active: Vec<Job>,
    /// recurrent-state slot free-list for the Qwen3.5 GDN layers (`0..max_batch`);
    /// empty / unused for the homogeneous cache.
    slots_free: Vec<i64>,
    /// byte-prefix trie over the vocab for constrained decoding, built on first
    /// `compile_grammar` / `compile_json_schema` and shared by every filter.
    vocab_trie: Option<std::sync::Arc<VocabTrie>>,
    /// Persistent device-side scratch for the batched decode step, sized once at
    /// `[max_batch, …]`. Each step writes the live `[bsz, …]` sub-view via a
    /// single H2D `copy_` instead of allocating a fresh `to_device` tensor —
    /// removes the per-token cudaMalloc churn from the varying batch/page counts.
    dec_ids: Tensor,     // [max_batch] i64  — the token fed to each row
    dec_block: Tensor,   // [max_batch, num_pages] i32 — per-row block table (full pool width)
    dec_seqlens: Tensor, // [max_batch] i32  — per-row pre-append length / RoPE offset
    dec_slots: Tensor,   // [max_batch] i32  — per-row GDN recurrent slot
}

impl Generator {
    /// `num_pages` sizes the shared cache; `max_batch` caps concurrent decode
    /// rows; `max_chunk` (rounded up to a page) bounds prefill work per step.
    /// `kv_bits` `(k, v)` quantizes a Qwen3.5 cache from the start (`(0, 0)` =
    /// fp16) — pass the real bits here rather than allocating fp16 and having
    /// `enable_cache_quant` throw it away (a 200k fp16 hybrid pool is ~13 GB).
    pub fn new(
        model: Model,
        tok: Tok,
        num_pages: i64,
        max_batch: usize,
        max_chunk: i64,
        kv_bits: (i64, i64),
    ) -> Self {
        let device = model.device();
        let cache = if model.config.arch_kind == ArchKind::Qwen35 {
            GenCache::Qwen35(Qwen35PagedCache::new_hist(
                &model.config, num_pages, max_batch as i64, 0, kv_bits, device,
            ))
        } else {
            GenCache::Plain(PagedCache::new(&model.config, num_pages, device))
        };
        let max_chunk = (max_chunk.max(PAGE_SIZE) + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;
        Self {
            model,
            tok,
            cache,
            pages: PageTable::new(num_pages),
            device,
            max_chunk,
            max_batch,
            ngram_match_min: 0,
            num_draft_tokens: 0,
            dflash2_slots: 0,
            rq_budget: 0,
            q35_kv_bits: kv_bits,
            draft_cal: {
                let c = std::env::var("EXL3_DRAFT_CONFIDENCE")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0);
                if c > 0.0 && c < 1.0 { Some(crate::draft_conf::DraftConfidence::new(c)) } else { None }
            },
            pending_conf: None,
            accept_p: 0.8,
            accept_obs: 0,
            cost_r: 0.1,
            gdn_ckpts: HashMap::new(),
            cpu_cache: None,
            gdn_ckpt_lru: VecDeque::new(),
            spec: Speculator::None,
            ngram_accepted: 0,
            ngram_drafted: 0,
            next_serial: 0,
            queue: VecDeque::new(),
            active: Vec::new(),
            slots_free: (0..max_batch as i64).rev().collect(),
            vocab_trie: None,
            dec_ids: Tensor::zeros([max_batch as i64], (Kind::Int64, device)),
            dec_block: Tensor::zeros([max_batch as i64, num_pages], (Kind::Int, device)),
            dec_seqlens: Tensor::zeros([max_batch as i64], (Kind::Int, device)),
            dec_slots: Tensor::zeros([max_batch as i64], (Kind::Int, device)),
        }
    }

    /// Enable n-gram speculative decoding (`generator.py` `ngram_match_min` /
    /// `num_draft_tokens`). `match_min` is the shortest suffix that may seed a
    /// draft; `draft_tokens` (default 4 upstream) is the per-round draft length.
    /// Must be called before any job starts.
    ///
    /// For a Qwen3.5 cache this rebuilds the GDN state pools with
    /// `max_history = draft_tokens` per slot so a rejected speculative forward can
    /// be rewound. Extra VRAM ≈ `draft_tokens * max_batch * Σ_gdn_layers
    /// (n_v_heads * k_head_dim * v_head_dim * 4)` — ~3 MB × draft × max_batch ×
    /// (#linear layers) for the 27B, so keep `max_batch` modest on a 24 GB card.
    pub fn enable_ngram(&mut self, match_min: usize, draft_tokens: i64) {
        assert!(self.active.is_empty() && self.queue.is_empty(), "enable_ngram before enqueue");
        self.ngram_match_min = match_min;
        self.num_draft_tokens = if draft_tokens > 0 { draft_tokens } else { 4 };
        // A speculative forward consumes up to `num_draft_tokens + 1` tokens (the
        // committed token + the draft); the history buffer must hold the whole
        // stream so any accepted prefix is recoverable.
        self.rebuild_q35_cache(self.num_draft_tokens + 1);
    }

    /// `(accepted_draft_tokens, total_drafted_tokens)` across the run so far.
    /// Shared by the n-gram, draft-model and MTP speculative paths.
    pub fn ngram_stats(&self) -> (u64, u64) {
        (self.ngram_accepted, self.ngram_drafted)
    }
    /// alias — the counters are shared across all speculative-decode strategies.
    pub fn spec_stats(&self) -> (u64, u64) {
        (self.ngram_accepted, self.ngram_drafted)
    }

    /// Enable prefix-cache dedup (`generator/pagetable.py`): complete leading
    /// prompt pages whose content matches an earlier job's are shared instead of
    /// re-prefilled. Full-page granularity only. For a Qwen3.5 hybrid target the
    /// KV pages are shared as usual AND the GDN recurrent/conv state is restored
    /// from a checkpoint taken at the boundary page (`gdn_ckpts`), so a follow-up
    /// turn only prefills its new tail. Must be called before any job is enqueued.
    pub fn enable_prefix_cache(&mut self) {
        assert!(
            self.active.is_empty() && self.queue.is_empty(),
            "enable_prefix_cache before enqueue"
        );
        self.pages.dedup = true;
    }

    /// Cap on live GDN prefix-cache checkpoints. Held in host RAM
    /// (`gdn_snapshot_cpu`, ~100 MB each for the 27B), restored H2D on a hit —
    /// far cheaper than re-prefilling the shared prefix, and a bigger LRU keeps
    /// a long multi-turn conversation resuming instead of falling back to a full
    /// re-prefill (which also spikes VRAM). `EXL3_GDN_CKPTS` overrides.
    fn gdn_ckpt_cap(&self) -> usize {
        std::env::var("EXL3_GDN_CKPTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(24)
    }

    /// `(cached_prompt_tokens, total_prompt_tokens)` served from shared prefix
    /// pages vs. ingested from scratch, across the run so far.
    pub fn prefix_stats(&self) -> (u64, u64) {
        self.pages.prefix_stats()
    }

    /// Enable the pinned host-RAM CPU cache tier (`generator/cpu_cache.py`):
    /// hashed KV pages evicted from the GPU pool spill to `max_host_tokens` of
    /// pinned system memory and are restored on a later prefix hit instead of
    /// being re-prefilled. Implies `enable_prefix_cache`. Homogeneous fp16 cache
    /// only. Must be called before any job is enqueued.
    pub fn enable_cpu_cache(&mut self, max_host_tokens: i64) {
        assert!(
            self.active.is_empty() && self.queue.is_empty(),
            "enable_cpu_cache before enqueue"
        );
        assert!(
            matches!(self.cache, GenCache::Plain(_)),
            "CPU cache tier supports only the homogeneous fp16 KV cache"
        );
        self.pages.dedup = true;
        self.cpu_cache = Some(CpuPageCache::new(
            &self.model.config,
            max_host_tokens,
            self.device,
        ));
    }

    /// `(restored_pages, pushed_pages)` for the CPU cache tier, or `(0, 0)`.
    pub fn cpu_cache_stats(&self) -> (u64, u64) {
        self.cpu_cache.as_ref().map(|c| c.stats()).unwrap_or((0, 0))
    }

    // --- constrained decoding ------------------------------------------------

    fn trie(&mut self) -> std::sync::Arc<VocabTrie> {
        if self.vocab_trie.is_none() {
            let v = self.model.config.vocab_size as usize;
            self.vocab_trie = Some(std::sync::Arc::new(VocabTrie::build(&self.tok, v)));
        }
        self.vocab_trie.clone().unwrap()
    }

    /// Compile a GBNF-subset grammar into a [`FilterState`] to hand to `JobSpec`.
    /// Completing the grammar stops the job.
    pub fn compile_grammar(&mut self, gbnf: &str) -> Result<FilterState> {
        let g = std::sync::Arc::new(
            crate::filter::gbnf::parse(gbnf).map_err(|e| anyhow::anyhow!("grammar: {e}"))?,
        );
        let trie = self.trie();
        Ok(FilterState::new(crate::filter::gbnf::Matcher::new(g), trie, None, true))
    }

    /// Compile a JSON-Schema (subset) into a constrained-decoding filter.
    pub fn compile_json_schema(&mut self, schema: &serde_json::Value) -> Result<FilterState> {
        let g = std::sync::Arc::new(
            crate::filter::json_schema::compile(schema)
                .map_err(|e| anyhow::anyhow!("json schema: {e}"))?,
        );
        let trie = self.trie();
        Ok(FilterState::new(crate::filter::gbnf::Matcher::new(g), trie, None, true))
    }

    /// Constrain output to any well-formed JSON object (`response_format:
    /// {type: json_object}`).
    pub fn compile_json_object(&mut self) -> Result<FilterState> {
        let g = std::sync::Arc::new(
            crate::filter::json_schema::any_json().map_err(|e| anyhow::anyhow!("json: {e}"))?,
        );
        let trie = self.trie();
        Ok(FilterState::new(crate::filter::gbnf::Matcher::new(g), trie, None, true))
    }

    /// Enable fair-scheduling requeue with a per-job token budget (`generator.py`
    /// `max_rq_tokens`). A job past the budget is transparently reaped and
    /// re-enqueued with its generated text folded into the prompt, so its pages
    /// return to the pool and long generations keep passing through prefix-cache
    /// allocation. Best paired with `enable_prefix_cache`. `budget == 0` disables.
    pub fn enable_requeue(&mut self, budget: usize) {
        self.rq_budget = budget;
    }

    /// Content-hash and register any newly-complete pages of every active job so
    /// later jobs sharing the prefix can reuse them (`pagetable.py` page hashing).
    fn hash_complete_pages(&mut self) {
        if !self.pages.dedup {
            return;
        }
        let hybrid = self.cache.is_qwen35();
        for j in 0..self.active.len() {
            let complete = (self.active[j].kv_pos / PAGE_SIZE) as usize;
            while self.active[j].hashed_pages < complete
                && self.active[j].hashed_pages < self.active[j].block.len()
            {
                let pidx = self.active[j].hashed_pages;
                let phys = self.active[j].block[pidx];
                let prev = if pidx == 0 {
                    None
                } else {
                    self.pages.page_hash(self.active[j].block[pidx - 1])
                };
                let start = pidx * PAGE_SIZE as usize;
                let toks: Vec<i64> =
                    self.active[j].seq[start..start + PAGE_SIZE as usize].to_vec();
                self.pages.register_page_hash(phys, prev, &toks);
                self.active[j].hashed_pages += 1;
            }

            // Hybrid: when the slot sits exactly on a page boundary (after a
            // prefill chunk, or every 256th decoded token) its GDN state is
            // resumable — checkpoint it against the boundary page's hash so a
            // later turn sharing this prefix can restore it instead of
            // re-prefilling. Keyed by content hash, so it stays valid even after
            // the physical page is repurposed.
            let kp = self.active[j].kv_pos;
            if hybrid && kp >= PAGE_SIZE && kp % PAGE_SIZE == 0 {
                let pidx = (kp / PAGE_SIZE - 1) as usize;
                if pidx < self.active[j].block.len() {
                    if let Some(h) = self.pages.page_hash(self.active[j].block[pidx]) {
                        if !self.gdn_ckpts.contains_key(&h) {
                            let slot = self.active[j].slot;
                            let cp = match &self.cache {
                                GenCache::Qwen35(c) => c.gdn_snapshot_cpu(slot),
                                _ => unreachable!(),
                            };
                            self.gdn_ckpts.insert(h, cp);
                            self.gdn_ckpt_lru.push_back(h);
                            let cap = self.gdn_ckpt_cap();
                            while self.gdn_ckpt_lru.len() > cap {
                                if let Some(old) = self.gdn_ckpt_lru.pop_front() {
                                    self.gdn_ckpts.remove(&old);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rebuild the Qwen3.5 paged cache with `hist` per-token GDN history planes
    /// and the current `q35_kv_bits`. Frees the old pool BEFORE allocating the
    /// new one — otherwise both are briefly live and a large pool OOMs at load
    /// (`Generator::new` builds an fp16 pool that `enable_cache_quant` replaces).
    fn rebuild_q35_cache(&mut self, hist: i64) {
        if !self.cache.is_qwen35() {
            return;
        }
        let cfg = &self.model.config;
        // minimal placeholder — drops the current (possibly multi-GB) pool
        self.cache = GenCache::Qwen35(Qwen35PagedCache::new_hist(cfg, 1, 1, 0, (0, 0), self.device));
        self.cache = GenCache::Qwen35(Qwen35PagedCache::new_hist(
            cfg,
            self.pages.num_pages(),
            self.max_batch as i64,
            hist,
            self.q35_kv_bits,
            self.device,
        ));
    }

    /// For a Qwen3.5 target, rebuild the paged cache with `draft_tokens + 1`
    /// per-token history planes so a rejected speculative forward can be rewound
    /// (`gdn_rewind`). No-op for the homogeneous cache.
    fn enable_hist_cache(&mut self, draft_tokens: i64) {
        self.rebuild_q35_cache(draft_tokens + 1);
    }

    /// Enable MTP self-speculation inside the batched loop (Qwen3.5 only). Loads
    /// nothing — pass a headless `MtpModel` (`MtpModel::load_headless`). Must be
    /// called before any job is enqueued.
    pub fn enable_mtp(&mut self, mtp: MtpModel, draft_tokens: i64) {
        assert!(self.active.is_empty() && self.queue.is_empty(), "enable_mtp before enqueue");
        assert!(self.cache.is_qwen35(), "MTP speculation requires a Qwen3.5 target");
        self.num_draft_tokens = if draft_tokens > 0 { draft_tokens } else { 4 };
        self.enable_hist_cache(self.num_draft_tokens);
        let cache = MtpBatchedCache::new(&self.model.config, self.pages.num_pages(), self.device);
        self.spec = Speculator::Mtp(MtpBatched { model: mtp, cache });
    }

    /// Enable DFlash2 block speculation. The drafter's context K/V are projected
    /// from the target's tapped hidden states, so it must see every position the
    /// target ingests — `prefill_round` feeds it each chunk's taps, exactly as
    /// it primes the MTP head. Must be called before any job is enqueued.
    pub fn enable_dflash2(&mut self, d: crate::dflash2::DFlash2Model, draft_tokens: i64) {
        assert!(self.active.is_empty() && self.queue.is_empty(), "enable_dflash2 before enqueue");
        assert!(self.cache.is_qwen35(), "DFlash2 speculation requires a Qwen3.5 target");
        // A block proposes `block_size - 1` tokens, but `draft_num_tokens` may
        // verify fewer. That is worth honouring rather than always taking the
        // full block: the hybrid cache keeps `draft_tokens + 1` GDN history
        // planes across all 48 linear-attention layers, so a width-8 verify
        // costs ~60% more history VRAM than MTP's width-5 — several GB on a 24
        // GB card. Drafting is one forward either way, so capping the width
        // trades a little acceptance for a lot of KV pool.
        let full = d.params.block_size - 1;
        self.num_draft_tokens = if draft_tokens > 0 { draft_tokens.min(full) } else { full };
        self.enable_hist_cache(self.num_draft_tokens);
        // One window cache per slot is ~50 MiB, and `max_batch` is often "auto"
        // (large), so allocating one per slot quietly costs multiple GB of KV
        // pool. Cap the number of *drafting* slots: concurrency beyond this
        // still generates, just without speculation, which is a far better
        // trade than a pool too small to hold the context.
        let cap: usize = std::env::var("EXL3_DFLASH2_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let slots = self.max_batch.max(1).min(cap.max(1));
        self.dflash2_slots = slots;
        let b = crate::dflash2::DFlash2Batched::new(d, slots, self.device);
        crate::sinfo!(
            "dflash2: block {} ({} verified/round of {} drafted), {} drafting slot(s) \
             x {:.0} MiB window cache",
            b.model.params.block_size,
            self.num_draft_tokens,
            full,
            slots,
            b.bytes_per_slot() as f64 / (1024.0 * 1024.0),
        );
        self.spec = Speculator::DFlash2(b);
    }

    /// Enable separate-draft-model speculation inside the batched loop. The draft
    /// model's KV pool is sized to the target's page pool. Must be called before
    /// any job is enqueued.
    pub fn enable_draft(&mut self, draft: DraftModel, draft_tokens: i64) {
        assert!(self.active.is_empty() && self.queue.is_empty(), "enable_draft before enqueue");
        self.num_draft_tokens = if draft_tokens > 0 { draft_tokens } else { 4 };
        self.enable_hist_cache(self.num_draft_tokens);
        self.spec = Speculator::Draft(draft);
    }

    /// Quantize the shared KV cache to `k_bits`/`v_bits` (2..=8) — port of
    /// `cache/quant.py`. Must be called before any job is enqueued (it rebuilds
    /// the cache pool). Attention dequantizes the prefix into fp16 scratch each
    /// step, so this trades decode speed for ~`16/bits`× less cache memory.
    pub fn enable_cache_quant(&mut self, k_bits: i64, v_bits: i64) {
        assert!(self.active.is_empty() && self.queue.is_empty(), "enable_cache_quant before enqueue");
        let np = self.pages.num_pages();
        if self.cache.is_qwen35() {
            // Only the `full_attention` KV pages are quantized; the GDN recurrent
            // state stays fp32. Usually a no-op — `Generator::new` already built
            // the pool at these bits; rebuild only if the bits actually change
            // (order-independent w.r.t. enable_mtp, which preserves the history).
            if self.q35_kv_bits == (k_bits, v_bits) {
                return;
            }
            self.q35_kv_bits = (k_bits, v_bits);
            let hist = if self.ngram_match_min > 0 || !matches!(self.spec, Speculator::None) {
                self.num_draft_tokens + 1
            } else {
                0
            };
            self.rebuild_q35_cache(hist);
            return;
        }
        self.cache = GenCache::Quant(QuantPagedCache::new(
            &self.model.config, np, k_bits, v_bits, self.device,
        ));
    }

    /// Resize the shared KV page pool, freeing the old pool before allocating
    /// the new one.
    ///
    /// Meant to be called by the host **after** the vision tower, MTP head and
    /// any draft model are loaded: only then is the remaining VRAM actually
    /// known, so the pool can be sized to fill it instead of being guessed from
    /// a fixed constant. Must be called before any job is enqueued.
    pub fn resize_pool(&mut self, num_pages: i64) {
        assert!(self.active.is_empty() && self.queue.is_empty(), "resize_pool before enqueue");
        let num_pages = num_pages.max(1);
        if num_pages == self.pages.num_pages() {
            return;
        }
        let dedup = self.pages.dedup;
        self.pages = PageTable::new(num_pages);
        self.pages.dedup = dedup;

        let hist = if self.ngram_match_min > 0 || !matches!(self.spec, Speculator::None) {
            self.num_draft_tokens + 1
        } else {
            0
        };
        if self.cache.is_qwen35() {
            self.rebuild_q35_cache(hist);
        } else {
            let cfg = &self.model.config;
            let (kb, vb) = match &self.cache {
                GenCache::Quant(q) => (q.k_bits, q.v_bits),
                _ => (0, 0),
            };
            // drop the old pool first, then allocate at the new size
            self.cache = GenCache::Plain(PagedCache::new(cfg, 1, self.device));
            self.cache = if kb > 0 || vb > 0 {
                GenCache::Quant(QuantPagedCache::new(cfg, num_pages, kb, vb, self.device))
            } else {
                GenCache::Plain(PagedCache::new(cfg, num_pages, self.device))
            };
        }
        if let Speculator::Mtp(m) = &mut self.spec {
            m.cache = MtpBatchedCache::new(&self.model.config, 1, self.device);
            m.cache = MtpBatchedCache::new(&self.model.config, num_pages, self.device);
        }
        self.dec_block = Tensor::zeros(
            [self.max_batch as i64, num_pages],
            (Kind::Int, self.device),
        );
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
    /// Size of the shared page pool (for sizing a draft model's KV pool).
    pub fn num_pages(&self) -> i64 {
        self.pages.num_pages()
    }
    pub fn tokenizer(&self) -> &Tok {
        &self.tok
    }

    /// Run one multimodal generation to completion, single-stream, **reusing the
    /// shared paged KV pool** (no dedicated vision cache — matches upstream, where
    /// image jobs go through the normal generator).
    ///
    /// * `embeds` — `[1, prompt_len, h]` input hidden state with vision-tower
    ///   embeddings already spliced over the image-placeholder rows.
    /// * `seq_ids` — the `prompt_len` token ids (image span = placeholder pads);
    ///   used for sampler history only.
    /// * `rope_table` — `[max_len, rot/2]` MRoPE angle table covering the prompt
    ///   plus `max_new` decode positions.
    /// * `on_token(id) -> Ok(true)` stops generation (a stop string was hit).
    /// * `cancel` — checked each step; set → returns `MmFinish::Cancelled`.
    ///
    /// Blocks the engine for the duration (like the batched path's prefill would
    /// for one big job). Requires a free GDN slot and enough free pages.
    #[allow(clippy::too_many_arguments)]
    pub fn mm_generate(
        &mut self,
        embeds: &Tensor,
        seq_ids: &[i64],
        rope_table: &Tensor,
        sampler: &SamplerSettings,
        max_new: usize,
        stop_tokens: &HashSet<i64>,
        cancel: &std::sync::atomic::AtomicBool,
        mut on_token: impl FnMut(i64) -> Result<bool>,
    ) -> Result<(MmFinish, usize)> {
        use std::sync::atomic::Ordering;
        if !matches!(self.cache, GenCache::Qwen35(_)) {
            anyhow::bail!("mm_generate requires a Qwen3.5 hybrid cache");
        }
        let prompt_len = seq_ids.len() as i64;

        let need = pages_for(prompt_len + max_new as i64 + 1) as usize;
        if self.pages.num_free() < need {
            anyhow::bail!(
                "KV pool has {} free pages, image request needs {need} — lower max_tokens or wait for other requests",
                self.pages.num_free()
            );
        }
        let Some(slot) = self.slots_free.pop() else {
            anyhow::bail!("all GDN recurrent slots are busy — retry when other requests finish");
        };
        let block: Vec<i32> = self.pages.alloc(need)?;

        let dev = self.device;
        let block_t = Tensor::from_slice(&block).reshape([1, block.len() as i64]).to_device(dev);
        let slots_t = Tensor::from_slice(&[slot as i32]).to_device(dev);
        let rope = Some(rope_table);
        let model = &self.model;
        let max_chunk = self.max_chunk;
        let cache = match &self.cache {
            GenCache::Qwen35(c) => c,
            _ => unreachable!(),
        };

        let r = (|| -> Result<(MmFinish, usize)> {
            cache.reset_slot(slot);
            let fwd = |x: &Tensor, pos: i64| {
                let seqlens = Tensor::from_slice(&[pos as i32]).to_device(dev);
                model.forward_qwen35_batched_embed(
                    x, cache, &block_t, &seqlens, &slots_t, true, false, rope,
                )
            };

            // --- chunked prefill (leave the last prompt token for step 1) ---
            let mut pos: i64 = 0;
            let ingest = (prompt_len - 1).max(0);
            while pos < ingest {
                let end = (pos + max_chunk).min(ingest);
                let chunk = embeds.narrow(1, pos, end - pos).contiguous();
                let _ = fwd(&chunk, pos);
                pos = end;
            }

            // --- step 1: the remaining prompt tail (usually one token) ---
            let last = embeds.narrow(1, pos, prompt_len - pos).contiguous();
            let logits = fwd(&last, pos);
            pos = prompt_len;

            let mut hist: Vec<i64> = seq_ids.to_vec();
            let mut next = sampler.sample(&logits.select(0, 0).select(0, 0), &hist);

            let mut generated = 0usize;
            let finish = loop {
                if cancel.load(Ordering::Relaxed) {
                    break MmFinish::Cancelled;
                }
                if stop_tokens.contains(&next) {
                    break MmFinish::Stop;
                }
                let stop_str = on_token(next)?;
                hist.push(next);
                generated += 1;
                if stop_str {
                    break MmFinish::Stop;
                }
                if generated >= max_new {
                    break MmFinish::Length;
                }
                let x = model.embed_tokens(
                    &Tensor::from_slice(&[next]).reshape([1, 1]).to_device(dev),
                );
                let l = fwd(&x, pos);
                pos += 1;
                next = sampler.sample(&l.select(0, 0).select(0, 0), &hist);
            };
            Ok((finish, generated))
        })();

        self.pages.release(&block);
        self.slots_free.push(slot);
        r
    }

    pub fn enqueue(&mut self, spec: JobSpec) -> u64 {
        let serial = self.next_serial;
        self.next_serial += 1;

        // --- token healing: drop the last prompt token, constrain step 1 to the
        // tokens whose surface piece starts with the dropped token's piece.
        let mut seq = spec.prompt;
        let mut prefix_token = None;
        let mut unhealed_piece = String::new();
        let mut heal_mask = None;
        if spec.token_healing && seq.len() > 1 {
            let last = *seq.last().unwrap();
            let piece = self.tok.piece(last);
            let allowed = self.tok.tokens_with_prefix(&piece);
            if !allowed.is_empty() {
                seq.pop();
                prefix_token = Some(last);
                unhealed_piece = piece;
                heal_mask = Some(allow_mask(
                    self.model.config.vocab_size,
                    &allowed,
                    self.device,
                ));
            }
        }
        let heal_offset = prefix_token.is_some() as usize;
        let prompt_len = seq.len() as i64;

        let loop_detector = spec.stop_on_loop.map(|(w, reps)| {
            LoopDetector::new(w, Some((w / reps.max(1) as i64).max(1) as usize))
        });

        let (neg_seq, cfg_scale) = match spec.cfg {
            Some((neg, scale)) => {
                assert!(
                    !self.cache.is_qwen35(),
                    "CFG is not supported for the Qwen3.5 hybrid target"
                );
                assert!(
                    self.ngram_match_min == 0 && matches!(self.spec, Speculator::None),
                    "CFG and speculative decoding are mutually exclusive"
                );
                (Some(neg), scale)
            }
            None => (None, 0.0),
        };

        self.queue.push_back(Job {
            serial,
            seq,
            prompt_len,
            stream_prompt_len: prompt_len,
            requeue_pending: false,
            kv_pos: 0,
            block: Vec::new(),
            slot: -1,
            sampler: spec.sampler,
            max_new: spec.max_new,
            min_new: spec.min_new,
            new_tokens: 0,
            stop_tokens: spec.stop_tokens,
            stop_strings: spec.stop_strings,
            started_emitted: false,
            emitted_text: String::new(),
            gen_text: String::new(),
            detok_prefix: 0,
            detok_read: 0,
            done: false,
            eos_reason: None,
            sam: None,
            draft_block: Vec::new(),
            mtp_h_prev: None,
            mtp_carry_h: None,
            spec_primed: false,
            spec_prime_from: 0,
            hashed_pages: 0,
            accepted_draft_tokens: 0,
            rejected_draft_tokens: 0,
            prefix_token,
            unhealed_piece,
            heal_mask,
            heal_pending: heal_offset == 1,
            heal_offset,
            loop_detector,
            filters: spec.filters,
            neg_seq,
            neg_block: Vec::new(),
            neg_kv_pos: 0,
            cfg_scale,
            cfg_primed: false,
        });
        serial
    }

    pub fn num_remaining(&self) -> usize {
        self.queue.len() + self.active.len()
    }

    /// Abandon a job (client disconnected). A queued job is dropped outright; an
    /// active job is marked done so the next `iterate()` reaps it and returns its
    /// pages / recurrent slot.
    pub fn cancel(&mut self, serial: u64) {
        self.queue.retain(|j| j.serial != serial);
        if let Some(j) = self.active.iter_mut().find(|j| j.serial == serial) {
            j.done = true;
            j.eos_reason = Some("cancelled".into());
        }
    }

    /// One scheduling + inference round. Returns a stream event per job that
    /// made progress this round.
    pub fn iterate(&mut self) -> Result<Vec<StreamEvent>> {
        let _no_grad = tch::no_grad_guard();
        let mut out = Vec::new();
        let mem = std::env::var("EXL3_MEM_DEBUG").is_ok();
        let m = |tag: &str| if mem { eprintln!("[mem] {tag}: {} MiB free", crate::ffi::cuda_free_mib()); };

        m("iter-start");
        self.admit_jobs(&mut out)?;
        m("post-admit");
        self.prefill_round(&mut out)?;
        m("post-prefill");
        self.decode_round(&mut out)?;
        m("post-decode");
        self.hash_complete_pages();
        m("post-hash");

        // reap finished jobs (return pages) and requeue budget-exceeded ones
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].done {
                let job = self.active.remove(i);
                self.release_job_resources(&job);
            } else if self.active[i].requeue_pending {
                let mut job = self.active.remove(i);
                self.release_job_resources(&job);
                // fold generation into the prompt; keep client-visible state
                job.prompt_len = job.seq.len() as i64;
                job.kv_pos = 0;
                job.block = Vec::new();
                job.draft_block = Vec::new();
                job.slot = -1;
                job.hashed_pages = 0;
                job.spec_primed = false;
                job.mtp_h_prev = None;
                job.mtp_carry_h = None;
                job.sam = None;
                job.heal_pending = false;
                job.heal_mask = None;
                job.requeue_pending = false;
                self.queue.push_front(job);
            } else {
                i += 1;
            }
        }
        Ok(out)
    }

    /// Run every queued/active job to completion, collecting the final text per
    /// job in enqueue order (`generator.py` `generate()`).
    pub fn run_all(&mut self) -> Result<Vec<(u64, String)>> {
        let mut done: Vec<(u64, String)> = Vec::new();
        while self.num_remaining() > 0 {
            for ev in self.iterate()? {
                if ev.eos {
                    done.push((ev.serial, ev.full_text.unwrap_or_default()));
                }
            }
        }
        done.sort_by_key(|(s, _)| *s);
        Ok(done)
    }

    // --- scheduling -------------------------------------------------------------

    /// Return a job's shared pages, draft-model pages and GDN slot to their pools.
    fn release_job_resources(&mut self, job: &Job) {
        self.pages.release(&job.block);
        if !job.neg_block.is_empty() {
            self.pages.release(&job.neg_block);
        }
        if !job.draft_block.is_empty() {
            if let Speculator::Draft(d) = &mut self.spec {
                d.release(&job.draft_block);
            }
        }
        if job.slot >= 0 {
            // hand back the drafter's window cache with the slot, or a long-lived
            // server accumulates one per slot ever used
            if let Speculator::DFlash2(b) = &mut self.spec {
                b.release(job.slot as usize);
            }
            self.slots_free.push(job.slot);
        }
    }

    fn admit_jobs(&mut self, out: &mut Vec<StreamEvent>) -> Result<()> {
        while self.active.len() < self.max_batch {
            let Some(job) = self.queue.front() else { break };
            // a CFG job runs a second (negative) sequence — it needs pages for
            // both and occupies two decode rows.
            let neg_need = job
                .neg_seq
                .as_ref()
                .map(|n| pages_for(n.len() as i64 + job.gen_reserve()) as usize)
                .unwrap_or(0);
            let need = job.pages_needed() as usize + neg_need;
            if need > self.pages.num_pages() as usize {
                // Not "wait for room" — this job cannot fit even in an empty
                // pool, so waiting would hang it forever with no reply. Fail it
                // now with a reason the caller can act on. Reachable whenever
                // the pool ends up smaller than `max_seq_len` (it is sized to
                // the VRAM left after load, so anything else holding VRAM —
                // a pruned draft head, say — can shrink it below the config).
                let job = self.queue.pop_front().unwrap();
                let reason = format!(
                    "prompt + max_new needs {} tokens but the KV pool holds {}                      (raise VRAM or lower cache_size / max_tokens)",
                    need as i64 * PAGE_SIZE,
                    self.pages.num_pages() * PAGE_SIZE,
                );
                crate::swarn!("job {}: {reason}", job.serial);
                out.push(StreamEvent {
                    serial: job.serial,
                    stage: Stage::Streaming,
                    text: String::new(),
                    token: None,
                    eos: true,
                    eos_reason: Some(reason),
                    full_text: Some(String::new()),
                    new_tokens: 0,
                });
                continue;
            }
            if self.pages.num_free() < need {
                // not enough room yet; wait for an active job to finish
                break;
            }
            if job.neg_seq.is_some() && self.active.len() + 2 > self.max_batch {
                break;
            }
            let mut job = self.queue.pop_front().unwrap();
            if neg_need > 0 {
                job.neg_block = self.pages.alloc(neg_need)?;
            }
            // prefix-cache: share complete leading prompt pages already resident
            // from an earlier job (no-op — `alloc` — unless `enable_prefix_cache`).
            let (block, mut matched) = self
                .pages
                .alloc_prefix(&job.seq[..job.prompt_len as usize], job.gen_reserve())?;
            job.block = block;

            // CPU cache tier (homogeneous fp16 only): snapshot pages just evicted
            // from the GPU reclaim pool, then restore any prompt pages past the
            // GPU prefix match that are resident in pinned host RAM.
            if let Some(mut cpu) = self.cpu_cache.take() {
                if let GenCache::Plain(c) = &self.cache {
                    for (phys, h) in self.pages.drain_evicted() {
                        cpu.push(h, c, phys);
                    }
                    let full_pages = job.prompt_len as usize / PAGE_SIZE as usize;
                    let mut gp = (matched / PAGE_SIZE) as usize;
                    let mut prev = if gp > 0 {
                        self.pages.page_hash(job.block[gp - 1])
                    } else {
                        None
                    };
                    while gp < full_pages {
                        let s = gp * PAGE_SIZE as usize;
                        let h = chain_hash(prev, &job.seq[s..s + PAGE_SIZE as usize]);
                        let phys = job.block[gp];
                        if !cpu.restore(&h, c, phys) {
                            break;
                        }
                        self.pages.register_restored(phys, h);
                        prev = Some(h);
                        gp += 1;
                        matched += PAGE_SIZE;
                    }
                } else {
                    let _ = self.pages.drain_evicted();
                }
                self.cpu_cache = Some(cpu);
            }

            if let GenCache::Qwen35(c) = &self.cache {
                job.slot = self.slots_free.pop().expect("slot free-list empty (max_batch bug)");
                c.reset_slot(job.slot);
                // Hybrid: the KV pages are shared, but attention also needs the
                // GDN recurrent/conv state at the boundary. Restore it from the
                // checkpoint keyed by that page's hash; if there is none, the KV
                // pages still get reused but GDN re-prefills from 0.
                let mut resume = 0i64;
                if matched > 0 {
                    let pidx = (matched / PAGE_SIZE - 1) as usize;
                    let cp = self
                        .pages
                        .page_hash(job.block[pidx])
                        .and_then(|h| self.gdn_ckpts.get(&h));
                    if let Some(cp) = cp {
                        c.gdn_restore(job.slot, cp);
                        resume = matched;
                    }
                }
                if resume > 0 {
                    job.kv_pos = resume.min((job.prompt_len - 1).max(0));
                    job.hashed_pages = (job.kv_pos / PAGE_SIZE) as usize;
                    // MTP KV for [0, resume) is already resident in the reused
                    // pages — prime the head over the tail only.
                    job.spec_prime_from = job.kv_pos;
                }
                if std::env::var_os("EXL3_PREFIX_DBG").is_some() {
                    eprintln!(
                        "[prefix] prompt={} matched_pages={} resume={resume} ckpts={}",
                        job.prompt_len,
                        matched / PAGE_SIZE,
                        self.gdn_ckpts.len()
                    );
                }
            } else if matched > 0 {
                job.kv_pos = matched.min((job.prompt_len - 1).max(0));
                job.hashed_pages = (job.kv_pos / PAGE_SIZE) as usize;
            }
            if self.ngram_match_min > 0 {
                job.sam = Some(BcSam::new());
            }
            if let Speculator::Draft(d) = &mut self.spec {
                job.draft_block = d.admit(job.prompt_len + job.gen_reserve())?;
            }
            if !job.started_emitted {
                out.push(StreamEvent {
                    serial: job.serial,
                    stage: Stage::Started,
                    text: String::new(),
                    token: None,
                    eos: false,
                    eos_reason: None,
                    full_text: None,
                    new_tokens: 0,
                });
                job.started_emitted = true;
            }
            self.active.push(job);
        }
        Ok(())
    }

    // --- prefill --------------------------------------------------------------

    fn prefill_round(&mut self, out: &mut Vec<StreamEvent>) -> Result<()> {
        for j in 0..self.active.len() {
            if self.active[j].prefill_done() {
                continue;
            }
            let (start, prompt_len) = {
                let job = &self.active[j];
                (job.kv_pos, job.prompt_len)
            };
            // ingest up to max_chunk, leaving the final prompt token for the
            // first decode step (matches job.py prefill()). Stop on a page
            // boundary — including on the last chunk, so `hash_complete_pages`
            // can take a GDN prefix-cache checkpoint there; the sub-page tail
            // (< 256 tokens) is prefilled on the next round.
            let mut end = (start + self.max_chunk).min(prompt_len - 1);
            let aligned = end / PAGE_SIZE * PAGE_SIZE;
            if aligned > start {
                // at least one full page ahead — stop on its boundary
                end = aligned;
            }
            if end <= start {
                // nothing to ingest at all (prompt_len <= 1)
                self.active[j].kv_pos = (prompt_len - 1).max(0);
                continue;
            }

            let (ids, block, seqlens, slots) = {
                let job = &self.active[j];
                let ids = Tensor::from_slice(&job.seq[start as usize..end as usize])
                    .reshape([1, end - start])
                    .to_device(self.device);
                let block = Tensor::from_slice(&job.block)
                    .reshape([1, job.block.len() as i64])
                    .to_device(self.device);
                let seqlens = Tensor::from_slice(&[start as i32]).to_device(self.device);
                let slots = Tensor::from_slice(&[job.slot.max(0) as i32]).to_device(self.device);
                (ids, block, seqlens, slots)
            };

            // With MTP active, take the chunk's post-final-norm trunk hidden and
            // prime the MTP head over the same token range right away, then drop
            // it. Priming per chunk (instead of concatenating every chunk and
            // running one `prompt_len`-wide MTP forward in `prime_spec`) keeps
            // both the retained hiddens and the MTP forward's activations
            // O(chunk_size) instead of O(prompt_len) — the latter peaked at
            // ~4 GB on a 19k prompt and was the reason long prompts OOM'd.
            let dbg = std::env::var("EXL3_MEM_DEBUG").is_ok();
            let tpf = std::time::Instant::now();
            let free_before = if dbg { crate::ffi::cuda_free_mib() } else { 0 };
            let df2 = matches!(self.spec, Speculator::DFlash2(_));
            let mtp = matches!(self.spec, Speculator::Mtp(_)) && self.active[j].spec_prime_from < prompt_len;
            if df2 {
                // Capture this chunk's taps and hand them straight to the
                // drafter; nothing chunk-sized is retained, so the drafter's
                // priming stays O(chunk) like the MTP path.
                let taps = match &self.spec {
                    Speculator::DFlash2(b) => b.model.params.target_layer_ids.clone(),
                    _ => unreachable!(),
                };
                if let GenCache::Qwen35(c) = &self.cache {
                    let (_h, tap_states) = self.model.forward_qwen35_batched_h_taps(
                        &ids, c, &block, &seqlens, &slots, false, &taps,
                    );
                    let slot = self.active[j].slot.max(0) as usize;
                    if slot < self.dflash2_slots {
                        if let Speculator::DFlash2(b) = &mut self.spec {
                            b.ingest(slot, &tap_states, start);
                        }
                    }
                } else {
                    anyhow::bail!("DFlash2 speculation requires a Qwen3.5 target");
                }
            } else if mtp {
                if let GenCache::Qwen35(c) = &self.cache {
                    let (h, _) = self.model.forward_qwen35_batched_h(
                        &ids, c, &block, &seqlens, &slots, false, false,
                    );
                    self.mtp_prime_chunk(j, &h, start, end);
                } else {
                    let _ = self.cache.forward(&self.model, &ids, &block, &seqlens, &slots, true, false);
                }
            } else {
                let _ = self.cache.forward(&self.model, &ids, &block, &seqlens, &slots, true, false);
            }

            if dbg {
                tch::Cuda::synchronize(0);
                let (a, g, m, n) = crate::model::trunk_prof_take();
                let (gp, gc, gr, go) = crate::model::gdn_prof_take();
                let stages = if a + g + m + n > 0.0 {
                    format!(
                        "  [attn {a:.0} gdn {g:.0} mlp {m:.0} norm {n:.0} ms] \
                         [gdn: proj {gp:.0} conv {gc:.0} rule {gr:.0} out {go:.0}]"
                    )
                } else {
                    String::new()
                };
                eprintln!(
                    "[mem] prefill chunk [{start}..{end}] ({} tok): {:.1}ms  free {} -> {} MiB{stages}",
                    end - start,
                    tpf.elapsed().as_secs_f64() * 1000.0,
                    free_before,
                    crate::ffi::cuda_free_mib()
                );
            }

            let job = &mut self.active[j];
            job.kv_pos = end;
            out.push(StreamEvent {
                serial: job.serial,
                stage: Stage::Prefill { done: end, total: prompt_len },
                text: String::new(),
                token: None,
                eos: false,
                eos_reason: None,
                full_text: None,
                new_tokens: 0,
            });
        }
        Ok(())
    }

    // --- CFG (classifier-free guidance) -------------------------------------
    //
    // Per CFG job: one positive-context forward + one negative-context forward,
    // logits mixed `uncond + scale*(cond-uncond)`, sampled once, the token
    // appended to both sequences. Not batched across CFG jobs (rare, usually a
    // single request). Non-hybrid caches only (enforced in `enqueue`).

    fn cfg_one_forward(&self, tok: i64, pos: i64, block: &[i32]) -> Tensor {
        let ids = Tensor::from_slice(&[tok]).reshape([1, 1]).to_device(self.device);
        let blk = Tensor::from_slice(block)
            .reshape([1, block.len() as i64])
            .to_device(self.device);
        let sl = Tensor::from_slice(&[pos as i32]).to_device(self.device);
        let logits = self.cache.forward(&self.model, &ids, &blk, &sl, &sl, true, false);
        logits.squeeze_dim(1).squeeze_dim(0).to_kind(Kind::Float) // [vocab]
    }

    fn cfg_decode_round(&mut self, rows: &[usize], out: &mut Vec<StreamEvent>) -> Result<()> {
        for &j in rows {
            if !self.active[j].cfg_primed {
                let neg = self.active[j].neg_seq.clone().unwrap();
                let nlen = neg.len() as i64;
                if nlen > 1 {
                    let ids = Tensor::from_slice(&neg[..nlen as usize - 1])
                        .reshape([1, nlen - 1])
                        .to_device(self.device);
                    let blk = Tensor::from_slice(&self.active[j].neg_block)
                        .reshape([1, self.active[j].neg_block.len() as i64])
                        .to_device(self.device);
                    let sl = Tensor::from_slice(&[0i32]).to_device(self.device);
                    let _ = self.cache.forward(&self.model, &ids, &blk, &sl, &sl, true, false);
                }
                self.active[j].neg_kv_pos = (nlen - 1).max(0);
                self.active[j].cfg_primed = true;
            }

            let scale = self.active[j].cfg_scale;
            let cond = self.cfg_one_forward(
                self.active[j].seq[self.active[j].kv_pos as usize],
                self.active[j].kv_pos,
                &self.active[j].block,
            );
            let neg_tok = {
                let ns = self.active[j].neg_seq.as_ref().unwrap();
                ns[self.active[j].neg_kv_pos as usize]
            };
            let uncond =
                self.cfg_one_forward(neg_tok, self.active[j].neg_kv_pos, &self.active[j].neg_block);
            let mixed = &uncond + scale * (&cond - &uncond); // [vocab] f32

            self.active[j].kv_pos += 1;
            self.active[j].neg_kv_pos += 1;

            let mut row = mixed;
            {
                let job = &mut self.active[j];
                if !job.filters.is_empty() {
                    crate::filter::apply_filters(&mut job.filters, &mut row);
                }
            }
            let next = self.active[j].sampler.sample(&row, &self.active[j].seq);
            {
                let job = &mut self.active[j];
                job.seq.push(next);
                job.neg_seq.as_mut().unwrap().push(next);
                job.new_tokens += 1;
            }
            let ev = self.build_event(j, next);
            out.push(ev);
        }
        Ok(())
    }

    // --- decode --------------------------------------------------------------

    fn decode_round(&mut self, out: &mut Vec<StreamEvent>) -> Result<()> {
        let all_rows: Vec<usize> = (0..self.active.len())
            .filter(|&j| self.active[j].prefill_done() && !self.active[j].done)
            .collect();
        // CFG jobs take a dedicated per-job path (positive + negative forward,
        // logits mixed); everything else batches normally.
        let (cfg_rows, rows): (Vec<usize>, Vec<usize>) =
            all_rows.iter().partition(|&&j| self.active[j].neg_seq.is_some());
        if !cfg_rows.is_empty() {
            self.cfg_decode_round(&cfg_rows, out)?;
        }
        if rows.is_empty() {
            return Ok(());
        }

        // constrained decoding needs a per-position logit mask that advances with
        // each accepted token — suppress speculation while any row has an active
        // filter (matches upstream `iterate_gen` returning `None` for filtered jobs).
        let spec_ok = rows.iter().all(|&j| self.active[j].filters.is_empty());
        if spec_ok && self.ngram_match_min > 0 && self.ngram_decode_round(&rows, out)? {
            return Ok(());
        }
        if spec_ok
            && !matches!(self.spec, Speculator::None)
            && self.spec_decode_round(&rows, out)?
        {
            return Ok(());
        }

        let bsz = rows.len() as i64;
        // block-table width = the widest per-row page count (each admitted job
        // holds all the pages it will ever need). Matches the single-sequence
        // `infer` path's `num_pages_per_seq`, so the attention kernel derives the
        // same chunk count and stays bit-identical to `infer`.
        let max_pages = rows.iter().map(|&j| self.active[j].pages_held()).max().unwrap() as i64;

        let mut feed = Vec::with_capacity(rows.len());
        let mut seqlens = Vec::with_capacity(rows.len());
        let mut slots = Vec::with_capacity(rows.len());
        let mut block = vec![0i32; (bsz * max_pages) as usize];
        for (r, &j) in rows.iter().enumerate() {
            let job = &self.active[j];
            feed.push(job.seq[job.kv_pos as usize]);
            seqlens.push(job.kv_pos as i32);
            slots.push(job.slot.max(0) as i32);
            for (p, &phys) in job.block.iter().enumerate() {
                block[r * max_pages as usize + p] = phys;
            }
        }

        // one H2D copy per buffer, into the pre-allocated persistent scratch
        let mut ids_v = self.dec_ids.narrow(0, 0, bsz);
        ids_v.copy_(&Tensor::from_slice(&feed));
        let mut seqlens_t = self.dec_seqlens.narrow(0, 0, bsz);
        seqlens_t.copy_(&Tensor::from_slice(&seqlens));
        let mut slots_t = self.dec_slots.narrow(0, 0, bsz);
        slots_t.copy_(&Tensor::from_slice(&slots));
        let mut block_dst = self.dec_block.narrow(0, 0, bsz).narrow(1, 0, max_pages);
        block_dst.copy_(&Tensor::from_slice(&block).reshape([bsz, max_pages]));
        let block_t = block_dst.contiguous(); // kernel requires a contiguous block table

        let logits = self.cache.forward(
            &self.model,
            &ids_v.reshape([bsz, 1]),
            &block_t,
            &seqlens_t,
            &slots_t,
            true,
            false, // plain decode: GDN advances one committed token, no rewind
        ); // [bsz, 1, vocab]
        let logits = logits.squeeze_dim(1); // [bsz, vocab]

        // Fast path: every row is plain greedy (no temperature, no penalties, no
        // pending heal mask) → one batched argmax and a single device→host sync,
        // instead of `bsz` separate `.int64_value()` reads that each stall the
        // pipeline.
        let greedy_batch = rows.iter().all(|&j| {
            let s = &self.active[j].sampler;
            s.temperature <= 0.0
                && !s.needs_past_ids()
                && !self.active[j].heal_pending
                && self.active[j].filters.is_empty()
        });
        if greedy_batch {
            let toks = logits.argmax(1, false).to_device(Device::Cpu);
            for (r, &j) in rows.iter().enumerate() {
                let next = toks.int64_value(&[r as i64]);
                {
                    let job = &mut self.active[j];
                    job.kv_pos += 1;
                    job.seq.push(next);
                    job.new_tokens += 1;
                }
                out.push(self.build_event(j, next));
            }
            return Ok(());
        }

        for (r, &j) in rows.iter().enumerate() {
            let row = logits.get(r as i64);
            let (_next, ev) = self.step_job(j, &row)?;
            out.push(ev);
        }
        Ok(())
    }

    // --- n-gram speculative decode ------------------------------------------
    //
    // Behavioural port of `iterate_ngram_gen` + the draft-verification branch of
    // `iterate_gen`. Each decoding job proposes a draft from its suffix
    // automaton; the batch is trimmed to the shortest draft (as upstream does),
    // then a single `q_len = min_len + 1` forward verifies all positions at
    // once. Greedy longest-accepted-prefix matching per job; rejected draft
    // positions leave stale K/V that the next round overwrites.
    //
    // Returns `Ok(false)` when no speculation happened this round (some job had
    // no draft) so the caller falls back to plain single-token decode.
    fn ngram_decode_round(&mut self, rows: &[usize], out: &mut Vec<StreamEvent>) -> Result<bool> {
        let window = self.num_draft_tokens;
        let mm = self.ngram_match_min;
        let mut drafts: Vec<Vec<i64>> = Vec::with_capacity(rows.len());
        for &j in rows {
            drafts.push(self.active[j].get_ngram_draft(window, mm));
        }
        self.spec_verify_round(rows, drafts, false, out)
    }

    // --- shared speculative verify path -----------------------------------
    //
    // Behavioural port of the draft-verification branch of `iterate_gen`. Given a
    // per-row draft, trim the batch to the shortest (`min_len`, all-or-nothing as
    // upstream), run one `q_len = min_len + 1` forward, greedily accept the
    // longest matching prefix per row, roll back GDN state, then (MTP only) sync
    // the draft head's KV over the accepted positions with the verify hiddens.
    //
    // `want_hidden` routes through `forward_qwen35_batched_h` so the MTP head can
    // consume the trunk hidden states. Returns `Ok(false)` if no row had a draft.
    fn spec_verify_round(
        &mut self,
        rows: &[usize],
        mut drafts: Vec<Vec<i64>>,
        want_hidden: bool,
        out: &mut Vec<StreamEvent>,
    ) -> Result<bool> {
        let mut min_len = self.num_draft_tokens.max(0);
        for d in &drafts {
            min_len = min_len.min(d.len() as i64);
        }
        if min_len == 0 {
            return Ok(false);
        }
        for d in &mut drafts {
            d.truncate(min_len as usize);
        }

        let bsz = rows.len() as i64;
        let q_len = min_len + 1;
        let max_pages = rows.iter().map(|&j| self.active[j].pages_held()).max().unwrap() as i64;
        let hybrid = self.cache.is_qwen35();

        let mut feed = Vec::with_capacity((bsz * q_len) as usize);
        let mut seqlens = Vec::with_capacity(rows.len());
        let mut slots = Vec::with_capacity(rows.len());
        let mut block = vec![0i32; (bsz * max_pages) as usize];
        for (r, &j) in rows.iter().enumerate() {
            let job = &self.active[j];
            feed.push(job.seq[job.kv_pos as usize]);
            feed.extend_from_slice(&drafts[r]);
            seqlens.push(job.kv_pos as i32);
            slots.push(job.slot.max(0) as i32);
            for (p, &phys) in job.block.iter().enumerate() {
                block[r * max_pages as usize + p] = phys;
            }
        }

        if want_hidden && !self.cache.is_qwen35() {
            anyhow::bail!("MTP speculation requires a Qwen3.5 target");
        }

        // Stage inputs into the persistent device scratch (one H2D copy each
        // instead of a fresh `to_device` per round).
        let ids = {
            let mut s = self.dec_seqlens.narrow(0, 0, bsz);
            s.copy_(&Tensor::from_slice(&seqlens));
            let mut sl = self.dec_slots.narrow(0, 0, bsz);
            sl.copy_(&Tensor::from_slice(&slots));
            let mut b = self.dec_block.narrow(0, 0, bsz).narrow(1, 0, max_pages);
            b.copy_(&Tensor::from_slice(&block).reshape([bsz, max_pages]));
            Tensor::from_slice(&feed).reshape([bsz, q_len]).to_device(self.device)
        };
        let bt = self.dec_block.narrow(0, 0, bsz).narrow(1, 0, max_pages).contiguous();
        let sq = self.dec_seqlens.narrow(0, 0, bsz);
        let so = self.dec_slots.narrow(0, 0, bsz);

        // DFlash2 rebuilds its context from the target's tapped hidden states,
        // so the verify has to hand them back — captured in the same forward
        // rather than costing a second pass.
        let df2_taps: Vec<i64> = match &self.spec {
            Speculator::DFlash2(b) => b.model.params.target_layer_ids.clone(),
            _ => Vec::new(),
        };
        let mut vtaps: Vec<Tensor> = Vec::new();
        // The trunk verify forward (the expensive 64-layer pass).
        let (vhid, logits): (Option<Tensor>, Tensor) = if !df2_taps.is_empty() {
            let c = match &self.cache {
                GenCache::Qwen35(c) => c,
                _ => unreachable!(),
            };
            let (h, t) = self
                .model
                .forward_qwen35_batched_h_taps(&ids, c, &bt, &sq, &so, true, &df2_taps);
            let l = self.model.lm_head_on(&h);
            vtaps = t;
            (Some(h), l)
        } else if want_hidden {
            let c = match &self.cache {
                GenCache::Qwen35(c) => c,
                _ => unreachable!(),
            };
            let (h, l) =
                self.model.forward_qwen35_batched_h(&ids, c, &bt, &sq, &so, true, true);
            (Some(h), l)
        } else {
            let l = self
                .cache
                .forward(&self.model, &ids, &bt, &sq, &so, false, hybrid);
            (None, l)
        };

        // pass 1: per-row accept + stats + GDN rewind + (MTP) draft-KV sync
        let mut spec = std::mem::replace(&mut self.spec, Speculator::None);

        // Greedy fast path: when every row is plain greedy, one batched argmax +
        // a single D2H copy replaces up to `bsz * q_len` per-position
        // `.int64_value()` pipeline stalls (this is what `bin/infer`'s MTP loop
        // does; the generic path below stalls once per accepted token).
        let all_greedy = rows.iter().all(|&j| {
            let s = &self.active[j].sampler;
            s.temperature <= 0.0
                && !s.needs_past_ids()
                && !self.active[j].heal_pending
                && self.active[j].filters.is_empty()
        });
        let greedy_toks: Option<Vec<Vec<i64>>> = if all_greedy {
            let am = logits.argmax(-1, false).to_kind(Kind::Int64).to_device(Device::Cpu); // [bsz, q_len]
            Some(
                (0..bsz)
                    .map(|r| (0..q_len).map(|c| am.int64_value(&[r, c])).collect())
                    .collect(),
            )
        } else {
            None
        };

        let mut all_accepted: Vec<Vec<i64>> = Vec::with_capacity(rows.len());
        for (r, &j) in rows.iter().enumerate() {
            let row_logits = logits.get(r as i64); // [q_len, vocab]
            let draft = &drafts[r];
            let q_old = self.active[j].kv_pos;

            let mut accepted: Vec<i64> = Vec::with_capacity(draft.len() + 1);
            if let Some(gt) = &greedy_toks {
                let row = &gt[r as usize];
                accepted.push(row[0]);
                for (i, &dtok) in draft.iter().enumerate() {
                    if dtok != accepted[i] {
                        break;
                    }
                    accepted.push(row[i + 1]);
                }
            } else {
                let seq_snapshot = self.active[j].seq.clone();
                let t0 = self.active[j].sampler.sample(&row_logits.get(0), &seq_snapshot);
                accepted.push(t0);
                let mut past = seq_snapshot;
                for (i, &dtok) in draft.iter().enumerate() {
                    if dtok != accepted[i] {
                        break;
                    }
                    past.push(accepted[i]);
                    let ti = self.active[j].sampler.sample(&row_logits.get(i as i64 + 1), &past);
                    accepted.push(ti);
                }
            }

            let n_acc_draft = (accepted.len() - 1) as u64;
            {
                // MLE update: `a` successes, plus one failure if the window was
                // not exhausted. Free — this outcome is already on the host.
                let a = n_acc_draft as f32;
                let trials = a + if (n_acc_draft as i64) < min_len { 1.0 } else { 0.0 };
                if trials > 0.0 {
                    const DECAY: f32 = 0.94;
                    self.accept_p = self.accept_p * DECAY + (a / trials) * (1.0 - DECAY);
                    self.accept_obs = self.accept_obs.saturating_add(1);
                }
            }
            // Label the calibrator from what the verifier actually tested: every
            // accepted drafted position, plus the first rejection (if any).
            if let Some((crows, _, cconf)) = &self.pending_conf {
                if let Some(ci) = crows.iter().position(|&x| x == j) {
                    if let Some(conf) = cconf.get(ci) {
                        let a = n_acc_draft as usize;
                        let labels: Vec<(f32, bool)> = conf
                            .iter()
                            .take(a)
                            .map(|&c| (c, true))
                            .chain(conf.get(a).map(|&c| (c, false)))
                            .collect();
                        if let Some(cal) = self.draft_cal.as_mut() {
                            for (c, ok) in labels {
                                cal.add_label(c, ok);
                            }
                        }
                    }
                }
            }
            self.active[j].accepted_draft_tokens += n_acc_draft;
            self.active[j].rejected_draft_tokens += min_len as u64 - n_acc_draft;
            self.ngram_accepted += n_acc_draft;
            self.ngram_drafted += min_len as u64;

            if let GenCache::Qwen35(c) = &self.cache {
                c.gdn_rewind(self.active[j].slot, accepted.len() as i64, q_len);
            }
            if r + 1 == rows.len() {
                // one decay per verification round, after every row is labelled
                self.pending_conf = None;
                if let Some(cal) = self.draft_cal.as_mut() {
                    cal.decay_step();
                }
            }

            if let Speculator::DFlash2(b) = &mut spec {
                // Accepted means the verify fed the *correct* tokens at
                // positions `q_old ..= q_old + k`, so those taps are real; the
                // first rejected position's tap was computed from a wrong token
                // and must not enter the cache.
                let keep = accepted.len() as i64; // k accepted drafts + the bonus token
                let slot = self.active[j].slot.max(0) as usize;
                if slot >= self.dflash2_slots {
                    all_accepted.push(accepted);
                    continue;
                }
                let row_taps: Vec<Tensor> = vtaps
                    .iter()
                    .map(|t| t.narrow(0, r as i64, 1).narrow(1, 0, keep))
                    .collect();
                b.ingest(slot, &row_taps, q_old);
            }
            if let (Speculator::Mtp(mtp), Some(vh)) = (&spec, &vhid) {
                let block_row = Tensor::from_slice(&self.active[j].block)
                    .reshape([1, self.active[j].block.len() as i64])
                    .to_device(self.device);
                let vh_row = vh.narrow(0, r as i64, 1); // [1, q_len, h]
                let k_draft = accepted.len() - 1;
                mtp.model
                    .sync_row(&self.model, &mtp.cache, &vh_row, &accepted[..k_draft], q_old, &block_row);
                self.active[j].mtp_h_prev = Some(
                    vh_row
                        .narrow(1, accepted.len() as i64 - 1, 1)
                        .contiguous(),
                );
            }
            all_accepted.push(accepted);
        }
        self.spec = spec;

        // pass 2: commit accepted tokens + stream events
        for (r, &j) in rows.iter().enumerate() {
            for tok in std::mem::take(&mut all_accepted[r]) {
                self.active[j].kv_pos += 1;
                self.active[j].seq.push(tok);
                self.active[j].new_tokens += 1;
                let ev = self.build_event(j, tok);
                let eos = ev.eos;
                out.push(ev);
                if eos {
                    break;
                }
            }
        }
        Ok(true)
    }

    // --- model-based speculators (draft model / MTP) ----------------------

    fn spec_decode_round(&mut self, rows: &[usize], out: &mut Vec<StreamEvent>) -> Result<bool> {
        let unprimed: Vec<usize> = rows
            .iter()
            .copied()
            .filter(|&j| !self.active[j].spec_primed)
            .collect();
        for j in unprimed {
            let dbg = std::env::var("EXL3_MEM_DEBUG").is_ok();
            let t = std::time::Instant::now();
            self.prime_spec(j)?;
            if dbg {
                tch::Cuda::synchronize(0);
                eprintln!("[mem] prime_spec: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        match &self.spec {
            Speculator::Mtp(_) => self.mtp_decode_round(rows, out),
            Speculator::Draft(_) => self.draft_decode_round(rows, out),
            Speculator::DFlash2(_) => self.dflash2_decode_round(rows, out),
            Speculator::None => Ok(false),
        }
    }

    /// Prime the MTP head's KV over one prefill chunk's token range.
    ///
    /// `h` is the trunk's post-final-norm hidden `[1, end-start, hidden]` for
    /// positions `[start, end)`. MTP position `p` is fed `(token[p],
    /// trunk_hidden[p-1])`, so the hidden for the chunk's *first* position comes
    /// from `mtp_carry_h` (the previous chunk's last hidden); the chunk's own
    /// last hidden becomes the carry for the next chunk. Nothing but that
    /// one-token carry outlives the call, which is what keeps the MTP prime's
    /// memory flat in prompt length.
    fn mtp_prime_chunk(&mut self, j: usize, h: &Tensor, start: i64, end: i64) {
        let Speculator::Mtp(mtp) = &self.spec else { return };
        let n = end - start;
        if n <= 0 {
            return;
        }
        let toks: Vec<i64> = self.active[j].seq[start as usize..end as usize].to_vec();
        let block_row = Tensor::from_slice(&self.active[j].block)
            .reshape([1, self.active[j].block.len() as i64])
            .to_device(self.device);
        mtp.model.prime_row(
            &self.model,
            &mtp.cache,
            h,
            &toks,
            start,
            self.active[j].mtp_carry_h.as_ref(),
            &block_row,
        );
        // carry the chunk's last hidden (position end-1) into the next chunk
        self.active[j].mtp_carry_h = Some(h.narrow(1, n - 1, 1).contiguous());
    }

    /// Draft-window length for the next round.
    ///
    /// Throughput per round is `(1 + sum_{i=1..n} p^i) / (V + n*d)`: the verify
    /// cost `V` is paid whatever the window length — it re-reads every weight
    /// regardless of `q_len` — and each draft step adds `d`, dominated by the
    /// `lm_head` (1.36 ms of a ~32 ms round on this model, because the head is
    /// stored at 8 bits = 1.27 GB and already runs at 93% of memory bandwidth).
    /// Only the RATIO `d/V` enters, and it is measured online (`cost_r`).
    ///
    /// A single fixed window is wrong because the optimum moves with acceptance:
    /// 3 at `p = 0.55`, 4+ at `p = 0.72`, and real traffic spans both. Never
    /// exceeds `draft_num_tokens`, so the config stays the cap.
    fn draft_window(&self) -> i64 {
        let nmax = self.num_draft_tokens;
        if nmax <= 1 || self.accept_obs < 8 {
            return nmax; // warm up on full windows so the estimate gets data
        }
        let r = DRAFT_COST_RATIO
            .get_or_init(|| {
                std::env::var("EXL3_DRAFT_COST_RATIO")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
            })
            .unwrap_or(self.cost_r)
            .clamp(0.005, 1.0);
        let p = self.accept_p.clamp(0.01, 0.999);
        let (mut best_n, mut best) = (1i64, f32::MIN);
        let mut tokens = 1.0f32; // the verify's own token
        let mut pw = 1.0f32;
        for n in 1..=nmax {
            pw *= p;
            tokens += pw;
            let score = tokens / (1.0 + n as f32 * r);
            if score > best {
                best = score;
                best_n = n;
            }
        }
        best_n
    }

    /// One-shot per-job priming of the speculator's KV state.
    fn prime_spec(&mut self, j: usize) -> Result<()> {
        let prompt_len = self.active[j].prompt_len;
        let prompt: Vec<i64> = self.active[j].seq[..prompt_len as usize].to_vec();
        match &self.spec {
            Speculator::Draft(d) => {
                d.prime_row(&prompt, &self.active[j].draft_block);
            }
            Speculator::DFlash2(_) => {
                // `prefill_round` fed every ingested position's taps already,
                // and the drafter's block starts at the first un-ingested
                // position, so there is nothing left to prime.
            }
            Speculator::Mtp(mtp) => {
                // `prefill_round` already primed the MTP KV for every position
                // it ingested (`mtp_prime_chunk`). Prefill stops one token short
                // of the prompt (the last token is the first decode step's
                // input), so only MTP position `prompt_len - 1` is left — a
                // single-token forward fed by the carried trunk hidden at
                // `prompt_len - 2`.
                let carry = self.active[j].mtp_carry_h.take();
                if let (Some(carry), true) = (carry, prompt_len >= 1) {
                    let last = prompt_len - 1;
                    let block_row = Tensor::from_slice(&self.active[j].block)
                        .reshape([1, self.active[j].block.len() as i64])
                        .to_device(self.device);
                    mtp.model.prime_row(
                        &self.model,
                        &mtp.cache,
                        &carry,
                        &prompt[last as usize..],
                        last,
                        Some(&carry),
                        &block_row,
                    );
                }
            }
            Speculator::None => {}
        }
        self.active[j].spec_primed = true;
        Ok(())
    }

    /// MTP round: draft `num_draft_tokens` per row via the MTP head, then verify
    /// through the trunk. Any row without a trunk hidden yet (freshly primed) or
    /// mid token-healing takes a plain hidden decode step first (upstream returns
    /// `None` from `iterate_draftmodel_mtp_gen` in the same case).
    fn mtp_decode_round(&mut self, rows: &[usize], out: &mut Vec<StreamEvent>) -> Result<bool> {
        let need_plain = rows
            .iter()
            .any(|&j| self.active[j].mtp_h_prev.is_none() || self.active[j].heal_pending);
        if need_plain {
            self.mtp_plain_round(rows, out)?;
            return Ok(true);
        }

        let n = self.draft_window();
        let bsz = rows.len() as i64;
        let max_pages = rows.iter().map(|&j| self.active[j].pages_held()).max().unwrap() as i64;
        let mut h_prev_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
        let mut c = Vec::with_capacity(rows.len());
        let mut q = Vec::with_capacity(rows.len());
        let mut block = vec![0i32; (bsz * max_pages) as usize];
        for (r, &j) in rows.iter().enumerate() {
            let job = &self.active[j];
            h_prev_rows.push(job.mtp_h_prev.as_ref().unwrap().shallow_clone());
            c.push(job.seq[job.kv_pos as usize]);
            q.push(job.kv_pos);
            for (p, &phys) in job.block.iter().enumerate() {
                block[r * max_pages as usize + p] = phys;
            }
        }
        let h_prev = Tensor::cat(&h_prev_rows, 0); // [bsz,1,h]
        let block_t = Tensor::from_slice(&block)
            .reshape([bsz, max_pages])
            .to_device(self.device);

        let dbg = std::env::var("EXL3_MEM_DEBUG").is_ok();
        let t_draft = std::time::Instant::now();
        let (drafts_dev, confs) = match &self.spec {
            Speculator::Mtp(mtp) => mtp.model.draft_n_batched(
                &self.model,
                &mtp.cache,
                &h_prev,
                &c,
                &q,
                &block_t,
                n,
                self.draft_cal.as_ref(),
            ),
            _ => unreachable!(),
        };
        let n = drafts_dev.size()[1]; // the calibrator may have cut the window short
        let dh = drafts_dev.to_kind(Kind::Int64).to_device(Device::Cpu); // [bsz,n]
        let drafts: Vec<Vec<i64>> = (0..bsz)
            .map(|r| (0..n).map(|c2| dh.int64_value(&[r, c2])).collect())
            .collect();
        self.pending_conf = Some((rows.to_vec(), drafts.clone(), confs));
        let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;
        let t_ver = std::time::Instant::now();
        let r = self.spec_verify_round(rows, drafts, true, out);
        let ver_ms = t_ver.elapsed().as_secs_f64() * 1000.0;
        {
            // Both halves are already host-synchronised (each ends in a device->host
            // readback), so this is a real measurement, not launch time.
            let obs = (draft_ms / n.max(1) as f64) / ver_ms.max(1e-3);
            const DECAY: f64 = 0.9;
            self.cost_r = if self.accept_obs == 0 {
                obs as f32
            } else {
                (self.cost_r as f64 * DECAY + obs * (1.0 - DECAY)) as f32
            };
        }
        if dbg {
            tch::Cuda::synchronize(0);
            use std::sync::atomic::Ordering;
            let deq = crate::modules::DEQ_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
            let att = crate::modules::ATTN_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
            let (ta, tg, tm, tn) = crate::model::trunk_prof_take();
            let (gp, gc, gr, go) = crate::model::gdn_prof_take();
            if ta + tg + tm + tn > 0.0 {
                eprintln!(
                    "[mem]   trunk: attn {ta:.1} gdn {tg:.1} mlp {tm:.1} norm {tn:.1} ms \
                     | gdn: proj {gp:.1} conv {gc:.1} rule {gr:.1} out {go:.1}"
                );
            }
            eprintln!(
                "[mem] mtp step: n={} p={:.3} r={:.3}  draft {:.2}ms  verify {:.2}ms  [kv-dequant {:.2}ms attn {:.2}ms]  (ctx {})",
                n,
                self.accept_p,
                self.cost_r,
                draft_ms,
                ver_ms,
                deq,
                att,
                self.active[rows[0]].kv_pos
            );
        }
        r
    }

    /// Plain hidden-exporting decode step (q_len 1) — advances one token per row
    /// and captures each row's trunk hidden as the next `mtp_h_prev`.
    fn mtp_plain_round(&mut self, rows: &[usize], out: &mut Vec<StreamEvent>) -> Result<()> {
        let bsz = rows.len() as i64;
        let max_pages = rows.iter().map(|&j| self.active[j].pages_held()).max().unwrap() as i64;
        let mut feed = Vec::with_capacity(rows.len());
        let mut seqlens = Vec::with_capacity(rows.len());
        let mut slots = Vec::with_capacity(rows.len());
        let mut block = vec![0i32; (bsz * max_pages) as usize];
        for (r, &j) in rows.iter().enumerate() {
            let job = &self.active[j];
            feed.push(job.seq[job.kv_pos as usize]);
            seqlens.push(job.kv_pos as i32);
            slots.push(job.slot.max(0) as i32);
            for (p, &phys) in job.block.iter().enumerate() {
                block[r * max_pages as usize + p] = phys;
            }
        }
        let ids = Tensor::from_slice(&feed).reshape([bsz, 1]).to_device(self.device);
        let block_t = Tensor::from_slice(&block)
            .reshape([bsz, max_pages])
            .to_device(self.device);
        let seqlens_t = Tensor::from_slice(&seqlens).to_device(self.device);
        let slots_t = Tensor::from_slice(&slots).to_device(self.device);
        let (vhid, logits) = match &self.cache {
            GenCache::Qwen35(cc) => self.model.forward_qwen35_batched_h(
                &ids, cc, &block_t, &seqlens_t, &slots_t, false, true,
            ),
            _ => anyhow::bail!("MTP speculation requires a Qwen3.5 target"),
        };
        for (r, &j) in rows.iter().enumerate() {
            self.active[j].mtp_h_prev = Some(
                vhid.narrow(0, r as i64, 1).narrow(1, 0, 1).contiguous(),
            );
            let row = logits.get(r as i64).get(0);
            let (_next, ev) = self.step_job(j, &row)?;
            out.push(ev);
        }
        Ok(())
    }

    /// DFlash2 round: one block forward per row, then a single shared `lm_head`
    /// + selector pass for the whole batch, then the usual verify.
    fn dflash2_decode_round(&mut self, rows: &[usize], out: &mut Vec<StreamEvent>) -> Result<bool> {
        let anchors: Vec<i64> = rows
            .iter()
            .map(|&j| self.active[j].seq[self.active[j].kv_pos as usize])
            .collect();
        let mut states = Vec::with_capacity(rows.len());
        for (r, &j) in rows.iter().enumerate() {
            let slot = self.active[j].slot.max(0) as usize;
            if slot >= self.dflash2_slots {
                // this row has no window cache; fall back to plain decode for
                // the whole round rather than drafting a subset
                return Ok(false);
            }
            let Speculator::DFlash2(b) = &self.spec else { unreachable!() };
            let Some(cache) = b.cache(slot) else {
                return Ok(false);
            };
            let emb = b.model.block_input(&self.model, anchors[r]);
            states.push(b.model.block_state(&emb, cache));
        }
        let states = Tensor::cat(&states, 0); // [bsz, block, H]
        let drafts = match &self.spec {
            Speculator::DFlash2(b) => b.model.paths_from_states(&self.model, &states, &anchors),
            _ => unreachable!(),
        };
        self.spec_verify_round(rows, drafts, true, out)
    }

    /// Draft-model round: draft `num_draft_tokens` per row with the AR draft
    /// model, then verify through the target.
    fn draft_decode_round(&mut self, rows: &[usize], out: &mut Vec<StreamEvent>) -> Result<bool> {
        let n = self.num_draft_tokens;
        let feed: Vec<i64> = rows
            .iter()
            .map(|&j| self.active[j].seq[self.active[j].kv_pos as usize])
            .collect();
        let pos: Vec<i64> = rows.iter().map(|&j| self.active[j].kv_pos).collect();
        let drafts = match &self.spec {
            Speculator::Draft(d) => {
                let blocks: Vec<&[i32]> =
                    rows.iter().map(|&j| self.active[j].draft_block.as_slice()).collect();
                d.draft_batch(&feed, &pos, &blocks, n)
            }
            _ => unreachable!(),
        };
        self.spec_verify_round(rows, drafts, false, out)
    }

    /// Advance one job by a single sampled token and build its stream event.
    fn step_job(&mut self, j: usize, logits_row: &Tensor) -> Result<(i64, StreamEvent)> {
        let next = {
            let job = &mut self.active[j];
            job.kv_pos += 1;
            // token healing: constrain the first sampled token to the allowed set
            let masked = if job.heal_pending {
                let mut r = logits_row * 1.0;
                let _ = r.f_add_(job.heal_mask.as_ref().unwrap()).unwrap();
                Some(r)
            } else {
                None
            };
            let row = masked.as_ref().unwrap_or(logits_row);
            let filtered = if !job.filters.is_empty() {
                let mut r = row.to_kind(Kind::Float);
                crate::filter::apply_filters(&mut job.filters, &mut r);
                Some(r)
            } else {
                None
            };
            let row = filtered.as_ref().unwrap_or(row);
            let next = job.sampler.sample(row, &job.seq);
            if job.heal_pending {
                job.heal_pending = false;
                job.heal_mask = None;
            }
            job.seq.push(next);
            job.new_tokens += 1;
            next
        };
        let ev = self.build_event(j, next);
        Ok((next, ev))
    }

    /// Build the stream event for a token already appended to `job.seq`
    /// (`new_tokens` already incremented). Runs stop-token / stop-string / max
    /// checks and the incremental-detokenisation diff.
    fn build_event(&mut self, j: usize, next: i64) -> StreamEvent {
        let job = &mut self.active[j];

        let hit_stop_token =
            job.stop_tokens.contains(&next) && job.gen_count() > job.min_new;
        let hit_max = job.gen_count() >= job.max_new;
        // loop detection: feed the just-generated token (healed token included,
        // matching job.py which feeds the whole held-token buffer)
        let hit_loop = job
            .loop_detector
            .as_mut()
            .map(|ld| ld.feed(next))
            .unwrap_or(false);
        // advance constrained-decoding filters; a completed grammar stops the job
        let mut hit_filter = false;
        for f in &mut job.filters {
            hit_filter |= f.feed(next);
        }

        // decode everything generated so far, then diff against what we've emitted.
        // With token healing the first gen id is the healed token, whose piece
        // starts with `unhealed_piece`; strip that so only the *added* text shows.
        // Steady state uses the bounded incremental detokeniser; on eos (or when
        // stop-strings need an exact full string) fall back to a one-shot decode
        // so the returned `full_text` is byte-identical to the non-incremental path.
        let eos_now = hit_stop_token || hit_max || hit_loop || hit_filter;
        let mut full = if job.stop_strings.is_empty() && !eos_now {
            job.extend_detok(&self.tok);
            job.gen_text.clone()
        } else {
            let gen_ids: Vec<i64> = job.seq[job.stream_prompt_len as usize..].to_vec();
            self.tok.decode(&gen_ids).unwrap_or_default()
        };
        if job.heal_offset == 1 {
            if let Some(rest) = full.strip_prefix(job.unhealed_piece.as_str()) {
                full = rest.to_string();
            }
        }

        let mut eos = false;
        let mut eos_reason: Option<String> = None;
        let mut visible = full.clone();

        // stop strings: truncate at the earliest match and finish
        for s in &job.stop_strings {
            if let Some(pos) = full.find(s.as_str()) {
                visible.truncate(pos);
                eos = true;
                eos_reason = Some("stop_string".into());
            }
        }
        if !eos && hit_stop_token {
            // stop token's text is not part of the output
            let mut v = self
                .tok
                .decode(&job.seq[job.stream_prompt_len as usize..job.seq.len() - 1])
                .unwrap_or_default();
            if job.heal_offset == 1 {
                if let Some(rest) = v.strip_prefix(job.unhealed_piece.as_str()) {
                    v = rest.to_string();
                }
            }
            visible = v;
            eos = true;
            eos_reason = Some("stop_token".into());
        }
        if !eos && hit_max {
            eos = true;
            eos_reason = Some("max_new_tokens".into());
        }
        if !eos && hit_loop {
            eos = true;
            eos_reason = Some("loop_detected".into());
        }
        if !eos && hit_filter {
            eos = true;
            eos_reason = Some("filter_completed".into());
        }

        // hold back a tail that might still grow into a stop string (unless finishing)
        let hold = if eos {
            0
        } else {
            job.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1)
        };
        let safe_len = visible.len().saturating_sub(hold);
        let safe_len = floor_char_boundary(&visible, safe_len);
        let emit_target = &visible[..safe_len.max(prefix_len(&visible, &job.emitted_text))];

        let new_text = emit_target
            .strip_prefix(job.emitted_text.as_str())
            .unwrap_or("")
            .to_string();
        job.emitted_text.push_str(&new_text);

        if eos {
            job.done = true;
            job.eos_reason = eos_reason.clone();
        } else if self.rq_budget > 0
            && job.neg_seq.is_none()
            && job.gen_count() >= self.rq_budget
        {
            // fair-scheduling: fold generation into the prompt and re-enqueue.
            // Draft positions past `kv_pos` are already uncommitted (each spec
            // round only advances `kv_pos` over accepted tokens), so nothing to
            // roll back here — the reap phase does the physical requeue.
            job.requeue_pending = true;
        }

        let full_text = if eos { Some(visible.clone()) } else { None };
        let new_tokens = job.gen_count();
        let serial = job.serial;

        StreamEvent {
            serial,
            stage: Stage::Streaming,
            text: new_text,
            token: Some(next),
            eos,
            eos_reason,
            full_text,
            new_tokens,
        }
    }
}

/// Additive logit mask: `0.0` on `allowed` token ids, `-inf` everywhere else.
fn allow_mask(vocab: i64, allowed: &[i64], device: Device) -> Tensor {
    let m = Tensor::full([vocab], f64::NEG_INFINITY, (Kind::Float, device));
    let idx = Tensor::from_slice(allowed).to_device(device);
    let zeros = Tensor::zeros([allowed.len() as i64], (Kind::Float, device));
    m.index_copy(0, &idx, &zeros)
}

/// Largest `i <= max` that is a char boundary of `s`.
fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Length of the longest prefix of `s` that `already` also starts with.
fn prefix_len(s: &str, already: &str) -> usize {
    let n = s.len().min(already.len());
    let mut i = 0;
    let sb = s.as_bytes();
    let ab = already.as_bytes();
    while i < n && sb[i] == ab[i] {
        i += 1;
    }
    floor_char_boundary(s, i)
}
