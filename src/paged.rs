//! Shared paged KV cache pool + page allocator — equivalence-only port of
//! `cache/cache.py` storage and `generator/pagetable.py` allocation (grade —).
//!
//! v1 scope: a flat pool of 256-token pages per layer, shared by every job, with
//! a free-list allocator. **No prefix-cache dedup** — upstream hashes completed
//! pages and shares identical prefixes between sequences (`pagetable.py`
//! `referenced_pages` / `unreferenced_pages`); here every job gets fresh pages
//! and releases them on completion. No CPU offload, no page rotation.

use crate::config::{Config, LayerKind};
use anyhow::{bail, Result};
use std::collections::{HashMap, VecDeque};
use tch::{Device, Kind, Tensor};

pub const PAGE_SIZE: i64 = 256;

/// Per-layer K/V page storage: `[num_pages, PAGE_SIZE, n_kv_heads, head_dim]` fp16.
pub struct PagedCache {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub num_pages: i64,
    pub n_layers: usize,
}

impl PagedCache {
    pub fn new(cfg: &Config, num_pages: i64, device: Device) -> Self {
        let hd = cfg.head_dim;
        // `kv_heads_eff`, not `num_kv_heads`: `Attention::forward` repeats the
        // KV heads up to a GQA ratio the kernel supports, and the kernel checks
        // this plane's head count against what it is handed. GLM-4-9B (32 q / 2
        // kv, ratio 16) is the case that needs it.
        let nkv = cfg.kv_heads_eff().0;
        let mk = || Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device));
        Self {
            k: (0..cfg.num_hidden_layers).map(|_| mk()).collect(),
            v: (0..cfg.num_hidden_layers).map(|_| mk()).collect(),
            num_pages,
            n_layers: cfg.num_hidden_layers as usize,
        }
    }

    /// Total token capacity across all pages.
    pub fn capacity(&self) -> i64 {
        self.num_pages * PAGE_SIZE
    }
}

/// Quantized variant of [`PagedCache`] — port of `cache/quant.py` storage for
/// the batched generator. Per-layer packed `int32` codes + `fp16` group scales.
/// Attention dequantizes into a compact `bsz*pages_per_seq` fp16 scratch it
/// allocates per call (no fixed pool-sized cost — see `Attn::PagedQuant`).
pub struct QuantPagedCache {
    pub qk: Vec<Tensor>,
    pub qv: Vec<Tensor>,
    pub sk: Vec<Tensor>,
    pub sv: Vec<Tensor>,
    pub k_bits: i64,
    pub v_bits: i64,
    pub compand_a: f32,
    pub num_pages: i64,
}

impl QuantPagedCache {
    pub fn new(cfg: &Config, num_pages: i64, k_bits: i64, v_bits: i64, device: Device) -> Self {
        assert!(
            (2..=8).contains(&k_bits) && (2..=8).contains(&v_bits),
            "cache bits must be 2..=8"
        );
        let hd = cfg.head_dim;
        // `kv_heads_eff`, not `num_kv_heads`: `Attention::forward` repeats the
        // KV heads up to a GQA ratio the kernel supports, and the kernel checks
        // this plane's head count against what it is handed. GLM-4-9B (32 q / 2
        // kv, ratio 16) is the case that needs it.
        let nkv = cfg.kv_heads_eff().0;
        let token_dim = nkv * hd;
        assert!(token_dim % 32 == 0, "n_kv*head_dim must be a multiple of 32");
        let groups = token_dim / 32;
        let mkq = |bits: i64| Tensor::zeros([num_pages, PAGE_SIZE, groups * bits], (Kind::Int, device));
        let mks = || Tensor::zeros([num_pages, PAGE_SIZE, groups], (Kind::Half, device));
        let n = cfg.num_hidden_layers;
        Self {
            qk: (0..n).map(|_| mkq(k_bits)).collect(),
            qv: (0..n).map(|_| mkq(v_bits)).collect(),
            sk: (0..n).map(|_| mks()).collect(),
            sv: (0..n).map(|_| mks()).collect(),
            k_bits,
            v_bits,
            compand_a: 0.0,
            num_pages,
        }
    }
}

/// Shared hybrid cache for the batched Qwen3.5 generator: KV page pools for the
/// `full_attention` layers, per-slot recurrent state pools for the
/// `linear_attention` layers. One `Vec` entry per decoder layer, in order.
pub enum Q35BatchLayer {
    Kv { k: Tensor, v: Tensor },
    /// KV pages quantized to `k_bits`/`v_bits` — packed `int32` codes + `fp16`
    /// per-group scales, dequantized into the cache's shared `k_scratch`/
    /// `v_scratch` before attention (mirrors the dense [`QuantPagedCache`]).
    KvQuant { qk: Tensor, qv: Tensor, sk: Tensor, sv: Tensor },
    Gdn { conv_state: Tensor, recurrent_state: Tensor },
}

pub struct Qwen35PagedCache {
    pub layers: Vec<Q35BatchLayer>,
    pub num_pages: i64,
    pub max_slots: i64,
    /// Per-step recurrent snapshots kept for speculative-decode rewind. 0 = no
    /// history (`conv_state` is `[.., K]`, `recurrent_state` is `[.., 1, ..]`).
    pub max_history: i64,
    /// KV-cache quant bit widths for the `full_attention` layers (`(0, 0)` =
    /// unquantized fp16 pages). GDN recurrent state is never quantized.
    pub k_bits: i64,
    pub v_bits: i64,
    pub compand_a: f32,
    conv_k: i64,
}

impl Qwen35PagedCache {
    pub fn new(cfg: &Config, num_pages: i64, max_slots: i64, device: Device) -> Self {
        Self::new_hist(cfg, num_pages, max_slots, 0, (0, 0), device)
    }

    /// `max_history > 0` reserves `max_history` per-token recurrent/conv snapshots
    /// so a speculative forward of up to `max_history` draft tokens can be rewound
    /// to the accepted prefix. Memory cost per GDN layer per slot ≈
    /// `max_history * num_v_heads * k_head_dim * v_head_dim * 4` bytes
    /// (~3 MB/token for Qwen3.8-27B) — keep `max_slots` small unless the card is big.
    pub fn new_hist(
        cfg: &Config,
        num_pages: i64,
        max_slots: i64,
        max_history: i64,
        kv_bits: (i64, i64),
        device: Device,
    ) -> Self {
        let hd = cfg.head_dim;
        let nkv = cfg.kv_heads_eff().0;
        let g = cfg.gdn.expect("Qwen35PagedCache::new on a non-Qwen3.5 config");
        let ck = g.conv_kernel_size;
        let (k_bits, v_bits) = kv_bits;
        let quant = k_bits > 0 || v_bits > 0;
        if quant {
            assert!(
                (2..=8).contains(&k_bits) && (2..=8).contains(&v_bits),
                "cache bits must be 2..=8"
            );
            assert!((nkv * hd) % 32 == 0, "n_kv*head_dim must be a multiple of 32");
        }
        let groups = nkv * hd / 32;
        let layers = cfg
            .layer_types
            .iter()
            .map(|lk| match lk {
                LayerKind::FullAttention if quant => Q35BatchLayer::KvQuant {
                    qk: Tensor::zeros([num_pages, PAGE_SIZE, groups * k_bits], (Kind::Int, device)),
                    qv: Tensor::zeros([num_pages, PAGE_SIZE, groups * v_bits], (Kind::Int, device)),
                    sk: Tensor::zeros([num_pages, PAGE_SIZE, groups], (Kind::Half, device)),
                    sv: Tensor::zeros([num_pages, PAGE_SIZE, groups], (Kind::Half, device)),
                },
                LayerKind::FullAttention => Q35BatchLayer::Kv {
                    k: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
                    v: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
                },
                LayerKind::LinearAttention => Q35BatchLayer::Gdn {
                    conv_state: Tensor::zeros(
                        [max_slots, g.fdim_qkv(), ck + max_history],
                        (Kind::BFloat16, device),
                    ),
                    recurrent_state: Tensor::zeros(
                        [max_slots, max_history + 1, g.num_v_heads, g.k_head_dim, g.v_head_dim],
                        (Kind::Float, device),
                    ),
                },
            })
            .collect();
        Self {
            layers,
            num_pages,
            max_slots,
            max_history,
            k_bits,
            v_bits,
            compand_a: 0.0,
            conv_k: ck,
        }
    }

    pub fn capacity(&self) -> i64 {
        self.num_pages * PAGE_SIZE
    }

    /// Zero the recurrent state for one slot across every GDN layer — call when a
    /// job is admitted so it starts from a clean state.
    pub fn reset_slot(&self, slot: i64) {
        for l in &self.layers {
            if let Q35BatchLayer::Gdn { conv_state, recurrent_state } = l {
                let _ = conv_state.narrow(0, slot, 1).zero_();
                let _ = recurrent_state.narrow(0, slot, 1).zero_();
            }
        }
    }

    /// After a **history** forward of `consumed` tokens on `slot`, commit exactly
    /// the first `keep` (drop `consumed - keep` rejected speculative tokens):
    ///
    /// * recurrent: `[slot, 0] <- [slot, keep]` (per-token snapshot), unless
    ///   `keep == consumed` (already the current state).
    /// * conv: the history conv1d writes the fresh rolling window to the *tail*
    ///   of the buffer, never to `[:, :K]` (that's the non-history layout), so the
    ///   working window `[:, :K]` must ALWAYS be restored — `p = (K + max_history)
    ///   - (consumed - keep)`, window = `[:, p-K : p]` (mirrors
    ///   `GDNLayerState.rewind`; `keep == consumed` picks the very last K entries).
    ///
    /// Requires `max_history >= consumed` and `1 <= keep <= consumed`.
    pub fn gdn_rewind(&self, slot: i64, keep: i64, consumed: i64) {
        debug_assert!(self.max_history >= consumed && keep >= 1 && keep <= consumed);
        let k = self.conv_k;
        let p = (k + self.max_history) - (consumed - keep);
        for l in &self.layers {
            if let Q35BatchLayer::Gdn { conv_state, recurrent_state } = l {
                if keep < consumed {
                    let src = recurrent_state.select(0, slot).select(0, keep);
                    let _ = recurrent_state.select(0, slot).select(0, 0).copy_(&src);
                }
                let cs = conv_state.select(0, slot); // [fdim, K + max_history]
                let win = cs.narrow(1, p - k, k).contiguous();
                let _ = cs.narrow(1, 0, k).copy_(&win);
            }
        }
    }

    /// Copy `slot`'s live GDN state OUT — the working conv window (`[fdim, K]`
    /// bf16) and SSM recurrent state (`[Nv, k_hd, v_hd]` f32) of every linear
    /// layer, at whatever position the slot is currently at. Used as a
    /// prefix-cache checkpoint taken at a page boundary.
    pub fn gdn_snapshot(&self, slot: i64) -> GdnCheckpoint {
        let k = self.conv_k;
        let mut conv = Vec::new();
        let mut rec = Vec::new();
        for l in &self.layers {
            if let Q35BatchLayer::Gdn { conv_state, recurrent_state } = l {
                conv.push(conv_state.select(0, slot).narrow(1, 0, k).contiguous());
                rec.push(recurrent_state.select(0, slot).select(0, 0).contiguous());
            }
        }
        GdnCheckpoint { conv, rec }
    }

    /// As [`gdn_snapshot`] but the state is copied to (pinned) host RAM. The
    /// generator keeps an LRU of these across a whole conversation, so keeping
    /// them off the GPU matters — each is ~100 MB. Restore does an H2D copy
    /// (~a few ms), vastly cheaper than re-prefilling the shared prefix.
    pub fn gdn_snapshot_cpu(&self, slot: i64) -> GdnCheckpoint {
        let cp = self.gdn_snapshot(slot);
        let cpu = |t: &Tensor| t.to_device(Device::Cpu);
        GdnCheckpoint {
            conv: cp.conv.iter().map(cpu).collect(),
            rec: cp.rec.iter().map(cpu).collect(),
        }
    }

    /// Restore a [`gdn_snapshot`] into `slot` (call after `reset_slot`, before the
    /// tail prefill). Only `[:, :K]` / `[slot, 0]` are written — the history
    /// planes stay zeroed and are repopulated by the first speculative forward.
    pub fn gdn_restore(&self, slot: i64, cp: &GdnCheckpoint) {
        let k = self.conv_k;
        let mut i = 0;
        for l in &self.layers {
            if let Q35BatchLayer::Gdn { conv_state, recurrent_state } = l {
                let dev = recurrent_state.device();
                let _ = conv_state
                    .select(0, slot)
                    .narrow(1, 0, k)
                    .copy_(&cp.conv[i].to_device(dev));
                let _ = recurrent_state
                    .select(0, slot)
                    .select(0, 0)
                    .copy_(&cp.rec[i].to_device(dev));
                i += 1;
            }
        }
    }
}

/// Prefix-cache snapshot of one slot's GDN state at a page boundary (one entry
/// per linear-attention layer). ~148 MB for Qwen3.8-27B; the generator keeps a
/// small LRU of these keyed by the boundary page's chained content hash.
pub struct GdnCheckpoint {
    pub conv: Vec<Tensor>,
    pub rec: Vec<Tensor>,
}

/// Chained content hash of one complete page = `hash(prev_page_hash ++ token_ids)`,
/// truncated to 128 bits — two keyed `DefaultHasher` passes (SipHash), strong
/// enough for non-adversarial prompt-prefix dedup (`pagetable.py`
/// `_tensor_blake2b_checksum`, but over token ids, which suffices for a
/// deterministic model).
pub type PageHash = [u8; 16];

pub fn chain_hash(prev: Option<PageHash>, token_ids: &[i64]) -> PageHash {
    use std::hash::{Hash, Hasher};
    let mk = |seed: u64| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut h);
        prev.hash(&mut h);
        token_ids.hash(&mut h);
        h.finish()
    };
    let a = mk(0x9E37_79B9_7F4A_7C15);
    let b = mk(0xC2B2_AE3D_27D4_EB4F);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&a.to_le_bytes());
    out[8..].copy_from_slice(&b.to_le_bytes());
    out
}

/// Free-list page allocator with prefix-cache dedup — port of
/// `generator/pagetable.py` allocation (grade —, v1: full-page reuse only, no
/// partial pages, no CPU tier, LRU eviction of unreferenced hashed pages).
///
/// Physical page ids are `0..num_pages`. Each page carries a refcount (a hashed
/// prefix page is shared by every job that matched it) and, once it holds a
/// complete 256-token span, that span's chained content hash. A page whose
/// refcount hits 0 but still has a valid hash is kept "reclaimable" — a later
/// job with the same prefix revives it instead of re-prefilling — until the free
/// list runs dry and it is repurposed (LRU).
pub struct PageTable {
    num_pages: i64,
    free: Vec<i32>,
    /// refcount per physical page
    rc: Vec<u32>,
    /// chained content hash per physical page (once the page is complete)
    hash: Vec<Option<PageHash>>,
    /// chained hash -> physical page (may currently be reclaimable, rc == 0)
    by_hash: HashMap<PageHash, i32>,
    /// rc==0 pages that still carry a hash, oldest first (approx LRU)
    reclaimable: VecDeque<i32>,
    /// `(phys, old_hash)` for hashed pages repurposed since the last drain — the
    /// generator snapshots these to the CPU tier before they are overwritten.
    evicted: Vec<(i32, PageHash)>,
    /// dedup metrics
    pub cached_tokens: u64,
    pub prompt_tokens: u64,
    /// prefix-cache dedup enabled (else `alloc_prefix` == `alloc`)
    pub dedup: bool,
}

impl PageTable {
    pub fn new(num_pages: i64) -> Self {
        Self {
            num_pages,
            free: (0..num_pages as i32).rev().collect(),
            rc: vec![0; num_pages as usize],
            hash: vec![None; num_pages as usize],
            by_hash: HashMap::new(),
            reclaimable: VecDeque::new(),
            evicted: Vec::new(),
            cached_tokens: 0,
            prompt_tokens: 0,
            dedup: false,
        }
    }

    /// Pages that could be handed out (free + reclaimable). Conservative — a
    /// prefix hit consumes none of these, so an `alloc_prefix` after this check
    /// passes always succeeds.
    pub fn num_free(&self) -> usize {
        self.free.len() + self.reclaimable.len()
    }

    pub fn num_pages(&self) -> i64 {
        self.num_pages
    }

    pub fn prefix_stats(&self) -> (u64, u64) {
        (self.cached_tokens, self.prompt_tokens)
    }

    /// Take one physical page for fresh use (drops any stale hash it carried).
    fn take_one(&mut self) -> i32 {
        let id = self.free.pop().unwrap_or_else(|| {
            let id = self
                .reclaimable
                .pop_front()
                .expect("page table exhausted (num_free check skipped?)");
            if let Some(h) = self.hash[id as usize].take() {
                if self.by_hash.get(&h) == Some(&id) {
                    self.by_hash.remove(&h);
                }
                self.evicted.push((id, h));
            }
            id
        });
        self.rc[id as usize] = 1;
        self.hash[id as usize] = None;
        id
    }

    /// Add a reference to an already-live-or-reclaimable page (prefix hit).
    fn acquire(&mut self, phys: i32) {
        if self.rc[phys as usize] == 0 {
            if let Some(pos) = self.reclaimable.iter().position(|&x| x == phys) {
                self.reclaimable.remove(pos);
            }
        }
        self.rc[phys as usize] += 1;
    }

    /// Allocate `n` fresh pages, or fail if the pool can't satisfy the request.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<i32>> {
        if self.num_free() < n {
            bail!("page table exhausted: need {n}, {} free", self.num_free());
        }
        Ok((0..n).map(|_| self.take_one()).collect())
    }

    /// Allocate a block for a job whose prompt is `prompt_ids`, reserving
    /// `gen_reserve` extra tokens for generation. Complete leading pages whose
    /// chained hash is already in the table are shared (no prefill needed);
    /// returns `(block, matched_tokens)` where `matched_tokens` is how many
    /// leading prompt tokens are already resident in the shared pages.
    pub fn alloc_prefix(
        &mut self,
        prompt_ids: &[i64],
        gen_reserve: i64,
    ) -> Result<(Vec<i32>, i64)> {
        let need = pages_for(prompt_ids.len() as i64 + gen_reserve) as usize;
        if !self.dedup {
            return Ok((self.alloc(need)?, 0));
        }
        let full_pages = prompt_ids.len() / PAGE_SIZE as usize;
        let mut hits: Vec<i32> = Vec::new();
        let mut prev: Option<PageHash> = None;
        for p in 0..full_pages {
            let chunk = &prompt_ids[p * PAGE_SIZE as usize..(p + 1) * PAGE_SIZE as usize];
            let h = chain_hash(prev, chunk);
            match self.by_hash.get(&h).copied() {
                Some(phys) => {
                    hits.push(phys);
                    prev = Some(h);
                }
                None => break,
            }
        }
        for &phys in &hits {
            self.acquire(phys);
        }
        let mut block = hits.clone();
        let fresh = need - hits.len();
        for _ in 0..fresh {
            block.push(self.take_one());
        }
        let matched = hits.len() as i64 * PAGE_SIZE;
        self.cached_tokens += matched as u64;
        self.prompt_tokens += prompt_ids.len() as u64;
        Ok((block, matched))
    }

    /// Register the chained content hash of a now-complete page so later jobs can
    /// share it. `prev` is the preceding page's hash (`page_hash(block[idx-1])`),
    /// `None` for the first page. Idempotent.
    pub fn register_page_hash(&mut self, phys: i32, prev: Option<PageHash>, token_ids: &[i64]) {
        if self.hash[phys as usize].is_some() || token_ids.len() != PAGE_SIZE as usize {
            return;
        }
        let h = chain_hash(prev, token_ids);
        self.hash[phys as usize] = Some(h);
        self.by_hash.entry(h).or_insert(phys);
    }

    pub fn page_hash(&self, phys: i32) -> Option<PageHash> {
        self.hash[phys as usize]
    }

    /// `(phys, old_hash)` pairs for hashed pages repurposed since the last call —
    /// the generator snapshots each to the CPU tier before prefill overwrites it.
    pub fn drain_evicted(&mut self) -> Vec<(i32, PageHash)> {
        std::mem::take(&mut self.evicted)
    }

    /// Mark a fresh page as holding the content for `h` (its K/V were just
    /// restored from the CPU tier). The page must already be referenced.
    pub fn register_restored(&mut self, phys: i32, h: PageHash) {
        self.hash[phys as usize] = Some(h);
        self.by_hash.entry(h).or_insert(phys);
    }

    pub fn release(&mut self, pages: &[i32]) {
        for &id in pages {
            let rc = &mut self.rc[id as usize];
            if *rc > 0 {
                *rc -= 1;
            }
            if self.rc[id as usize] == 0 {
                if self.hash[id as usize].is_some() {
                    self.reclaimable.push_back(id);
                } else {
                    self.free.push(id);
                }
            }
        }
    }
}

/// Pages needed to hold `tokens` tokens.
pub fn pages_for(tokens: i64) -> i64 {
    (tokens + PAGE_SIZE - 1) / PAGE_SIZE
}
