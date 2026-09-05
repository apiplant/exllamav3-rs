//! Pinned host-RAM second tier for evicted hashed KV pages — behavioural port of
//! `generator/cpu_cache.py` `CPUPageCache` (grade —, v1: homogeneous fp16 cache
//! only, LRU eviction, no draft-cache segment).
//!
//! When GPU page pressure repurposes a complete, content-hashed page
//! ([`PageTable::drain_evicted`]), its K/V across every layer are copied to a
//! fixed-size pinned host slot keyed by the page's chained hash. A later job
//! whose prompt prefix hashes to that key skips a prefill pass over the page and
//! pays one host→device copy per layer tensor instead.

use crate::config::Config;
use crate::paged::{pages_for, PageHash, PagedCache, PAGE_SIZE};
use std::collections::{HashMap, VecDeque};
use tch::{Device, Kind, Tensor};

pub struct CpuPageCache {
    /// one pinned host tensor per slot: `[n_layers, 2, PAGE_SIZE, n_kv, head_dim]` fp16
    slots: Vec<Tensor>,
    slot_hash: Vec<Option<PageHash>>,
    map: HashMap<PageHash, usize>,
    /// slot indices, front = least recently used
    lru: VecDeque<usize>,
    free: Vec<usize>,
    n_layers: usize,
    /// metrics
    pub restored_pages: u64,
    pub pushed_pages: u64,
}

impl CpuPageCache {
    /// `max_tokens` sizes the tier (rounded to whole pages). Each slot is
    /// `n_layers * 2 * PAGE_SIZE * n_kv * head_dim * 2` bytes of pinned host RAM.
    pub fn new(cfg: &Config, max_tokens: i64, device: Device) -> Self {
        let n_layers = cfg.num_hidden_layers as usize;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let max_slots = pages_for(max_tokens).max(1) as usize;
        let mk = || {
            let t = Tensor::zeros(
                [n_layers as i64, 2, PAGE_SIZE, nkv, hd],
                (Kind::Half, Device::Cpu),
            );
            // pin for async H2D/D2H; falls back to pageable if pinning fails
            t.pin_memory(device)
        };
        Self {
            slots: (0..max_slots).map(|_| mk()).collect(),
            slot_hash: vec![None; max_slots],
            map: HashMap::new(),
            lru: VecDeque::new(),
            free: (0..max_slots).rev().collect(),
            n_layers,
            restored_pages: 0,
            pushed_pages: 0,
        }
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.restored_pages, self.pushed_pages)
    }

    fn take_slot(&mut self) -> usize {
        if let Some(s) = self.free.pop() {
            return s;
        }
        // evict the LRU slot
        let s = self.lru.pop_front().expect("cpu cache has no slots");
        if let Some(h) = self.slot_hash[s].take() {
            self.map.remove(&h);
        }
        s
    }

    /// Snapshot page `phys` of `cache` (all layers) to a host slot under `h`.
    /// Idempotent — a hash already resident is left as-is.
    pub fn push(&mut self, h: PageHash, cache: &PagedCache, phys: i32) {
        if self.map.contains_key(&h) {
            return;
        }
        let s = self.take_slot();
        let dst = &self.slots[s];
        for l in 0..self.n_layers {
            let _ = dst
                .select(0, l as i64)
                .select(0, 0)
                .copy_(&cache.k[l].select(0, phys as i64));
            let _ = dst
                .select(0, l as i64)
                .select(0, 1)
                .copy_(&cache.v[l].select(0, phys as i64));
        }
        self.slot_hash[s] = Some(h);
        self.map.insert(h, s);
        self.lru.push_back(s);
        self.pushed_pages += 1;
    }

    /// Restore hash `h` into page `phys` of `cache`. Returns `true` if `h` was in
    /// the tier (and the copy was issued).
    pub fn restore(&mut self, h: &PageHash, cache: &PagedCache, phys: i32) -> bool {
        let Some(&s) = self.map.get(h) else { return false };
        let src = &self.slots[s];
        for l in 0..self.n_layers {
            // `copy_` does the host→device transfer directly
            let _ = cache.k[l]
                .select(0, phys as i64)
                .copy_(&src.select(0, l as i64).select(0, 0));
            let _ = cache.v[l]
                .select(0, phys as i64)
                .copy_(&src.select(0, l as i64).select(0, 1));
        }
        // bump recency
        if let Some(pos) = self.lru.iter().position(|&x| x == s) {
            self.lru.remove(pos);
        }
        self.lru.push_back(s);
        self.restored_pages += 1;
        true
    }
}
