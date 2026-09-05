//! Paged KV cache — equivalence-only port of `cache/cache.py` + the generator's
//! page table (grade —).
//!
//! Upstream `Cache` is a paged, optionally-quantized ring of layer tensors whose
//! pages are handed out by the generator's `PageTable`. This port needs only the
//! single-sequence, non-quantized, never-evicted case: a contiguous run of pages
//! (`block_table = [0, 1, 2, …]`) and a one-element `cache_seqlens`.
//!
//! Why paged and not a plain `[1, max_len, …]` buffer: the paged kernels
//! (`paged_kv_cache_update`, `bighead_attn_paged`) read the current sequence
//! length from `cache_seqlens` **on the device**, so a single CUDA-graph capture
//! of the decode step replays correctly at every position — the pointer is
//! stable, only the value changes. A shape-based length would bake the position
//! into the captured graph.

use crate::config::{Config, LayerKind};
use crate::qwen3_5::GdnState;
use tch::{Device, Kind, Tensor};

pub const PAGE_SIZE: i64 = 256;

pub struct PagedKvCache {
    /// per layer, `[num_pages, PAGE_SIZE, n_kv_heads, head_dim]` fp16
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    /// `[1, num_pages]` int32, identity mapping `0..num_pages`
    pub block_table: Tensor,
    /// `[1]` int32 on device — tokens currently stored (pre-append length)
    pub seqlens: Tensor,
    pub max_len: i64,
}

impl PagedKvCache {
    pub fn new(cfg: &Config, max_len: i64, device: Device) -> Self {
        let num_pages = (max_len + PAGE_SIZE - 1) / PAGE_SIZE;
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
            block_table: Tensor::arange(num_pages, (Kind::Int, device)).reshape([1, num_pages]),
            seqlens: Tensor::zeros([1], (Kind::Int, device)),
            max_len: num_pages * PAGE_SIZE,
        }
    }

    /// Length bump. Called eagerly after the ungraphed prefill, and recorded into
    /// the graph for decode (so each replay advances by exactly one). Uses the
    /// out-param form (aliasing `seqlens`) to stay a `&self` in-place op.
    pub fn advance(&self, n: i64) {
        let _ = self.seqlens.add_scalar_out(&self.seqlens, n);
    }
}

/// Quantized single-sequence paged KV cache — port of `cache/quant.py`
/// `CacheLayer_quant` (grade —, single-seq / no-eviction subset).
///
/// Per layer K and V are stored as packed `int32` codes (`qk`/`qv`, one group of
/// 32 values → `k_bits` int32-packed bits) plus `fp16` per-group scales
/// (`sk`/`sv`). Attention dequantizes the stored prefix into a shared pair of
/// `fp16` scratch page pools (`k_scratch`/`v_scratch`) and then runs the normal
/// `bighead_attn_paged` — matching upstream's `CacheLayer_quant.get_kv` path
/// (the online-dequant-in-kernel path is not ported).
pub struct QuantPagedKvCache {
    pub qk: Vec<Tensor>,
    pub qv: Vec<Tensor>,
    pub sk: Vec<Tensor>,
    pub sv: Vec<Tensor>,
    /// shared fp16 dequant scratch, `[num_pages, PAGE_SIZE, n_kv, head_dim]`
    pub k_scratch: Tensor,
    pub v_scratch: Tensor,
    pub block_table: Tensor,
    pub seqlens: Tensor,
    pub k_bits: i64,
    pub v_bits: i64,
    pub compand_a: f32,
    pub max_len: i64,
}

impl QuantPagedKvCache {
    pub fn new(cfg: &Config, max_len: i64, k_bits: i64, v_bits: i64, device: Device) -> Self {
        assert!((2..=8).contains(&k_bits) && (2..=8).contains(&v_bits), "cache bits must be 2..=8");
        let num_pages = (max_len + PAGE_SIZE - 1) / PAGE_SIZE;
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
        let mkf = || Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device));
        let n = cfg.num_hidden_layers;
        Self {
            qk: (0..n).map(|_| mkq(k_bits)).collect(),
            qv: (0..n).map(|_| mkq(v_bits)).collect(),
            sk: (0..n).map(|_| mks()).collect(),
            sv: (0..n).map(|_| mks()).collect(),
            k_scratch: mkf(),
            v_scratch: mkf(),
            block_table: Tensor::arange(num_pages, (Kind::Int, device)).reshape([1, num_pages]),
            seqlens: Tensor::zeros([1], (Kind::Int, device)),
            k_bits,
            v_bits,
            compand_a: 0.0,
            max_len: num_pages * PAGE_SIZE,
        }
    }

    pub fn advance(&self, n: i64) {
        let _ = self.seqlens.add_scalar_out(&self.seqlens, n);
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5 hybrid cache — KV pages for the `full_attention` layers, a recurrent
// `GdnState` for each `linear_attention` layer. All layers share one
// `block_table` / `seqlens` pair (token positions are absolute, not per-kind).
// ---------------------------------------------------------------------------

pub enum Q35LayerCache {
    Kv { k: Tensor, v: Tensor },
    Gdn(GdnState),
}

pub struct Qwen35Cache {
    pub layers: Vec<Q35LayerCache>,
    pub block_table: Tensor,
    pub seqlens: Tensor,
    pub max_len: i64,
}

impl Qwen35Cache {
    pub fn new(cfg: &Config, max_len: i64, device: Device) -> Self {
        let num_pages = (max_len + PAGE_SIZE - 1) / PAGE_SIZE;
        let hd = cfg.head_dim;
        let nkv = cfg.kv_heads_eff().0;
        let gdn = cfg.gdn.expect("Qwen35Cache::new on a non-Qwen3.5 config");
        let layers = cfg
            .layer_types
            .iter()
            .map(|lk| match lk {
                LayerKind::FullAttention => Q35LayerCache::Kv {
                    k: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
                    v: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
                },
                LayerKind::LinearAttention => Q35LayerCache::Gdn(GdnState::new(&gdn, device)),
            })
            .collect();
        Self {
            layers,
            block_table: Tensor::arange(num_pages, (Kind::Int, device)).reshape([1, num_pages]),
            seqlens: Tensor::zeros([1], (Kind::Int, device)),
            max_len: num_pages * PAGE_SIZE,
        }
    }

    pub fn advance(&self, n: i64) {
        let _ = self.seqlens.add_scalar_out(&self.seqlens, n);
    }

    /// Rewind to an empty sequence, reusing the allocation. Zeroes `seqlens` and
    /// every GDN recurrent/conv state (stale KV rows are masked by `seqlens`).
    pub fn reset(&self) {
        let _ = self.seqlens.shallow_clone().zero_();
        for l in &self.layers {
            if let Q35LayerCache::Gdn(g) = l {
                let _ = g.conv_state.shallow_clone().zero_();
                let _ = g.recurrent_state.shallow_clone().zero_();
            }
        }
    }
}

/// Cache for the qwen4_exp hybrid stack.
///
/// Deliberately **contiguous, single-sequence** rather than paged: the QSA
/// layers attend under a per-query selection mask, which the paged kernels
/// cannot express (see `Attention::forward_masked`), and the indexer needs the
/// full raw-key history in one plane to pool blocks from. Alongside K/V it
/// carries the indexer's raw keys — unnormed and unroped, since a pooled block
/// key is normed after the fp32 mean and roped at the block's *start* position,
/// so nothing about a token's own key can be baked in early.
pub enum Q4LayerCache {
    Full {
        k: Tensor,
        v: Tensor,
        /// `[1, max_len, indexer_head_dim]` fp16 raw indexer keys.
        raw_k: Tensor,
    },
    Gdn(GdnState),
}

pub struct Qwen4Cache {
    pub layers: Vec<Q4LayerCache>,
    /// PLE conv window + trailing token ids, one slot.
    pub ple: crate::ple_state::PleState,
    /// Tokens already committed, i.e. the absolute position of the next query.
    pub past_len: std::cell::Cell<i64>,
    pub max_len: i64,
}

impl Qwen4Cache {
    pub fn new(cfg: &Config, max_len: i64, device: Device) -> Self {
        let q4 = cfg.qwen4.as_ref().expect("Qwen4Cache::new on a non-qwen4_exp config");
        let hd = cfg.head_dim;
        // The native KV head count, not `kv_heads_eff`: the QSA layers attend in
        // tch under a mask rather than through the paged kernel, so there is no
        // kernel-supported GQA ratio to round up to.
        let nkv = cfg.num_kv_heads;
        let gdn = cfg.gdn.expect("qwen4_exp config without GDN params");
        let layers = cfg
            .layer_types
            .iter()
            .map(|lk| match lk {
                LayerKind::FullAttention => Q4LayerCache::Full {
                    k: Tensor::zeros([1, max_len, nkv, hd], (Kind::Half, device)),
                    v: Tensor::zeros([1, max_len, nkv, hd], (Kind::Half, device)),
                    raw_k: Tensor::zeros([1, max_len, q4.indexer_head_dim], (Kind::Half, device)),
                },
                LayerKind::LinearAttention => Q4LayerCache::Gdn(GdnState::new(&gdn, device)),
            })
            .collect();
        Self {
            layers,
            ple: crate::ple_state::PleState::new(
                1,
                q4.hc_mult * cfg.hidden_size,
                (q4.ple_conv_kernel_size - 1) * q4.ngram_size,
                q4.ngram_size - 1,
                0,
                q4.ple_eos_token_id,
                device,
            ),
            past_len: std::cell::Cell::new(0),
            max_len,
        }
    }

    pub fn advance(&self, n: i64) {
        self.past_len.set(self.past_len.get() + n);
    }

    /// Rewind to an empty sequence, reusing the allocation.
    pub fn reset(&self) {
        self.past_len.set(0);
        self.ple.reset_slot(0);
        for l in &self.layers {
            if let Q4LayerCache::Gdn(g) = l {
                let _ = g.conv_state.shallow_clone().zero_();
                let _ = g.recurrent_state.shallow_clone().zero_();
            }
        }
    }
}
