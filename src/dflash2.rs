//! DFlash2 speculative draft model — port of the MiaAI-Lab exllamav3 fork
//! (`architecture/dflash2.py` + `modules/arch_specific/dflash2.py`), which in
//! turn decodes the `dflash` package's `DFlash2DraftModel`.
//!
//! # Why this exists
//!
//! Our MTP drafter costs one full `lm_head` pass per drafted token — measured
//! at 1.93 ms/step, of which ~1.90 ms is streaming the 1.27 GB output
//! embedding. Drafting `n` tokens therefore costs `n × T_d`, and that term
//! dominates a speculative round.
//!
//! DFlash2 breaks the linearity. It drafts a whole block of `block_size` (8)
//! positions in **one** forward:
//!
//! ```text
//!   block input = target.embed([anchor, MASK, MASK, …])   # 8 rows
//!   x = fc(concat(target hidden @ layers 5,19,33,47,61))  # context K/V only
//!   x = 5 × DFlash2Block(x)                               # conv-wrapped GQA
//!   logits = target.lm_head(norm(x))                      # ONE pass, 8 rows
//!   path   = selector.walk(top-16 per row)                # cheap chaining
//! ```
//!
//! So the `n × lm_head` term collapses to a single pass, and cross-position
//! coherence — the thing an independent per-position prediction gets wrong — is
//! restored by the selector's chained top-k walk rather than by re-running the
//! model. The selector walk is also what makes this subsume tree speculation:
//! it explores `top_k` candidates per slot without any extra model passes.
//!
//! # Coupling to the target
//!
//! The drafter has no embedding table and no `lm_head`; it borrows both from
//! the target, and its context K/V is *projected from the target's own hidden
//! states* (`update_kv_from_target`) rather than produced by running the draft
//! model over the context. It is therefore tied to a specific target
//! checkpoint; a finetuned target shifts the hidden states it was trained
//! against, which costs acceptance (never correctness — verification is exact).
//!
//! # Dtypes
//!
//! The residual stream is bf16 by design and reaches |x| ~ 1.5e5, far outside
//! fp16 range. Following the reference: residual stream, conv `finish` outputs
//! and the final state are bf16; everything downstream of an RMSNorm is
//! bounded (<= ~50) and uses the stock fp16 EXL3 modules.

use crate::config::Config;
use crate::modules::{Attention, GatedMlp, Linear};
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result};
use tch::{Device, Kind, Tensor};

/// DFlash2-specific knobs from `config.json -> dflash_config`.
#[derive(Clone, Debug)]
pub struct DFlash2Params {
    /// Positions drafted per forward, including the anchor row (8).
    pub block_size: i64,
    pub conv_kernel_size: i64,
    pub conv_group_size: i64,
    /// Token id filling the `block_size - 1` noise rows.
    pub mask_token_id: i64,
    pub selector_rank: i64,
    pub selector_top_k: i64,
    /// Target layers whose outputs feed `fc` (5 taps for this checkpoint).
    pub target_layer_ids: Vec<i64>,
    pub sliding_window: i64,
}

impl DFlash2Params {
    pub fn from_config(raw: &serde_json::Value) -> Result<Self> {
        let d = raw
            .get("dflash_config")
            .ok_or_else(|| anyhow::anyhow!("config.json has no `dflash_config` — not a DFlash2 draft model"))?;
        let gi = |k: &str, def: i64| d.get(k).and_then(|v| v.as_i64()).unwrap_or(def);
        let need = |k: &str| -> Result<i64> {
            d.get(k)
                .and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("dflash_config.{k} missing"))
        };
        Ok(Self {
            block_size: gi("block_size", 8),
            conv_kernel_size: gi("conv_kernel_size", 2),
            conv_group_size: gi("conv_group_size", 16),
            mask_token_id: need("mask_token_id")?,
            selector_rank: need("selector_rank")?,
            selector_top_k: need("selector_top_k")?,
            target_layer_ids: d
                .get("target_layer_ids")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default(),
            sliding_window: raw.get("sliding_window").and_then(|v| v.as_i64()).unwrap_or(2048),
        })
    }
}

/// RMSNorm over a bf16 residual stream. `ext.rms_norm` is fp16-only and the
/// stream here overflows fp16, so this computes in fp32 and emits bf16.
struct BfRmsNorm {
    weight: Tensor, // [H], f32
    eps: f64,
}

impl BfRmsNorm {
    fn load(stc: &SafeTensors, key: &str, eps: f32, device: Device) -> Result<Self> {
        let w = stc.get(&format!("{key}.weight"), device, false, true)?;
        Ok(Self { weight: w.to_kind(Kind::Float), eps: eps as f64 })
    }

    /// `[.., H]` any float dtype -> `[.., H]` in `out`.
    fn forward(&self, x: &Tensor, out: Kind) -> Tensor {
        let xf = x.to_kind(Kind::Float);
        let var = xf.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float);
        (xf * (var + self.eps).rsqrt() * &self.weight).to_kind(out)
    }
}

/// Two-tap grouped dynamic causal convolution (`GroupedDynamicCausalConv`).
///
/// `output[t] = Σ_tap (base[tap] + dyn[tap][t]) ⊙ x[t - tap]`, causal, tap 0 =
/// current position. The dynamic half is predicted per position by
/// `kernel_projection`; `base_kernel[0]` is used before the sub-op (`prepare`)
/// and `base_kernel[1]` after it (`finish`), which is what lets one conv
/// straddle the attention/MLP it wraps.
struct DynConv {
    /// `[2, kernel_size, H]` — prepare base, finish base. f32 for the math.
    base_kernel: Tensor,
    /// `H -> 2 * kernel_size * groups`, plain fp16 in the checkpoint.
    proj: Linear,
    kernel_size: i64,
    group_size: i64,
    groups: i64,
}

impl DynConv {
    fn load(stc: &SafeTensors, key: &str, p: &DFlash2Params, hidden: i64, device: Device) -> Result<Self> {
        let groups = hidden / p.conv_group_size;
        Ok(Self {
            base_kernel: stc
                .get(&format!("{key}.base_kernel"), device, false, true)?
                .to_kind(Kind::Float),
            proj: Linear::load(
                stc,
                &format!("{key}.kernel_projection"),
                None,
                hidden,
                2 * p.conv_kernel_size * groups,
                device,
                true,
                0.0,
            )?,
            kernel_size: p.conv_kernel_size,
            group_size: p.conv_group_size,
            groups,
        })
    }

    /// Shared convolution kernel. `hidden` `[b,l,H]`, `dynamic`
    /// `[b,l,taps,groups]`, `base` `[taps,H]`; math in f32, result cast to `out`.
    fn convolve(&self, hidden: &Tensor, dynamic: &Tensor, base: &Tensor, out: Kind) -> Tensor {
        let (b, l, h) = {
            let s = hidden.size();
            (s[0], s[1], s[2])
        };
        let blocks = hidden.to_kind(Kind::Float).reshape([b, l, self.groups, self.group_size]);
        let dynamic = dynamic
            .to_kind(Kind::Float)
            .reshape([b, l, self.kernel_size, self.groups, 1]);
        let mut acc = Tensor::zeros_like(&blocks);
        for tap in 0..self.kernel_size {
            // causal shift: tap `t` reads position `i - t`, zero-padded at the left
            let values = if tap == 0 {
                blocks.shallow_clone()
            } else {
                let keep = blocks.narrow(1, 0, l - tap);
                Tensor::cat(
                    &[Tensor::zeros([b, tap, self.groups, self.group_size], (Kind::Float, hidden.device())), keep],
                    1,
                )
            };
            let kernel = base.select(0, tap).reshape([1, 1, self.groups, self.group_size]);
            acc = acc + kernel * &values;
            acc = acc.addcmul(&dynamic.select(2, tap), &values);
        }
        acc.reshape([b, l, h]).to_kind(out)
    }

    /// Pre-sub-op half. `x` is post-norm (bounded), so fp16 is safe here.
    /// Returns `(convolved, dynamic-half saved for `finish`)`.
    fn prepare(&self, x: &Tensor) -> (Tensor, Tensor) {
        let x = x.to_kind(Kind::Half);
        let dyn_ = self.proj.forward(&x);
        let s = x.size();
        let dyn_ = dyn_.reshape([s[0], s[1], 2, self.kernel_size, self.groups]);
        let y = self.convolve(&x, &dyn_.select(2, 0), &self.base_kernel.select(0, 0), Kind::Half);
        (y, dyn_.select(2, 1))
    }

    /// Post-sub-op half, on the residual-magnitude stream — bf16.
    fn finish(&self, x: &Tensor, dynamic: &Tensor) -> Tensor {
        self.convolve(x, dynamic, &self.base_kernel.select(0, 1), Kind::BFloat16)
    }
}

/// One conv-wrapped decoder layer:
/// `x += attn_conv.finish(attn(attn_conv.prepare(norm(x))))`, likewise for MLP.
struct DFlash2Block {
    attn: Attention,
    mlp: GatedMlp,
    attn_norm: BfRmsNorm,
    mlp_norm: BfRmsNorm,
    attn_conv: DynConv,
    mlp_conv: DynConv,
}

impl DFlash2Block {
    fn load(
        stc: &SafeTensors,
        idx: i64,
        cfg: &Config,
        p: &DFlash2Params,
        device: Device,
    ) -> Result<Self> {
        let key = format!("layers.{idx}");
        Ok(Self {
            attn: Attention::load(stc, &format!("{key}.self_attn"), cfg, device)?,
            mlp: GatedMlp::load(stc, &format!("{key}.mlp"), cfg, device)?,
            attn_norm: BfRmsNorm::load(stc, &format!("{key}.input_layernorm"), cfg.rms_norm_eps, device)?,
            mlp_norm: BfRmsNorm::load(stc, &format!("{key}.post_attention_layernorm"), cfg.rms_norm_eps, device)?,
            attn_conv: DynConv::load(stc, &format!("{key}.attention_conv"), p, cfg.hidden_size, device)?,
            mlp_conv: DynConv::load(stc, &format!("{key}.mlp_conv"), p, cfg.hidden_size, device)?,
        })
    }
}

/// Top-k candidate selector (`CandidateSelector`).
///
/// Replaces per-row argmax. For each block row it takes the draft logits'
/// top-`k` candidates, then walks the block greedily scoring each candidate
/// against the previously chosen token:
///
/// ```text
///   S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>
/// ```
///
/// with `A`/`B` the predecessor/successor codebooks and `H` a rank-`r`
/// projection of the draft state. This is what turns `block_size` independent
/// per-position guesses into one coherent sequence, for two embedding lookups
/// and a dot product per row.
struct Selector {
    hidden_proj: Linear,
    pred_codebook: Tensor, // [vocab, rank] f32
    succ_codebook: Tensor, // [vocab, rank] f32
    top_k: i64,
}

impl Selector {
    fn load(stc: &SafeTensors, cfg: &Config, p: &DFlash2Params, device: Device) -> Result<Self> {
        Ok(Self {
            hidden_proj: Linear::load(
                stc,
                "candidate_selector.hidden_projection",
                None,
                cfg.hidden_size,
                p.selector_rank,
                device,
                true,
                0.0,
            )?,
            pred_codebook: stc
                .get("candidate_selector.predecessor_codebook", device, false, true)?
                .to_kind(Kind::Float),
            succ_codebook: stc
                .get("candidate_selector.successor_codebook", device, false, true)?
                .to_kind(Kind::Float),
            top_k: p.selector_top_k,
        })
    }

    /// Greedy walk. `hidden` `[b, rows, H]` post-norm draft state, `logits`
    /// `[b, rows, V]` f32, `anchor` `[b]` the token the block follows.
    /// Returns `[b, rows]` token ids.
    ///
    /// Greedy at every temperature: the proposal is verified by accept-while-
    /// match against the target's own samples, so the path only has to be a
    /// good guess — it can never move the output distribution.
    fn walk(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchor: &Tensor,
        id_map: Option<&Tensor>,
    ) -> Tensor {
        let rows = logits.size()[1];
        let (unary, cands) = logits.topk(self.top_k, -1, true, false);
        // A pruned draft head argmaxes over compact indices; the codebooks are
        // indexed by real token id, so translate before they are used.
        let cands = match id_map {
            None => cands,
            Some(m) => {
                let sh = cands.size();
                m.index_select(0, &cands.reshape([-1])).reshape(&sh[..])
            }
        };
        let unary = unary.to_kind(Kind::Float);
        let gate = self
            .hidden_proj
            .forward(&hidden.to_kind(Kind::Half))
            .to_kind(Kind::Float); // [b, rows, rank]

        let mut pred = anchor.to_kind(Kind::Int64);
        let mut path = Vec::with_capacity(rows as usize);
        for i in 0..rows {
            let a = self.pred_codebook.index_select(0, &pred); // [b, rank]
            let ci = cands.select(1, i); // [b, k]
            let b_emb = self
                .succ_codebook
                .index_select(0, &ci.reshape([-1]))
                .reshape([ci.size()[0], self.top_k, -1]); // [b, k, rank]
            let ag = (a * gate.select(1, i)).unsqueeze(1); // [b, 1, rank]
            let scores = unary.select(1, i) + (ag * b_emb).sum_dim_intlist(-1, false, Kind::Float);
            let idx = scores.argmax(-1, false).unsqueeze(-1); // [b, 1]
            pred = ci.gather(-1, &idx, false).squeeze_dim(-1);
            path.push(pred.shallow_clone());
        }
        Tensor::stack(&path, 1)
    }
}

/// The loaded DFlash2 drafter.
pub struct DFlash2Model {
    pub cfg: Config,
    pub params: DFlash2Params,
    /// `concat(taps) -> hidden`; `in = hidden × |target_layer_ids|`.
    fc: Linear,
    hidden_norm: BfRmsNorm,
    blocks: Vec<DFlash2Block>,
    norm: BfRmsNorm,
    selector: Selector,
    pub device: Device,
}

impl DFlash2Model {
    pub fn load(dir: &std::path::Path, device: Device) -> Result<Self> {
        // Name the directory in every failure. A bare `No such file or
        // directory` from here is indistinguishable from the target model
        // failing to load, and says nothing about which path was wrong.
        let ctx = || format!("loading DFlash2 drafter from {}", dir.display());
        let cfg = Config::from_dir(dir).with_context(ctx)?;
        let params = DFlash2Params::from_config(&cfg.raw).with_context(ctx)?;
        let stc = SafeTensors::open(dir, &[]).with_context(ctx)?;
        let taps = params.target_layer_ids.len() as i64;
        anyhow::ensure!(taps > 0, "dflash_config.target_layer_ids is empty");

        let fc = Linear::load(&stc, "fc", None, cfg.hidden_size * taps, cfg.hidden_size, device, true, 0.0)?;
        let hidden_norm = BfRmsNorm::load(&stc, "hidden_norm", cfg.rms_norm_eps, device)?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| DFlash2Block::load(&stc, i, &cfg, &params, device))
            .collect::<Result<Vec<_>>>()?;
        let norm = BfRmsNorm::load(&stc, "norm", cfg.rms_norm_eps, device)?;
        let selector = Selector::load(&stc, &cfg, &params, device)?;

        Ok(Self { cfg, params, fc, hidden_norm, blocks, norm, selector, device })
    }

    /// Project the target's tapped hidden states into the drafter's stream.
    /// `taps` are `[b, l, H]` per tapped layer, in `target_layer_ids` order;
    /// returns `[b, l, H]` fp16 — the input to every layer's `k_proj`/`v_proj`
    /// when filling the draft KV cache.
    pub fn project_taps(&self, taps: &[Tensor]) -> Tensor {
        let cat = Tensor::cat(taps, -1);
        let x = self.fc.forward(&cat.to_kind(Kind::Half));
        self.hidden_norm.forward(&x, Kind::Half)
    }

    pub fn num_layers(&self) -> i64 {
        self.cfg.num_hidden_layers
    }
}

/// Rolling context K/V for the drafter's 5 sliding-window layers.
///
/// Sliding attention caps what a block row can see at `sliding_window`, so this
/// only ever has to retain that many positions — the draft cache is O(window),
/// not O(context), no matter how long the conversation gets.
pub struct DFlash2Cache {
    /// per layer, `[1, cap, kv_heads, head_dim]` fp16
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    /// valid entries, occupying slots `0..len`
    len: i64,
    /// absolute position of slot 0
    start: i64,
    cap: i64,
}

impl DFlash2Cache {
    pub fn new(m: &DFlash2Model, device: Device) -> Self {
        // window + slack, so the compaction below runs rarely rather than once
        // per round
        let cap = m.params.sliding_window + 512;
        let (kvh, hd) = (m.cfg.num_kv_heads, m.cfg.head_dim);
        let mk = || Tensor::zeros([1, cap, kvh, hd], (Kind::Half, device));
        Self {
            k: (0..m.num_layers()).map(|_| mk()).collect(),
            v: (0..m.num_layers()).map(|_| mk()).collect(),
            len: 0,
            start: 0,
            cap,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.start = 0;
    }

    /// Valid entries currently held.
    pub fn len(&self) -> i64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slot capacity: `sliding_window + 512`.
    pub fn cap(&self) -> i64 {
        self.cap
    }

    /// Absolute position just past the last cached entry.
    pub fn end_pos(&self) -> i64 {
        self.start + self.len
    }

    /// Make room for `n` more entries, dropping the oldest in bulk.
    ///
    /// `n` is a whole prefill chunk, not one token, so evicting down to
    /// `window` is not enough on its own: the slack above the window is only
    /// `cap - window` (512), and a chunk larger than that would still overrun
    /// the buffer. Bounding `keep` by `cap - n` as well keeps the invariant
    /// `keep + n <= cap` for any chunk size. Callers must still clamp `n` to
    /// `cap` before calling — see `update_kv_from_target`.
    fn ensure_room(&mut self, n: i64, window: i64) {
        if self.len + n <= self.cap {
            return;
        }
        // keep only what the window can still reach, and only what leaves room
        let keep = window.min(self.len).min((self.cap - n).max(0));
        let drop = self.len - keep;
        if drop > 0 {
            for t in self.k.iter_mut().chain(self.v.iter_mut()) {
                let src = t.narrow(1, drop, keep).copy();
                t.narrow(1, 0, keep).copy_(&src);
            }
            self.len = keep;
            self.start += drop;
        }
    }
}

impl DFlash2Model {
    /// DFlash2's `update_kv_from_target`: fold `n` new target positions into the
    /// draft cache. `taps` are the target's hidden states at
    /// `target_layer_ids`, each `[1, n, H]`, for absolute positions
    /// `base .. base + n`.
    ///
    /// This is the whole trick: the drafter never runs over the context, it just
    /// re-projects what the target already computed.
    pub fn update_kv_from_target(&self, cache: &mut DFlash2Cache, taps: &[Tensor], base: i64) {
        let n = taps[0].size()[1];
        if n == 0 {
            return;
        }
        debug_assert_eq!(taps.len(), self.params.target_layer_ids.len());
        // A chunk bigger than the whole buffer can only be stored in part, and
        // the part that matters is the tail — attention reaches back at most
        // `sliding_window` from the newest position. Drop the head and advance
        // `base` with it so the RoPE positions still describe what is stored.
        let sliced: Option<Vec<Tensor>> = (n > cache.cap).then(|| {
            let off = n - cache.cap;
            taps.iter().map(|t| t.narrow(1, off, cache.cap)).collect()
        });
        let (base, n) = match &sliced {
            Some(_) => (base + (n - cache.cap), cache.cap),
            None => (base, n),
        };
        let taps: &[Tensor] = sliced.as_deref().unwrap_or(taps);
        cache.ensure_room(n, self.params.sliding_window);
        if cache.len == 0 {
            cache.start = base;
        }
        let stream = self.project_taps(taps); // [1, n, H] fp16
        let positions = Tensor::from_slice(&[base as i32]).to_device(self.device);
        let at = cache.len;
        for (i, blk) in self.blocks.iter().enumerate() {
            let (k, v) = blk.attn.kv_from_hidden(&stream, &positions);
            cache.k[i].narrow(1, at, n).copy_(&k);
            cache.v[i].narrow(1, at, n).copy_(&v);
        }
        cache.len += n;
    }

    /// One draft block. `anchor` is the token the block follows (already
    /// committed); `embed` supplies the target's token embeddings.
    ///
    /// Returns the post-norm block state `[1, block_size, H]`. Row 0 is the
    /// anchor; rows `1..` are the mask positions that predict themselves.
    pub fn block_state(&self, embed: &Tensor, cache: &DFlash2Cache) -> Tensor {
        let base_pos = cache.end_pos();
        let mut x = embed.to_kind(Kind::BFloat16);
        for (i, blk) in self.blocks.iter().enumerate() {
            let y = blk.attn_norm.forward(&x, Kind::Half);
            let (y, kern) = blk.attn_conv.prepare(&y);
            let y = blk.attn.forward_block_windowed(
                &y,
                &cache.k[i].narrow(1, 0, cache.len),
                &cache.v[i].narrow(1, 0, cache.len),
                cache.start,
                base_pos,
                self.params.sliding_window,
            );
            x = x + blk.attn_conv.finish(&y, &kern);

            let y = blk.mlp_norm.forward(&x, Kind::Half);
            let (y, kern) = blk.mlp_conv.prepare(&y);
            let y = blk.mlp.forward(&y);
            x = x + blk.mlp_conv.finish(&y, &kern);
        }
        self.norm.forward(&x, Kind::BFloat16)
    }
}

impl DFlash2Model {
    /// Draft `block_size - 1` tokens following `anchor`.
    ///
    /// One draft forward + one target `lm_head` pass covers the whole block —
    /// where the MTP drafter pays a full `lm_head` stream *per token*. The
    /// selector then walks a coherent path through each row's top-`k`
    /// candidates, which is what keeps the block from being `n` independent
    /// (and mutually inconsistent) guesses.
    ///
    /// The proposal is only ever a guess: it is verified against the target by
    /// the usual accept-while-match rule, so nothing here can change the output.
    /// That is also why it is safe to read the block logits off a
    /// vocabulary-pruned draft head.
    pub fn draft(
        &self,
        target: &crate::model::Model,
        cache: &DFlash2Cache,
        anchor: i64,
    ) -> Vec<i64> {
        let _no_grad = tch::no_grad_guard();
        let bs = self.params.block_size;
        // [anchor, MASK, MASK, …] embedded with the *target's* table
        let mut ids = vec![anchor];
        ids.extend(std::iter::repeat(self.params.mask_token_id).take((bs - 1) as usize));
        let ids = Tensor::from_slice(&ids).reshape([1, bs]).to_device(self.device);
        let embed = target.embed_tokens(&ids);

        let state = self.block_state(&embed, cache); // [1, bs, H]
        let mut v = self.paths_from_states(target, &state, &[anchor]);
        v.pop().unwrap_or_default()
    }

    /// The embedded `[anchor, MASK…]` block input for a row.
    pub fn block_input(&self, target: &crate::model::Model, anchor: i64) -> Tensor {
        let bs = self.params.block_size;
        let mut ids = vec![anchor];
        ids.extend(std::iter::repeat(self.params.mask_token_id).take((bs - 1) as usize));
        let ids = Tensor::from_slice(&ids).reshape([1, bs]).to_device(self.device);
        target.embed_tokens(&ids)
    }

    /// Turn per-row block states into per-row draft paths.
    ///
    /// `states` is `[bsz, block_size, H]` — the rows' post-norm draft states,
    /// stacked. Deliberately batched across rows: the block layers are small,
    /// but `lm_head` is the 1.27 GB read, and running it once for the whole
    /// batch is what keeps a round's cost flat in both draft length *and* batch
    /// size.
    pub fn paths_from_states(
        &self,
        target: &crate::model::Model,
        states: &Tensor,
        anchors: &[i64],
    ) -> Vec<Vec<i64>> {
        let _no_grad = tch::no_grad_guard();
        let bs = self.params.block_size;
        // rows 1.. predict their own position; row 0 is the anchor
        let rows = states.narrow(1, 1, bs - 1).contiguous();
        let (logits, id_map) = target.draft_logits_on(&rows);
        let logits = logits.to_kind(Kind::Float);
        let anchor_t = Tensor::from_slice(anchors).to_device(self.device);
        let path = self.selector.walk(&rows, &logits, &anchor_t, id_map);
        let host = path.to_kind(Kind::Int64).to_device(Device::Cpu);
        (0..anchors.len() as i64)
            .map(|r| (0..bs - 1).map(|c| host.int64_value(&[r, c])).collect())
            .collect()
    }
}

/// The DFlash2 drafter plus one rolling context cache per generator slot.
///
/// Each slot's cache is `O(sliding_window)`, not `O(context)`, and is allocated
/// on first use so idle slots cost nothing.
pub struct DFlash2Batched {
    pub model: DFlash2Model,
    caches: Vec<Option<DFlash2Cache>>,
    device: Device,
}

impl DFlash2Batched {
    /// Allocates every slot's window cache up front. Lazy allocation would hide
    /// this cost from the KV-pool sizing that runs right after load, and the
    /// memory would then come due on the first request instead — an OOM at
    /// serve time rather than a smaller pool at start time.
    pub fn new(model: DFlash2Model, max_slots: usize, device: Device) -> Self {
        let caches = (0..max_slots.max(1))
            .map(|_| Some(DFlash2Cache::new(&model, device)))
            .collect();
        Self { model, caches, device }
    }

    /// Per-slot cache, allocated on first touch.
    pub fn cache_mut(&mut self, slot: usize) -> &mut DFlash2Cache {
        if slot >= self.caches.len() {
            self.caches.resize_with(slot + 1, || None);
        }
        if self.caches[slot].is_none() {
            self.caches[slot] = Some(DFlash2Cache::new(&self.model, self.device));
        }
        self.caches[slot].as_mut().unwrap()
    }

    /// Fold `n` new target positions into one slot's cache. A method rather
    /// than `cache_mut` + a call on `model`, because those two borrows conflict.
    pub fn ingest(&mut self, slot: usize, taps: &[Tensor], base: i64) {
        if slot >= self.caches.len() {
            self.caches.resize_with(slot + 1, || None);
        }
        if self.caches[slot].is_none() {
            self.caches[slot] = Some(DFlash2Cache::new(&self.model, self.device));
        }
        let c = self.caches[slot].as_mut().unwrap();
        self.model.update_kv_from_target(c, taps, base);
    }

    pub fn cache(&self, slot: usize) -> Option<&DFlash2Cache> {
        self.caches.get(slot).and_then(|c| c.as_ref())
    }

    /// Hand a finished job's cache back for reuse. The buffer stays allocated —
    /// it was counted at startup and the next job on this slot needs it again.
    pub fn release(&mut self, slot: usize) {
        if let Some(Some(c)) = self.caches.get_mut(slot) {
            c.reset();
        }
    }

    /// Bytes one slot's window cache occupies — the drafter's real per-job cost.
    pub fn bytes_per_slot(&self) -> i64 {
        let p = &self.model.params;
        let c = &self.model.cfg;
        // k+v, fp16, over (window + slack) positions, per layer
        2 * 2 * (p.sliding_window + 512) * c.num_kv_heads * c.head_dim * c.num_hidden_layers
    }
}
