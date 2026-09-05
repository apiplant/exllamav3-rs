//! Qwen3.5 MTP (multi-token-prediction) draft head — port of
//! `architecture/qwen3_5_mtp.py` + `modules/arch_specific/qwen3_5_mtp.py`
//! (grade C: single-sequence greedy self-speculation for `bin/infer`,
//! draft length 1).
//!
//! The MTP head is a tiny second model that shares the trunk's `embed_tokens`
//! and `lm_head`. Given the trunk's post-final-norm hidden state `hₚ` at
//! position `p` and the token `tₚ` fed there, it predicts `tₚ₊₁`:
//!
//! ```text
//!   e  = pre_fc_norm_embedding( embed(tₚ) )        # RMSNorm(bias 1) -> fp16
//!   h  = pre_fc_norm_hidden( hₚ )                   # RMSNorm(bias 1) -> fp16
//!   x  = fc( cat(e, h) )                            # Linear 2H -> H
//!   x  = x + self_attn( input_layernorm(x) )       # 1 gated GQA layer, own KV
//!   x  = x + mlp( post_attention_layernorm(x) )
//!   g  = mtp.norm(x)                                # RMSNorm(bias 1) -> fp16
//!   logits = trunk.lm_head(g)
//! ```
//!
//! `bin/infer` runs one MTP step per committed token to draft its successor,
//! then verifies with a `q_len = 2` trunk forward (accept-longest-prefix, same
//! rule as the n-gram path). Best case: 2 tokens per trunk forward.

use crate::cache::PAGE_SIZE;
use crate::config::Config;
use crate::model::Model;
use crate::modules::{Attention, Attn, GatedMlp, Linear, RmsNorm};
use crate::safetensors::SafeTensors;
use anyhow::Result;
use tch::{Device, Kind, Tensor};

/// Single-sequence paged KV cache for the MTP head's lone attention layer.
struct MtpKvCache {
    k: Tensor,
    v: Tensor,
    block_table: Tensor,
    seqlens: Tensor,
}

impl MtpKvCache {
    fn new(cfg: &Config, max_len: i64, device: Device) -> Self {
        let num_pages = (max_len + PAGE_SIZE - 1) / PAGE_SIZE;
        let nkv = cfg.kv_heads_eff().0;
        let hd = cfg.head_dim;
        Self {
            k: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
            v: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
            block_table: Tensor::arange(num_pages, (Kind::Int, device)).reshape([1, num_pages]),
            seqlens: Tensor::zeros([1], (Kind::Int, device)),
        }
    }
    fn set_seqlen(&self, n: i64) {
        let src = Tensor::from_slice(&[n as i32]).to_device(self.seqlens.device());
        let _ = self.seqlens.shallow_clone().copy_(&src);
    }
    fn get_seqlen(&self) -> i64 {
        self.seqlens.int64_value(&[0])
    }
    fn advance(&self, n: i64) {
        let _ = self.seqlens.add_scalar_out(&self.seqlens, n);
    }
}

pub struct MtpModel {
    pre_fc_norm_hidden: RmsNorm,
    pre_fc_norm_embedding: RmsNorm,
    fc: Linear,
    input_norm: RmsNorm,
    attn: Attention,
    post_norm: RmsNorm,
    mlp: GatedMlp,
    final_norm: RmsNorm,
    /// single-sequence KV for `bin/infer --mtp` (`None` for the batched generator
    /// path, which supplies its own `MtpBatchedCache`).
    kv: Option<MtpKvCache>,
}

/// Shared page pool for the batched MTP head's lone attention layer, addressed
/// with the job's own `block` table + a per-row seqlen.
///
/// Default storage is **Q8** (packed int32 codes + fp16 scales, `Attn::PagedQuant`
/// online dequant): ~half the VRAM of an fp16 store, sized to the whole page
/// pool. Q8 is near-lossless and the MTP alignment (`prime_row` / `draft_n`
/// pairing) is what governs acceptance, not the draft KV precision. Set
/// `EXL3_MTP_KV_FP16=1` for the fp16 store (`Attn::Paged`) if you have the VRAM
/// headroom and want to rule the quant path out.
pub enum MtpBatchedCache {
    Fp16 { k: Tensor, v: Tensor },
    Q8 { qk: Tensor, qv: Tensor, sk: Tensor, sv: Tensor },
}

impl MtpBatchedCache {
    pub const BITS: i64 = 8;

    pub fn new(cfg: &Config, num_pages: i64, device: Device) -> Self {
        let nkv = cfg.kv_heads_eff().0;
        let hd = cfg.head_dim;
        assert!((nkv * hd) % 32 == 0, "MTP n_kv*head_dim must be a multiple of 32");
        if std::env::var("EXL3_MTP_KV_FP16").ok().as_deref() == Some("1") {
            return Self::Fp16 {
                k: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
                v: Tensor::zeros([num_pages, PAGE_SIZE, nkv, hd], (Kind::Half, device)),
            };
        }
        let groups = nkv * hd / 32;
        let b = Self::BITS;
        Self::Q8 {
            qk: Tensor::zeros([num_pages, PAGE_SIZE, groups * b], (Kind::Int, device)),
            qv: Tensor::zeros([num_pages, PAGE_SIZE, groups * b], (Kind::Int, device)),
            sk: Tensor::zeros([num_pages, PAGE_SIZE, groups], (Kind::Half, device)),
            sv: Tensor::zeros([num_pages, PAGE_SIZE, groups], (Kind::Half, device)),
        }
    }
}

/// The MTP head plus its batched KV pool, held by `Speculator::Mtp`.
pub struct MtpBatched {
    pub model: MtpModel,
    pub cache: MtpBatchedCache,
}

impl MtpModel {
    pub fn load(dir: &std::path::Path, device: Device, trunk: &Model) -> Result<Self> {
        let cfg = &trunk.config;
        let stc = SafeTensors::open(dir, &[])?;
        let eps = cfg.rms_norm_eps;
        let nb = 1.0_f32; // MTP norms all carry constant_bias 1.0

        if !stc.has("mtp.fc.trellis") && !stc.has("mtp.fc.weight") {
            anyhow::bail!("checkpoint has no `mtp.*` tensors — this model has no MTP head");
        }
        let h = cfg.hidden_size;
        Ok(Self {
            pre_fc_norm_hidden: RmsNorm::load_biased(&stc, "mtp.pre_fc_norm_hidden", eps, nb, device)?,
            pre_fc_norm_embedding: RmsNorm::load_biased(
                &stc,
                "mtp.pre_fc_norm_embedding",
                eps,
                nb,
                device,
            )?,
            fc: Linear::load(&stc, "mtp.fc", None, h * 2, h, device, true, 0.0)?,
            input_norm: RmsNorm::load_biased(&stc, "mtp.layers.0.input_layernorm", eps, nb, device)?,
            attn: Attention::load(&stc, "mtp.layers.0.self_attn", cfg, device)?,
            post_norm: RmsNorm::load_biased(
                &stc,
                "mtp.layers.0.post_attention_layernorm",
                eps,
                nb,
                device,
            )?,
            mlp: GatedMlp::load(&stc, "mtp.layers.0.mlp", cfg, device)?,
            final_norm: RmsNorm::load_biased(&stc, "mtp.norm", eps, nb, device)?,
            kv: Some(MtpKvCache::new(
                cfg,
                trunk.config.max_position_embeddings.max(4096),
                device,
            )),
        })
    }

    /// Load the MTP head without the single-sequence KV pool (the batched
    /// generator supplies an `MtpBatchedCache` instead).
    pub fn load_headless(dir: &std::path::Path, device: Device, trunk: &Model) -> Result<Self> {
        let mut m = Self::load(dir, device, trunk)?;
        m.kv = None;
        Ok(m)
    }

    // ---- batched path (dynamic generator) ---------------------------------

    /// One batched MTP forward over `[bsz, 1]` tokens against `cache`, addressed
    /// by `block_table` `[bsz, P]` / `seqlens` `[bsz]` i32 (the paged attn kernel
    /// appends one K/V entry per row at `seqlens[row]`). Does **not** advance
    /// `seqlens` — the caller rebuilds it. Returns
    /// `(mtp_hidden [bsz,1,h] fp16, logits [bsz,vocab] f32 | None)`.
    #[allow(clippy::too_many_arguments)]
    fn bstep(
        &self,
        trunk: &Model,
        cache: &MtpBatchedCache,
        trunk_hidden: &Tensor,
        ids: &Tensor,
        block_table: &Tensor,
        seqlens: &Tensor,
        want_logits: bool,
    ) -> (Tensor, Option<Tensor>) {
        let _no_grad = tch::no_grad_guard();
        let emb = trunk.embed_tokens(ids); // [bsz,1,h]
        let e = self.pre_fc_norm_embedding.forward(&emb);
        let hh = self.pre_fc_norm_hidden.forward(trunk_hidden);
        let x = Tensor::cat(&[e, hh], -1); // [bsz,1,2h]
        let x = self.fc.forward(&x); // [bsz,1,h]
        let a_in = self.input_norm.forward(&x);
        let attn_spec = match cache {
            MtpBatchedCache::Fp16 { k, v } => Attn::Paged {
                k_cache: k,
                v_cache: v,
                block_table,
                seqlens,
                rope_table: None,
            },
            MtpBatchedCache::Q8 { qk, qv, sk, sv } => Attn::PagedQuant {
                qk,
                qv,
                sk,
                sv,
                k_scratch: None, // compact per-call scratch
                v_scratch: None,
                block_table,
                seqlens,
                compand_a: 0.0,
                rope_table: None,
            },
        };
        let a_out = self.attn.forward(&a_in, &attn_spec);
        let x2 = &x + a_out;
        let m_in = self.post_norm.forward(&x2);
        let m_out = self.mlp.forward(&m_in);
        let x3 = x2 + m_out;
        let g = self.final_norm.forward(&x3); // [bsz,1,h] fp16
        let logits = want_logits.then(|| self.head_logits(trunk, &g));
        (g, logits)
    }

    /// Per-row prompt teacher-forcing (bsz 1): write MTP KV for positions
    /// `base .. base + toks.len()`. MTP KV position `p` is fed
    /// `(token[p], trunk_hidden[p-1])` — the DeepSeek-V3 MTP alignment the head
    /// was trained with (`shifted_hidden` in `job.py`): it predicts `token[p+1]`
    /// from the token *at* `p` and the trunk state from *before* it. `hiddens`
    /// `[1, toks.len(), h]` is the trunk hidden for `toks` (index 0 == position
    /// `base`); position `base` uses `carry` — the trunk hidden at `base - 1` —
    /// or a zero hidden when `carry` is `None` (Python's `carry_hidden`
    /// fallback), exact for `base == 0` and a negligible one-slot approximation
    /// on a prefix-cache resume. A caller priming a long prompt one chunk at a
    /// time passes the previous chunk's last hidden as `carry`, which keeps the
    /// prime exact across chunk boundaries and its working set O(chunk).
    /// `block_row` `[1, P]` maps the sequence's pages.
    #[allow(clippy::too_many_arguments)]
    pub fn prime_row(
        &self,
        trunk: &Model,
        cache: &MtpBatchedCache,
        hiddens: &Tensor,
        toks: &[i64],
        base: i64,
        carry: Option<&Tensor>,
        block_row: &Tensor,
    ) {
        let dev = hiddens.device();
        let n = toks.len() as i64;
        if n == 0 {
            return;
        }
        let hdim = hiddens.size()[2];
        // shifted: MTP KV position `base+i` sees `trunk_hidden[base+i-1]`
        // (`shifted_hidden` in `job.py`); position `base` uses `carry`.
        let lead = match carry {
            Some(c) => c.to_kind(hiddens.kind()).reshape([1, 1, hdim]),
            None => Tensor::zeros([1, 1, hdim], (hiddens.kind(), dev)),
        };
        let shifted = if n > 1 {
            Tensor::cat(&[lead, hiddens.narrow(1, 0, n - 1)], 1)
        } else {
            lead
        }; // [1, n, h]
        let ids = Tensor::from_slice(toks).reshape([1, n]).to_device(dev);
        let sl = Tensor::from_slice(&[base as i32]).to_device(dev);
        // one prefill forward populates MTP KV for [base, base+n) at once
        let _ = self.bstep(trunk, cache, &shifted, &ids, block_row, &sl, false);
    }

    /// Draft `n` tokens per row following each row's position `q[r]`. MTP KV
    /// `[0, q[r])` is already resident (primed / synced from earlier rounds).
    /// Step 0 writes MTP KV `@ q[r]` from `(c[r], h_prev[r])` where `h_prev` is
    /// the *real* trunk hidden at `q[r]-1` (DeepSeek MTP alignment); the chain
    /// then feeds the head its own output state, as upstream does. Returns
    /// `([bsz, window] i64 on device, per-row confidences)`.
    ///
    /// With `cal` set, the block is cut short at the first position where the
    /// running estimate of "this whole block gets accepted" drops below the
    /// calibrator's target (upstream's `draft_confidence`, default 0.4). Each
    /// position skipped saves a whole `lm_head` pass — the single most expensive
    /// thing in a draft step — and narrows the verify forward with it. `window`
    /// is shared across the batch, matching upstream, so the verify stays
    /// rectangular.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_n_batched(
        &self,
        trunk: &Model,
        cache: &MtpBatchedCache,
        h_prev: &Tensor, // [bsz,1,h] — trunk hidden at q-1
        c: &[i64],
        q: &[i64],
        block_table: &Tensor, // [bsz,P]
        n: i64,
        cal: Option<&crate::draft_conf::DraftConfidence>,
    ) -> (Tensor, Vec<Vec<f32>>) {
        let dev = h_prev.device();
        let bsz = c.len() as i64;
        // All n position vectors in ONE host->device copy. Building them per step
        // put a pageable H2D transfer inside the draft loop, which serializes
        // against the work already queued; the loop now just narrows a row.
        let sl_all = {
            let mut v: Vec<i32> = Vec::with_capacity((n * bsz) as usize);
            for off in 0..n {
                v.extend(q.iter().map(|&qi| (qi + off) as i32));
            }
            Tensor::from_slice(&v).reshape([n, bsz]).to_device(dev)
        };
        let sl = |off: i64| -> Tensor { sl_all.select(0, off) };
        let col = |xs: &[i64]| Tensor::from_slice(xs).reshape([bsz, 1]).to_device(dev);

        // `EXL3_MEM_DEBUG=1`: split a draft step into the MTP layer forward and
        // the lm_head, which is the same weight read as the trunk's (636 MB at
        // 4 bits) and therefore the floor for a draft step.
        let dbg = std::env::var("EXL3_MEM_DEBUG").is_ok();
        let mut t_layer = 0f64;
        let mut t_head = 0f64;

        let mut drafts: Vec<Tensor> = Vec::with_capacity(n as usize);
        let mut confs: Vec<Vec<f32>> = vec![Vec::new(); bsz as usize];
        let mut reach = vec![1.0f32; bsz as usize];

        let mut h = h_prev.shallow_clone();
        let mut tok = col(c);
        for i in 0..n {
            let t0 = std::time::Instant::now();
            let nh = if i == 0 {
                self.bstep_h(trunk, cache, h_prev, &tok, block_table, &sl(0))
            } else {
                self.bstep_h(trunk, cache, &h, &tok, block_table, &sl(i))
            };
            if dbg {
                tch::Cuda::synchronize(0);
                t_layer += t0.elapsed().as_secs_f64() * 1000.0;
            }
            let t1 = std::time::Instant::now();
            // Draft-only head: may be vocabulary-pruned, in which case its
            // argmax is a compact index and `id_map` turns it back into a token.
            let (logits, id_map) = trunk.draft_logits_on(&nh);
            let logits = logits.squeeze_dim(1);
            if dbg {
                tch::Cuda::synchronize(0);
                t_head += t1.elapsed().as_secs_f64() * 1000.0;
            }
            let unmap = |ids: Tensor| match id_map {
                None => ids.reshape([bsz, 1]),
                Some(m) => m.index_select(0, &ids.reshape([-1])).reshape([bsz, 1]),
            };
            h = nh;
            match cal {
                None => tok = unmap(logits.argmax(-1, false)),
                Some(cal) => {
                    let (conf, ids) = logits.max_dim(-1, false);
                    tok = unmap(ids);
                    let cv: Vec<f32> = Vec::<f32>::try_from(conf.to_kind(Kind::Float).to_device(Device::Cpu))
                        .unwrap_or_else(|_| vec![f32::INFINITY; bsz as usize]);
                    for (r, &v) in cv.iter().enumerate() {
                        confs[r].push(v);
                        reach[r] *= cal.estimate(v);
                    }
                }
            }
            drafts.push(tok.shallow_clone());
            if let Some(cal) = cal {
                let best = reach.iter().cloned().fold(0.0f32, f32::max);
                if i + 1 < n && best < cal.confidence {
                    break;
                }
            }
        }
        if dbg {
            eprintln!("[mem]   draft: layer {t_layer:.2}ms  lm_head {t_head:.2}ms  ({n} steps)");
        }
        (Tensor::cat(&drafts, 1), confs) // [bsz, window]
    }

    /// The MTP layer forward without the `lm_head` — `draft_n_batched` splits
    /// them so the two can be timed (and, later, scheduled) independently.
    #[allow(clippy::too_many_arguments)]
    fn bstep_h(
        &self,
        trunk: &Model,
        cache: &MtpBatchedCache,
        trunk_hidden: &Tensor,
        ids: &Tensor,
        block_table: &Tensor,
        seqlens: &Tensor,
    ) -> Tensor {
        self.bstep(trunk, cache, trunk_hidden, ids, block_table, seqlens, false).0
    }

    /// Trunk `lm_head` over an MTP hidden state: `[bsz, 1, h] -> [bsz, vocab]`.
    /// This is the same 636 MB weight read the trunk does, so it is the floor
    /// for a draft step.
    fn head_logits(&self, trunk: &Model, g: &Tensor) -> Tensor {
        trunk
            .lm_head_on(g)
            .squeeze_dim(1)
            .narrow(1, 0, trunk.config.vocab_size)
    }

    /// Per-row (bsz 1) re-run of the MTP head over the `committed` accepted
    /// positions with the verify forward's *real* trunk hiddens, overwriting the
    /// KV entries `draft_n_batched` wrote from drafted tokens. `committed[j]` is
    /// the token now at position `q+1+j`; `vhid` `[1, >=k+1, h]` holds the trunk
    /// hidden at positions `q, q+1, …`, so position `q+1+j` is fed
    /// `(committed[j], vhid[j])` = `(token[p], trunk_hidden[p-1])`.
    pub fn sync_row(
        &self,
        trunk: &Model,
        cache: &MtpBatchedCache,
        vhid: &Tensor,
        committed: &[i64],
        q: i64,
        block_row: &Tensor,
    ) {
        let dev = vhid.device();
        for (j, &t) in committed.iter().enumerate() {
            let h_j = vhid.narrow(1, j as i64, 1);
            let ids = Tensor::from_slice(&[t]).reshape([1, 1]).to_device(dev);
            let sl = Tensor::from_slice(&[(q + 1 + j as i64) as i32]).to_device(dev);
            let _ = self.bstep(trunk, cache, &h_j, &ids, block_row, &sl, false);
        }
    }

    /// One MTP forward: appends K/V at the current MTP seqlen (position `pos`),
    /// bumps it by 1, returns `(mtp_hidden [1,1,h] fp16, logits [vocab] f32)`.
    fn step(&self, trunk: &Model, trunk_hidden: &Tensor, ids: &Tensor, want_logits: bool) -> (Tensor, Option<Tensor>) {
        let _no_grad = tch::no_grad_guard();
        let emb = trunk.embed_tokens(ids); // ids [1,1] i64 on device -> [1,1,h]
        let e = self.pre_fc_norm_embedding.forward(&emb);
        let hh = self.pre_fc_norm_hidden.forward(&trunk_hidden.reshape([1, 1, -1]));
        let x = Tensor::cat(&[e, hh], -1); // [1,1,2h]
        let x = self.fc.forward(&x); // [1,1,h] f32

        let a_in = self.input_norm.forward(&x);
        let kv = self.kv.as_ref().expect("single-stream MtpKvCache (use bstep for batched)");
        let a_out = self.attn.forward(
            &a_in,
            &Attn::Paged {
                k_cache: &kv.k,
                v_cache: &kv.v,
                block_table: &kv.block_table,
                seqlens: &kv.seqlens,
                rope_table: None,
            },
        );
        let x2 = &x + a_out;
        let m_in = self.post_norm.forward(&x2);
        let m_out = self.mlp.forward(&m_in);
        let x3 = x2 + m_out;
        kv.advance(1);

        let g = self.final_norm.forward(&x3); // [1,1,h] fp16
        // Draft-only, so this may be the vocabulary-pruned head; `draft_n`
        // maps its compact argmax back to a token id.
        let logits = want_logits.then(|| trunk.draft_logits_on(&g).0.reshape([-1]));
        (g, logits)
    }

    /// Teacher-force the MTP KV over prompt positions `0 .. toks.len()`, so its
    /// attention sees the full prefix. MTP KV position `p` is fed
    /// `(token[p], trunk_hidden[p-1])` (DeepSeek-V3 MTP alignment; position 0
    /// uses a zero hidden). `hiddens` is the trunk's post-final-norm hidden state
    /// for the whole prompt, `[1, len, h]`.
    pub fn prime(&self, trunk: &Model, hiddens: &Tensor, toks: &[i64]) {
        let _no_grad = tch::no_grad_guard();
        let dev = hiddens.device();
        let n = toks.len() as i64;
        let kv = self.kv.as_ref().expect("single-stream MtpKvCache");
        kv.set_seqlen(0);
        if n == 0 {
            return;
        }
        let hdim = hiddens.size()[2];
        let zero = Tensor::zeros([1, 1, hdim], (hiddens.kind(), dev));
        let shifted = if n > 1 {
            Tensor::cat(&[zero, hiddens.narrow(1, 0, n - 1)], 1)
        } else {
            zero
        }; // [1, n, h]
        let ids = Tensor::from_slice(toks).reshape([1, n]).to_device(dev);

        // Single prefill forward through the MTP head's attention layer — only
        // its K/V need to land in `kv`; the mlp / norm / logits are not cached.
        let emb = trunk.embed_tokens(&ids);
        let e = self.pre_fc_norm_embedding.forward(&emb);
        let hh = self.pre_fc_norm_hidden.forward(&shifted);
        let x = self.fc.forward(&Tensor::cat(&[e, hh], -1));
        let a_in = self.input_norm.forward(&x);
        let _ = self.attn.forward(
            &a_in,
            &Attn::Paged {
                k_cache: &kv.k,
                v_cache: &kv.v,
                block_table: &kv.block_table,
                seqlens: &kv.seqlens,
                rope_table: None,
            },
        );
        kv.advance(n);
    }

    /// Draft the single token following position `q` (`c` = tok at `q`), given
    /// the trunk hidden `h_prev` at `q-1`.
    pub fn draft_one(&self, trunk: &Model, h_prev: &Tensor, c: i64, q: i64) -> i64 {
        self.draft_n(trunk, h_prev, c, q, 1).int64_value(&[0])
    }

    /// Draft `n` tokens following position `q` (positions `q+1 .. q+n`). MTP KV
    /// `[0, q)` is already resident. Step 0 writes MTP KV `@ q` from
    /// `(c, h_prev)` — `h_prev` the *real* trunk hidden at `q-1` — then the chain
    /// feeds the head its own output state, leaving MTP seqlen `q+n`.
    ///
    /// Returns an `[n]` i64 tensor **on device** — the whole chain stays on the
    /// GPU (no per-step `argmax`→host sync; upstream does the same). Greedy.
    pub fn draft_n(&self, trunk: &Model, h_prev: &Tensor, c: i64, q: i64, n: i64) -> Tensor {
        debug_assert!(n >= 1);
        let dev = h_prev.device();
        self.kv.as_ref().unwrap().set_seqlen(q);
        // a pruned draft head argmaxes over compact indices, not token ids
        let unmap = |ids: Tensor| match trunk.draft_id_map() {
            None => ids.reshape([1, 1]),
            Some(m) => m.index_select(0, &ids.reshape([1])).reshape([1, 1]),
        };
        let (mut h, logits) = self.step(trunk, h_prev, &tok1(dev, c), true); // writes MTP KV @ q
        let mut tok = unmap(logits.unwrap().argmax(0, false)); // [1,1] i64, on device
        let mut drafts = Vec::with_capacity(n as usize);
        drafts.push(tok.shallow_clone());
        for _ in 1..n {
            let (nh, lg) = self.step(trunk, &h, &tok, true); // writes MTP KV @ q+i
            h = nh;
            tok = unmap(lg.unwrap().argmax(0, false));
            drafts.push(tok.shallow_clone());
        }
        Tensor::cat(&drafts, 0).reshape([-1])
    }

    /// After the trunk accepted `k` drafted tokens this round (`committed` are the
    /// real tokens now at positions `q+1 .. q+k`), re-run the MTP head over those
    /// positions with the verify forward's *real* trunk hiddens — overwriting the
    /// approximate KV entries written from the drafted tokens — so the next
    /// `draft_n` has a contiguous, exact prefix. `hiddens` `[1, >=k+1, h]` are
    /// the verify hiddens at positions `q, q+1, …`, so position `q+1+j` is fed
    /// `(committed[j], hiddens[j])` = `(token[p], trunk_hidden[p-1])`.
    pub fn sync_after_accept(&self, trunk: &Model, hiddens: &Tensor, committed: &[i64], q: i64) {
        if !committed.is_empty() {
            let dev = hiddens.device();
            self.kv.as_ref().unwrap().set_seqlen(q + 1);
            for (j, &t) in committed.iter().enumerate() {
                let h_j = hiddens.narrow(1, j as i64, 1);
                let _ = self.step(trunk, &h_j, &tok1(dev, t), false); // writes MTP KV @ q+1+j
            }
        }
        let _ = self.kv.as_ref().unwrap().get_seqlen(); // silence unused in release
    }
}

/// A single token id as a `[1, 1]` i64 tensor on `dev` (MTP forward input shape).
fn tok1(dev: Device, id: i64) -> Tensor {
    Tensor::from_slice(&[id]).reshape([1, 1]).to_device(dev)
}
