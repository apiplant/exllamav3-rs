//! `Model.from_config` + `model_ls.forward_ls` for Qwen3 (grade —, equivalence only).

use crate::cache::{PagedKvCache, Q35LayerCache, Qwen35Cache};
use crate::config::{ArchKind, Config, LayerKind};
use crate::modules::{Attn, Attention, Embedding, GatedMlp, Linear, RmsNorm, TransformerBlock};
use crate::qwen3_5::GatedDeltaNet;
use crate::safetensors::SafeTensors;
use anyhow::Result;
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// Build the attention context for one Qwen3.5 `full_attention` layer in the
/// batched path — plain paged fp16, or the dequant-into-scratch quantized path.
fn q35_full_ctx<'a>(
    lc: &'a crate::paged::Q35BatchLayer,
    cache: &'a crate::paged::Qwen35PagedCache,
    block_table: &'a Tensor,
    seqlens: &'a Tensor,
    rope_table: Option<&'a Tensor>,
) -> Attn<'a> {
    use crate::paged::Q35BatchLayer;
    match lc {
        Q35BatchLayer::Kv { k, v } => Attn::Paged {
            k_cache: k,
            v_cache: v,
            block_table,
            seqlens,
            rope_table,
        },
        Q35BatchLayer::KvQuant { qk, qv, sk, sv } => Attn::PagedQuant {
            qk,
            qv,
            sk,
            sv,
            k_scratch: None, // compact per-call scratch (batched path)
            v_scratch: None,
            block_table,
            seqlens,
            compand_a: cache.compand_a,
            rope_table,
        },
        Q35BatchLayer::Gdn { .. } => unreachable!("q35_full_ctx on a GDN layer"),
    }
}

impl Model {
    /// Shared body of the batched Qwen3.5 decoder loop: 64 layers of
    /// norm → (paged full-attn | GDN) → norm → gated MLP, over input hidden `x0`
    /// `[bsz, q_len, h]`. Returns the pre-final-norm hidden `[bsz, q_len, h]`.
    #[allow(clippy::too_many_arguments)]
    fn q35_batched_stack(
        &self,
        x0: Tensor,
        cache: &crate::paged::Qwen35PagedCache,
        block_table: &Tensor,
        seqlens: &Tensor,
        slots: &Tensor,
        gdn_history: bool,
        rope_table: Option<&Tensor>,
        // DFlash2 taps: layer indices whose *output* to capture, ascending, and
        // where to put them. The drafter's context K/V are projected from these
        // rather than computed by running the drafter, so they are the only
        // coupling between target and draft model.
        taps: Option<(&[i64], &mut Vec<Tensor>)>,
    ) -> Tensor {
        use crate::paged::Q35BatchLayer;
        let (tap_ids, tap_out) = match taps {
            Some((ids, out)) => (ids, Some(out)),
            None => (&[][..], None),
        };
        let mut tap_out = tap_out;
        // `EXL3_TRUNK_PROF=1`: attribute trunk time to full-attention / GDN /
        // MLP / norm+residual. Each stage is bracketed by a stream sync so the
        // numbers are real per-stage costs and not "whatever was queued" — the
        // whole point is that one wall-clock number for the forward tells you
        // nothing about which stage to work on. Debug only; the syncs serialize.
        let prof = *TRUNK_PROF.get_or_init(|| std::env::var("EXL3_TRUNK_PROF").is_ok());
        let dev_idx = 0;
        let t = || {
            if prof {
                tch::Cuda::synchronize(dev_idx);
            }
            std::time::Instant::now()
        };
        let mut x = x0;
        for (i, layer) in self.q35_layers.iter().enumerate() {
            let t0 = t();
            let y = layer.input_norm.forward(&x);
            let t1 = t();
            let y = match (&layer.attn, &cache.layers[i]) {
                (Q35Attn::Full(a), lc @ (Q35BatchLayer::Kv { .. } | Q35BatchLayer::KvQuant { .. })) => {
                    a.forward(&y, &q35_full_ctx(lc, cache, block_table, seqlens, rope_table))
                }
                (Q35Attn::Linear(g), Q35BatchLayer::Gdn { conv_state, recurrent_state }) => {
                    g.forward(&y, conv_state, recurrent_state, Some(slots), gdn_history)
                }
                _ => unreachable!("Qwen35PagedCache layer kind mismatch at {i}"),
            };
            let t2 = t();
            let x2 = &x + y;
            let z = layer.post_norm.forward(&x2);
            let t3 = t();
            let z = layer.mlp.forward(&z);
            x = x2 + z;
            let t4 = t();
            if prof {
                use std::sync::atomic::Ordering::Relaxed;
                let is_full = matches!(&layer.attn, Q35Attn::Full(_));
                TP_NORM.fetch_add((t1 - t0).as_nanos() as u64 + (t3 - t2).as_nanos() as u64, Relaxed);
                if is_full {
                    TP_ATTN.fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
                } else {
                    TP_GDN.fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
                }
                TP_MLP.fetch_add((t4 - t3).as_nanos() as u64, Relaxed);
            }
            if let Some(out) = tap_out.as_mut() {
                if tap_ids.contains(&(i as i64)) {
                    out.push(x.shallow_clone());
                }
            }
        }
        x
    }
}

/// One Qwen3.5 decoder layer: norm → (gated full-attn | gated-delta-net) → norm → gated MLP.
pub struct Q35Layer {
    input_norm: RmsNorm,
    attn: Q35Attn,
    post_norm: RmsNorm,
    mlp: GatedMlp,
}

pub enum Q35Attn {
    Full(Attention),
    Linear(GatedDeltaNet),
}

/// One qwen4_exp decoder layer. There are no input/post layernorms: each
/// sublayer is entered through a gated-residual hyper-connection site, which
/// both normalizes and collapses the stream stack, and left through the same
/// site's per-stream write-back gate.
pub struct Qwen4Layer {
    attn_hc: crate::hc::GatedResidual,
    attn: Q4Attn,
    mlp_hc: crate::hc::GatedResidual,
    mlp: crate::moe::BlockSparseMlp,
}

pub enum Q4Attn {
    /// QSA sparse full attention: the indexer picks the visible blocks, the
    /// attention runs under that mask.
    Qsa { attn: Attention, indexer: crate::qsa::QsaIndexer },
    Linear(GatedDeltaNet),
}

/// `EXL3_TRUNK_PROF=1` accumulators — see `q35_batched_stack`. Reported (and
/// reset) by [`trunk_prof_take`].
static TRUNK_PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
pub static TP_ATTN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TP_GDN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TP_MLP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TP_NORM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub static TP_G_PROJ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TP_G_CONV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TP_G_RULE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TP_G_OUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn trunk_prof_on() -> &'static bool {
    TRUNK_PROF.get_or_init(|| std::env::var("EXL3_TRUNK_PROF").is_ok())
}

/// GDN sub-stages `(proj, conv, delta_rule, norm+out_proj)` in ms.
pub fn gdn_prof_take() -> (f64, f64, f64, f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let ms = |a: &std::sync::atomic::AtomicU64| a.swap(0, Relaxed) as f64 / 1e6;
    (ms(&TP_G_PROJ), ms(&TP_G_CONV), ms(&TP_G_RULE), ms(&TP_G_OUT))
}

/// `(full_attn, gdn, mlp, norm)` milliseconds accumulated since the last call.
pub fn trunk_prof_take() -> (f64, f64, f64, f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let ms = |a: &std::sync::atomic::AtomicU64| a.swap(0, Relaxed) as f64 / 1e6;
    (ms(&TP_ATTN), ms(&TP_GDN), ms(&TP_MLP), ms(&TP_NORM))
}

pub struct Model {
    pub config: Config,
    embed: Embedding,
    /// Homogeneous Qwen3 stack (empty for Qwen3.5).
    blocks: Vec<TransformerBlock>,
    /// Hybrid Qwen3.5 stack (empty for Qwen3).
    q35_layers: Vec<Q35Layer>,
    /// qwen4_exp stack (empty for everything else).
    qwen4_layers: Vec<Qwen4Layer>,
    /// PLE injection sites, `(layer_idx, layer)`, run *ahead* of their block.
    ple_layers: Vec<(usize, crate::ple::PleLayer)>,
    /// The hashed n-gram table PLE reads from. Present iff `ple_layers` is.
    ngram: Option<crate::ngram::NGramEmbedding>,
    /// Combine-less mixer that collapses the stream stack before the head.
    /// qwen4_exp has no final model norm; this stands in its place.
    hc_mixer: Option<crate::hc::GatedResidual>,
    /// Final model norm. `None` on qwen4_exp, which has none: the combine-less
    /// hyper-connection mixer collapses the stream stack straight into the head.
    norm: Option<RmsNorm>,
    lm_head: Linear,
    /// Cheap `lm_head` over a subset of the vocabulary, used only to draft
    /// (see [`Model::enable_draft_head`]).
    draft_head: Option<DraftHead>,
    /// 128-blocks pinned into the draft head on top of the frequency prefix,
    /// learned from what the trunk actually emits (see [`Model::adapt_draft_head`]).
    draft_pinned: std::collections::BTreeSet<i64>,
    /// Per-128-block membership of the live draft head, for a cheap miss test.
    draft_keep: Vec<bool>,
    /// New 128-blocks discovered since the last rebuild.
    draft_misses: usize,
    /// Rebuilds so far, and whether we have given up on pruning entirely.
    draft_rebuilds: usize,
    draft_giveup: bool,
    device: Device,
}

/// A vocabulary-pruned `lm_head` plus the map from its compact output index
/// back to a real token id.
pub struct DraftHead {
    head: Linear,
    id_map: Tensor,
    /// `0` for kept ids, `-inf` for the padding the 128-block grid drags in.
    mask: Option<Tensor>,
}

impl Model {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        Self::load_with(dir, device, &crate::config::ConfigOverrides::default())
    }

    pub fn load_with(
        dir: &Path,
        device: Device,
        ov: &crate::config::ConfigOverrides,
    ) -> Result<Self> {
        Self::load_with_progress(dir, device, ov, None)
    }

    /// Like [`load_with`], but invokes `progress` with a fraction in `0.0..=1.0`
    /// after each transformer layer is loaded (layers dominate load time).
    pub fn load_with_progress(
        dir: &Path,
        device: Device,
        ov: &crate::config::ConfigOverrides,
        progress: Option<&dyn Fn(f32)>,
    ) -> Result<Self> {
        let report = |f: f32| {
            if let Some(p) = progress {
                p(f);
            }
        };
        let config = Config::from_dir_with(dir, ov)?;
        let mut stc = SafeTensors::open(dir, &[])?;
        // The trunk is ~93% of the checkpoint and loading it is disk-bound, so
        // read ahead of the module constructors instead of leaving the drive
        // idle through every H2D copy and dequant. Vision/MTP open their own
        // handles and are small enough to read inline.
        stc.start_prefetch();
        let pfx = config.key_prefix.clone();
        let eps = config.rms_norm_eps;
        let nb = config.norm_constant_bias;

        let embed = Embedding::load(&stc, &format!("{pfx}.embed_tokens"), device)?;

        let mut blocks = Vec::new();
        let mut q35_layers = Vec::new();
        let mut qwen4_layers = Vec::new();
        let mut ple_layers = Vec::new();
        let mut ngram = None;
        let mut hc_mixer = None;
        match config.arch_kind {
            ArchKind::Llama
            | ArchKind::Qwen3
            | ArchKind::Qwen3Moe
            | ArchKind::Glm4
            | ArchKind::Glm4Moe => {
                for i in 0..config.num_hidden_layers {
                    blocks.push(TransformerBlock::load(
                        &stc,
                        &format!("{pfx}.layers.{i}"),
                        &config,
                        device,
                        i,
                    )?);
                    report((i + 1) as f32 / config.num_hidden_layers as f32);
                }
            }
            ArchKind::Qwen35 => {
                for i in 0..config.num_hidden_layers {
                    let key = format!("{pfx}.layers.{i}");
                    let attn = match config.layer_types[i as usize] {
                        LayerKind::FullAttention => Q35Attn::Full(Attention::load(
                            &stc,
                            &format!("{key}.self_attn"),
                            &config,
                            device,
                        )?),
                        LayerKind::LinearAttention => Q35Attn::Linear(GatedDeltaNet::load(
                            &stc,
                            &format!("{key}.linear_attn"),
                            &config,
                            device,
                        )?),
                    };
                    q35_layers.push(Q35Layer {
                        input_norm: RmsNorm::load_biased(&stc, &format!("{key}.input_layernorm"), eps, nb, device)?,
                        attn,
                        post_norm: RmsNorm::load_biased(&stc, &format!("{key}.post_attention_layernorm"), eps, nb, device)?,
                        mlp: GatedMlp::load(&stc, &format!("{key}.mlp"), &config, device)?,
                    });
                    report((i + 1) as f32 / config.num_hidden_layers as f32);
                }
            }
            ArchKind::Qwen4Exp => {
                let q4 = config.qwen4.clone().expect("qwen4_exp config without qwen4 params");
                // `use_combine` = "has a write-back gate": every per-layer site
                // does, the final mixer does not (it only collapses the stack).
                let hc = |key: &str, use_combine: bool| {
                    crate::hc::GatedResidual::load(
                        &stc,
                        key,
                        q4.hc_mult,
                        config.hidden_size,
                        eps,
                        use_combine,
                        device,
                    )
                };
                for i in 0..config.num_hidden_layers {
                    let key = format!("{pfx}.layers.{i}");
                    // PLE sites run ahead of their block, injecting into the raw
                    // stream stack rather than into a sublayer's output.
                    if q4.ple_layer_ids.contains(&i) {
                        ple_layers.push((
                            i as usize,
                            crate::ple::PleLayer::load(
                                &stc,
                                &format!("{key}.ple"),
                                q4.hc_mult,
                                config.hidden_size,
                                q4.ple_embed_dim,
                                q4.ngram_size,
                                q4.ple_conv_kernel_size,
                                eps,
                                device,
                            )?,
                        ));
                    }
                    let attn = match config.layer_types[i as usize] {
                        LayerKind::FullAttention => Q4Attn::Qsa {
                            attn: Attention::load(&stc, &format!("{key}.self_attn"), &config, device)?,
                            indexer: crate::qsa::QsaIndexer::load(
                                &stc,
                                &format!("{key}.self_attn.indexer"),
                                config.hidden_size,
                                q4.indexer_n_heads,
                                q4.indexer_head_dim,
                                q4.indexer_budget,
                                q4.indexer_compress_ratio,
                                eps,
                                &crate::rope::RoPE::new(device, &config.rope),
                                device,
                            )?,
                        },
                        LayerKind::LinearAttention => Q4Attn::Linear(GatedDeltaNet::load(
                            &stc,
                            &format!("{key}.linear_attn"),
                            &config,
                            device,
                        )?),
                    };
                    qwen4_layers.push(Qwen4Layer {
                        attn_hc: hc(&format!("{key}.attn_hyper_connection"), true)?,
                        attn,
                        mlp_hc: hc(&format!("{key}.mlp_hyper_connection"), true)?,
                        mlp: crate::moe::BlockSparseMlp::load(&stc, &format!("{key}.mlp"), &config, device)?,
                    });
                    report((i + 1) as f32 / config.num_hidden_layers as f32);
                }
                if !ple_layers.is_empty() {
                    ngram = Some(crate::ngram::NGramEmbedding::load(
                        &stc,
                        &format!("{pfx}.ple_embedding"),
                        q4.ngram_size,
                        q4.heads_per_ngram,
                        q4.ple_embed_dim,
                        q4.ple_eos_token_id,
                        device,
                    )?);
                }
                hc_mixer = Some(hc(&format!("{pfx}.hyper_connection_mixer"), false)?);
            }
        }

        // qwen4_exp carries no `{pfx}.norm`; the hyper-connection mixer below
        // takes its place.
        let norm = match config.arch_kind {
            ArchKind::Qwen4Exp => None,
            _ => Some(RmsNorm::load_biased(&stc, &format!("{pfx}.norm"), eps, nb, device)?),
        };

        let head_alt = if config.tie_word_embeddings && !stc.has("lm_head.trellis") && !stc.has("lm_head.weight") {
            Some(format!("{pfx}.embed_tokens"))
        } else {
            None
        };
        let lm_head = Linear::load(
            &stc,
            "lm_head",
            head_alt.as_deref(),
            config.hidden_size,
            config.vocab_size,
            device,
            false,
            0.0,
        )?;

        let mut m = Self {
            config,
            embed,
            blocks,
            q35_layers,
            qwen4_layers,
            ple_layers,
            ngram,
            hc_mixer,
            norm,
            lm_head,
            draft_head: None,
            draft_pinned: Default::default(),
            draft_keep: Vec::new(),
            draft_misses: 0,
            draft_rebuilds: 0,
            draft_giveup: false,
            device,
        };
        // Opt-in: a draft-only `lm_head` over the frequent head of the
        // vocabulary. Output-identical by construction (drafts are verified
        // exactly); trades a little acceptance and ~0.3 GB of VRAM for a much
        // cheaper draft step.
        m.refresh_draft_head(&[])?;
        Ok(m)
    }

    /// Qwen3.5 hybrid forward against a [`Qwen35Cache`]. Returns last-position
    /// logits `(vocab,)` f32. Does not bump `cache.seqlens` (caller does, via
    /// `prefill_qwen35` after prefill; the GDN layers are self-advancing).
    pub fn forward_qwen35(&self, ids: &Tensor, cache: &Qwen35Cache) -> Tensor {
        let x = self.q35_stack(self.embed.forward(ids), cache, None, &[]);
        self.head(&x)
    }

    /// The 64-layer hybrid decoder loop (no final norm / head). `x` is the input
    /// hidden state (`embed(ids)` normally, or spliced text+image embeddings for
    /// the multimodal path); `rope_table` is the optional MRoPE angle table.
    fn q35_stack(
        &self,
        x0: Tensor,
        cache: &Qwen35Cache,
        rope_table: Option<&Tensor>,
        deepstack: &[Tensor],
    ) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let mut x = x0;
        for (i, layer) in self.q35_layers.iter().enumerate() {
            // Deepstack image features, added at the image token positions ahead
            // of the layer that consumes them (zero elsewhere, so a plain add).
            if let Some(d) = deepstack.get(i) {
                let k = x.kind();
                x = x + d.to_kind(k);
            }
            let y = layer.input_norm.forward(&x);
            let y = match (&layer.attn, &cache.layers[i]) {
                (Q35Attn::Full(a), Q35LayerCache::Kv { k, v }) => a.forward(
                    &y,
                    &Attn::Paged {
                        k_cache: k,
                        v_cache: v,
                        block_table: &cache.block_table,
                        seqlens: &cache.seqlens,
                        rope_table,
                    },
                ),
                (Q35Attn::Linear(g), Q35LayerCache::Gdn(st)) => {
                    g.forward(&y, &st.conv_state, &st.recurrent_state, None, false)
                }
                _ => unreachable!("Qwen35Cache layer kind mismatch at {i}"),
            };
            let x2 = &x + y;
            let z = layer.post_norm.forward(&x2);
            let z = layer.mlp.forward(&z);
            x = x2 + z;
        }
        x
    }

    /// qwen4_exp forward against a [`Qwen4Cache`]. `ids` are the *new* tokens;
    /// they are appended to the cache's own token history for the n-gram hashing.
    /// Returns last-position logits `(vocab,)` f32.
    ///
    /// Does not bump `past_len` — the caller does, as on the Qwen3.5 path. The
    /// recurrent state (GDN, and the PLE conv/token window) *is* advanced here,
    /// because it can only be advanced by running.
    pub fn forward_qwen4(&self, ids: &Tensor, cache: &crate::cache::Qwen4Cache) -> Tensor {
        let (streams, _) = self.qwen4_stack(ids, cache);
        let seq = streams.size()[1];
        let last = streams.narrow(1, seq - 1, 1).contiguous();
        self.qwen4_head(&last)
    }

    /// Collapse the stream stack and project to logits. qwen4_exp has no final
    /// model norm — the combine-less mixer normalizes and collapses in one step,
    /// which is exactly what the norm would otherwise have done.
    fn qwen4_head(&self, streams: &Tensor) -> Tensor {
        let mixer = self.hc_mixer.as_ref().expect("qwen4_head on a non-qwen4_exp model");
        let (_, x) = mixer.mix(streams);
        self.lm_head
            .forward(&x.to_kind(Kind::Half))
            .reshape([self.config.vocab_size])
            .to_kind(Kind::Float)
    }

    /// The qwen4_exp decoder loop. Returns the fp32 stream stack
    /// `[1, seq, hc_mult, hidden]` and the PLE conv columns of this chunk (empty
    /// when the model has no PLE site), which the caller has already had
    /// committed to the cache.
    ///
    /// Unlike the Qwen3.5 loop there is no running `x`: the residual *is* the
    /// stack, every sublayer reads it through its site's gate and writes back
    /// through the same site's per-stream gate.
    fn qwen4_stack(
        &self,
        ids: &Tensor,
        cache: &crate::cache::Qwen4Cache,
    ) -> (Tensor, Option<Tensor>) {
        let _no_grad = tch::no_grad_guard();
        let q4 = self.config.qwen4.as_ref().expect("qwen4_stack on a non-qwen4_exp model");
        let past_len = cache.past_len.get();
        let seq = ids.size()[1];
        if past_len + seq > cache.max_len {
            panic!("qwen4 cache overflow: {past_len} + {seq} > {}", cache.max_len);
        }

        let mut streams = crate::hc::expand_streams(&self.embed.forward(ids), q4.hc_mult);

        // n-gram features for this chunk, hashed against the cache's trailing
        // token history so a chunked decode hashes like a single-shot prefill.
        let emb = self.ngram.as_ref().map(|ng| {
            let hist = cache.ple.token_input(0, ids);
            ng.forward(&hist).to_device(streams.device())
        });
        let mut ple_cols = None;

        for (i, layer) in self.qwen4_layers.iter().enumerate() {
            if let Some((_, ple)) = self.ple_layers.iter().find(|(li, _)| *li == i) {
                let emb = emb.as_ref().expect("a PLE site without an n-gram table");
                let state = cache.ple.window(0).unsqueeze(0);
                let (inj, stream) = ple.forward_streams(&streams, emb, Some(&state));
                let _ = streams.f_add_(&inj).unwrap();
                // `stream` is state ++ this chunk; only the new columns are pushed.
                ple_cols = Some(stream.narrow(2, ple.conv_state_len(), seq).squeeze_dim(0));
            }

            let (post, mixed) = layer.attn_hc.mix(&streams);
            let y = match (&layer.attn, &cache.layers[i]) {
                (
                    Q4Attn::Qsa { attn, indexer },
                    crate::cache::Q4LayerCache::Full { k, v, raw_k },
                ) => {
                    let x = mixed.to_kind(Kind::Half);
                    let mask = self.qsa_mask(indexer, &x, raw_k, past_len, seq);
                    attn.forward_masked(&x, k, v, past_len, &mask)
                }
                (Q4Attn::Linear(g), crate::cache::Q4LayerCache::Gdn(st)) => g.forward(
                    &mixed.to_kind(Kind::Half),
                    &st.conv_state,
                    &st.recurrent_state,
                    None,
                    false,
                ),
                _ => unreachable!("Qwen4Cache layer kind mismatch at {i}"),
            };
            layer.attn_hc.apply_(&mut streams, &y, &post.expect("a layer site with no write-back gate"));

            let (post, mixed) = layer.mlp_hc.mix(&streams);
            let y = layer.mlp.forward(&mixed.to_kind(Kind::Half));
            layer.mlp_hc.apply_(&mut streams, &y, &post.expect("a layer site with no write-back gate"));
        }

        // Commit the PLE recurrence for this chunk once the whole stack is done,
        // so a panic mid-forward leaves the cache on the last good position.
        if let Some(cols) = &ple_cols {
            cache.ple.push(0, cols, ids);
        }
        (streams, ple_cols)
    }

    /// Project the indexer for this chunk, append its raw keys to the cache,
    /// pool every complete block of the whole history and return the QSA
    /// selection mask `[seq, past_len + seq]`.
    fn qsa_mask(
        &self,
        indexer: &crate::qsa::QsaIndexer,
        x: &Tensor,
        raw_k_cache: &Tensor,
        past_len: i64,
        seq: i64,
    ) -> Tensor {
        let (cos, sin) = indexer.rope_tables(past_len, seq, x.device());
        let (q, raw_k) = indexer.project(x, &cos, &sin);
        let _ = raw_k_cache
            .narrow(1, past_len, seq)
            .copy_(&raw_k.squeeze_dim(0).to_kind(raw_k_cache.kind()).unsqueeze(0));

        let total = past_len + seq;
        let hist = raw_k_cache.narrow(1, 0, total);
        // Pooled keys are roped at their block's start position, so the tables
        // have to span the whole history, not just this chunk.
        let (cos_f, sin_f) = indexer.rope_tables(0, total, x.device());
        let pooled = indexer.pool_keys(&hist, &cos_f, &sin_f);
        indexer.p.token_mask(&q, &pooled, past_len, total).squeeze_dim(0)
    }

    /// Multimodal Qwen3.5 forward: `x0` is the spliced text+image embedding
    /// stream `[1, seq, h]`, `rope_table` the MRoPE angle table, `deepstack` the
    /// per-layer image features (empty on every Qwen3.5 checkpoint to hand, which
    /// all carry `deepstack_visual_indexes: []`). Returns
    /// `(post-final-norm hidden [1, seq, h] fp16, logits [1, seq, vocab] f32)`.
    /// Does not bump `cache.seqlens`.
    pub fn forward_qwen35_mm(
        &self,
        x0: &Tensor,
        cache: &Qwen35Cache,
        rope_table: Option<&Tensor>,
        deepstack: &[Tensor],
    ) -> (Tensor, Tensor) {
        let x = self.q35_stack(x0.shallow_clone(), cache, rope_table, deepstack);
        let normed = self.norm().forward(&x);
        let logits = self.lm_head.forward(&normed).to_kind(Kind::Float);
        (normed, logits)
    }

    /// As `forward_qwen35`, but returns the full post-final-norm hidden state
    /// `[1, seq, h]` (fp16) alongside the full logits `[1, seq, vocab]` (f32).
    /// The MTP draft head consumes the hidden state; the trunk's own next token
    /// comes from the logits. Does not bump `cache.seqlens`.
    pub fn forward_qwen35_hidden(&self, ids: &Tensor, cache: &Qwen35Cache) -> (Tensor, Tensor) {
        self.forward_qwen35_mm(&self.embed.forward(ids), cache, None, &[])
    }

    /// Whether every sparse MLP in the stack took the multi-GEMM path. Only then
    /// is a decode step free of the routing readback, and so capturable.
    pub fn moe_is_fused(&self) -> bool {
        let mut any = false;
        for b in &self.blocks {
            if let Some(m) = b.sparse_mlp() {
                any = true;
                if !m.is_fused() {
                    return false;
                }
            }
        }
        any
    }

    /// Token embedding, `[1, seq]` i64 → `[1, seq, h]` (embedding out-kind).
    /// The final model norm, for the paths that have one. qwen4_exp does not,
    /// and never reaches any of them.
    fn norm(&self) -> &RmsNorm {
        self.norm
            .as_ref()
            .expect("this architecture has no final model norm")
    }

    pub fn embed_tokens(&self, ids: &Tensor) -> Tensor {
        self.embed.forward(ids)
    }

    /// Trunk final RMSNorm applied to an external hidden state `[.., h]`.
    pub fn final_norm(&self, x: &Tensor) -> Tensor {
        self.norm().forward(x)
    }

    /// Trunk `lm_head` on an already-normed hidden state `[.., h]` → `[.., vocab]` f32.
    pub fn lm_head_on(&self, normed: &Tensor) -> Tensor {
        self.lm_head.forward(normed).to_kind(Kind::Float)
    }

    /// Build the pruned draft `lm_head`, keeping token ids `[0, cut)` plus the
    /// special/added tokens that sit at the very top of the vocabulary.
    ///
    /// A decode step is bandwidth-bound and `lm_head` is the largest single
    /// weight in the model (1.27 GB at 4 bits for a 248k vocab), so a
    /// speculative round spends most of its draft time re-reading it: measured
    /// here, a draft step costs 1.93 ms of which ~1.90 ms is exactly the time to
    /// stream `lm_head` once. Drafting from a quarter of the vocabulary cuts
    /// that roughly fourfold.
    ///
    /// This cannot change what the model emits. Verification is exact, so a
    /// token the pruned head cannot propose is simply never drafted and the
    /// trunk's own token stands; the only thing at risk is the acceptance rate.
    /// Token ids are roughly frequency-ordered (BPE merge order), so the tail
    /// being dropped is rare: on a mixed prose/code sample, ids `< 65536` — 26%
    /// of the vocabulary — account for 97.8% of tokens.
    ///
    /// `cut` comes from `EXL3_DRAFT_VOCAB` (`0` disables). Costs a copy of the
    /// kept slice of the trellis in VRAM.
    ///
    /// `extra` pins additional ids into the keep set. A frequency prefix is an
    /// English/code prior: measured on this checkpoint, `cut = 65536` leaves CJK
    /// acceptance at 9% (vs 30% unpruned) and even `131072` costs Arabic 10
    /// points, because those tokens live high in the id space. Passing the
    /// prompt's own tokens fixes that — whatever script the conversation is in,
    /// its tokens are in the set, and continuations reuse them heavily.
    pub fn enable_draft_head(&mut self, cut: i64, extra: &[i64]) -> Result<()> {
        if cut <= 0 || cut >= self.config.vocab_size {
            self.draft_head = None;
            return Ok(());
        }
        // Specials (`<|im_start|>`, EOS, …) live above `vocab_size`'s tail and
        // must survive the prune — dropping EOS would stall every draft chain at
        // the end of a turn.
        const SPECIAL_TAIL: i64 = 512;
        const BLK: i64 = 128;
        let v = self.config.vocab_size;
        self.draft_pinned
            .extend(extra.iter().filter(|&&i| i >= 0 && i < v).map(|i| i / BLK));
        let mut ranges = vec![(0, cut), ((v - SPECIAL_TAIL).max(cut), v)];
        // `prune_out` snaps to the 128-block grid and merges the overlaps for us
        ranges.extend(self.draft_pinned.iter().map(|&b| (b * BLK, b * BLK + BLK)));
        let t0 = std::time::Instant::now();
        let first = self.draft_head.is_none();
        let (head, ids) = self.lm_head.prune_out(&ranges)?;
        self.draft_keep = vec![false; (self.lm_head.out_features / BLK) as usize];
        for &i in &ids {
            self.draft_keep[(i / BLK) as usize] = true;
        }
        self.draft_misses = 0;
        let dev = self.device;
        let mask = ids.iter().any(|&i| i >= v).then(|| {
            let m: Vec<f32> = ids.iter().map(|&i| if i < v { 0.0 } else { f32::NEG_INFINITY }).collect();
            Tensor::from_slice(&m).to_device(dev)
        });
        let id_map = Tensor::from_slice(&ids).to_device(dev);
        // Always reported: this is a *copy* of the kept slice of `lm_head`, so
        // it costs real VRAM that would otherwise be KV pool. On a 24 GB card a
        // 53% head is ~640 MiB — about 32k tokens of context.
        let frac = ids.len() as f64 / v as f64;
        if first {
            crate::sinfo!(
                "draft head: lm_head pruned {} -> {} outputs ({:.0}% of the weight read, \
                 ~{:.0} MiB VRAM); lower EXL3_DRAFT_VOCAB or unset it to give the KV pool that back",
                v,
                ids.len(),
                frac * 100.0,
                frac * self.lm_head_bytes() as f64 / (1024.0 * 1024.0),
            );
        }
        let _ = t0;
        self.draft_head = Some(DraftHead { head, id_map, mask });
        Ok(())
    }

    /// (Re)build the draft head at the cut named by `EXL3_DRAFT_VOCAB`, pinning
    /// `extra` — normally the prompt's own token ids. A no-op when unset.
    pub fn refresh_draft_head(&mut self, extra: &[i64]) -> Result<()> {
        if self.draft_giveup {
            return Ok(());
        }
        let cut: i64 = match std::env::var("EXL3_DRAFT_VOCAB") {
            Ok(v) => v.parse().unwrap_or(0),
            Err(_) => return Ok(()),
        };
        self.enable_draft_head(cut, extra)
    }

    /// Bytes the full `lm_head` occupies — the base a pruned draft head's VRAM
    /// cost is a fraction of.
    fn lm_head_bytes(&self) -> i64 {
        self.lm_head.nbytes()
    }

    /// Feed the tokens the trunk actually committed back into the draft head.
    ///
    /// A frequency prefix is an English/code prior and does not transfer: on
    /// this checkpoint `cut = 65536` drops CJK acceptance from 30% to 9%, and
    /// even `131072` costs Arabic 10 points, because those scripts live high in
    /// the id space. Rather than guess a cut per language, watch what the trunk
    /// emits — every committed token outside the keep set is precisely a token
    /// the draft head could never have proposed — and fold the misses back in.
    /// A rebuild costs ~2 ms (a third of one round), so it pays for itself
    /// almost immediately and then stops firing once the set has converged.
    ///
    /// No-op unless a draft head is built.
    pub fn adapt_draft_head(&mut self, committed: &[i64]) -> Result<()> {
        const BLK: i64 = 128;
        /// New 128-blocks to discover before rebuilding. A rebuild costs ~2 ms
        /// against a ~7 ms round, so adapting eagerly is nearly free — and it
        /// has to be eager: an Arabic generation puts half its tokens outside a
        /// 131072 prefix, spread over ~160 blocks, so a lazy threshold never
        /// converges inside one response.
        const REBUILD_AFTER: usize = 4;
        if self.draft_head.is_none() {
            return Ok(());
        }
        for &t in committed {
            let b = t / BLK;
            if let Some(false) = self.draft_keep.get(b as usize) {
                // `draft_keep` only moves at a rebuild, so this counts blocks
                // discovered *since* the last one, not since this call
                if self.draft_pinned.insert(b) {
                    self.draft_misses += 1;
                }
            }
        }
        if self.draft_misses < REBUILD_AFTER {
            return Ok(());
        }
        // Give up if the keep set will not settle. Some scripts are simply not
        // concentrated in the low ids — an Arabic response on this checkpoint
        // puts half its tokens outside a 131072 prefix and keeps finding new
        // blocks, so drafting from a subset costs more acceptance than the
        // cheaper head wins back. Falling back to the full head bounds that
        // downside to the handful of rounds it took to notice.
        const GIVE_UP_AFTER: usize = 16;
        self.draft_rebuilds += 1;
        if self.draft_rebuilds > GIVE_UP_AFTER {
            self.draft_giveup = true;
            self.draft_head = None;
            self.draft_keep.clear();
            return Ok(());
        }
        self.refresh_draft_head(&[])?;
        Ok(())
    }

    /// Map from the draft head's compact output index to a token id, when the
    /// head is pruned.
    pub fn draft_id_map(&self) -> Option<&Tensor> {
        self.draft_head.as_ref().map(|d| &d.id_map)
    }

    /// Logits for a *draft* step over an already-normed hidden state. Falls back
    /// to the full head when no draft head is built. The returned logits are
    /// indexed compactly; `Some(id_map)` maps an argmax over them back to a
    /// token id.
    pub fn draft_logits_on(&self, normed: &Tensor) -> (Tensor, Option<&Tensor>) {
        match &self.draft_head {
            // the full head is padded past `vocab_size`; trim as `lm_head_on`'s
            // callers do, so a padding column can never win the argmax
            None => {
                let y = self.lm_head_on(normed);
                let d = y.dim() as i64 - 1;
                (y.narrow(d, 0, self.config.vocab_size), None)
            }
            Some(d) => {
                let mut y = d.head.forward(normed).to_kind(Kind::Float);
                if let Some(m) = &d.mask {
                    y += m;
                }
                (y, Some(&d.id_map))
            }
        }
    }

    pub fn prefill_qwen35(&self, ids: &Tensor, cache: &Qwen35Cache) -> Tensor {
        let logits = self.forward_qwen35(ids, cache);
        cache.advance(ids.size()[1]);
        logits
    }

    /// As `forward_qwen35_batched`, but also returns the full post-final-norm
    /// hidden state `[bsz, q_len, h]` (fp16) for the MTP draft head. Always
    /// computes every position (`last_only` is implicitly false). `want_logits`
    /// false skips the `lm_head` — MTP priming only needs the hidden states, and
    /// full-sequence logits for a long prompt are `[1, len, vocab]` f32 (several
    /// GB); the returned logits tensor is then empty.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen35_batched_h(
        &self,
        ids: &Tensor,
        cache: &crate::paged::Qwen35PagedCache,
        block_table: &Tensor,
        seqlens: &Tensor,
        slots: &Tensor,
        gdn_history: bool,
        want_logits: bool,
    ) -> (Tensor, Tensor) {
        let _no_grad = tch::no_grad_guard();
        let x0 = self.embed.forward(ids);
        let x =
            self.q35_batched_stack(x0, cache, block_table, seqlens, slots, gdn_history, None, None);
        let normed = self.norm().forward(&x); // [bsz, q_len, h] half
        let logits = if want_logits {
            self.lm_head.forward(&normed).to_kind(Kind::Float)
        } else {
            Tensor::zeros([0], (Kind::Float, normed.device()))
        };
        (normed, logits)
    }

    /// As [`Model::forward_qwen35_batched_h`], but also captures the hidden
    /// state at the end of each layer in `tap_ids` (ascending). Used to feed a
    /// DFlash2 drafter, whose context K/V are a projection of these rather than
    /// anything it computes itself.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen35_batched_h_taps(
        &self,
        ids: &Tensor,
        cache: &crate::paged::Qwen35PagedCache,
        block_table: &Tensor,
        seqlens: &Tensor,
        slots: &Tensor,
        gdn_history: bool,
        tap_ids: &[i64],
    ) -> (Tensor, Vec<Tensor>) {
        let _no_grad = tch::no_grad_guard();
        let x0 = self.embed.forward(ids);
        let mut taps = Vec::with_capacity(tap_ids.len());
        let x = self.q35_batched_stack(
            x0,
            cache,
            block_table,
            seqlens,
            slots,
            gdn_history,
            None,
            Some((tap_ids, &mut taps)),
        );
        (self.norm().forward(&x), taps)
    }

    /// Batched Qwen3.5 forward for the dynamic generator. `ids` `[bsz, q_len]`;
    /// `block_table` `[bsz, P]` i32, `seqlens` `[bsz]` i32 (pre-append length /
    /// RoPE offset for the full-attn layers), `slots` `[bsz]` i32 (recurrent
    /// pool slot per row for the GDN layers). Returns `[bsz, q_len, vocab]` f32
    /// (or `[bsz, 1, vocab]` if `last_only`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen35_batched(
        &self,
        ids: &Tensor,
        cache: &crate::paged::Qwen35PagedCache,
        block_table: &Tensor,
        seqlens: &Tensor,
        slots: &Tensor,
        last_only: bool,
        gdn_history: bool,
    ) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x0 = self.embed.forward(ids);
        self.q35_batched_finish(
            self.q35_batched_stack(x0, cache, block_table, seqlens, slots, gdn_history, None, None),
            last_only,
        )
    }

    /// As `forward_qwen35_batched`, but the input hidden state is supplied
    /// directly (`embeds` `[bsz, q_len, h]`) instead of being looked up from
    /// token ids — for the multimodal path, where image-token positions carry
    /// vision-tower embeddings. `rope_table` is the MRoPE angle table.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen35_batched_embed(
        &self,
        embeds: &Tensor,
        cache: &crate::paged::Qwen35PagedCache,
        block_table: &Tensor,
        seqlens: &Tensor,
        slots: &Tensor,
        last_only: bool,
        gdn_history: bool,
        rope_table: Option<&Tensor>,
    ) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        self.q35_batched_finish(
            self.q35_batched_stack(
                embeds.shallow_clone(),
                cache,
                block_table,
                seqlens,
                slots,
                gdn_history,
                rope_table,
                None,
            ),
            last_only,
        )
    }

    fn q35_batched_finish(&self, x: Tensor, last_only: bool) -> Tensor {
        let x = if last_only {
            let q = x.size()[1];
            x.narrow(1, q - 1, 1).contiguous()
        } else {
            x
        };
        let x = self.norm().forward(&x);
        self.lm_head.forward(&x).to_kind(Kind::Float)
    }

    fn head(&self, x: &Tensor) -> Tensor {
        let seq = x.size()[1];
        let last = x.narrow(1, seq - 1, 1).contiguous(); // (1,1,h)
        let last = self.norm().forward(&last);
        self.lm_head
            .forward(&last)
            .reshape([self.config.vocab_size])
            .to_kind(Kind::Float)
    }

    /// As `head`, but folds the deferred `pending` residual add (last block's
    /// MLP output) into the final norm at the last position.
    fn head_res(&self, x: &Tensor, pending: Option<&Tensor>) -> Tensor {
        let seq = x.size()[1];
        let last = x.narrow(1, seq - 1, 1).contiguous(); // (1,1,h)
        let normed = match pending {
            Some(p) => {
                let p_last = p.narrow(1, seq - 1, 1).contiguous();
                self.norm().forward_res(&last, Some(&p_last))
            }
            None => self.norm().forward(&last),
        };
        self.lm_head
            .forward(&normed)
            .reshape([self.config.vocab_size])
            .to_kind(Kind::Float)
    }

    /// Final norm + `lm_head` for the batched homogeneous path, folding the
    /// deferred `pending` residual add into the norm. Returns `[.., vocab]` f32.
    fn finalize_head(&self, x: Tensor, pending: Option<Tensor>, last_only: bool) -> Tensor {
        let q = x.size()[1];
        let (x, pending) = if last_only && q > 1 {
            (
                x.narrow(1, q - 1, 1).contiguous(),
                pending.map(|p| p.narrow(1, q - 1, 1).contiguous()),
            )
        } else {
            (x, pending)
        };
        let normed = self.norm().forward_res(&x, pending.as_ref());
        self.lm_head.forward(&normed).to_kind(Kind::Float)
    }

    /// Full forward over `ids` (1, seq), no cache; logits (vocab,) for the last position.
    pub fn forward_last(&self, ids: &Tensor, position: i64) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x = self.embed.forward(ids); // (1, seq, h) f32
        let ctx = Attn::NoCache { past_len: position };
        let mut pending: Option<Tensor> = None;
        for blk in &self.blocks {
            pending = Some(blk.forward(&x, &ctx, pending.as_ref()));
        }
        self.head_res(&x, pending.as_ref())
    }

    /// One paged forward: reads `cache.seqlens` (device) as the current position,
    /// appends this call's K/V, returns last-position logits. Does **not** bump
    /// `seqlens` — the caller does (eagerly after prefill, or inside the captured
    /// graph for decode). `ids` is the whole prompt on prefill, `[1,1]` per step.
    pub fn forward_paged(&self, ids: &Tensor, cache: &PagedKvCache) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x = self.embed.forward(ids);
        let mut pending: Option<Tensor> = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            let ctx = Attn::Paged {
                k_cache: &cache.k[i],
                v_cache: &cache.v[i],
                block_table: &cache.block_table,
                seqlens: &cache.seqlens,
                rope_table: None,
            };
            pending = Some(blk.forward(&x, &ctx, pending.as_ref()));
        }
        self.head_res(&x, pending.as_ref())
    }

    /// Multimodal forward on the paged (non-hybrid) path: `x0` is the spliced
    /// text+image embedding stream `[1, seq, h]`, `rope_table` the MRoPE angle
    /// table, `deepstack` the per-layer image features to fold in.
    ///
    /// Returns `(post-final-norm hidden [1, seq, h], logits [1, seq, vocab])` and
    /// does **not** advance the cache — the caller does, matching
    /// `forward_qwen35_mm`.
    ///
    /// Deepstack entry `i` is added to the residual before block `i`. It is a
    /// full-width tensor that is zero everywhere but the image token positions,
    /// so this is a plain add rather than a scatter; the residual add of the
    /// previous block's MLP is still deferred in `pending`, and since both are
    /// additions into the same stream the order does not matter.
    pub fn forward_paged_mm(
        &self,
        x0: &Tensor,
        cache: &PagedKvCache,
        rope_table: Option<&Tensor>,
        deepstack: &[Tensor],
    ) -> (Tensor, Tensor) {
        let _no_grad = tch::no_grad_guard();
        let x = x0.shallow_clone();
        let mut pending: Option<Tensor> = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            if let Some(d) = deepstack.get(i) {
                let _ = x.shallow_clone().f_add_(&d.to_kind(x.kind())).unwrap();
            }
            let ctx = Attn::Paged {
                k_cache: &cache.k[i],
                v_cache: &cache.v[i],
                block_table: &cache.block_table,
                seqlens: &cache.seqlens,
                rope_table,
            };
            pending = Some(blk.forward(&x, &ctx, pending.as_ref()));
        }
        let normed = self.norm().forward_res(&x, pending.as_ref());
        let logits = self.lm_head.forward(&normed).to_kind(Kind::Float);
        (normed, logits)
    }

    /// Prefill: run the prompt through the paged path and advance the cache.
    pub fn prefill(&self, ids: &Tensor, cache: &PagedKvCache) -> Tensor {
        let logits = self.forward_paged(ids, cache);
        cache.advance(ids.size()[1]);
        logits
    }

    /// As `forward_paged`, but against a quantized KV cache. Each layer quantizes
    /// its fresh K/V into the packed store and dequantizes the prefix into the
    /// shared fp16 scratch pools before attending.
    pub fn forward_paged_quant(&self, ids: &Tensor, cache: &crate::cache::QuantPagedKvCache) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x = self.embed.forward(ids);
        let mut pending: Option<Tensor> = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            let ctx = Attn::PagedQuant {
                qk: &cache.qk[i],
                qv: &cache.qv[i],
                sk: &cache.sk[i],
                sv: &cache.sv[i],
                k_scratch: Some(&cache.k_scratch),
                v_scratch: Some(&cache.v_scratch),
                block_table: &cache.block_table,
                seqlens: &cache.seqlens,
                compand_a: cache.compand_a,
                rope_table: None,
            };
            pending = Some(blk.forward(&x, &ctx, pending.as_ref()));
        }
        self.head_res(&x, pending.as_ref())
    }

    pub fn prefill_quant(&self, ids: &Tensor, cache: &crate::cache::QuantPagedKvCache) -> Tensor {
        let logits = self.forward_paged_quant(ids, cache);
        cache.advance(ids.size()[1]);
        logits
    }

    /// Batched paged forward for the dynamic generator. `ids` is `[bsz, q_len]`
    /// (uniform q_len across the batch — the generator groups jobs by shape).
    /// `kc`/`vc` are the shared per-layer page pools; `block_table` `[bsz, P]` i32
    /// maps each row's logical pages to physical ones; `seqlens` `[bsz]` i32 is the
    /// per-row pre-append length (also the RoPE offset). Returns logits
    /// `[bsz, q_len, vocab]` (or `[bsz, 1, vocab]` if `last_only`) as f32.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged_batched(
        &self,
        ids: &Tensor,
        kc: &[Tensor],
        vc: &[Tensor],
        block_table: &Tensor,
        seqlens: &Tensor,
        last_only: bool,
    ) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x = self.embed.forward(ids);
        let mut pending: Option<Tensor> = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            let ctx = Attn::Paged {
                k_cache: &kc[i],
                v_cache: &vc[i],
                block_table,
                seqlens,
                rope_table: None,
            };
            pending = Some(blk.forward(&x, &ctx, pending.as_ref()));
        }
        self.finalize_head(x, pending, last_only)
    }

    /// As `forward_paged_batched`, but against a quantized paged cache.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged_batched_quant(
        &self,
        ids: &Tensor,
        qc: &crate::paged::QuantPagedCache,
        block_table: &Tensor,
        seqlens: &Tensor,
        last_only: bool,
    ) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x = self.embed.forward(ids);
        let mut pending: Option<Tensor> = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            let ctx = Attn::PagedQuant {
                qk: &qc.qk[i],
                qv: &qc.qv[i],
                sk: &qc.sk[i],
                sv: &qc.sv[i],
                k_scratch: None, // compact per-call scratch (batched path)
                v_scratch: None,
                block_table,
                seqlens,
                compand_a: qc.compand_a,
                rope_table: None,
            };
            pending = Some(blk.forward(&x, &ctx, pending.as_ref()));
        }
        self.finalize_head(x, pending, last_only)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
