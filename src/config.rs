//! Port of `model/config.py` + `architecture/{qwen3,qwen3_5}.py` (grade B / A).
//! Only the fields the Qwen3 / Qwen3.5 text paths consume are surfaced.

use crate::rope::{RopeSettings, RopeStyle};
use anyhow::{bail, Result};
use std::path::Path;

/// Which transformer family the checkpoint is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArchKind {
    /// Llama-shaped dense GQA stack: no QK-norm, no post-norms. Covers
    /// `LlamaForCausalLM`, `MistralForCausalLM` and `Qwen2ForCausalLM` — Qwen2
    /// differs from Llama only by bias on Q/K/V, and `Linear` picks the bias up
    /// from the checkpoint on its own.
    Llama,
    /// `Qwen3ForCausalLM` — homogeneous GQA transformer.
    Qwen3,
    /// `Qwen3MoeForCausalLM` — Qwen3 blocks with a block-sparse expert MLP.
    Qwen3Moe,
    /// `Glm4MoeForCausalLM` — GLM-4.5/4.6. Despite the name it is *not* the
    /// GLM4 block: NeoX RoPE rather than GPT-J, no sandwich norms, optional
    /// QK-norm, and a DeepSeek-style sigmoid router with a selection bias, a
    /// routed scaling factor, an always-on shared expert, and the first
    /// `first_k_dense_replace` layers left dense.
    Glm4Moe,
    /// `Qwen4ExpForConditionalGeneration` — Qwen3.8-Flash-Next. The Qwen3.5
    /// hybrid stack plus four things: gated-residual hyper-connections in place
    /// of the input/post layernorms, QSA sparse attention on the full-attention
    /// layers, a sigmoid GDN output gate, and PLE n-gram injection ahead of one
    /// or more early layers. No final model norm — the combine-less mixer
    /// collapses the stream stack before the head.
    Qwen4Exp,
    /// `Glm4ForCausalLM` — Llama-shaped but GPT-J RoPE and sandwich norms:
    /// each sublayer output is normed *before* its residual add.
    Glm4,
    /// `Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM` — hybrid
    /// gated-delta-net + gated full-attention, text path only.
    Qwen35,
}

/// Structural traits of a decoder block, read off `ArchKind` rather than
/// re-matched at every call site.
impl ArchKind {
    /// Per-head RMSNorm on Q and K, fused into the RoPE kernel. Qwen3 and
    /// Qwen3.5 have it; Llama-shaped stacks and GLM4 do not.
    /// Note this is only the *default*; GLM4-MoE gates it on `use_qk_norm` in
    /// `config.json`, so callers read `Config::qk_norm` rather than this.
    pub fn has_qk_norm(self) -> bool {
        matches!(self, ArchKind::Qwen3 | ArchKind::Qwen3Moe | ArchKind::Qwen35 | ArchKind::Qwen4Exp)
    }

    /// Sandwich norms: `post_self_attn_layernorm` / `post_mlp_layernorm`
    /// applied to each sublayer's output before it re-enters the residual.
    pub fn has_post_norms(self) -> bool {
        matches!(self, ArchKind::Glm4)
    }

    /// Hybrid linear-attention stack, loaded as `Q35Layer` rather than
    /// `TransformerBlock`.
    pub fn is_hybrid(self) -> bool {
        matches!(self, ArchKind::Qwen35 | ArchKind::Qwen4Exp)
    }

    /// Carries the Qwen3-VL vision tower, loaded and run by [`crate::vision`].
    /// `Qwen4ExpForConditionalGeneration` subclasses the identical tower with no
    /// changes upstream, so it is in this set too.
    pub fn has_qwen3_vl_tower(self) -> bool {
        matches!(self, ArchKind::Qwen35 | ArchKind::Qwen4Exp)
    }
}

/// Per-layer attention flavour for the hybrid Qwen3.5 stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerKind {
    FullAttention,
    LinearAttention,
}

/// How the router turns hidden states into (expert, weight) pairs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RouterKind {
    /// Qwen3-MoE: top-k over the raw logits, softmax over the selected k.
    Std,
    /// DeepSeek-V3 / GLM4-MoE (`routing_ds3_nogroup`, `router_type = "dots"`):
    /// score = `sigmoid(logit)`, selection ranks by `score + e_score_correction_bias`
    /// but the *weights* use the unbiased score, normalized over the selected
    /// set and multiplied by `routed_scaling_factor`.
    Dots,
}

/// Block-sparse (mixture-of-experts) MLP dimensions.
#[derive(Clone, Debug)]
pub struct MoeParams {
    pub num_experts: i64,
    pub num_experts_per_tok: i64,
    /// Per-expert MLP width — `moe_intermediate_size`, not `intermediate_size`.
    pub moe_intermediate_size: i64,
    /// Renormalize the top-k routing weights so they sum to 1. Qwen3-MoE
    /// requires it; the routing kernel softmaxes over the selected k, which is
    /// the same thing.
    pub norm_topk_prob: bool,
    /// Only every `decoder_sparse_step`-th layer is sparse (1 = all of them).
    pub decoder_sparse_step: i64,
    /// Layer indices forced to a plain dense MLP despite the schedule.
    pub mlp_only_layers: Vec<i64>,
    /// Routing flavour.
    pub router: RouterKind,
    /// The first N layers are dense regardless of the schedule (GLM4-MoE
    /// `first_k_dense_replace`; 0 on Qwen3-MoE).
    pub first_k_dense_replace: i64,
    /// `Dots` only: routed weights are scaled by this after normalization.
    pub routed_scaling_factor: f64,
    /// Width of the always-on shared expert running alongside the routed ones,
    /// or 0 when the architecture has none (Qwen3-MoE).
    pub shared_expert_intermediate_size: i64,
    /// The shared expert's output passes through `sigmoid(shared_expert_gate(x))`
    /// before it is added (Qwen2-MoE lineage, and qwen4_exp). GLM4-MoE has no
    /// such gate and adds the shared expert unweighted.
    pub shared_gate: bool,
}

impl MoeParams {
    /// Whether layer `idx` uses experts rather than a dense `GatedMlp`.
    pub fn is_sparse_layer(&self, idx: i64) -> bool {
        idx >= self.first_k_dense_replace
            && !self.mlp_only_layers.contains(&idx)
            && self.num_experts > 0
            && self.decoder_sparse_step > 0
            && (idx + 1) % self.decoder_sparse_step == 0
    }
}

/// Gated-delta-net dimensions (Qwen3.5 `linear_attention` layers).
#[derive(Clone, Copy, Debug)]
pub struct GdnParams {
    pub conv_kernel_size: i64,
    pub num_k_heads: i64,
    pub num_v_heads: i64,
    pub k_head_dim: i64,
    pub v_head_dim: i64,
    pub beta_scale: f32,
}

impl GdnParams {
    /// `2*Nk*Hk + Nv*Hv` — the conv1d / mixed_qkv feature width.
    pub fn fdim_qkv(&self) -> i64 {
        2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim
    }
    pub fn k_dim(&self) -> i64 {
        self.num_k_heads * self.k_head_dim
    }
    pub fn v_dim(&self) -> i64 {
        self.num_v_heads * self.v_head_dim
    }
}

/// qwen4_exp extras: everything the Qwen3.5 hybrid config does not already
/// cover. Hyper-connection width, the QSA indexer's shape, and the PLE /
/// n-gram embedding parameters.
#[derive(Clone, Debug)]
pub struct Qwen4Params {
    /// `hc_count` — parallel fp32 residual streams.
    pub hc_mult: i64,
    pub indexer_n_heads: i64,
    pub indexer_head_dim: i64,
    /// Tokens each query may attend to, before its own tail block.
    pub indexer_budget: i64,
    pub indexer_compress_ratio: i64,
    /// Layer indices (0-based here; `ple_layer_ids` in the checkpoint is 1-based)
    /// that carry a PLE injection ahead of the block.
    pub ple_layer_ids: Vec<i64>,
    pub ple_embed_dim: i64,
    pub ple_conv_kernel_size: i64,
    pub ngram_size: i64,
    pub heads_per_ngram: i64,
    /// The eos id the n-gram hashing segments on. Read from the *text* config,
    /// which may differ from the generator's stop ids.
    pub ple_eos_token_id: i64,
}

/// CLI / caller overrides applied on top of `config.json` at load time.
/// Mirrors the tabbyAPI `config.yml` knobs of the same name.
#[derive(Clone, Debug, Default)]
pub struct ConfigOverrides {
    /// Clamp `max_position_embeddings` (context length) to this.
    pub max_seq_len: Option<i64>,
    /// Linear RoPE position scaling (tabby `rope_scale`); `inv_freq /= scale`.
    pub rope_scale: Option<f64>,
    /// NTK-aware RoPE base scaling (tabby `rope_alpha`);
    /// `rope_theta *= alpha ** (rotary_dim / (rotary_dim - 2))`.
    pub rope_alpha: Option<f64>,
}

pub struct Config {
    pub arch: String,
    pub arch_kind: ArchKind,
    /// Per-head RMSNorm on Q and K. Defaults to `arch_kind.has_qk_norm()`, but
    /// GLM4-MoE carries the tensors only when `use_qk_norm` is set.
    pub qk_norm: bool,
    /// Weight-name prefix for the decoder stack (`"model"` or `"model.language_model"`).
    pub key_prefix: String,
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: i64,
    pub num_q_heads: i64,
    pub num_kv_heads: i64,
    pub head_dim: i64,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<i64>,
    pub eos_token_ids: Vec<i64>,
    pub max_position_embeddings: i64,
    pub rope: RopeSettings,
    // --- Qwen3.5 hybrid extras (empty / defaults for plain Qwen3) ---
    /// `constant_bias` added to every RMSNorm weight (1.0 for Qwen3.5, 0.0 for Qwen3).
    pub norm_constant_bias: f32,
    /// `attn_output_gate` — full-attention layers carry an interleaved output gate.
    pub attn_output_gate: bool,
    /// Per-layer flavour; `num_hidden_layers` entries. All `FullAttention` for plain Qwen3.
    pub layer_types: Vec<LayerKind>,
    pub gdn: Option<GdnParams>,
    /// Nonlinearity on the GDN output gate (`output_gate_type`). Silu everywhere
    /// but qwen4_exp, which selects sigmoid.
    pub gdn_gate_act: crate::ffi::GateAct,
    /// MoE params; `None` on dense checkpoints.
    pub moe: Option<MoeParams>,
    /// qwen4_exp extras; `None` on every other architecture.
    pub qwen4: Option<Qwen4Params>,
    /// Qwen-VL vision tower config, `None` for text-only checkpoints.
    pub vision: Option<VisionConfig>,
    /// `[t, h, w]` MRoPE section widths (freq dims per axis); `None` ⇒ plain RoPE.
    pub mrope_section: Option<[i64; 3]>,
    pub vision_start_token_id: i64,
    pub vision_end_token_id: i64,
    pub image_token_id: i64,
    pub raw: serde_json::Value,
}

/// `vision_config` for Qwen3-VL / Qwen3.5 (`qwen3_vl.py::read_qwen3_vl_vision_config`
/// + `read_qwen3_vl_pp_config`). Vision weights in this checkpoint are dense bf16.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub depth: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_heads: i64,
    pub head_dim: i64,
    pub out_hidden_size: i64,
    pub patch_size: i64,
    pub temporal_patch_size: i64,
    pub spatial_merge_size: i64,
    pub num_position_embeddings: i64,
    /// Vision block indexes whose output is tapped by a deepstack merger and fed
    /// back into the first N text layers. Empty on Qwen3.5 (and qwen4_exp);
    /// `[8, 16, 24]` on Qwen3-VL.
    pub deepstack_visual_indexes: Vec<i64>,
    pub layernorm_eps: f64,
    pub rope_theta: f64,
    // preprocessor_config.json
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    pub min_pixels: i64,
    pub max_pixels: i64,
}

fn get_i64(v: &serde_json::Value, key: &str) -> Option<i64> {
    v.get(key).and_then(|x| x.as_i64())
}
fn get_f64(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}
fn get_bool(v: &serde_json::Value, key: &str) -> Option<bool> {
    v.get(key).and_then(|x| x.as_bool())
}

impl Config {
    /// The paged-attention kernels are instantiated for GQA group ratios
    /// {1,2,3,4,5,6,7,8} at head_dim {128,256} (and {1,2,4,8} elsewhere). When
    /// `num_q_heads / num_kv_heads` isn't a supported ratio, the KV heads are
    /// repeated up to the smallest multiple of `num_kv_heads` that both divides
    /// `num_q_heads` and yields a supported ratio. Returns `(kv_heads_eff,
    /// repeat_factor)`; `repeat_factor == 1` when the native ratio is fine
    /// (which it is for Qwen3.5's 24/4 = 6 at head_dim 256).
    pub fn kv_heads_eff(&self) -> (i64, i64) {
        let wide = matches!(self.head_dim, 128 | 256);
        let ok = |g: i64| {
            (1..=8).contains(&g) && (wide || matches!(g, 1 | 2 | 4 | 8))
        };
        if self.num_q_heads % self.num_kv_heads == 0 && ok(self.num_q_heads / self.num_kv_heads) {
            return (self.num_kv_heads, 1);
        }
        for r in 2.. {
            let kv = self.num_kv_heads * r;
            if self.num_q_heads % kv == 0 && ok(self.num_q_heads / kv) {
                return (kv, r);
            }
            if kv >= self.num_q_heads {
                return (self.num_q_heads, self.num_q_heads / self.num_kv_heads);
            }
        }
        unreachable!()
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_with(dir, &ConfigOverrides::default())
    }

    pub fn from_dir_with(dir: &Path, ov: &ConfigOverrides) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let archs = raw["architectures"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("no architectures in config.json"))?;
        if archs.len() != 1 {
            bail!("expected exactly one architecture");
        }
        let arch = archs[0].as_str().unwrap().to_string();

        let (arch_kind, key_prefix) = match arch.as_str() {
            // MiMo, SeedOss and IQuestCoder are declared upstream as `LlamaModel`
            // subclasses with no overrides at all — only a distinct arch string
            // and chat template — so they are the Llama path exactly.
            "LlamaForCausalLM" | "MistralForCausalLM" | "Qwen2ForCausalLM"
            | "MiMoForCausalLM" | "SeedOssForCausalLM" | "IQuestCoderForCausalLM" => {
                (ArchKind::Llama, "model".to_string())
            }
            "Glm4ForCausalLM" => (ArchKind::Glm4, "model".to_string()),
            "Qwen3ForCausalLM" => (ArchKind::Qwen3, "model".to_string()),
            "Qwen3MoeForCausalLM" => (ArchKind::Qwen3Moe, "model".to_string()),
            // Qwen3-VL is the Qwen3 block with a vision tower bolted on and the
            // decoder moved under `model.language_model`; the block structure,
            // and so the ArchKind, is unchanged. Same for the MoE variant.
            "Qwen3VLForConditionalGeneration" => {
                (ArchKind::Qwen3, "model.language_model".to_string())
            }
            "Qwen3VLMoeForConditionalGeneration" => {
                (ArchKind::Qwen3Moe, "model.language_model".to_string())
            }
            "Qwen4ExpForConditionalGeneration" => {
                (ArchKind::Qwen4Exp, "model.language_model".to_string())
            }
            // SolarOpen is declared upstream as a Glm4Moe subclass with only a
            // distinct arch string and chat template, so it is this path exactly.
            "Glm4MoeForCausalLM" | "SolarOpenForCausalLM" => {
                (ArchKind::Glm4Moe, "model".to_string())
            }
            // DFlash2 speculative draft model. A Qwen3 block stack (so it reads
            // through the Qwen3 path) but rooted at the checkpoint top level —
            // its tensors are `layers.N.*`, `fc.*`, `norm.*`, with no `model.`
            // prefix, no embedding table and no `lm_head` (it borrows the
            // target's). The DFlash2-specific fields live in `dflash_config`
            // and are read by `DFlash2Config`.
            "DFlash2DraftModel" => (ArchKind::Qwen3, String::new()),
            "Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM" => {
                (ArchKind::Qwen35, "model.language_model".to_string())
            }
            other => bail!(
                "unsupported architecture {other}. This port implements \
                 Llama/Mistral/Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM, \
                 Glm4ForCausalLM, Glm4MoeForCausalLM, \
                 Qwen4ExpForConditionalGeneration and \
                 Qwen3_5For{{ConditionalGeneration,CausalLM}} (see PLAN.md)"
            ),
        };

        // Multimodal checkpoints nest the decoder config under `text_config`
        // (Qwen3.5, qwen4_exp, Qwen3-VL); text-only ones keep it at the root.
        // Keyed on presence rather than on the arch, so a VL variant of an
        // existing block structure needs no new case here.
        let tc: &serde_json::Value = raw.get("text_config").unwrap_or(&raw);

        let hidden_size = get_i64(tc, "hidden_size").unwrap();
        let num_q_heads = get_i64(tc, "num_attention_heads").unwrap();
        let num_kv_heads = get_i64(tc, "num_key_value_heads").unwrap_or(num_q_heads);
        let head_dim = get_i64(tc, "head_dim").unwrap_or(hidden_size / num_q_heads);
        let num_hidden_layers = get_i64(tc, "num_hidden_layers").unwrap();
        // `--max-seq-len` (tabby `max_seq_len`) overrides the checkpoint context
        // length. Used for cache sizing and by the YaRN/longrope ramps.
        let max_position_embeddings = match ov.max_seq_len {
            Some(m) if m > 0 => m,
            _ => get_i64(tc, "max_position_embeddings").unwrap_or(8192),
        };

        // eos/bos can live at the top level or under text_config.
        let eos_src = tc.get("eos_token_id").or_else(|| raw.get("eos_token_id"));
        let eos = match eos_src {
            Some(serde_json::Value::Array(a)) => {
                a.iter().filter_map(|v| v.as_i64()).collect()
            }
            Some(serde_json::Value::Number(n)) => vec![n.as_i64().unwrap()],
            _ => vec![],
        };
        let bos_token_id = get_i64(tc, "bos_token_id").or_else(|| get_i64(&raw, "bos_token_id"));

        // RoPE — Qwen3.5 keeps params in `text_config.rope_parameters` (partial rotary,
        // interleaved MRoPE which for a pure-text sequence collapses to standard partial NEOX).
        let rope_params = tc.get("rope_parameters").or_else(|| tc.get("rope_scaling"));
        let rope_theta = rope_params
            .and_then(|p| get_f64(p, "rope_theta"))
            .or_else(|| get_f64(tc, "rope_theta"))
            .unwrap_or(10000.0);
        let partial_rotary_factor = rope_params
            .and_then(|p| get_f64(p, "partial_rotary_factor"))
            .or_else(|| get_f64(tc, "partial_rotary_factor"))
            .unwrap_or(1.0);
        // Only pass a `rope_scaling` object through to RopeSettings when it actually selects
        // a non-default scaling type; interleaved MRoPE is handled as plain partial RoPE here.
        let rope_scaling = rope_params
            .filter(|p| {
                p.get("rope_type")
                    .and_then(|t| t.as_str())
                    .map(|t| !matches!(t, "default" | "mrope"))
                    .unwrap_or(false)
            })
            .cloned();

        // RoPE overrides (tabby `rope_alpha` / `rope_scale`). `rope_alpha` is
        // NTK-aware base scaling; `rope_scale` is linear position compression,
        // injected as a `linear` rope_scaling type (composes with any existing
        // scaling only if there was none — matches tabby, which forbids both).
        let rotary_dim_eff = get_i64(tc, "rotary_dim")
            .unwrap_or((head_dim as f64 * partial_rotary_factor) as i64);
        let rope_theta = match ov.rope_alpha {
            Some(a) if a > 0.0 && a != 1.0 => {
                rope_theta * a.powf(rotary_dim_eff as f64 / (rotary_dim_eff as f64 - 2.0))
            }
            _ => rope_theta,
        };
        let rope_scaling = match ov.rope_scale {
            Some(s) if s > 0.0 && s != 1.0 => Some(serde_json::json!({
                "rope_type": "linear",
                "factor": s,
            })),
            _ => rope_scaling,
        };

        let rope = RopeSettings {
            head_dim,
            rope_theta,
            rotary_dim: get_i64(tc, "rotary_dim"),
            partial_rotary_factor,
            max_position_embeddings: Some(max_position_embeddings),
            original_max_position_embeddings: get_i64(tc, "original_max_position_embeddings"),
            // GLM4 rotates adjacent pairs (GPT-J) rather than halves (NeoX);
            // everything else in this port is NeoX.
            rope_style: match arch_kind {
                ArchKind::Glm4 => RopeStyle::Gptj,
                _ => RopeStyle::Neox,
            },
            rope_scaling,
        };

        let rms_norm_eps = tc["rms_norm_eps"].as_f64().unwrap() as f32;
        let tie_word_embeddings = get_bool(&raw, "tie_word_embeddings")
            .or_else(|| get_bool(tc, "tie_word_embeddings"))
            .unwrap_or(false);

        // Hybrid layer schedule + GDN params (Qwen3.5 only).
        let (norm_constant_bias, attn_output_gate, layer_types, gdn) = match arch_kind {
            ArchKind::Llama
            | ArchKind::Qwen3
            | ArchKind::Qwen3Moe
            | ArchKind::Glm4
            | ArchKind::Glm4Moe => (
                0.0,
                false,
                vec![LayerKind::FullAttention; num_hidden_layers as usize],
                None,
            ),
            ArchKind::Qwen35 | ArchKind::Qwen4Exp => {
                let interval = get_i64(tc, "full_attention_interval").unwrap_or(4);
                let layer_types = match tc.get("layer_types").and_then(|v| v.as_array()) {
                    Some(a) => {
                        if a.len() != num_hidden_layers as usize {
                            bail!("text_config.layer_types length != num_hidden_layers");
                        }
                        a.iter()
                            .map(|v| match v.as_str() {
                                Some("full_attention") => Ok(LayerKind::FullAttention),
                                Some("linear_attention") => Ok(LayerKind::LinearAttention),
                                other => bail!("unknown layer type {other:?}"),
                            })
                            .collect::<Result<Vec<_>>>()?
                    }
                    None => (0..num_hidden_layers)
                        .map(|i| {
                            if (i + 1) % interval == 0 {
                                LayerKind::FullAttention
                            } else {
                                LayerKind::LinearAttention
                            }
                        })
                        .collect(),
                };
                let gdn = GdnParams {
                    conv_kernel_size: get_i64(tc, "linear_conv_kernel_dim").unwrap_or(4),
                    num_k_heads: get_i64(tc, "linear_num_key_heads").unwrap_or(16),
                    num_v_heads: get_i64(tc, "linear_num_value_heads").unwrap_or(32),
                    k_head_dim: get_i64(tc, "linear_key_head_dim").unwrap_or(128),
                    v_head_dim: get_i64(tc, "linear_value_head_dim").unwrap_or(128),
                    beta_scale: 1.0,
                };
                (
                    1.0,
                    get_bool(tc, "attn_output_gate").unwrap_or(false),
                    layer_types,
                    Some(gdn),
                )
            }
        };

        // GLM4-MoE checkpoints append `num_nextn_predict_layers` MTP layers to
        // the tensor file but keep `num_hidden_layers` at the trunk depth, so no
        // adjustment is needed here — the extra layers are simply never loaded.
        let moe = match arch_kind {
            ArchKind::Glm4Moe => {
                let moe_interm = get_i64(tc, "moe_intermediate_size")
                    .ok_or_else(|| anyhow::anyhow!("MoE checkpoint has no moe_intermediate_size"))?;
                let n_shared = get_i64(tc, "n_shared_experts").unwrap_or(1);
                Some(MoeParams {
                    num_experts: get_i64(tc, "n_routed_experts").unwrap_or(128),
                    num_experts_per_tok: get_i64(tc, "num_experts_per_tok").unwrap_or(8),
                    moe_intermediate_size: moe_interm,
                    // Upstream asserts it; the `dots` router normalizes over the
                    // selected set unconditionally, which is the same thing.
                    norm_topk_prob: true,
                    decoder_sparse_step: 1,
                    mlp_only_layers: vec![],
                    router: RouterKind::Dots,
                    first_k_dense_replace: get_i64(tc, "first_k_dense_replace").unwrap_or(3),
                    routed_scaling_factor: get_f64(tc, "routed_scaling_factor").unwrap_or(2.5),
                    // The shared expert is one GatedMLP `n_shared_experts` times
                    // as wide as a routed one, not `n_shared_experts` of them.
                    shared_expert_intermediate_size: moe_interm * n_shared,
                    shared_gate: false,
                })
            }
            ArchKind::Qwen3Moe => Some(MoeParams {
                num_experts: get_i64(tc, "num_experts")
                    .or_else(|| get_i64(tc, "num_local_experts"))
                    .ok_or_else(|| anyhow::anyhow!("MoE checkpoint has no num_experts"))?,
                num_experts_per_tok: get_i64(tc, "num_experts_per_tok")
                    .ok_or_else(|| anyhow::anyhow!("MoE checkpoint has no num_experts_per_tok"))?,
                moe_intermediate_size: get_i64(tc, "moe_intermediate_size")
                    .ok_or_else(|| anyhow::anyhow!("MoE checkpoint has no moe_intermediate_size"))?,
                // Upstream asserts this is true for Qwen3-MoE and the routing
                // kernel has no un-normalized mode, so refuse rather than
                // silently produce different weights.
                norm_topk_prob: {
                    let n = get_bool(tc, "norm_topk_prob").unwrap_or(true);
                    if !n {
                        bail!("norm_topk_prob = false is not implemented");
                    }
                    n
                },
                decoder_sparse_step: get_i64(tc, "decoder_sparse_step").unwrap_or(1),
                mlp_only_layers: tc
                    .get("mlp_only_layers")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
                    .unwrap_or_default(),
                router: RouterKind::Std,
                first_k_dense_replace: 0,
                routed_scaling_factor: 1.0,
                shared_expert_intermediate_size: 0,
                shared_gate: false,
            }),
            // qwen4_exp: plain top-k softmax routing like Qwen3-MoE, but with a
            // sigmoid-gated shared expert alongside (the Qwen2-MoE arrangement).
            ArchKind::Qwen4Exp => {
                let moe_interm = get_i64(tc, "moe_intermediate_size")
                    .ok_or_else(|| anyhow::anyhow!("MoE checkpoint has no moe_intermediate_size"))?;
                Some(MoeParams {
                    num_experts: get_i64(tc, "num_experts")
                        .or_else(|| get_i64(tc, "num_local_experts"))
                        .ok_or_else(|| anyhow::anyhow!("MoE checkpoint has no num_experts"))?,
                    num_experts_per_tok: get_i64(tc, "num_experts_per_tok")
                        .ok_or_else(|| anyhow::anyhow!("MoE checkpoint has no num_experts_per_tok"))?,
                    moe_intermediate_size: moe_interm,
                    norm_topk_prob: true,
                    decoder_sparse_step: 1,
                    mlp_only_layers: vec![],
                    router: RouterKind::Std,
                    first_k_dense_replace: 0,
                    routed_scaling_factor: 1.0,
                    shared_expert_intermediate_size: get_i64(tc, "shared_expert_intermediate_size")
                        .unwrap_or(moe_interm),
                    shared_gate: true,
                })
            }
            _ => None,
        };

        // --- qwen4_exp extras ---
        let qwen4 = match arch_kind {
            ArchKind::Qwen4Exp => Some(Qwen4Params {
                hc_mult: get_i64(tc, "hc_count").unwrap_or(4),
                indexer_n_heads: get_i64(tc, "indexer_n_heads")
                    .ok_or_else(|| anyhow::anyhow!("qwen4_exp config has no indexer_n_heads"))?,
                indexer_head_dim: get_i64(tc, "indexer_head_dim")
                    .ok_or_else(|| anyhow::anyhow!("qwen4_exp config has no indexer_head_dim"))?,
                indexer_budget: get_i64(tc, "indexer_budget")
                    .ok_or_else(|| anyhow::anyhow!("qwen4_exp config has no indexer_budget"))?,
                indexer_compress_ratio: get_i64(tc, "indexer_compress_ratio").unwrap_or(4),
                // 1-based in the checkpoint, 0-based here.
                ple_layer_ids: tc
                    .get("ple_layer_ids")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).map(|i| i - 1).collect())
                    .unwrap_or_default(),
                ple_embed_dim: get_i64(tc, "ple_embed_dim").unwrap_or(0),
                ple_conv_kernel_size: get_i64(tc, "ple_conv_kernel_size").unwrap_or(4),
                ngram_size: get_i64(tc, "ngram_size").unwrap_or(0),
                heads_per_ngram: get_i64(tc, "heads_per_ngram").unwrap_or(0),
                ple_eos_token_id: get_i64(tc, "eos_token_id")
                    .or_else(|| eos.first().copied())
                    .unwrap_or(0),
            }),
            _ => None,
        };

        // --- vision tower (Qwen3-VL / Qwen3.5) ---
        let mrope_section = rope_params
            .and_then(|p| p.get("mrope_section"))
            .and_then(|v| v.as_array())
            .filter(|a| a.len() == 3)
            .map(|a| {
                [
                    a[0].as_i64().unwrap_or(0),
                    a[1].as_i64().unwrap_or(0),
                    a[2].as_i64().unwrap_or(0),
                ]
            });
        let vision = match raw.get("vision_config") {
            Some(vc) if get_i64(vc, "depth").is_some() => {
                let hs = get_i64(vc, "hidden_size").unwrap();
                let nh = get_i64(vc, "num_heads").unwrap();
                let deepstack_visual_indexes: Vec<i64> = vc
                    .get("deepstack_visual_indexes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
                    .unwrap_or_default();
                let prep: serde_json::Value = std::fs::read(dir.join("preprocessor_config.json"))
                    .ok()
                    .and_then(|b| serde_json::from_slice(&b).ok())
                    .unwrap_or(serde_json::json!({}));
                let arr3 = |v: &serde_json::Value, k: &str, d: [f64; 3]| -> [f64; 3] {
                    v.get(k)
                        .and_then(|a| a.as_array())
                        .filter(|a| a.len() == 3)
                        .map(|a| [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()])
                        .unwrap_or(d)
                };
                let size = prep.get("size");
                let short = size
                    .and_then(|s| get_i64(s, "shortest_edge"))
                    .unwrap_or(4 * 28 * 28);
                let long = size
                    .and_then(|s| get_i64(s, "longest_edge"))
                    .unwrap_or(16384 * 28 * 28);
                Some(VisionConfig {
                    depth: get_i64(vc, "depth").unwrap(),
                    hidden_size: hs,
                    intermediate_size: get_i64(vc, "intermediate_size").unwrap(),
                    num_heads: nh,
                    head_dim: hs / nh,
                    out_hidden_size: get_i64(vc, "out_hidden_size").unwrap_or(hidden_size),
                    patch_size: get_i64(vc, "patch_size").unwrap_or(16),
                    temporal_patch_size: get_i64(vc, "temporal_patch_size").unwrap_or(2),
                    spatial_merge_size: get_i64(vc, "spatial_merge_size").unwrap_or(2),
                    num_position_embeddings: get_i64(vc, "num_position_embeddings").unwrap_or(2304),
                    deepstack_visual_indexes,
                    layernorm_eps: 1e-6,
                    rope_theta: 10000.0,
                    image_mean: arr3(&prep, "image_mean", [0.5, 0.5, 0.5]),
                    image_std: arr3(&prep, "image_std", [0.5, 0.5, 0.5]),
                    min_pixels: short,
                    max_pixels: long,
                })
            }
            _ => None,
        };
        let vision_start_token_id = get_i64(&raw, "vision_start_token_id").unwrap_or(151652);
        let vision_end_token_id = get_i64(&raw, "vision_end_token_id").unwrap_or(151653);
        let image_token_id = get_i64(&raw, "image_token_id")
            .or_else(|| get_i64(tc, "image_token_id"))
            .unwrap_or(151655);

        Ok(Config {
            arch,
            arch_kind,
            qk_norm: match arch_kind {
                ArchKind::Glm4Moe => get_bool(tc, "use_qk_norm").unwrap_or(false),
                k => k.has_qk_norm(),
            },
            key_prefix,
            vocab_size: get_i64(tc, "vocab_size")
                .or_else(|| get_i64(&raw, "vocab_size"))
                .unwrap(),
            hidden_size,
            intermediate_size: get_i64(tc, "intermediate_size").unwrap(),
            num_hidden_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            tie_word_embeddings,
            bos_token_id,
            eos_token_ids: eos,
            max_position_embeddings,
            rope,
            norm_constant_bias,
            attn_output_gate,
            layer_types,
            gdn,
            gdn_gate_act: match tc.get("output_gate_type").and_then(|v| v.as_str()) {
                Some("sigmoid") => crate::ffi::GateAct::Sigmoid,
                _ => crate::ffi::GateAct::Silu,
            },
            moe,
            qwen4,
            vision,
            mrope_section,
            vision_start_token_id,
            vision_end_token_id,
            image_token_id,
            raw,
        })
    }
}
