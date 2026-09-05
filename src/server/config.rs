//! `config.yml` parser — a drop-in for the TabbyAPI config file.
//!
//! Every documented key is accepted (unknown keys are ignored, not rejected).
//! Which keys are actually *honored* by this server is documented on each field
//! and summarised by [`ServerConfig::report`]. Keys that are parsed but not yet
//! acted on are marked `(accepted, not honored)`.

use serde::Deserialize;
use std::path::{Path, PathBuf};

fn de_string_or_number<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // cache_mode / rope_alpha etc. may be written as `Q4`, `8,8`, `1.0` or left blank.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum V {
        S(String),
        F(f64),
        I(i64),
        Null,
    }
    Ok(match Option::<V>::deserialize(d)? {
        None | Some(V::Null) => None,
        Some(V::S(s)) if s.trim().is_empty() => None,
        Some(V::S(s)) => Some(s),
        Some(V::F(f)) => Some(f.to_string()),
        Some(V::I(i)) => Some(i.to_string()),
    })
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct NetworkCfg {
    pub host: Option<String>,
    pub port: Option<u16>,
    /// Skip API-key auth entirely. Honored.
    pub disable_auth: bool,
    /// Refuse to fetch remote image URLs in vision requests. Honored.
    pub disable_fetch_requests: bool,
    /// Include Rust backtraces / error detail in API error bodies. Honored.
    pub send_tracebacks: bool,
    /// Only `OAI` is implemented; `Kobold` is ignored.
    pub api_servers: Option<Vec<String>>,
    /// Seconds between SSE keep-alive comments (0 disables). Honored.
    pub sse_ping_interval: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct LoggingCfg {
    pub log_prompt: bool,
    pub log_generation_params: bool,
    pub log_requests: bool,
    pub log_chat_completion_requests: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ModelCfg {
    /// Directory searched for models. Honored.
    pub model_dir: Option<String>,
    /// (accepted, not honored) runtime model loading from requests.
    pub inline_model_loading: bool,
    /// (accepted, not honored)
    pub use_dummy_models: bool,
    /// Extra names reported by `/v1/models` alongside the loaded model. Honored.
    pub dummy_model_names: Option<Vec<String>>,
    /// Model to load on startup (a sub-directory of `model_dir`). Honored.
    pub model_name: Option<String>,
    /// (accepted, not honored)
    pub use_as_default: Option<Vec<String>>,
    /// (accepted, not honored) — always exllamav3 here.
    #[serde(deserialize_with = "de_string_or_number")]
    pub backend: Option<String>,

    /// Context length. Honored (feeds `ConfigOverrides::max_seq_len` + cache size).
    /// `-1` means "pull from config.json".
    pub max_seq_len: Option<i64>,
    /// KV cache size in tokens (rounded up to a multiple of 256). Honored.
    pub cache_size: Option<i64>,
    /// `FP16` | `Q8` | `Q6` | `Q4` | `k,v` bit pair. Honored (non-hybrid arch).
    #[serde(deserialize_with = "de_string_or_number")]
    pub cache_mode: Option<String>,

    /// (accepted, not honored) — single GPU only in this port.
    pub tensor_parallel: bool,
    #[serde(deserialize_with = "de_string_or_number")]
    pub tensor_parallel_backend: Option<String>,
    pub gpu_split_auto: Option<bool>,
    pub autosplit_reserve: Option<Vec<f64>>,
    pub gpu_split: Option<Vec<f64>>,
    /// (accepted, not honored) — MoE CPU offload not ported.
    pub cpu_moe_offload_layers: Option<i64>,
    pub cpu_moe_split_experts: Option<i64>,
    pub cpu_moe_threads: Option<i64>,

    /// Linear RoPE position scaling. Honored.
    pub rope_scale: Option<f64>,
    /// NTK-aware RoPE base scaling (`auto` is treated as unset). Honored.
    #[serde(deserialize_with = "de_string_or_number")]
    pub rope_alpha: Option<String>,
    /// Prompt-ingestion chunk size. Honored.
    pub chunk_size: Option<i64>,
    /// (accepted, not honored) — cache is always chunk-allocated here.
    pub output_chunking: Option<bool>,
    /// Max concurrent generation jobs (batched path). Honored.
    pub max_batch_size: Option<usize>,
    /// Share identical prompt-prefix KV pages between requests (`pagetable.py`
    /// prefix cache). Honored (non-hybrid architectures only).
    pub prefix_cache: bool,
    /// Fair-scheduling requeue budget: a job past this many generated tokens is
    /// transparently reaped and re-enqueued so its pages return to the pool
    /// (tabbyAPI `max_rq_tokens`). Honored. `0` / unset = off.
    pub max_rq_tokens: Option<i64>,
    /// Pinned host-RAM KV cache tier size in tokens (`generator/cpu_cache.py`).
    /// Honored (non-hybrid architectures only); implies `prefix_cache`.
    pub cpu_cache_tokens: Option<i64>,

    /// Named template in `tokenizer_config.json`, or a raw Jinja string.
    /// A named selection is honored; a raw Jinja string falls back to ChatML.
    #[serde(deserialize_with = "de_string_or_number")]
    pub prompt_template: Option<String>,

    /// Enable the vision tower. Honored (Qwen3.5).
    pub vision: bool,
    /// (accepted, not honored)
    pub vision_offload: bool,

    /// Merged into every chat request's template vars. Honored for the keys the
    /// built-in ChatML renderer understands (`enable_thinking`).
    pub template_vars_default: Option<serde_yaml_ng::Value>,
    pub template_vars_force: Option<serde_yaml_ng::Value>,

    /// Split `<think>…</think>` into `reasoning_content`. Honored.
    pub reasoning: bool,
    pub reasoning_start_token: Option<String>,
    pub reasoning_end_token: Option<String>,
    /// `auto` | `always` | `never`. Honored.
    #[serde(deserialize_with = "de_string_or_number")]
    pub start_in_reasoning: Option<String>,
    /// (accepted, not honored)
    pub tool_calls_in_reasoning: Option<bool>,
    /// (accepted, not honored) — reasoning budget enforcement not ported.
    pub reasoning_budget_tokens: Option<i64>,
    pub reasoning_budget_message: Option<String>,

    /// `qwen3_coder` / `hermes` style `<tool_call>` JSON parsing. Honored
    /// (any non-empty value enables Qwen-style tool-call extraction).
    #[serde(deserialize_with = "de_string_or_number")]
    pub tool_format: Option<String>,
    /// (accepted, not honored) — Harmony / Muse Glimmer formats not ported.
    pub harmony: Option<bool>,
    pub muse_glimmer: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct DraftCfg {
    /// `model` | `disabled` | `mtp` | `ngram`. Honored.
    #[serde(deserialize_with = "de_string_or_number")]
    pub draft_mode: Option<String>,
    pub draft_model_dir: Option<String>,
    /// (accepted, not honored) — a separate draft *model* is not ported; use
    /// `mtp` (self-speculation) or `ngram`.
    #[serde(deserialize_with = "de_string_or_number")]
    pub draft_model_name: Option<String>,
    pub draft_rope_scale: Option<f64>,
    #[serde(deserialize_with = "de_string_or_number")]
    pub draft_rope_alpha: Option<String>,
    #[serde(deserialize_with = "de_string_or_number")]
    pub draft_cache_mode: Option<String>,
    pub draft_gpu_split: Option<Vec<f64>>,
    /// Draft length per iteration (`mtp` and `ngram`). Honored.
    pub draft_num_tokens: Option<i64>,
    /// (accepted, not honored) — dynamic draft length not ported.
    pub dynamic_draft: Option<bool>,
    /// Minimum suffix-match length for `ngram` drafting. Honored.
    pub ngram_match_min: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct SamplingCfg {
    /// A preset name from `sampler-overrides/`. `safe_defaults` is recognised;
    /// any other name is loaded from `<config_dir>/sampler-overrides/<name>.yml`
    /// if present, else ignored. Honored (as request-parameter fallbacks).
    #[serde(deserialize_with = "de_string_or_number")]
    pub override_preset: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct DeveloperCfg {
    pub unsafe_launch: bool,
    /// Force every response non-streaming. Honored.
    pub disable_request_streaming: bool,
    pub realtime_process_priority: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ServerConfig {
    pub network: NetworkCfg,
    pub logging: LoggingCfg,
    pub model: ModelCfg,
    pub draft_model: DraftCfg,
    pub sampling: SamplingCfg,
    pub developer: DeveloperCfg,
    // lora / embeddings / memory are accepted but not acted on.
    #[serde(default)]
    pub lora: serde_yaml_ng::Value,
    #[serde(default)]
    pub embeddings: serde_yaml_ng::Value,
    #[serde(default)]
    pub memory: serde_yaml_ng::Value,
}

/// Sampler-parameter fallbacks applied when a request omits a field.
#[derive(Debug, Clone)]
pub struct SamplerDefaults {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub repetition_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
}

impl Default for SamplerDefaults {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl ServerConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let cfg: ServerConfig = serde_yaml_ng::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;
        Ok(cfg)
    }

    pub fn model_dir(&self) -> PathBuf {
        let d = self.model.model_dir.as_deref().unwrap_or("models");
        // Relative to the process CWD, not the config file's directory — a
        // relative PathBuf resolves against CWD in any subsequent fs call, so
        // just hand it back as-is.
        PathBuf::from(d)
    }

    pub fn model_path(&self) -> anyhow::Result<PathBuf> {
        let name = self
            .model
            .model_name
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("config.model.model_name is required"))?;
        Ok(self.model_dir().join(name))
    }

    pub fn host(&self) -> String {
        self.network.host.clone().unwrap_or_else(|| "127.0.0.1".into())
    }
    pub fn port(&self) -> u16 {
        self.network.port.unwrap_or(5000)
    }

    /// `rope_alpha` as a number (`auto` / blank → None).
    pub fn rope_alpha(&self) -> Option<f64> {
        let s = self.model.rope_alpha.as_deref()?.trim().to_lowercase();
        if s.is_empty() || s == "auto" || s == "none" {
            return None;
        }
        s.parse().ok()
    }
    pub fn draft_rope_alpha(&self) -> Option<f64> {
        let s = self.draft_model.draft_rope_alpha.as_deref()?.trim().to_lowercase();
        if s.is_empty() || s == "auto" || s == "none" {
            return None;
        }
        s.parse().ok()
    }

    /// KV-cache bit width from `cache_mode` (0 = fp16). Accepts `FP16/Q8/Q6/Q4`
    /// and the `k,v` pair form (the lower of the two is used).
    pub fn cache_bits(&self) -> anyhow::Result<i64> {
        let Some(m) = self.model.cache_mode.as_deref() else {
            return Ok(0);
        };
        parse_cache_mode(m)
    }

    pub fn sampler_defaults(&self) -> SamplerDefaults {
        let mut d = SamplerDefaults::default();
        if self.sampling.override_preset.as_deref() == Some("safe_defaults") {
            // TabbyAPI's noob-friendly fallbacks.
            d.temperature = 1.0;
            d.top_p = 1.0;
            d.top_k = 0;
            d.min_p = 0.05;
            d.repetition_penalty = 1.0;
        }
        d
    }

    /// Effective context length: `min(max_position_embeddings-ish, cache_size)`
    /// resolution is left to `ConfigOverrides`; this returns the explicit request.
    pub fn ctx_len(&self) -> Option<i64> {
        match (self.model.max_seq_len, self.model.cache_size) {
            (Some(a), Some(b)) if a > 0 && b > 0 => Some(a.min(b)),
            (Some(a), _) if a > 0 => Some(a),
            (_, Some(b)) if b > 0 => Some(b),
            _ => None,
        }
    }

    pub fn draft_mode(&self) -> DraftMode {
        match self
            .draft_model
            .draft_mode
            .as_deref()
            .unwrap_or("model")
            .trim()
            .to_lowercase()
            .as_str()
        {
            "mtp" => DraftMode::Mtp,
            "dflash2" | "dflash" => {
                if self.draft_model.draft_model_dir.is_some() {
                    DraftMode::DFlash2
                } else {
                    DraftMode::Disabled
                }
            }
            "ngram" => DraftMode::Ngram,
            "disabled" => DraftMode::Disabled,
            // `model` mode: a separate AR draft model, if a directory is given.
            "model" => {
                if self.draft_model.draft_model_dir.is_some() {
                    DraftMode::Draft
                } else {
                    DraftMode::Disabled
                }
            }
            _ => DraftMode::Disabled,
        }
    }

    /// Draft model directory for `draft_mode: model`.
    pub fn draft_model_path(&self) -> Option<std::path::PathBuf> {
        // Relative to the process CWD, as for `model_dir` above.
        let dir = std::path::PathBuf::from(self.draft_model.draft_model_dir.as_ref()?);
        // `draft_model_dir` is a search directory, as for the main model; join
        // `draft_model_name` when one is given, and otherwise treat the dir as
        // the checkpoint itself (how it behaved before names were honoured).
        Some(match self.draft_model.draft_model_name.as_deref().map(str::trim) {
            Some(n) if !n.is_empty() => dir.join(n),
            _ => dir,
        })
    }

    /// `enable_thinking` default resolved from `reasoning` + template vars.
    pub fn enable_thinking_default(&self) -> Option<bool> {
        for v in [&self.model.template_vars_force, &self.model.template_vars_default] {
            if let Some(serde_yaml_ng::Value::Mapping(m)) = v {
                if let Some(x) = m.get(serde_yaml_ng::Value::from("enable_thinking")) {
                    if let Some(b) = x.as_bool() {
                        return Some(b);
                    }
                }
            }
        }
        None
    }

    /// Human-readable summary of what was applied, printed at startup.
    pub fn report(&self) -> String {
        let mut s = String::new();
        let p = |s: &mut String, k: &str, v: String| {
            s.push_str(&format!("  {k:<22} {v}\n"));
        };
        p(&mut s, "model", self.model_path().map(|p| p.display().to_string()).unwrap_or_default());
        p(&mut s, "listen", format!("{}:{}", self.host(), self.port()));
        p(&mut s, "auth", if self.network.disable_auth { "disabled".into() } else { "api key required".into() });
        p(&mut s, "max_seq_len", self.ctx_len().map(|v| v.to_string()).unwrap_or_else(|| "model default".into()));
        p(&mut s, "cache_mode", self.model.cache_mode.clone().unwrap_or_else(|| "FP16".into()));
        p(&mut s, "chunk_size", self.model.chunk_size.map(|v| v.to_string()).unwrap_or_else(|| "one-shot".into()));
        p(&mut s, "rope_scale", self.model.rope_scale.map(|v| v.to_string()).unwrap_or_else(|| "1.0".into()));
        p(&mut s, "rope_alpha", self.rope_alpha().map(|v| v.to_string()).unwrap_or_else(|| "model default".into()));
        p(&mut s, "max_batch_size", self.model.max_batch_size.map(|v| v.to_string()).unwrap_or_else(|| "auto".into()));
        p(&mut s, "prefix_cache", self.model.prefix_cache.to_string());
        p(&mut s, "max_rq_tokens", self.model.max_rq_tokens.map(|v| v.to_string()).unwrap_or_else(|| "off".into()));
        p(&mut s, "cpu_cache_tokens", self.model.cpu_cache_tokens.map(|v| v.to_string()).unwrap_or_else(|| "off".into()));
        p(&mut s, "draft_mode", format!("{:?}", self.draft_mode()));
        p(&mut s, "draft_num_tokens", self.draft_model.draft_num_tokens.map(|v| v.to_string()).unwrap_or_else(|| "4".into()));
        p(&mut s, "vision", self.model.vision.to_string());
        p(&mut s, "reasoning", self.model.reasoning.to_string());
        p(&mut s, "tool_format", self.model.tool_format.clone().unwrap_or_else(|| "off".into()));
        p(&mut s, "sampler preset", self.sampling.override_preset.clone().unwrap_or_else(|| "none".into()));
        s
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftMode {
    Disabled,
    Mtp,
    Ngram,
    Draft,
    /// DFlash2 block drafter — one forward proposes a whole block of tokens,
    /// then a candidate selector walks a coherent path through them. Needs
    /// `draft_model_dir` + `draft_model_name` pointing at a DFlash2 checkpoint
    /// trained for this target.
    DFlash2,
}

pub fn parse_cache_mode(m: &str) -> anyhow::Result<i64> {
    let m = m.trim();
    if let Some((a, b)) = m.split_once(',') {
        let a: i64 = a.trim().parse().map_err(|_| anyhow::anyhow!("bad cache_mode {m:?}"))?;
        let b: i64 = b.trim().parse().map_err(|_| anyhow::anyhow!("bad cache_mode {m:?}"))?;
        return Ok(a.min(b));
    }
    Ok(match m.to_ascii_uppercase().as_str() {
        "FP16" | "F16" | "" => 0,
        "Q8" | "8" => 8,
        "Q6" | "6" => 6,
        "Q4" | "4" => 4,
        "Q3" | "3" => 3,
        "Q2" | "2" => 2,
        other => anyhow::bail!("unknown cache_mode {other:?} (FP16|Q8|Q6|Q4 or `k,v`)"),
    })
}
