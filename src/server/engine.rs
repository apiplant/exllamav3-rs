//! The inference engine thread. Owns the model, tokenizer, `Generator`, and
//! (optionally) the MTP head / vision tower. All CUDA work happens here on one
//! OS thread; HTTP handlers talk to it over channels.
//!
//! Text requests all run through the batched [`Generator`] (continuous batching +
//! optional MTP / draft-model / n-gram speculative decode + KV-cache quant, per
//! `config.yml`).
//! An image request is served one-at-a-time through the vision tower + MRoPE
//! path, which blocks the engine while it runs.
//!
//! libtorch errors (OOM most often) `unwrap` inside `tch`, so the CUDA calls are
//! wrapped in `catch_cuda` — a panic there becomes a clean per-request error and
//! the engine keeps serving instead of the whole server dying.

use crate::async_gen::AsyncGenerator;
use crate::config::{ArchKind, ConfigOverrides};
use crate::generator::{Generator, JobSpec, Stage};
use crate::model::Model;
use crate::mtp::MtpModel;
use crate::sampler::SamplerSettings;
use crate::server::config::{DraftMode, ServerConfig};
use crate::server::oai::FinishReason;
use crate::tokenizer::Tok;
use crate::vision::{mrope_angle_table, mrope_pos_ids, VisionModel};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};
use tch::{Device, Kind, Tensor};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Load the model with an animated `⠋ loading model ▕███░░░▏ 47% (3s)` bar on
/// stderr driven by real per-layer load progress (same as `bin/infer`). Plain
/// line when stderr is not a TTY.
fn load_model_bar(
    dir: &std::path::Path,
    device: Device,
    ov: &crate::config::ConfigOverrides,
) -> Result<Model> {
    if !std::io::stderr().is_terminal() {
        eprintln!("loading model...");
        return Model::load_with(dir, device, ov);
    }
    let permil = Arc::new(AtomicU32::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let anim = {
        let (permil, done) = (Arc::clone(&permil), Arc::clone(&done));
        std::thread::spawn(move || {
            let start = Instant::now();
            const W: usize = 28;
            let mut frame = 0usize;
            while !done.load(Ordering::Relaxed) {
                let frac = permil.load(Ordering::Relaxed) as f32 / 1000.0;
                let fill = (frac * W as f32).round() as usize;
                let bar: String = "█".repeat(fill) + &"░".repeat(W - fill);
                eprint!(
                    "\r\x1b[2K\x1b[36m{}\x1b[0m loading model \x1b[32m▕{}▏\x1b[0m {:3.0}% \x1b[2m({:.0}s)\x1b[0m",
                    SPINNER[frame % SPINNER.len()],
                    bar,
                    frac * 100.0,
                    start.elapsed().as_secs_f64(),
                );
                std::io::stderr().flush().ok();
                frame += 1;
                std::thread::sleep(Duration::from_millis(80));
            }
        })
    };
    let cb = {
        let permil = Arc::clone(&permil);
        move |f: f32| permil.store((f * 1000.0) as u32, Ordering::Relaxed)
    };
    let r = Model::load_with_progress(dir, device, ov, Some(&cb));
    done.store(true, Ordering::Relaxed);
    anim.join().ok();
    eprint!("\r\x1b[2K");
    std::io::stderr().flush().ok();
    Ok(r?)
}

/// What the engine streams back for one request.
#[derive(Debug, Clone)]
pub enum Chunk {
    /// visible answer text delta
    Content(String),
    /// chain-of-thought delta (`reasoning_content`)
    Reasoning(String),
    /// terminal event
    Done {
        finish: FinishReason,
        completion_tokens: usize,
        stats: GenStats,
    },
    Error(String),
}

/// Per-request inference accounting, for the TabbyAPI-style `Metrics` log line.
#[derive(Debug, Clone, Default)]
pub struct GenStats {
    /// prompt tokens served from a shared prefix / CPU-cache page (not re-prefilled)
    pub cached_prompt_tokens: usize,
    /// prompt tokens actually ingested this request
    pub new_prompt_tokens: usize,
    /// wall time spent prefilling this request's prompt
    pub prefill_secs: f64,
    /// wall time from end of prefill to eos
    pub gen_secs: f64,
    /// speculative-decode draft tokens accepted / drafted for this request
    pub draft_accepted: u64,
    pub draft_total: u64,
}

/// Constrained-decoding request, from `response_format` / `guided_*` (feature 4).
pub enum GrammarSpec {
    /// any well-formed JSON object (`response_format: {type: json_object}`)
    JsonObject,
    /// JSON-Schema subset (`response_format: {type: json_schema, ...}`)
    JsonSchema(serde_json::Value),
    /// raw GBNF-subset grammar string (`response_format: {type: grammar, ...}`)
    Gbnf(String),
}

/// A generation request handed to the engine.
pub struct EngineRequest {
    pub prompt_ids: Vec<i64>,
    pub images: Vec<String>,
    pub max_new: usize,
    pub min_new: usize,
    pub sampler: SamplerSettings,
    pub stop_strings: Vec<String>,
    pub seed: Option<i64>,
    /// constrained decoding (`None` = unconstrained)
    pub grammar: Option<GrammarSpec>,
    /// classifier-free guidance: `(negative_prompt_ids, scale)`
    pub cfg: Option<(Vec<i64>, f64)>,
    /// split `<think>…</think>` into `Chunk::Reasoning`
    pub parse_reasoning: bool,
    pub reasoning_start: String,
    pub reasoning_end: String,
    pub start_in_reasoning: bool,
    pub reply: flume::Sender<Chunk>,
    pub cancel: Arc<AtomicBool>,
}

pub struct EngineConfig {
    pub model_dir: std::path::PathBuf,
    pub overrides: ConfigOverrides,
    pub cache_bits: i64,
    pub ctx_len: i64,
    pub max_batch: usize,
    pub chunk_size: i64,
    pub draft_mode: DraftMode,
    pub draft_num_tokens: i64,
    pub ngram_match_min: usize,
    /// directory of the separate AR draft model (`draft_mode: model`)
    pub draft_model_dir: Option<std::path::PathBuf>,
    /// prompt-prefix KV-page sharing (non-hybrid only)
    pub prefix_cache: bool,
    /// fair-scheduling requeue budget in generated tokens (0 = off)
    pub rq_budget: usize,
    /// pinned host-RAM KV cache tier size in tokens (0 = off)
    pub cpu_cache_tokens: i64,
    pub vision: bool,
    pub gpu: usize,
}

impl EngineConfig {
    pub fn from_server_config(cfg: &ServerConfig, gpu: usize) -> Result<Self> {
        let model_dir = cfg.model_path()?;
        let alpha = cfg.rope_alpha();
        let ctx = cfg.ctx_len().unwrap_or(0);
        Ok(Self {
            model_dir,
            overrides: ConfigOverrides {
                max_seq_len: if ctx > 0 { Some(ctx) } else { None },
                rope_scale: cfg.model.rope_scale.filter(|&s| s != 1.0),
                rope_alpha: alpha,
            },
            cache_bits: cfg.cache_bits()?,
            ctx_len: ctx,
            max_batch: cfg.model.max_batch_size.unwrap_or(0),
            chunk_size: cfg.model.chunk_size.unwrap_or(0),
            draft_mode: cfg.draft_mode(),
            // 4, matching upstream's `default_draft_size` for this arch. 3 won a
            // synthetic temperature-0 continuation benchmark, but real traffic
            // spans a much wider acceptance range (22-78% observed) and the
            // marginal draft step is cheap next to the verify: +1 step costs
            // ~2.3 ms of a ~34 ms round while buying ~0.7 tokens at 70%
            // acceptance. Tune per workload if acceptance is consistently low.
            draft_num_tokens: cfg.draft_model.draft_num_tokens.unwrap_or(4).max(1),
            ngram_match_min: cfg.draft_model.ngram_match_min.unwrap_or(2),
            draft_model_dir: cfg.draft_model_path(),
            prefix_cache: cfg.model.prefix_cache,
            rq_budget: cfg.model.max_rq_tokens.unwrap_or(0).max(0) as usize,
            cpu_cache_tokens: cfg.model.cpu_cache_tokens.unwrap_or(0).max(0),
            vision: cfg.model.vision,
            gpu,
        })
    }
}

/// Metadata the HTTP layer needs without touching the engine.
#[derive(Clone)]
pub struct ModelMeta {
    pub id: String,
    pub n_ctx: i64,
    pub vocab_size: i64,
    pub arch: String,
    pub eos: Vec<i64>,
    pub bos: Option<i64>,
    pub has_vision: bool,
    pub mode: &'static str,
}

/// Handle held by HTTP handlers.
#[derive(Clone)]
pub struct EngineHandle {
    pub tx: flume::Sender<EngineRequest>,
    pub meta: ModelMeta,
    /// tokenizer clone-free access: encode/decode via a dedicated channel is
    /// overkill, so we keep an `Arc<Tok>` (it is `Send + Sync`).
    pub tok: Arc<Tok>,
}

impl EngineHandle {
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        self.tok.encode(text)
    }
}

/// Spawn the engine thread. Blocks (in the caller) only until the model is
/// loaded, then returns a handle; generation runs on the spawned thread.
/// VRAM cost of one token of KV pool, summed over every pool `resize_pool`
/// allocates: the trunk's paged KV (all layers for a dense arch, only the
/// `full_attention` layers for the Qwen3.5 hybrid — the GDN recurrent state is
/// per-slot, not per-token) plus the MTP head's own Q8 KV when `mtp` is set.
///
/// Quantized pages are packed `int32` codes (`groups * bits` per token per
/// tensor) plus one `fp16` scale per group, i.e. `16/bits` smaller than fp16
/// plus a ~12.5% scale overhead — the scales are why the true ratio is not the
/// clean `16/bits` the old hand-tuned cap assumed.
fn pool_bytes_per_token(cfg: &crate::config::Config, is_hybrid: bool, bits: i64, mtp: bool) -> i64 {
    use crate::config::LayerKind;
    let hd = cfg.head_dim;
    let nkv = cfg.kv_heads_eff().0;
    let groups = nkv * hd / 32;
    let per_layer = if bits > 0 {
        2 * (groups * bits * 4 + groups * 2) // qk+qv int32 codes, sk+sv fp16 scales
    } else {
        2 * nkv * hd * 2 // k+v fp16
    };
    let kv_layers = if is_hybrid {
        cfg.layer_types.iter().filter(|l| matches!(l, LayerKind::FullAttention)).count() as i64
    } else {
        cfg.layer_types.len() as i64
    };
    let mut b = kv_layers * per_layer;
    if mtp {
        // one full-attention layer, always Q8 (`MtpBatchedCache::BITS`)
        b += 2 * (groups * 8 * 4 + groups * 2);
    }
    b
}

pub fn spawn(ec: EngineConfig, id: String) -> Result<EngineHandle> {
    install_panic_hook();
    let (ready_tx, ready_rx) = flume::bounded::<Result<(ModelMeta, Arc<Tok>)>>(1);
    let (tx, rx) = flume::unbounded::<EngineRequest>();

    std::thread::Builder::new()
        .name("exl3-engine".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let device = Device::Cuda(ec.gpu);
            let built = catch_cuda(|| -> Result<Engine> {
                let model = load_model_bar(&ec.model_dir, device, &ec.overrides)
                    .with_context(|| format!("loading model from {}", ec.model_dir.display()))?;
                let tok = Tok::load(&ec.model_dir)
                    .with_context(|| format!("loading tokenizer from {}", ec.model_dir.display()))?;
                Engine::new(model, tok, &ec, id, device)
            })
            .and_then(|r| r);
            match built {
                Ok(mut engine) => {
                    let _ = ready_tx.send(Ok((engine.meta.clone(), engine.tok_arc.clone())));
                    // A single-thread tokio runtime pinned to THIS OS thread (all
                    // CUDA work must stay here); `LocalSet` lets the non-`Send`
                    // engine run as an async task. Mirrors the asyncio model of
                    // `async_generator.py`.
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .expect("engine tokio runtime");
                    let local = tokio::task::LocalSet::new();
                    local.block_on(&rt, engine.run(rx));
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })?;

    let (meta, tok) = ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("engine thread died during startup"))??;
    Ok(EngineHandle { tx, meta, tok })
}

/// Replace the default panic hook with one that prints a single clean line for
/// libtorch/CUDA errors instead of a screenful of C++ stack frames. The panic
/// still unwinds (and is caught by `catch_cuda`); this only tames the output.
fn install_panic_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = info
                .payload()
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| info.payload().downcast_ref::<&str>().copied())
                .unwrap_or("panic");
            if msg.contains("Torch(") || msg.contains("CUDA") {
                crate::swarn!("{}", pretty_torch_err(msg));
            } else {
                crate::swarn!("panic: {}{}",
                    msg.lines().next().unwrap_or(msg),
                    info.location().map(|l| format!(" (at {l})")).unwrap_or_default(),
                );
            }
        }));
    });
}

struct Engine {
    /// the batched generator, wrapped in the async fan-out layer. Derefs to
    /// `Generator`, so all the setup / `iterate()` / `cancel()` calls are
    /// unchanged; the engine loop runs it on a `current_thread` tokio runtime.
    gen: AsyncGenerator,
    tok_arc: Arc<Tok>,
    vision: Option<VisionModel>,
    meta: ModelMeta,
    device: Device,
    gpu: usize,
    ctx_len: i64,
    /// CFG is unavailable on a hybrid cache or with a speculator active
    cfg_ok: bool,
}

/// Per-active-job bookkeeping for the batched path.
struct JobState {
    reply: flume::Sender<Chunk>,
    cancel: Arc<AtomicBool>,
    raw: String,
    emitted_content: usize,
    emitted_reasoning: usize,
    parse_reasoning: bool,
    r_start: String,
    r_end: String,
    start_in_reasoning: bool,
    completion_tokens: usize,
    /// metrics accounting
    prompt_tokens: usize,
    enqueued: Instant,
    prefill_end: Option<Instant>,
    /// `(spec_accepted, spec_drafted, prefix_cached, prefix_total)` at enqueue
    base_spec: (u64, u64),
    base_prefix: (u64, u64),
}

impl Engine {
    fn new(model: Model, tok: Tok, ec: &EngineConfig, id: String, device: Device) -> Result<Self> {
        let arch = model.config.arch.clone();
        let is_hybrid = model.config.arch_kind == ArchKind::Qwen35;
        let n_ctx = if ec.ctx_len > 0 {
            ec.ctx_len
        } else {
            model.config.max_position_embeddings
        };
        // MTP and separate-draft speculation both add `(draft+1)` fp32 recurrent
        // -history planes per batch slot on a hybrid target — cap the batch there.
        let spec_hist = matches!(ec.draft_mode, DraftMode::Mtp | DraftMode::Draft) && is_hybrid;

        // --- KV pool sizing -------------------------------------------------
        // TabbyAPI's `cache_size` is the TOTAL shared pool (tokens). It used to
        // be clamped by a hand-tuned constant, which was both wrong (it refused
        // pool sizes that fit) and unsafe (it accepted ones that then OOM'd
        // mid-request). Instead the generator starts with a placeholder pool and
        // `resize_pool` fills the VRAM that is *actually* left once the vision
        // tower, MTP head and draft model are loaded — see `fit_pool` below.
        // `EXL3_MAX_CACHE` still caps it, for leaving room for other processes.
        let requested = if ec.ctx_len > 0 { ec.ctx_len } else { n_ctx };
        let pool_cap = std::env::var("EXL3_MAX_CACHE")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(i64::MAX);
        let want_tokens = requested.min(pool_cap).max(1024);
        // placeholder: one page. The real pool is allocated by `resize_pool`.
        let num_pages = 1;

        let max_batch = if spec_hist {
            if ec.max_batch > 0 { ec.max_batch.min(2) } else { 1 }
        } else if ec.max_batch > 0 {
            ec.max_batch
        } else if is_hybrid {
            4
        } else {
            8
        };
        let max_chunk = if ec.chunk_size > 0 { ec.chunk_size } else { 2048 };

        let vision = if ec.vision {
            if !is_hybrid {
                crate::swarn!("vision requested but arch {arch} has no ported vision tower — ignoring");
                None
            } else {
                Some(VisionModel::load(&ec.model_dir, &model.config, device)?)
            }
        } else {
            None
        };

        let eos = model.config.eos_token_ids.clone();
        let bos = model.config.bos_token_id;
        let vocab_size = model.config.vocab_size;

        // Build the KV pool quantized from the start (hybrid: `Generator::new`
        // takes the bits; a 200k fp16 hybrid pool is ~13 GB and would OOM before
        // `enable_cache_quant` could replace it). GDN recurrent state stays fp32.
        let kv_bits = if ec.cache_bits > 0 && is_hybrid {
            (ec.cache_bits, ec.cache_bits)
        } else {
            (0, 0)
        };
        let mut g = Generator::new(model, tok, num_pages, max_batch, max_chunk, kv_bits);
        if ec.cache_bits > 0 && !is_hybrid {
            g.enable_cache_quant(ec.cache_bits, ec.cache_bits);
        }
        match ec.draft_mode {
            DraftMode::Mtp => {
                if !is_hybrid {
                    anyhow::bail!("draft_mode: mtp requires a Qwen3.5 checkpoint with an MTP head");
                }
                let mtp = MtpModel::load_headless(&ec.model_dir, device, g.model())?;
                g.enable_mtp(mtp, ec.draft_num_tokens);
            }
            DraftMode::Ngram => g.enable_ngram(ec.ngram_match_min, ec.draft_num_tokens),
            DraftMode::DFlash2 => {
                let dir = ec.draft_model_dir.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("draft_mode: dflash2 needs draft_model_dir + draft_model_name")
                })?;
                if !is_hybrid {
                    anyhow::bail!("draft_mode: dflash2 requires a Qwen3.5 target");
                }
                // `Generator::new` has already claimed the full KV pool, and the
                // drafter needs ~1.4 GB of weights plus a window cache per slot.
                // Shrink the pool first and hand the memory back to the driver —
                // freeing alone only returns it to the caching allocator, which
                // the driver still counts as used — then let the sizing block
                // below re-expand into whatever is genuinely left.
                let dbg = std::env::var("EXL3_MEM_DEBUG").is_ok();
                let m = |tag: &str| {
                    if dbg {
                        eprintln!("[df2] {tag}: {} MiB free", crate::ffi::cuda_free_mib());
                    }
                };
                m("before resize");
                g.resize_pool(crate::paged::pages_for(1024));
                crate::ffi::CudaGraph::empty_cache();
                m("after resize+empty_cache");
                let d = crate::dflash2::DFlash2Model::load(dir, device)?;
                m("after drafter load");
                g.enable_dflash2(d, ec.draft_num_tokens);
                m("after enable");
            }
            DraftMode::Draft => {
                let dir = ec.draft_model_dir.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("draft_mode: model needs draft_model_dir set")
                })?;
                let np = crate::paged::pages_for(want_tokens);
                let draft = crate::draft::DraftModel::load(dir, device, g.model(), np)?;
                g.enable_draft(draft, ec.draft_num_tokens);
            }
            DraftMode::Disabled => {}
        }

        // Everything that competes with the KV pool for VRAM is loaded now, so
        // ask the driver how much is actually left and size the pool to fill it,
        // minus a working-set reserve for prefill/decode activations. This is
        // what lets a `cache_size: 204800` config load on a 24 GB card exactly
        // as it does under the Python port, instead of being clipped by a
        // hand-tuned constant.
        // What the KV pool can actually hold. Advertising the configured
        // `max_seq_len` when the pool came out smaller invites requests the
        // server can never serve — and those used to hang, not error.
        let served_ctx;
        {
            let per_tok = pool_bytes_per_token(
                &g.model().config,
                is_hybrid,
                ec.cache_bits,
                matches!(ec.draft_mode, DraftMode::Mtp),
            );
            // Loading leaves the caching allocator holding the dequant/H2D
            // staging buffers, and the driver counts those as used — so ask for
            // them back before measuring, or the pool is sized against a free
            // figure that is hundreds of MiB pessimistic.
            crate::ffi::CudaGraph::empty_cache();
            let free_mib = crate::ffi::cuda_free_mib().max(0);
            let reserve_mib = std::env::var("EXL3_VRAM_RESERVE_MB")
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or_else(|| {
                    // DFlash2 allocates its round working set *after* this
                    // sizing runs: the five tap tensors captured during verify,
                    // the block forward's score matrix, and `paths_from_states`.
                    // Without a term for them the pool claims the VRAM they
                    // need and the first request OOMs instead of the pool
                    // simply coming out smaller.
                    let df2 = if matches!(ec.draft_mode, DraftMode::DFlash2) { 1024 } else { 0 };
                    768 + 512 * max_chunk / 2048 + 64 * max_batch as i64 + df2
                });
            let avail = ((free_mib - reserve_mib).max(0)) * 1024 * 1024;
            let fits = if per_tok > 0 { avail / per_tok } else { want_tokens };
            let pool_tokens = want_tokens.min(fits).max(1024);
            if pool_tokens < want_tokens {
                crate::swarn!("KV pool sized to {pool_tokens} tokens (config asked for {requested}): \
                     {free_mib} MiB free after load, {reserve_mib} MiB reserved for activations, \
                     {} B/token. Free VRAM, lower cache_size, or set EXL3_VRAM_RESERVE_MB.",
                    per_tok
                );
            }
            g.resize_pool(crate::paged::pages_for(pool_tokens));
            served_ctx = pool_tokens;
            crate::sinfo!("KV pool: {pool_tokens} tokens ({:.2} GiB), {} MiB VRAM free after",
                (pool_tokens * per_tok) as f64 / (1024.0 * 1024.0 * 1024.0),
                crate::ffi::cuda_free_mib()
            );
        }

        if ec.prefix_cache {
            g.enable_prefix_cache();
            if is_hybrid {
                crate::sinfo!("prefix_cache: on (hybrid — GDN state checkpointed at page boundaries)");
            }
        }
        if ec.rq_budget > 0 {
            g.enable_requeue(ec.rq_budget);
        }
        if ec.cpu_cache_tokens > 0 {
            if is_hybrid || ec.cache_bits > 0 {
                crate::sinfo!("cpu_cache_tokens ignored: homogeneous fp16 KV cache only");
            } else {
                g.enable_cpu_cache(ec.cpu_cache_tokens);
            }
        }

        let tok_arc = Arc::new(Tok::load(&ec.model_dir)?);

        let mode = match ec.draft_mode {
            DraftMode::Mtp => "mtp",
            DraftMode::Ngram => "ngram",
            DraftMode::Draft => "draft-model",
            DraftMode::DFlash2 => "dflash2",
            DraftMode::Disabled => "batched",
        };

        let meta = ModelMeta {
            id,
            n_ctx: served_ctx,
            vocab_size,
            arch,
            eos,
            bos,
            has_vision: vision.is_some(),
            mode,
        };

        Ok(Self {
            gen: AsyncGenerator::new(g),
            tok_arc,
            vision,
            meta,
            device,
            gpu: ec.gpu,
            ctx_len: n_ctx,
            cfg_ok: !is_hybrid && matches!(ec.draft_mode, DraftMode::Disabled),
        })
    }

    /// The engine loop, driven on a `current_thread` tokio runtime (see
    /// [`spawn`]). `iterate()` runs inline — we are already on the one CUDA
    /// thread — and we `yield_now().await` after each round so the runtime can
    /// service the request channel and any other local task, mirroring
    /// `async_generator.py::_run_iteration`'s `await asyncio.sleep(0)`.
    async fn run(&mut self, rx: flume::Receiver<EngineRequest>) {
        let mut jobs: HashMap<u64, JobState> = HashMap::new();

        loop {
            // If nothing is active, await the next request. Otherwise poll.
            if jobs.is_empty() {
                match rx.recv_async().await {
                    Ok(req) => self.admit(req, &mut jobs),
                    Err(_) => return, // all senders dropped -> shutdown
                }
            }
            while let Ok(req) = rx.try_recv() {
                self.admit(req, &mut jobs);
            }
            if jobs.is_empty() {
                continue;
            }

            // Drop cancelled jobs.
            let cancelled: Vec<u64> = jobs
                .iter()
                .filter(|(_, s)| s.cancel.load(Ordering::Relaxed))
                .map(|(k, _)| *k)
                .collect();
            for k in cancelled {
                self.gen.cancel(k);
                jobs.remove(&k);
            }
            if jobs.is_empty() {
                continue;
            }

            let events = match catch_cuda(|| self.gen.iterate()) {
                Ok(Ok(e)) => e,
                Ok(Err(e)) => {
                    for (_, s) in jobs.drain() {
                        let _ = s.reply.send(Chunk::Error(format!("engine: {e}")));
                    }
                    continue;
                }
                Err(panic) => {
                    // a libtorch error (usually OOM) unwound iterate(); report it
                    // to every in-flight job, free their pages, keep serving.
                    let msg = panic.to_string();
                    crate::swarn!("{msg}");
                    let keys: Vec<u64> = jobs.keys().copied().collect();
                    for (_, s) in jobs.drain() {
                        let _ = s.reply.send(Chunk::Error(msg.clone()));
                    }
                    for k in keys {
                        self.gen.cancel(k);
                    }
                    let _ = catch_cuda(|| self.gen.iterate()); // best-effort reap
                    continue;
                }
            };

            for ev in events {
                let Some(st) = jobs.get_mut(&ev.serial) else { continue };
                match ev.stage {
                    Stage::Started | Stage::Prefill { .. } => {}
                    Stage::Streaming => {
                        st.prefill_end.get_or_insert_with(Instant::now);
                        if !ev.text.is_empty() {
                            st.raw.push_str(&ev.text);
                            st.completion_tokens = ev.new_tokens;
                            flush_job(st, false);
                        }
                        if ev.eos {
                            st.completion_tokens = ev.new_tokens;
                            flush_job(st, true);
                            let finish = ev
                                .eos_reason
                                .as_deref()
                                .map(FinishReason::from_eos_reason)
                                .unwrap_or(FinishReason::Stop);
                            let now = Instant::now();
                            let prefill_end = st.prefill_end.unwrap_or(now);
                            let (sa, sd) = self.gen.spec_stats();
                            let (pc, _pt) = self.gen.prefix_stats();
                            let cached = (pc.saturating_sub(st.base_prefix.0) as usize)
                                .min(st.prompt_tokens);
                            let stats = GenStats {
                                cached_prompt_tokens: cached,
                                new_prompt_tokens: st.prompt_tokens - cached,
                                prefill_secs: prefill_end
                                    .duration_since(st.enqueued)
                                    .as_secs_f64(),
                                gen_secs: now.duration_since(prefill_end).as_secs_f64(),
                                draft_accepted: sa.saturating_sub(st.base_spec.0),
                                draft_total: sd.saturating_sub(st.base_spec.1),
                            };
                            let _ = st.reply.send(Chunk::Done {
                                finish,
                                completion_tokens: st.completion_tokens,
                                stats,
                            });
                            jobs.remove(&ev.serial);
                        }
                    }
                }
            }

            // hand control back to the runtime between rounds
            tokio::task::yield_now().await;
        }
    }

    fn admit(&mut self, mut req: EngineRequest, jobs: &mut HashMap<u64, JobState>) {
        // context guard
        if req.prompt_ids.len() as i64 >= self.ctx_len {
            let _ = req.reply.send(Chunk::Error(format!(
                "prompt is {} tokens, context window is {}",
                req.prompt_ids.len(),
                self.ctx_len
            )));
            return;
        }
        if let Some(seed) = req.seed {
            tch::manual_seed(seed);
        }

        // image requests run single-stream (block the engine); everything else,
        // MTP / n-gram speculation included, goes through the batched Generator.
        if !req.images.is_empty() {
            if self.vision.is_none() {
                let _ = req.reply.send(Chunk::Error(
                    "this model / config has no vision support (set model.vision: true)".into(),
                ));
                return;
            }
            self.run_vision(req);
            return;
        }
        let grammar = req.grammar.take();
        let prompt_tokens = req.prompt_ids.len();
        let base_spec = self.gen.spec_stats();
        let base_prefix = self.gen.prefix_stats();
        let enqueued = Instant::now();
        let mut spec = JobSpec::new(req.prompt_ids, req.max_new);
        spec.sampler = req.sampler;
        spec.min_new = req.min_new;
        spec.stop_tokens = self.meta.eos.iter().copied().collect();
        spec.stop_strings = req.stop_strings;
        let filter = match grammar {
            None => Ok(None),
            Some(GrammarSpec::JsonObject) => self.gen.compile_json_object().map(Some),
            Some(GrammarSpec::JsonSchema(s)) => self.gen.compile_json_schema(&s).map(Some),
            Some(GrammarSpec::Gbnf(g)) => self.gen.compile_grammar(&g).map(Some),
        };
        match filter {
            Ok(Some(f)) => spec.filters.push(f),
            Ok(None) => {}
            Err(e) => {
                let _ = req.reply.send(Chunk::Error(format!("response_format: {e}")));
                return;
            }
        }
        if let Some((neg, scale)) = req.cfg.take() {
            if !self.cfg_ok {
                let _ = req.reply.send(Chunk::Error(
                    "classifier-free guidance is unavailable with this model's cache type or with speculative decoding enabled".into(),
                ));
                return;
            }
            spec.cfg = Some((neg, scale));
        }
        let serial = self.gen.enqueue(spec);
        jobs.insert(
            serial,
            JobState {
                reply: req.reply,
                cancel: req.cancel,
                raw: String::new(),
                emitted_content: 0,
                emitted_reasoning: 0,
                parse_reasoning: req.parse_reasoning,
                r_start: req.reasoning_start,
                r_end: req.reasoning_end,
                start_in_reasoning: req.start_in_reasoning,
                completion_tokens: 0,
                prompt_tokens,
                enqueued,
                prefill_end: None,
                base_spec,
                base_prefix,
            },
        );
    }

    // --- single-stream vision -------------------------------------------------

    fn run_vision(&mut self, req: EngineRequest) {
        let res = match catch_cuda(|| self.vision_generate(&req)) {
            Ok(r) => r,
            Err(panic) => {
                crate::swarn!("{panic}");
                Err(panic)
            }
        };
        finish_single(&req, res);
    }

    fn vision_generate(&mut self, req: &EngineRequest) -> Result<(FinishReason, usize)> {
        use crate::generator::MmFinish;
        use crate::rope::RoPE;
        let _g = tch::no_grad_guard();
        if req.images.len() != 1 {
            anyhow::bail!("this build supports exactly one image per request");
        }
        let device = self.device;

        // Build the spliced input embeddings + MRoPE table (needs an immutable
        // borrow of the model / vision tower — scope it so `mm_generate` can take
        // `&mut self.gen` afterwards).
        let (embeds, seq, angle_table, max_new) = {
            let model = self.gen.model();
            let vision = self
                .vision
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no vision tower loaded"))?;
            let cfg = &model.config;
            let merge = cfg.vision.as_ref().unwrap().spatial_merge_size;

            // the HTTP layer passes the *text* prompt ids with a literal `<image>`
            // word; split the decoded text around it and re-tokenise each side.
            let text = self.tok_arc.decode(&req.prompt_ids)?;
            let (before, after) = match text.split_once("<image>") {
                Some((a, b)) => (a.to_string(), b.to_string()),
                None => (String::new(), text.clone()),
            };
            let before_ids = self.tok_arc.encode(&before)?;
            let after_ids = self.tok_arc.encode(&after)?;

            let img_path = resolve_image(&req.images[0])?;
            let ie = vision.embed_image(img_path.to_str().unwrap_or_default())?;
            let n_img = ie.num_tokens();

            let vs = cfg.vision_start_token_id;
            let ve = cfg.vision_end_token_id;
            let pad = cfg.image_token_id;
            let mut seq: Vec<i64> = Vec::new();
            seq.extend_from_slice(&before_ids);
            seq.push(vs);
            seq.extend(std::iter::repeat(pad).take(n_img as usize));
            seq.push(ve);
            seq.extend_from_slice(&after_ids);
            let seq_len = seq.len() as i64;
            if seq_len >= self.ctx_len {
                anyhow::bail!(
                    "image + prompt is {seq_len} tokens, context window is {}",
                    self.ctx_len
                );
            }
            let img_first = before_ids.len() as i64 + 1;
            let img_span = (img_first, img_first + n_img);

            let emb_of = |slice: &[i64]| -> Tensor {
                if slice.is_empty() {
                    Tensor::zeros([1, 0, cfg.hidden_size], (Kind::Half, device))
                } else {
                    model
                        .embed_tokens(&Tensor::from_slice(slice).reshape([1, -1]).to_device(device))
                        .to_kind(Kind::Half)
                }
            };
            let img_emb = ie.embeddings.to_device(device).to_kind(Kind::Half).unsqueeze(0);
            let embeds = Tensor::cat(
                &[emb_of(&before_ids), emb_of(&[vs]), img_emb, emb_of(&[ve]), emb_of(&after_ids)],
                1,
            );

            let max_new = (req.max_new as i64).min((self.ctx_len - seq_len - 1).max(1)) as usize;
            let max_len = seq_len + max_new as i64 + 8;
            let rope = RoPE::new(device, &cfg.rope);
            let half = rope.inv_freq.size()[0];
            let (pt, ph, pw, next_base) =
                mrope_pos_ids(seq_len, img_span, (ie.grid_t, ie.grid_h, ie.grid_w), merge);
            let section = cfg.mrope_section.unwrap_or([half, 0, 0]);
            let angle_table = mrope_angle_table(
                &rope.inv_freq, (&pt, &ph, &pw), next_base, max_len, section, device,
            );
            (embeds, seq, angle_table, max_new)
        };

        let eos: std::collections::HashSet<i64> = self.meta.eos.iter().copied().collect();
        let tok = self.tok_arc.clone();
        let mut streamer = Streamer::new(req);

        let (finish, generated) = self.gen.mm_generate(
            &embeds,
            &seq,
            &angle_table,
            &req.sampler,
            max_new,
            &eos,
            &req.cancel,
            |id| streamer.push(&tok, id),
        )?;

        tch::Cuda::synchronize(self.gpu as i64);
        streamer.finish(); // release the reasoning-delimiter hold-back
        let finish = match finish {
            MmFinish::Stop => FinishReason::Stop,
            MmFinish::Length => FinishReason::Length,
            MmFinish::Cancelled => FinishReason::Cancelled,
        };
        Ok((finish, generated))
    }
}

/// Incremental detokeniser + reasoning splitter for the single-stream paths.
struct Streamer<'a> {
    req: &'a EngineRequest,
    ids: Vec<i64>,
    raw: String,
    emitted_content: usize,
    emitted_reasoning: usize,
    hit_stop: bool,
}

impl<'a> Streamer<'a> {
    fn new(req: &'a EngineRequest) -> Self {
        Self {
            req,
            ids: Vec::new(),
            raw: String::new(),
            emitted_content: 0,
            emitted_reasoning: 0,
            hit_stop: false,
        }
    }
    /// Feed one token. Returns `true` once a stop string has been hit (the caller
    /// should stop generating).
    fn push(&mut self, tok: &Tok, id: i64) -> Result<bool> {
        self.ids.push(id);
        // decode the whole gen so far (short) for correct multi-byte handling
        let mut full = tok.decode(&self.ids).unwrap_or_default();
        // honor stop strings (the single-stream paths have no Generator to do it)
        for s in &self.req.stop_strings {
            if s.is_empty() {
                continue;
            }
            if let Some(pos) = full.find(s.as_str()) {
                full.truncate(pos);
                self.hit_stop = true;
            }
        }
        if full.len() > self.raw.len() {
            self.raw = full;
            self.emit(false);
        }
        Ok(self.hit_stop)
    }

    /// Emit the pending deltas. `final_flush` on the last call (eos / stop) so the
    /// delimiter hold-back is released and any trailing bytes go out.
    fn emit(&mut self, final_flush: bool) {
        let mut send = |c: Chunk| {
            let _ = self.req.reply.send(c);
        };
        send_reasoning_deltas(
            self.req.parse_reasoning,
            &self.raw,
            &self.req.reasoning_start,
            &self.req.reasoning_end,
            self.req.start_in_reasoning,
            &mut self.emitted_content,
            &mut self.emitted_reasoning,
            final_flush,
            &mut send,
        );
    }

    /// Release the hold-back and emit any remaining text. Call once after the
    /// generation loop, before `Chunk::Done`.
    fn finish(&mut self) {
        self.emit(true);
    }
}

/// Trailing bytes of `raw` to withhold before splitting: the longest suffix that
/// is a proper prefix of a reasoning delimiter, so a `<think>` / `</think>` split
/// across tokens never leaks a fragment into the wrong stream. Delimiters are
/// ASCII, so byte work is fine.
fn reasoning_holdback(raw: &str, delims: [&str; 2]) -> usize {
    let b = raw.as_bytes();
    let maxd = delims.iter().map(|d| d.len()).max().unwrap_or(0);
    let hi = maxd.saturating_sub(1).min(b.len());
    for l in (1..=hi).rev() {
        let tail = &b[b.len() - l..];
        if delims.iter().any(|d| d.len() > l && d.as_bytes().starts_with(tail)) {
            return l;
        }
    }
    0
}

/// Split `raw` into `(reasoning, content)` and send whatever hasn't been emitted
/// yet through `send`. Until `final_flush`, a few trailing bytes are withheld so
/// a reasoning delimiter that straddles two tokens is classified in one piece
/// (otherwise `<think>` leaks its first chars into `content` and `</think>` its
/// last chars into `reasoning_content`).
#[allow(clippy::too_many_arguments)]
fn send_reasoning_deltas(
    parse_reasoning: bool,
    raw: &str,
    r_start: &str,
    r_end: &str,
    start_in_reasoning: bool,
    emitted_content: &mut usize,
    emitted_reasoning: &mut usize,
    final_flush: bool,
    send: &mut dyn FnMut(Chunk),
) {
    if !parse_reasoning {
        if raw.len() > *emitted_content {
            send(Chunk::Content(raw[*emitted_content..].to_string()));
            *emitted_content = raw.len();
        }
        return;
    }

    let view = if final_flush {
        raw
    } else {
        let mut keep = raw.len() - reasoning_holdback(raw, [r_start, r_end]);
        while keep > 0 && !raw.is_char_boundary(keep) {
            keep -= 1;
        }
        &raw[..keep]
    };

    let (reasoning, content) =
        crate::server::chat::split_reasoning(view, r_start, r_end, start_in_reasoning);
    if reasoning.len() > *emitted_reasoning {
        send(Chunk::Reasoning(reasoning[*emitted_reasoning..].to_string()));
        *emitted_reasoning = reasoning.len();
    }
    if content.len() > *emitted_content {
        send(Chunk::Content(content[*emitted_content..].to_string()));
        *emitted_content = content.len();
    }
}

fn flush_job(st: &mut JobState, final_flush: bool) {
    let mut send = |c: Chunk| {
        let _ = st.reply.send(c);
    };
    send_reasoning_deltas(
        st.parse_reasoning,
        &st.raw,
        &st.r_start,
        &st.r_end,
        st.start_in_reasoning,
        &mut st.emitted_content,
        &mut st.emitted_reasoning,
        final_flush,
        &mut send,
    );
}

/// Turn an OpenAI `image_url` value into a local file path. Supports `data:`
/// URIs (written to a temp file) and plain filesystem paths / `file://` URLs.
/// `http(s)://` is rejected (remote fetch not implemented).
fn resolve_image(spec: &str) -> Result<std::path::PathBuf> {
    let s = spec.trim();
    if let Some(rest) = s.strip_prefix("data:") {
        let comma = rest
            .find(',')
            .ok_or_else(|| anyhow::anyhow!("malformed data: URI"))?;
        let meta = &rest[..comma];
        let payload = &rest[comma + 1..];
        let bytes = if meta.contains("base64") {
            b64_decode(payload)?
        } else {
            payload.as_bytes().to_vec()
        };
        let ext = if meta.contains("png") {
            "png"
        } else if meta.contains("webp") {
            "webp"
        } else {
            "jpg"
        };
        let mut p = std::env::temp_dir();
        p.push(format!(
            "exl3-img-{}.{ext}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, &bytes)?;
        return Ok(p);
    }
    if let Some(path) = s.strip_prefix("file://") {
        return Ok(std::path::PathBuf::from(path));
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        anyhow::bail!("remote image URLs are not supported — pass a data: URI or a local path");
    }
    Ok(std::path::PathBuf::from(s))
}

fn b64_decode(s: &str) -> Result<Vec<u8>> {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut inv = [255u8; 256];
    for (i, &c) in T.iter().enumerate() {
        inv[c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &c in s.trim().as_bytes() {
        if c == b'=' || c == b'\n' || c == b'\r' || c == b' ' {
            continue;
        }
        let v = inv[c as usize];
        if v == 255 {
            anyhow::bail!("invalid base64");
        }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}

/// Run a CUDA-touching closure, converting a panic (libtorch errors `unwrap`
/// inside `tch`) into a clean `Err` instead of unwinding the engine thread.
fn catch_cuda<T>(f: impl FnOnce() -> T) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(e) => {
            let raw = e
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("engine panic");
            Err(anyhow::anyhow!(pretty_torch_err(raw)))
        }
    }
}

/// Collapse a multi-page libtorch panic string into one useful line.
pub fn pretty_torch_err(s: &str) -> String {
    // pull the message out of `Torch("…")` if present
    let inner = s
        .find("Torch(\"")
        .map(|i| &s[i + 7..])
        .and_then(|t| t.rfind("\")").map(|j| &t[..j]))
        .unwrap_or(s);
    // stop at the C++ stack trace
    let head = inner
        .split("\nException raised from")
        .next()
        .unwrap_or(inner)
        .split("\\nException raised from")
        .next()
        .unwrap_or(inner)
        .replace("\\n", " ")
        .trim()
        .to_string();

    if head.contains("out of memory") {
        let grab = |a: &str, b: &str| -> Option<String> {
            let i = head.find(a)? + a.len();
            let rest = &head[i..];
            let j = rest.find(b)?;
            Some(rest[..j].trim().to_string())
        };
        let tried = grab("Tried to allocate ", " ").unwrap_or_default();
        let free = grab("of which ", " is free").unwrap_or_default();
        let cap = grab("total capacity of ", " of which").unwrap_or_default();
        return format!(
            "CUDA out of memory (tried to allocate {tried}, {free} free of {cap}). \
             Lower `cache_size` / set EXL3_MAX_CACHE, turn off `vision:` or `draft_mode: mtp`, \
             or use a smaller quant."
        );
    }
    // generic: first line only
    head.lines().next().unwrap_or(&head).to_string()
}

fn finish_single(req: &EngineRequest, res: Result<(FinishReason, usize)>) {
    match res {
        Ok((finish, n)) => {
            let _ = req.reply.send(Chunk::Done {
                finish,
                completion_tokens: n,
                stats: GenStats::default(),
            });
        }
        Err(e) => {
            let _ = req.reply.send(Chunk::Error(e.to_string()));
        }
    }
}
