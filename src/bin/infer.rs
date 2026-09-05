//! Minimal inference driver — greedy / temperature-top-k-top-p sampling over a
//! paged KV cache: one prefill, then one token per step. The steady-state decode
//! step is captured into a CUDA graph and replayed, collapsing ~1k kernel
//! launches per token into a single replay (`--no-graph` to disable).

use anyhow::Result;
use clap::Parser;
use exl3::{
    cache::{PagedKvCache, QuantPagedKvCache, Qwen35Cache},
    config::ArchKind,
    ffi::CudaGraph,
    model::Model,
    tokenizer::Tok,
};
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tch::{Device, Kind, Tensor};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Attribute a load to its disk and PCIe halves under `EXL3_LOAD_PROF`. Guessing
/// which half dominates is how you optimise the wrong one.
fn report_load_prof(wall: f64) {
    if std::env::var("EXL3_LOAD_PROF").is_err() {
        return;
    }
    let (rd, h2d, bytes) = exl3::safetensors::load_prof_take();
    let gbps = |b: u64, s: f64| if s > 0.0 { b as f64 / s / 1e9 } else { 0.0 };
    eprintln!(
        "[load] {wall:.2}s wall — read {rd:.2}s thread-time ({:.2} GB/s aggregate), \
         h2d {h2d:.2}s ({:.2} GB/s), {:.2} GiB, effective {:.2} GB/s",
        gbps(bytes, rd),
        gbps(bytes, h2d),
        bytes as f64 / (1u64 << 30) as f64,
        gbps(bytes, wall),
    );
}

/// Load the model, drawing an animated `⠋ loading model ▕███░░░▏ 47% (3s)` bar on
/// stderr driven by the real per-layer load progress. Falls back to a plain line
/// when stderr is not a TTY.
fn load_model_bar(
    dir: &std::path::Path,
    device: Device,
    ov: &exl3::config::ConfigOverrides,
) -> anyhow::Result<Model> {
    if !std::io::stderr().is_terminal() {
        eprintln!("loading model...");
        let t0 = Instant::now();
        let r = Model::load_with(dir, device, ov);
        report_load_prof(t0.elapsed().as_secs_f64());
        return r;
    }
    let permil = Arc::new(AtomicU32::new(0)); // progress in 0..=1000
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
    let t0 = Instant::now();
    let r = Model::load_with_progress(dir, device, ov, Some(&cb));
    let wall = t0.elapsed().as_secs_f64();
    done.store(true, Ordering::Relaxed);
    anim.join().ok();
    eprint!("\r\x1b[2K"); // wipe the bar
    std::io::stderr().flush().ok();
    report_load_prof(wall);
    Ok(r?)
}

#[derive(Parser)]
struct Args {
    /// Model directory (config.json + *.safetensors + tokenizer.json)
    #[arg(long)]
    model: String,
    #[arg(long)]
    prompt: String,
    /// Wrap the prompt in the Qwen chat template
    #[arg(long)]
    chat: bool,
    #[arg(long, default_value_t = 64)]
    max_new: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0)]
    top_k: i64,
    #[arg(long, default_value_t = 1.0)]
    top_p: f64,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    /// Print prefill / decode timing to stderr
    #[arg(long)]
    timing: bool,
    /// Disable CUDA graph capture of the decode step
    #[arg(long)]
    no_graph: bool,
    /// Quantize the KV cache to N bits (2..=8; 0 = fp16). Forces --no-graph.
    #[arg(long, default_value_t = 0)]
    cache_bits: i64,
    /// KV cache mode (tabby `cache_mode`): FP16 | Q8 | Q6 | Q4. Alias for
    /// --cache-bits; --cache-bits wins if both are given.
    #[arg(long)]
    cache_mode: Option<String>,
    /// Max context length for cache allocation (tabby `max_seq_len` /
    /// `cache_size`). Defaults to prompt + --max-new.
    #[arg(long)]
    max_seq_len: Option<i64>,
    /// Linear RoPE position scaling (tabby `rope_scale`).
    #[arg(long)]
    rope_scale: Option<f64>,
    /// NTK-aware RoPE base scaling (tabby `rope_alpha`).
    #[arg(long)]
    rope_alpha: Option<f64>,
    /// Prefill chunk size in tokens (tabby `chunk_size`); 0 = one-shot prefill.
    #[arg(long, default_value_t = 0)]
    chunk_size: i64,
    /// Reasoning: keep the model's `<think>` block enabled in the chat template
    /// (default). Pair with --chat.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    reasoning: bool,
    /// Disable thinking: append an empty `<think></think>` block after the chat
    /// template so the model answers directly.
    #[arg(long)]
    no_think: bool,
    /// Load the model's MTP head as a self-speculative draft model (Qwen3.5).
    #[arg(long)]
    mtp: bool,
    /// Number of tokens the MTP head drafts per verify step (`--mtp`).
    #[arg(long, default_value_t = 4)]
    draft_n: i64,
    /// Image file(s) to feed to the vision tower; `<image>` in the prompt marks
    /// the insertion point (repeatable).
    #[arg(long)]
    vision: Vec<String>,
    /// Time the trunk verify forward at a range of `q_len` and exit (`--mtp`).
    /// Answers what an extra *verified* position costs, which a `--draft-n`
    /// sweep cannot: that bundles the sequential MTP draft step with it.
    #[arg(long)]
    bench_qlen: bool,
    /// DFlash2 draft model directory. Drafts a whole block per forward instead
    /// of one token per `lm_head` pass, so the draft cost stops scaling with
    /// the number of drafted tokens.
    #[arg(long)]
    dflash2: Option<String>,
}

/// Map a tabby-style cache mode string to a bit width. FP16 -> 0.
fn cache_mode_bits(s: &str) -> anyhow::Result<i64> {
    match s.trim().to_ascii_uppercase().as_str() {
        "FP16" | "F16" | "" => Ok(0),
        "Q8" | "8" => Ok(8),
        "Q6" | "6" => Ok(6),
        "Q4" | "4" => Ok(4),
        "Q3" | "3" => Ok(3),
        "Q2" | "2" => Ok(2),
        other => anyhow::bail!("unknown --cache-mode {other:?} (FP16|Q8|Q6|Q4)"),
    }
}

/// fp16 or quantized paged KV cache, behind one interface for the decode loop.
enum KvCache {
    Plain(PagedKvCache),
    Quant(QuantPagedKvCache),
    Qwen35(Qwen35Cache),
    Qwen4(exl3::cache::Qwen4Cache),
}

impl KvCache {
    fn advance(&self, n: i64) {
        match self {
            KvCache::Plain(c) => c.advance(n),
            KvCache::Quant(c) => c.advance(n),
            KvCache::Qwen35(c) => c.advance(n),
            KvCache::Qwen4(c) => c.advance(n),
        }
    }
    fn forward(&self, model: &Model, ids: &Tensor) -> Tensor {
        match self {
            KvCache::Plain(c) => model.forward_paged(ids, c),
            KvCache::Quant(c) => model.forward_paged_quant(ids, c),
            KvCache::Qwen35(c) => model.forward_qwen35(ids, c),
            KvCache::Qwen4(c) => model.forward_qwen4(ids, c),
        }
    }
}

fn sample(logits: &Tensor, a: &Args) -> i64 {
    if a.temperature <= 0.0 {
        return logits.argmax(0, false).int64_value(&[]);
    }
    let mut l = logits / a.temperature;
    if a.top_k > 0 {
        let (vals, _) = l.topk(a.top_k.min(l.size()[0]), 0, true, true);
        let kth = vals.get(vals.size()[0] - 1).double_value(&[]);
        l = l.where_scalarother(&l.ge(kth), f64::NEG_INFINITY);
    }
    let mut probs = l.softmax(0, Kind::Float);
    if a.top_p < 1.0 {
        let (sorted, idx) = probs.sort(0, true);
        let cum = sorted.cumsum(0, Kind::Float);
        let mask = cum.le(a.top_p);
        let mask = mask.logical_or(&Tensor::arange(mask.size()[0], (Kind::Int64, mask.device())).eq(0));
        let kept = sorted.where_scalarother(&mask, 0.0);
        probs = probs.zeros_like().scatter(0, &idx, &kept);
        probs = &probs / probs.sum(Kind::Float);
    }
    probs.multinomial(1, true).int64_value(&[0])
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    let device = Device::Cuda(args.gpu);
    let dir = std::path::PathBuf::from(&args.model);

    // --cache-mode is an alias for --cache-bits (explicit --cache-bits wins).
    if args.cache_bits == 0 {
        if let Some(m) = &args.cache_mode {
            args.cache_bits = cache_mode_bits(m)?;
        }
    }
    let overrides = exl3::config::ConfigOverrides {
        max_seq_len: args.max_seq_len,
        rope_scale: args.rope_scale,
        rope_alpha: args.rope_alpha,
    };
    let mut model = load_model_bar(&dir, device, &overrides)?;
    let tok = Tok::load(&dir)?;

    // --- multimodal: load the vision tower and run the image + text path ---
    if !args.vision.is_empty() {
        let vm = exl3::vision::VisionModel::load(&dir, &model.config, device)?;
        let opts = exl3::vision::VisionInferOpts {
            image_paths: &args.vision,
            prompt: &args.prompt,
            chat: args.chat,
            no_think: args.no_think || !args.reasoning,
            max_new: args.max_new,
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            max_seq_len: args.max_seq_len,
            timing: args.timing,
        };
        return exl3::vision::run_infer(&model, &vm, &tok, &opts);
    }

    if args.mtp && model.config.arch_kind != ArchKind::Qwen35 {
        anyhow::bail!("--mtp requires a Qwen3.5 checkpoint with an MTP head");
    }
    let mtp = if args.mtp {
        Some(exl3::mtp::MtpModel::load(&dir, device, &model)?)
    } else {
        None
    };

    let mut text = if args.chat {
        Tok::qwen_chat_prompt(&args.prompt, None)
    } else {
        args.prompt.clone()
    };
    // Reasoning toggle: --no-think (or --reasoning=false) appends an empty
    // think block so the model skips chain-of-thought.
    if args.chat && (args.no_think || !args.reasoning) {
        text.push_str("<think>\n\n</think>\n\n");
    }
    let text = text;
    let ids = tok.encode(&text)?;
    // Pin the prompt's own tokens into the draft head's keep set, so drafting
    // stays strong in whatever script the prompt is written in (no-op unless
    // `EXL3_DRAFT_VOCAB` is set).
    model.refresh_draft_head(&ids)?;

    if let Some(dir) = args.dflash2.clone() {
        let d = exl3::dflash2::DFlash2Model::load(std::path::Path::new(&dir), device)?;
        return run_dflash2(&mut model, &tok, &d, &args, &text, &ids, device);
    }
    if let Some(mtp) = &mtp {
        return run_mtp(&mut model, &tok, mtp, &args, &text, &ids, device);
    }

    // Stream the prompt then tokens straight to stdout; stats go to stderr at the
    // end (matches py-infer.py).
    let mut out = std::io::stdout();
    write!(out, "{text}").ok();
    out.flush().ok();

    let prompt_len = ids.len() as i64;
    let max_len = match args.max_seq_len {
        Some(m) if m > 0 => m.max(prompt_len + 8),
        _ => prompt_len + args.max_new as i64 + 8,
    };
    let qwen4 = model.config.arch_kind == ArchKind::Qwen4Exp;
    let hybrid = model.config.arch_kind == ArchKind::Qwen35;
    let quantized = args.cache_bits > 0 && !hybrid && !qwen4;
    let cache = if qwen4 {
        KvCache::Qwen4(exl3::cache::Qwen4Cache::new(&model.config, max_len, device))
    } else if hybrid {
        if args.cache_bits > 0 {
            eprintln!("(--cache-bits ignored: Qwen3.5 KV-quant not wired yet)");
        }
        KvCache::Qwen35(Qwen35Cache::new(&model.config, max_len, device))
    } else if quantized {
        eprintln!("KV cache: {}-bit quantized", args.cache_bits);
        KvCache::Quant(QuantPagedKvCache::new(
            &model.config, max_len, args.cache_bits, args.cache_bits, device,
        ))
    } else {
        KvCache::Plain(PagedKvCache::new(&model.config, max_len, device))
    };
    let eos = model.config.eos_token_ids.clone();
    let vocab = model.config.vocab_size;

    // graph capture of the quantized path is not wired yet (extra kernels in the
    // step); it gives ~0 speedup on the 8bpw model anyway.
    // qwen4_exp's QSA attention runs in tch over a growing contiguous cache, so
    // its shapes change every step — not capturable.
    //
    // Nor is a MoE stack: `BlockSparseMlp::forward` reads the routing table back
    // to the host once per layer to bucket rows by expert, and a D2H copy of an
    // unpinned tensor is illegal during capture. Making it capturable means
    // routing on the device (upstream's fused multi-GEMM), not a flag here.
    // The MoE decode step is capturable only on the multi-GEMM path, where the
    // routing never leaves the device.
    let moe = model.config.moe.is_some() && !model.moe_is_fused();
    if moe && !args.no_graph {
        eprintln!("(CUDA graph disabled: the MoE router round-trips through the host)");
    }
    let use_graph = !args.no_graph && !quantized && !qwen4 && !moe;

    // --- warmup: prime the exl3_gemm autotuner + lazy kernel loads on the stream
    // capture will use, so nothing capture-illegal (timing loops) runs mid-capture.
    if use_graph {
        CudaGraph::use_side_stream();
        let wcache = if hybrid {
            KvCache::Qwen35(Qwen35Cache::new(&model.config, 512, device))
        } else {
            KvCache::Plain(PagedKvCache::new(&model.config, 512, device))
        };
        let dummy = Tensor::zeros([1, 1], (Kind::Int64, device));
        let warmup_iters: i64 = std::env::var("EXL3_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        for _ in 0..warmup_iters {
            let _ = wcache.forward(&model, &dummy);
            wcache.advance(1);
        }
        CudaGraph::sync_side_stream();
    }

    // --- prefill
    // EXL3_PREFILL_HOLDBACK=1: mimic the batched generator, which ingests only
    // prompt_len-1 tokens during prefill and computes the last prompt token's
    // K/V + logits as a q_len=1 decode step (a different attention kernel).
    let holdback = std::env::var("EXL3_PREFILL_HOLDBACK").is_ok() && prompt_len > 1;
    let pf_len = if holdback { prompt_len - 1 } else { prompt_len };
    let t_prefill = Instant::now();
    // Chunked prefill (tabby `chunk_size`): feed the prompt in page-friendly
    // slices so peak activation memory stays bounded on long contexts. Only the
    // final chunk's logits matter here.
    let chunk = if args.chunk_size > 0 { args.chunk_size } else { pf_len.max(1) };
    let prefill_logits = {
        let mut last = Tensor::zeros([vocab], (Kind::Float, device));
        let mut off = 0i64;
        while off < pf_len {
            let n = (pf_len - off).min(chunk);
            let slice = &ids[off as usize..(off + n) as usize];
            let input = Tensor::from_slice(slice).reshape([1, n]).to_device(device);
            last = cache.forward(&model, &input);
            cache.advance(n);
            off += n;
        }
        last
    };
    tch::Cuda::synchronize(args.gpu as i64);
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let mut next = if holdback {
        ids[prompt_len as usize - 1]
    } else {
        sample(&prefill_logits, &args)
    };
    let mut generated = 0usize;

    // static I/O buffers for the captured graph
    let mut cur_tok = Tensor::zeros([1, 1], (Kind::Int64, device));
    let mut logits_buf = Tensor::zeros([vocab], (Kind::Float, device));

    let mut graph: Option<CudaGraph> = None;
    let mut graphed = false;

    let t_decode = Instant::now();
    let mut skip_emit = holdback; // first pass feeds the held-back prompt token
    for _ in 0..(args.max_new + holdback as usize) {
        if eos.contains(&next) && !skip_emit {
            break;
        }
        if !skip_emit {
            generated += 1;
            write!(out, "{}", tok.decode(&[next])?).ok();
            out.flush().ok();
        }
        skip_emit = false;

        // feed `next`, obtain logits for the following token
        cur_tok.copy_(&Tensor::from_slice(&[next]).reshape([1, 1]).to_device(device));

        if use_graph && !graphed {
            let g = CudaGraph::new();
            let ok = !g.is_null()
                && g.capture(|| {
                    let l = cache.forward(&model, &cur_tok);
                    logits_buf.copy_(&l);
                    cache.advance(1);
                })
                && g.replay();
            if ok {
                graph = Some(g);
                graphed = true;
                next = sample(&logits_buf, &args);
                continue;
            }
            // capture failed — fall back to eager for the rest of the run
            eprintln!("(cuda graph unavailable, running eager)");
        }

        if graphed {
            if !graph.as_ref().unwrap().replay() {
                anyhow::bail!("cuda graph replay failed mid-run");
            }
            next = sample(&logits_buf, &args);
        } else {
            let l = cache.forward(&model, &cur_tok);
            cache.advance(1);
            next = sample(&l, &args);
        }
    }
    tch::Cuda::synchronize(args.gpu as i64);
    let decode_s = t_decode.elapsed().as_secs_f64();
    writeln!(out).ok();
    eprintln!(
        "\x1b[2m{generated} tokens in {decode_s:.2}s — {:.1} tok/s\x1b[0m",
        generated as f64 / decode_s.max(1e-9)
    );

    if args.timing && graphed {
        // pure GPU cost: N replays, one sync
        let n = 100;
        let t = Instant::now();
        for _ in 0..n {
            let _ = graph.as_ref().unwrap().replay();
        }
        CudaGraph::sync_side_stream();
        let s = t.elapsed().as_secs_f64();
        eprintln!("raw replay: {n} steps in {s:.3}s ({:.1} tok/s, {:.2} ms/step)", n as f64 / s, s * 1000.0 / n as f64);
    }

    if args.timing {
        eprintln!(
            "prefill: {prompt_len} tok in {prefill_s:.3}s ({:.1} tok/s)  |  \
             decode: {generated} tok in {decode_s:.3}s ({:.1} tok/s){}",
            prompt_len as f64 / prefill_s,
            generated as f64 / decode_s.max(1e-9),
            if graphed { "  [cuda graph]" } else { "  [eager]" },
        );
    }
    Ok(())
}

/// MTP self-speculative decode (Qwen3.5, `--mtp`). Each round the MTP head drafts
/// `--draft-n` tokens (chained, all on-device), then one `q_len = n+1` trunk
/// forward verifies them (accept-longest-prefix). Greedy. Eager only (no CUDA
/// graph — variable q_len + a second model in the step). Output is streamed to
/// stdout and is byte-identical to non-speculative greedy decode.
fn run_mtp(
    model: &mut Model,
    tok: &Tok,
    mtp: &exl3::mtp::MtpModel,
    args: &Args,
    text: &str,
    ids: &[i64],
    device: Device,
) -> Result<()> {
    use exl3::paged::{pages_for, Qwen35PagedCache};
    let eos = model.config.eos_token_ids.clone();
    let prompt_len = ids.len() as i64;
    let max_len = match args.max_seq_len {
        Some(m) if m > 0 => m.max(prompt_len + args.max_new as i64 + 8),
        _ => prompt_len + args.max_new as i64 + 8,
    };
    if args.temperature > 0.0 {
        eprintln!("(--mtp: greedy self-speculation — --temperature is ignored)");
    }
    let gpu = match device {
        Device::Cuda(i) => i as i64,
        _ => 0,
    };
    let mut out = std::io::stdout();
    write!(out, "{text}").ok();
    out.flush().ok();

    // Trunk verify runs through the batched hybrid cache so the GDN layers get
    // per-token history and can be rewound when a draft is rejected. History
    // depth must cover a full verify forward (q_len = n_draft + 1).
    let n_draft = args.draft_n.max(1);
    let num_pages = pages_for(max_len);
    let cache = Qwen35PagedCache::new_hist(&model.config, num_pages, 1, n_draft + 1, (0, 0), device);
    let block_table =
        Tensor::arange(num_pages, (Kind::Int, device)).reshape([1, num_pages]);
    let slots = Tensor::from_slice(&[0i32]).to_device(device);
    let tok1 = |id: i64| Tensor::from_slice(&[id]).reshape([1, 1]).to_device(device);

    // one-shot prefill with hidden export, then prime the MTP KV
    let t_prefill = Instant::now();
    let input = Tensor::from_slice(ids).reshape([1, prompt_len]).to_device(device);
    let sl0 = Tensor::from_slice(&[0i32]).to_device(device);
    let (hiddens, logits) =
        model.forward_qwen35_batched_h(&input, &cache, &block_table, &sl0, &slots, false, true);
    mtp.prime(model, &hiddens, ids);
    tch::Cuda::synchronize(gpu);
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let mut c = logits
        .select(0, 0)
        .select(0, prompt_len - 1)
        .argmax(0, false)
        .int64_value(&[]); // token @ q, held back
    let mut q = prompt_len;
    let mut h_prev = hiddens.narrow(1, prompt_len - 1, 1).contiguous(); // trunk hidden @ q-1

    if args.bench_qlen {
        // Re-run the forward at a fixed seqlen without advancing the cache, so
        // every q_len is measured against identical context; it rewrites the
        // same KV slots each time, which is meaningless numerically and
        // identical in cost.
        const ITERS: u32 = 60;
        let sl = Tensor::from_slice(&[q as i32]).to_device(device);
        eprintln!("[bench] trunk verify forward, ctx={prompt_len}, {ITERS} iters");
        let mut prev: Option<(i64, f64)> = None;
        for ql in [1i64, 2, 3, 4, 5, 6, 8, 12, 16] {
            let vin = Tensor::zeros([1, ql], (Kind::Int64, device));
            for _ in 0..3 {
                let _ = model
                    .forward_qwen35_batched_h(&vin, &cache, &block_table, &sl, &slots, true, true);
            }
            tch::Cuda::synchronize(args.gpu as i64);
            let t = Instant::now();
            for _ in 0..ITERS {
                let _ = model
                    .forward_qwen35_batched_h(&vin, &cache, &block_table, &sl, &slots, true, true);
            }
            tch::Cuda::synchronize(args.gpu as i64);
            let ms = t.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
            let marg = match prev {
                Some((pq, pms)) => format!("  (+{:.3} ms/pos)", (ms - pms) / (ql - pq) as f64),
                None => String::new(),
            };
            eprintln!("[bench] q_len={ql:2}  {ms:6.2} ms{marg}");
            prev = Some((ql, ms));
        }
        // And the other half of a round: the sequential MTP draft chain. Its
        // slope is the per-step cost that a deeper chain pays and a *wider*
        // tree would not.
        let mut dprev: Option<(i64, f64)> = None;
        for n in [1i64, 2, 4, 8] {
            for _ in 0..3 {
                let _ = mtp.draft_n(model, &h_prev, c, q, n);
            }
            tch::Cuda::synchronize(args.gpu as i64);
            let t = Instant::now();
            for _ in 0..ITERS {
                let _ = mtp.draft_n(model, &h_prev, c, q, n);
            }
            tch::Cuda::synchronize(args.gpu as i64);
            let ms = t.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
            let marg = match dprev {
                Some((pn, pms)) => format!("  (+{:.3} ms/step)", (ms - pms) / (n - pn) as f64),
                None => String::new(),
            };
            eprintln!("[bench] draft_n={n:2}  {ms:6.2} ms{marg}");
            dprev = Some((n, ms));
        }
        return Ok(());
    }

    let mut generated = 0usize;
    let mut drafted = 0u64;
    let mut accepted_extra = 0u64;

    // --- CUDA graph capture of the q_len = n_draft+1 verify forward (the 64-layer
    // trunk pass — the expensive part of each MTP round). Dynamic position lives
    // in `vsl_buf` (RoPE offset + K/V write offset for the paged kernels), so one
    // capture replays at every offset; `block_table` already spans all pages so it
    // never grows. The MTP head draft chain stays eager (tiny, and its q_len=1
    // shape varies with acceptance). `--no-graph` disables.
    let hdim = model.config.hidden_size;
    let vocab = model.config.vocab_size;
    let vq = n_draft + 1;
    let mut vin_buf = Tensor::zeros([1, vq], (Kind::Int64, device));
    let mut vsl_buf = Tensor::zeros([1], (Kind::Int, device));
    let mut vhid_buf = Tensor::zeros([1, vq, hdim], (Kind::Half, device));
    let mut vlog_buf = Tensor::zeros([1, vq, vocab], (Kind::Float, device));
    let mut vgraph: Option<CudaGraph> = None;
    let mut vgraphed = false;
    if !args.no_graph {
        // Prime the exl3_gemm autotuner + lazy kernel loads for this shape on a
        // throwaway cache so the real cache/GDN state is untouched.
        CudaGraph::use_side_stream();
        let wpages = pages_for(512);
        let wcache =
            Qwen35PagedCache::new_hist(&model.config, wpages, 1, n_draft + 1, (0, 0), device);
        let wbt = Tensor::arange(wpages, (Kind::Int, device)).reshape([1, wpages]);
        let wsl = Tensor::zeros([1], (Kind::Int, device));
        let witers: i64 = std::env::var("EXL3_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        for _ in 0..witers {
            let _ = model.forward_qwen35_batched_h(&vin_buf, &wcache, &wbt, &wsl, &slots, true, true);
        }
        CudaGraph::sync_side_stream();
    }

    let t_decode = Instant::now();
    'outer: while generated < args.max_new {
        if eos.contains(&c) {
            break;
        }
        write!(out, "{}", tok.decode(&[c])?).ok();
        out.flush().ok();
        generated += 1;
        if generated >= args.max_new || q + 2 >= max_len {
            break;
        }

        // draft `n` tokens after `c` (positions q+1 .. q+n), clamped to the cache;
        // the whole chain stays on the GPU (no per-step host sync)
        let n = n_draft.min(max_len - q - 1).max(1);
        let drafts_t = mtp.draft_n(model, &h_prev, c, q, n); // [n] i64, on device
        drafted += n as u64;

        // verify all of them in one q_len = n+1 trunk forward through the history
        // cache. Graph path (steady state, n == n_draft): stage inputs into the
        // static buffers and replay. Eager fallback for the tail (n shrinks near
        // max_len) or --no-graph.
        let use_vgraph = !args.no_graph && n == n_draft;
        let (vhid, vlog): (Tensor, Tensor) = if use_vgraph {
            let vin = Tensor::cat(&[tok1(c), drafts_t.reshape([1, n])], 1); // [1, n+1]
            vin_buf.copy_(&vin);
            vsl_buf.copy_(&Tensor::from_slice(&[q as i32]).to_device(device));
            if !vgraphed {
                let g = CudaGraph::new();
                let ok = !g.is_null()
                    && g.capture(|| {
                        let (h, l) = model.forward_qwen35_batched_h(
                            &vin_buf, &cache, &block_table, &vsl_buf, &slots, true, true,
                        );
                        vhid_buf.copy_(&h);
                        vlog_buf.copy_(&l);
                    })
                    && g.replay();
                if ok {
                    vgraph = Some(g);
                    vgraphed = true;
                } else {
                    eprintln!("(mtp verify: cuda graph unavailable, running eager)");
                    let (h, l) = model.forward_qwen35_batched_h(
                        &vin_buf, &cache, &block_table, &vsl_buf, &slots, true, true,
                    );
                    vhid_buf.copy_(&h);
                    vlog_buf.copy_(&l);
                }
            } else if !vgraph.as_ref().unwrap().replay() {
                anyhow::bail!("mtp verify: cuda graph replay failed mid-run");
            }
            (vhid_buf.shallow_clone(), vlog_buf.shallow_clone())
        } else {
            let vin = Tensor::cat(&[tok1(c), drafts_t.reshape([1, n])], 1); // [1, n+1]
            let sl = Tensor::from_slice(&[q as i32]).to_device(device);
            model.forward_qwen35_batched_h(&vin, &cache, &block_table, &sl, &slots, true, true)
        };
        // greedy trunk tokens for positions q+1 .. q+n+1 (r[n] is the bonus token
        // available when every draft is accepted)
        let r_t = vlog.select(0, 0).narrow(0, 0, n + 1).argmax(-1, false); // [n+1] i64, on device

        // one device->host copy for the whole round: n drafts ++ (n+1) trunk tokens
        let combo = Tensor::cat(&[drafts_t, r_t], 0).to_device(Device::Cpu);
        let get = |i: i64| combo.int64_value(&[i]);
        let r: Vec<i64> = (0..=n).map(|j| get(n + j)).collect();
        let k = (0..n)
            .take_while(|&j| get(j) == r[j as usize])
            .count();

        let l = k as i64 + 1; // committed this round: r[0..=k] at positions q+1..q+l
        cache.gdn_rewind(0, l, n + 1);
        mtp.sync_after_accept(model, &vhid, &r[..k], q);

        // emit the k accepted tokens; hold r[k] back as the next `c`
        for &t in &r[..k] {
            write!(out, "{}", tok.decode(&[t])?).ok();
            generated += 1;
            if eos.contains(&t) {
                out.flush().ok();
                accepted_extra += k as u64;
                break 'outer;
            }
        }
        out.flush().ok();
        accepted_extra += k as u64;

        c = r[k];
        // r[..=k] are the trunk's own tokens for this round; teach the draft
        // head any it could not have proposed (no-op once converged)
        model.adapt_draft_head(&r[..=k])?;
        if k >= 1 {
            h_prev = vhid.narrow(1, k as i64, 1).contiguous(); // trunk hidden @ new q-1
            q += l;
        } else {
            h_prev = vhid.narrow(1, 0, 1).contiguous();
            q += 1;
        }
    }
    tch::Cuda::synchronize(gpu);
    let decode_s = t_decode.elapsed().as_secs_f64();
    writeln!(out).ok();

    let pct = if drafted > 0 { 100.0 * accepted_extra as f64 / drafted as f64 } else { 0.0 };
    eprintln!(
        "\x1b[2m{generated} tokens in {decode_s:.2}s — {:.1} tok/s  (MTP: {accepted_extra}/{drafted} drafts accepted, {pct:.0}%)\x1b[0m",
        generated as f64 / decode_s.max(1e-9),
    );
    if args.timing {
        eprintln!(
            "prefill: {prompt_len} tok in {prefill_s:.3}s ({:.1} tok/s)  |  decode: {generated} tok in {decode_s:.3}s ({:.1} tok/s)  [mtp]",
            prompt_len as f64 / prefill_s,
            generated as f64 / decode_s.max(1e-9),
        );
        // Attribute the verify forward. It is ~97% of a speculative round, so
        // this is the breakdown that decides what is worth optimising.
        if *exl3::model::trunk_prof_on() {
            tch::Cuda::synchronize(0);
            let (att, gdn, mlp, norm) = exl3::model::trunk_prof_take();
            let (gp, gc, gr, go) = exl3::model::gdn_prof_take();
            eprintln!(
                "[trunk] full_attn {att:.0}ms  gdn {gdn:.0}ms  mlp {mlp:.0}ms  norm {norm:.0}ms\n\
                 [gdn]   proj {gp:.0}ms  conv {gc:.0}ms  delta_rule {gr:.0}ms  norm+out {go:.0}ms"
            );
        }
    }
    Ok(())
}

/// Single-sequence DFlash2 speculative decode.
///
/// A round is: one draft forward proposing a whole block, one trunk verify over
/// the block, accept-while-match, then fold the verify's own hidden states back
/// into the drafter's cache. The drafter never runs over the context — its K/V
/// come from the target's taps — so its cost per round is fixed regardless of
/// how many tokens it proposes, which is the whole point versus MTP's
/// `n × lm_head`.
#[allow(clippy::too_many_arguments)]
fn run_dflash2(
    model: &mut Model,
    tok: &Tok,
    d: &exl3::dflash2::DFlash2Model,
    args: &Args,
    text: &str,
    ids: &[i64],
    device: Device,
) -> Result<()> {
    use exl3::paged::{pages_for, Qwen35PagedCache};
    let eos = model.config.eos_token_ids.clone();
    let prompt_len = ids.len() as i64;
    let bs = d.params.block_size; // 8
    let _n = bs - 1; // drafted tokens per round
    let max_len = match args.max_seq_len {
        Some(m) if m > 0 => m.max(prompt_len + args.max_new as i64 + bs + 8),
        _ => prompt_len + args.max_new as i64 + bs + 8,
    };
    let gpu = match device {
        Device::Cuda(i) => i as i64,
        _ => 0,
    };
    let taps = d.params.target_layer_ids.clone();
    let mut out = std::io::stdout();
    write!(out, "{text}").ok();
    out.flush().ok();

    let num_pages = pages_for(max_len);
    let cache = Qwen35PagedCache::new_hist(&model.config, num_pages, 1, bs, (0, 0), device);
    let block_table = Tensor::arange(num_pages, (Kind::Int, device)).reshape([1, num_pages]);
    let slots = Tensor::from_slice(&[0i32]).to_device(device);
    let mut dcache = exl3::dflash2::DFlash2Cache::new(d, device);

    // Prefill: the trunk consumes the prompt and hands the drafter its taps.
    let t_prefill = Instant::now();
    let input = Tensor::from_slice(ids).reshape([1, prompt_len]).to_device(device);
    let sl0 = Tensor::from_slice(&[0i32]).to_device(device);
    let (hiddens, tap_states) = model.forward_qwen35_batched_h_taps(
        &input, &cache, &block_table, &sl0, &slots, false, &taps,
    );
    d.update_kv_from_target(&mut dcache, &tap_states, 0);
    let logits = model.lm_head_on(&hiddens.narrow(1, prompt_len - 1, 1));
    tch::Cuda::synchronize(gpu);
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // The prefill's own token sits at position q and is a real output token —
    // the rounds below only ever emit what comes *after* the current anchor, so
    // it has to be emitted here or the response loses its first token.
    let mut c = logits.reshape([-1]).argmax(0, false).int64_value(&[]);
    let mut q = prompt_len;
    if eos.contains(&c) {
        writeln!(out).ok();
        return Ok(());
    }
    write!(out, "{}", tok.decode(&[c])?).ok();
    out.flush().ok();

    let mut generated = 1u64;
    let mut drafted = 0u64;
    let mut accepted_extra = 0u64;
    let t_dec = Instant::now();

    'outer: while generated < args.max_new as u64 {
        // ---- draft: one forward for the whole block
        let dr = d.draft(model, &dcache, c);
        let nd = dr.len() as i64;
        drafted += nd as u64;

        // ---- verify: [c, drafts] in one trunk forward, with taps
        let mut vin_ids = vec![c];
        vin_ids.extend_from_slice(&dr);
        let vin = Tensor::from_slice(&vin_ids).reshape([1, nd + 1]).to_device(device);
        let sl = Tensor::from_slice(&[q as i32]).to_device(device);
        let (vhid, vtaps) = model.forward_qwen35_batched_h_taps(
            &vin, &cache, &block_table, &sl, &slots, true, &taps,
        );
        let vlog = model.lm_head_on(&vhid);
        let r_t = vlog.select(0, 0).argmax(-1, false).to_device(Device::Cpu);
        let r: Vec<i64> = (0..=nd).map(|j| r_t.int64_value(&[j])).collect();
        let k = (0..nd).take_while(|&j| dr[j as usize] == r[j as usize]).count() as i64;

        // Committed this round: r[0..k] at q+1..q+k, then r[k] becomes the next
        // anchor at q+k+1. Verify inputs q..q+k were the *correct* tokens (that
        // is what acceptance means), so their taps are valid and can be folded
        // into the draft cache; the first rejected position's tap cannot.
        cache.gdn_rewind(0, k + 1, nd + 1);
        let keep: Vec<Tensor> = vtaps.iter().map(|t| t.narrow(1, 0, k + 1)).collect();
        d.update_kv_from_target(&mut dcache, &keep, q);

        for &t in &r[..k as usize] {
            if eos.contains(&t) {
                out.flush().ok();
                accepted_extra += k as u64;
                break 'outer;
            }
            write!(out, "{}", tok.decode(&[t])?).ok();
            generated += 1;
        }
        out.flush().ok();
        accepted_extra += k as u64;

        c = r[k as usize];
        if eos.contains(&c) {
            break 'outer;
        }
        write!(out, "{}", tok.decode(&[c])?).ok();
        generated += 1;
        out.flush().ok();
        q += k + 1;
        model.adapt_draft_head(&r[..=k as usize])?;
    }

    tch::Cuda::synchronize(gpu);
    let decode_s = t_dec.elapsed().as_secs_f64();
    let pct = if drafted > 0 { 100.0 * accepted_extra as f64 / drafted as f64 } else { 0.0 };
    writeln!(out).ok();
    eprintln!(
        "\x1b[2mprefill {prompt_len} tokens in {prefill_s:.2}s\x1b[0m"
    );
    eprintln!(
        "\x1b[2m{generated} tokens in {decode_s:.2}s — {:.1} tok/s  (DFlash2: {accepted_extra}/{drafted} drafts accepted, {pct:.0}%)\x1b[0m",
        generated as f64 / decode_s
    );
    Ok(())
}
