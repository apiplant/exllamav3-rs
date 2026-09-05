//! Demo driver for the dynamic-batching `Generator`: enqueue several prompts at
//! once and stream all their completions concurrently through one shared cache.

use anyhow::Result;
use clap::Parser;
use exl3::{
    generator::{Generator, JobSpec, Stage},
    model::Model,
    sampler::SamplerSettings,
    tokenizer::Tok,
};
use std::collections::HashSet;
use std::io::Write;
use std::time::Instant;
use tch::Device;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    /// One or more prompts (repeat --prompt); each becomes a concurrent job
    #[arg(long = "prompt", required = true)]
    prompts: Vec<String>,
    #[arg(long)]
    chat: bool,
    #[arg(long, default_value_t = 128)]
    max_new: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0.0)]
    min_p: f64,
    #[arg(long, default_value_t = 1.0)]
    rep_penalty: f64,
    /// presence penalty (additive)
    #[arg(long, default_value_t = 0.0)]
    pres_penalty: f64,
    /// frequency penalty (additive, scaled by count)
    #[arg(long, default_value_t = 0.0)]
    freq_penalty: f64,
    /// penalty sustain window in tokens (0 = whole history)
    #[arg(long, default_value_t = 0)]
    sustain: i64,
    /// penalty linear-decay window in tokens after the sustain window
    #[arg(long, default_value_t = 0)]
    decay: i64,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    /// shared cache size in 256-token pages
    #[arg(long, default_value_t = 64)]
    pages: i64,
    /// max concurrent decode rows
    #[arg(long, default_value_t = 8)]
    max_batch: usize,
    #[arg(long, default_value_t = 512)]
    max_chunk: i64,
    /// minimum suffix-match length for n-gram speculative decoding (0 = off)
    #[arg(long, default_value_t = 0)]
    ngram_min: usize,
    /// draft length per round for speculative decoding (n-gram / MTP / draft-model)
    #[arg(long, default_value_t = 4)]
    ngram_draft: i64,
    /// MTP self-speculation inside the batched loop (Qwen3.5 checkpoint with an MTP head)
    #[arg(long)]
    mtp: bool,
    /// separate autoregressive draft model directory (speculative decoding)
    #[arg(long)]
    draft_model: Option<String>,
    /// regenerate the last prompt token constrained to its text prefix
    #[arg(long)]
    token_healing: bool,
    /// loop-detector window in tokens (0 = off)
    #[arg(long, default_value_t = 0)]
    loop_window: i64,
    /// minimum repetitions across the window to call it a loop
    #[arg(long, default_value_t = 3)]
    loop_reps: usize,
    /// quantize the shared KV cache to N bits (2..=8; 0 = fp16)
    #[arg(long, default_value_t = 0)]
    cache_bits: i64,
    /// share identical prompt-prefix KV pages between concurrent jobs
    #[arg(long)]
    prefix_cache: bool,
    /// fair-scheduling: requeue a job once it has generated this many tokens
    /// (0 = off), bounding its cache footprint
    #[arg(long, default_value_t = 0)]
    requeue_budget: usize,
    /// pinned host-RAM CPU cache tier size in tokens (0 = off; implies --prefix-cache)
    #[arg(long, default_value_t = 0)]
    cpu_cache_tokens: i64,
    /// constrain output to a GBNF-subset grammar (string or @path)
    #[arg(long)]
    grammar: Option<String>,
    /// constrain output to a JSON schema (path to a .json file)
    #[arg(long)]
    json_schema: Option<String>,
    /// classifier-free guidance: negative prompt (applied to every --prompt)
    #[arg(long)]
    negative_prompt: Option<String>,
    /// CFG scale (1.0 = no effect; >1 sharpens toward the positive prompt)
    #[arg(long, default_value_t = 1.5)]
    cfg_scale: f64,
    #[arg(long)]
    stream: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cuda(args.gpu);
    let dir = std::path::PathBuf::from(&args.model);

    eprintln!("loading model...");
    let model = Model::load(&dir, device)?;
    let tok = Tok::load(&dir)?;
    let eos: HashSet<i64> = model.config.eos_token_ids.iter().copied().collect();

    let kv_bits = if args.cache_bits > 0 { (args.cache_bits, args.cache_bits) } else { (0, 0) };
    let mut g = Generator::new(model, tok, args.pages, args.max_batch, args.max_chunk, kv_bits);
    if args.ngram_min > 0 {
        g.enable_ngram(args.ngram_min, args.ngram_draft);
        eprintln!("n-gram speculative decode: match_min={}, draft={}", args.ngram_min, args.ngram_draft);
    } else if args.mtp {
        let mtp = exl3::mtp::MtpModel::load_headless(&dir, device, g.model())?;
        g.enable_mtp(mtp, args.ngram_draft);
        eprintln!("MTP speculative decode: draft={}", args.ngram_draft);
    } else if let Some(dm) = &args.draft_model {
        let draft = exl3::draft::DraftModel::load(
            std::path::Path::new(dm),
            device,
            g.model(),
            g.num_pages(),
        )?;
        g.enable_draft(draft, args.ngram_draft);
        eprintln!("draft-model speculative decode: {dm} draft={}", args.ngram_draft);
    }
    if args.cache_bits > 0 {
        g.enable_cache_quant(args.cache_bits, args.cache_bits);
        eprintln!("KV cache: {}-bit quantized", args.cache_bits);
    }
    if args.prefix_cache {
        g.enable_prefix_cache();
        eprintln!("prefix-cache dedup: on");
    }
    if args.requeue_budget > 0 {
        g.enable_requeue(args.requeue_budget);
        eprintln!("fair-scheduling requeue: budget {} tokens", args.requeue_budget);
    }
    if args.cpu_cache_tokens > 0 {
        g.enable_cpu_cache(args.cpu_cache_tokens);
        eprintln!("CPU cache tier: {} tokens pinned host RAM", args.cpu_cache_tokens);
    }

    let sampler = SamplerSettings {
        temperature: args.temperature,
        min_p: args.min_p,
        rep_penalty: args.rep_penalty,
        pres_penalty: args.pres_penalty,
        freq_penalty: args.freq_penalty,
        sustain_range: args.sustain,
        decay_range: args.decay,
        ..SamplerSettings::greedy()
    };

    let grammar_src = match &args.grammar {
        Some(s) if s.starts_with('@') => Some(std::fs::read_to_string(&s[1..])?),
        Some(s) => Some(s.clone()),
        None => None,
    };
    let schema_val: Option<serde_json::Value> = match &args.json_schema {
        Some(path) => Some(serde_json::from_str(&std::fs::read_to_string(path)?)?),
        None => None,
    };

    for p in &args.prompts {
        let text = if args.chat { Tok::qwen_chat_prompt(p, None) } else { p.clone() };
        let ids = g.tokenizer().encode(&text)?;
        let mut spec = JobSpec::new(ids, args.max_new);
        spec.sampler = sampler.clone();
        spec.stop_tokens = eos.clone();
        spec.token_healing = args.token_healing;
        if args.loop_window > 0 {
            spec.stop_on_loop = Some((args.loop_window, args.loop_reps));
        }
        if let Some(src) = &grammar_src {
            spec.filters = vec![g.compile_grammar(src)?];
        } else if let Some(sv) = &schema_val {
            spec.filters = vec![g.compile_json_schema(sv)?];
        }
        if let Some(np) = &args.negative_prompt {
            let ntext = if args.chat { Tok::qwen_chat_prompt(np, None) } else { np.clone() };
            let nids = g.tokenizer().encode(&ntext)?;
            spec.cfg = Some((nids, args.cfg_scale));
        }
        g.enqueue(spec);
    }

    let t0 = Instant::now();
    let mut total_tokens = 0usize;
    let n_jobs = args.prompts.len();

    while g.num_remaining() > 0 {
        for ev in g.iterate()? {
            match ev.stage {
                Stage::Started => eprintln!("[job {}] started", ev.serial),
                Stage::Prefill { done, total } => {
                    eprintln!("[job {}] prefill {done}/{total}", ev.serial)
                }
                Stage::Streaming => {
                    total_tokens += 1;
                    if args.stream && !ev.text.is_empty() {
                        print!("\x1b[36m[{}]\x1b[0m{}", ev.serial, ev.text);
                        std::io::stdout().flush().ok();
                    }
                    if ev.eos {
                        let reason = ev.eos_reason.unwrap_or_default();
                        if let Some(full) = ev.full_text {
                            if !args.stream {
                                println!("\n===== job {} ({} tok, {reason}) =====\n{full}", ev.serial, ev.new_tokens);
                            } else {
                                eprintln!("\n[job {}] done: {reason} ({} tok)", ev.serial, ev.new_tokens);
                            }
                        }
                    }
                }
            }
        }
    }

    let s = t0.elapsed().as_secs_f64();
    eprintln!(
        "\n{n_jobs} jobs, {total_tokens} tokens in {s:.2}s — {:.1} tok/s aggregate",
        total_tokens as f64 / s
    );
    if args.ngram_min > 0 || args.mtp || args.draft_model.is_some() {
        let (acc, drafted) = g.spec_stats();
        let rate = if drafted > 0 { acc as f64 / drafted as f64 * 100.0 } else { 0.0 };
        eprintln!("speculative: {acc}/{drafted} draft tokens accepted ({rate:.1}%)");
    }
    if args.prefix_cache || args.cpu_cache_tokens > 0 {
        let (cached, total) = g.prefix_stats();
        let rate = if total > 0 { cached as f64 / total as f64 * 100.0 } else { 0.0 };
        eprintln!("prefix-cache: {cached}/{total} prompt tokens reused ({rate:.1}%)");
    }
    if args.cpu_cache_tokens > 0 {
        let (restored, pushed) = g.cpu_cache_stats();
        eprintln!("CPU cache tier: {restored} pages restored, {pushed} pages spilled");
    }
    Ok(())
}
