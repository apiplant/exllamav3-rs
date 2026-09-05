//! OpenAI-compatible inference server, driven by a TabbyAPI-style `config.yml`.
//!
//! ```text
//! server --config /mnt/extra/ai/llm/tabbyAPI/config.yml
//! server --config config.yml --host 0.0.0.0 --port 8080
//! ```
//!
//! All model / draft / sampling / networking behaviour comes from the config
//! file; CLI flags only override `network.host` / `network.port` / the GPU
//! index and point at the file. See `exl3::server::config` for the honored keys.

use anyhow::Result;
use clap::Parser;
use exl3::server::config::ServerConfig;
use exl3::server::engine::{self, EngineConfig};
use exl3::server::http::{self, AppState};
use ntex::web::{self, App, HttpServer};
use std::sync::Arc;

#[derive(Parser)]
#[command(about = "OpenAI-compatible EXL3 inference server")]
struct Args {
    /// Path to config.yml (TabbyAPI-compatible).
    #[arg(long, default_value = "config.yml")]
    config: String,
    /// Override network.host.
    #[arg(long)]
    host: Option<String>,
    /// Override network.port.
    #[arg(long)]
    port: Option<u16>,
    /// CUDA device index.
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    /// API key required on requests (overrides EXL3_API_KEY; ignored if
    /// network.disable_auth is set).
    #[arg(long)]
    api_key: Option<String>,
}

#[ntex::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    exl3::server::log::init();

    let mut cfg = ServerConfig::load(std::path::Path::new(&args.config))?;
    if let Some(h) = args.host {
        cfg.network.host = Some(h);
    }
    if let Some(p) = args.port {
        cfg.network.port = Some(p);
    }

    println!("exllamav3-rs server");
    println!("config: {}", args.config);
    print!("{}", cfg.report());

    let api_key = if cfg.network.disable_auth {
        None
    } else {
        args.api_key
            .clone()
            .or_else(|| std::env::var("EXL3_API_KEY").ok())
            .or_else(|| {
                // generate one so the server is not silently open
                let k = format!("sk-exl3-{:x}", exl3::server::oai::now());
                println!("\n  no API key configured — generated: {k}");
                println!("  (set network.disable_auth: true, --api-key, or EXL3_API_KEY to change)");
                Some(k)
            })
    };

    let ec = EngineConfig::from_server_config(&cfg, args.gpu)?;
    let model_id = cfg
        .model
        .model_name
        .clone()
        .unwrap_or_else(|| "exl3-model".into());

    println!();
    let t0 = std::time::Instant::now();
    // On a startup failure (usually CUDA OOM) exit hard via `_exit`: a
    // half-initialised CUDA context deadlocks in its atexit handler (it waits on
    // the engine thread, which is stuck in the failed context), so
    // `std::process::exit` — which runs atexit handlers — hangs and leaves a
    // multi-GB zombie holding the GPU. `_exit` skips all of that.
    let handle = match engine::spawn(ec, model_id) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("\nfailed to start: {e}");
            unsafe { libc::_exit(1) };
        }
    };
    exl3::sinfo!(
        "model ready in {:.1}s — arch={} ctx={} mode={} vision={}",
        t0.elapsed().as_secs_f64(),
        handle.meta.arch,
        handle.meta.n_ctx,
        handle.meta.mode,
        handle.meta.has_vision,
    );

    let host = cfg.host();
    let port = cfg.port();
    let state = AppState {
        eng: handle,
        cfg: Arc::new(cfg),
        api_key,
    };

    exl3::sinfo!("listening on http://{host}:{port}  (POST /v1/chat/completions)");

    // Hard-exit on Ctrl-C / SIGTERM. ntex's graceful shutdown waits on the engine
    // thread, and a CUDA context can take seconds to tear down — long enough that
    // a quick restart races the old process for VRAM. Just exit.
    ntex::rt::spawn(async {
        let mut term =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
        exl3::swarn!("Shutdown signal received. Exiting.");
        unsafe { libc::_exit(0) };
    });

    HttpServer::new(move || {
        App::new()
            .state(state.clone())
            // allow large bodies (base64 images in vision requests)
            .state(web::types::PayloadConfig::new(64 * 1024 * 1024))
            .wrap(http::AccessLog)
            .wrap(
                web::middleware::DefaultHeaders::new()
                    .header("access-control-allow-origin", "*")
                    .header("access-control-allow-headers", "authorization, content-type, x-api-key")
                    .header("access-control-allow-methods", "GET, POST, OPTIONS"),
            )
            .route("/health", web::get().to(http::health))
            .route("/v1/models", web::get().to(http::list_models))
            .route("/v1/internal/model/info", web::get().to(http::model_info))
            .route("/v1/chat/completions", web::post().to(http::chat_completions))
            .route("/v1/completions", web::post().to(http::completions))
            .route("/v1/responses", web::post().to(http::responses))
            .default_service(web::route().to(http::fallback))
    })
    .bind((host.as_str(), port))?
    .workers(2)
    .run()
    .await?;

    Ok(())
}
