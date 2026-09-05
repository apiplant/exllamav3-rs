//! A llama.cpp-style HTTP server with an OpenAI-compatible API, driven by a
//! TabbyAPI-style `config.yml`.
//!
//! ```text
//! server --config /path/to/config.yml
//! ```
//!
//! Endpoints: `GET /v1/models`, `GET /health`, `POST /v1/chat/completions`,
//! `POST /v1/completions`, `GET /v1/internal/model/info`. Streaming (SSE) and
//! non-streaming are both supported. See [`config::ServerConfig`] for which
//! `config.yml` keys are honored.

pub mod chat;
pub mod config;
pub mod engine;
pub mod http;
pub mod log;
pub mod oai;

pub use config::ServerConfig;
pub use engine::{EngineConfig, EngineHandle};
