//! ntex HTTP layer: OpenAI-compatible routes over the engine channel.

use crate::server::chat::{extract_tool_calls, render_chatml, ToolCall};
use crate::server::config::ServerConfig;
use crate::server::engine::{Chunk, EngineHandle, EngineRequest, GenStats, GrammarSpec};
use crate::server::oai::{self, FinishReason};
use crate::server::responses;
use ntex::util::Bytes;
use ntex::web::{self, HttpRequest, HttpResponse};
use serde_json::{json, Value};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct AppState {
    pub eng: EngineHandle,
    pub cfg: Arc<ServerConfig>,
    pub api_key: Option<String>,
}

/// Sets the cancel flag if the response future is dropped (client disconnect).
struct CancelGuard(Arc<AtomicBool>);
impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn authorized(st: &AppState, req: &HttpRequest) -> bool {
    let Some(want) = &st.api_key else { return true };
    let hdr = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_start_matches("Bearer ").trim().to_string())
        .or_else(|| {
            req.headers()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim().to_string())
        });
    hdr.as_deref() == Some(want.as_str())
}

fn err_response(code: u16, msg: &str, kind: &str) -> HttpResponse {
    let body = oai::error_body(msg, kind);
    let mut b = match code {
        400 => HttpResponse::BadRequest(),
        401 => HttpResponse::Unauthorized(),
        404 => HttpResponse::NotFound(),
        _ => HttpResponse::InternalServerError(),
    };
    b.json(&body)
}

fn parse_body(raw: &Bytes) -> Result<Value, HttpResponse> {
    serde_json::from_slice::<Value>(raw)
        .map_err(|e| err_response(400, &format!("invalid JSON body: {e}"), "invalid_request_error"))
}

// --- GET /v1/models, /health, model info ------------------------------------

pub async fn list_models(st: web::types::State<AppState>) -> HttpResponse {
    let m = &st.eng.meta;
    let mut data = vec![json!({
        "id": m.id, "object": "model", "created": oai::now(), "owned_by": "exllamav3-rs",
        "meta": { "n_ctx": m.n_ctx, "arch": m.arch, "vision": m.has_vision, "mode": m.mode },
    })];
    if let Some(names) = &st.cfg.model.dummy_model_names {
        for n in names {
            data.push(json!({ "id": n, "object": "model", "created": oai::now(), "owned_by": "exllamav3-rs" }));
        }
    }
    HttpResponse::Ok().json(&json!({ "object": "list", "data": data }))
}

pub async fn fallback(req: HttpRequest) -> HttpResponse {
    if req.method() == ntex::http::Method::OPTIONS {
        return HttpResponse::NoContent().finish();
    }
    err_response(404, "not found", "not_found")
}

pub async fn health() -> HttpResponse {
    HttpResponse::Ok().json(&json!({ "status": "ok" }))
}

pub async fn model_info(st: web::types::State<AppState>) -> HttpResponse {
    let m = &st.eng.meta;
    HttpResponse::Ok().json(&json!({
        "id": m.id, "arch": m.arch, "n_ctx": m.n_ctx, "vocab_size": m.vocab_size,
        "vision": m.has_vision, "mode": m.mode,
        "eos_token_ids": m.eos, "bos_token_id": m.bos,
    }))
}

// --- shared request assembly ----------------------------------------------

struct Prepared {
    req: EngineRequest,
    rx: flume::Receiver<Chunk>,
    cancel: Arc<AtomicBool>,
    stream: bool,
    prompt_tokens: usize,
    model_id: String,
    parse_tools: bool,
    is_chat: bool,
    echo_text: String,
}

/// Parse constrained-decoding from an OpenAI `response_format` block or the
/// vLLM-style `guided_json` / `guided_grammar` extensions (feature 4).
fn parse_grammar(body: &Value) -> Option<GrammarSpec> {
    if let Some(g) = body.get("guided_grammar").and_then(Value::as_str) {
        return Some(GrammarSpec::Gbnf(g.to_string()));
    }
    if let Some(j) = body.get("guided_json") {
        return Some(GrammarSpec::JsonSchema(j.clone()));
    }
    let rf = body.get("response_format")?;
    match rf.get("type").and_then(Value::as_str) {
        Some("json_object") => Some(GrammarSpec::JsonObject),
        Some("json_schema") => {
            let schema = rf
                .get("json_schema")
                .and_then(|js| js.get("schema"))
                .or_else(|| rf.get("schema"))
                .cloned()
                .unwrap_or(Value::Null);
            Some(GrammarSpec::JsonSchema(schema))
        }
        Some("grammar") => rf
            .get("grammar")
            .and_then(Value::as_str)
            .map(|s| GrammarSpec::Gbnf(s.to_string())),
        _ => None,
    }
}

fn enable_thinking(cfg: &ServerConfig, body: &Value) -> Option<bool> {
    let from_req = body
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool)
        .or_else(|| body.get("enable_thinking").and_then(Value::as_bool))
        .or_else(|| {
            body.get("reasoning_effort").and_then(Value::as_str).map(|e| e != "none")
        });
    // template_vars_force wins; then request; then default; then `reasoning` off => empty block
    let force = mapping_bool(&cfg.model.template_vars_force, "enable_thinking");
    let deflt = mapping_bool(&cfg.model.template_vars_default, "enable_thinking");
    force
        .or(from_req)
        .or(deflt)
        .or(if cfg.model.reasoning { None } else { Some(false) })
}

fn mapping_bool(v: &Option<serde_yaml_ng::Value>, key: &str) -> Option<bool> {
    if let Some(serde_yaml_ng::Value::Mapping(m)) = v {
        return m.get(serde_yaml_ng::Value::from(key)).and_then(|x| x.as_bool());
    }
    None
}

fn prepare(st: &AppState, body: &Value, is_chat: bool) -> Result<Prepared, HttpResponse> {
    let eng = &st.eng;
    let cfg = &st.cfg;
    let m = &eng.meta;

    let r_start = cfg
        .model
        .reasoning_start_token
        .clone()
        .unwrap_or_else(|| "<think>".into());
    let r_end = cfg
        .model
        .reasoning_end_token
        .clone()
        .unwrap_or_else(|| "</think>".into());

    let (prompt_text, images, mut start_in_reasoning, echo_text) = if is_chat {
        let messages = body
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| err_response(400, "`messages` is required", "invalid_request_error"))?;
        let tools = body.get("tools").and_then(Value::as_array).map(|v| v.as_slice());
        let et = enable_thinking(cfg, body);
        // `reasoning_effort: "none"` is the OpenAI spelling for "don't think" and
        // is already folded into `et`; it is not a template effort level.
        let effort = body
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .filter(|e| *e != "none");
        let rp = render_chatml(messages, tools, et, effort, &r_end);
        (rp.text, rp.images, rp.starts_in_reasoning, String::new())
    } else {
        let prompt = match body.get("prompt") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) => a.first().and_then(Value::as_str).unwrap_or("").to_string(),
            _ => return Err(err_response(400, "`prompt` is required", "invalid_request_error")),
        };
        let echo = if body.get("echo").and_then(Value::as_bool).unwrap_or(false) {
            prompt.clone()
        } else {
            String::new()
        };
        (prompt, Vec::new(), false, echo)
    };

    match cfg
        .model
        .start_in_reasoning
        .as_deref()
        .unwrap_or("auto")
        .trim()
    {
        "always" => start_in_reasoning = true,
        "never" => start_in_reasoning = false,
        _ => {}
    }

    let prompt_ids = eng
        .encode(&prompt_text)
        .map_err(|e| err_response(500, &format!("tokenise: {e}"), "server_error"))?;
    let prompt_tokens = prompt_ids.len();

    // classifier-free guidance: `negative_prompt` + `guidance_scale`/`cfg_scale`
    let cfg_pair = match body.get("negative_prompt").and_then(Value::as_str) {
        Some(neg) if !neg.is_empty() => {
            let ntext = if is_chat {
                let msgs = json!([{ "role": "user", "content": neg }]);
                render_chatml(msgs.as_array().unwrap(), None, Some(false), None, &r_end).text
            } else {
                neg.to_string()
            };
            let scale = body
                .get("guidance_scale")
                .or_else(|| body.get("cfg_scale"))
                .and_then(Value::as_f64)
                .unwrap_or(1.5);
            match eng.encode(&ntext) {
                Ok(ids) => Some((ids, scale)),
                Err(e) => {
                    return Err(err_response(
                        500,
                        &format!("tokenise negative_prompt: {e}"),
                        "server_error",
                    ))
                }
            }
        }
        _ => None,
    };
    let ctx_room = (m.n_ctx as usize).saturating_sub(prompt_tokens);

    let sp = oai::parse_sample_params(body, &cfg.sampler_defaults(), ctx_room);
    if sp.n > 1 {
        return Err(err_response(
            400,
            "`n` > 1 is not supported by this server",
            "invalid_request_error",
        ));
    }

    let parse_tools = is_chat
        && cfg.model.tool_format.as_deref().map(|s| !s.is_empty()).unwrap_or(false)
        && body.get("tools").and_then(Value::as_array).map(|t| !t.is_empty()).unwrap_or(false);

    let stream = sp.stream && !cfg.developer.disable_request_streaming;

    // ChatML end marker — always a stop for chat turns (Qwen `<|im_end|>` is not
    // always in eos_token_ids).
    let mut stop_strings = sp.stop_strings;
    if is_chat {
        stop_strings.push("<|im_end|>".to_string());
        stop_strings.push("<|endoftext|>".to_string());
    }

    let (reply, rx) = flume::bounded::<Chunk>(512);
    let cancel = Arc::new(AtomicBool::new(false));

    let req = EngineRequest {
        prompt_ids,
        images,
        max_new: sp.max_new,
        min_new: sp.min_new,
        sampler: sp.sampler,
        stop_strings,
        seed: sp.seed,
        grammar: parse_grammar(body),
        cfg: cfg_pair,
        parse_reasoning: cfg.model.reasoning,
        reasoning_start: r_start,
        reasoning_end: r_end,
        start_in_reasoning,
        reply,
        cancel: cancel.clone(),
    };

    Ok(Prepared {
        req,
        rx,
        cancel,
        stream,
        prompt_tokens,
        model_id: m.id.clone(),
        parse_tools,
        is_chat,
        echo_text,
    })
}

// --- POST /v1/chat/completions & /v1/completions -------------------------

pub async fn chat_completions(
    st: web::types::State<AppState>,
    http_req: HttpRequest,
    raw: Bytes,
) -> HttpResponse {
    handle(st, http_req, raw, true).await
}

pub async fn completions(
    st: web::types::State<AppState>,
    http_req: HttpRequest,
    raw: Bytes,
) -> HttpResponse {
    handle(st, http_req, raw, false).await
}

async fn handle(
    st: web::types::State<AppState>,
    http_req: HttpRequest,
    raw: Bytes,
    is_chat: bool,
) -> HttpResponse {
    if !authorized(&st, &http_req) {
        return err_response(401, "missing or invalid API key", "authentication_error");
    }
    let body = match parse_body(&raw) {
        Ok(b) => b,
        Err(r) => return r,
    };
    if st.cfg.logging.log_requests {
        crate::sinfo!("Request body: {}", serde_json::to_string(&body).unwrap_or_default());
    }
    let t_recv = Instant::now();
    let prep = match prepare(&st, &body, is_chat) {
        Ok(p) => p,
        Err(r) => return r,
    };

    let Prepared {
        req,
        rx,
        cancel,
        stream,
        prompt_tokens,
        model_id,
        parse_tools,
        is_chat,
        echo_text,
    } = prep;

    if st.eng.tx.send_async(req).await.is_err() {
        return err_response(500, "engine unavailable", "server_error");
    }

    let ping = st.cfg.network.sse_ping_interval.unwrap_or(15);
    let id = oai::rand_id(if is_chat { "chatcmpl" } else { "cmpl" });

    let log_m = st.cfg.logging.log_chat_completion_requests;
    if log_m {
        crate::sinfo!(
            "Received {}{} request {id}",
            if is_chat { "chat completion" } else { "completion" },
            if stream { " streaming" } else { "" },
        );
    }
    let m = Metrics { id: id.clone(), t_recv, stream, is_chat, enabled: log_m };

    if stream {
        stream_response(
            rx, cancel, id, model_id, is_chat, parse_tools, prompt_tokens, echo_text, ping, m,
        )
    } else {
        buffered_response(
            rx, cancel, id, model_id, is_chat, parse_tools, prompt_tokens, echo_text, m,
        )
        .await
    }
}

/// Inference-speed logging for one request, gated on
/// `logging.log_chat_completion_requests`. Mirrors TabbyAPI's `Metrics (ID: …)`
/// line: queue latency, generation throughput, and total context.
struct Metrics {
    id: String,
    t_recv: Instant,
    stream: bool,
    is_chat: bool,
    enabled: bool,
}

impl Metrics {
    fn parsed_tools(&self, n: usize) {
        if self.enabled && n > 0 {
            crate::sinfo!("Parsed {n} tool calls in chat completion request {}", self.id);
        }
    }

    fn finish(&self, t_end: Instant, completion_tokens: usize, s: &GenStats) {
        if !self.enabled {
            return;
        }
        let total = t_end.duration_since(self.t_recv).as_secs_f64();
        let queue = (total - s.prefill_secs - s.gen_secs).max(0.0);
        let prompt_tps = if s.prefill_secs > 1e-6 {
            s.new_prompt_tokens as f64 / s.prefill_secs
        } else {
            0.0
        };
        let gen_tps = if s.gen_secs > 1e-6 {
            completion_tokens as f64 / s.gen_secs
        } else {
            0.0
        };
        let context = s.cached_prompt_tokens + s.new_prompt_tokens;
        let draft_pct = if s.draft_total > 0 {
            s.draft_accepted as f64 / s.draft_total as f64 * 100.0
        } else {
            0.0
        };
        crate::sinfo!(
            "Finished {}{} request {}",
            if self.is_chat { "chat completion" } else { "completion" },
            if self.stream { " streaming" } else { "" },
            self.id,
        );
        use crate::server::log::num;
        crate::sinfo!(
            "Metrics (ID: {}): {} tokens generated in {} seconds \
             (Queue: {} s, Process: {} cached tokens and {} new tokens at {} T/s, \
             Generate: {} T/s, Context: {} tokens, Draft: {} / {} tokens accepted ({}%))",
            self.id,
            num(completion_tokens),
            num(format_args!("{total:.2}")),
            num(format_args!("{queue:.2}")),
            num(s.cached_prompt_tokens),
            num(s.new_prompt_tokens),
            num(format_args!("{prompt_tps:.2}")),
            num(format_args!("{gen_tps:.2}")),
            num(context),
            num(s.draft_accepted),
            num(s.draft_total),
            num(format_args!("{draft_pct:.2}")),
        );
    }
}

async fn buffered_response(
    rx: flume::Receiver<Chunk>,
    cancel: Arc<AtomicBool>,
    id: String,
    model_id: String,
    is_chat: bool,
    parse_tools: bool,
    prompt_tokens: usize,
    echo_text: String,
    metrics: Metrics,
) -> HttpResponse {
    let _guard = CancelGuard(cancel);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut finish = FinishReason::Stop;
    let mut completion_tokens = 0usize;
    let mut error: Option<String> = None;
    let mut stats = GenStats::default();

    while let Ok(c) = rx.recv_async().await {
        match c {
            Chunk::Content(s) => content.push_str(&s),
            Chunk::Reasoning(s) => reasoning.push_str(&s),
            Chunk::Done { finish: f, completion_tokens: n, stats: gs } => {
                finish = f;
                completion_tokens = n;
                stats = gs;
                break;
            }
            Chunk::Error(e) => {
                error = Some(e);
                break;
            }
        }
    }
    if let Some(e) = error {
        return err_response(500, &e, "server_error");
    }
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    if parse_tools {
        let (calls, cleaned) = extract_tool_calls(&content);
        if !calls.is_empty() {
            metrics.parsed_tools(calls.len());
            tool_calls = calls;
            content = cleaned;
            finish = FinishReason::ToolCalls;
        }
    }
    metrics.finish(Instant::now(), completion_tokens, &stats);

    if is_chat {
        HttpResponse::Ok().json(&oai::chat_completion(
            &id, &model_id, &content, &reasoning, &tool_calls, finish, prompt_tokens,
            completion_tokens,
        ))
    } else {
        let text = format!("{echo_text}{content}");
        HttpResponse::Ok().json(&oai::text_completion(
            &id, &model_id, &text, finish, prompt_tokens, completion_tokens,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
/// Length of `raw` that is safe to stream as content when tool-call parsing is
/// on: everything before the first `<tool_call>`, minus (unless `final_flush`) a
/// trailing partial-`<tool_call>` prefix that a later token might complete.
fn tool_safe_len(raw: &str, final_flush: bool) -> usize {
    const TAG: &str = "<tool_call>";
    if let Some(i) = raw.find(TAG) {
        return i;
    }
    if final_flush {
        return raw.len();
    }
    let b = raw.as_bytes();
    for l in (1..TAG.len().min(b.len() + 1)).rev() {
        if TAG.as_bytes().starts_with(&b[b.len() - l..]) {
            return b.len() - l;
        }
    }
    raw.len()
}

fn stream_response(
    rx: flume::Receiver<Chunk>,
    cancel: Arc<AtomicBool>,
    id: String,
    model_id: String,
    is_chat: bool,
    parse_tools: bool,
    prompt_tokens: usize,
    echo_text: String,
    ping_secs: u64,
    metrics: Metrics,
) -> HttpResponse {
    let s = async_stream::stream! {
        let _guard = CancelGuard(cancel);
        let _ = prompt_tokens;

        // opening role delta (chat only)
        if is_chat {
            let v = oai::chat_chunk(&id, &model_id, json!({ "role": "assistant", "content": "" }), None);
            yield Ok::<Bytes, io::Error>(sse(&v));
        } else if !echo_text.is_empty() {
            let v = oai::text_chunk(&id, &model_id, &echo_text, None);
            yield Ok(sse(&v));
        }

        let mut full_content = String::new();
        // bytes of `full_content` already streamed as a content delta. With
        // `parse_tools`, content from the first `<tool_call>` on is withheld (and
        // a trailing partial-`<tool_call>` prefix) so the raw XML never reaches
        // the client — it comes back as `tool_calls` in the final chunk instead.
        let mut emitted_content: usize = 0;
        let ping = if ping_secs > 0 { Some(Duration::from_secs(ping_secs)) } else { None };

        loop {
            let chunk = match ping {
                Some(d) => {
                    let recv = std::pin::pin!(rx.recv_async());
                    let timer = std::pin::pin!(ntex::time::sleep(d));
                    match futures_util::future::select(recv, timer).await {
                        futures_util::future::Either::Left((Ok(c), _)) => c,
                        futures_util::future::Either::Left((Err(_), _)) => break,
                        futures_util::future::Either::Right(_) => {
                            yield Ok(Bytes::from_static(b": ping\n\n"));
                            continue;
                        }
                    }
                }
                None => match rx.recv_async().await {
                    Ok(c) => c,
                    Err(_) => break,
                },
            };
            match chunk {
                Chunk::Content(t) => {
                    full_content.push_str(&t);
                    let upto = if parse_tools {
                        tool_safe_len(&full_content, false)
                    } else {
                        full_content.len()
                    };
                    if upto > emitted_content {
                        let delta = full_content[emitted_content..upto].to_string();
                        emitted_content = upto;
                        let v = if is_chat {
                            oai::chat_chunk(&id, &model_id, json!({ "content": delta }), None)
                        } else {
                            oai::text_chunk(&id, &model_id, &delta, None)
                        };
                        yield Ok(sse(&v));
                    }
                }
                Chunk::Reasoning(t) => {
                    let v = if is_chat {
                        oai::chat_chunk(&id, &model_id, json!({ "reasoning_content": t }), None)
                    } else {
                        oai::text_chunk(&id, &model_id, &t, None)
                    };
                    yield Ok(sse(&v));
                }
                Chunk::Error(e) => {
                    let v = oai::error_body(&e, "server_error");
                    yield Ok(sse(&v));
                    yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    return;
                }
                Chunk::Done { mut finish, completion_tokens, stats } => {
                    let mut tool_calls: Vec<ToolCall> = Vec::new();
                    if parse_tools {
                        let (calls, cleaned) = extract_tool_calls(&full_content);
                        if !calls.is_empty() {
                            metrics.parsed_tools(calls.len());
                            tool_calls = calls;
                            finish = FinishReason::ToolCalls;
                        }
                        // flush any prose that was held back behind the tool-call
                        // hold-back tail but isn't part of a `<tool_call>` block.
                        if cleaned.len() > emitted_content {
                            let delta = cleaned[emitted_content..].to_string();
                            let v = if is_chat {
                                oai::chat_chunk(&id, &model_id, json!({ "content": delta }), None)
                            } else {
                                oai::text_chunk(&id, &model_id, &delta, None)
                            };
                            yield Ok(sse(&v));
                        }
                    }
                    if is_chat {
                        let delta = if tool_calls.is_empty() {
                            json!({})
                        } else {
                            json!({ "tool_calls": tool_calls })
                        };
                        let v = oai::chat_chunk(&id, &model_id, delta, Some(finish));
                        yield Ok(sse(&v));
                    } else {
                        let v = oai::text_chunk(&id, &model_id, "", Some(finish));
                        yield Ok(sse(&v));
                    }
                    metrics.finish(Instant::now(), completion_tokens, &stats);
                    yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    return;
                }
            }
        }
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    };

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .streaming(Box::pin(s))
}

fn sse(v: &Value) -> Bytes {
    let mut s = String::from("data: ");
    s.push_str(&serde_json::to_string(v).unwrap_or_default());
    s.push_str("\n\n");
    Bytes::from(s)
}

/// SSE frame with an explicit `event:` line, as the Responses API stream uses
/// (Codex pairs the `event:` kind with the `data:` payload).
fn sse_event(kind: &str, v: &Value) -> Bytes {
    let mut s = String::from("event: ");
    s.push_str(kind);
    s.push_str("\ndata: ");
    s.push_str(&serde_json::to_string(v).unwrap_or_default());
    s.push_str("\n\n");
    Bytes::from(s)
}

// --- POST /v1/responses (OpenAI Responses API, for Codex) -----------------

pub async fn responses(
    st: web::types::State<AppState>,
    http_req: HttpRequest,
    raw: Bytes,
) -> HttpResponse {
    if !authorized(&st, &http_req) {
        return err_response(401, "missing or invalid API key", "authentication_error");
    }
    let body = match parse_body(&raw) {
        Ok(b) => b,
        Err(r) => return r,
    };
    if st.cfg.logging.log_requests {
        crate::sinfo!("Request body: {}", serde_json::to_string(&body).unwrap_or_default());
    }
    let chat_body = match responses::to_chat_body(&body) {
        Ok(b) => b,
        Err(e) => return err_response(400, &e, "invalid_request_error"),
    };

    let t_recv = Instant::now();
    let prep = match prepare(&st, &chat_body, true) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let Prepared { req, rx, cancel, stream, prompt_tokens, model_id, parse_tools, .. } = prep;

    if st.eng.tx.send_async(req).await.is_err() {
        return err_response(500, "engine unavailable", "server_error");
    }

    let id = oai::rand_id("resp");
    let log_m = st.cfg.logging.log_chat_completion_requests;
    if log_m {
        crate::sinfo!("Received responses{} request {id}", if stream { " streaming" } else { "" });
    }
    let m = Metrics { id: id.clone(), t_recv, stream, is_chat: true, enabled: log_m };
    let ping = st.cfg.network.sse_ping_interval.unwrap_or(15);

    if stream {
        responses_stream(rx, cancel, id, model_id, parse_tools, prompt_tokens, ping, m)
    } else {
        responses_buffered(rx, cancel, id, model_id, parse_tools, prompt_tokens, m).await
    }
}

async fn responses_buffered(
    rx: flume::Receiver<Chunk>,
    cancel: Arc<AtomicBool>,
    id: String,
    model_id: String,
    parse_tools: bool,
    prompt_tokens: usize,
    metrics: Metrics,
) -> HttpResponse {
    let _guard = CancelGuard(cancel);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut finish = FinishReason::Stop;
    let mut completion_tokens = 0usize;
    let mut stats = GenStats::default();
    let mut error: Option<String> = None;

    while let Ok(c) = rx.recv_async().await {
        match c {
            Chunk::Content(s) => content.push_str(&s),
            Chunk::Reasoning(s) => reasoning.push_str(&s),
            Chunk::Done { finish: f, completion_tokens: n, stats: gs } => {
                finish = f;
                completion_tokens = n;
                stats = gs;
                break;
            }
            Chunk::Error(e) => {
                error = Some(e);
                break;
            }
        }
    }
    if let Some(e) = error {
        return HttpResponse::Ok().json(&responses::response_failed(&id, &model_id, &e));
    }

    let mut tool_calls = Vec::new();
    if parse_tools {
        let (calls, cleaned) = extract_tool_calls(&content);
        if !calls.is_empty() {
            metrics.parsed_tools(calls.len());
            tool_calls = calls;
            content = cleaned;
            finish = FinishReason::ToolCalls;
        }
    }
    metrics.finish(Instant::now(), completion_tokens, &stats);

    HttpResponse::Ok().json(&responses::response_object(
        &id, &model_id, &content, &reasoning, &tool_calls, finish, prompt_tokens, completion_tokens,
    ))
}

#[allow(clippy::too_many_arguments)]
fn responses_stream(
    rx: flume::Receiver<Chunk>,
    cancel: Arc<AtomicBool>,
    id: String,
    model_id: String,
    parse_tools: bool,
    prompt_tokens: usize,
    ping_secs: u64,
    metrics: Metrics,
) -> HttpResponse {
    let s = async_stream::stream! {
        let _guard = CancelGuard(cancel);
        let msg_id = format!("msg_{id}");
        let rs_id = format!("rs_{id}");
        let mut seq: u64 = 0;
        let mut ev = |kind: &str, mut v: Value| {
            // Codex deserialises the `data:` JSON into a struct keyed by `type`;
            // a missing `type` is a hard parse error ("stream disconnected
            // before completion"). Keep it in the body as well as the `event:`
            // line.
            v["type"] = json!(kind);
            v["sequence_number"] = json!(seq);
            seq += 1;
            sse_event(kind, &v)
        };

        yield Ok::<Bytes, io::Error>(ev(
            "response.created",
            json!({ "response": responses::response_skeleton(&id, &model_id) }),
        ));

        let mut full_content = String::new();
        let mut emitted: usize = 0;
        let mut reasoning_acc = String::new();
        let mut msg_added = false;
        let mut reason_added = false;
        let mut reason_done = false;
        let mut next_idx: u64 = 0;
        let mut reason_idx: u64 = 0;
        let mut msg_idx: u64 = 0;

        let ping = if ping_secs > 0 { Some(Duration::from_secs(ping_secs)) } else { None };

        loop {
            let chunk = match ping {
                Some(d) => {
                    let recv = std::pin::pin!(rx.recv_async());
                    let timer = std::pin::pin!(ntex::time::sleep(d));
                    match futures_util::future::select(recv, timer).await {
                        futures_util::future::Either::Left((Ok(c), _)) => c,
                        futures_util::future::Either::Left((Err(_), _)) => break,
                        futures_util::future::Either::Right(_) => {
                            yield Ok(Bytes::from_static(b": ping\n\n"));
                            continue;
                        }
                    }
                }
                None => match rx.recv_async().await {
                    Ok(c) => c,
                    Err(_) => break,
                },
            };
            match chunk {
                Chunk::Reasoning(t) => {
                    reasoning_acc.push_str(&t);
                    if !reason_added {
                        reason_added = true;
                        reason_idx = next_idx;
                        next_idx += 1;
                        yield Ok(ev("response.output_item.added", json!({
                            "output_index": reason_idx,
                            "item": { "type": "reasoning", "id": rs_id, "summary": [], "content": [] },
                        })));
                    }
                    yield Ok(ev("response.reasoning_text.delta", json!({
                        "item_id": rs_id, "output_index": reason_idx, "content_index": 0, "delta": t,
                    })));
                }
                Chunk::Content(t) => {
                    full_content.push_str(&t);
                    let upto = if parse_tools {
                        tool_safe_len(&full_content, false)
                    } else {
                        full_content.len()
                    };
                    if upto > emitted {
                        if !msg_added {
                            // close the reasoning item before the message opens
                            if reason_added && !reason_done {
                                reason_done = true;
                                yield Ok(ev("response.output_item.done", json!({
                                    "output_index": reason_idx,
                                    "item": {
                                        "type": "reasoning", "id": rs_id, "summary": [],
                                        "content": [{ "type": "reasoning_text", "text": reasoning_acc }],
                                    },
                                })));
                            }
                            msg_added = true;
                            msg_idx = next_idx;
                            next_idx += 1;
                            yield Ok(ev("response.output_item.added", json!({
                                "output_index": msg_idx,
                                "item": {
                                    "type": "message", "id": msg_id, "role": "assistant",
                                    "status": "in_progress", "content": [],
                                },
                            })));
                        }
                        let delta = full_content[emitted..upto].to_string();
                        emitted = upto;
                        yield Ok(ev("response.output_text.delta", json!({
                            "item_id": msg_id, "output_index": msg_idx, "content_index": 0, "delta": delta,
                        })));
                    }
                }
                Chunk::Error(e) => {
                    yield Ok(ev("response.failed", json!({
                        "response": responses::response_failed(&id, &model_id, &e),
                    })));
                    return;
                }
                Chunk::Done { mut finish, completion_tokens, stats } => {
                    let mut tool_calls: Vec<ToolCall> = Vec::new();
                    let mut final_text = full_content.clone();
                    if parse_tools {
                        let (calls, cleaned) = extract_tool_calls(&full_content);
                        if !calls.is_empty() {
                            metrics.parsed_tools(calls.len());
                            tool_calls = calls;
                            finish = FinishReason::ToolCalls;
                            final_text = cleaned;
                        }
                    }
                    // flush any held-back prose before closing the message item
                    if msg_added && final_text.len() > emitted {
                        let delta = final_text[emitted..].to_string();
                        yield Ok(ev("response.output_text.delta", json!({
                            "item_id": msg_id, "output_index": msg_idx, "content_index": 0, "delta": delta,
                        })));
                    }
                    if reason_added && !reason_done {
                        yield Ok(ev("response.output_item.done", json!({
                            "output_index": reason_idx,
                            "item": {
                                "type": "reasoning", "id": rs_id, "summary": [],
                                "content": [{ "type": "reasoning_text", "text": reasoning_acc }],
                            },
                        })));
                    }
                    if msg_added {
                        yield Ok(ev("response.output_item.done", json!({
                            "output_index": msg_idx,
                            "item": {
                                "type": "message", "id": msg_id, "role": "assistant", "status": "completed",
                                "content": [{ "type": "output_text", "text": final_text, "annotations": [] }],
                            },
                        })));
                    }
                    for (i, tc) in tool_calls.iter().enumerate() {
                        let idx = next_idx;
                        next_idx += 1;
                        let item = json!({
                            "type": "function_call",
                            "id": format!("fc_{id}_{i}"),
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                            "status": "completed",
                        });
                        yield Ok(ev("response.output_item.added",
                            json!({ "output_index": idx, "item": item })));
                        yield Ok(ev("response.output_item.done",
                            json!({ "output_index": idx, "item": item })));
                    }
                    metrics.finish(Instant::now(), completion_tokens, &stats);
                    yield Ok(ev("response.completed", json!({
                        "response": responses::response_object(
                            &id, &model_id, &final_text, &reasoning_acc, &tool_calls,
                            finish, prompt_tokens, completion_tokens,
                        ),
                    })));
                    return;
                }
            }
        }
        // stream ended without a Done — emit a best-effort completion
        yield Ok(ev("response.completed", json!({
            "response": responses::response_object(
                &id, &model_id, &full_content, &reasoning_acc, &[],
                FinishReason::Stop, prompt_tokens, 0,
            ),
        })));
    };

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .streaming(Box::pin(s))
}

// --- access log middleware ---------------------------------------------------

/// TabbyAPI-style one-line access log: `127.0.0.1:59228 - "POST /v1/… HTTP/1.1" 200`.
pub struct AccessLog;

impl<S> ntex::service::Middleware<S> for AccessLog {
    type Service = AccessLogMw<S>;
    fn create(&self, service: S) -> Self::Service {
        AccessLogMw { service }
    }
}

pub struct AccessLogMw<S> {
    service: S,
}

impl<S, E> ntex::service::Service<web::WebRequest<E>> for AccessLogMw<S>
where
    S: ntex::service::Service<web::WebRequest<E>, Response = web::WebResponse>,
{
    type Response = web::WebResponse;
    type Error = S::Error;

    ntex::forward_poll!(service);
    ntex::forward_ready!(service);
    ntex::forward_shutdown!(service);

    async fn call(
        &self,
        req: web::WebRequest<E>,
        ctx: ntex::service::ServiceCtx<'_, Self>,
    ) -> Result<Self::Response, Self::Error> {
        let peer = req
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "-".into());
        let line = format!(
            "{} {} {:?}",
            req.method(),
            req.uri()
                .path_and_query()
                .map(|p| p.as_str())
                .unwrap_or("/"),
            req.version(),
        );
        let res = ctx.call(&self.service, req).await?;
        let code = res.status().as_u16();
        use crate::server::log::{paint, status};
        crate::sinfo!(
            "{} - {} {}",
            paint("1;30", &peer),
            status(code, format_args!("{:?}", line)),
            status(code, code),
        );
        Ok(res)
    }
}
