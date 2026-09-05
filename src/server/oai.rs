//! OpenAI-compatible request/response shapes and the mapping to internal
//! generation parameters.

use crate::sampler::SamplerSettings;
use crate::server::config::SamplerDefaults;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
}

pub fn rand_id(prefix: &str) -> String {
    // good enough: time + a process-unique counter
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let n = C.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    format!("{prefix}-{t:x}{n:x}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    Cancelled,
    Error,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::Cancelled => "stop",
            FinishReason::Error => "error",
        }
    }
    pub fn from_eos_reason(r: &str) -> Self {
        match r {
            "max_new_tokens" => FinishReason::Length,
            "cancelled" => FinishReason::Cancelled,
            _ => FinishReason::Stop,
        }
    }
}

/// Sampling / limits parsed from a request, before tokenisation.
#[derive(Debug, Clone)]
pub struct SampleParams {
    pub max_new: usize,
    pub min_new: usize,
    pub sampler: SamplerSettings,
    pub stop_strings: Vec<String>,
    pub seed: Option<i64>,
    pub stream: bool,
    pub n: usize,
}

fn f64_at(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}
fn i64_at(v: &Value, key: &str) -> Option<i64> {
    v.get(key).and_then(Value::as_i64)
}

/// Parse the shared sampling block. `ctx_room` caps `max_tokens` to the space
/// left in the context window.
pub fn parse_sample_params(body: &Value, d: &SamplerDefaults, ctx_room: usize) -> SampleParams {
    let temperature = f64_at(body, "temperature").unwrap_or(d.temperature).max(0.0);
    let top_p = f64_at(body, "top_p").unwrap_or(d.top_p);
    let top_k = i64_at(body, "top_k").unwrap_or(d.top_k);
    let min_p = f64_at(body, "min_p").unwrap_or(d.min_p);
    // OpenAI presence/frequency penalties are additive in [-2, 2].
    let pres = f64_at(body, "presence_penalty").unwrap_or(d.presence_penalty);
    let freq = f64_at(body, "frequency_penalty").unwrap_or(d.frequency_penalty);
    // repetition_penalty (multiplicative) — non-OpenAI but widely sent.
    let rep = f64_at(body, "repetition_penalty").unwrap_or(d.repetition_penalty);

    let sampler = SamplerSettings {
        temperature,
        top_k,
        top_p,
        min_p,
        rep_penalty: rep,
        pres_penalty: pres,
        freq_penalty: freq,
        sustain_range: i64_at(body, "penalty_range").unwrap_or(0),
        decay_range: 0,
    };

    let want = i64_at(body, "max_tokens")
        .or_else(|| i64_at(body, "max_completion_tokens"))
        .unwrap_or(512)
        .max(1) as usize;
    let max_new = want.min(ctx_room.max(1));

    let mut stop_strings = Vec::new();
    match body.get("stop") {
        Some(Value::String(s)) => stop_strings.push(s.clone()),
        Some(Value::Array(a)) => {
            for s in a {
                if let Some(s) = s.as_str() {
                    stop_strings.push(s.to_string());
                }
            }
        }
        _ => {}
    }

    SampleParams {
        max_new,
        min_new: i64_at(body, "min_tokens").unwrap_or(0).max(0) as usize,
        sampler,
        stop_strings,
        seed: i64_at(body, "seed"),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        n: i64_at(body, "n").unwrap_or(1).max(1) as usize,
    }
}

// --- response builders ------------------------------------------------------

pub fn chat_completion(
    id: &str,
    model: &str,
    content: &str,
    reasoning: &str,
    tool_calls: &[crate::server::chat::ToolCall],
    finish: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    let mut msg = json!({ "role": "assistant", "content": if content.is_empty() && !tool_calls.is_empty() { Value::Null } else { Value::String(content.to_string()) } });
    if !reasoning.is_empty() {
        msg["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        msg["tool_calls"] = serde_json::to_value(tool_calls).unwrap();
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish.as_str(),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

pub fn chat_chunk(id: &str, model: &str, delta: Value, finish: Option<FinishReason>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish.map(|f| f.as_str()),
        }]
    })
}

pub fn text_completion(
    id: &str,
    model: &str,
    text: &str,
    finish: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": now(),
        "model": model,
        "choices": [{ "index": 0, "text": text, "finish_reason": finish.as_str(), "logprobs": Value::Null }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

pub fn text_chunk(id: &str, model: &str, text: &str, finish: Option<FinishReason>) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": now(),
        "model": model,
        "choices": [{ "index": 0, "text": text, "finish_reason": finish.map(|f| f.as_str()), "logprobs": Value::Null }]
    })
}

pub fn error_body(msg: &str, kind: &str) -> Value {
    json!({ "error": { "message": msg, "type": kind, "code": kind } })
}
