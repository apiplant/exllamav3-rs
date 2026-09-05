//! OpenAI **Responses API** (`POST /v1/responses`) translation layer.
//!
//! Codex (`codex --profile arch`, `wire_api = "responses"`) speaks this protocol
//! and nothing else. Rather than re-implement prompting, this module rewrites a
//! Responses request into the chat-completions `Value` shape that
//! [`crate::server::http::prepare`] already consumes, and rewrites engine output
//! back into the `response.*` SSE event stream / `response` object Codex expects.
//!
//! Scope: text, reasoning, function calls, image input, `text.format` JSON
//! schema — streaming and buffered. Not covered: hosted tools (`web_search`
//! etc.), `previous_response_id` / `store` conversation state.

use crate::server::chat::ToolCall;
use crate::server::oai::{now, FinishReason};
use serde_json::{json, Map, Value};

// --- request: Responses -> chat-completions body ---------------------------

/// Rewrite a Responses request body into a chat-completions body understood by
/// [`crate::server::http::prepare`] (called with `is_chat = true`).
pub fn to_chat_body(body: &Value) -> Result<Value, String> {
    let mut messages: Vec<Value> = Vec::new();

    // `instructions` -> leading system message (render_chatml only honours a
    // system message at index 0).
    if let Some(instr) = body.get("instructions").and_then(Value::as_str) {
        if !instr.trim().is_empty() {
            messages.push(json!({ "role": "system", "content": instr }));
        }
    }

    match body.get("input") {
        Some(Value::String(s)) => messages.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => translate_input_items(items, &mut messages)?,
        Some(Value::Null) | None => {}
        Some(other) => return Err(format!("`input` must be a string or array, got {other}")),
    }

    if messages.iter().all(|m| m.get("role").and_then(Value::as_str) == Some("system")) {
        return Err("`input` produced no user/assistant/tool messages".into());
    }

    let mut out = Map::new();
    out.insert("messages".into(), Value::Array(messages));
    out.insert(
        "stream".into(),
        json!(body.get("stream").and_then(Value::as_bool).unwrap_or(false)),
    );

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let nested: Vec<Value> = tools.iter().filter_map(nest_tool).collect();
        if !nested.is_empty() {
            out.insert("tools".into(), Value::Array(nested));
        }
    }
    if let Some(tc) = body.get("tool_choice") {
        out.insert("tool_choice".into(), tc.clone());
    }

    // param mapping
    if let Some(v) = body.get("max_output_tokens").cloned() {
        out.insert("max_tokens".into(), v);
    }
    if let Some(eff) = body.get("reasoning").and_then(|r| r.get("effort")).cloned() {
        out.insert("reasoning_effort".into(), eff);
    }
    for k in ["temperature", "top_p", "top_k", "min_p", "stop", "seed", "frequency_penalty", "presence_penalty"] {
        if let Some(v) = body.get(k).cloned() {
            out.insert(k.into(), v);
        }
    }

    // `text.format` -> `response_format` (the shape parse_grammar already reads)
    if let Some(fmt) = body.get("text").and_then(|t| t.get("format")) {
        match fmt.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                out.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": { "schema": fmt.get("schema").cloned().unwrap_or(Value::Null) }
                    }),
                );
            }
            Some("json_object") => {
                out.insert("response_format".into(), json!({ "type": "json_object" }));
            }
            _ => {}
        }
    }

    Ok(Value::Object(out))
}

/// Map the `input` array to chat messages. `function_call` items coalesce onto a
/// synthetic `assistant` message; `reasoning` items fold into the preceding
/// assistant turn's `reasoning_content`.
fn translate_input_items(items: &[Value], messages: &mut Vec<Value>) -> Result<(), String> {
    for item in items {
        // A bare `{role, content}` with no `type` is a message (Codex sends both).
        let ty = item.get("type").and_then(Value::as_str).unwrap_or("message");
        match ty {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let content = normalize_content(item.get("content").unwrap_or(&Value::Null));
                messages.push(json!({ "role": role, "content": content }));
            }
            "function_call" => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                let args = match item.get("arguments") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => "{}".into(),
                };
                let call_id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or("call_0")
                    .to_string();
                let tc = json!({
                    "id": call_id,
                    "type": "function",
                    "function": { "name": name, "arguments": args }
                });
                match messages.last_mut().filter(|m| {
                    m.get("role").and_then(Value::as_str) == Some("assistant")
                }) {
                    Some(Value::Object(m)) => {
                        m.entry("tool_calls")
                            .or_insert_with(|| Value::Array(Vec::new()))
                            .as_array_mut()
                            .unwrap()
                            .push(tc);
                    }
                    _ => messages.push(json!({
                        "role": "assistant", "content": "", "tool_calls": [tc]
                    })),
                }
            }
            "function_call_output" => {
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("call_0");
                let out = match item.get("output") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                messages.push(json!({
                    "role": "tool", "tool_call_id": call_id, "content": out
                }));
            }
            "reasoning" => {
                let text = collect_text(item.get("summary"))
                    .or_else(|| collect_text(item.get("content")))
                    .unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                if let Some(Value::Object(m)) = messages.last_mut().filter(|m| {
                    m.get("role").and_then(Value::as_str) == Some("assistant")
                }) {
                    m.insert("reasoning_content".into(), json!(text));
                }
            }
            other => return Err(format!("unsupported input item type `{other}`")),
        }
    }
    Ok(())
}

/// A Responses content value -> the `String | [{type:text|image_url}]` shape
/// `render_chatml` / `content_to_text` accept.
fn normalize_content(content: &Value) -> Value {
    match content {
        Value::String(s) => Value::String(s.clone()),
        Value::Array(parts) => {
            let mapped: Vec<Value> = parts
                .iter()
                .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") | Some("text") => Some(json!({
                        "type": "text",
                        "text": p.get("text").and_then(Value::as_str).unwrap_or_default()
                    })),
                    Some("input_image") | Some("image_url") => {
                        let url = match p.get("image_url") {
                            Some(Value::String(u)) => Some(u.clone()),
                            Some(Value::Object(o)) => {
                                o.get("url").and_then(Value::as_str).map(String::from)
                            }
                            _ => None,
                        };
                        url.map(|u| json!({ "type": "image_url", "image_url": { "url": u } }))
                    }
                    _ => None,
                })
                .collect();
            Value::Array(mapped)
        }
        Value::Null => Value::String(String::new()),
        other => Value::String(other.to_string()),
    }
}

/// Concatenate `text` fields from a Responses `summary` / `content` part array.
fn collect_text(v: Option<&Value>) -> Option<String> {
    let arr = v?.as_array()?;
    let joined = arr
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");
    (!joined.is_empty()).then_some(joined)
}

/// Responses flat function tool `{type:function, name, ...}` -> chat nested
/// `{type:function, function:{...}}`. Hosted tools (`web_search`, …) are dropped.
fn nest_tool(t: &Value) -> Option<Value> {
    match t.get("type").and_then(Value::as_str) {
        Some("function") | None => {
            if t.get("function").is_some() {
                return Some(t.clone()); // already nested
            }
            let mut f = Map::new();
            for k in ["name", "description", "parameters", "strict"] {
                if let Some(v) = t.get(k) {
                    f.insert(k.into(), v.clone());
                }
            }
            Some(json!({ "type": "function", "function": Value::Object(f) }))
        }
        _ => None,
    }
}

// --- response: internal -> Responses object -------------------------------

fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": format!("rs_{id}"),
        "summary": [],
        "content": [{ "type": "reasoning_text", "text": text }],
    })
}

fn message_item(id: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "id": format!("msg_{id}"),
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "text": text, "annotations": [] }],
    })
}

fn function_call_item(id: &str, idx: usize, tc: &ToolCall) -> Value {
    json!({
        "type": "function_call",
        "id": format!("fc_{id}_{idx}"),
        "call_id": tc.id,
        "name": tc.function.name,
        "arguments": tc.function.arguments,
        "status": "completed",
    })
}

/// Build the full `response` object (also used as the `response.completed`
/// payload).
#[allow(clippy::too_many_arguments)]
pub fn response_object(
    id: &str,
    model: &str,
    content: &str,
    reasoning: &str,
    tool_calls: &[ToolCall],
    finish: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    let mut output = Vec::new();
    if !reasoning.is_empty() {
        output.push(reasoning_item(id, reasoning));
    }
    if !content.is_empty() {
        output.push(message_item(id, content));
    }
    for (i, tc) in tool_calls.iter().enumerate() {
        output.push(function_call_item(id, i, tc));
    }

    let (status, incomplete) = match finish {
        FinishReason::Length => (
            "incomplete",
            json!({ "reason": "max_output_tokens" }),
        ),
        _ => ("completed", Value::Null),
    };

    json!({
        "id": id,
        "object": "response",
        "created_at": now(),
        "model": model,
        "status": status,
        "output": output,
        "usage": usage(prompt_tokens, completion_tokens),
        "incomplete_details": incomplete,
        "error": Value::Null,
        "parallel_tool_calls": true,
    })
}

/// Skeleton `response` for the `response.created` event.
pub fn response_skeleton(id: &str, model: &str) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now(),
        "model": model,
        "status": "in_progress",
        "output": [],
        "error": Value::Null,
    })
}

/// `response.failed` payload.
pub fn response_failed(id: &str, model: &str, message: &str) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now(),
        "model": model,
        "status": "failed",
        "output": [],
        "error": { "code": "server_error", "message": message },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instructions_become_system() {
        let b = json!({ "instructions": "be terse", "input": "hi" });
        let c = to_chat_body(&b).unwrap();
        let msgs = c["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be terse");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hi");
    }

    #[test]
    fn function_call_roundtrip() {
        let b = json!({
            "input": [
                { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "weather?" }] },
                { "type": "function_call", "name": "get_weather", "arguments": "{\"city\":\"Rome\"}", "call_id": "c1" },
                { "type": "function_call_output", "call_id": "c1", "output": "{\"t\":19}" }
            ]
        });
        let c = to_chat_body(&b).unwrap();
        let msgs = c["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"][0]["type"], "text");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "c1");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "c1");
    }

    #[test]
    fn tools_get_nested() {
        let b = json!({
            "input": "hi",
            "tools": [
                { "type": "function", "name": "f", "description": "d", "parameters": { "type": "object" } },
                { "type": "web_search" }
            ]
        });
        let c = to_chat_body(&b).unwrap();
        let tools = c["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "f");
    }

    #[test]
    fn reasoning_effort_and_max_tokens_map() {
        let b = json!({
            "input": "hi",
            "max_output_tokens": 64,
            "reasoning": { "effort": "low" }
        });
        let c = to_chat_body(&b).unwrap();
        assert_eq!(c["max_tokens"], 64);
        assert_eq!(c["reasoning_effort"], "low");
    }

    #[test]
    fn text_format_becomes_response_format() {
        let b = json!({
            "input": "hi",
            "text": { "format": { "type": "json_schema", "schema": { "type": "object" } } }
        });
        let c = to_chat_body(&b).unwrap();
        assert_eq!(c["response_format"]["type"], "json_schema");
        assert_eq!(c["response_format"]["json_schema"]["schema"]["type"], "object");
    }

    #[test]
    fn empty_input_is_rejected() {
        assert!(to_chat_body(&json!({ "instructions": "x" })).is_err());
    }
}
