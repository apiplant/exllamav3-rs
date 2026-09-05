//! Minimal ChatML prompt rendering + reasoning / tool-call extraction.
//!
//! This port has no Jinja engine, so `chat_template.jinja` is not executed —
//! `render_chatml` transcribes it by hand for the Qwen2/Qwen3/Qwen3.5 family (the
//! models this crate supports). `prompt_template` selection of a *named* template
//! is honored only in that a non-default name is logged and ignored.
//!
//! **Transcription drift is a correctness bug, not a cosmetic one.** A prompt that
//! differs from the trained format puts the model out of distribution and it
//! starts leaking tool-call syntax (its own, or a foreign dialect) into content.
//! `tests/chat_template/check.py` renders fixtures through both this code and the
//! real Jinja and requires them byte-identical; run it after touching this file.

use serde_json::Value;

/// One chat message flattened to text + any image markers.
pub struct RenderedPrompt {
    pub text: String,
    /// data-URI / URL / path for each `<image>` marker, in order.
    pub images: Vec<String>,
    /// whether the templated prompt ends inside an unclosed reasoning block.
    pub starts_in_reasoning: bool,
}

fn content_to_text(content: &Value, images: &mut Vec<String>) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            out.push_str(t);
                        }
                    }
                    Some("image_url") => {
                        if let Some(u) = p
                            .get("image_url")
                            .and_then(|iu| iu.get("url"))
                            .and_then(Value::as_str)
                        {
                            images.push(u.to_string());
                            out.push_str("<image>");
                        }
                    }
                    _ => {}
                }
            }
            out
        }
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Serialise like Jinja's `tojson`, which is Python `json.dumps` with its default
/// `(", ", ": ")` separators — NOT serde's compact form. The tool schemas are the
/// largest part of an agentic prompt, so the difference re-tokenises hundreds of
/// tokens per request and is exactly the kind of drift that costs format fidelity.
fn to_json_jinja(v: &Value) -> String {
    struct F;
    impl serde_json::ser::Formatter for F {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, F);
    match serde::Serialize::serialize(v, &mut ser) {
        Ok(()) => String::from_utf8(buf).unwrap_or_default(),
        Err(_) => serde_json::to_string(v).unwrap_or_default(),
    }
}

/// Build the tool preamble, verbatim from the Qwen3.5 / Qwen3-Coder
/// `chat_template.jinja` tool branch.
///
/// This model is trained to emit tool calls as **pseudo-XML**
/// (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`),
/// not as the Hermes `{"name":…,"arguments":…}` JSON this used to ask for. Asking
/// for JSON put the model out of distribution: it complied some of the time and
/// fell back to its trained XML the rest, which the old JSON-only parser dropped
/// on the floor — the intermittent empty `{}` / "Tool not found" that the Python
/// port (which renders the model's own template) never produced.
fn tools_preamble(tools: &[Value]) -> String {
    let mut s = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for t in tools {
        // OpenAI shape: {"type":"function","function":{...}} — the template emits
        // the whole tool object, not just the inner `function`.
        s.push('\n');
        s.push_str(&to_json_jinja(t));
    }
    s.push_str(
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with \
         NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
         <parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
         <parameter=example_parameter_2>\nThis is the value for the second parameter\n\
         that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n\
         <IMPORTANT>\nReminder:\n\
         - Function calls MUST follow the specified format: an inner <function=...></function> \
         block must be nested within <tool_call></tool_call> XML tags\n\
         - Required parameters MUST be specified\n\
         - You may provide optional reasoning for your function call in natural language BEFORE \
         the function call, but NOT after\n\
         - If there is no function call available, answer the question like normal with your \
         current knowledge and do not tell the user about function calls\n</IMPORTANT>",
    );
    s
}

/// Render one assistant `tool_calls` entry the way the model's template does, so
/// a multi-turn tool conversation is replayed in the same form the model emits.
fn render_tool_call(f: &Value, out: &mut String) {
    let name = f.get("name").and_then(Value::as_str).unwrap_or("");
    out.push_str("<tool_call>\n<function=");
    out.push_str(name);
    out.push_str(">\n");
    // `arguments` is a JSON *string* on the wire; the template iterates its keys.
    let args: Option<Value> = f
        .get("arguments")
        .and_then(|a| match a {
            Value::String(s) => serde_json::from_str(s).ok(),
            other => Some(other.clone()),
        });
    if let Some(Value::Object(map)) = args {
        for (k, v) in map {
            out.push_str("<parameter=");
            out.push_str(&k);
            out.push_str(">\n");
            match v {
                Value::String(s) => out.push_str(&s),
                other => out.push_str(&serde_json::to_string(&other).unwrap_or_default()),
            }
            out.push_str("\n</parameter>\n");
        }
    }
    out.push_str("</function>\n</tool_call>");
}

/// The template's `reasoning_instructions`: a system-prompt prefix selected by
/// `reasoning_effort`, emitted whenever thinking is not explicitly disabled.
///
/// Note `medium` deliberately yields nothing — only `xhigh` (the template's
/// default when the client says nothing) and `low` carry text.
fn reasoning_instructions(enable_thinking: Option<bool>, effort: Option<&str>) -> &'static str {
    if enable_thinking == Some(false) {
        return "";
    }
    match effort.unwrap_or("xhigh") {
        "xhigh" => "Reasoning effort is set to xhigh. Please think carefully through the task, \
                    validate key assumptions, consider plausible alternatives, and prioritize \
                    correctness, consistency, and clarity in the final answer.",
        "low" => "Reasoning effort is set to low. Keep your thinking brief and focused, moving \
                  directly to the conclusion without unnecessary elaboration.",
        // `medium` -> no instructions. An unrecognised effort raises in the
        // template; here we fall back to no instructions rather than 500.
        _ => "",
    }
}

/// Render an OpenAI `messages` array to a ChatML prompt string, following the
/// model's own `chat_template.jinja` (there is no Jinja engine in this port, so
/// the template is transcribed by hand — keep the two in sync).
///
/// `enable_thinking == Some(false)` appends an empty `<think></think>` block so a
/// reasoning model answers directly. Anything else — including `None`, the common
/// case where the client sends nothing — prefills `<think>\n`, because the model
/// is trained so that EVERY assistant turn opens with it.
pub fn render_chatml(
    messages: &[Value],
    tools: Option<&[Value]>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<&str>,
    reasoning_end_token: &str,
) -> RenderedPrompt {
    let mut images = Vec::new();
    let mut out = String::new();

    let ri = reasoning_instructions(enable_thinking, reasoning_effort);

    // The template only honours a system message at index 0.
    let sys0 = messages
        .first()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        .map(|m| content_to_text(m.get("content").unwrap_or(&Value::Null), &mut images))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        out.push_str("<|im_start|>system\n");
        if !ri.is_empty() {
            out.push_str(ri);
            out.push_str("\n\n");
        }
        out.push_str(&tools_preamble(tools));
        if let Some(s) = &sys0 {
            out.push_str("\n\n");
            out.push_str(s);
        }
        out.push_str("<|im_end|>\n");
    } else if sys0.is_some() || !ri.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !ri.is_empty() {
            out.push_str(ri);
            if sys0.is_some() {
                out.push_str("\n\n");
            }
        }
        if let Some(s) = &sys0 {
            out.push_str(s);
        }
        out.push_str("<|im_end|>\n");
    }

    for (i, m) in messages.iter().enumerate() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        if role == "system" {
            continue;
        }
        let content = content_to_text(m.get("content").unwrap_or(&Value::Null), &mut images);
        let content = content.trim();
        match role {
            "tool" => {
                // Consecutive tool messages share ONE user turn, each as its own
                // `<tool_response>` block; `<|im_end|>` closes the whole run.
                let prev_is_tool = i > 0
                    && messages[i - 1].get("role").and_then(Value::as_str) == Some("tool");
                let next_is_tool = messages
                    .get(i + 1)
                    .and_then(|n| n.get("role").and_then(Value::as_str))
                    == Some("tool");
                if !prev_is_tool {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if !next_is_tool {
                    out.push_str("<|im_end|>\n");
                }
            }
            "assistant" => {
                // The template renders a `<think>` block on EVERY assistant turn
                // (`preserve_thinking` defaults to true), filled from
                // `reasoning_content` when the client round-trips it. Omitting it
                // is what leaves the model unsure where it is in the format, and
                // is a prime source of leaked/stray tool-call tags.
                out.push_str("<|im_start|>assistant\n<think>\n");
                let reasoning = m
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                out.push_str(reasoning);
                out.push_str("\n</think>\n\n");
                out.push_str(content);
                // replay any tool calls the client sent back, in the model's own
                // pseudo-XML form (the template separates the first block from
                // non-empty content by a blank line, later ones by a newline)
                if let Some(tcs) = m.get("tool_calls").and_then(Value::as_array) {
                    for (j, tc) in tcs.iter().enumerate() {
                        let f = tc.get("function").unwrap_or(tc);
                        if j == 0 {
                            if !content.is_empty() {
                                out.push_str("\n\n");
                            }
                        } else {
                            out.push('\n');
                        }
                        render_tool_call(f, &mut out);
                    }
                }
                out.push_str("<|im_end|>\n");
            }
            _ => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    out.push_str("<|im_start|>assistant\n");
    let starts_in_reasoning = enable_thinking != Some(false);
    if starts_in_reasoning {
        out.push_str("<think>\n");
    } else {
        out.push_str("<think>\n\n</think>\n\n");
    }
    let _ = reasoning_end_token;
    RenderedPrompt { text: out, images, starts_in_reasoning }
}

/// Streaming-friendly reasoning splitter: fed the *entire* raw output so far, it
/// re-derives the `(reasoning, content)` partition and the caller emits only the
/// new suffix of each. Cheap because completions are short.
pub fn split_reasoning(
    raw: &str,
    start_tok: &str,
    end_tok: &str,
    start_in_reasoning: bool,
) -> (String, String) {
    let in_reasoning = start_in_reasoning
        || raw.trim_start().starts_with(start_tok);

    if !in_reasoning {
        return (String::new(), raw.to_string());
    }
    let body = raw.trim_start().strip_prefix(start_tok).unwrap_or(raw);
    match body.split_once(end_tok) {
        Some((r, c)) => (
            r.trim_matches(['\n', ' ']).to_string(),
            c.trim_start_matches('\n').to_string(),
        ),
        None => (body.trim_start_matches('\n').to_string(), String::new()),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFn,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolCallFn {
    pub name: String,
    pub arguments: String,
}

/// Extract `<tool_call>{...}</tool_call>` blocks (Qwen / Hermes style). Returns
/// the calls and `content` with the blocks removed.
/// Find `<tag=NAME>` … `</tag>` blocks, returning `(name, inner)` pairs. The
/// model's format nests `<function=…>` inside `<tool_call>` and `<parameter=…>`
/// inside that; neither ever nests within itself, so a flat scan is enough.
fn xml_blocks<'a>(s: &'a str, tag: &str) -> Vec<(&'a str, &'a str)> {
    let open = format!("<{tag}=");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(i) = rest.find(&open) {
        let after = &rest[i + open.len()..];
        let Some(gt) = after.find('>') else { break };
        let name = after[..gt].trim();
        let body = &after[gt + 1..];
        let Some(e) = body.find(&close) else { break };
        out.push((name, &body[..e]));
        rest = &body[e + close.len()..];
    }
    out
}

/// Coerce a `<parameter=…>` body to JSON: the template writes objects, arrays,
/// numbers and booleans as-is and strings raw, so try JSON first and fall back to
/// a JSON string. (Mirrors upstream's `coerce_param_value`.)
fn coerce_param(raw: &str) -> Value {
    let t = raw.trim();
    if t.is_empty() {
        return Value::String(String::new());
    }
    serde_json::from_str::<Value>(t).unwrap_or_else(|_| Value::String(t.to_string()))
}

/// Parse one `<tool_call>` body into `(name, arguments_json)` pairs. Accepts the
/// Qwen3.5 / Qwen3-Coder pseudo-XML the model is trained to emit, and falls back
/// to the Hermes JSON object (including the OpenAI-nested `{"function": {...}}`
/// shape) so a model that emits either is handled. A call with no name is never
/// produced — the caller keeps the block as text instead.
fn parse_call_body(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (fname, fbody) in xml_blocks(body, "function") {
        if fname.is_empty() {
            continue;
        }
        let mut args = serde_json::Map::new();
        for (pname, pbody) in xml_blocks(fbody, "parameter") {
            if !pname.is_empty() {
                args.insert(pname.to_string(), coerce_param(pbody));
            }
        }
        out.push((
            fname.to_string(),
            serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
        ));
    }
    if !out.is_empty() {
        return out;
    }
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let v = v.get("function").unwrap_or(&v);
        if let Some(name) = v.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) {
            let args = match v.get("arguments") {
                Some(Value::String(s)) => s.clone(),
                Some(a) => serde_json::to_string(a).unwrap_or_else(|_| "{}".into()),
                None => "{}".into(),
            };
            out.push((name.to_string(), args));
        }
    }
    out
}

pub fn extract_tool_calls(content: &str) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = content;
    while let Some(open) = rest.find("<tool_call>") {
        cleaned.push_str(&rest[..open]);
        let after = &rest[open + "<tool_call>".len()..];
        let Some(close) = after.find("</tool_call>") else {
            // unterminated — keep as text
            cleaned.push_str(&rest[open..]);
            rest = "";
            break;
        };
        let body = after[..close].trim();
        let parsed = parse_call_body(body);
        if parsed.is_empty() {
            // not a call shape we understand — leave it in the text rather than
            // silently dropping it or emitting a nameless call the client would
            // report as "Tool not found"
            cleaned.push_str(&rest[open..open + "<tool_call>".len() + close + "</tool_call>".len()]);
        }
        for (name, args) in parsed {
            calls.push(ToolCall {
                id: format!("call_{}", calls.len()),
                kind: "function".into(),
                function: ToolCallFn { name, arguments: args },
            });
        }
        rest = &after[close + "</tool_call>".len()..];
    }
    cleaned.push_str(rest);
    (calls, cleaned.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn names(c: &str) -> Vec<(String, String)> {
        extract_tool_calls(c)
            .0
            .into_iter()
            .map(|t| (t.function.name, t.function.arguments))
            .collect()
    }

    /// The form this model is actually trained to emit.
    #[test]
    fn parses_qwen3_coder_xml() {
        let raw = "Let me check.\n<tool_call>\n<function=get_weather>\n\
                   <parameter=city>\nParis\n</parameter>\n\
                   <parameter=units>\nc\n</parameter>\n</function>\n</tool_call>";
        let (calls, cleaned) = extract_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let a: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(a["city"], "Paris");
        assert_eq!(a["units"], "c");
        assert_eq!(cleaned, "Let me check.");
    }

    /// Non-string parameters come back typed, not as strings.
    #[test]
    fn coerces_parameter_types() {
        let raw = "<tool_call>\n<function=add_numbers>\n<parameter=a>\n17\n</parameter>\n\
                   <parameter=b>\n25\n</parameter>\n<parameter=opts>\n{\"round\": true}\n</parameter>\n\
                   </function>\n</tool_call>";
        let a: Value = serde_json::from_str(&names(raw)[0].1).unwrap();
        assert_eq!(a["a"], 17);
        assert_eq!(a["b"], 25);
        assert_eq!(a["opts"]["round"], true);
    }

    /// Two calls in one block, and two separate blocks.
    #[test]
    fn parses_multiple_calls() {
        let two_in_one = "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n\
                          <function=g>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(names(two_in_one).len(), 2);
        let two_blocks = "<tool_call>\n<function=f>\n</function>\n</tool_call>\n\
                          <tool_call>\n<function=g>\n</function>\n</tool_call>";
        let n = names(two_blocks);
        assert_eq!(n.len(), 2);
        assert_eq!(n[0].0, "f");
        assert_eq!(n[1].0, "g");
        assert_eq!(n[0].1, "{}"); // no parameters -> empty object, never a null name
    }

    /// Hermes JSON still works, including the OpenAI-nested shape.
    #[test]
    fn parses_hermes_json_fallback() {
        let flat = r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>"#;
        assert_eq!(names(flat)[0].0, "get_weather");
        let nested =
            r#"<tool_call>{"function": {"name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}}</tool_call>"#;
        let n = names(nested);
        assert_eq!(n[0].0, "get_weather");
        assert_eq!(n[0].1, r#"{"city":"Rome"}"#);
    }

    /// A block we cannot understand must stay visible as text, NOT become a
    /// nameless call the client reports as "Tool not found" with `{}` args.
    #[test]
    fn never_emits_a_nameless_call() {
        for raw in [
            "<tool_call>{}</tool_call>",
            "<tool_call>{\"arguments\": {\"a\": 1}}</tool_call>",
            "<tool_call>\ngarbage\n</tool_call>",
            "<tool_call>\n<function=>\n</function>\n</tool_call>",
        ] {
            let (calls, cleaned) = extract_tool_calls(raw);
            assert!(calls.is_empty(), "{raw} produced {calls:?}");
            assert!(cleaned.contains("<tool_call>"), "{raw} was silently dropped");
        }
    }

    /// An unterminated block is kept as text (streaming may not have closed it).
    #[test]
    fn keeps_unterminated_block() {
        let raw = "text <tool_call>\n<function=f>";
        let (calls, cleaned) = extract_tool_calls(raw);
        assert!(calls.is_empty());
        assert!(cleaned.contains("<tool_call>"));
    }

    /// The prompt must advertise the XML format, not Hermes JSON — asking for
    /// JSON is what put this model out of distribution.
    #[test]
    fn preamble_asks_for_xml_and_keeps_system_prompt() {
        let tools = [json!({"type": "function", "function": {"name": "f", "parameters": {}}})];
        let msgs = [json!({"role": "system", "content": "You are terse."}),
                    json!({"role": "user", "content": "hi"})];
        let p = render_chatml(&msgs, Some(&tools), Some(false), None, "</think>").text;
        assert!(p.contains("<function=example_function_name>"));
        assert!(!p.contains(r#"{"name": <function-name>"#));
        assert!(p.contains("You are terse."));
        // tool block first, user's system prompt after it
        assert!(p.find("# Tools").unwrap() < p.find("You are terse.").unwrap());
    }

    /// A replayed assistant tool call round-trips into the same XML form.
    #[test]
    fn replays_tool_calls_as_xml() {
        let tools = [json!({"type": "function", "function": {"name": "get_weather"}})];
        let msgs = [
            json!({"role": "user", "content": "weather?"}),
            json!({"role": "assistant", "content": "",
                   "tool_calls": [{"type": "function", "function":
                       {"name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}}]}),
            json!({"role": "tool", "content": "{\"temp\": 19}"}),
        ];
        let p = render_chatml(&msgs, Some(&tools), Some(false), None, "</think>").text;
        assert!(p.contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>"));
        assert!(p.contains("<tool_response>"));
        // and the replayed block parses back to what we started with (skip the
        // preamble's own `<function=example_function_name>` illustration)
        let assistant = &p[p.find("<|im_start|>assistant").unwrap()..];
        let n = names(assistant);
        assert_eq!(n[0].0, "get_weather");
        assert_eq!(n[0].1, r#"{"city":"Rome"}"#);
    }

    // --- fidelity to chat_template.jinja -------------------------------------
    //
    // Each of these covers a place the hand-written renderer had drifted from the
    // model's own template. Drift here is not cosmetic: it puts the model out of
    // distribution and it starts emitting stray/foreign tool-call syntax.

    /// Every assistant turn is framed with `<think>…</think>`, filled from
    /// `reasoning_content` when the client round-trips it. Dropping the frame in
    /// history is what makes a multi-turn tool conversation degrade.
    #[test]
    fn assistant_history_keeps_think_frame() {
        let msgs = [
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello", "reasoning_content": "be brief"}),
            json!({"role": "user", "content": "again"}),
        ];
        let p = render_chatml(&msgs, None, Some(true), None, "</think>").text;
        assert!(p.contains("<|im_start|>assistant\n<think>\nbe brief\n</think>\n\nhello<|im_end|>"));
        // and one with no reasoning_content still gets the empty frame
        let msgs2 = [
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let p2 = render_chatml(&msgs2, None, Some(true), None, "</think>").text;
        assert!(p2.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>"));
    }

    /// `None` (the client said nothing) must prefill `<think>\n`, like the
    /// template's `add_generation_prompt` branch — NOT leave the turn bare.
    #[test]
    fn unspecified_thinking_prefills_think() {
        let msgs = [json!({"role": "user", "content": "hi"})];
        for et in [None, Some(true)] {
            let rp = render_chatml(&msgs, None, et, None, "</think>");
            assert!(rp.text.ends_with("<|im_start|>assistant\n<think>\n"), "{et:?}");
            assert!(rp.starts_in_reasoning, "{et:?}");
        }
        let rp = render_chatml(&msgs, None, Some(false), None, "</think>");
        assert!(rp.text.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!rp.starts_in_reasoning);
    }

    /// The reasoning-effort preamble leads the system turn (default `xhigh`),
    /// and precedes the tool block when there are tools.
    #[test]
    fn emits_reasoning_instructions() {
        let msgs = [json!({"role": "user", "content": "hi"})];
        let p = render_chatml(&msgs, None, None, None, "</think>").text;
        assert!(p.contains("Reasoning effort is set to xhigh."));
        let low = render_chatml(&msgs, None, None, Some("low"), "</think>").text;
        assert!(low.contains("Reasoning effort is set to low."));
        // `medium` carries no instructions, and with no system prompt that means
        // no system turn at all
        let med = render_chatml(&msgs, None, None, Some("medium"), "</think>").text;
        assert!(!med.contains("Reasoning effort"));
        assert!(!med.contains("<|im_start|>system"));
        // disabled thinking suppresses them
        let off = render_chatml(&msgs, None, Some(false), None, "</think>").text;
        assert!(!off.contains("Reasoning effort"));
        // with tools: instructions first, then the tool block
        let tools = [json!({"type": "function", "function": {"name": "f"}})];
        let t = render_chatml(&msgs, Some(&tools), None, None, "</think>").text;
        assert!(t.find("Reasoning effort").unwrap() < t.find("# Tools").unwrap());
    }

    /// Consecutive tool results share ONE user turn, each in its own
    /// `<tool_response>` block — not a separate turn apiece.
    #[test]
    fn groups_consecutive_tool_responses() {
        let msgs = [
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": "ok"}),
            json!({"role": "tool", "content": "a"}),
            json!({"role": "tool", "content": "b"}),
            json!({"role": "user", "content": "next"}),
        ];
        let p = render_chatml(&msgs, None, Some(false), None, "</think>").text;
        assert!(p.contains(
            "<|im_start|>user\n<tool_response>\na\n</tool_response>\n<tool_response>\nb\n</tool_response><|im_end|>"
        ));
        assert_eq!(p.matches("<tool_response>").count(), 2);
    }

    /// A system message that is not first is ignored, as the template requires
    /// (it raises); it must never be hoisted into the system turn.
    #[test]
    fn ignores_late_system_message() {
        let msgs = [
            json!({"role": "user", "content": "hi"}),
            json!({"role": "system", "content": "SNEAKY"}),
        ];
        let p = render_chatml(&msgs, None, Some(false), None, "</think>").text;
        assert!(!p.contains("SNEAKY"));
    }
}

/// Test-only: render the fixture cases in `$EXL3_TPL_CASES` to `$EXL3_TPL_OUT`,
/// so the hand-transcribed renderer can be diffed against the model's real
/// `chat_template.jinja` rendered by Jinja itself.
#[cfg(test)]
#[test]
fn dump_template_fixtures() {
    let (Ok(inp), Ok(outp)) = (std::env::var("EXL3_TPL_CASES"), std::env::var("EXL3_TPL_OUT"))
    else {
        return;
    };
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(inp).unwrap()).unwrap();
    let mut outs = Vec::new();
    for c in &cases {
        let msgs = c["messages"].as_array().unwrap();
        let tools = c.get("tools").and_then(Value::as_array).map(|v| v.as_slice());
        let et = c.get("enable_thinking").and_then(Value::as_bool);
        let effort = c.get("reasoning_effort").and_then(Value::as_str);
        outs.push(render_chatml(msgs, tools, et, effort, "</think>").text);
    }
    std::fs::write(outp, serde_json::to_string(&outs).unwrap()).unwrap();
}
