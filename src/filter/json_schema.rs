//! JSON-Schema (subset) → GBNF compiler. Emits a grammar string that
//! [`super::gbnf::parse`] turns into a matcher. Mirrors what
//! `formatron` / `llguidance` do, without the dependency.
//!
//! Supported: `type` (object/array/string/number/integer/boolean/null),
//! `properties` + `required` + `additionalProperties:false`, `items`, `enum`,
//! `const`, `anyOf`/`oneOf`, `$ref` → `#/$defs/<name>` or `#/definitions/<name>`.
//! Unsupported keywords are ignored (the value is left unconstrained by type).

use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write;

const PRELUDE: &str = r#"
ws     ::= [ \t\n]*
string ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* "\""
integer ::= "-"? ( "0" | [1-9] [0-9]* )
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
boolean ::= "true" | "false"
null   ::= "null"
value  ::= object | array | string | number | boolean | null
object ::= "{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* )? ws "}"
array  ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
"#;

pub fn compile(schema: &Value) -> Result<super::gbnf::Grammar, String> {
    let defs = collect_defs(schema);
    let mut c = Ctx { defs, rules: BTreeMap::new(), n: 0 };
    let root = c.node(schema);
    let mut out = String::from(PRELUDE);
    // no leading/trailing `ws` in root — the value must start immediately, else a
    // greedy model emits unbounded whitespace (it is always permitted by `ws*`).
    let _ = writeln!(out, "root ::= {root}");
    for (name, body) in &c.rules {
        let _ = writeln!(out, "{name} ::= {body}");
    }
    super::gbnf::parse(&out)
}

/// "any JSON value" grammar for `response_format: {type: json_object}`.
pub fn any_json() -> Result<super::gbnf::Grammar, String> {
    super::gbnf::parse(&format!("{PRELUDE}\nroot ::= object\n"))
}

fn collect_defs(schema: &Value) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(o)) = schema.get(key) {
            for (k, v) in o {
                m.insert(k.clone(), v.clone());
            }
        }
    }
    m
}

struct Ctx {
    defs: BTreeMap<String, Value>,
    rules: BTreeMap<String, String>,
    n: usize,
}

impl Ctx {
    fn fresh(&mut self, body: String) -> String {
        self.n += 1;
        let name = format!("n{}", self.n);
        self.rules.insert(name.clone(), body);
        name
    }

    /// Return a grammar symbol matching this schema node.
    fn node(&mut self, s: &Value) -> String {
        let s = &s.clone();
        if let Some(Value::String(r)) = s.get("$ref") {
            let name = r.rsplit('/').next().unwrap_or("").to_string();
            if let Some(def) = self.defs.get(&name).cloned() {
                let key = format!("ref_{name}");
                if !self.rules.contains_key(&key) {
                    self.rules.insert(key.clone(), "value".into()); // break cycles
                    let body = self.node(&def);
                    self.rules.insert(key.clone(), body);
                }
                return key;
            }
            return "value".into();
        }
        if let Some(c) = s.get("const") {
            return self.fresh(lit(c));
        }
        if let Some(Value::Array(vs)) = s.get("enum") {
            let alts: Vec<String> = vs.iter().map(lit).collect();
            return self.fresh(alts.join(" | "));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(vs)) = s.get(key) {
                let alts: Vec<String> = vs.iter().map(|v| self.node(v)).collect();
                return self.fresh(alts.join(" | "));
            }
        }
        match s.get("type").and_then(Value::as_str) {
            Some("string") => "string".into(),
            Some("integer") => "integer".into(),
            Some("number") => "number".into(),
            Some("boolean") => "boolean".into(),
            Some("null") => "null".into(),
            Some("array") => {
                let item = s
                    .get("items")
                    .map(|it| self.node(it))
                    .unwrap_or_else(|| "value".into());
                self.fresh(format!(
                    "\"[\" ws ( {item} ( ws \",\" ws {item} )* )? ws \"]\""
                ))
            }
            Some("object") => {
                let props = match s.get("properties") {
                    Some(Value::Object(o)) => o.clone(),
                    _ => Default::default(),
                };
                let required: std::collections::HashSet<String> = s
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                if props.is_empty() {
                    return "object".into();
                }
                // Required properties first (fixed schema order), then each
                // optional property as `( ws "," ws <kv> )?`. When there are no
                // required properties the first optional carries no leading comma
                // and the rest may only appear after it — a fixed-order subset,
                // which is what schema-constrained decoders converge to anyway.
                let kv = |k: &str, vt: &str| format!("\"\\\"{k}\\\"\" ws \":\" ws {vt}");
                let mut req: Vec<String> = Vec::new();
                let mut opt: Vec<String> = Vec::new();
                for (k, v) in &props {
                    let vt = self.node(v);
                    if required.contains(k) {
                        req.push(kv(k, &vt));
                    } else {
                        opt.push(kv(k, &vt));
                    }
                }
                let body = if !req.is_empty() {
                    let head = req.join(" ws \",\" ws ");
                    let tail: String = opt
                        .iter()
                        .map(|o| format!(" ( ws \",\" ws {o} )?"))
                        .collect();
                    format!("{head}{tail}")
                } else {
                    // ( opt0 ( ws "," ws opt1 )? ( ws "," ws opt2 )? ... )?
                    let inner: String = opt
                        .iter()
                        .enumerate()
                        .map(|(i, o)| {
                            if i == 0 {
                                o.clone()
                            } else {
                                format!(" ( ws \",\" ws {o} )?")
                            }
                        })
                        .collect();
                    format!("( {inner} )?")
                };
                self.fresh(format!("\"{{\" ws {body} ws \"}}\""))
            }
            _ => "value".into(),
        }
    }
}

fn lit(v: &Value) -> String {
    // exact JSON serialization as a quoted grammar literal
    let s = serde_json::to_string(v).unwrap_or_default();
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}
