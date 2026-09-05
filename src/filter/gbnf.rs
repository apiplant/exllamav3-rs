//! GBNF-subset grammar engine — a codepoint-level nondeterministic stack machine
//! modelled on llama.cpp's `llama_grammar` (the proven, compact design).
//!
//! Supported syntax: `name ::= ...`, string literals `"..."`, char classes
//! `[a-z]` / `[^...]`, `.` (any), postfix `* + ?`, `|` alternation, `()` groups,
//! rule references, `{m,n}` counts. The entry rule is `root`.

use std::collections::BTreeSet;

/// One element of a compiled rule (llama.cpp layout).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Elem {
    /// end of an alternate (also end of rule when it is the last)
    End,
    /// alternate separator
    Alt,
    /// reference to rules[value]
    RuleRef(u32),
    /// exact codepoint
    Char(u32),
    /// negated set follows (as a run of Char / CharRngUpper terminated by End-of-set)
    CharNot(u32),
    /// upper bound of a range whose lower bound was the preceding Char/CharNot
    CharRngUpper(u32),
    /// additional alternative codepoint in a `[...]` set
    CharAlt(u32),
}

type Rule = Vec<Elem>;

pub struct Grammar {
    rules: Vec<Rule>,
    root: u32,
}

// --- parser --------------------------------------------------------------

pub fn parse(src: &str) -> Result<Grammar, String> {
    let mut p = Parser { rules: Vec::new(), names: Vec::new() };
    let mut lines: Vec<(String, String)> = Vec::new();
    // join continuation: split on top-level `name ::=`
    let mut cur_name = String::new();
    let mut cur_body = String::new();
    for raw in src.lines() {
        let line = strip_comment(raw);
        if line.trim().is_empty() {
            continue;
        }
        if let Some(idx) = find_def(line) {
            if !cur_name.is_empty() {
                lines.push((cur_name.clone(), cur_body.clone()));
            }
            cur_name = line[..idx].trim().to_string();
            cur_body = line[idx + 3..].to_string();
        } else {
            cur_body.push(' ');
            cur_body.push_str(line);
        }
    }
    if !cur_name.is_empty() {
        lines.push((cur_name, cur_body));
    }
    if lines.is_empty() {
        return Err("empty grammar".into());
    }
    // pre-register names so forward refs resolve
    for (name, _) in &lines {
        p.intern(name);
    }
    for (name, body) in &lines {
        let id = p.intern(name) as usize;
        let rule = p.parse_alternates(body)?;
        if p.rules.len() <= id {
            p.rules.resize(id + 1, Vec::new());
        }
        p.rules[id] = rule;
    }
    let root = p
        .names
        .iter()
        .position(|n| n == "root")
        .ok_or("grammar has no `root` rule")? as u32;
    Ok(Grammar { rules: p.rules, root })
}

fn strip_comment(l: &str) -> &str {
    match l.find('#') {
        Some(i) => &l[..i],
        None => l,
    }
}
fn find_def(l: &str) -> Option<usize> {
    l.find("::=")
}

struct Parser {
    rules: Vec<Rule>,
    names: Vec<String>,
}

impl Parser {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return i as u32;
        }
        self.names.push(name.to_string());
        self.rules.push(Vec::new());
        (self.names.len() - 1) as u32
    }

    fn fresh_rule(&mut self, r: Rule) -> u32 {
        let name = format!("__r{}", self.rules.len());
        let id = self.intern(&name);
        self.rules[id as usize] = r;
        id
    }

    fn parse_alternates(&mut self, body: &str) -> Result<Rule, String> {
        let toks: Vec<char> = body.chars().collect();
        let mut pos = 0usize;
        let seqs = self.parse_seq_alts(&toks, &mut pos, false)?;
        Ok(seqs)
    }

    /// Parse `alt ( "|" alt )*` into a flat Rule (`... Alt ... End`).
    fn parse_seq_alts(
        &mut self,
        s: &[char],
        pos: &mut usize,
        in_group: bool,
    ) -> Result<Rule, String> {
        let mut out: Rule = Vec::new();
        loop {
            let seq = self.parse_sequence(s, pos, in_group)?;
            out.extend(seq);
            skip_ws(s, pos);
            if *pos < s.len() && s[*pos] == '|' {
                *pos += 1;
                out.push(Elem::Alt);
                continue;
            }
            break;
        }
        out.push(Elem::End);
        Ok(out)
    }

    fn parse_sequence(
        &mut self,
        s: &[char],
        pos: &mut usize,
        in_group: bool,
    ) -> Result<Vec<Elem>, String> {
        let mut out: Vec<Elem> = Vec::new();
        loop {
            skip_ws(s, pos);
            if *pos >= s.len() {
                break;
            }
            let c = s[*pos];
            if c == '|' {
                break;
            }
            if c == ')' {
                if in_group {
                    break;
                }
                return Err("unmatched )".into());
            }
            let atom_start = out.len();
            match c {
                '"' => {
                    *pos += 1;
                    while *pos < s.len() && s[*pos] != '"' {
                        let ch = self.read_escaped(s, pos)?;
                        out.push(Elem::Char(ch));
                    }
                    if *pos >= s.len() {
                        return Err("unterminated string".into());
                    }
                    *pos += 1;
                }
                '[' => {
                    *pos += 1;
                    let neg = *pos < s.len() && s[*pos] == '^';
                    if neg {
                        *pos += 1;
                    }
                    let mut first = true;
                    while *pos < s.len() && s[*pos] != ']' {
                        let lo = self.read_escaped(s, pos)?;
                        let hi = if *pos + 1 < s.len() && s[*pos] == '-' && s[*pos + 1] != ']' {
                            *pos += 1;
                            self.read_escaped(s, pos)?
                        } else {
                            lo
                        };
                        if first {
                            out.push(if neg { Elem::CharNot(lo) } else { Elem::Char(lo) });
                            first = false;
                        } else {
                            out.push(Elem::CharAlt(lo));
                        }
                        if hi != lo {
                            out.push(Elem::CharRngUpper(hi));
                        }
                    }
                    if *pos >= s.len() {
                        return Err("unterminated char class".into());
                    }
                    *pos += 1;
                    if first {
                        return Err("empty char class".into());
                    }
                }
                '.' => {
                    *pos += 1;
                    // any codepoint = [^] — represent as CharNot(0) with no alts,
                    // which our matcher treats as "any except nothing" = any.
                    out.push(Elem::CharNot(0));
                    out.push(Elem::CharRngUpper(0)); // exclude just NUL
                }
                '(' => {
                    *pos += 1;
                    let sub = self.parse_seq_alts(s, pos, true)?;
                    skip_ws(s, pos);
                    if *pos >= s.len() || s[*pos] != ')' {
                        return Err("missing )".into());
                    }
                    *pos += 1;
                    let id = self.fresh_rule(sub);
                    out.push(Elem::RuleRef(id));
                }
                c if is_name_char(c) => {
                    let start = *pos;
                    while *pos < s.len() && is_name_char(s[*pos]) {
                        *pos += 1;
                    }
                    let name: String = s[start..*pos].iter().collect();
                    let id = self.intern(&name);
                    out.push(Elem::RuleRef(id));
                }
                other => return Err(format!("unexpected char {other:?} in grammar")),
            }

            // postfix repetition on the atom just parsed
            skip_ws(s, pos);
            if *pos < s.len() && matches!(s[*pos], '*' | '+' | '?') {
                let op = s[*pos];
                *pos += 1;
                let atom: Vec<Elem> = out.split_off(atom_start);
                let id = self.wrap_repeat(atom, op);
                out.push(Elem::RuleRef(id));
            } else if *pos < s.len() && s[*pos] == '{' {
                *pos += 1;
                let start = *pos;
                while *pos < s.len() && s[*pos] != '}' {
                    *pos += 1;
                }
                let spec: String = s[start..*pos].iter().collect();
                *pos += 1;
                let atom: Vec<Elem> = out.split_off(atom_start);
                let id = self.wrap_count(atom, &spec)?;
                out.push(Elem::RuleRef(id));
            }
        }
        Ok(out)
    }

    /// atom* / atom+ / atom?  →  a fresh recursive rule.
    fn wrap_repeat(&mut self, atom: Vec<Elem>, op: char) -> u32 {
        // sub = the atom as its own rule
        let mut sub = atom;
        sub.push(Elem::End);
        let sub_id = self.fresh_rule(sub);
        match op {
            '?' => {
                // R ::= sub |
                self.fresh_rule(vec![Elem::RuleRef(sub_id), Elem::Alt, Elem::End])
            }
            '*' => {
                // R ::= sub R |
                let name = format!("__r{}", self.rules.len());
                let id = self.intern(&name);
                self.rules[id as usize] = vec![
                    Elem::RuleRef(sub_id),
                    Elem::RuleRef(id),
                    Elem::Alt,
                    Elem::End,
                ];
                id
            }
            '+' => {
                let name = format!("__r{}", self.rules.len());
                let id = self.intern(&name);
                // star = __sN ::= sub __sN |
                let star = format!("__s{}", self.rules.len());
                let star_id = self.intern(&star);
                self.rules[star_id as usize] = vec![
                    Elem::RuleRef(sub_id),
                    Elem::RuleRef(star_id),
                    Elem::Alt,
                    Elem::End,
                ];
                self.rules[id as usize] =
                    vec![Elem::RuleRef(sub_id), Elem::RuleRef(star_id), Elem::End];
                id
            }
            _ => unreachable!(),
        }
    }

    fn wrap_count(&mut self, atom: Vec<Elem>, spec: &str) -> Result<u32, String> {
        let (min, max) = match spec.split_once(',') {
            Some((a, b)) => (
                a.trim().parse::<u32>().map_err(|_| "bad {m,n}")?,
                if b.trim().is_empty() {
                    None
                } else {
                    Some(b.trim().parse::<u32>().map_err(|_| "bad {m,n}")?)
                },
            ),
            None => {
                let n = spec.trim().parse::<u32>().map_err(|_| "bad {n}")?;
                (n, Some(n))
            }
        };
        let mut sub = atom.clone();
        sub.push(Elem::End);
        let sub_id = self.fresh_rule(sub);
        let mut seq: Vec<Elem> = Vec::new();
        for _ in 0..min {
            seq.push(Elem::RuleRef(sub_id));
        }
        match max {
            Some(mx) => {
                for _ in min..mx {
                    // optional sub
                    let opt = self.fresh_rule(vec![Elem::RuleRef(sub_id), Elem::Alt, Elem::End]);
                    seq.push(Elem::RuleRef(opt));
                }
            }
            None => {
                let star = self.wrap_repeat(atom, '*');
                seq.push(Elem::RuleRef(star));
            }
        }
        seq.push(Elem::End);
        Ok(self.fresh_rule(seq))
    }

    fn read_escaped(&self, s: &[char], pos: &mut usize) -> Result<u32, String> {
        let c = s[*pos];
        if c != '\\' {
            *pos += 1;
            return Ok(c as u32);
        }
        *pos += 1;
        if *pos >= s.len() {
            return Err("trailing backslash".into());
        }
        let e = s[*pos];
        *pos += 1;
        Ok(match e {
            'n' => '\n' as u32,
            'r' => '\r' as u32,
            't' => '\t' as u32,
            '\\' => '\\' as u32,
            '"' => '"' as u32,
            ']' => ']' as u32,
            '[' => '[' as u32,
            '-' => '-' as u32,
            '/' => '/' as u32,
            'x' => {
                let h: String = s[*pos..(*pos + 2).min(s.len())].iter().collect();
                *pos += 2;
                u32::from_str_radix(&h, 16).map_err(|_| "bad \\x")?
            }
            'u' => {
                let h: String = s[*pos..(*pos + 4).min(s.len())].iter().collect();
                *pos += 4;
                u32::from_str_radix(&h, 16).map_err(|_| "bad \\u")?
            }
            other => other as u32,
        })
    }
}

fn skip_ws(s: &[char], pos: &mut usize) {
    while *pos < s.len() && s[*pos].is_whitespace() {
        *pos += 1;
    }
}
fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_'
}

// --- matcher ------------------------------------------------------------

/// A stack is a list of positions into rule-element arrays, innermost last.
type Pos = (u32, usize); // (rule id, index)

#[derive(Clone)]
pub struct Matcher {
    grammar: std::sync::Arc<Grammar>,
    /// current set of live stacks (an empty stack = a fully-parsed branch)
    stacks: Vec<Vec<Pos>>,
    /// fed a byte no live stack could accept — grammar violated
    dead: bool,
}

impl Matcher {
    pub fn new(grammar: std::sync::Arc<Grammar>) -> Self {
        let mut m = Matcher { grammar, stacks: Vec::new(), dead: false };
        m.reset();
        m
    }

    pub fn reset(&mut self) {
        let root = self.grammar.root;
        self.dead = false;
        let init: Vec<Vec<Pos>> = alt_starts(&self.grammar, root)
            .into_iter()
            .map(|s| vec![(root, s)])
            .collect();
        self.stacks = advance_stacks(&self.grammar, init);
    }

    /// No further token can extend the parse (every live branch is fully
    /// consumed) — the job should stop. A `dead` matcher is NOT complete.
    pub fn is_complete(&self) -> bool {
        !self.dead && !self.stacks.is_empty() && self.stacks.iter().all(|s| s.is_empty())
    }

    /// Grammar was violated (last accepted piece had no valid continuation).
    pub fn is_dead(&self) -> bool {
        self.dead
    }

    /// The current parse is at a point where stopping would be valid.
    pub fn can_end(&self) -> bool {
        !self.dead && self.stacks.iter().any(|s| s.is_empty())
    }

    /// Feed the raw bytes of one emitted token piece.
    pub fn accept_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() || self.dead || self.is_complete() {
            return;
        }
        let text = String::from_utf8_lossy(bytes).into_owned();
        for ch in text.chars() {
            // drop the "already parsed" branches before consuming more input
            let live: Vec<Vec<Pos>> =
                self.stacks.iter().filter(|s| !s.is_empty()).cloned().collect();
            if live.is_empty() {
                self.dead = true;
                return;
            }
            self.stacks = step_char(&self.grammar, &live, ch as u32);
            if self.stacks.is_empty() {
                self.dead = true;
                return;
            }
        }
    }

    /// A cheap hash of the live stack set, for mask caching.
    pub fn signature(&self) -> u64 {
        let mut h = 1469598103934665603u64;
        for st in &self.stacks {
            for &(r, i) in st {
                h ^= r as u64;
                h = h.wrapping_mul(1099511628211);
                h ^= i as u64;
                h = h.wrapping_mul(1099511628211);
            }
            h ^= 0xABCD;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }

    pub fn acceptor(&self) -> GrammarAcceptor {
        GrammarAcceptor {
            grammar: self.grammar.clone(),
            stacks: self.stacks.clone(),
            pending: Vec::new(),
        }
    }
}

/// Expand any leading rule refs / consumed elements so every stack's top is a
/// terminal (char element) or the stack is empty (parse complete).
fn advance_stacks(g: &Grammar, stacks: Vec<Vec<Pos>>) -> Vec<Vec<Pos>> {
    let mut out: Vec<Vec<Pos>> = Vec::new();
    let mut seen: BTreeSet<Vec<Pos>> = BTreeSet::new();
    let mut work = stacks;
    while let Some(stack) = work.pop() {
        if stack.is_empty() {
            if seen.insert(stack.clone()) {
                out.push(stack);
            }
            continue;
        }
        let (rid, idx) = *stack.last().unwrap();
        let elem = g.rules[rid as usize][idx];
        match elem {
            Elem::End => {
                let mut ns = stack.clone();
                ns.pop();
                if let Some((prid, pidx)) = ns.last().copied() {
                    ns.pop();
                    ns.push((prid, pidx + 1));
                }
                work.push(ns);
            }
            Elem::Alt => {
                // an alternate boundary reached by falling through means the
                // previous alternate matched fully; skip to this rule's End.
                let rule = &g.rules[rid as usize];
                let mut j = idx;
                while j < rule.len() && rule[j] != Elem::End {
                    j += 1;
                }
                let mut ns = stack.clone();
                ns.pop();
                ns.push((rid, j));
                work.push(ns);
            }
            Elem::RuleRef(sub) => {
                // for each alternate of `sub`, push a branch
                let starts = alt_starts(g, sub);
                for st in starts {
                    let mut ns = stack.clone();
                    ns.push((sub, st));
                    work.push(ns);
                }
            }
            Elem::Char(_) | Elem::CharNot(_) => {
                if seen.insert(stack.clone()) {
                    out.push(stack);
                }
            }
            Elem::CharRngUpper(_) | Elem::CharAlt(_) => {
                // should never be a stack top; skip past
                let mut ns = stack.clone();
                let l = ns.len() - 1;
                ns[l].1 += 1;
                work.push(ns);
            }
        }
    }
    out
}

/// Indices at which each alternate of `rule` begins.
fn alt_starts(g: &Grammar, rule: u32) -> Vec<usize> {
    let mut v = vec![0usize];
    let elems = &g.rules[rule as usize];
    for (i, e) in elems.iter().enumerate() {
        if *e == Elem::Alt {
            v.push(i + 1);
        }
    }
    v
}

/// Does the char-set starting at `elems[idx]` (a Char/CharNot possibly followed
/// by CharRngUpper / CharAlt) match codepoint `c`?
fn char_set_matches(elems: &[Elem], idx: usize, c: u32) -> (bool, usize) {
    let mut i = idx;
    let (negated, first) = match elems[i] {
        Elem::Char(v) => (false, v),
        Elem::CharNot(v) => (true, v),
        _ => return (false, idx + 1),
    };
    let mut matched = false;
    // handle first element (may have a range upper next)
    let mut lo = first;
    i += 1;
    let check = |lo: u32, hi: u32, c: u32| c >= lo && c <= hi;
    if i < elems.len() {
        if let Elem::CharRngUpper(hi) = elems[i] {
            if check(lo, hi, c) {
                matched = true;
            }
            i += 1;
        } else if lo != 0 || !negated {
            if c == lo {
                matched = true;
            }
        }
    } else if c == lo {
        matched = true;
    }
    // additional alts
    while i < elems.len() {
        match elems[i] {
            Elem::CharAlt(v) => {
                lo = v;
                i += 1;
                if i < elems.len() {
                    if let Elem::CharRngUpper(hi) = elems[i] {
                        if check(lo, hi, c) {
                            matched = true;
                        }
                        i += 1;
                        continue;
                    }
                }
                if c == lo {
                    matched = true;
                }
            }
            _ => break,
        }
    }
    let res = if negated { !matched } else { matched };
    (res, i)
}

fn step_char(g: &Grammar, stacks: &[Vec<Pos>], c: u32) -> Vec<Vec<Pos>> {
    let mut next: Vec<Vec<Pos>> = Vec::new();
    for stack in stacks {
        let Some(&(rid, idx)) = stack.last() else { continue };
        let elems = &g.rules[rid as usize];
        if idx >= elems.len() {
            continue;
        }
        let (ok, after) = char_set_matches(elems, idx, c);
        if ok {
            let mut ns = stack.clone();
            let l = ns.len() - 1;
            ns[l] = (rid, after);
            next.push(ns);
        }
    }
    advance_stacks(g, next)
}

/// [`ByteAcceptor`]-shaped view over a matcher state, for the vocab-trie walk.
/// The trie feeds raw bytes; `pending` buffers a partial UTF-8 sequence until a
/// full codepoint is available, which is then fed to the codepoint-level grammar.
#[derive(Clone)]
pub struct GrammarAcceptor {
    grammar: std::sync::Arc<Grammar>,
    stacks: Vec<Vec<Pos>>,
    pending: Vec<u8>,
}

impl super::ByteAcceptor for GrammarAcceptor {
    fn step(&self, byte: u8) -> Option<Self> {
        let mut buf = self.pending.clone();
        buf.push(byte);
        match std::str::from_utf8(&buf) {
            Ok(s) => {
                let mut stacks = self.stacks.clone();
                for ch in s.chars() {
                    stacks = step_char(&self.grammar, &stacks, ch as u32);
                    if stacks.is_empty() {
                        return None;
                    }
                }
                Some(Self { grammar: self.grammar.clone(), stacks, pending: Vec::new() })
            }
            // valid but incomplete multi-byte prefix — keep buffering
            Err(e) if e.error_len().is_none() && buf.len() < 4 => {
                Some(Self { grammar: self.grammar.clone(), stacks: self.stacks.clone(), pending: buf })
            }
            _ => None,
        }
    }
    fn can_end(&self) -> bool {
        self.pending.is_empty() && self.stacks.iter().any(|s| s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::ByteAcceptor;
    use std::sync::Arc;

    fn m(src: &str) -> Matcher {
        Matcher::new(Arc::new(parse(src).unwrap()))
    }
    /// feed a whole string, return (still-live, complete)
    fn run(src: &str, input: &str) -> (bool, bool) {
        let mut mm = m(src);
        mm.accept_bytes(input.as_bytes());
        (!mm.stacks.is_empty() || mm.is_complete(), mm.is_complete())
    }

    #[test]
    fn literal_alternation() {
        assert_eq!(run("root ::= \"yes\" | \"no\"", "yes").1, true);
        assert_eq!(run("root ::= \"yes\" | \"no\"", "no").1, true);
        let mut mm = m("root ::= \"yes\" | \"no\"");
        mm.accept_bytes(b"y");
        assert!(!mm.is_complete() && !mm.stacks.is_empty());
        mm.accept_bytes(b"x");
        assert!(mm.is_dead() && !mm.is_complete());
    }

    #[test]
    fn repetition_and_class() {
        // one or more digits
        let mut mm = m("root ::= [0-9]+");
        mm.accept_bytes(b"1");
        assert!(mm.acceptor().can_end());
        mm.accept_bytes(b"42");
        assert!(mm.acceptor().can_end());
        let mut bad = m("root ::= [0-9]+");
        bad.accept_bytes(b"a");
        assert!(bad.stacks.is_empty());
    }

    #[test]
    fn optional_and_group() {
        let g = "root ::= \"a\" (\"b\" | \"c\")? \"d\"";
        assert_eq!(run(g, "ad").1, true);
        assert_eq!(run(g, "abd").1, true);
        assert_eq!(run(g, "acd").1, true);
        assert_eq!(run(g, "aed").1, false);
    }

    #[test]
    fn json_any() {
        let gr = super::super::json_schema::any_json().unwrap();
        let mut mm = Matcher::new(Arc::new(gr));
        mm.accept_bytes(b"{\"a\": 1}");
        assert!(mm.acceptor().can_end(), "valid object should be acceptable");
    }

    #[test]
    fn json_schema_object() {
        let schema: serde_json::Value = serde_json::json!({
            "type": "object",
            "properties": { "name": {"type": "string"}, "age": {"type": "integer"} },
            "required": ["name", "age"]
        });
        let gr = super::super::json_schema::compile(&schema).unwrap();
        let mut mm = Matcher::new(Arc::new(gr));
        mm.accept_bytes(b"{\"name\": \"bob\" , \"age\": 42}");
        assert!(mm.acceptor().can_end());
    }

    #[test]
    fn json_schema_full_object_stream() {
        // feed a realistic model continuation byte-by-byte and confirm every
        // prefix stays live (this is what the mask walk needs)
        let schema: serde_json::Value = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age"]
        });
        let gr = Arc::new(super::super::json_schema::compile(&schema).unwrap());
        let full = r#"{"name": "Alice", "age": 30, "city": "Paris"}"#;
        let mut mm = Matcher::new(gr.clone());
        for (i, ch) in full.char_indices() {
            let before_dead = mm.is_dead();
            mm.accept_bytes(&[ch as u8]);
            assert!(
                !mm.is_dead() || before_dead,
                "died feeding {:?} at byte {i} of {full}",
                ch
            );
        }
        assert!(mm.acceptor().can_end());
        // opening brace must be allowed from the start state; leading whitespace
        // is NOT (root has no leading `ws` — see json_schema::compile)
        let acc = Matcher::new(gr.clone()).acceptor();
        assert!(acc.step(b'{').is_some(), "start state rejects '{{'");
        assert!(acc.step(b' ').is_none(), "start state should reject leading ws");
        // a full CJK codepoint (3 bytes) must be rejected at the start state
        let cjk = "中".as_bytes();
        let mut a = Some(Matcher::new(gr).acceptor());
        for &b in cjk {
            a = a.and_then(|x| x.step(b));
        }
        assert!(a.is_none(), "start state should reject a CJK char");
    }

    #[test]
    fn json_schema_optional_prop() {
        let schema: serde_json::Value = serde_json::json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "note": {"type": "string"}
            },
            "required": ["id"]
        });
        let gr = Arc::new(super::super::json_schema::compile(&schema).unwrap());
        // required only
        let mut a = Matcher::new(gr.clone());
        a.accept_bytes(b"{\"id\": 7}");
        assert!(a.acceptor().can_end());
        // required + optional
        let mut b = Matcher::new(gr.clone());
        b.accept_bytes(b"{\"id\": 7, \"note\": \"hi\"}");
        assert!(b.acceptor().can_end());
        // missing required -> rejected
        let mut c = Matcher::new(gr);
        c.accept_bytes(b"{\"note\": \"hi\"}");
        assert!(c.is_dead());
    }
}
