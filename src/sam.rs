//! Byte-cost suffix automaton — 1:1 port of `exllamav3_ext/sam.{h,cpp}` (`BC_SAM`),
//! grade A. Drives n-gram speculative decoding: it is fed the running token
//! sequence and returns the span `[start, end)` of the longest suffix of the
//! sequence that also occurred earlier, so the tokens right after that earlier
//! occurrence can be used as a draft.

/// Suffix automaton over an append-only token stream (with rebuild-on-shrink).
pub struct BcSam {
    link: Vec<i32>,
    max_len: Vec<i32>,
    min_end: Vec<i32>,
    first_edge: Vec<i32>,

    edge_token: Vec<i32>,
    edge_to: Vec<i32>,
    edge_next: Vec<i32>,

    last: i32,
    match_state: i32,
    match_len: i32,
    pos: i64,
}

impl Default for BcSam {
    fn default() -> Self {
        let mut s = BcSam {
            link: Vec::new(),
            max_len: Vec::new(),
            min_end: Vec::new(),
            first_edge: Vec::new(),
            edge_token: Vec::new(),
            edge_to: Vec::new(),
            edge_next: Vec::new(),
            last: 0,
            match_state: 0,
            match_len: 0,
            pos: 0,
        };
        s.reset(0);
        s
    }
}

impl BcSam {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self, reserve_tokens: i64) {
        assert!(reserve_tokens >= 0, "reserve_tokens must be >= 0");
        let r = reserve_tokens as usize;

        self.link.clear();
        self.max_len.clear();
        self.min_end.clear();
        self.first_edge.clear();
        self.edge_token.clear();
        self.edge_to.clear();
        self.edge_next.clear();

        let state_cap = if r > 0 { 2 * r + 1 } else { 1 };
        let edge_cap = if r > 0 { 3 * r + 1 } else { 0 };
        self.link.reserve(state_cap);
        self.max_len.reserve(state_cap);
        self.min_end.reserve(state_cap);
        self.first_edge.reserve(state_cap);
        self.edge_token.reserve(edge_cap);
        self.edge_to.reserve(edge_cap);
        self.edge_next.reserve(edge_cap);

        // root
        self.link.push(-1);
        self.max_len.push(0);
        self.min_end.push(0x7fff_ffff);
        self.first_edge.push(-1);

        self.last = 0;
        self.match_state = 0;
        self.match_len = 0;
        self.pos = 0;
    }

    pub fn length(&self) -> i64 {
        self.pos
    }

    /// Feed one token, returning `(start, end)` (end exclusive) of the longest
    /// matching earlier suffix, or `(-1, -1)` when there is none.
    pub fn accept(&mut self, token: i64) -> (i64, i64) {
        let (state, match_len) = self.advance_match(token as i32);

        let mut start = -1i64;
        let mut end = -1i64;
        if match_len > 0 {
            let source_end = self.min_end[state as usize];
            start = source_end as i64 - match_len as i64 + 1;
            end = source_end as i64 + 1;
        }

        self.extend(token as i32);
        (start, end)
    }

    /// Feed the full token sequence so far (as i64 slice). Tracks how much it has
    /// already consumed; rebuilds from scratch if the sequence shrank.
    pub fn accept_tensor(&mut self, tokens: &[i64]) -> (i64, i64) {
        let total = tokens.len() as i64;
        if total < self.length() {
            self.reset(total);
        }

        let offset = self.length() as usize;
        let len = tokens.len() - offset;
        if len < 1 {
            return (-1, -1);
        }
        let slice = &tokens[offset..];

        for &t in &slice[..len - 1] {
            self.advance_match(t as i32);
            self.extend(t as i32);
        }
        let (state, match_len) = self.advance_match(slice[len - 1] as i32);
        self.extend(slice[len - 1] as i32);

        let mut start = -1i64;
        let mut end = -1i64;
        if match_len > 0 {
            let source_end = self.min_end[state as usize];
            start = source_end as i64 - match_len as i64 + 1;
            end = source_end as i64 + 1;
        }
        (start, end)
    }

    fn new_state(&mut self, max_len: i32, link: i32, min_end: i32) -> i32 {
        let idx = self.link.len() as i32;
        self.link.push(link);
        self.max_len.push(max_len);
        self.min_end.push(min_end);
        self.first_edge.push(-1);
        idx
    }

    fn add_edge(&mut self, from: i32, token: i32, to: i32) {
        self.edge_token.push(token);
        self.edge_to.push(to);
        self.edge_next.push(self.first_edge[from as usize]);
        self.first_edge[from as usize] = self.edge_to.len() as i32 - 1;
    }

    fn find_edge(&self, state: i32, token: i32) -> i32 {
        let mut e = self.first_edge[state as usize];
        while e != -1 {
            if self.edge_token[e as usize] == token {
                return e;
            }
            e = self.edge_next[e as usize];
        }
        -1
    }

    fn advance_match(&mut self, token: i32) -> (i32, i32) {
        let mut state = self.match_state;
        let mut length = self.match_len;

        let mut edge = self.find_edge(state, token);
        while state != 0 && edge == -1 {
            state = self.link[state as usize];
            length = length.min(self.max_len[state as usize]);
            edge = self.find_edge(state, token);
        }

        if edge != -1 {
            state = self.edge_to[edge as usize];
            length += 1;
        } else {
            state = 0;
            length = 0;
        }

        self.match_state = state;
        self.match_len = length;
        (state, length)
    }

    fn extend(&mut self, token: i32) {
        let pos32 = self.pos as i32;

        let cur = self.new_state(self.max_len[self.last as usize] + 1, 0, pos32);
        let mut p = self.last;

        while p != -1 && self.find_edge(p, token) == -1 {
            self.add_edge(p, token, cur);
            p = self.link[p as usize];
        }

        if p == -1 {
            self.link[cur as usize] = 0;
        } else {
            let e = self.find_edge(p, token);
            let q = self.edge_to[e as usize];
            if self.max_len[p as usize] + 1 == self.max_len[q as usize] {
                self.link[cur as usize] = q;
            } else {
                let clone =
                    self.new_state(self.max_len[p as usize] + 1, self.link[q as usize], self.min_end[q as usize]);

                // Copy q's outgoing transitions into clone.
                let mut ee = self.first_edge[q as usize];
                while ee != -1 {
                    let (tok, to) = (self.edge_token[ee as usize], self.edge_to[ee as usize]);
                    self.add_edge(clone, tok, to);
                    ee = self.edge_next[ee as usize];
                }

                while p != -1 {
                    let pe = self.find_edge(p, token);
                    if pe == -1 || self.edge_to[pe as usize] != q {
                        break;
                    }
                    self.edge_to[pe as usize] = clone;
                    p = self.link[p as usize];
                }

                self.link[q as usize] = clone;
                self.link[cur as usize] = clone;
            }
        }

        self.last = cur;
        self.pos += 1;
    }
}
