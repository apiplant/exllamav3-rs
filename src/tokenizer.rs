//! Thin wrapper over the `tokenizers` crate loading the model's `tokenizer.json`
//! (grade C — exllamav3's own trie / piece-healing bookkeeping is not reproduced).

use anyhow::{anyhow, Result};
use std::path::Path;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

pub struct Tok {
    inner: Tokenizer,
    /// lazily built id → surface piece table (raw vocab only), for token healing
    pieces: OnceLock<Vec<String>>,
}

impl Tok {
    pub fn load(dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        Ok(Self { inner, pieces: OnceLock::new() })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("{e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
    }

    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner.decode(&ids, false).map_err(|e| anyhow!("{e}"))
    }

    /// Raw (non-special) vocab size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// id → surface piece table, mirroring `tokenizer.py` `_get_fixed_vocab`: each
    /// piece is `decode([space, id])` with the leading-space piece stripped, so
    /// byte-level-BPE word-boundary markers become real leading spaces. Built once.
    fn pieces(&self) -> &[String] {
        self.pieces.get_or_init(|| {
            let n = self.vocab_size() as u32;
            let space_id = self
                .inner
                .encode(" ", false)
                .ok()
                .and_then(|e| e.get_ids().first().copied());
            let Some(space_id) = space_id else {
                // fall back to plain per-id decode
                return (0..n)
                    .map(|i| self.inner.decode(&[i], false).unwrap_or_default())
                    .collect();
            };
            let prefix = self.inner.decode(&[space_id], false).unwrap_or_default();
            let seqs: Vec<Vec<u32>> = (0..n).map(|i| vec![space_id, i]).collect();
            let refs: Vec<&[u32]> = seqs.iter().map(|v| v.as_slice()).collect();
            match self.inner.decode_batch(&refs, false) {
                Ok(v) => v
                    .into_iter()
                    .map(|s| s.strip_prefix(&prefix).map(str::to_string).unwrap_or(s))
                    .collect(),
                Err(_) => (0..n)
                    .map(|i| self.inner.decode(&[i], false).unwrap_or_default())
                    .collect(),
            }
        })
    }

    /// Full id → surface piece table (see [`Tok::piece`]). Used to build the
    /// constrained-decoding vocab trie (`filter::VocabTrie`).
    pub fn piece_table(&self) -> &[String] {
        self.pieces()
    }

    /// Surface piece for a single token id (empty if out of range).
    pub fn piece(&self, id: i64) -> String {
        self.pieces().get(id as usize).cloned().unwrap_or_default()
    }

    /// Token ids whose surface piece starts with `prefix` — `tokenizer.py`
    /// `get_tokens_with_prefix_string`, used for token healing.
    pub fn tokens_with_prefix(&self, prefix: &str) -> Vec<i64> {
        if prefix.is_empty() {
            return Vec::new();
        }
        self.pieces()
            .iter()
            .enumerate()
            .filter(|(_, p)| p.starts_with(prefix))
            .map(|(i, _)| i as i64)
            .collect()
    }

    /// `Qwen3Model.default_chat_prompt`.
    pub fn qwen_chat_prompt(prompt: &str, system: Option<&str>) -> String {
        let mut p = String::new();
        if let Some(s) = system {
            p.push_str("<|im_start|>system\n");
            p.push_str(s);
            p.push_str("<|im_end|>\n");
        }
        p.push_str("<|im_start|>user\n");
        p.push_str(prompt);
        p.push_str("<|im_end|>\n");
        p.push_str("<|im_start|>assistant\n");
        p
    }
}
