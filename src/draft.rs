//! Separate autoregressive draft model for the batched `Generator` — behavioural
//! port of `generator/generator.py` `iterate_draftmodel_gen` (grade —).
//!
//! A small AR model (e.g. Qwen3-0.6B) proposes `n` future tokens per round; the
//! target verifies them all in one `q_len = n+1` forward (the shared verify path,
//! `Generator::spec_verify_round`, identical to the n-gram path). The draft model
//! keeps its own paged KV pool, allocated in lockstep with the target's pages.
//!
//! Not ported: dynamic draft length / confidence calibration
//! (`DraftConfidenceCalibrator`), recurrent draft models (asserted off, as
//! upstream), CFG / multi-sequence jobs, MRoPE.

use crate::model::Model;
use crate::paged::{pages_for, PageTable, PagedCache};
use anyhow::{bail, Result};
use std::path::Path;
use tch::{Device, Kind, Tensor};

pub struct DraftModel {
    model: Model,
    cache: PagedCache,
    pages: PageTable,
    device: Device,
}

impl DraftModel {
    /// Load a draft model and size its KV pool to the same token capacity as the
    /// target's shared pool (`num_pages` target pages).
    pub fn load(dir: &Path, device: Device, target: &Model, num_pages: i64) -> Result<Self> {
        let model = Model::load(dir, device)?;
        if model.config.arch_kind == crate::config::ArchKind::Qwen35 {
            bail!("draft model must not be recurrent/hybrid (Qwen3.5)");
        }
        if model.config.vocab_size != target.config.vocab_size {
            bail!(
                "draft vocab {} != target vocab {}",
                model.config.vocab_size,
                target.config.vocab_size
            );
        }
        let cache = PagedCache::new(&model.config, num_pages, device);
        Ok(Self {
            model,
            cache,
            pages: PageTable::new(num_pages),
            device,
        })
    }

    /// Allocate draft pages for a job needing `total_tokens` of context; returns
    /// the physical page list to store on the `Job`.
    pub fn admit(&mut self, total_tokens: i64) -> Result<Vec<i32>> {
        self.pages.alloc(pages_for(total_tokens) as usize)
    }

    pub fn release(&mut self, block: &[i32]) {
        self.pages.release(block);
    }

    fn block_tensor(&self, block: &[i32]) -> Tensor {
        Tensor::from_slice(block)
            .reshape([1, block.len() as i64])
            .to_device(self.device)
    }

    /// Prime the draft KV over prompt positions `0 .. prompt.len()-2` (mirrors the
    /// target, which holds its last prompt token back for the first decode step).
    pub fn prime_row(&self, prompt: &[i64], block: &[i32]) {
        let n = prompt.len() as i64 - 1;
        if n <= 0 {
            return;
        }
        let ids = Tensor::from_slice(&prompt[..n as usize])
            .reshape([1, n])
            .to_device(self.device);
        let sl0 = Tensor::from_slice(&[0i32]).to_device(self.device);
        let _ = self.model.forward_paged_batched(
            &ids,
            &self.cache.k,
            &self.cache.v,
            &self.block_tensor(block),
            &sl0,
            true,
        );
    }

    /// Draft `n` greedy tokens per row. `feed[r]` is the token at position
    /// `pos[r]` (the target's held-back token), `blocks[r]` the row's draft page
    /// list. Returns one `Vec<i64>` of length `n` per row.
    pub fn draft_batch(
        &self,
        feed: &[i64],
        pos: &[i64],
        blocks: &[&[i32]],
        n: i64,
    ) -> Vec<Vec<i64>> {
        let _no_grad = tch::no_grad_guard();
        let bsz = feed.len() as i64;
        let max_pages = blocks.iter().map(|b| b.len()).max().unwrap_or(1) as i64;
        let mut bt = vec![0i32; (bsz * max_pages) as usize];
        for (r, b) in blocks.iter().enumerate() {
            bt[r * max_pages as usize..r * max_pages as usize + b.len()].copy_from_slice(b);
        }
        let block_table = Tensor::from_slice(&bt)
            .reshape([bsz, max_pages])
            .to_device(self.device);

        let mut cur = Tensor::from_slice(feed).reshape([bsz, 1]).to_device(self.device);
        let mut cols: Vec<Tensor> = Vec::with_capacity(n as usize);
        for i in 0..n {
            let sl: Vec<i32> = pos.iter().map(|&p| (p + i) as i32).collect();
            let seqlens = Tensor::from_slice(&sl).to_device(self.device);
            let logits = self.model.forward_paged_batched(
                &cur,
                &self.cache.k,
                &self.cache.v,
                &block_table,
                &seqlens,
                true,
            ); // [bsz,1,vocab]
            cur = logits.squeeze_dim(1).argmax(-1, false).reshape([bsz, 1]);
            cols.push(cur.shallow_clone());
        }
        let out = Tensor::cat(&cols, 1).to_kind(Kind::Int64).to_device(Device::Cpu); // [bsz,n]
        (0..bsz)
            .map(|r| (0..n).map(|c| out.int64_value(&[r, c])).collect())
            .collect()
    }
}
