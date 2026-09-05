//! `Model::enable_draft_head` must be *exact* on the vocabulary it keeps.
//!
//! The pruned head drops output blocks; it does not approximate the ones it
//! keeps. If that ever stops holding, drafts silently degrade in a way that
//! only shows up as a mysterious acceptance-rate drop, so pin it down here.
//!
//! Needs a checkpoint; set `EXL3_TEST_MODEL` (default: the 0.6B in ./models).

use tch::{Device, Kind, Tensor};

#[test]
fn pruned_draft_head_matches_full_head() {
    let dir = std::env::var("EXL3_TEST_MODEL")
        .unwrap_or_else(|_| "models/Qwen3-0.6B-exl3-8.0bpw_H8".into());
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        eprintln!("skipping: no model at {}", dir.display());
        return;
    }
    if !tch::Cuda::is_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let dev = Device::Cuda(0);
    let mut model = exl3::model::Model::load(&dir, dev).expect("load");
    let cut = 8192.min(model.config.vocab_size / 2);
    model.enable_draft_head(cut, &[]).expect("enable_draft_head");

    let h = model.config.hidden_size;
    let x = Tensor::randn([1, 4, h], (Kind::Half, dev));
    let full = model.lm_head_on(&x);
    let (pruned, id_map) = model.draft_logits_on(&x);
    let id_map = id_map.expect("draft head should be built");

    // every kept output must equal the full head's logit for that token id
    let want = full.index_select(2, &id_map);
    let diff = (&pruned - &want).abs().max().double_value(&[]);
    let scale = want.abs().max().double_value(&[]);
    // Not bit-identical: the trellis GEMM picks its split-K/tiling from N, so a
    // narrower head reduces in a different order and lands within an fp16 ulp.
    // The kept blocks are still the *same* arithmetic, and an ulp on a draft
    // logit can at worst flip a near-tie into a rejected draft.
    eprintln!("max abs diff {diff} over logits of magnitude {scale}");
    assert!(diff / scale < 1e-3, "pruned head diverges: {diff} on scale {scale}");

    // and the ids it kept are the frequent prefix plus the special tail
    let ids = Vec::<i64>::try_from(id_map.to_device(Device::Cpu)).unwrap();
    assert!(ids.contains(&0) && ids.contains(&(cut - 1)));
    assert!(ids.contains(&(model.config.vocab_size - 1)), "special tail dropped");
    assert!((ids.len() as i64) < model.config.vocab_size);
}
