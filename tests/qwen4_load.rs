//! End-to-end load + forward of a synthetic qwen4_exp checkpoint.
//!
//! There is no public `Qwen4ExpForConditionalGeneration` exl3 checkpoint, so
//! this cannot check numerics against a reference. What it does check is that
//! the wiring holds: config parsing, every tensor key the loaders ask for, the
//! hybrid layer schedule, the hyper-connection sites, the PLE + n-gram path, the
//! QSA selection mask, and that a chunked decode advances the caches the same
//! way a single prefill does.
//!
//! Build the fixture with `python3 tests/make_qwen4_fixture.py <dir>`; the test
//! skips itself if the fixture is absent (it needs torch + safetensors).

use exl3::cache::Qwen4Cache;
use exl3::config::{ArchKind, Config, ConfigOverrides};
use exl3::model::Model;
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};

fn fixture() -> Option<PathBuf> {
    let d = std::env::var("QWEN4_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/qwen4_fixture"));
    d.join("config.json").exists().then_some(d)
}

fn device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

#[test]
fn config_reads_the_qwen4_extras() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen4 fixture");
        return;
    };
    let cfg = Config::from_dir_with(&dir, &ConfigOverrides::default()).unwrap();
    assert_eq!(cfg.arch_kind, ArchKind::Qwen4Exp);
    assert_eq!(cfg.key_prefix, "model.language_model");
    let q4 = cfg.qwen4.as_ref().unwrap();
    assert_eq!(q4.hc_mult, 4);
    // `ple_layer_ids` is 1-based in the checkpoint and 0-based in the config.
    assert_eq!(q4.ple_layer_ids, vec![1]);
    // The sigmoid GDN output gate is what distinguishes this from Qwen3.5.
    assert_eq!(cfg.gdn_gate_act, exl3::ffi::GateAct::Sigmoid);
    assert!(cfg.moe.as_ref().unwrap().shared_gate);
    assert!(cfg.arch_kind.is_hybrid());
}

#[test]
fn loads_and_forwards() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen4 fixture");
        return;
    };
    let dev = device();
    let model = Model::load(&dir, dev).unwrap();
    let cache = Qwen4Cache::new(&model.config, 512, dev);

    let ids = Tensor::from_slice(&[5i64, 9, 12, 7, 3, 11, 4, 8, 2, 6]).view([1, 10]).to_device(dev);
    let logits = model.forward_qwen4(&ids, &cache);
    cache.advance(ids.size()[1]);

    assert_eq!(logits.size(), vec![model.config.vocab_size]);
    assert_eq!(cache.past_len.get(), 10);
    // Zeros would be finite and correctly shaped, and a dead model passes every
    // other check in this file.
    assert!(
        f64::try_from(logits.abs().max()).unwrap() > 1e-3,
        "logits are all zero — the fixture is not exercising the model"
    );
    assert!(
        bool::try_from(logits.isfinite().all()).unwrap(),
        "logits went non-finite: a NaN here means a norm, a mask row with nothing \
         visible, or an uninitialized buffer somewhere in the stack"
    );
}

/// Prefill then decode token by token must land on the same logits as prefilling
/// the whole sequence at once. This is the property every piece of per-slot
/// state in the stack exists to preserve — the GDN recurrence, the PLE conv
/// window and token history, the QSA pooled keys and the K/V caches — so it is
/// the one test that exercises all of them against each other.
#[test]
fn chunked_decode_matches_one_shot_prefill() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen4 fixture");
        return;
    };
    let dev = device();
    let model = Model::load(&dir, dev).unwrap();
    let toks: Vec<i64> = vec![5, 9, 12, 7, 3, 11, 4, 8, 2, 6, 14, 21];
    let all = Tensor::from_slice(&toks).view([1, -1]).to_device(dev);
    let n = toks.len() as i64;

    let one = Qwen4Cache::new(&model.config, 512, dev);
    let want = model.forward_qwen4(&all, &one);
    one.advance(n);

    let split = Qwen4Cache::new(&model.config, 512, dev);
    let head = all.narrow(1, 0, n - 3);
    let _ = model.forward_qwen4(&head, &split);
    split.advance(n - 3);
    let mut got = None;
    for i in n - 3..n {
        got = Some(model.forward_qwen4(&all.narrow(1, i, 1), &split));
        split.advance(1);
    }
    let got = got.unwrap();

    assert_eq!(split.past_len.get(), one.past_len.get());
    let d = f64::try_from((&want - &got).abs().max()).unwrap();
    let scale = f64::try_from(want.abs().max()).unwrap().max(1.0);
    assert!(
        d / scale < 2e-2,
        "chunked decode diverged from prefill: max abs diff {d} (scale {scale})"
    );
}

/// The QSA mask is the whole point of the full-attention layers: it must stay
/// causal and must not let a query see more than its budget plus its own tail
/// block, at every position of a real forward.
#[test]
fn qsa_mask_stays_causal_and_bounded() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen4 fixture");
        return;
    };
    let dev = device();
    let cfg = Config::from_dir_with(&dir, &ConfigOverrides::default()).unwrap();
    let q4 = cfg.qwen4.as_ref().unwrap();
    let p = exl3::qsa::QsaParams::new(
        q4.indexer_n_heads,
        q4.indexer_head_dim,
        q4.indexer_budget,
        q4.indexer_compress_ratio,
    );
    let total = 64;
    let nb = total / p.compress_ratio;
    let q = Tensor::randn([1, total, p.n_heads, p.head_dim], (Kind::Float, dev));
    let pooled = Tensor::randn([1, nb, p.head_dim], (Kind::Float, dev));
    let mask = p.token_mask(&q, &pooled, 0, total).squeeze_dim(0);

    for s in 0..total {
        let row = mask.select(0, s);
        assert!(!bool::try_from(row.narrow(0, s + 1, total - s - 1).any()).unwrap());
        let n = f64::try_from(row.to_kind(Kind::Int64).sum(Kind::Int64)).unwrap();
        assert!(n >= 1.0, "row {s} can see nothing at all");
        assert!(n <= (p.token_budget + p.compress_ratio) as f64, "row {s} sees {n}");
    }
}
