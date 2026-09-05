//! End-to-end load + forward of a synthetic Glm4Moe checkpoint.
//!
//! The smallest published `Glm4MoeForCausalLM` exl3 quant is ~27 GB (GLM-4.5-Air
//! at 2.0bpw), which does not fit the 24 GB card this port is developed on, so
//! this architecture cannot be checked against real weights here. What it can be
//! checked for is that the paths unique to it actually execute: the `dots`
//! router and its selection bias, the `first_k_dense_replace` leading dense
//! layers, the ungated shared expert, and NeoX (not GPT-J) RoPE despite the
//! shared "Glm4" name.
//!
//! Build the fixture with `python3 tests/make_glm4moe_fixture.py <dir>`; the
//! test skips itself if it is absent.

use exl3::cache::PagedKvCache;
use exl3::config::{ArchKind, Config, ConfigOverrides, RouterKind};
use exl3::model::Model;
use std::path::PathBuf;
use tch::{Device, Tensor};

fn fixture() -> Option<PathBuf> {
    let d = std::env::var("GLM4MOE_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/glm4moe_fixture"));
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
fn config_reads_the_glm4moe_extras() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no glm4moe fixture");
        return;
    };
    let cfg = Config::from_dir_with(&dir, &ConfigOverrides::default()).unwrap();
    assert_eq!(cfg.arch_kind, ArchKind::Glm4Moe);
    // Despite the name it is not the GLM4 block: no sandwich norms.
    assert!(!cfg.arch_kind.has_post_norms());
    // QK-norm is per checkpoint here, not per architecture.
    assert!(cfg.qk_norm);

    let moe = cfg.moe.as_ref().unwrap();
    assert_eq!(moe.router, RouterKind::Dots);
    assert_eq!(moe.routed_scaling_factor, 2.5);
    assert_eq!(moe.first_k_dense_replace, 1);
    // One shared expert of `moe_intermediate_size * n_shared_experts`, ungated.
    assert_eq!(moe.shared_expert_intermediate_size, moe.moe_intermediate_size);
    assert!(!moe.shared_gate);

    // Layer 0 is dense, the rest are sparse — the leading-dense rule is what
    // decides which tensors each block even looks for.
    assert!(!moe.is_sparse_layer(0));
    for i in 1..cfg.num_hidden_layers {
        assert!(moe.is_sparse_layer(i), "layer {i} should be sparse");
    }
}

#[test]
fn loads_and_forwards() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no glm4moe fixture");
        return;
    };
    let dev = device();
    let model = Model::load(&dir, dev).unwrap();
    let cache = PagedKvCache::new(&model.config, 512, dev);

    let ids = Tensor::from_slice(&[5i64, 9, 12, 7, 3, 11, 4, 8]).view([1, 8]).to_device(dev);
    let logits = model.prefill(&ids, &cache);
    cache.advance(ids.size()[1]);

    assert_eq!(logits.size(), vec![model.config.vocab_size]);
    assert!(bool::try_from(logits.isfinite().all()).unwrap(), "logits went non-finite");
    // Zeros are finite and correctly shaped. An early version of this fixture
    // wrote zero-weight RMSNorms, which zeroed the whole model and still passed
    // every other check here.
    assert!(
        f64::try_from(logits.abs().max()).unwrap() > 1e-3,
        "logits are all zero — the fixture is not exercising the model"
    );
}

/// Prefill then decode must agree with prefilling the whole sequence — the same
/// property the KV cache and the router have to preserve together. A router that
/// read stale rows, or a leading-dense layer loaded as sparse, shows up here.
#[test]
fn chunked_decode_matches_one_shot_prefill() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no glm4moe fixture");
        return;
    };
    let dev = device();
    let model = Model::load(&dir, dev).unwrap();
    let toks: Vec<i64> = vec![5, 9, 12, 7, 3, 11, 4, 8, 2, 6];
    let all = Tensor::from_slice(&toks).view([1, -1]).to_device(dev);
    let n = toks.len() as i64;

    // `prefill` advances the cache itself — advancing again here would put the
    // decode steps at the wrong positions, which is exactly what it looks like
    // when the model is broken.
    let one = PagedKvCache::new(&model.config, 512, dev);
    let want = model.prefill(&all, &one);

    let split = PagedKvCache::new(&model.config, 512, dev);
    let _ = model.prefill(&all.narrow(1, 0, n - 2), &split);
    let mut got = None;
    for i in n - 2..n {
        got = Some(model.forward_paged(&all.narrow(1, i, 1), &split));
        split.advance(1);
    }

    let d = f64::try_from((&want - &got.unwrap()).abs().max()).unwrap();
    let scale = f64::try_from(want.abs().max()).unwrap().max(1.0);
    assert!(d / scale < 2e-2, "chunked decode diverged: max abs diff {d} (scale {scale})");
}
