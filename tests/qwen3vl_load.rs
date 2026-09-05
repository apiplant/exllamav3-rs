//! End-to-end load + forward of a synthetic Qwen3-VL checkpoint.
//!
//! The Qwen3 block and the Qwen3-VL tower are each exercised against real
//! checkpoints elsewhere. What only exists when the two are loaded *together* is
//! the deepstack path: taps after selected vision blocks, patch mergers that
//! normalize after the spatial shuffle rather than before, and the per-layer
//! injection of those features into the text residual. This fixture covers that
//! seam without waiting on a multi-gigabyte download.
//!
//! Build it with `python3 tests/make_qwen3vl_fixture.py <dir>`; the test skips
//! itself if the fixture is absent.

use exl3::cache::PagedKvCache;
use exl3::config::{ArchKind, Config, ConfigOverrides};
use exl3::model::Model;
use exl3::vision::VisionModel;
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};

fn fixture() -> Option<PathBuf> {
    let d = std::env::var("QWEN3VL_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/qwen3vl_fixture"));
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
fn config_reads_a_vl_checkpoint_as_the_plain_block() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen3vl fixture");
        return;
    };
    let cfg = Config::from_dir_with(&dir, &ConfigOverrides::default()).unwrap();
    // Qwen3-VL is the Qwen3 block with the decoder moved under a longer prefix;
    // it is deliberately not its own ArchKind.
    assert_eq!(cfg.arch_kind, ArchKind::Qwen3);
    assert_eq!(cfg.key_prefix, "model.language_model");
    assert!(!cfg.arch_kind.is_hybrid());
    // `text_config` nesting is detected by presence, not by architecture.
    assert_eq!(cfg.hidden_size, 512);
    assert_eq!(cfg.num_hidden_layers, 4);

    let vc = cfg.vision.as_ref().expect("vision_config");
    assert_eq!(vc.deepstack_visual_indexes, vec![1, 2]);
    assert_eq!(cfg.mrope_section, Some([32, 16, 16]));
}

/// The tower must produce one deepstack block per configured index, each shaped
/// like the merged image features — that is what the text side adds.
#[test]
fn tower_emits_one_deepstack_block_per_index() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen3vl fixture");
        return;
    };
    let dev = device();
    let cfg = Config::from_dir_with(&dir, &ConfigOverrides::default()).unwrap();
    let vm = VisionModel::load(&dir, &cfg, dev).unwrap();

    let img = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([(x * 4) as u8, (y * 4) as u8, 128])
    });
    let path = std::env::temp_dir().join("qwen3vl_fixture_probe.png");
    img.save(&path).unwrap();

    let ie = vm.embed_image(path.to_str().unwrap()).unwrap();
    assert_eq!(ie.deepstack.len(), 2, "one block per deepstack_visual_index");
    for d in &ie.deepstack {
        assert_eq!(d.size(), ie.embeddings.size());
        assert!(bool::try_from(d.isfinite().all()).unwrap());
    }
    // Distinct taps of a randomly-weighted tower must not collapse onto each
    // other; identical blocks would mean the same merger ran twice.
    // A dead tower produces zeros, which are finite and correctly shaped — the
    // first version of this fixture had zero-weight LayerNorms and every check
    // below passed on an all-zero model.
    assert!(f64::try_from(ie.embeddings.abs().max()).unwrap() > 1e-3, "tower output is all zeros");
    let spread = f64::try_from((&ie.deepstack[0] - &ie.deepstack[1]).abs().max()).unwrap();
    assert!(spread > 1e-3, "deepstack taps are identical: {spread}");
}

/// Deepstack has to actually reach the decoder. Running the same spliced stream
/// with and without it must change the logits — if the injection were dropped
/// (or added outside the image span) the model would still produce fluent text
/// while ignoring most of the image.
#[test]
fn deepstack_changes_the_logits() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen3vl fixture");
        return;
    };
    let dev = device();
    let model = Model::load(&dir, dev).unwrap();
    let cfg = &model.config;
    let seq = 16i64;

    let x = Tensor::randn([1, seq, cfg.hidden_size], (Kind::Half, dev));
    // Non-zero only over an "image span", as the real path builds it.
    let ds: Vec<Tensor> = (0..2)
        .map(|_| {
            let t = Tensor::zeros([1, seq, cfg.hidden_size], (Kind::Half, dev));
            let _ = t
                .narrow(1, 4, 6)
                .copy_(&Tensor::randn([1, 6, cfg.hidden_size], (Kind::Half, dev)));
            t
        })
        .collect();

    let c0 = PagedKvCache::new(cfg, 128, dev);
    let (_, plain) = model.forward_paged_mm(&x, &c0, None, &[]);
    let c1 = PagedKvCache::new(cfg, 128, dev);
    let (_, with_ds) = model.forward_paged_mm(&x, &c1, None, &ds);

    assert_eq!(plain.size(), with_ds.size());
    assert!(bool::try_from(with_ds.isfinite().all()).unwrap());
    let d = f64::try_from((&plain - &with_ds).abs().max()).unwrap();
    assert!(d > 1e-3, "deepstack made no difference to the logits: {d}");
}

/// A checkpoint with no deepstack must be unaffected — this is the Qwen3.5 case,
/// and it is what guarantees the refactor did not change the verified path.
#[test]
fn no_deepstack_matches_the_plain_forward() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen3vl fixture");
        return;
    };
    let dev = device();
    let model = Model::load(&dir, dev).unwrap();
    let cfg = &model.config;
    let ids = Tensor::from_slice(&[5i64, 9, 12, 7, 3, 11]).view([1, 6]).to_device(dev);

    let c0 = PagedKvCache::new(cfg, 128, dev);
    let want = model.prefill(&ids, &c0);

    let c1 = PagedKvCache::new(cfg, 128, dev);
    let (_, logits) = model.forward_paged_mm(&model.embed_tokens(&ids), &c1, None, &[]);
    let got = logits.select(0, 0).select(0, ids.size()[1] - 1);

    let d = f64::try_from((&want - &got).abs().max()).unwrap();
    assert!(d < 1e-3, "mm forward diverged from the plain one with no image: {d}");
}

/// Chunked decode must match a one-shot prefill on the plain paged path. This is
/// the control for the same check on the MoE fixtures: if it fails here, with a
/// dense Qwen3 block and no experts, the divergence is not about routing.
#[test]
fn chunked_decode_matches_one_shot_prefill() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: no qwen3vl fixture");
        return;
    };
    let dev = device();
    let model = Model::load(&dir, dev).unwrap();
    let toks: Vec<i64> = vec![5, 9, 12, 7, 3, 11, 4, 8, 2, 6];
    let all = Tensor::from_slice(&toks).view([1, -1]).to_device(dev);
    let n = toks.len() as i64;

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
