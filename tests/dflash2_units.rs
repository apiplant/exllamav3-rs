//! Narrow the DFlash2 forward down to its pieces, so a GPU fault points at one
//! of them instead of at "somewhere in the draft path".
use tch::{Device, Kind, Tensor};

fn model() -> Option<exl3::dflash2::DFlash2Model> {
    let _ = exl3::ffi::cuda_free_mib();
    let dir = std::env::var("EXL3_DFLASH2_MODEL")
        .unwrap_or_else(|_| "models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw".into());
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        eprintln!("skipping: no model");
        return None;
    }
    Some(exl3::dflash2::DFlash2Model::load(&dir, Device::Cuda(0)).expect("load"))
}

#[test]
fn project_taps_runs() {
    let Some(m) = model() else { return };
    let dev = Device::Cuda(0);
    let h = m.cfg.hidden_size;
    for len in [1i64, 8, 14] {
        let taps: Vec<Tensor> = (0..m.params.target_layer_ids.len())
            .map(|_| Tensor::randn([1, len, h], (Kind::Half, dev)))
            .collect();
        let y = m.project_taps(&taps);
        tch::Cuda::synchronize(0);
        eprintln!("project_taps len={len} -> {:?} finite={}", y.size(), y.isfinite().all().int64_value(&[]));
        assert_eq!(y.size(), vec![1, len, h]);
    }
}

#[test]
fn update_kv_from_target_runs() {
    let Some(m) = model() else { return };
    let dev = Device::Cuda(0);
    let h = m.cfg.hidden_size;
    let mut cache = exl3::dflash2::DFlash2Cache::new(&m, dev);
    for (base, len) in [(0i64, 14i64), (14, 1), (15, 8)] {
        let taps: Vec<Tensor> = (0..m.params.target_layer_ids.len())
            .map(|_| Tensor::randn([1, len, h], (Kind::Half, dev)))
            .collect();
        m.update_kv_from_target(&mut cache, &taps, base);
        tch::Cuda::synchronize(0);
        eprintln!("update_kv base={base} len={len} -> end_pos={}", cache.end_pos());
    }
    assert_eq!(cache.end_pos(), 23);
}

/// A prefill chunk larger than the cache's slack above the sliding window used
/// to overrun the buffer: `ensure_room` evicted down to `window`, leaving only
/// `cap - window` (512) free, and a wider chunk then failed its `narrow` with
/// "start (2048) + length (N) exceeds dimension size (2560)". The positions in
/// that chunk were silently never stored, so the drafter ran on a stale window.
#[test]
fn ingest_chunk_larger_than_window_slack() {
    let Some(model) = model() else { return };
    let device = Device::Cuda(0);
    let mut c = exl3::dflash2::DFlash2Cache::new(&model, device);
    let h = model.cfg.hidden_size;
    let taps = |n: i64| -> Vec<Tensor> {
        (0..model.params.target_layer_ids.len())
            .map(|_| Tensor::randn([1, n, h], (Kind::Half, device)))
            .collect()
    };
    // fill past the window, then push chunks wider than the 512-token slack
    let mut pos = 0i64;
    for n in [2048i64, 1280, 1280, 2048] {
        let t = taps(n);
        model.update_kv_from_target(&mut c, &t, pos);
        pos += n;
        assert!(c.len() <= c.cap(), "len {} exceeded cap {}", c.len(), c.cap());
        assert_eq!(c.end_pos(), pos, "end_pos drifted from the ingested stream");
    }
    // a single chunk bigger than the whole buffer keeps only its tail
    let n = c.cap() + 777;
    let t = taps(n);
    model.update_kv_from_target(&mut c, &t, pos);
    pos += n;
    assert!(c.len() <= c.cap());
    assert_eq!(c.end_pos(), pos, "oversized chunk lost track of the stream head");
}
