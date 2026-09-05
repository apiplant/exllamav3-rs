//! Loading the DFlash2 drafter: every tensor found, shapes as the checkpoint
//! declares them. Set `EXL3_DFLASH2_MODEL` to override the path.

#[test]
fn loads_dflash2_checkpoint() {
    let dir = std::env::var("EXL3_DFLASH2_MODEL")
        .unwrap_or_else(|_| "models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw".into());
    let dir = std::path::PathBuf::from(dir);
    // Touch the CUDA shim before any tch call: the linker drops libtorch_cuda
    // from a test binary that pulls no CUDA symbol, and then every CUDA op
    // fails with "not available for this backend" rather than anything obvious.
    let _ = exl3::ffi::cuda_free_mib();
    if !dir.exists() {
        eprintln!("skipping: no model at {}", dir.display());
        return;
    }
    let m = exl3::dflash2::DFlash2Model::load(&dir, tch::Device::Cuda(0)).expect("load dflash2");
    assert_eq!(m.params.block_size, 8);
    assert_eq!(m.params.selector_top_k, 16);
    assert_eq!(m.params.target_layer_ids, vec![5, 19, 33, 47, 61]);
    assert_eq!(m.num_layers(), 5);
    assert_eq!(m.cfg.hidden_size, 5120);
    eprintln!(
        "loaded: {} layers, block {}, taps {:?}, window {}",
        m.num_layers(),
        m.params.block_size,
        m.params.target_layer_ids,
        m.params.sliding_window
    );
}
