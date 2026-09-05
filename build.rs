// Compiles the ExLlamaV3 CUDA kernels verbatim and links them, plus the C-ABI
// shim, against the same libtorch that `tch` / `torch-sys` link.
//
// Each .cu is compiled whole-program (no `-dc` / device-link) exactly like the
// upstream `torch.utils.cpp_extension` JIT build — the EXL3 comp_units define
// `__device__` globals with external linkage that collide under `nvcc -dlink`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn torch_paths() -> (PathBuf, PathBuf) {
    if let Ok(p) = std::env::var("LIBTORCH") {
        let p = PathBuf::from(p);
        return (p.join("include"), p.join("lib"));
    }
    let out = Command::new("python3")
        .args(["-c", "import torch,os;print(os.path.dirname(torch.__file__))"])
        .output()
        .expect("need `torch` installed (pip) or LIBTORCH set");
    let base = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    (base.join("include"), base.join("lib"))
}

fn main() {
    // CUDA kernel sources are vendored into the crate (crate/kernels/), compiled
    // verbatim. Formerly pointed at the upstream python package via a symlink.
    let ext = PathBuf::from("kernels")
        .canonicalize()
        .expect("crate/kernels not found");
    let (torch_inc, torch_lib) = torch_paths();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=csrc/exl3_shim.cpp");
    println!("cargo:rerun-if-changed=csrc/exl3_shim.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");

    let mut sources: Vec<PathBuf> = Vec::new();
    collect_cu(&ext.join("quant"), &mut sources);
    for f in [
        "graph.cu", "norm.cu", "rope.cu", "hgemm.cu", "attention.cu",
        "activation.cu", "add.cu", "softcap.cu",
        "gdn.cu",             // gated-delta-net (Qwen3.5 linear attention)
        "gdn_chunk.cu",       // WY-chunked gated delta rule (prefill)
    ] {
        sources.push(ext.join(f));
    }
    for f in [
        "cache.cu",           // paged KV cache update + cache_rotate
        "rep_pen.cu",         // repetition / presence / frequency penalties
        "sampling_basic.cu",  // argmax_sample / gumbel_sample
        "gumbel.cu",          // gumbel_noise_*
    ] {
        sources.push(ext.join("generator").join(f));
    }
    sources.push(ext.join("cache/q_cache.cu")); // KV cache quantization
    sources.push(ext.join("cuda_drv.cpp"));
    sources.push(PathBuf::from("csrc/exl3_shim.cpp"));

    let includes = vec![
        ext.clone(),
        torch_inc.clone(),
        torch_inc.join("torch/csrc/api/include"),
    ];

    let gencode = {
        let arch = std::env::var("TORCH_CUDA_ARCH_LIST").unwrap_or_default();
        if arch.is_empty() {
            vec!["-arch=native".to_string()]
        } else {
            arch.split([',', ' '])
                .filter(|s| !s.is_empty())
                .map(|a| {
                    let a = a.trim().replace('.', "");
                    format!("-gencode=arch=compute_{a},code=sm_{a}")
                })
                .collect()
        }
    };

    let base_flags: Vec<&str> = vec![
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler=-fPIC",
        // host gcc here is far newer than libtorch 2.12 expects
        "-Xcompiler=-fpermissive",
        "-Xcompiler=-w",
        "-diag-suppress=177,20012,20014",
        "-D_GLIBCXX_USE_CXX11_ABI=1", // pip libtorch 2.12 is cxx11-ABI
    ];

    // newest mtime across all kernel/shim headers — any of them can be included
    // by any .cu, so treat them as a dependency of every object
    fn newest_header(dir: &std::path::Path, acc: &mut std::time::SystemTime) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                newest_header(&p, acc);
            } else if matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("cuh") | Some("h") | Some("hpp")
            ) {
                if let Ok(t) = e.metadata().and_then(|m| m.modified()) {
                    if t > *acc {
                        *acc = t;
                    }
                }
            }
        }
    }
    let mut newest_dep = std::time::SystemTime::UNIX_EPOCH;
    newest_header(std::path::Path::new("kernels"), &mut newest_dep);
    newest_header(std::path::Path::new("csrc"), &mut newest_dep);

    let mut objs = Vec::new();
    for src in &sources {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let obj = out_dir.join(format!("{stem}.o"));
        // Compare against the newest of the .cu AND every header it could pull
        // in. Comparing only against the .cu meant editing a .cuh silently kept
        // a stale object file — kernel changes appeared to have no effect until
        // some .cu happened to be touched.
        let up_to_date = obj.exists()
            && std::fs::metadata(&obj).and_then(|m| m.modified()).ok()
                >= Some(newest_dep.max(
                    std::fs::metadata(src)
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                ));
        if !up_to_date {
            let mut cmd = Command::new("nvcc");
            cmd.args(&base_flags).args(&gencode);
            for i in &includes {
                cmd.arg("-I").arg(i);
            }
            cmd.arg("-c").arg(src).arg("-o").arg(&obj);
            let status = cmd.status().expect("nvcc not found");
            assert!(status.success(), "nvcc failed on {}", src.display());
        }
        objs.push(obj);
    }

    let lib = out_dir.join("libexl3kernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .args(&objs)
        .status()
        .unwrap();
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=exl3kernels");
    println!("cargo:rustc-link-search=native={}", torch_lib.display());
    for l in ["torch", "torch_cpu", "torch_cuda", "c10", "c10_cuda"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", torch_lib.display());
    if let Ok(cuda) = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")) {
        println!("cargo:rustc-link-search=native={cuda}/lib64");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn collect_cu(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            collect_cu(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("cu") {
            out.push(p);
        }
    }
}
