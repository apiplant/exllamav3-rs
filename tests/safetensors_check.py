#!/usr/bin/env python3
"""Differential test: our safetensors reader vs Python's `safetensors`.

The loader reads each tensor by absolute file offset, split across threads for
large tensors. Both the offset base and the split are silent-failure arithmetic
— a wrong `begin` still yields a correctly shaped tensor of the wrong weights —
so compare a byte-level fingerprint of every tensor against the reference reader.

    python crate/tests/safetensors_check.py [MODEL_DIR]
"""
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile

CRATE = pathlib.Path(__file__).resolve().parent.parent
DEFAULT = "/mnt/extra/ai/llm/qwen3.8-27B/Qwen3.8-27B-heretic-ara-exl3-4.0bpw"


def fingerprint(nbytes: int, samples) -> str:
    """Must match `dump_tensor_checksums` in src/safetensors.rs exactly."""
    h = 1469598103934665603
    M = 0xFFFFFFFFFFFFFFFF
    h = ((h ^ nbytes) * 1099511628211) & M
    for x in samples:
        h = ((h ^ x) * 1099511628211) & M
    return f"{h:016x}"


def reference(model):
    import torch
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                t = h.get_tensor(k).contiguous()
                # `.view(torch.uint8).flatten()` is the tensor's bytes in order,
                # and works for the bf16/fp8 dtypes numpy cannot represent. The
                # two obvious alternatives both fail at this scale:
                # `bytes(t.untyped_storage())` iterates in Python, and
                # `ctypes.string_at` overflows its size argument past 2 GB.
                # reshape(-1) first: a 0-dim scalar tensor cannot be dtype-viewed
                u8 = t.reshape(-1).view(torch.uint8)
                out[k] = fingerprint(u8.numel(), u8[::4096].tolist())
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "ours.json"
        env = dict(os.environ, LIBTORCH_USE_PYTORCH="1",
                   LIBTORCH_BYPASS_VERSION_CHECK="1", TORCH_CUDA_ARCH_LIST="8.9",
                   EXL3_ST_MODEL=model, EXL3_ST_OUT=str(out))
        subprocess.run(["cargo", "test", "--release", "dump_tensor_checksums"],
                       cwd=CRATE, env=env, check=True, stdout=subprocess.DEVNULL)
        ours = json.loads(out.read_text())

    ref = reference(model)
    missing = sorted(set(ref) - set(ours))
    bad = sorted(k for k in ref if k in ours and ref[k] != ours[k])
    for k in bad[:10]:
        print(f"MISMATCH {k}: reference={ref[k]} ours={ours[k]}")
    for k in missing[:10]:
        print(f"MISSING {k}")
    print(f"tensors: {len(ref)}  missing: {len(missing)}  mismatched: {len(bad)}")
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
