#!/usr/bin/env bash
# Numerical + timing check of the tensor-core attention kernel against the
# scalar reference. Needs the GPU free (a CUDA context is ~0.5 GB).
#   ./run-attn-check.sh [fp16|q4|q8|all]
set -euo pipefail
cd "$(dirname "$0")"
export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
# Compile for the GPU actually present unless the caller pins it (e.g. to build
# a binary for a different card). nvidia-smi reports "8.9"; fall back to 8.9 only
# if it cannot be queried.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
fi
TORCH_LIB="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
cargo build --release --bin attn_check >&2
exec ./target/release/attn_check "${1:-all}"
