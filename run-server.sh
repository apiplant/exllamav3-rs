#!/usr/bin/env bash
# Convenience wrapper: sets the env libtorch + the CUDA kernels need, builds the
# server, and runs it. All server config comes from --config <config.yml>.
#
#   ./run-server.sh --config config.example.yml
#   ./run-server.sh --config /mnt/extra/ai/llm/tabbyAPI/config.yml --host 0.0.0.0
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

cargo build --release --bin server >&2
exec ./target/release/server "$@"
