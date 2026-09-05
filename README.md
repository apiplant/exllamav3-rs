# exllamav3-rs

A Rust port of [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) —
EXL3-quantized LLM inference, with an OpenAI-compatible server.

The CUDA kernels are **not** rewritten. Upstream's `exllamav3_ext/**` tree is vendored
verbatim into `kernels/` and compiled by `build.rs`, so kernel numerics are identical to
Python by construction. A small C-ABI shim (`csrc/exl3_shim.cpp`) re-exports the ops the
inference path needs, taking `at::Tensor*` — ABI-identical to `tch`'s tensor handle. The
model logic, loader, generator, sampler and server are Rust.

It still links libtorch (via `tch` / `torch-sys`) for tensor plumbing, the caching
allocator and CUDA-graph capture. The upstream kernels are PyTorch C++ extensions —
~2500 `at::Tensor` references across 123 of them — so "no libtorch" would mean rewriting
those, not just this crate.

## Where it diverges from upstream

Beyond the port itself, several things were changed or added. The ones that alter
behaviour or performance:

**New kernels**
- **Tensor-core attention** (`kernels/attention_mma*.cuh`). Upstream's fast path is
  Triton; this is an `mma.sync` flash kernel that reads the fp16 cache *or* the packed
  quant codes directly. Rows of the mma M dimension are (query, query-head) pairs, so a
  short speculative-verify batch fills its tiles instead of wasting most of them.
  `EXL3_MMA_ATTN=0` restores the scalar path.
- **Coalesced GDN conv1d prefill.** The upstream kernel is one thread per channel walking
  the sequence — right for a 1-token decode, badly uncoalesced for a long chunk (measured
  114 GB/s of ~1008). A tiled kernel that loads along the sequence and stores along
  channels gets 551 GB/s, bit-exact.
- **Register-resident delta-rule prefill.** The recurrence kept its state in global memory
  and touched it three times per step. With `save_history == false` only the final state
  is needed, so each thread holds its slice in registers and writes back once.
- **WY-chunked gated delta rule** (`src/gdn_chunk.rs` + `kernels/gdn_chunk.cu`), on by
  default for prefill chunks ≥ 128 tokens. `EXL3_GDN_CHUNK=0` reverts to the sequential
  kernel, which decode and speculative verification still use.

**Added**
- OpenAI-compatible HTTP server that reads a [tabbyAPI](https://github.com/theroyallab/tabbyAPI)
  `config.yml` as-is — chat + completions, SSE streaming, stop strings, tool calls, vision.
- Automatic KV-pool sizing: the pool is fitted to the VRAM actually left after the weights,
  vision tower and draft model load, rather than a hand-tuned constant.
- GDN-aware prefix cache for the hybrid arch — a follow-up turn restores the recurrent and
  conv state from a checkpoint at the shared-prefix boundary, not just the KV pages.
- Adaptive speculative draft window, sized per round from an online cost model.
- DFlash2 draft-model support, n-gram speculative decode, token healing, loop detection.

Full engineering log, including things that were **measured and rejected**, is in
[`PLAN.md`](PLAN.md).

## Status — what is actually tested

Honest summary. "Verified" means run end-to-end and checked against the Python reference
or a known-good output; greedy output is byte-identical to upstream on every prompt tested.

[`ARCHITECTURES.md`](ARCHITECTURES.md) has the full checklist against all 62 upstream
architectures. In short:

| Architecture | State |
|---|---|
| `Qwen3ForCausalLM` | **Verified** — Qwen3-0.6B-exl3-8.0bpw |
| `Qwen3_5For{ConditionalGeneration,CausalLM}` | **Verified** — Qwen3.8-27B at 3.5 and 4.0bpw: text, MTP self-speculation, vision tower, KV-cache quant, prefix cache |
| `DFlash2DraftModel` | **Verified** as a drafter for the above |
| `Qwen3MoeForCausalLM` | **Untested** — block-sparse experts implemented, no checkpoint run yet |
| `Glm4ForCausalLM` | **Untested** — code written, no checkpoint run yet |
| `Llama` / `Mistral` / `Qwen2` / `MiMo` / `SeedOss` / `IQuestCoder` | **Untested** — same dense path, no checkpoint run yet |

That is 4 verified and 8 implemented-but-untested of 63; the other 51 are **not
implemented** and error out by name at load. Vision beyond Qwen3.5, MLA attention, SSM,
hyper-connections and per-layer embeddings are all absent, as are quantization/conversion,
tensor parallelism and LoRA.

The MoE implementation is deliberately unfused — one grouped GEMM per active expert rather
than upstream's fused multi-GEMM, which is not reachable through our C-ABI shim. Same math,
same output, lower throughput; it stays as the reference if the fused path is added.

Other known limits:
- Numerical logit diff against the Python reference has not been measured directly; the
  evidence is byte-identical greedy output, which is weaker.
- The host GCC here is newer than nvcc officially supports (built with `-fpermissive`).
- Single-GPU only.

Benchmark numbers in `PLAN.md` are annotated with the card they were measured on. They are
one machine's results, not a general claim.

## Build

Requires a CUDA toolkit and a PyTorch install to link libtorch against.

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1   # tch 0.26 expects a specific torch version
export CUDA_HOME=/opt/cuda
export LD_LIBRARY_PATH="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64"
cargo build --release
```

`TORCH_CUDA_ARCH_LIST` defaults to the compute capability of the GPU present; set it
explicitly to build for a different card. The first build compiles the CUDA tree and takes
a while.

The wrapper scripts set all of this up for you:

```bash
./run-infer.sh  --model /path/to/model-exl3 --prompt "The capital of France is"
./run-server.sh --config sample-configs/config.yml --host 0.0.0.0
```

## Quickstart: 3090/4090

Both a 3090 and a 4090 have 24 GB, so the same model and config work on either
card (the 4090 is just faster).

**1. Pick a model.** Qwen3.8-27B at 4-bit is ~17 GB of weights, leaving room
for a solid KV cache. Two good EXL3 quants:

- [`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw) —
  stock instruct model, calibrated on an agentic trace rather than generic text.
- [`Honkware/Qwen3.8-27B-heretic-ara-exl3-4.0bpw`](https://huggingface.co/Honkware/Qwen3.8-27B-heretic-ara-exl3-4.0bpw) —
  same base model with refusal directions ablated via [Heretic](https://github.com/p-e-w/heretic) (uncensored).

```bash
huggingface-cli download Honkware/Qwen3.8-27B-heretic-ara-exl3-4.0bpw \
    --local-dir models/Qwen3.8-27B-heretic-ara-exl3-4.0bpw
```

**2. Also grab the DFlash2 draft model** — a block drafter tied to this target
checkpoint, ~1.4 GB, good for a solid speedup over MTP:

```bash
huggingface-cli download Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw \
    --local-dir models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw
```

**3. Run the server** with the matching sample config, which sets `draft_mode:
dflash2` against that draft model plus vision for a 24 GB card:

```bash
./run-server.sh --config sample-configs/4090/config-dflash.yml --host 0.0.0.0
```

Edit `model.model_name` and `draft_model.draft_model_name` in
`sample-configs/4090/config-dflash.yml` to match whichever target/draft pair
you downloaded, and `model_dir` / `draft_model_dir` if you didn't put them
under `models/`. See that file and `sample-configs/config.example.yml` for
what every key does and how to tune `cache_size` if it doesn't fit.

## Binaries

| binary | purpose |
|---|---|
| `infer` | single-prompt generation; MTP, vision, cache quant, timing |
| `gen` | dynamic-batching generator across N concurrent jobs |
| `server` | OpenAI-compatible HTTP server (tabbyAPI config format) |
| `attn_check` | differential + timing harness for the attention, GDN and EXL3 kernels |

See [`sample-configs/`](sample-configs/) for server configs and what the keys mean.

## Licence and credit

MIT, same as upstream. All CUDA kernels, the EXL3 format and the original design are
turboderp's work — see [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3).
This repository tracks upstream v1.4.6.

DFlash2 support and the DFlash2 draft-model checkpoints referenced above
come from MiaAI-Lab's exllamav3 fork — see [MiaAI-Lab/exllamav3](https://github.com/MiaAI-Lab/exllamav3).
