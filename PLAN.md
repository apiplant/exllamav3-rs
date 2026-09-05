# exllamav3-rs — Rust port of ExLlamaV3 + DFlash

Goal: run EXL3-quantized models in Rust at native speed by compiling the original
CUDA kernels unchanged and driving them from Rust via `tch` (libtorch).

## Strategy

* **CUDA / C++** — *not rewritten*. The upstream `exllamav3_ext/**` tree is
  **vendored into the repo at `kernels/`** (was a symlink to the pip
  package) and compiled verbatim by `build.rs` (same flags as
  the upstream JIT builder in `ext.py`: `-O3 --use_fast_math -lineinfo`), linked
  against the libtorch that ships with the pip `torch` wheel (`tch` /
  `torch-sys`, `LIBTORCH_USE_PYTORCH=1`). A tiny C-ABI layer, `csrc/exl3_shim.cpp`,
  re-exports the handful of ops the inference path needs, taking `at::Tensor*`
  (ABI-identical to `tch`'s `*mut C_tensor`). Kernel numerics are therefore
  identical to Python by construction.
* **Logic files** — ported line-for-line, same math, same order of operations,
  documented per-file below with an accuracy grade.
* **Wiring files** (loader orchestration, generator loop, CLI) — reimplemented to
  be idiomatic Rust; only behavioural equivalence is promised, not structural.

## Accuracy grades

| grade | meaning |
|-------|---------|
| A | operation-for-operation identical to the Python source |
| B | same math, minor structural difference (fewer code paths, different fast-path gating) |
| C | behaviourally equivalent for the supported subset, diverges outside it |
| S | stub / not yet implemented |
| — | wiring file, equivalence-only by design |

## Ported files

### Kernels (compiled, not ported)

| upstream | rust | grade | notes |
|----------|------|-------|-------|
| `exllamav3_ext/**` (vendored → `kernels/`) | `build.rs` compiles a subset | A | verbatim compile; flags mirror `ext.py`. Compiled set: `quant/**`, `graph`, `norm`, `rope`, `hgemm`, `attention`, `activation`, `add`, `softcap`, `generator/{cache,rep_pen,sampling_basic,gumbel}`, `cuda_drv`. Dead pybind11/`Python.h` includes stripped from `graph.{cuh,cu}` so no binary needs libpython. |
| `exllamav3_ext/bindings.cpp` (pybind) | `csrc/exl3_shim.cpp` | — | C-ABI re-exports: `exl3_gemm`, `rms_norm`, `rms_norm_res_in`, `gated_rms_norm`, `rope`, `hgemm`, `bighead_attn`(`_paged`), `paged_kv_cache_update`, `softcap`, `add`, `silu/gelu/relu2_mul`, `apply_rep_pens`, `apply_pres_freq_pens`, `argmax_sample`, `gumbel_sample`, `gumbel_noise_f32`, `cache_rotate`, plus the CUDA-graph capture shim. Not exposed: MoE/CPU-offload, TP/`parallel/**`, quant conversion, gated-delta-net/mamba, multimodal, `stloader` jobs, `partial_strings_match` (needs `py::buffer`). |

### Logic

| upstream | rust | grade | notes |
|----------|------|-------|-------|
| `loader/safetensors.py` `read_header`/`validate_header`/`convert_dtype` | `src/safetensors.rs` | A | header size bounds, offset/shape/dtype cross-checks, `tensor_name_fixes` all ported. Load path uses mmap + `tch` `Tensor::f_from_blob` instead of the multithreaded `stloader` C++ (wiring). |
| `util/rope.py` `RopeSettings`, `RoPE`, `yarn_inv_freq`, all `_rope_params_*`, `compute_sincos` | `src/rope.rs` | A | every rope_type branch ported (default/linear/llama3/yarn/longrope/proportional). `apply()` calls the `rope` CUDA kernel with the same args. mrope: **B** — `RoPE::apply` takes an optional 2-D `[max_pos, rotary_dim/2]` angle table (the `rope` kernel already indexes it by `positions[b]+token_pos`); text MRoPE for Qwen3.5 vision built in `src/vision.rs` (`mrope_pos_ids`, `mrope_angle_table`). 1-D path unchanged when the table is `None`. |
| `model/config.py` `Config`, `read_dict` semantics | `src/config.rs` | B | generic `read_cfg` chain + `InferParams` defaults ported; only the fields the text path reads are surfaced. |
| `architecture/qwen3.py` `Qwen3Config`, `Qwen3Model` module tree | `src/config.rs` (`ArchParams`), `src/model.rs` (`build_qwen3`) | A | head_dim/kv fallback, tie_word_embeddings + `lm_head` alt-key, RMSNorm eps, NEOX rope, q/k norm, GatedMLP silu — all matched. |
| `modules/embedding.py` `Embedding.forward` (plain path) | `src/modules/embedding.rs` | B | `index_select` + optional `multiplier`/`normalize`; indexed/deepstack/pinned-staging paths omitted (multimodal). |
| `modules/rmsnorm.py` `RMSNorm.forward` | `src/modules/rmsnorm.rs` | A | calls `rms_norm` kernel, `constant_bias`/`constant_scale` passed through, 2-D flatten rule matched. `residual_in` fuse: **A** (`RmsNorm::forward_res` → `norm.cu` `RES_IN`, bit-identical to unfused add; Qwen3 path only, Qwen3.5 still separate). |
| `modules/linear.py` `Linear`, `modules/quant/exl3.py` `LinearEXL3`, `quant/fp16.py` `LinearFP16` | `src/modules/linear.rs` | B | pad-to-128, input zero-extend, bias, softcap, pre/post scale ported. EXL3 forward always takes the fused `exl3_gemm` path (upstream splits <144-row → GEMV, ≥ → reconstruct+hgemm; both are the same kernel family, numerically within trellis rounding). No LoRA, no fused Q/G·K/V MGEMM, no slicing/TP. |
| `modules/mlp.py` `GatedMLP.forward` | `src/modules/mlp.rs` | B | gate→`silu_mul`(fused kernel)→down, fp16 interm. No `intermediate_split_size` slicing, no bsz-1 `BC_MLP` graph fast-path (perf-only; result identical). |
| `modules/attn.py` `Attention.decode_flash_attn_nc` | `src/modules/attention.rs` | C | q/k/v proj, fused q/k RMSNorm+RoPE via `rope` kernel, GQA attention, o_proj. Two paths, selected by `modules::Attn`: **NoCache** (`bighead_attn`, matches `forward_last`) and **Paged** (`paged_kv_cache_update` + `bighead_attn_paged` over a real page table; `cache_seqlens` read on-device so one CUDA-graph capture replays at any offset). RoPE offset comes from the `positions` device tensor in the paged path. Both bit-identical to the no-cache result (bottom-right causal masking). Quantized KV, sinks, gates, sliding window, softcap: **S**. |
| `cache/cache.py` `Cache` + generator page table | `src/cache.rs` `PagedKvCache` | — | single sequence, contiguous page run (`block_table = 0..n`), one-element `cache_seqlens` (device int32). PAGE_SIZE 256 per the kernel. No eviction/rotation, no multi-seq. |
| `cache/quant.py` `CacheLayer_quant` | `src/cache.rs` `QuantPagedKvCache` / `src/paged.rs` `QuantPagedCache` | — | K/V stored as packed int32 codes (`qk`/`qv` `[pages,256,dim/32*bits]`) + fp16 group scales (`sk`/`sv` `[pages,256,dim/32]`), `dim = n_kv*head_dim`, `bits ∈ 2..=8`, linear quant (`compand_a = 0`). `Attn::PagedQuant` has two paths: the **single-seq / CUDA-graph** path (`infer`) folds fresh rows in with `quant_cache_paged`, bulk-dequants the prefix into a caller-owned fp16 scratch pool, attends fp16. The **batched** path (server / `gen`) does online in-kernel dequant — `bighead_attn_paged_q` → `attn_decode_split_kernel_q<D,G,BITS>` unpacks each KV tile warp-collectively (`dequant_block_x4`) into shared memory as it attends, **zero fp16 cache-sized scratch** (Python's Triton `_fns_qc` path); long-q prefill falls back to a compact `bsz*pages_per_seq` window scratch, freed per call. Wired into `infer --cache-bits N` / `--cache-mode`, the batched generator (`Generator::enable_cache_quant`, `Generator::new` takes `kv_bits`, `gen --cache-bits N`), and the **hybrid Qwen3.5 cache** (`Qwen35PagedCache` `KvQuant` variant — full-attn KV pages only, GDN recurrent state stays fp32; `cache_mode Q4/Q6/Q8`, composes with MTP + n-gram; MTP draft-head KV is Q8). Online kernel verified coherent on 27B (text/mtp/streaming/6.5k-ctx-recall/vision); 8-bit byte-identical to fp16 on the dense test prompt. |
| generator decode-graph capture (`generator/*.py` `Graph`) | `src/ffi.rs` `CudaGraph` + `csrc/exl3_shim.cpp` | — | raw `at::cuda::CUDAGraph` capture/replay of the whole decode step on a pooled side stream (not upstream's param-patching `Graph`). Dynamic position lives entirely in device tensors (`cache_seqlens` for the paged kernels + RoPE `positions`), so no per-step patching is needed. Warmup runs the step 6× first so `exl3_gemm` autotuning (a timing loop, capture-illegal) is locked before capture. `--no-graph` disables. |
| `modules/transformer.py` `TransformerBlock.forward` | `src/modules/transformer.rs` | B | pre-norm attn + residual, pre-norm mlp + residual — residual adds now **fused into the next norm** (`forward(resid, ctx, pending)` returns the deferred MLP output; the model loop threads it). Post-norms, hyperconnections, layer/resid scalars: **S** (none used by Qwen3). |
| `model/model_ls.py` `forward_ls` | `src/model.rs` `Model::forward` | — | single-device sequential module loop. autosplit / TP / deferred load: **S**. |
| `tokenizer/tokenizer.py` | `src/tokenizer.rs` | C | delegates to the `tokenizers` crate loading `tokenizer.json`; exllamav3's trie/added-token bookkeeping not reproduced. `pieces()` (lazy `decode([space,id])`-strip fixed-vocab, mirrors `_get_fixed_vocab`), `piece(id)`, `tokens_with_prefix(str)` ported for token healing. Fine for greedy/basic sampling. |
| `generator/generator.py` `Generator` + `generator/job.py` `Job` | `src/generator.rs` | — | **dynamic continuous batching**: N jobs share one `PagedCache`; each `iterate()` does one page-aligned prefill chunk per prompt-ingesting job, then one batched decode step (`forward_paged_batched`, per-row `block_table`/`cache_seqlens`) for every job past prefill. Per-job sampler + stop tokens + stop strings + min/max-new, streaming text events matching `iterate()`'s dict shape, free-list page reclaim, queue when the batch/cache is full. n-gram speculative decode, token healing (`JobSpec::token_healing`) and streaming loop detection (`JobSpec::stop_on_loop`) ported (see rows below). Done since: prefix-cache dedup (`pagetable.py`; full-page only), n-gram/MTP/separate-draft speculative decode, CFG, grammar/JSON-schema filters (hand-rolled, no kbnf/llguidance dep), async wrapper (`AsyncGenerator`), CPU cache offload, fair-scheduling requeue. **S**: partial-page prefix reuse, DFlash + dynamic draft length + confidence calibration, MRoPE in-batch, per-token top-k probs. Not CUDA-graphed (batch shape varies). |
| `exllamav3_ext/sam.{h,cpp}` `BC_SAM` | `src/sam.rs` `BcSam` | A | suffix automaton for n-gram drafting, 1:1 port (append-only + rebuild-on-shrink, `accept` / `accept_tensor` return the `[start,end)` span of the longest earlier-occurring suffix). |
| `generator.py` `iterate_ngram_gen` + draft-verify branch of `iterate_gen` + `job.py` `get_ngram_draft` | `src/generator.rs` `ngram_decode_round` / `Job::get_ngram_draft` | — | per-job `BcSam` proposes a draft (tokens after the earliest occurrence of the longest suffix, `>= ngram_match_min`); batch trimmed to the shortest draft (all-or-nothing per round, as upstream); one `q_len = min_len+1` paged forward verifies all rows; greedy longest-accepted-prefix, rejected positions leave stale K/V the next round overwrites. `enable_ngram(match_min, draft_tokens)`, `ngram_stats()`. **S**: dynamic/confidence truncation, draft stats list, filter advancing, requeue/rewind interplay. |
| `generator/loop_detect.py` `LoopDetector` | `src/loop_detect.rs` `LoopDetector` | A | flat-latency streaming loop detector (circular buffer + per-period heap-scheduled backlog scans); fires when the whole `window_size`-token window is one repeating sequence. Verified fire-point + period against the Python reference. `JobSpec::stop_on_loop = Some((window, min_reps))`, `gen --loop-window/--loop-reps`, eos reason `loop_detected`. |
| `job.py` token healing (`prefix_token` / `prepare_logit_mask` allowed-tokens / first-token strip) | `src/generator.rs` (`Job` heal fields) + `tokenizer.rs` | — | drop the last prompt token, additive `-inf` logit mask on step 1 restricting to `tokens_with_prefix(piece(dropped))`, healed token excluded from min/max-new accounting, `unhealed_piece` stripped from the first emitted text. n-gram speculation suppressed for the healed step. `JobSpec::token_healing`, `gen --token-healing`. |
| `generator/pagetable.py` page allocation | `src/paged.rs` `PagedCache` + `PageTable` | — | flat pool of 256-tok pages per layer + free-list allocator. No hash-indexed prefix sharing, no rotation, no CPU tier. |
| `generator/sampler/custom.py` `SS_*` chain | `src/sampler.rs` `SamplerSettings` | C | ported stages in upstream order: rep penalty (`SS_RepP`), presence/frequency (`SS_PresFreqP`), temperature, top-k, top-p, min-p, then argmax / categorical. Penalties now use the `apply_rep_pens` / `apply_pres_freq_pens` **CUDA kernels** on CUDA with the real `sustain_range` / `decay_range` windowing (CPU fallback keeps the full-history HashMap path); `gen` exposes `--pres-penalty --freq-penalty --sustain --decay`. **Diffs**: `multinomial` instead of the exact Gumbel-noise kernel (RNG not bit-identical — upstream disclaims determinism too; `gumbel_sample` is now wired through `ffi` but the chain doesn't use it yet); DRY/XTC/mirostat/typical/quadratic/adaptive-p/skew/grammar not ported. |
| `generator/*.py` (single-stream) | `src/bin/infer.rs` | — | greedy + temp/top-k/top-p loop, KV-cached prefill + 1-tok decode, CUDA-graphed (`--timing` reports tok/s, `--no-graph` disables). Tokens stream to stdout, stats to stderr at the end (matches `py-infer.py`). Model load shows an animated real-progress bar (`Model::load_with_progress` fires per transformer layer). Also drives the Qwen3.5 MTP path (`--mtp`, `run_mtp`) and the vision path (`--vision`, `vision::run_infer`). tabbyAPI knobs: `--max-seq-len --rope-scale --rope-alpha --chunk-size --cache-mode --reasoning/--no-think --draft-n`. |
| — (demo) | `src/bin/gen.rs` | — | multi-prompt concurrent driver for `Generator` (`--prompt` repeatable, `--max-batch`, `--pages`, `--stream`). |
| TabbyAPI `endpoints/OAI/**` + `config.yml` schema | `src/server/**` + `src/bin/server.rs` | — | OpenAI-compatible HTTP server (ntex + flume). One engine thread owns the Model / `Generator` / MTP head / vision tower. Endpoints: `GET /v1/models`, `GET /health`, `GET /v1/internal/model/info`, `POST /v1/chat/completions`, `POST /v1/completions` — streaming (SSE) + non-streaming, API-key auth, permissive CORS. `config.yml` parsed by `server/config.rs` (every documented key accepted; honored subset: model_dir/model_name, max_seq_len/cache_size, cache_mode, chunk_size, rope_scale/rope_alpha, max_batch_size, vision, reasoning + tokens + start_in_reasoning, tool_format, template_vars_*, draft_mode (`mtp`/`ngram`/`disabled`), draft_num_tokens, ngram_match_min, sampling.override_preset, network.*, developer.disable_request_streaming). ChatML prompt rendering is hand-rolled (`server/chat.rs`, no Jinja engine) — covers the Qwen family + `<tool_call>` JSON extraction + `<think>` reasoning split. Text (batched / mtp / ngram / draft) all flows through `Generator::iterate`. **Vision**: an image request runs single-stream via `Generator::mm_generate` — vision-tower embeds spliced into the input stream, MRoPE angle table, KV pages drawn from the **shared paged pool** (no dedicated vision cache; `Model::forward_qwen35_batched_embed`), blocking the batch for its duration. Matches upstream, where image jobs go through the normal generator — `vision: true` only adds the ~0.9 GB ViT. **S**: separate draft *model* (use mtp/ngram), Jinja templates, Harmony/Muse formats, multi-image / http(s) image URLs, batched (non-blocking) multimodal, runtime model hot-swap / `inline_model_loading`, `n` > 1, LoRA, embeddings endpoint, Kobold API. |

## Not started

MoE / block-sparse, MLA attention, Mamba2 (gated-delta-net + short-conv **are**
done for Qwen3.5), all non-Qwen3/Qwen3.5 architectures, quantization/conversion
(`conversion/**`, `quantize.cu` driver), tensor-parallel (single GPU here), LoRA.

**DONE — Qwen3.5 (`qwen3_5`) for Qwen3.8-27B**: hybrid 64-layer model, 3×
`linear_attention` (gated-delta-net: short conv + recurrent SSM state, `gdn.cu`
kernels) then 1× gated `full_attention` (head_dim 256, partial-rotary 0.25,
`attn_output_gate`), interleaved MRoPE (`mrope_section [11,11,10]`), MTP head,
vision tower. KV cache quant (done) reduces the full-attn layers' footprint; the
27B-4bpw weights (~14 GB) fit one 4090. Text inference (`infer` + batched `gen`),
MTP self-speculation, and the vision tower all run end-to-end and verified
byte-identical to the upstream reference. Hybrid KV-cache quant is done —
`cache_mode Q4/Q6/Q8` quantizes only the full-attn KV pages (GDN recurrent state
stays fp32); verified on 27B/4090 (Q6 loads a 49k-token pool at 22 GB that would
OOM at fp16, coherent, composes with MTP). Still **S**: MoE variant of the arch,
MTP inside the batched `Generator` (works single-stream via `run_mtp` / the
server's mtp engine mode).

Simplifications taken (all bit-exact for a text-only sequence): text MRoPE for
pure-text prompts == standard partial NEOX RoPE with `rotary_dim =
head_dim*partial_rotary_factor = 64` (equal t/h/w positions ⇒ interleaving is a
no-op; the angle-table MRoPE path is used only when an image is present); the
sequential `cuda_recurrent_gated_delta_rule` kernel serves both prefill and decode
(no fla/triton chunked path).

Phases: [x] 1 — config (`ArchKind`, `text_config` unwrap, `layer_types`,
`GdnParams`, `key_prefix`), `gdn.cu` compiled, shim/ffi for
`gated_delta_net_fused_op_2`, `cuda_recurrent_gated_delta_rule`,
`cuda_causal_conv1d_update`, `mul_sigmoid_`, `deinterleave_qg`. Qwen3 build still
green. [x] 2 — gated full-attention: `modules::Attention` gained `output_gate`
(interleaved `[q|gate]` q_proj via `deinterleave_qg`, `o *= sigmoid(gate)` before
o_proj) + `norm_cbias` (feeds the fused Q/K RMSNorm `constant_bias`); partial
RoPE 64 already handled by `RoPE::new` from `cfg.rope`. [x] 3 — `src/qwen3_5.rs`:
`GatedDeltaNet` (split in_proj_qkv/z/b/a, `gdn_fused_op_2` → conv1d+SiLU →
`recurrent_gated_delta_rule` → gated RMSNorm(gate=z) → out_proj) + `GdnState`
(conv_state bf16 `[1,fdim_qkv,K]` + recurrent_state f32 `[1,1,Nv,khd,vhd]`).
Compiles. [x] 4 — hybrid assembly: `Model` gained `q35_layers: Vec<Q35Layer>`
(`{input_norm, attn: Q35Attn::{Full(Attention)|Linear(GatedDeltaNet)}, post_norm,
mlp: GatedMlp}`); `Model::{forward,prefill}_qwen35` against `cache::Qwen35Cache`
(`Vec<Q35LayerCache::{Kv{k,v}|Gdn(GdnState)}>` + shared block_table/seqlens).
`Model::load` branches on `cfg.arch_kind`. `Config::kv_heads_eff()` → the paged
attn kernels only allow GQA ratio ∈{1,2,4,8}; Qwen3.5's 24/4=6 is repeated to
24/12=2 (`Attention.kv_repeat`, `repeat_interleave` after RoPE; cache pools sized
12). [x] 5 — `infer --model <27B>` runs end-to-end on the 4090: coherent output
("The capital of France is Paris. …", chat + `<think>` too), ~12.3 tok/s decode
eager, prefill 5.3 tok/s. Also fixed a false-positive assert in `ffi::exl3_gemm`
(rc==0 is the int8-GEMV fast path = success, not "no compatible shape";
default-on for `mul1` tensors, which the 27B has). [x] 6 — batched generator:
`GenCache::Qwen35(paged::Qwen35PagedCache)` (`Q35BatchLayer::{Kv|Gdn}` per layer;
GDN pools `[max_slots, …]` indexed by a per-job recurrent slot); `Generator`
gained `slots_free` (free-list `0..max_batch`), `Job.slot`, slot alloc + zero in
`admit_jobs` / release on reap; `Model::forward_qwen35_batched` (full-attn →
`Attn::Paged`, GDN → `GatedDeltaNet::forward` with `slots`); chunked prefill and
batched decode both pass a `slots` tensor. n-gram spec decode (initially asserted
off) and hybrid KV-cache quant now both work on the Qwen3.5 cache. Verified: `gen` 2 concurrent jobs coherent,
single-job `gen` == `infer` greedy (byte-identical continuation). ~8.7 tok/s agg
for 2 jobs.

Post-port work:
[x] KV-repeat trim — `kernels/attention.cu` now instantiates the paged +
non-paged attn kernels for GQA ratios 3/5/6/7 at head_dim 128/256 (the kernel
body is already generic over `G`). Qwen3.5's 24/4 = 6 runs native, 4 KV heads,
no repeat — 3× less KV cache for the full-attn layers. `kv_heads_eff()` widened.
[x] CUDA-graph the hybrid decode — capture works (int8-GEMV path captures fine),
output correct. (Originally measured ~0 speedup at 12 tok/s; that was the
`force_num_sms = -1` shim bug — see the fixed item in the build-status list.
27B-4bpw decode is now ~51 tok/s.) Left on-by-default (`--no-graph` disables).
[x] **Prefill speed — fixed, 75×.** Root cause was NOT the GDN recurrent kernel
(that's ~2 s of a 9 s prefill and mostly its projections). It was `Linear` always
using the EXL3 trellis GEMM, which re-derives the weight per m-tile and gets ~no
batching benefit. `modules::Linear::forward` now mirrors
`quant/exl3.py::reconstruct_hgemm` (non-fused): for `rows > 144` (and
`out_features <= 32768`, unpadded), `had_r_128(x)` → `reconstruct(trellis)` to a
dense fp16 weight once → `hgemm` → `had_r_128(y)`. New shim/ffi: `had_r_128`,
`reconstruct`. Result: 27B 601-tok prefill **41 s → 0.55 s (1086 tok/s)**; Qwen3
0.6B 401-tok prefill 520 → 3197 tok/s. Output coherent, decode unchanged.
(A chunked-parallel GDN delta-rule kernel was written and then removed — GDN
recurrence isn't the bottleneck, and it couldn't be validated against the
sequential path, which decode still relies on.)
[x] GDN history/rewind — n-gram speculative decode now works for Qwen3.5.
`GatedDeltaNet::forward` gained a `history` flag (→ the recurrent + conv kernels'
history modes: per-token recurrent snapshots + full conv stream in the buffer
tail). `paged::Qwen35PagedCache::new_hist(max_history)` reserves the extra planes
(`recurrent_state [slots, max_history+1, …]`, `conv_state [slots, fdim, K +
max_history]`); `enable_ngram` rebuilds the Qwen3.5 cache with `max_history =
draft+1`. After a speculative forward, `gdn_rewind(slot, keep, consumed)` copies
`recurrent_state[slot,0] <- [slot,keep]` and ALWAYS restores the conv window
`[:, :K]` from `[:, p-K:p]` (`p = (K+max_history) - (consumed-keep)`) — the
history conv writes the fresh window to the tail, never to `[:, :K]`. Verified
byte-identical to plain decode at 0 / 40 / 90 % acceptance, batch 1 and 2;
repetitive prompt 12 → 137 tok/s (11×) on the 27B. VRAM ≈ 3 MB × (draft+1) ×
max_batch × #linear-layers extra — batch 1 ≈ 900 MB, scales with `max_batch`.
[x] **Long-q attention through `bighead_attn_paged` — crash fix + prefill/decode
speedup.** The vendored split-K wrapper sizes an fp32 partials buffer in the fixed
16 MB device workspace and *doubles* `kv_chunk_size` until it fits — an infinite
loop when a single chunk (`n_chunks == 1`) still overflows, i.e. `q_len *
n_q_heads * (dim+2) > 4 M` (~2 k-token prefill on Qwen3-0.6B). It manifested as
`GPU assert: invalid argument attention.cu` once the doubled `kv_chunk_size`
overflowed i64. Fix: the sizing loop now terminates at `n_chunks == 1`, and that
case takes a new `single` path — one CTA does the whole causal reduction in
registers (it already did) and writes `o` directly, skipping the workspace
round-trip *and* the separate `attn_reduce` launch. Applies to
`attn_chunked_paged_kernel` + `attn_chunked_kernel` (head_dim 64/128/256; the
512×256 variants keep the old path and are guarded). Verified byte-identical to
Python greedy on a 3055-token prompt (Qwen3-0.6B and 27B); short-context decode
also hits this path now (n_chunks==1) — infer 0.6B ~260→300 tok/s. The proper
flash-decode paged kernel (tiled KV inner loop / tensor cores, split-K by real
seqlen not padded page capacity) is still the real long-context decode fix — see
`decode_flash_attn` / `fn_triton_paged_attn_decode` upstream; `TritonKernel` AOT
infra is vendored (`kernels/triton_kernel.*`) but unwired.
[x] **Flash-decode paged-attention kernel.** `kernels/attention.cu`
`attn_decode_split_kernel<D,G>` — CUDA port of upstream
`triton_paged.py::_paged_attn_decode_split_kernel` (+ combine, for which the
existing `attn_reduce_kernel` is reused). `bighead_attn_paged` routes `q_len <= 16`
(decode + n-gram verify), head_dim 64/128/256, to it; large-q prefill keeps the
chunked+`single` path. One CTA per (batch, kv_head, q_pos, kv-split); the KV range
is walked in `DecTile<D>::N`-key tiles (32 for d64/d128, 16 for d256) staged
through shared memory, **double-buffered with `cp.async`** (`__pipeline_memcpy_async`
/ `_commit` / `_wait_prior`): tile *t+1*'s K+V load overlaps tile *t*'s compute.
Vectorised `half2`; **one `__syncthreads` per tile** (vs the scalar kernel's
two-per-*key*); online-softmax accumulator per GQA sibling head; out-of-range tail
positions clamp onto the last valid token (masked by the score loop bound).
`num_splits` sized on the host to ~2× the SM count (`min(2·SMs/programs,
maxk/(4·32), 128)`, clamped to the fp32 workspace), so long-context decode keeps
every SM busy. `num_splits == 1` writes `o` directly (no combine pass).
Verified byte-identical to Python greedy over a 150-token generation and on a
3055-token prompt; ngram == plain decode byte-identical; gen 4-job coherent
(990 tok/s agg); 27B (G=6, d256) coherent.
Decode raw-replay ms/step (Qwen3-0.6B-8bpw / 4090, was scalar → tiled → +cp.async):
~1.5k ctx **5.9 → 3.0 ms** (89 → 336 tok/s), ~4k **~3.2 ms**, ~8k **crash → 3.9 ms**
(255 tok/s). At parity with Python's Triton flash-decode (~3.0→3.3 ms over
0→1k). Decode is now HBM-bandwidth bound (KV re-read per step); tensor cores
would be near-zero gain at GQA 2/6 (mostly-padded `mma` tiles) — not pursued.
[x] **MTP self-speculative draft head (Qwen3.5).** `src/mtp.rs` `MtpModel` —
port of `architecture/qwen3_5_mtp.py` + `modules/arch_specific/qwen3_5_mtp.py`.
Shares the trunk's `embed_tokens` / `lm_head`; own 1-layer stack (2× pre-fc
RMSNorm(bias 1) + `fc` 2H→H Linear @ `mtp_bits`, then input_layernorm →
gated GQA `self_attn` (own `MtpKvCache`, head_dim 256, GQA 6, interleaved
`[q|gate]`, q/k-norm, partial RoPE 64) → post_attention_layernorm → GatedMLP →
`mtp.norm`). `bin/infer --mtp [--draft-n N]` (default N=4): one-shot prefill
exports the trunk's post-final-norm hidden (`Model::forward_qwen35_batched_h`),
primes the MTP KV; each round the MTP head **chains N steps** to draft N tokens
(feeding its own output hidden + drafted token back, entirely on-device — no
per-step host sync), then one `q_len = N+1` trunk forward through the batched
**history** hybrid cache (`Qwen35PagedCache::new_hist`, `gdn_rewind(0, l, N+1)`)
verifies them greedy, accept-longest-prefix. `sync_after_accept` re-runs the MTP
head over the accepted positions with the verify forward's *real* trunk hiddens.
One device→host copy per round (drafts ++ trunk argmax). Eager only (no graph).
Verified **byte-identical to plain greedy** over 120–240 tokens on Qwen3.8-27B;
py-infer `--mtp` (upstream `draft_model`) produces the same text.
Speed (27B-4bpw / 4090): baseline 52 tok/s → **85–105 tok/s at N=4** (~1.6–2×,
prompt-dependent — 42–57 % acceptance), matching/beating upstream py-infer
(~96–98). `py-infer.py` gained `--mtp` + `--vision` + all the context/RoPE/cache
flags for parity, and reports the same accepted/total + prefill/decode stats.
[x] **infer/py-infer CLI: tabbyAPI-style knobs.** `--max-seq-len` (context /
cache size), `--rope-scale` (linear) + `--rope-alpha` (NTK base) via new
`config::ConfigOverrides` → `Model::load_with`, `--chunk-size` (chunked
prefill — `bin/infer` was one-shot), `--cache-mode FP16|Q8|Q6|Q4` (alias for
`--cache-bits`), `--reasoning`/`--no-think` (empty `<think></think>` block in
the chat template), `--mtp`, `--vision <img>` (rejected until the vision
phase). `py-infer.py` mirrors every flag.
[x] **Vision tower (Qwen3.5 multimodal).** `src/vision.rs` — port of
`architecture/qwen3_vl.py` `Qwen3VLVisionModel` + `modules/arch_specific/qwen3_vl.py`.
Vision weights are dense bf16 (this checkpoint doesn't quantize them), run once
per image, so the whole tower is plain `tch` math — no CUDA kernels:
preprocess (`qwen2_smart_resize` + normalize + the 9-D patchify reshape) →
`patch_embed` (Conv-flat == Linear 1536→1152) → bilinear-interpolated learned
pos-embed (`fast_pos_embed_interpolate`) → 27× { LayerNorm, non-causal MHSA with
2-D RoPE (`qwen2_position_embedding_grid_2d`, head_dim 72, plain fp16 softmax),
LayerNorm, MLP gelu-tanh } → merger (LayerNorm, 2×2 shuffle, Linear
4608→4608→gelu→5120) → `[N/4, 5120]` image embeddings.
**MRoPE**: the vendored `rope` kernel already accepts a 2-D `[max_pos,
rotary_dim/2]` f32 *angle table* (indexed by `positions[b] + token_pos`), so
text MRoPE needed no shim/kernel change — just a device angle table threaded
through a new `Attn::Paged { rope_table }` field and `RoPE::apply`'s
`freq_table` arg. `vision::mrope_pos_ids` ports `rope.cu::gen_mrope_pos_ids`
(text tokens t=h=w contiguous; image span uses the grid h/w), `mrope_angle_table`
builds the `[max_len, 32]` table (rows past the prompt continue the plain ramp,
so decode is standard). `Model::forward_qwen35_mm(embeds, cache, rope_table)`
takes the spliced text+image embedding stream.
`infer --vision <img> --prompt "<image>\n..."`: splice = `[…text, vision_start,
img_emb×N, vision_end, …text]`. Verified on Qwen3.8-27B: cat.png → *"a small
kitten wearing a colorful knitted hat"* (upstream py-infer: *"…crocheted hat"* —
1-synonym diff at tok ~10, from CatmullRom-vs-Pillow-BICUBIC resize + fp16 ViT).
strawberry.png → correct. ~49 tok/s eager. One image per prompt; deepstack
(`deepstack_visual_indexes`) and video not ported (both empty/unused here).
`py-infer.py` gained `--vision` (upstream `vision` component + `get_image_embeddings`).
[ ] MoE variant.

Generator gaps (v1 is a working core — see the table row): DFlash speculative
decode, recurrent-state checkpoints beyond the prefix-cache boundary, MRoPE in
the batched path, per-token top-k probs, partial-page prefix reuse, the exact
Gumbel sampler RNG + DRY/XTC/mirostat/typical/quadratic/adaptive-p stages.
Done: n-gram / MTP / separate-draft-model speculative decode, token healing, loop
detection, **prefix-cache dedup** (`src/paged.rs` chained-hash page registry +
refcounts + reclaimable-LRU; GDN-state checkpoints so hybrid targets prefix-share
too), **fair-scheduling requeue** (`enable_requeue`; `stream_prompt_len` keeps it
invisible to the client), **CFG** (`JobSpec.cfg`, per-job positive+negative
forward + logit mix; non-hybrid), **grammar / JSON-schema constrained decoding**
(`src/filter/` — hand-rolled GBNF-subset engine + JSON-Schema→GBNF compiler, **no
kbnf/llguidance/formatron dep**; `response_format` / `guided_json` on the server),
**CPU cache offload** (`src/cpu_cache.rs` pinned host tier restored on prefix
hit), **async wrapper** (`src/async_gen.rs` `AsyncGenerator`; the server engine
loop is now an async task on a `current_thread` tokio runtime + `LocalSet`).
Progress: sustain/decay-windowed penalties now run on the real CUDA kernels
(`apply_rep_pens` / `apply_pres_freq_pens`); `gumbel_sample` / `argmax_sample` /
`cache_rotate` are wired through `ffi` ready for the exact-RNG sampler and
prefix-cache rotation work.

## Build

```
cd crate
export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9      # match your GPU
cargo build --release                                   # ~9 min first time (~140 .cu files)

export LD_LIBRARY_PATH="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$CUDA_HOME/lib64:$(python3 -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')"
./target/release/infer --model ../models/Qwen3-0.6B-exl3-8.0bpw_H8 \
    --chat --prompt "Name three primary colors." --max-new 60
```

Requires: CUDA toolkit (`nvcc`) in PATH, a pip `torch` (2.12 / cxx11-ABI here), an
NVIDIA GPU. `build.rs` compiles each `.cu` whole-program (no device-link — the
EXL3 comp_units carry external-linkage `__device__` globals) and links the pip
libtorch + CUDA runtime/driver only. Kernel sources are vendored at
`kernels/`; the `../exllamav3` symlink is kept solely as the Python
reference. libpython is **no longer** needed (dead pybind11/`Python.h` includes
were removed from `graph.{cuh,cu}`).

### Build status  ✅ working end-to-end

- [x] kernel subtree (`quant/**` + norm/rope/hgemm/attention/graph/cuda_drv) compiles
- [x] shim links against pip libtorch 2.12 (cxx11 ABI, host gcc 16 via `-fpermissive`)
- [x] `infer` generates coherent text from Qwen3-0.6B-exl3-8.0bpw on an RTX 4090
- [x] Paged KV cache: O(n) decode. Same output.
- [x] CUDA graph capture of the decode step (paged kernels + on-device seqlens; `--no-graph` to disable). Correct output.
- [x] **exl3_gemm shim was passing `force_num_sms = -1`** (meant "use device SM count", but the kernel treats any non-zero value literally). This collapsed the cooperative-GEMM grid to a single SM (`MAX(MIN(max_slices, num_sms), 1)`), running every m=1 decode matmul on 1/128th of a 4090. Fixed to `0` (matches upstream `BC_LinearEXL3::run_gr`). **Qwen3-0.6B-8bpw decode 24 → ~260–280 tok/s (infer), ~228 tok/s (gen 1 job); 27B-4bpw decode 12 → 51 tok/s.** CUDA graph now also matters again (~4.9 ms/step replay vs ~5–6 eager on 0.6B). Was never a kernel/bpw limitation.
- [ ] fp32 residual stream is **correct** (matches Python `transformer.py`), not a perf bug — removed from the headroom list.
- [x] dynamic-batching `Generator` (`src/generator.rs`, `bin/gen.rs`): N jobs / one paged cache, chunked prefill + batched decode, per-job sampler + stop conditions + streaming. Single-job output matches `infer` byte-for-byte. Qwen3-0.6B-8bpw/4090 (after the force_num_sms fix + decode-loop cleanup below): 1 job ~210, 4 jobs ~650, 8 jobs ~1440 tok/s aggregate.
- [x] Decode-loop host-overhead cleanup (`generator.rs` `decode_round`, `sampler.rs`): (2) persistent `[max_batch,…]` device scratch for ids/block_table/seqlens/slots, written per step with one `copy_` each instead of a fresh `to_device` alloc; block_table now passed at full pool-page width (kernels derive pages-per-seq from `size(1)`). (3) all-greedy batches take one batched `argmax` + a single device→host copy instead of `bsz` separate `.int64_value()` syncs — this is the bulk of the multi-job speedup. (4) incremental (vLLM-style two-window) detokenisation via `Job::extend_detok` / `gen_text`, with a one-shot full decode on eos so `full_text` stays byte-identical. (5) `sampler::sample` skips the redundant f32 copy when already f32. Greedy output unchanged; streamed text == final text.
- [x] Per-layer op-fusion pass (`modules.rs` + `model.rs`), closing the ~1.2 ms/tok fixed gap vs python (both cuda-graphed, same GEMMs; python fuses via C++ `BC_*` ops). Three changes, all preserving greedy output on short generations: (1) `Linear::forward` no longer re-copies an already-fp16 decode input (`to_kind(Half)` was running 7×/layer/tok). (2) `GatedMlp::forward` uses the fused `ffi::silu_mul` kernel (3 tch ops → 1, matches upstream `BC_GatedMLP`). (3) **Residual-add→next-norm fusion**: `RmsNorm::forward_res` calls `norm.cu`'s `RES_IN` mode (`r += add` in place, rounded exactly as an unfused add, then `norm(r)`); the Qwen3 `Model::forward_*` loops now carry a deferred `pending` sublayer output that each block's `forward(resid, ctx, pending)` folds into its attn-norm, and `head_res` / `finalize_head` fold the last one — eliminating both standalone `x + y` residual kernels per layer. Net: **`infer` 0.6B-8bpw short-context 279 → ~336 tok/s (python ~332)**; `gen` 1-job ~239, 8-job ~1600 agg. 27B (Qwen3.5 path) untouched — still 51 tok/s. Verified: `EXL3_NOFUSE_RES` A/B shows the fused RES_IN path is bit-identical to the unfused add. NOTE: `infer` and single-job `gen` can diverge on *long* greedy generations (~byte 400+ on rare prompts). Root cause confirmed: `infer` prefills all N prompt tokens in one `q_len=N` causal-attention call; the generator (like upstream `job.py`) holds the last prompt token back and computes its K/V as a `q_len=1` decode step — a different attention kernel, ~1e-4 fp16 delta on that one token's cached K/V, which tips a near-tie ~40 tokens later. `EXL3_PREFILL_HOLDBACK=1` makes `infer` mimic the generator → then byte-identical to `gen` on every tested prompt. Not the fusion, not GEMM autotune (0.6B m=1 is deterministic int8-GEMV); both paths are valid greedy. The remaining long-context perf gap is the attention kernel (see below).
- [x] n-gram speculative decode (`src/sam.rs` `BcSam` grade-A port of `BC_SAM`, `Generator::enable_ngram` / `ngram_decode_round`, `gen --ngram-min N --ngram-draft K`). Greedy output bit-identical to plain decode; ~19–38% draft acceptance on repetitive/code prompts, marginal on free-form reasoning. Multi-job verified.
- [x] CUDA kernel tree vendored into `kernels/` (was a symlink); `build.rs` compiles from there. libpython dependency removed (dead pybind11 includes stripped from `graph.{cuh,cu}`).
- [x] Kernel surface widened: shim + `ffi.rs` now expose activation muls, `add`, `softcap`, `rms_norm_res_in`, `gated_rms_norm`, the rep/pres/freq penalty kernels, `argmax_sample` / `gumbel_sample` / `gumbel_noise_f32`, `cache_rotate` (compiled set extended in `build.rs`).
- [x] Sampler penalties run on the `apply_rep_pens` / `apply_pres_freq_pens` CUDA kernels with sustain/decay windows (`SamplerSettings::{sustain_range,decay_range}`, `gen --pres-penalty/--freq-penalty/--sustain/--decay`). Greedy/`infer` output unchanged.
- [x] `infer` output: tokens stream straight to stdout, stats to stderr at the end (matches `py-infer.py`); the old in-place `LiveView` token counter that rewrote scrollback is removed. Model load draws an animated colored progress bar driven by real per-layer load progress (`Model::load_with_progress`).
- [x] Generator gaps — token healing (`JobSpec::token_healing`, `gen --token-healing`: drops last prompt token, masks step 1 to prefix-sharing tokens, strips the unhealed piece from output) and streaming loop detection (`src/loop_detect.rs` grade-A port of `LoopDetector`, `JobSpec::stop_on_loop`, `gen --loop-window/--loop-reps`, eos `loop_detected`). Verified against the Python reference + on Qwen3-0.6B (healing completes `"Par"`→`"Paris"`; loop detector fires on a `ha ha…` prompt, no false trigger on normal text).
- [x] KV cache quantization (`cache/q_cache.cu` kernels exposed; `QuantPagedKvCache` / `QuantPagedCache`, `Attn::PagedQuant`; `infer --cache-bits N`, `gen --cache-bits N` / `Generator::enable_cache_quant`). 8-bit byte-identical to fp16; 2–5 bit coherent with graceful degradation. Also wired for the **hybrid Qwen3.5 cache** (full-attn pages only, GDN state fp32). **In-kernel online-dequant** for the batched path (`attn_decode_split_kernel_q`, `bighead_attn_paged_q`) — no fp16 cache-sized scratch, decode + spec-verify; long-q prefill uses a compact per-call window. The batched path now attends **straight off the packed codes by default** via the query-tiled `attn_prefill_paged_kernel_q<D,G,BITS>` (each KV tile dequantized once into shared memory and shared across the CTA's 8 query warps x G heads) — nothing materializes an fp16 KV window, which was costing ~3.3 GB of write+read (~21 ms) per decode step at 50k context. `EXL3_KVQ_WINDOW=1` forces the old bulk-dequant path; `attn_decode_split_kernel_q` still serves q_len==1. Verified on 27B/4090: tabby Q4 + mtp + vision loads the **full 204800-token pool** (22.5 GB, peak 23.3 GB at 64k context); text/streaming/vision/needle-recall all correct.
- [x] OpenAI-compatible HTTP server (`src/server/**`, `bin/server.rs`) — loads the real tabbyAPI `config.yml` as-is; verified on Qwen3.8-27B (chat non-stream + SSE, stop strings, `tool_format: qwen3_coder` tool calls, vision via data: URI, `/v1/models`) and 0.6B (batched path, 3 concurrent requests, 400 handling). Text (batched/mtp/ngram/draft) all through `Generator::iterate`; image requests single-stream via `Generator::mm_generate` on the **shared KV pool** (no dedicated vision cache — matches upstream). Real tabby config (Q4 + mtp + vision + 204800) loads at ~21 GB and stays flat through text + vision, matching Python tabby's ~21.65 GB.
- [x] **GDN-aware prefix cache** for the Qwen3.5 hybrid arch (`prefix_cache: true`) — a follow-up turn reuses the KV pages **and** restores the GDN recurrent/conv state from a checkpoint taken at the shared-prefix boundary page (`Qwen35PagedCache::gdn_snapshot`/`gdn_restore`, LRU keyed by chained page hash), then prefills only the new user turn. MTP priming made prefix-aware too (`prime_row(.., base, ..)` over the tail only). Verified on 27B: follow-up turns **7.8 s → 0.6 s** with mtp (**3.4 s → 0.4 s** without) at ~4k context, answers correct incl. recall of a fact buried at the start of the cached prefix. Streaming tool-call XML no longer leaks (held back like `<think>` until the block completes → returned as `tool_calls`).
- [x] **Perf/VRAM pass vs the Python port** (27B/4090, benchmarked against a live tabbyAPI run of the same model+config). Three root causes found and fixed: (1) `prime_spec` concatenated every prefill chunk's trunk hidden into one `prompt_len`-wide MTP forward — ~4 GB at a 19k prompt and growing with context, which is what actually OOM'd long prompts; the MTP head is now primed per prefill chunk (`Generator::mtp_prime_chunk`, one-token `Job::mtp_carry_h`, `MtpModel::prime_row(.., carry, ..)`), working set O(chunk_size). (2) The hand-tuned `hybrid_cap` pool constant is gone: `Generator::new` takes a placeholder pool and `Generator::resize_pool` fills the VRAM actually left after the vision tower / MTP / draft model load, from `cuda_free_mib()` minus a chunk-scaled activation reserve (`EXL3_VRAM_RESERVE_MB`) over `pool_bytes_per_token()` — so `cache_size: 204800` is honored instead of being clipped to 114688. (3) Prompt ingestion ran on a *decode* kernel (`attn_chunked_paged_kernel`: one CTA per query, one KV position per iteration, 4 barriers each, ~2.3 TFLOP/s) — replaced by `attn_prefill_paged_kernel<D,G>`, 8 query warps per CTA with KV staged in shared memory and split-K so it also serves the q_len 2..16 speculative verify. Net, ttft/decode: ctx 16k 9.7 s/64 tok/s -> **9.0 s/78**, ctx 32k 25.1/57 -> **22.2/63**, ctx 64k 71.0/42 -> **59.8/58**, with 204800 loading where it previously would not. Remaining gap to Python is entirely attention FLOP/s (their Triton kernels use tensor cores; ours is a scalar shuffle-reduce loop at ~10-15 TFLOP/s, and profiling says it is latency-bound rather than issue-bound, so an mma.sync flash kernel is the next real project). The context-free trunk already matches Python (~1650 tok/s at chunk 2048).
- [x] **Tensor-core attention** (`kernels/attention_mma{,_kernel}.cuh`, `attn_flash_mma_impl<D,G,QUANT,BITS>`). The scalar query-tiled kernels ran ~10-15 TFLOP/s on an RTX 4090 doing per-lane FMAs plus a 5-step shuffle reduction; this does both the QK and PV products with `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, off the fp16 cache **or** straight off the packed quant codes (a KV tile is dequantized once into shared memory and consumed by the whole CTA). Rows of the mma M dimension are (query, query-head) pairs — every query head sharing a kv head attends over the same K/V, so they stack into one tile, which is what makes a 5-token speculative verify fill its tiles instead of using 5 of every 16 rows. A CTA covers RT=6 row tiles x PAIR=D/128 warps, keeping the O accumulator at 64 registers per thread for any head dim; split-K over the KV range reuses the existing `attn_reduce_kernel`. The S accumulator's two n8 halves are exactly the A fragment of the following PV product, so P never round-trips through shared memory. On by default; `EXL3_MMA_ATTN=0` restores the scalar path. `bin/attn_check` (`./run-attn-check.sh`) diffs both kernels over identical inputs across 12 shapes x {fp16, Q4, Q8} — all agree to fp16 rounding (max 1e-3 relative). Measured on the 27B (Q4 + mtp + vision, 204800 pool, temperature 0), ttft/decode: 16k 9.09s/74 -> **7.48s/77**, 32k 22.3/65 -> **16.4/72**, 64k 59.6/57 -> **37.4/64**; the isolated 2048q x 48000ctx layer goes 155ms -> 53ms.
- [x] **Qwen3-Coder tool-call format.** `tool_format: qwen3_coder` advertised the Hermes `{"name":…,"arguments":…}` JSON in the prompt and parsed only that, but this model's own `chat_template.jinja` uses pseudo-XML (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`). The model followed its training part of the time, and those calls were dropped or surfaced as a nameless call — the intermittent `{}` / "Tool not found" that the Python port never produced. `chat.rs` now renders the template's own tool preamble, replays assistant `tool_calls` as XML, parses XML first with the JSON forms (flat and OpenAI-nested) as fallback, coerces parameter values to JSON types, and never emits a call without a name (an unparseable block stays visible as text). Covered by unit tests in `server::chat::tests`.
- [x] **GDN causal conv1d: coalesced prefill kernel.** The existing kernel is one thread per channel walking the sequence — correct for a 1-token decode step, wrong for a 2048-token chunk: `x` is `(bsz, dim, seqlen)`, so consecutive threads read addresses `seqlen*2` bytes apart (every warp load touches 32 sectors to use 2 bytes of each) and the grid is only `ceil(dim/256)` = 40 CTAs. Measured **114 GB/s of ~1008**, exactly the ~1/8 the coalescing loss predicts. New `conv1d_prefill_kernel<ACT>` gives each CTA a [64 channel x 64 token] tile, loading along the sequence (contiguous in `x`) into shared memory and storing along channels (contiguous in `out`) so both directions coalesce, with the smem row stride padded odd in 4-byte words for conflict-free channel-major reads; `conv1d_state_writeback_kernel` rolls the conv window separately. Engaged at `seqlen >= 64` outside graph capture. **114 -> 551 GB/s (4.9x), bit-exact** against the per-channel kernel on both output and state (`attn_check conv` verifies this). 50 -> 7 ms per prefill chunk.
- [x] **Delta-rule state in registers (prefill).** The recurrence was not compute bound — it kept the recurrent state in GLOBAL memory and touched it three times per step, ~18 GB per layer for a 2048-token chunk (~3.6 ms at L2 speed vs 4.87 ms measured) on ~150 GFLOP of actual work. `cuda_recurrent_gdr_kernel_128_reg<V_SPLIT>` holds each thread's 32-float slice in registers for the whole sequence when `save_history == false` and writes back once: 184 -> 138 ms/chunk. Verified against the global-memory kernel by `attn_check gdn` (`EXL3_GDN_NOREG=1` forces the old one).
- [x] **Draft length 4 -> 3.** Upstream declares `default_draft_size: 4` but pairs it with confidence-based truncation (`generator/draft_confidence.py`: drafter exports the argmax logit, an online binned calibrator maps it to observed acceptance, and the block is cut where the running acceptance product falls below 0.4). Ported as `src/draft_conf.rs` with tests — and measured a LOSS here (76.5 vs 82.0 tok/s at 16k), because reading the confidence costs a device sync per draft step while upstream pays a host round-trip regardless. A sync-free variant (window sized from an EMA of accepted length) was also a loss: the verify re-reads all 8.5 GB of weights whatever its `q_len`, so a shorter window only buys more verify rounds. The window wants to be as long as acceptance repays one extra `lm_head` (0.686 ms) per draft step — measured 3 (2: 66-68, 3: 75-82, 4: 64-75, 7: ~25% worse). `EXL3_DRAFT_CONFIDENCE=0.4` re-enables upstream's truncation.
- [x] Measured and rejected, with numbers: the EXL3 trellis GEMV is already at 92-103% of memory bandwidth (`attn_check exl3`), so decode has no headroom there; double-buffering the mma attention tiles is a regression (52.8 -> 84.1 ms — occupancy limited, not barrier limited); `MMA_BN` is structurally pinned at 16 by the PV mma k-dimension; and CUDA graphs are worth only ~6% on this model (`infer` 52.3 vs 49.4 tok/s with `--no-graph`), so the server not capturing them is not the decode gap.
- [x] Both WY-chunk bugs guarded by tests, not just fixed: `gdn_chunk::tests::survives_strong_decay` runs |g| up to 8 (where the cumulative decay is `exp(-512)` and flushes to zero, so the old `beta/A` form divides by zero) and asserts finite output against the scalar reference; and `attn_check` routes every measurement through a single `timed()` helper, so timing the reference and the candidate differently — which manufactured a 1.10x that the model contradicted — is now structurally impossible.
- [x] **WY-chunked gated delta rule** (`src/gdn_chunk.rs` + `kernels/gdn_chunk.cu`), on by default for prefill chunks >= 128 tokens; `EXL3_GDN_CHUNK=0` reverts to the sequential kernel, which decode and speculative verification keep using and which remains the validation reference. Two fused kernels: `gdn_chunk_wy_kernel` (K K^T, decay ratios, M, triangular solve, W/Uv — one CTA per (V head, chunk), one thread per right-hand-side column) and `gdn_chunk_fused_kernel` (state scan **and** outputs on the tensor cores, mma m16n8k16, state resident in registers; S is an accumulator in the state update but a B operand in `W@S`/`Q@S`, so it round-trips through shared memory once per chunk). **1.16x at seq 2048 / 1.26x at 4096** on the rule; in the model 138 -> 116 ms per prefill chunk and ttft 64k 34.71 -> 34.36 s. What moved it, measured: half2 loads in the K K^T dot product (WY 2.65 -> 1.51 ms), X in registers rather than 64 KB of shared memory (3.18 -> 2.65), removing 64 per-CTA barriers the substitution never needed (each thread touches only its own column), and deriving every decay on chip instead of materializing `exp(A_t - A_j)` as an `[nv, nc, C, C]` array (1.5 ms of write-then-read-back, and it had forced `Q K^T` to broaden from nk to nv heads). Numerics: the naive form divides by the cumulative decay and NaNs at realistic gate magnitudes — solved for `Ũ = A u` so every decay appears only as a bounded ratio, which is why fla's formulation looks the way it does; `attn_check gdnchunk` takes `EXL3_GDN_TEST_G` and is verified to |g| = 2.0. Remaining at seq 4096: prep 1.29 / QK^T 0.14 / WY 1.51 / scan+outputs 2.05 against fla's 1.255 total.
- [x] **`gdn.cu` dispatch/grid mismatch (latent, silent).** `gridDim.z` is `v_split`, but the `history` branch only instantiated `V_SPLIT == 4` and fell through to `<true, 1>` otherwise, so each CTA computed the whole v range and applied the state update twice. Unreachable at the old default and instantly wrong at any other value. Both branches now cover 1/2/4/8 for the 128/128 kernel, non-128 paths clamp to what they instantiate, and the fallbacks `TORCH_CHECK(false, ...)`. Default `v_split` 4 -> **2** (~12% faster across seq 512..4096; not monotonic — 8 is ~80% slower).
- [ ] numerical logit diff vs Python reference not yet measured (greedy output is byte-identical to upstream on every tested prompt — Qwen3 0.6B and Qwen3.5 27B, text + MTP + vision)
- [ ] host GCC is newer than nvcc officially supports; a matching toolchain is safer

## Adaptive draft window (landed)

`draft_num_tokens` now defaults to 4 and acts as a **cap**; each round picks the
window that maximises the speculative-decoding cost model

    throughput ∝ (1 + Σ_{i=1..n} p^i) / (1 + n·r)

from two quantities measured online and free:

- `p` — per-token acceptance, MLE from each verify's own outcome (`a` successes,
  plus one failure when the window was not exhausted), EMA 0.94.
- `r` — one draft step's cost as a fraction of one verify's, EMA 0.9 over the
  round timers. **This has to be measured.** The draft step is a fixed cost while
  the verify grows with context, so `r` runs ~0.25 on short prompts and converges
  to 0.076 at 10-12k — which is exactly why fixed n=3 beat fixed n=4 on the short
  synthetic bench while n=4 won on realistic traffic. A single fixed window is
  wrong for traffic that spans both.

`EXL3_DRAFT_COST_RATIO` pins `r` for experiments. First 8 rounds run at the full
cap so the estimator gets data.

Measured (temp 0, 27B, 4090):

| workload            | fixed n=3 | fixed n=4 | adaptive |
|---------------------|-----------|-----------|----------|
| synthetic ctx~4000  | 85.5      | 79.2      | 83.9     |
| realistic ctx~12000 | —         | 98.3      | 98.1     |

It tracks the better of the two fixed settings on each workload. The residual on
the synthetic bench is warm-up: 8 of ~65 rounds run at the cap.
