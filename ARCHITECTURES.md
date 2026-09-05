# Architecture support

Every architecture registered in ExLlamaV3 (upstream `exllamav3/architecture/architectures.py`,
tracked at **v1.4.4**), and where this port stands on each.

The upstream list was extracted mechanically from the `arch_string` of each
`architecture/*.py`, so it is complete rather than curated. The **Needs** column is
likewise derived from which modules each file imports.

T/s are from running on RTX 4090 + 64GB DDR5.

## Legend

| mark | meaning |
|---|---|
| ✅ **Verified** | runs end-to-end, checked against a real checkpoint |
| 🟡 **Untested** | implemented, builds, but never run against a checkpoint of this arch |
| ❌ **Not implemented** | recognised as unsupported; `Config::from_dir` errors out by name |

Subsystem shorthand in **Needs**: **MoE** block-sparse experts · **MLA** multi-head latent
attention · **GDN** gated delta net · **SSM** Mamba-style state space · **Vision** vision
tower · **HC** hyper-connections · **PLE** per-layer embeddings / n-gram embeddings ·
**QSA** query-sparse attention.

## Status summary

| | count |
|---|---|
| ✅ Verified | 6 (5 upstream + 1 port-only) |
| 🟡 Implemented, untested | 11 |
| ❌ Not implemented | 46 |
| **Total** | **63** (62 upstream + 1 port-only) |

Counting subsystems across the 46 unsupported rows, the biggest levers are **MoE** (25 rows,
now partly landed) and **Vision** (19); then GDN 6, MLA 3, SSM 3, HC 2. Note
that 12 of the 46 need *no* new subsystem at all — they are dense Llama-shaped variants
that only need their arch string registered and their config quirks handled.

## Measured throughput

Single-stream greedy decode, RTX 4090, prompt `"The capital of France is"`, CUDA graph
capture on, from `./run-infer.sh --timing`. One number per *architecture*, not per model —
these are different model sizes and bit widths, so they compare paths, not quality.

| Architecture | Checkpoint | bpw | decode | prefill |
|---|---|---|---|---|
| `Qwen3ForCausalLM` | Qwen3-0.6B | 8.0 | **368 tok/s** | 1641 tok/s |
| `Glm4ForCausalLM` | GLM-4-9B-0414 | 4.0 | **125 tok/s** | 579 tok/s |
| `Qwen3_5ForConditionalGeneration` | Qwen3.8-27B | 3.5 | **56.7 tok/s** | 84 tok/s |
| `Qwen3MoeForCausalLM` | Qwen3-30B-A3B | 3.0 | **154 tok/s** | 56 tok/s |
| `Qwen4ExpForConditionalGeneration` | — | — | — | no checkpoint exists to run |
| `Glm4MoeForCausalLM` | — | — | — | smallest published exl3 quant is 27 GB, over this card |

Prefill here is only 5 tokens, so its figure is dominated by launch overhead and is not a
throughput measurement; it is included because a wildly different ratio between the two
columns is a useful smoke signal.

Both first runs turned up a bug rather than a number.

Qwen3-30B-A3B aborted at CUDA graph capture: the per-expert MoE loop reads the routing table
back to the host once per layer to bucket rows by expert, and a D2H copy of an unpinned
tensor is illegal during capture. Running it eager worked but gave 32 tok/s, far off what 3B
active parameters at 3.0bpw should manage. The fix was to route on the device — see
[MoE decode](#moe-decode-the-multi-gemm-path) below, which took it to 154 tok/s.

Running GLM-4 for the first time turned up the other one: the paged KV
cache sized its planes to `num_kv_heads`, while `Attention::forward` repeats KV heads up to
a GQA ratio the attention kernel supports. GLM-4-9B is 32 q / 2 kv — ratio 16, which the
kernel does not take — so the repeat to 4 heads hit a cache built for 2 and the kernel threw
`k_cache n_kv_heads mismatch`. Every paged cache now sizes on `kv_heads_eff()`. Nothing
already ✅ was affected (their ratios need no repeat), which is exactly why it survived this
long: it is invisible until an architecture with an awkward GQA ratio shows up.

---

## Qwen

| Architecture | Status | Needs | Note |
|---|---|---|---|
| `Qwen3ForCausalLM` | ✅ Verified | — | Qwen3-0.6B-exl3-8.0bpw — **368 tok/s** |
| `Qwen3_5ForConditionalGeneration` | ✅ Verified | GDN, Vision | Qwen3.8-27B 3.5/4.0bpw: text, MTP, vision, cache quant, prefix cache — **56.7 tok/s** |
| `Qwen3_5ForCausalLM` | ✅ Verified | GDN | same path, text-only checkpoints |
| `Qwen2ForCausalLM` | 🟡 Untested | — | Llama + Q/K/V bias, picked up automatically |
| `Qwen3MoeForCausalLM` | ✅ Verified | MoE | Qwen3-30B-A3B-exl3-3.0bpw — **154 tok/s** |
| `Qwen3_5MoeForCausalLM` | ❌ | MoE, GDN | hybrid + experts; not yet wired |
| `Qwen3_5MoeForConditionalGeneration` | ❌ | MoE, GDN, Vision | |
| `Qwen3VLForConditionalGeneration` | 🟡 Untested | Vision | Same tower as Qwen3.5 plus deepstack; Qwen3 block under `model.language_model`. Runs on a synthetic checkpoint (`tests/qwen3vl_load.rs`); Qwen3-VL-8B 4.0bpw is downloaded and waiting on a free GPU |
| `Qwen3VLMoeForConditionalGeneration` | 🟡 Untested | MoE, Vision | As above with the Qwen3-MoE block — both halves are verified separately, the pairing is not |
| `Qwen3NextForCausalLM` | ❌ | MoE, GDN | |
| `Qwen2_5_VLForConditionalGeneration` | ❌ | Vision | A genuinely different tower (windowed attention, its own merger), not the Qwen3-VL one |
| `Qwen4ExpForConditionalGeneration` | 🟡 Untested | MoE, GDN, Vision, PLE, QSA, HC | Qwen3.8-Flash-Next; most new subsystems of any arch, all now implemented and wired. Loads and generates on a synthetic checkpoint (`tests/qwen4_load.rs`); no real one exists to check numerics or speed against. See below |

## GLM

| Architecture | Status | Needs | Note |
|---|---|---|---|
| `Glm4ForCausalLM` | ✅ Verified | — | GPT-J RoPE, sandwich norms, partial rotary. GLM-4-9B-0414-exl3-4.0bpw — **125 tok/s** |
| `Glm4MoeForCausalLM` | 🟡 Untested | MoE | GLM-4.5/4.6. NeoX RoPE (**not** GLM4's GPT-J), no sandwich norms, QK-norm gated on `use_qk_norm`, `dots` router + shared expert, first `first_k_dense_replace` layers dense. Loads and generates on a synthetic checkpoint (`tests/glm4moe_load.rs`); no real one fits this card — see below |
| `Glm4vForConditionalGeneration` | ❌ | Vision | |
| `Glm4vMoeForConditionalGeneration` | ❌ | MoE, Vision | |
| `GlmMoeDsaForCausalLM` | ❌ | MoE, MLA | |
| `Glm5NextForConditionalGeneration` | ❌ | MoE, MLA, GDN, Vision, HC | |

## DeepSeek

| Architecture | Status | Needs |
|---|---|---|
| `DeepseekV3ForCausalLM` | ❌ | MoE, MLA |
| `DeepseekV4ForCausalLM` | ❌ | MoE, HC |

## Gemma

| Architecture | Status | Needs |
|---|---|---|
| `Gemma2ForCausalLM` | ❌ | — (logit softcap, alternating window) |
| `Gemma3ForCausalLM` | ❌ | — |
| `Gemma3ForConditionalGeneration` | ❌ | Vision |
| `Gemma4ForConditionalGeneration` | ❌ | MoE, Vision |
| `Gemma4UnifiedForConditionalGeneration` | ❌ | MoE, Vision |

## Llama-shaped

Upstream declares these as `LlamaModel` subclasses with no overrides beyond the arch string
and a chat template, so they resolve to the same code path here.

| Architecture | Status | Needs | Note |
|---|---|---|---|
| `LlamaForCausalLM` | 🟡 Untested | — | |
| `MistralForCausalLM` | 🟡 Untested | — | pure Llama alias upstream |
| `MiMoForCausalLM` | 🟡 Untested | — | pure Llama alias upstream |
| `SeedOssForCausalLM` | 🟡 Untested | — | pure Llama alias upstream |
| `IQuestCoderForCausalLM` | 🟡 Untested | — | pure Llama alias upstream |
| `MixtralForCausalLM` | ❌ | MoE | expert keys are `block_sparse_moe.experts.N.w1/w2/w3`, not the Qwen layout |
| `Ministral3ForCausalLM` | ❌ | Vision | derives from Mistral3 |
| `Mistral3ForConditionalGeneration` | ❌ | Vision | |

## Everything else

| Architecture | Status | Needs |
|---|---|---|
| `AfmoeForCausalLM` | ❌ | MoE |
| `ApertusForCausalLM` | ❌ | — |
| `ArceeForCausalLM` | ❌ | — |
| `CohereForCausalLM` | ❌ | — |
| `Cohere2ForCausalLM` | ❌ | — |
| `DeciLMForCausalLM` | ❌ | — (variable per-layer shapes) |
| `DFlashDraftModel` | ❌ | SSM, Vision |
| `DFlashLagunaForCausalLM` | ❌ | Vision |
| `Dots1ForCausalLM` | ❌ | MoE |
| `Ernie4_5_ForCausalLM` | ❌ | — |
| `Ernie4_5_MoeForCausalLM` | ❌ | MoE |
| `Exaone4ForCausalLM` | ❌ | — |
| `GptOssForCausalLM` | ❌ | MoE (biased experts, padded hidden dim) |
| `HCXVisionV2ForCausalLM` | ❌ | Vision |
| `HyperCLOVAXForCausalLM` | ❌ | Vision |
| `HYV3ForCausalLM` | ❌ | MoE |
| `LagunaForCausalLM` | ❌ | MoE |
| `Lfm2MoeForCausalLM` | ❌ | MoE, Vision |
| `MiniMaxM2ForCausalLM` | ❌ | MoE |
| `MuseGlimmerForConditionalGeneration` | ❌ | Vision |
| `MuseGlimmerAssistantModel` | ❌ | SSM, Vision |
| `NemotronHForCausalLM` | ❌ | MoE, SSM |
| `Olmo3ForCausalLM` | ❌ | — |
| `OlmoHybridForCausalLM` | ❌ | GDN |
| `Phi3ForCausalLM` | ❌ | — |
| `SmolLM3ForCausalLM` | ❌ | — |
| `SolarOpenForCausalLM` | 🟡 Untested | MoE (derives from Glm4Moe — same path here, so the same fixture covers it) |
| `Step3p5ForCausalLM` | ❌ | MoE |
| `Step3p7ForConditionalGeneration` | ❌ | Vision |

## Port-only

| Architecture | Status | Note |
|---|---|---|
| `DFlash2DraftModel` | ✅ Verified | Speculative draft model, used with Qwen3.5. Not an upstream arch string; reads through the Qwen3 block path rooted at the checkpoint top level |

---

## What "untested" actually means here

Seven rows are 🟡. Those were written from the upstream source and the checkpoint's tensor
index, they compile, and their unit tests pass — but **no checkpoint of that architecture
has ever been loaded**. Do not read 🟡 as "probably fine". Known specific risks:

- **GLM4** — GPT-J RoPE (adjacent pairs, not halves) and the sandwich-norm placement are
  both easy to get subtly wrong in a way that still produces fluent-looking text.
- **Qwen3-MoE** — expert key names and the router weight orientation were verified against
  the real `model.safetensors.index.json`, but routing weights and expert dispatch have
  only been checked against a hand-derived reference, not a running model.
- **GLM4-MoE** — the `dots` router is the risk. `e_score_correction_bias` steers
  *selection* but must not reach the applied weights, and the weights normalize to
  `routed_scaling_factor` (2.5) rather than to 1. Both mistakes leave a model that
  generates fluent text at the wrong expert mixture. Unit-tested against the kernel's
  semantics in `src/moe.rs`, not against a checkpoint.
- **Llama-shaped** — lowest risk of the group, since upstream declares them as no-override
  subclasses, but "upstream says it's an alias" is not the same as having run one.

## Vision beyond Qwen3.5

The tower was never Qwen3.5-specific: upstream's `Qwen3VLVisionModel` is shared verbatim by
`Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`, the Qwen3.5 family
and qwen4_exp. What was Qwen3.5-specific here was everything *around* it — a hard
`arch_kind == Qwen35` gate, a decoder call that only spoke to `Qwen35Cache`, and a missing
piece of the tower.

Three changes opened it up:

- **The gate is now the checkpoint, not the arch.** `--vision` asks whether a `vision_config`
  is present, and the decoder is chosen by `is_hybrid()`. A VL variant of an existing block
  structure needs no new case. `text_config` nesting is likewise keyed on presence rather
  than on the architecture.
- **The VL archs are the blocks they already were.** `Qwen3VLForConditionalGeneration` is the
  Qwen3 block with the decoder moved under `model.language_model`, and the MoE variant is the
  Qwen3-MoE block the same way — so both are registrations against existing `ArchKind`s, not
  new stacks.
- **Deepstack.** This is the one piece of real new work. Qwen3.5 checkpoints carry
  `deepstack_visual_indexes: []`; Qwen3-VL carries `[8, 16, 24]`. Each index taps that vision
  block's output through its own patch merger, and the first three *text* layers add the
  result at the image token positions — the language model sees the image at three depths of
  the tower, not just through the final merger.

The deepstack mergers differ from the final one in exactly one way: they LayerNorm **after**
the 2x2 spatial shuffle rather than before, so the norm runs over the concatenated group
instead of per patch. Identical shapes, and both orderings produce plausible features, so
getting it backwards is silent — `vision.rs` has a unit test that pins the ordering against
a group-then-normalize reference.

`Model::forward_paged_mm` is the non-hybrid multimodal forward, adding deepstack entry `i` to
the residual ahead of block `i`; the hybrid stack takes the same argument, so a Qwen3.5-family
checkpoint that ships deepstack indexes would work without further changes. Qwen3.5 vision was
re-run after the refactor and still describes a test image correctly.

`tests/qwen3vl_load.rs` covers the seam on a synthetic checkpoint: one deepstack block per
configured index, distinct taps, logits that actually change when deepstack is supplied, and
a no-deepstack forward that matches the plain paged one. Two things it caught immediately:

- The first version of the fixture wrote **zero-weight** LayerNorms and RMSNorms. This port
  adds a constant bias of 1.0 only where the architecture stores its norm weights
  zero-centred (Qwen3.5, qwen4_exp); a Qwen3 RMSNorm and every vision LayerNorm use the
  stored weight as-is, so zeros silently zeroed the entire model — and every check passed,
  because zeros are finite and correctly shaped. The `Glm4Moe` fixture had the same defect,
  which means its earlier "loads and forwards" result was worth much less than it looked.
  All three fixtures now write realistic norm weights, and every fixture test asserts the
  logits are non-trivial.
- A double `cache.advance` in the `Glm4Moe` chunked-decode test, which looked exactly like a
  model bug (57% relative divergence on a *single* layer) until the same weights were run
  through the Qwen3 arch path and diverged identically.

## MoE decode: the multi-GEMM path

`BlockSparseMlp` originally ran one gate/up/down GEMM per active expert. To know which rows
went to which expert it copied the routing table to the host — a device sync per layer, 48
of them per token on Qwen3-30B-A3B, and an outright error under CUDA graph capture.

`exl3_mgemm` (already vendored in `kernels/quant/exl3_gemm.cu`, unused until now) takes a
pointer list of expert weights plus the selection *as a device tensor*, so three launches
replace `3 * top_k` and nothing leaves the GPU. Passing the routing weights to the down
projection makes the kernel scale and reduce the experts into one row, so the mixture comes
out of the same call.

Measured on Qwen3-30B-A3B at 3.0bpw, greedy, 4090:

| path | decode |
|---|---|
| per-expert loop, eager (was the only option) | 32.0 tok/s |
| multi-GEMM, eager | 19.1 tok/s |
| **multi-GEMM + CUDA graph** | **154 tok/s** |

The middle row is the interesting one: taken on its own the multi-GEMM is *slower* than the
loop it replaces at this shape. It wins because it is the only one of the two that can be
captured, and capture is worth ~8x here. Had I stopped at the eager comparison I would have
concluded the change was a regression and reverted it. Output is token-for-token identical
to the reference loop across 48 greedy tokens; `EXL3_NO_MGEMM_MOE=1` forces the old path for
A/B.

Two limits. Decode only — prefill (`rows > 1`) still runs the per-expert loop, where one
sync amortizes over the whole prompt. And EXL3 experts only, with a uniform bit rate and
codebook per projection and no padding between the config's dimensions and the loaded
weights; anything else declines the path rather than being coerced into it.

There is also a fully fused `exl3_moe` kernel vendored alongside, which would cover batched
decode. I wired it first and then removed it: it only supports the `mcg` and `mul1`
codebooks, and Qwen3-30B-A3B (like every MoE exl3 quant to hand) uses neither, so the path
could not be exercised on any checkpoint here. Shipping a fast path nothing can run is how
silent breakage gets in.

## Why Glm4Moe is still 🟡

The implementation is complete and executes: `tests/glm4moe_load.rs` loads a synthetic
checkpoint (`tests/make_glm4moe_fixture.py`) and generates from it, which covers the paths
unique to this architecture — the `dots` router with its `e_score_correction_bias`, the
`first_k_dense_replace` leading dense layers, the ungated shared expert, QK-norm gated on
`use_qk_norm`, and NeoX rather than GPT-J RoPE despite the shared "Glm4" name — plus the
usual chunked-decode-equals-prefill check.

What is missing is real weights, and that is a hardware limit rather than a to-do. The
smallest published `Glm4MoeForCausalLM` exl3 quant is GLM-4.5-Air at 2.0bpw, 28.4 GB
(2.25bpw is 31.5 GB; the REAP-82B-A12B at 2.5bpw is 27.1 GB). The development card is a
24 GB 4090 and this port has no weight offload, so none of them can be loaded at all —
never mind leaving room for a KV cache. `SolarOpenForCausalLM` is the same code path and is
blocked the same way.

So the router arithmetic is pinned by unit tests derived from the CUDA kernel
(`src/moe.rs`), and the block structure is pinned by the fixture, but nobody has yet
confirmed that a real GLM-4.5 checkpoint produces sensible text through this port.

## Synthetic checkpoint fixtures

Two architectures have no runnable real checkpoint here — `Qwen4ExpForConditionalGeneration`
(none published in exl3 form) and `Glm4MoeForCausalLM` (too large for the card). Both have a
generator under `tests/` that writes a tiny checkpoint with real tensor names and shapes and
random weights:

```
python3 tests/make_qwen4_fixture.py    target/qwen4_fixture
python3 tests/make_glm4moe_fixture.py  target/glm4moe_fixture
cargo test --test qwen4_load --test glm4moe_load
```

The tests skip themselves when the fixture directory is absent, so `cargo test` stays green
without torch installed. A fixture cannot check numerics — there is nothing to compare
against — but it does check that every tensor key the loaders ask for exists where they look
it up, that a forward produces finite logits, and that a chunked decode lands on the same
logits as a one-shot prefill, which is what ties the per-layer state together. Fixture head
dims are 128: the paged attention kernel is built for wide heads and aborts on a narrow one.

## Qwen4Exp: wired, unvalidated

`Qwen4ExpForConditionalGeneration` needed five subsystems beyond the Qwen3 block, more than
any other architecture in the list. All five are implemented, and the arch is now assembled:
`Config::from_dir` recognises it, `Model::load` builds the stack, and
`Model::forward_qwen4` generates against a `Qwen4Cache`. `infer` runs it.

What it has **not** had is a real checkpoint — none is published in exl3 form — so this is
🟡 and not ✅, and the gap is numerics, not plumbing. `tests/qwen4_load.rs` loads a synthetic
checkpoint (`tests/make_qwen4_fixture.py`: real tensor names and shapes, random weights) and
checks the things that can be checked without a reference: that every key the loaders ask
for exists where they look, that a forward produces finite logits, that the QSA mask stays
causal and inside its budget, and — the load-bearing one — that a chunked decode lands on
the same logits as a one-shot prefill, which exercises the GDN recurrence, the PLE conv
window and token history, the pooled indexer keys and the K/V caches against each other.

The pieces, and where they live:

- **HC** — `src/hc.rs`. `hc_mult` parallel fp32 residual streams replacing the input/post
  layernorms: `expand_streams` broadcasts the embedding into the stack, each sublayer site
  collapses it through a low-rank sigmoid gate, and the output is written back per stream
  through a `2 * sigmoid` gate. Ported from `_mix_ref`, upstream's own fp32 parity
  reference. Also on the critical path for `DeepseekV4ForCausalLM` and
  `Glm5NextForConditionalGeneration`, which use the full mHC mixer (Sinkhorn + combine
  matrix) rather than this elementwise flavour.
- **MoE + shared expert** — `src/moe.rs`. Both router flavours, plus the shared expert and
  its `sigmoid(shared_expert_gate(x))` weighting (qwen4_exp and the Qwen2-MoE lineage have
  one; GLM4-MoE adds its shared expert unweighted).
- **GDN output gate** — `output_gate_type`. The gated-RMSNorm kernel takes a gate activation
  (`kernels/norm.cu`, threaded through the shim and `ffi::GateAct`); Qwen3.5 keeps silu,
  qwen4_exp selects sigmoid.
- **PLE** — `src/ple.rs` (arithmetic, ported from `forward_streams_reference`) and
  `src/ple_state.rs` (per-slot conv window + trailing token ids, with the same tail-buffer
  rewind arithmetic as `Qwen35PagedCache`'s GDN planes).
- **N-gram embedding** — `src/ngram.rs`. Hashes each position's trailing 2..=`ngram_size`
  windows into `(ngram_size - 1) * heads_per_ngram` table rows, never across an eos boundary.
  Both table formats load, including the `exl3_ngram_trellis` packed form (tail-biting ring
  over the mul1 codebook), whose codec is ported here in torch ops.
- **QSA** — `src/qsa.rs`. An indexer head scores 4-token blocks (`relu(q·k)` summed over
  index heads, pooled keys roped at the block's *start*) and keeps the top
  `token_budget / compress_ratio` plus the query's own tail block. Both output forms are
  ported and cross-checked against each other: a dense causal mask and per-row index lists.
- **Vision** — the tower is the Qwen3.5 one, unchanged upstream, reached through
  `ArchKind::has_qwen3_vl_tower()`.

Known limits of the current wiring, all of which cost performance or capacity rather than
correctness:

| Limit | Consequence |
|---|---|
| **Contiguous, single-sequence cache** (`Qwen4Cache`) | The QSA layers attend under a per-query mask, which the paged kernels cannot express, and the indexer needs the whole raw-key history in one plane. So no paging, no batching, no prefix cache on this arch yet |
| **QSA runs in tch** | `Attention::forward_masked` materializes the `[heads, seq, total]` score matrix. Correct at any length, but it costs dense attention's memory and time — the sparsity is semantic, not yet a speedup. Upstream gathers only the selected rows; `kernels/dsa_topk.cu` is vendored and still unused |
| **No CUDA graph** | Shapes change every step on the masked path, so `infer` disables capture for this arch |
| **N-gram table must be resident** | A production table is tens of billions of parameters and upstream streams it from disk with threaded preads. `src/safetensors.rs` has no row-level reads, so real checkpoints will not fit even though sharding is handled |
| **No MTP head** | `qwen4_exp_mtp.py` is not ported |

## Download candidates (not yet fetched)

exl3 checkpoints found on Hugging Face for architectures that are 🟡 untested or ❌ not
implemented here. Links only — nothing below has been downloaded yet.

| Repo | Architecture | Row status | Size | Note |
|---|---|---|---|---|
| [lesj0610/DeepSeek-V2-Lite-Chat-exl3-4.0bpw](https://huggingface.co/lesj0610/DeepSeek-V2-Lite-Chat-exl3-4.0bpw) | `DeepseekV2ForCausalLM` | ❌ (V2 isn't even in the table — smaller cousin of V3) | ~7.7 GB, single shard | Real DeepSeek MLA+MoE arch, not a Qwen/Llama distill wearing a DeepSeek name. Smallest real DeepSeek exl3 quant found; closest available proxy for validating `DeepseekV3ForCausalLM`'s MLA/MoE path cheaply |
| [Ataromoku/Seed-OSS-36B-Instruct-exl3](https://huggingface.co/Ataromoku/Seed-OSS-36B-Instruct-exl3) (branch `4.0bpw_H6`) | `SeedOssForCausalLM` | 🟡 | 36B @ 4.0bpw ≈ 18 GB | Only exl3 SeedOss quant found; other branches are `4.53bpw_H6`, `6.0bpw_H6` (bigger) |
| [turboderp/SmolLM3-3B-exl3](https://huggingface.co/turboderp/SmolLM3-3B-exl3) | `SmolLM3ForCausalLM` | ❌ | 3B, small | Cheapest possible pickup in the "no new subsystem" tier — tiny download, quick to add |
| [turboderp/gemma-2-9b-it-exl3](https://huggingface.co/turboderp/gemma-2-9b-it-exl3) | `Gemma2ForCausalLM` | ❌ | 9B | Needs logit softcap + alternating local/global attention window, no new subsystem otherwise |
| [isogen/gemma-3-4b-it-exl3-8bpw-h8](https://huggingface.co/isogen/gemma-3-4b-it-exl3-8bpw-h8) | `Gemma3ForCausalLM` | ❌ | ~3 GB, smallest Gemma3 exl3 found | text-only variant (no `-it-vision` in the name); check config before assuming no vision tower |
| [turboderp/gemma-3-27b-it-exl3](https://huggingface.co/turboderp/gemma-3-27b-it-exl3) | `Gemma3ForConditionalGeneration` | ❌ | 27B | Vision variant, larger; only pull if Gemma3 vision becomes a priority |

Not found on Hugging Face as exl3 quants (checked, came up empty): `MiMoForCausalLM`,
`IQuestCoderForCausalLM`, `CohereForCausalLM` / `Cohere2ForCausalLM`, `Phi3ForCausalLM`,
`Olmo3ForCausalLM`, `DeciLMForCausalLM`, `ArceeForCausalLM`, `ApertusForCausalLM`,
`Exaone4ForCausalLM`, `Ernie4_5_ForCausalLM`. These architectures may only exist as
GGUF/AWQ/bnb quants upstream, or as unquantized checkpoints that would need converting
with exllamav3's own quantizer.

## Adding an architecture

Roughly, in order:

1. Register the `arch_string` in `Config::from_dir_with` (`src/config.rs`) and give it an
   `ArchKind`, or reuse one.
2. Express its block-structure differences through the `ArchKind` predicates
   (`has_qk_norm`, `has_post_norms`, `is_hybrid`) rather than new match arms at call sites.
3. Parse any extra config into a params struct next to `GdnParams` / `MoeParams`.
4. Add the module if it needs one, and wire it into `TransformerBlock` or the loader in
   `src/model.rs`.
5. Verify against a real checkpoint before moving the row to ✅ — and check the tensor names
   in `model.safetensors.index.json` first, which catches most layout mistakes for free.
