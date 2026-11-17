# Qwen4Exp implementation status

## Current milestone — RUNNING, NOT NUMERICALLY VALIDATED

The original local two-shard IQ3_S checkpoint now constructs and executes all48 layers on NumPy and CuPy/RTX5080, using packed native CPU routed experts from mapped canonical sources. No pruned model, full expert expansion, whole-PLE allocation, new checkpoint download, or all-resident GPU intermediate architecture is used.

**The first principal deliverable is NOT complete.** Independent full-model logits differ from the pinned reference by ~5–6% relativeL2. Matching argmax/short greedy IDs is not enough. Numerical debugging is stopped without a speculative production edit; the next experiment is listed below. Serving and performance phases are not started.

Status vocabulary: Not started; In progress; Implemented but unvalidated; Numerically validated; End-to-end validated; Performance optimized. Validation scope is always stated; an operator claim is not a model claim.

## Numbered patch ledger

| Patch | Objective / principal files | Status and evidence | Limitations / next dependency |
|---|---|---|---|
|0200 | Source/model compatibility audit; `reference_contract.md`, `tensor_inventory.{md,json}`, `gguf_metadata.json`, `memory_architecture.md`, `tests/test_0200_qwen4exp_contract.py` | Implemented but unvalidated provenance; complete1224-tensor header/operator inventory tested | Immutable checkpoint revision not embedded; payload hash not verified. Full model numerical evidence is separate |
|0201 | `mini_llm/external/gguf_store.py`, `tests/test_0201_gguf_store.py` | Implemented; header/ownership/range validation passes, including actual original namespace and >4GiB sparse offsets | No payload integrity hashing; little-endian contiguous GGML layouts only |
|0202 | `mini_llm/ops/quantized_weight.py`, `gguf_codebooks.{py,h}`, `tests/test_0202_iq_formats.py`, package-data and MIT notices | Numerically validated **CPU block decoding**: IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS/Q2_0 exactly match compiled reference | Not GPU kernel support; native packed execution0203 |
|0203 | `mini_llm/ops/native_packed_cpu.{py,cpp}`, `tests/test_0203_native_iq_experts.py` | Numerically validated **native packed CPU operators**; independent GGML lanes exact, synthetic experts maxabs3.8147e-6 | FP32 arithmetic even when returned dtype differs; throughput not benchmarked |
|0204 | `expert_source.py`, `expert_residency.py`, `quantized_linear.py`, `external/qwen4exp_gguf.py`, sparse tests and `scripts/probe_0204_qwen4exp_experts.py` | Implemented with numerically validated **CPU/supported-GPU subpaths**; all7 actual expert-format triples maxabs5.6617e-11; complete publication/in-flight tests | All-IQ GPU/top-10 GPU exit not met; fixed slots/asynchronous weight mover not implemented. Existing synchronous sparse materializer retained |
|0205 | `external/qwen4exp_hyperconnection.py`, standalone tests | Implemented but unvalidated against linked full reference; synthetic FP64 math and actual CuPy tests pass | Full HC boundary reference association currently suspect; do not call model parity |
|0206 | `external/qwen4exp_ple.py`, standalone tests | Implemented but unvalidated as a full linked-reference block; compiled Strata hash parity, bounded row decoding and synthetic activation/history tests pass | Fixed target geometry, one sequence, sequential history; independent checkpoint PLE activation parity pending |
|0207 | `external/qwen4exp_mixer.py`, standalone tests | Numerically validated **recurrence subpath** against pinned GGML and synthetic block oracle | Full checkpoint projection+state parity pending; sigmoidZ differs deliberately from Qwen3-Next default |
|0208 | `external/qwen4exp_attention.py`, standalone tests | Implemented but unvalidated as linked full QSA graph; synthetic2061-cell selection/state tests and compiled-source textRoPE/tie checks pass | Text scalar positions only; local libstdc++ tie membership, not portable STL promise; no multimodal/long-context performance claim |
|0209 | `external/qwen4exp_weights.py`, `qwen4exp_model.py`, combined graph tests | Implemented but unvalidated; actual CPU/GPU original-model forward succeeds | Independent logits discrepancy unresolved; diagnostic association must be verified first |
|0210 | Generation state/lifecycle in model; `tests/test_0210_qwen4exp_generation.py`, reference probe/harness/artifacts | In progress; actual3 greedy decisions, exact within-backend prefill/incremental/reset checks pass | Reference logits maxabs~1, relativeL2~5–6%; no end-to-end numerical validation |
|0211 | Existing OpenAI serving registration/tokenizer/template/streaming integration | Not started | Gated on numerically validated model; no new HTTP stack |
|0212 | Critical-path profiling | Not started | No optimized throughput claim from lifecycle timings |
|0213 | Native packed IQ GPU kernels | Not started | New IQ formats safely use CPU; no permanent GPU expansion |
|0214 | Fixed-slot asynchronous expert mover | Not started | Publication/readiness/quarantine contracts documented; current cache synchronous |
|0215 | Independent efficient prefill policy | Not started | Sequential correct-path candidate currently used, model parity unresolved |
|0216 | Adaptive routing/prefetch | Not started | Reuse existing profiler/cache seams later; no prediction-based omissions |
|0217 | CUDA graphs/long context | Not started | Dynamic positions/state must remain runtime values |

Each implemented patch has its own numbered document in this directory, with interfaces, tests, architecture and limitations. Numbers denote independently reviewable change groups, **not git commits**. No branch or commit was created. Qwen3-Next modules/training implementations were not rewritten. The legacy dense Llama format acceptance boundary was kept separate from the expanded CPU registry; old unsupported-format test sentinels and GPU capability parametrization were updated where necessary.

## Verified states and most recent commands

### Consolidated CPU inference/training-component regressions

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_BUILD_DIR=.pytest_cache/native_0203 \
OPENBLAS_NUM_THREADS=1 \
QWEN4EXP_LLAMA_SOURCE=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp \
.venv/bin/python -m pytest -q tests/test_020*.py \
  tests/test_0210_qwen4exp_generation.py tests/test_0106*.py \
  tests/test_linear.py tests/test_rmsnorm.py tests/test_swiglu.py \
  tests/test_moe_experts.py tests/test_moe_router.py tests/test_checkpoint.py
```

**1488 passed,184 skipped in29.52s**. This establishes the current new modules plus existing0106 suite and listed training primitives/checkpoint tests on CPU. It is not the whole repository training suite. Optional GPU/benchmarks/real-model opt-in cases skip as declared. No production code was changed after this verified run.

Earlier selected foundation run:481 passed,86 skipped. Subsequent operator/runtime selected run:466 passed,7 skipped. These overlap and must not be added to the1488 total.

### Targeted actual GPU kernels

Supported-format GPU GEMV and finite mixed-fallback tests: **24 passed,43 deselected** (six mixed cases plus18 GEMV cases exercising FP16/FP32 legacy kernels). New model/reference activation arithmetic is FP32; this legacy FP16 regression coverage is not a preference change. No new IQ GPU format is advertised.

A broader exploratory CuPy run identified six pre-existing backend-array-assumption cases that fail when CPU-oriented tests run under CuPy; they are listed in `0204_sparse_execution.md` and were left untouched. **The entire CuPy suite is not green/claimed green.** The18 CPU-only-format GPU-parametrization failures were fixed by testing GPU capability rather than the CPU registry; CPU format coverage remains intact.

### Original-checkpoint lifecycle on CPU and GPU

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_BUILD_DIR=.pytest_cache/native_0203 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
QWEN4EXP_MODEL_PATH=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/ \
.venv/bin/python -m pytest -q -s tests/test_0210_qwen4exp_generation.py
```

CPU: **1 passed in40.68s**. Repeat with `MINI_LLM_BACKEND=cupy`: **1 passed in30.06s**. Fixed IDs[1,2]; greedy IDs[220,100561,198] on both. Within each backend, prefill logits and fresh-request incremental logits match exactly. Six token forwards/test,2880 native CPU expert calls, zero decoded-expert cache bytes, zero GPU expert residents. This is self-consistency, **not independent correctness**.

Peak process RSS: CPU10,281,414,656 bytes (~9.58GiB); CuPy8,632,389,632 bytes (~8.04GiB). GPU pool snapshot4,414,358,016 used /4,420,487,680 reserved bytes (~4.12GiB); not global VRAM high-water. Owned-allocation planning excludes OS-controlled mmap residency, so the8GiB host budget is not an8GiB RSS limit. Hard RSS/page-cache control and long-context stress validation remain limitations.

### Independent reference — unresolved numerical result

Original pinned build's eager prefetch initially blocked safe loading. A **separate CPU-only source copy**, baseSHA171e8846b4af9766c354064cb776cb34a50f053f, changes only `init_mappings(true,...)` to `false`; its3643-file source comparison and exact patch are recorded. Original pinned source is still clean. No quantization/graph mathematics changed. The no-prefetch reference and production ran sequentially under a12GiB RSS cap; no duplicate resident model processes.

| Fixed input prefix | Max absolute logit difference | RelativeL2 | Actual/reference argmax | Top10 overlap |
|---|---:|---:|---|---:|
|[1] |0.958172 |0.061643 |271/271 |9/10 |
|[1,2] |1.006777 |0.050675 |220/220 |9/10 |

Max absolute log-probability differences:1.07785 and0.94804. FP32 stored logits; production native packed FP32 activations differ from GGML's quantized activation arithmetic. No broad acceptance tolerance was applied, and these metrics are **not a passing numerical gate**. No finite-difference step is involved. Reference peak RSS5.72GiB, production8.71GiB; combined probe27.16s.386 boundary comparisons were collected.

The reported first difference at layer0 HC attention read is **not established as the first mathematical divergence**: duplicate reference `hc_mixed` names appear to associate attention/FFN captures incorrectly (identical reference norms). Do not infer a production defect from this association. No production correction or repeated speculative run followed.

Artifacts: `0210_noprefetch_reference_validation.json`, `0210_noprefetch_reference.patch`, `0210_reference_diagnostics.md`. The earlier blocked preflight result is preserved separately, not overwritten as a success.

## Stop state and single next experiment

Known: exact packed decoders/native operators pass; original model runs with bounded owned allocations, mapped experts/PLE, deterministic selected combination and isolated states. Unknown: whether full-model differences arise predominantly from activation quantization, an actual graph/state error, or diagnostic misassociation. Existing evidence does not distinguish those causes.

**Next experiment:** fixed token1/layer0 only; capture reference tensor name **plus tensor/op/source identities and evaluation occurrence**, then associate its HC attention/FFN reads unambiguously with production. Only after verifying boundary association should the first real numerical divergence be classified. Do not alter production code or broaden tolerances merely to match argmax/text.
