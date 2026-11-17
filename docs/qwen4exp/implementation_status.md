# Qwen4Exp implementation status

## Current milestone — SHORT-TEXT VALIDATION GATE PASSED; INITIAL PROFILING

The original local two-shard IQ3_S checkpoint now constructs and executes all48 layers on NumPy and CuPy/RTX5080, using packed native CPU routed experts from mapped canonical sources. No pruned model, full expert expansion, whole-PLE allocation, new checkpoint download, or all-resident GPU intermediate architecture is used.

**Complete model/state mathematical equivalence is not yet claimed.** The current numerical investigation fixed the diagnostic indentation and analyzed retained states successfully. Identical-input MoE routing/experts/shared/combination tests at early and late layers agree at expected arithmetic precision after accounting for activation quantization. A one-time1.31e-6 early input perturbation produces2.31–6.33% logit changes in the diagnostic quantized path, demonstrating sensitivity at the observed discrepancy scale. No production mathematical defect is established. Subsequent0226 replays pass236 comparisons each for identical reference pre-state and continuous local state across16 tokens. Full-model CuPy incremental/prefill logits and all active-state hashes match exactly at prefixes1/2/4/8/16 (0227). The short-text gate has passed; initial profiling has begun without a kernel change or achieved speedup. Full independent model equivalence and long-context attribution remain outstanding. See numerical_root_cause.md. Production mathematics was not changed.

Status vocabulary: Not started; In progress; Implemented but unvalidated; Numerically validated; End-to-end validated; Performance optimized. Validation scope is always stated; an operator claim is not a model claim.

## Numbered patch ledger

| Patch | Objective / principal files | Status and evidence | Limitations / next dependency |
|---|---|---|---|
|0200 | Source/model compatibility audit; `reference_contract.md`, `tensor_inventory.{md,json}`, `gguf_metadata.json`, `memory_architecture.md`, `tests/test_0200_qwen4exp_contract.py` | Implemented but unvalidated provenance; complete1224-tensor header/operator inventory tested | Immutable checkpoint revision not embedded; payload hash not verified. Full model numerical evidence is separate |
|0201 | `mini_llm/external/gguf_store.py`, `tests/test_0201_gguf_store.py` | Implemented; header/ownership/range validation passes, including actual original namespace and >4GiB sparse offsets | No payload integrity hashing; little-endian contiguous GGML layouts only |
|0202 | `mini_llm/ops/quantized_weight.py`, `gguf_codebooks.{py,h}`, `tests/test_0202_iq_formats.py`, package-data and MIT notices | Numerically validated **CPU block decoding**: IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS/Q2_0 exactly match compiled reference | Not GPU kernel support; native packed execution0203 |
|0203 | `mini_llm/ops/native_packed_cpu.{py,cpp}`, `tests/test_0203_native_iq_experts.py` | Numerically validated **native packed CPU operators**; independent GGML lanes exact, synthetic experts maxabs3.8147e-6 | FP32 arithmetic even when returned dtype differs; throughput not benchmarked |
|0204 | `expert_source.py`, `expert_residency.py`, `quantized_linear.py`, `external/qwen4exp_gguf.py`, sparse tests and `scripts/probe_0204_qwen4exp_experts.py` | Implemented with numerically validated **CPU/supported-GPU subpaths**; all7 actual expert-format triples maxabs5.6617e-11; complete publication/in-flight tests | All-IQ GPU/top-10 GPU exit not met; fixed slots/asynchronous weight mover not implemented. Existing synchronous sparse materializer retained |
|0205 | `external/qwen4exp_hyperconnection.py`, standalone tests | Implemented but unvalidated against linked full reference; synthetic FP64 math and actual CuPy tests pass | Boundary identity verified; first real HC-read difference explained by BF16 rounding; not full-model parity |
|0206 | `external/qwen4exp_ple.py`, standalone tests | Implemented but unvalidated as a full linked-reference block; compiled Strata hash parity, bounded row decoding and synthetic activation/history tests pass | Fixed target geometry, one sequence, sequential history; independent checkpoint PLE activation parity pending |
|0207 | `external/qwen4exp_mixer.py`, standalone tests | Numerically validated **representative checkpoint state subpaths**, including16-token identical-input/continuous state replay against pinned GGML | All-layer independent state parity pending; sigmoidZ differs deliberately from Qwen3-Next default |
|0208 | `external/qwen4exp_attention.py`, standalone tests | Numerically validated **representative short-text QSA state/output subpaths** over16 reference tokens; synthetic2061-cell sparse selection/state and compiled textRoPE/tie checks pass | Text scalar positions only; local libstdc++ tie membership, not portable STL promise; no multimodal/long-context performance claim |
|0209 | `external/qwen4exp_weights.py`, `qwen4exp_model.py`, combined graph tests | Implemented but unvalidated; actual CPU/GPU original-model forward succeeds | Boundary association verified;5–6% baseline reference logit discrepancy not fully attributed; no demonstrated graph defect |
|0210 | Generation state/lifecycle in model; `tests/test_0210_qwen4exp_generation.py`, reference probe/harness/artifacts | In progress; actual3 greedy decisions, reset checks and16-token full-model incremental/prefill active-state consistency pass | Reference logits maxabs~1, relativeL2~5–6%; no end-to-end numerical validation |
|0211 | Existing OpenAI serving registration/tokenizer/template/streaming integration | Not started | Gated on numerically validated model; no new HTTP stack |
|0212 | Initial profiling; `scripts/probe_0227_short_sequences_and_profile.py`, `0212_initial_profile.md` | In progress; measured eight CuPy decode positions with host spans/CUDA events | Exclusive critical path, SSD/page faults, overlap efficiency and global VRAM high-water outstanding; no speedup claim |
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

The old HC-attention comparison is superseded by the verified0218–0220 captures: reshaped aliases overwrote occurrence-based artifacts. Corrected first HC-read error is0.00102728 relativeL2, explained by BF16 projection-input rounding. No production correction was needed.

Artifacts: `0210_noprefetch_reference_validation.json`, `0210_noprefetch_reference.patch`, `0210_reference_diagnostics.md`. The earlier blocked preflight result is preserved separately, not overwritten as a success.

## Investigation outcome and next measured work

Known: packed decoders/native operators pass; original model runs with bounded owned allocations, mapped experts/PLE, deterministic selected combination and isolated states. Diagnostic misassociation is resolved; BF16/Q8 arithmetic differences and amplification are demonstrated. Representative identical-input DeltaNet/QSA states pass over16 tokens, and full-model local state consistency passes. Complete attribution of the5–6% reference logit discrepancy remains open; no incorrect production operation has been demonstrated.

**Update through investigations0224–0225:** indentation is fixed; state reanalysis passes with35 comparisons and exact PLE row IDs, without model reloads. Six identical-input MoE cases have matching top10 sets; teacher-forced down/combination agree at roughly1e-7 precision. A shared-expert quantization threshold amplifies1.47e-7 input error into5.50e-4 output error. Controlled full-forward perturbations demonstrate several-percent sensitivity. Most recent focused diagnostics plus0106 inference regressions:1295 passed,176 skipped. No production/reference math changed. Those historical next checks are superseded by0226–0227 below; full mathematical equivalence remains unclaimed.

**Update through0226–0227:** both independent block-state policies pass236 comparisons; all full-model active-state hashes/logits match at prefixes1/2/4/8/16. Latest consolidated command:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -m pytest -q \
  tests/test_020*.py tests/test_021*.py tests/test_022*.py tests/test_0106*.py
```

**1492 passed,186 skipped in16.75s** on the final rerun (earlier run:16.54s). This overlaps earlier totals; do not add them. Separate real-checkpoint block replays and sequential CuPy runs passed. New diagnostic files: `tests/fixtures/qwen4exp_block_state_reference.cpp`, `scripts/probe_0226_identical_block_states.py`, `scripts/probe_0227_short_sequences_and_profile.py` and their focused tests. No production source changed.

The short-text validation gate is passed. Initial profile: mean instrumented token3.6965s; routed expert span1.55152s, complete GDN1.36484s, complete QSA0.55624s. These include waiting/nested work, not exclusive critical-path costs. Next: separate projection fallback from mixer core overhead and validate packed mixed-format GPU expert kernels against retained oracles. Fixed-slot asynchronous movement and long-context validation remain pending.
