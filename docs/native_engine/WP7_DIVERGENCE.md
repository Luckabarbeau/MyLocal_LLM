# WP7 divergence diagnosis — identical-input MoE evidence

Date: 2026-11-24. **Diagnosis only, not WP7/WP2 acceptance closure.**
Parent-owned status/manifest files were not modified. No production numerical
operation, tolerance, placement, or quantization policy was changed.

**Full-48 follow-up is now complete: see section 8 and
[`WP7_FULL_MODEL_DIAGNOSTICS.json`](WP7_FULL_MODEL_DIAGNOSTICS.json).**
The earlier unavailable-lineage blocker is resolved; the strict whole-model gate
still fails. Sections 1–7 preserve the earlier prefix/synthetic evidence rather
than silently replacing its scope or verified state.

Machine-readable results, including exact selected IDs/weights, all ten per-expert
error records, boundary metrics, quantization changes and library/source hashes:
[`WP7_DIVERGENCE_RESULTS.json`](WP7_DIVERGENCE_RESULTS.json).
Local raw activation/term snapshots: `logs/wp7_divergence/{actual,synthetic}/`.
The NPZ files contain diagnostics, not decoded model weight banks.

## 1. Starting failures, read before debugging

Latest recorded full gates in `WP7_DECODER.md`:

| Gate | First violating boundary | max abs | relative L2 | violations |
| --- | --- | ---: | ---: | ---: |
| Actual 48 layers, token 0 (fixed token ID 1) | layer 7 `hc_ffn_read` | 0.00155842304 | 0.000535675091 | 719 / 2560 |
| Original reduced 48-layer synthetic fixture, token 0 | layer 15 `post_moe` | 0.0569190979 | 0.0000434425855, measured in this diagnosis | 135 / 1024 |

All comparisons are FP32, fixed `rtol=2e-4, atol=2e-4`. No finite differences are
involved and there is no finite-difference step size. Componentwise relative
errors near zero are retained in JSON but are not used alone to declare a defect.

The synthetic diagnostic capture **reproduced the original failure exactly**:
max error `0.056919097900390625`, max componentwise relative error
`0.29268279189165264`, 135 violating entries. At the max-error location `[3,171]`,
native is `-30.713623046875`, Python is `-30.77054214477539`.

## 2. What was added, without altering execution math

Existing public MoE diagnostics returned input, output, selected IDs and weights,
but not router logits or individual expert terms. Added read-only C ABI:

- `ml_moe_diagnostic_for_test(owner, which, out, count)`.
- `ml_decoder_moe_diagnostic_for_test(decoder, layer, which, out, count)`.

They copy already existing completed-epoch buffers; **no diagnostic allocation,
new kernel, arithmetic, callback, or copy was inserted into step/submit/finish**.
Decoder access additionally requires enabled trace and a successful latest step.
Busy/invalid/reset states, null destinations, insufficient capacity and invalid
selection are rejected. IDs are copied as int32; all numerical buffers are FP32.

Captured: input, 512 router logits, 512 probabilities, exact ascending IDs and
once-scaled weights, ten CPU **unweighted** rows, ten weighted terms, sequential
ordered routed sum, shared gate/up/hidden/raw/score/output and combined output.

The existing decoder trace supplies the original FFN hidden, post-mixer streams
and post-MoE streams. Standalone HC read/write on retained actual weights supplies
paired gains and HC-write checks. Both native and Python replayed HC writes are
array-equal to their corresponding captured trajectory writes, separately.

Implementation files:

- `mini_llm/native_cpp/moe_runtime.inc`: shared read-only diagnostic helper.
- `mini_llm/native_cpp/decoder_runtime.inc`: gated retained-owner access.
- `mini_llm/native_cpp/runtime.h`, `mini_llm/native/__init__.py`: ABI declarations.
- `tests/native_decoder_divergence.py`: independent capture/replay/report phases.
- `tests/test_native_engine_divergence.py`: diagnostic safety/decomposition and
  retained identical-input evidence tests.

### Four-way experiment

Let `xN` be native decoder-trace FFN hidden and `xP` Python FFN hidden. Both are
retained as independent FP32 arrays, never substituted into the full trajectories.

1. Native standalone MoE and Python MoE both consume **the exact same xN bytes**.
2. Native standalone MoE and Python MoE both consume **the exact same xP bytes**.
3. Compare xN versus xP outputs **within each implementation**.
4. Compare standalone native replay to the native decoder's original buffers.

The Python diagnostic decomposition uses the unchanged router and GGML row pool,
not dense expert dequantization. It verifies array equality of its ordered sum
against `ExpertsInference.forward`, and of recombined output against
`Qwen4ExpMoE.__call__`. The native decomposition also satisfies exact
`terms = rows * weights`, sequential sum, and `output = routed + shared`.
No native diagnostics become inputs to the independent Python trajectory.

## 3. Synthetic: established cause is CPU Q8_K sensitivity, not route flip

### Identical-input results

| Quantity | Same xN: max abs | Same xP: max abs |
| --- | ---: | ---: |
| Router logits | **0** | **0** |
| Selected weights | **0** | **0** |
| All ten CPU unweighted rows | **0, every expert** | **0, every expert** |
| Weighted terms / ordered routed sum | **0** | **0** |
| Shared output | 1.74622983e-10 | 8.14907253e-10 |
| Combined MoE output | **0** | 2.98023224e-8 |
| HC write on the same lineage's streams | **0** | 7.62939453e-6 |

There are zero violations of the unchanged envelope for every identical-input
stage. Same-xP HC-write relative L2 is only `6.61257484e-10`; its residual streams
are hundreds in magnitude. Native capture versus native standalone replay is
array-equal, including original HC write. Python reconstructed HC write is also
array-equal to the Python trajectory.

Exact selected IDs, in both paths and both lineages:

```text
[2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
```

Every weight is exactly FP32 `0.10000000149011612`. The fixture has exact router
ties: tenth/eleventh margin is **zero**, but membership is unchanged. For xN,
tenth and eleventh logits are both `61.079524993896484`; for xP, both are
`61.07949447631836`. This is **not** evidence of a route flip.

### Different hidden lineages

| Quantity, xN versus xP | max abs | relative L2 |
| --- | ---: | ---: |
| FFN hidden | 2.98023224e-7 | 1.43439243e-7 |
| Router logits | 3.05175781e-5 | 4.99637004e-7 |
| All CPU rows | 0.1550350189 | 0.00549320972 |
| Ordered routed output | 0.05300712585 | 0.00550009074 |
| Shared output, native | 6.40284270e-10 | 8.57168418e-7 |
| Combined output, native | 0.05300712585 | 0.00550011351 |

Both implementations produce the **same changed CPU rows** when switching from xN
to xP. The large difference exists even when comparing Python against itself on
the two inputs. It is not a native CPU expert implementation discrepancy.

Per-expert cross-lineage max absolute errors:

| Expert | max abs |
| --- | ---: |
| 2 | 0.117952347 |
| 3 | 0.146777153 |
| 4 | 0.109217644 |
| 5 | 0.155035019 |
| 6 | 0.149766922 |
| 8 | 0.0767927170 |
| 9 | 0.121177673 |
| 10 | 0.108151913 |
| 11 | 0.140169144 |
| 12 | 0.117952347 |

Every corresponding **same-input** error above is exactly zero.

### Direct quantizer evidence

The unchanged GGML `quantize_row_q8_K` produces three differing bytes in the one
256-element block: one scale byte, **one signed code**, one block-sum byte.
These are not three independent code flips. Q8_0 input encoding is unchanged.

| Q8_K block 0, lane 42 | xN | xP |
| --- | ---: | ---: |
| FP32 input | -0.8652728796005249 | -0.8652730584144592 |
| stored d | 0.012101721949875355 | 0.01210172101855278 |
| actual signed code | **-71** | **-72** |
| input / stored d, diagnostic double reconstruction | -71.4999801833521 | -71.50000046174718 |

The stored-scale ratios illustrate proximity to a rounding boundary; they are
not a newly implemented quantizer and do not replace the authoritative GGML call.
The actual returned code bytes are the evidence. Thus the original synthetic
failure is explained by upstream FP32 differences crossing a CPU activation
quantization boundary, followed by the unchanged expert/HC graph.

## 4. Actual layer 6: resource-limited prefix, identical-input MoE passes

### Resource boundary and exact scope

An attempted original full native capture failed with CUDA `out of memory`.
A masked device query found an independent `train_model.py` process, PID 3665537,
using **11334 MiB on the specified GPU1**; free device memory was about 4477 MiB.
That process was not terminated or modified. No command used GPU0 for execution.
The previous full reference NPZ files were no longer present in pytest's retained
temporary directories. Therefore the exact previous full-model xP was not
available for replay.

The cheapest available actual experiment used the **first seven actual checkpoint
layers at their original geometry**, including actual PLE at layer 1, actual QSA
at layer 3, all original HC/GDN/MoE weights, real embedding token ID 1, default
FP32 paged QSA and all-CPU mmap experts. A native seven-layer decoder owns the
entire prefix execution and exposes its retained layer-6 input. No boundary is
forced. The Python prefix runs the unchanged `_step(..., project_logits=False)`
through the same seven blocks; HC observers only retain arrays. No dummy logits
or invented model operation is used. Native final projection does not affect the
already recorded prefix boundary.

**This is not an actual full-48 acceptance run and is not the exact earlier
full-model Python hidden lineage.** The new actual prefix measurements must not
silently replace the original layer-7 failure. All raw prefix arrays are retained.

### Layout control that mattered

Source comparison identified a real diagnostic setup distinction:

- Full `qwen4exp_model._build`: `router.W_router = dense(...).T`, a transpose view.
- Existing selected-layer `CheckpointMoE.oracle`: `cp.asarray(router.T.copy())`,
  C-contiguous transposed storage.

The initial prefix used that existing helper unchanged. Both input lineages then
produced bit-identical CPU rows, with only one differing Q8_K **scale byte**, no
Q8_K code change. That control is retained in the JSON under
`actual_c_router_control`, and raw `python_c_router.npz` locally.

A localized **diagnostic-only** adjustment restored the full model's transpose
view layout, without changing router values, policy, equations or production
code. It changed upstream FP32 reduction results enough to alter where CPU
quantization boundaries were crossed. This is evidence that the prefix setup's
arithmetic/layout details matter; it is not permission to claim the original
full-model input was reconstructed exactly.

### Identical-input results with full-model router layout

| Quantity | Same xN: max abs | Same xP: max abs |
| --- | ---: | ---: |
| Router logits | 1.90734863e-6 | 9.53674316e-7 |
| Selected weights | 8.94069672e-8 | 5.96046448e-8 |
| All ten CPU unweighted rows | **0, every expert** | **0, every expert** |
| Ordered routed output | 1.86264515e-8 | 1.49011612e-8 |
| Shared output | 5.58793545e-9 | 5.12227416e-9 |
| Combined output | 1.86264515e-8 | 1.49011612e-8 |
| HC write | 7.45058060e-9 | 1.49011612e-8 |

Same-input combined-output relative L2 errors are `2.63998131e-7` and
`1.93029416e-7`. Every same-input stage has zero violations of the unchanged
`2e-4 / 2e-4` envelope. Native retained decoder buffers and standalone replay are
array-equal for every captured numerical buffer. Both captured trajectory HC
writes are independently array-equal to their corresponding reconstructed writes.

Exact IDs in native/Python and xN/xP:

```text
[33, 60, 77, 152, 201, 223, 256, 283, 285, 379]
```

For xN, native selected weights are:

```text
[0.20068180561065674, 0.049662671983242035, 0.031956423074007034,
 0.030876170843839645, 0.09923552721738815, 0.3941163420677185,
 0.02748999185860157, 0.09713044762611389, 0.04033082723617554,
 0.02851971797645092]
```

All four exact weight vectors, router logits and arrays are retained in NPZ;
JSON retains exact selected weights, errors and margins.

### Router margin: no observed route flip

Native xN tenth expert is 256, eleventh is 144:

```text
tenth logit       = -4.465839862823486
eleventh logit    = -4.466910362243652
margin            =  0.0010704994201660156
same-input max router-logit error = 1.9073486328125e-6
```

Python xN margin is `0.0010700225830078125`. Both xP implementations have margin
`0.0010709762573242188`. Across xN/xP, max router-logit difference is
`2.956390380859375e-5`, still below half of the observed selection margin.
Membership remains identical in every measured comparison. There is no evidence
of a router membership flip in this actual prefix experiment.

### Different hidden lineages and Q8_K boundary

| Quantity, xN versus xP, native implementation | max abs | relative L2 |
| --- | ---: | ---: |
| FFN hidden | 7.02142715e-5 | 2.55457994e-5 |
| CPU unweighted rows | 0.000465281308 | 0.00291011411 |
| Ordered routed output | 0.000127278501 | 0.00234175508 |
| Shared output | 5.93776349e-7 | 2.61307221e-5 |
| Combined output | 0.000127276406 | 0.00223933115 |

The same CPU row changes occur in the Python implementation. Per-expert max
absolute errors (cross-lineage, not same-input):

```text
33:  0.000416856259   60:  0.000207873061   77:  0.000465281308
152: 0.000251268968  201:  0.000271506608  223:  0.000191420317
256: 0.000174866989  283:  0.000239035115  285:  0.000326903537
379: 0.000203532167
```

Q8_K has 14 differing bytes across its ten blocks, including scale changes and
one changed signed lane. Q8_0 input has two differing bytes. Direct Q8_K evidence
at block 1 lane 206, global input coordinate 462:

| Field | xN | xP |
| --- | ---: | ---: |
| FP32 input | 0.26499247550964355 | 0.2649804949760437 |
| stored d | -0.01606004498898983 | -0.016060225665569305 |
| actual signed code | **-17** | **-16** |
| input / stored d, diagnostic reconstruction | -16.50010791945555 | -16.499176318806143 |

Again, identical input gives bit-identical native/Python CPU terms, while switching
input changes those terms in **both** implementations. This establishes CPU Q8
sensitivity in this actual prefix. It supports, but does not conclusively prove,
the same cause for the unavailable original full-model lineage.

The prefix's layer-6 post-MoE max error is `6.67721033e-5`, relative L2
`0.000208029207`, with zero componentwise-envelope violations at that boundary.
Earlier amplification begins at layer 5 in this prefix; the old full model's first
sharp recorded increase was layer 6. Those observations are explicitly distinct.

## 5. Policy conclusion: no justified numerical production fix

Established:

- Synthetic original divergence is reproducible and explained by upstream FP32
  roundoff crossing Q8_K activation thresholds, **not a route flip**.
- Both actual prefix input lineages pass identical-input native/Python MoE checks;
  all ten CPU terms are bit-identical in each check.
- Current actual prefix shows the same quantization-threshold amplification
  mechanism, stable routing, and negligible same-input GPU shared/HC differences.
- No missing/doubled expert, reordered term, wrong weight application, or incorrect
  HC-write path is supported by these measurements.

No numerical production change is justified. In particular, no epsilon bias,
quantization smoothing, expert omission, changed router tie policy or tolerance
relaxation was introduced. Existing failing full trajectory tests remain intact.
An end-to-end tolerance does not imply continuity through discrete quantization:
component-sized FP32 perturbations can change an integer code even with a large
router selection margin. Conversely, these findings do not themselves authorize
replacing a requested full-model acceptance gate with a component gate.

The current `cpu_ggml_q8_gpu_reference_fp32` policy is retained exactly. If the
acceptance contract ultimately requires bitwise trajectory identity, upstream GPU
reduction/layout order before CPU quantization must also be pinned or matched;
a downstream approximate comparison cannot guarantee identical Q8 codes.
That is a policy/engineering decision, not a mathematical defect proven here.

## 6. Verification and commands

All GPU commands used only:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
```

Each command below was externally bounded to 180 seconds. No full-suite retry or
longer timeout was attempted. There was no successful new full-48/two-token gate;
its resource-blocked capture failed before execution and the targeted prefix was
explicitly substituted only for diagnosis.

Common environment for capture/tests:

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_BACKEND=cupy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH=.:tests
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Build:

```sh
cmake --build /tmp/mini-llm-wp1-build -j 4
```

**PASS**. The three existing `gguf_loader.cpp` misleading-indentation warnings
were emitted; no new numerical kernel was built or changed.

Capture/replay phases executed successfully:

```sh
.venv/bin/python tests/native_decoder_divergence.py synthetic-capture logs/wp7_divergence/synthetic
.venv/bin/python tests/native_decoder_divergence.py synthetic-replay logs/wp7_divergence/synthetic
.venv/bin/python tests/native_decoder_divergence.py synthetic-report logs/wp7_divergence/synthetic
.venv/bin/python tests/native_decoder_divergence.py actual-prefix logs/wp7_divergence/actual
.venv/bin/python tests/native_decoder_divergence.py actual-replay logs/wp7_divergence/actual
.venv/bin/python tests/native_decoder_divergence.py actual-report logs/wp7_divergence/actual
```

The actual prefix was rerun only after the concrete router-layout setup correction;
the previous C-router control was preserved. Report-only regeneration adds code
threshold details from the same saved arrays and does not rerun GPU trajectories.

Focused acceptance:

```sh
.venv/bin/python -m pytest -q tests/test_native_engine_divergence.py tests/test_native_engine_decoder.py::test_last_layer_failure_global_position tests/test_native_engine_decoder.py::test_factory_budget_preflight -x --tb=short
```

**6 passed in 3.91 s**. New tests verify buffer gating, unchanged allocation
accounting, exact decomposition, retained boundary input, reset rejection, and
all ten same-input CPU rows plus routed/shared/HC outputs for both experiments.
Retained-evidence cases skip if their explicitly captured local NPZ files are
absent; they ran and passed here, not skipped.

The first combined diagnostic test attempt aborted after sequential native-only
owners reloaded GGML. The decoder diagnostic test passed alone in a fresh process
(`1 passed in 1.33 s`). A test-module-only `ctypes.CDLL` keepalive then allowed
all six tests to pass together. No speculative production CPU-library lifetime
change was made. The earlier WP7 library-reload issue is not claimed fixed.

Broader, bounded relevant regression:

```sh
MINI_LLM_MOE_CHECKPOINT=1 .venv/bin/python -m pytest -q tests/test_native_engine_divergence.py tests/test_native_engine_moe.py tests/test_native_engine_moe_chaining.py tests/test_native_engine_cpu_experts.py tests/test_native_engine_hc.py -x --tb=short -rs
```

**129 passed in 24.57 s**, no skips. This includes the existing actual-checkpoint
MoE/chaining cases and preserves their verified math. It is not a new full-suite
result, not long-context quality validation, and not the still-open whole-decoder
trajectory acceptance.

## 7. Single next experiment for the remaining exact-original-lineage question

When GPU1 has sufficient free memory, capture the original full decoder and
unchanged full Python model in fresh sequential processes using the helper's
`actual-native` and `actual-python` phases, then run `actual-replay` and
`actual-report`. Keep all 48-layer boundary snapshots, token ID 1, original
constructor/layout and policy. Compare layer-6 xN/xP and each CPU term exactly as
above. This will distinguish whether the original layer-7 violation is the same
Q8 threshold mechanism or a discrepancy unique to that full-model lineage.

Do not use GPU0, terminate the other training process, silently replace the
original full-model test, or alter tolerances to obtain a green result.

## 8. Full-48 follow-up with fresh retained original lineages

### New environment evidence and admission

A new GPU1-only query showed **13089 MiB free**, 2752 MiB used, with the existing
training process PID 3665537 using 2722 MiB. It was not stopped, signalled,
reconfigured or otherwise controlled. GPU0 was not used.

Before each full capture, the helper validates the exact UUID mask, sole visible
logical device 0, and `cudaMemGetInfo` against a **fixed 10 GiB budget plus 2 GiB
additional headroom**. This is a fail-before-weight-allocation check, not an
automatic reduced-budget mode or a numerical fallback.

| Phase | CUDA free after context creation | Fixed budget + headroom | Excess |
| --- | ---: | ---: | ---: |
| Native full capture | 12859.3125 MiB | 12288 MiB | 571.3125 MiB |
| Python full capture | 12691.3125 MiB | 12288 MiB | 403.3125 MiB |

The native phase completed and exited before the Python process started. Native
weights/maps/session/CPU pools close at context exit; process exit releases its
CUDA context/pools as well. A GPU1-only query between phases showed 12919 MiB
free. The Python phase independently passed the same admission. No overlapping
full native/Python model residency, process control, long retry or GPU0 fallback.
Requested budgets and admission are not measured whole-process peak accounting.

### Captures and reset: complete actual graph, not the prefix

New artifacts are separate from the earlier prefix and synthetic captures:

```text
logs/wp7_divergence/actual_full/native.npz
logs/wp7_divergence/actual_full/python.npz
logs/wp7_divergence/actual_full/replay.npz
logs/wp7_divergence/actual_full/token1/{native,python,replay}.npz
logs/wp7_divergence/actual_full/layer7_token1/{native,python,replay}.npz
logs/wp7_divergence/actual_full/gdn8/states.npz
```

Both full model implementations execute the original **48 actual layers**, fixed
tokens `(1,7)`, original geometry/schema/constructor policy, all-CPU mmap routed
experts, FP32 GPU branches and FP32 paged native QSA. All 192 layer-boundary arrays
per token and all 248320 logits are retained. The full Python factory is unchanged;
there is no shortened reference, substituted boundary, forced route or hidden
fallback. Extra standalone MoE diagnostic jobs do not feed the trajectory.

Native reset and the same two-token replay are **array-equal for every logit**;
position commits `(1,2)`, resets to 0, then commits `(1,2)` and resets to 0 again.
Layer-6 native owner buffers are independently array-equal to standalone replay,
including its paired HC write. Each Python HC write also reconstructs its own
trajectory exactly. The previous prefix/synthetic artifacts and JSON remain
unchanged.

### Original failure reproduced exactly; strict acceptance still FAILS

| Token index / ID | First violating boundary | max abs | relative L2 | violations |
| --- | --- | ---: | ---: | ---: |
| 0 / 1 | layer 7 `hc_ffn_read` | 0.0015584230422973633 | 0.0005356750911888023 | 719 / 2560 |
| 1 / 7 | layer 8 `hc_ffn_read` | 0.000667273998260498 | 0.00032932771588950623 | 287 / 2560 |

Token 0 reproduces the earlier full-model failure exactly, including the earlier
layer-6 hidden error `8.344650268554688e-7`. At token 0's maximal violating
coordinate 624, native is `-0.0877465307712555`, reference is
`-0.08930495381355286`. For token 1's maximal violating coordinate 1971, native is
`-0.8229811191558838`, reference is `-0.8236483931541443`.

All dtype/tolerance reports remain FP32, `rtol=2e-4, atol=2e-4`. The original strict
test and its assertions are unchanged, not marked xfail, relaxed or replaced.
The retained full captures independently evaluate the same boundary/logit rule
and record `strict_whole_model_acceptance_passed=false`. Diagnostic tests passing
below **do not close WP7/WP2 or production quality acceptance**.

### First dense HC difference, isolated on an identical embedding

The native standalone first attention HC read consumes the exact Python-decoded
embedding row. Its hidden output is array-equal to the full native layer-0 trace.
The independently repeated Python first HC read is array-equal to the full Python
layer-0 trace. Thus this measurement isolates dense HC arithmetic before GDN,
QSA, routing or CPU quantization; it is not an embedding-codec discrepancy.

| First attention HC, identical input | Token ID 1 | Token ID 7 |
| --- | ---: | ---: |
| Hidden max abs | 4.76837158203125e-7 | 1.430511474609375e-6 |
| Hidden relative L2 | 6.938541844042036e-8 | 1.1067949244205178e-7 |
| Gains max abs | 5.587935447692871e-9 | 3.725290298461914e-8 |
| Violations of fixed envelope | 0 | 0 |

Example token-0 hidden coordinate 747: native `-1.7923853397369385`, Python
`-1.7923848628997803`. No wrong HC equation or factor is established. The native
FP32 reduction and Python dense GPU path are close, but not bit-identical.

### Full token 0, layer 6: same-input contract passes; Q8_K lane flips

| Same-input quantity | Both consume native hidden: max abs | Both consume Python hidden: max abs |
| --- | ---: | ---: |
| Router logits | 1.90734863e-6 | 1.90734863e-6 |
| Selected weights | 8.94069672e-8 | 5.96046448e-8 |
| Every one of ten CPU unweighted rows | **0** | **0** |
| Ordered routed output | 1.86264515e-8 | 1.11758709e-8 |
| Shared output | 5.58793545e-9 | 5.12227416e-9 |
| Combined output | 1.86264515e-8 | 1.11758709e-8 |
| Paired HC write | 7.45058060e-9 | 7.45058060e-9 |

All identical-input stages have zero violations. Exact IDs in all four cases:
`[33,60,77,152,201,223,256,283,285,379]`. Tenth/eleventh native margin is
`0.0010704994201660156`; the other three margins are approximately `0.00107002`
to `0.00107050`. Router-logit differences are orders smaller; there is no route
membership flip.

Across the two actual hidden lineages:

- Hidden max abs `8.34465027e-7`, relative L2 `2.27641651e-7`.
- CPU-row max abs `0.000377766788`, relative L2 `0.00264213406`.
- Routed max abs `0.000149392523`, relative L2 `0.00222906445`.
- Shared max abs only `8.49831849e-9`.
- Layer-6 post-MoE stream max abs `7.907301187515259e-5`, followed by the exact
  layer-7 FFN-read violation above.

Both native and Python produce the same changed CPU rows when the input switches
lineage. The unchanged GGML Q8_K encoder returns five differing bytes: scale
changes in blocks 2/3/9, one changed code, and its block sum. Q8_0 input bytes are
unchanged. The actual full-model threshold is **not** the earlier prefix's lane:

| Q8_K block 9, lane 190, input coordinate 2494 | Native lineage | Python lineage |
| --- | ---: | ---: |
| FP32 input | 0.4080895185470581 | 0.4080897271633148 |
| stored d | 0.014318935573101044 | 0.014318937435746193 |
| Actual signed code | **28** | **29** |
| Input / stored d, diagnostic reconstruction | 28.499989853552947 | 28.500000715454508 |

The actual encoded bytes, not a reimplemented quantizer, establish the threshold
crossing. This now resolves the original full-model-lineage question from section
7: the measured first amplification is CPU Q8 sensitivity to upstream FP32
roundoff, not a same-input CPU MoE implementation defect or a route flip.

### Token 1: recurrent state carry, not another layer-6 route/term change

At layer 6 for token ID 7, hidden max error is `1.22189522e-6`; all ten CPU rows are
identical **even across lineages**. Identical-input comparisons again have zero
CPU-row error and combined-output max error at most `1.49011612e-8`.

A cheap independent layer-7 MoE replay from the retained full-model inputs also
shows unchanged CPU rows across lineages and stable IDs. Same-input output and HC
write errors are at most `1.49011612e-8`. Its native and Python paired HC writes
exactly reconstruct their respective captured writes. Original layer-7 internal
owner buffers were not retained: that limitation is explicitly flagged in JSON;
its original aggregate HC write, not an invented owner snapshot, is the endpoint
check. This experiment rules out the initially suspected layer-7 CPU amplification.

Layer 8 is recurrent. Its prior token's attention hidden had already diverged by
`0.00327372551` because token 0's upstream Q8 amplification had propagated there.
Current token-1 attention hidden differs by only `4.29153442e-6`, but its GDN
**committed matrix and convolution history are not the same**.

A separate two-step replay feeds each full-model hidden history to both native
and unchanged Python layer-8 GDN, starting with zero state. Existing actual
weights and primitive APIs are reused; no full-model state is overwritten.

- Conditioned on either complete input history, native/Python GDN output max
  errors are at most `2.98023224e-8`.
- Convolution-history errors are at most `9.53674316e-7`; matrix errors at most
  `5.96046448e-8`. All fixed-envelope checks pass.
- Native GDN plus paired HC write reconstructs the original full-native
  post-mixer streams **exactly for both tokens**. Native replay on the Python
  input history reconstructs its Python post-mixer streams within `1.49011612e-8`.
- Holding the current hidden input **identical**, but using the two different
  recorded prior input histories, the unchanged Python recurrence produces
  output difference `0.000124253333`, relative L2 `0.00130516763`.
- The actual two-lineage GDN output difference is `0.000124245882`.
  The history-only controlled delta differs from that actual delta by max
  `2.12341547e-7`; residual L2 is `0.000632699106` of the actual delta norm.

Therefore expecting token-1 GDN outputs to agree merely because their current
inputs are close would omit the forward dependency on carried state. This is
expected state propagation in the measured graph, not evidence of a recurrent
implementation error. The following HC read amplifies that stream difference
into the token-1 layer-8 strict violation.

### Mathematical invariant and diagnostic acceptance scope

No incorrect production mathematical operation has been identified in this
measured causal path. The tests check the actual dependency chain:

```text
streams -> HC read(hidden, gains)
        -> route + CPU GGML(Q8(hidden)) + GPU shared
        -> ordered weighted sum + shared
        -> HC write(original streams, combined output, same read's gains)

GDN: (prior matrix, prior conv history, current hidden)
        -> (next matrix, next conv history, branch output)
```

The invalid inference would be:

```text
close upstream FP32 inputs
    implies identical discrete Q8 codes
    implies the same end-to-end componentwise error envelope
```

That is not a mathematical property of quantization. It does **not** make the
requested strict gate invalid or passed: the implementation still fails that
engineering requirement. Matching that strict trajectory may require matching
upstream numerical operation/layout/reduction order before every quantization
boundary. No such production redesign or policy change was attempted here.

The added diagnostic acceptance has a narrower, independently justified contract:
for identical retained inputs, routing/weighting, all ten CPU terms, shared branch,
ordered combination and HC write match; for identical complete input history,
GDN history/matrix/output match. It does not substitute forced inputs into a
full model, waive a failed logit gate, accept model quality, or certify every
unexamined layer/input. Tests explicitly keep diagnostic and strict status
separate. Existing tolerances remain `2e-4 / 2e-4` and CPU-row equality is exact.

### Observed short-token distribution drift; quality acceptance remains open

| Metric | Token index 0 / ID 1 | Token index 1 / ID 7 |
| --- | ---: | ---: |
| Logit max abs | 0.1547901630 | 0.4191079140 |
| Logit relative L2 | 0.01158315705 | 0.03783137204 |
| Violating logits / 248320 | 244163 | 247078 |
| Greedy top-1 native/reference | 271 / 271 | 16 / 16 |
| Top-10 membership overlap | 10 / 10 | 10 / 10 |
| KL(Python || native), stable FP64 reporting | 0.0002736619931 | 0.003868193475 |
| Probability total variation | 0.008869246762 | 0.03791556623 |

These are direct diagnostics of the two retained fixed-token distributions,
**not** perplexity, generated-text quality, long-context equivalence or an accepted
quality envelope. The observed drift is consistent with repeated quantization
thresholds and state propagation; its magnitude was measured, not bounded in
advance by a continuity claim. Top-1 agreement on two samples cannot establish
model quality. Both JSON quality flags remain false.

No native forward benchmark or accepted throughput was published. Test runtimes
below are verification durations, not inference performance acceptance.

### Executed verification for this follow-up

All GPU commands retained the exact GPU1 UUID mask. Every capture/replay/test
command had an external **180-second bound**. The selected GGML DSO is retained
within the two-epoch standalone replay to avoid the previously observed reload
issue; this is diagnostic process lifetime management, not a production fix.

Using section 6's explicit environment, successful commands were:

```sh
.venv/bin/python tests/native_decoder_divergence.py actual-native logs/wp7_divergence/actual_full
.venv/bin/python tests/native_decoder_divergence.py actual-python logs/wp7_divergence/actual_full
.venv/bin/python tests/native_decoder_divergence.py actual-replay logs/wp7_divergence/actual_full
.venv/bin/python tests/native_decoder_divergence.py actual-report logs/wp7_divergence/actual_full
.venv/bin/python tests/native_decoder_divergence.py actual-layer7-token1 logs/wp7_divergence/actual_full
.venv/bin/python tests/native_decoder_divergence.py actual-gdn8 logs/wp7_divergence/actual_full
```

The report-only phase reads saved arrays; it does not repeat trajectories. The
first layer-7 diagnostic invocation caught a one-line indentation edit artifact
before execution; a localized correction fixed it. No numerical runtime code
changed and no production rebuild was necessary for this follow-up.

Focused verification:

```sh
.venv/bin/python -m pytest -q tests/test_native_engine_divergence.py tests/test_native_engine_decoder.py::test_last_layer_failure_global_position tests/test_native_engine_decoder.py::test_factory_budget_preflight -x --tb=short
```

**11 passed in 4.54 s**, no skips. New tests cover both full-model token inputs,
layer-7 token-1 conditioned replay, state-conditioned GDN8, full-logit/status
separation and the recorded native reset/preflight contract. Earlier prefix and
synthetic evidence checks still pass unchanged.

Broader bounded relevant subset:

```sh
MINI_LLM_MOE_CHECKPOINT=1 .venv/bin/python -m pytest -q tests/test_native_engine_divergence.py tests/test_native_engine_moe.py tests/test_native_engine_moe_chaining.py tests/test_native_engine_cpu_experts.py tests/test_native_engine_hc.py -x --tb=short -rs
```

**134 passed in 25.18 s**, no skips. The prior timed-out full suite was not retried.
The original strict full-model test was not relaxed or relabelled; the fresh
retained full captures reproduce its first failure and evaluate every remaining
boundary/logit without aborting at the first violation.

The new JSON artifact retains all comparisons, exact IDs/weights, per-expert
metrics, preflight records, library/source/activation-file hashes and separate
strict/diagnostic status. Parent status and reference manifest are unchanged.
Full-model strict parity and longer model-quality acceptance remain OPEN.

Python compilation and `git diff --check` passed after the follow-up. The final
GPU1-only query showed PID 3665537 still running, now using 10898 MiB, and only
4913 MiB free. No diagnostic Python process remained in that compute-app listing.
Thus the other workload's memory use changed during this work; no further full
capture was attempted after that observation. Each successful full capture used
its own earlier recorded admission, not an assumption that free memory stayed
constant.
