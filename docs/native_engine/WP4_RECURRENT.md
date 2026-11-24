# WP4 recurrent native runtime integration

## Result and scope

A complete **batch-one recurrent mixer layer step** is independently validated,
including all five projections at the original checkpoint geometry. This is not
merely a projected-input primitive. It is **not full WP4 completion**: PLE,
private HC/branch chaining, decoder orchestration and end-to-end model parity
remain outstanding. Parent-owned status/manifest files were not changed.

Files added:

- `mini_llm/native_cpp/recurrent.cu`: reused validated decode arithmetic plus
  converter-adjusted coefficients and intermediate finite-status checks.
- `mini_llm/native_cpp/recurrent.h`: private launch interface.
- `mini_llm/native_cpp/recurrent_runtime.inc`: ownership, bounded storage,
  staged state transaction, ABI implementation and explicit diagnostics.
- `tests/test_native_engine_recurrent.py`: 24 recurrent cases.

Minimal integration additions are in `mini_llm/native_cpp/runtime.cpp`,
`runtime.h`, `CMakeLists.txt` and `mini_llm/native/__init__.py`. Existing HC,
QSA, projection kernels and Python production model dispatch were not rewritten.
The runtime header's older synthetic-runtime overview does not describe this new
operator; the recurrent ABI comment and this document specify its scope.

## Authority and inspected graph

Authority is the existing `mini_llm/external/qwen4exp_mixer.py`, not another
model's similar recurrence. Also inspected:

- `mini_llm/ops/gated_delta_inference.py` and `gated_delta_decode.py`;
- `mini_llm/ops/qwen4exp_elementwise_gpu.py`;
- `mini_llm/external/qwen4exp_recurrent_block_graph.py` and
  `qwen4exp_pre_moe_graph.py` for state binding and HC/mixer ordering;
- native QSA projection/layer and HC ownership modules;
- the actual uploaded GGUF descriptors, not just model names or assumed formats.

The three kernels' arithmetic was copied from the validated
`gated_delta_decode.py` source. Only the static head map is substituted with
`h % Hk`, status checks are added, and native launch wrappers are used. No
similar-other-model implementation is imported. The existing generic
`ml_qsa_projection_launch` is reused unchanged; **no new dequantizer was needed**.

## Actual layer-0 descriptors

Checkpoint is the two-shard model identified by
`docs/native_engine/WP2_GGUF_INVENTORY.json`, starting at:

```text
/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
```

Canonical row-major shapes below reverse the actual GGML fastest-first shape.
Native APIs validate the latter directly.

| Tensor under `blk.0.` | Canonical shape | Stored type | Row bytes |
| --- | --- | --- | ---: |
| `attn_qkv.weight` | [10240,2560] | Q6_K (14) | 2100 |
| `attn_gate.weight` | [6144,2560] | Q4_K (12) | 1440 |
| `ssm_beta.weight` | [48,2560] | BF16 (30) | 5120 |
| `ssm_alpha.weight` | [48,2560] | BF16 (30) | 5120 |
| `ssm_out.weight` | [2560,6144] | Q6_K (14) | 5040 |
| `ssm_conv1d.weight` | [10240,4] | F32 (0) | 16 |
| `ssm_norm.weight` | [128] | F32 (0) | 512 |
| `ssm_a` | [48] | F32 (0) | 192 |
| `ssm_dt.bias` | [48] | F32 (0) | 192 |

Metadata gives Dmodel=2560, Hk=16, Hv=48, head dimension=128, taps=4,
inner=6144. `Qwen4ExpMixerSpec.from_metadata` supplies the exact epsilon and
geometry to the test. Checkpoint tests assert all nine actual formats.

Public projection format admission is the existing
F32/BF16/Q4_K/Q5_K/Q6_K/IQ4_XS set. Actual-checkpoint recurrent validation
covers Q4_K, Q6_K and BF16; synthetic recurrent tests cover F32 and BF16.
Other packed formats retain their existing projection tests, but no additional
recurrent checkpoint coverage is claimed for them. Controls and activations
must be F32. F16 and BF16 controls are not silently decoded.

## Computational graph and mapping

All activations, recurrence storage and arithmetic are FP32. The only arithmetic
mode is explicitly named `reference_fp32`. Projection reconstructs lane-local
weights and multiplies **unquantized F32 input**; no Q8 activation path is used.

1. Project hidden independently to QKV, Z, beta logits and alpha.
2. Depthwise causal convolution over raw oldest-first history and current QKV;
   then SiLU. Channel packing is global `[all Q | all K | all V]`.
3. Sum-based L2 normalize Q and K with `sqrt(sum(x*x)+epsilon)`.
4. `beta = exp(-logaddexp(0,-beta_logits))` and
   `log_decay = A * logaddexp(0,alpha+dt)`.
5. For each value head h, use key head **h % Hk**, not h divided by the
   value/key ratio. Q alone is additionally scaled by `1/sqrt(head_dim)`.
6. `Mdecay = M * exp(log_decay)`;
   `prediction = k @ Mdecay`;
   `correction = (v-prediction)*beta`;
   `Mnext = Mdecay + outer(k,correction)`;
   `readout = scaled_q @ Mnext` (read UPDATED memory).
7. Per-value-head RMS normalize readout, multiply converted gamma, then
   sigmoid(Z), with FP32 multiplication boundaries preserved.
8. Apply output projection before validating status and committing state.

A is **already negative -exp(A_log)** and must not be exponentiated again.
Gamma is already converted and gets **no extra +1**. All V-indexed controls,
conv channels, gates and output columns must already use converter tiled order.
There is no extra permutation of matrix state. Physical matrix indexing remains
`((h*Dkey)+key)*Dvalue+value`, exactly logical `[Hv,Dkey,Dvalue]`; tiling only
changes thread work ownership. Conv history remains `[taps-1,channels]` oldest
first, containing raw projected values, never SiLU values or a hidden ring layout.

Coefficient softplus reproduces the inspected CuPy FP32 logaddexp routine,
including the equality branch's double `log(2.0)` literal. CUDA builds use the
existing `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true` flags.

## API, bounds, ownership and failure behavior

`Engine.recurrent_layer(spec, capacity, qkv, z, beta, alpha, out, conv_weight,
norm_weight, ssm_a, dt_bias, arithmetic='reference_fp32')` returns a native owner.
It provides `step(uploaded_hidden, position)`, `position`, `reset`, `close` and
explicit test-only `readback_for_test('output'|'history'|'matrix', destination)`.
The ABI is `ml_recurrent_*` in `runtime.h`.

- Batch=1; Dmodel 1..2560, Hk 1..16, Hv 1..48 divisible by Hk,
  dimension 1..128, taps 1..16, positive uint64 capacity. Packed column alignment
  is validated before any projection launch.
- Capacity bounds committed positions, not allocation size. State is constant
  size with respect to token count. Position must exactly equal committed length.
  Capacity exhaustion is ML_CAPACITY; position mismatch is ML_INVALID.
- Nine uploaded tensors are retained natively with their engine, independently
  of public upload/store/engine closes. Hidden upload is retained through stream
  completion, including error unwinding. No raw device pointers escape.
- One Engine stream; all calls use the parent runtime lock, device guard,
  completion guard and status/error boundary. No private error subsystem or
  Python per-operation native dispatch is introduced.
- No allocation inside step/reset. Stable committed and staged history/matrix,
  projected vectors, conv/coefficient/readout/gated/output scratch and status are
  allocated once and charged to Engine. A pinned four-byte status is also charged.
- Each step copies committed history/matrix to staging, runs the whole graph,
  checks scalar status, and only then copies staging back to stable committed
  addresses. Numerical validation failure preserves prior committed state and
  position, invalidates output, and allows a subsequent valid step. Metadata
  failures likewise never commit. Committed-length output is zeroed on ABI error.
- CUDA failures during execution or commit poison state until reset; diagnostic
  reads cannot expose potentially partially committed buffers. Reset is itself
  poisoned until its stream completes successfully. Injected CUDA-fault recovery
  has not been tested.
- Controls are finite checked at creation. Converted A's strict negative-value
  contract is checked by the coefficient kernel on each step, not at creation.
  Projection sources/results and conv/readout/gated/staged states are checked.
  L2 and RMS reduction intermediates are checked, so overflow cannot be hidden
  by an infinite normalization denominator producing finite zeros.
- Reset zeros all state/scratch, sets position=0 and invalidates output. Empty
  zero history/matrix diagnostics are available before any successful step.
  There is no state-import API, prefill API, batching, or external pointer binding.

With C=(2Hk+Hv)*dimension, I=Hv*dimension, H=(taps-1)*C and
M=Hv*dimension*dimension, workspace payload is:

```text
4 * (2H + 2M + 2C + 3I + 4Hv + Dmodel + 1) device bytes
4 pinned bytes
```

Uploads are additionally charged at their exact stored sizes. No full dense
packed-weight expansion exists in native execution.

## Exact executed validation / verified state

Device 0: NVIDIA GeForce RTX 5080. Build directory and shared library are the
existing `/tmp/mini-llm-wp1-build` artifacts. All commands ran from repository root.
No tests or tolerances were loosened following results.

Build (120-second bound):

```sh
cmake --build /tmp/mini-llm-wp1-build -j 4
```

**PASS.** Three pre-existing misleading-indentation warnings in
`gguf_loader.cpp` remain; no new recurrent warning reported.

Focused tests (180-second bound):

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 .venv/bin/python -m pytest -q -s tests/test_native_engine_recurrent.py -x
```

**Latest result: 24 passed in 2.34 s**, including child checkpoint
**1 passed in 1.41 s**. The first focused execution had 9 passing cases then
failed the overflow atomicity case because a nonfinite normalization reduction
could produce finite normalized zeros. The evidence-supported local fix added
status checks at L2/RMS reduction boundaries without changing equations or
thresholds. The same focused command then passed in full. No unresolved failure.

Full native regression with all existing complete-QSA checkpoint cases enabled
(180-second bound):

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 MINI_LLM_QSA_LAYER_CHECKPOINT=1 .venv/bin/python -m pytest -q tests/test_native_engine*.py -x
```

**472 passed, 3 skipped in 54.43 s.** Preserves all previous 448 passing cases,
including the 52 HC cases and enabled QSA checkpoint cases, and adds 24 recurrent
cases. The same three existing optional timing/projection/hidden-frontend cases
remain skipped. This does not assert that every repository test was run.

CuPy focused suite under CUDA memcheck (180-second bound):

```sh
MINI_LLM_BACKEND=cupy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q -s tests/test_native_engine_recurrent.py -x
```

**24 passed in 4.68 s; ERROR SUMMARY: 0 errors.** In this run the whole focused
suite, not only the checkpoint oracle, used actual CuPy references.

Also passed: `git --no-pager diff --check` restricted to edited tracked runtime,
header, CMake and binding files; Python compilation of binding and new tests.

### Coverage

- F32/BF16 projections; geometry tuples (Dmodel,Hk,Hv,dim,taps):
  (1,1,1,1,1), (7,2,6,5,4), (33,2,6,33,3), (65,3,9,128,4).
- Twelve nonconstant successive steps, every output and both persistent states
  compared to unchanged Python mixer; reset/zero-state comparison and three
  repeated steps. Non-power-of-two heads/dimensions and tiled sharing covered.
- Stable engine accounting across all steps; exact budget admission boundary,
  allocation failure cleanup and release accounting; retained ownership after
  public weights and engine close; stale handles and no-op zero destruction.
- NaN hidden, finite-input overflow, wrong position, wrong shape, stale upload,
  cross-engine input: no committed history/matrix/position changes; unavailable
  failed output, valid retry. Capacity exhaustion after 12 steps also preserves
  committed state. Invalid diagnostics and destination dtype rejected.
- Wrong control shape/format, nonfinite control, nonnegative converted A,
  unsupported arithmetic, zero capacity and over-limit geometry rejected.
- Synthetic tolerances fixed at rtol=3e-5, atol=3e-6, FP32 throughout.

### Mandatory actual-checkpoint gate

No optional skip flag. Under the NumPy broad suite, this test launches a bounded
120-second child with `MINI_LLM_BACKEND=cupy`; it explicitly asserts real CuPy and
will fail rather than substitute a CPU oracle. Under memcheck it runs directly
in the instrumented CuPy process.

Layer 0's actual nine weights drive the complete native graph. The unchanged
Python `Qwen4ExpRecurrentMixerInference` uses existing `quantized_linear` /
`DevicePackedGGUFWeight` GPU packed FP32 projections, lossless GPU BF16 expansion,
and the validated GPU delta decode kernels. Checkpoint bytes cross host as exact
stored bytes only; **no CPU numerical dequantization/math fallback**. Neither
oracle results nor native diagnostics feed the native step.

Sixteen successive steps, then reset and four repeated steps, compare output,
raw oldest-first conv history and logical key-by-value matrix after every step.
Original full geometry is used. Checkpoint tolerances predeclared at
rtol=2e-4, atol=2e-4, consistent with the existing complete-projection FP32 envelope;
actual errors were far smaller. Maximum absolute errors in both successful runs:

| Snapshot | Max absolute error |
| --- | ---: |
| Output | 3.5390257835388184e-08 |
| Raw conv history | 1.7136335372924805e-07 |
| Matrix memory | 1.1175870895385742e-08 |

Exact native engine accounting: **50,777,732 device bytes + 4 pinned bytes**;
peak accounted payload **50,777,736 bytes**. The test additionally measures CUDA
free-memory change including its GPU oracle/pools: **184,549,376 bytes** in the
focused child and **111,149,056 bytes** under memcheck. Both explicitly assert
less than **512 MiB GPU extra**; native Engine budget is capped at 128 MiB. These
free-memory deltas are observations of the bounded test process, not a profiler
claim about full-model peak memory. Only one actual recurrent checkpoint layer
was tested; no all-36-recurrent-layer parity claim.

## Concrete next task and remaining nonclaims

**Next task: implement the native bounded PLE step from the existing Qwen4Exp PLE
reference, preserving converted gamma, causal state/history and residual order;
validate actual checkpoint PLE state/output parity under the same budget and
failure-atomicity rules.** Do not mark WP4 complete before PLE is validated.

After PLE, add private device-buffer orchestration connecting HC read to native
GDN/QSA/PLE inputs and branch outputs back to HC write, without diagnostic
readback/reupload. This recurrent API currently consumes an uploaded F32 hidden
vector and owns its output privately; it does not yet consume HC's hidden pointer
or feed HC write directly. Existing Python captured block/chain graphs were
inspected but not replaced or wired to this separate Engine API.

Further work: full decoder/FFN/MoE ordering, session orchestration, actual combined
block/logit trajectories, long runs/all recurrent layers, prefill/batching,
performance/capture/asynchronous scheduling, fault injection and production
selection. No speedup, full-model residency feasibility, external-engine exact
execution parity, BF16 activation arithmetic or full WP4 completion is claimed.
