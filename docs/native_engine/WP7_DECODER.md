# WP7 — native whole-decoder owner and bounded execution gate

Date: 2026-11-24.

Full-48 follow-up: `WP7_DIVERGENCE.md`, section 8, and
`WP7_FULL_MODEL_DIAGNOSTICS.json` retain the original two-token full-model
lineages and explain the first amplification. Conditioned diagnostic gates pass;
the unchanged strict whole-model gate still fails. This does not close WP7/WP2.

**Implementation advanced; acceptance OPEN. Do not mark WP7 complete or claim
full-model numerical parity.** The actual 48-layer checkpoint executes natively,
including two fixed tokens, reset, and exact native replay. Comparison to the
unchanged full Python reference fails the fixed numerical envelope. WP6 long
full-model quality remains independently open and was not used to block this
bounded FP32 implementation.

Parent-owned `IMPLEMENTATION_STATUS.md` and `native_reference_manifest.json` are
intentionally unchanged. Existing WP6 files and unrelated training work were
preserved.

## Implemented graph and authority

Authorities inspected: `qwen4exp_model.py` (`_step`, constructors),
`qwen4exp_weights.py::inspect_headers`, `qwen4exp_gguf.py`,
`docs/qwen4exp/reference_contract.md`, the native HC/GDN/QSA/PLE/MoE owners,
and their existing chaining/actual-checkpoint tests.

`mini_llm/native_cpp/decoder_runtime.inc` owns a fixed execution vector. Python
constructs it once; `NativeDecoder.step(token)` makes **one C ABI call**, not a
Python layer loop:

1. Copy only this token's IQ4_XS embedding row from the retained shard mmap;
   decode with the existing lane decoder and duplicate into four HC streams.
2. For each bound layer, optional PLE before attention.
3. Attention HC read -> recurrent/GDN or complete paged QSA -> paired HC write.
4. Independent FFN HC read -> native static MoE (shared GPU branch and all selected
   routed CPU GGML experts) -> paired HC write.
5. Final HC collapse -> vocabulary projection -> FP32 host logits `[1,248320]`.
   No additional conventional output RMSNorm, sampling, or Python numerical seam.

No HC, recurrence, PLE, QSA, router, expert, or reduction equation was redesigned.
The PLE ABI body was extracted into a shared private device-input helper; the old
uploaded/HC APIs still use the same body. Projection additions reuse the existing
FP32 128-lane projection reduction and existing lane decoders. Vocabulary dispatch
supports F32, IQ4_NL, Q8_0, IQ4_XS, Q4_K, Q5_K, Q6_K, with its own 248320-row bound;
ordinary layer projection limits remain unchanged. IQ4_XS selected-row decoding
uses the already present decoder. Q8_0 uses half scale times signed byte lanes.
The actual `output.weight` is **Q6_K**, not IQ4_NL: `(2560,248320)` GGML shape,
row bytes 2100, packed bytes 521472000. Actual embedding is IQ4_XS, row bytes
1360; its full 337715200-byte table is **not** uploaded.

The arithmetic policy remains explicitly
`cpu_ggml_q8_gpu_reference_fp32`. CPU GGML activation quantization is intentional;
this is not a claim that CPU routed experts perform pure FP32 dots. There is no
GPU-Q8 arithmetic substitution, expert tail omission, or fallback.

## Python setup and WP2 whole-model binding

`mini_llm/native/decoder.py::NativeDecoder.open`:

- Opens the existing native multi-shard store and calls its authoritative
  `validate_qwen4exp()` before touching weight payloads.
- Requires actual width 2560, vocab 248320, rank 320, 48 layers, and the validated
  36-GDN/12-QSA schedule. Uses the validated header shapes/specs for all bindings.
- Binds all layer weights, optional PLE, final HC and output into native retained
  owners. All routed expert banks and the PLE table remain immutable mmap leases.
- Creates native paged **FP32** QSA by default, with explicit `ggml_cpu` selection.
- Closes setup's public store, lease, upload, weight, Engine and transferred-state
  wrappers on return. The decoder's native owners retain everything required.
- Does not replace or alter the existing Python `Qwen4ExpGGUFModel` path.

This supplies the previously missing whole-model payload-consumer binding for
WP2. Actual construction and native execution reach every layer. **WP2 numerical
acceptance must not be declared closed on this evidence**: the first-boundary
whole-model comparison below still fails. This document does not update the
parent status on the parent's behalf.

### Ownership and sessions

A Decoder is a session, distinct from Engine lifetime. The low-level
`Engine.decoder(embedding, output, final_hc, hc, layers, capacity=..., trace=...)`
consumes fresh mutable HC/GDN/QSA/PLE/MoE handles on successful publication.
Their registry entries are removed and Python wrappers become closed; stale C
aliases cannot mutate them. Duplicate mutable bindings, wrong engine/width/rank,
used/invalid/busy state, inadequate capacity, and unsupported output formats are
rejected before transfer. Immutable weight owners remain shareable within their
compatible Engine.

`NativeDecoder.open` initially creates **one strict session per Engine**. Separate
opens do not share immutable allocations yet; do not promise concurrent session
weight sharing or a new state-only session factory. Underlying native ownership
retains Engine, stream, uploads, maps and CPU pools independently of public handles.
All ABI calls use the existing registry mutex/error boundary; no second status
system, per-layer Python callback, public device pointer, or nested ABI lock.

### Transaction and reset

`decoder.state` returns `(global_position, failed)`.

- Global position increments only after every layer, final collapse, finite
  projection validation, and completed logits copy succeed.
- Any step exception marks the decoder unresumable. Committed per-layer history
  is not rolled back and cannot be used to resume through another public handle.
- Reset clears all GDN history/matrices/scratch, QSA history/page backing and
  generations, PLE histories/predecessors/rows, MoE logical lengths and validity,
  HC streams/tickets/collapse status, decoder scratch and diagnostic trace.
- Reset itself keeps the decoder failed until its completion fence succeeds.
- Native replay of actual tokens `(1,7)` after reset was array-equal for all
  248320 logits on both steps. No generated continuation was sampled.

## Budgets and limits

Factory defaults (explicit overrides accepted, no auto-cache/fallback):

| Budget/configuration | Default |
| --- | ---: |
| Engine requested device+pinned budget | 10 GiB |
| Stored dense/projection uploads plus exact HC/control expansion cap | 7 GiB |
| Mapped immutable shard-length cap | 96 GiB |
| Pageable QSA host backing budget, divided among 12 QSA layers | 1 GiB |
| Active session capacity | 256 tokens |
| Maximum accepted active capacity / pagecap | 262144 tokens |
| Page size | 16 tokens |
| Physical pages per QSA layer | 2 |
| CPU threads per native layer pool | 4 |
| Routed GPU expert placement | empty: all 512 experts CPU mmap |

Factory preflight checks mapped shard lengths, dense uploaded/expanded bytes,
a conservative device-workspace bound, and host metadata plus one page **before
GPU payload allocations**. Native Engine admission additionally guards every
requested allocation. Host metadata includes the actual `KVPageRequests` layout:
`(3+2*2051)*8` bytes, plus logical table and slot arrays. Lazy host page growth is
budget-checked by the existing page owner; it can still exhaust an explicitly
small host budget during a long session and then poisons the decoder.

Actual bounded gate configuration: capacity 4, page size 16, one physical page,
2 CPU threads per pool. Printed preflight plan:

```text
dense_bytes       = 6734983680
workspace_bound   =  546845440
mapped_bytes      = 83617662656
gpu_budget        =10737418240
host_page_budget  = 1073741824
```

The device bound is under 12 GiB with explicit headroom, but this is requested
allocation accounting, **not a measured whole-process CUDA peak**. CUDA context,
allocator/driver overhead, library internals, mmap/page-cache RSS, and CPU worker
metadata are not Engine payload accounting. Mapped shard lengths are not decoded
RAM allocation. No full-model CPU decoded expert banks, expert pinning, full
embedding/PLE table GPU upload, or automatic expert promotion.

Trace is explicit diagnostics only: four layer-boundary arrays per layer,
fixed bounded host storage `layers*4*4*D*sizeof(float)` (7.5 MiB at actual geometry),
outside the QSA host-page budget. Production default is trace off. Traces never
feed native execution or the independent reference. `trace_for_test` is available
only after a successful step; failed/reset sessions cannot expose stale results.

## Executed gates and exact observed failures

All GPU tests/builds/benchmark opt-ins used only:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
```

Physical GPU1 is the sole visible CUDA device, logical 0. Every test command had
a 180-second external tool limit; the reference child also has a 180-second
subprocess limit. No timeout was followed by an automatic longer retry.

### Build

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 cmake --build /tmp/mini-llm-wp1-build -j 4
```

Succeeded, including the final Q6_K vocabulary-dispatch correction. The first
build repeated the three existing `gguf_loader.cpp` misleading-indentation
warnings. No warning-free whole-repository build claim.

### Common test environment

Commands below used these literal settings (export them together for replay):

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_BACKEND=cupy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

### Safety gates: PASS

```sh
.venv/bin/python -m pytest -q tests/test_native_engine_decoder.py::test_last_layer_failure_global_position tests/test_native_engine_decoder.py::test_factory_budget_preflight --tb=short
```

Latest: **2 passed in 3.60 s**, followed by successful `git diff --check` and
Python compilation of binding/new tests. Verified files: decoder binding/runtime,
ABI/runtime integration, new tests and extracted PLE helper, as built above.

Late-layer NaN shared-gate fixture fails specifically in native MoE at layer 47:
global position stays 0, resume and diagnostic trace are rejected, reset clears
failure and permits replay to the same deliberately invalid layer. Preflight
tripwires prohibit upload while testing insufficient dense, total GPU, mmap and
host-page budgets.

### Reduced synthetic 48-layer numerical gate: FAIL, preserved

```sh
.venv/bin/python -m pytest -q -s tests/test_native_engine_decoder.py -x --tb=short
```

The fixture genuinely binds/executes 48 independent native recurrent/MoE states
at reduced width 256; immutable synthetic weights are shared. It is **not** an
actual-geometry surrogate gate: actual QSA/PLE coverage comes from the real
checkpoint and baseline tests. Oracle is the unchanged full Python model graph,
with the same all-CPU placement/GGML policy and packed GPU shared/output path.
Tripwires forbid Python HC/branch/MoE execution and upload inside native step.

First failing boundary, token 0: **layer 15, post_moe**. Fixed FP32
`rtol=2e-4, atol=2e-4`; no finite-difference epsilon applies.

- 135/1024 elements exceed the envelope.
- Max violating absolute difference: `0.0569191`.
- Max violating componentwise relative difference: `0.2926828`.
- Example `[0,1]`: native `48.707828521728516`, Python
  `48.69479751586914`, absolute error `0.013031005859375`.

The assertion stops before synthetic replay/lifetime acceptance; those portions
must not be described as passing. An initial test-only API correction supplied
required `create_generation_state(max_length=4)`. No numerical production fix or
tolerance relaxation was made in response to this failure.

### Actual all-48 two-token/reset comparison: FAIL, execution phase passes

```sh
MINI_LLM_DECODER_CHECKPOINT=1 .venv/bin/python -m pytest -q -s tests/test_native_engine_decoder.py::test_actual_all48_two_steps_reset -x --tb=short
```

Latest: **1 failed in 9.92 s**, not a timeout. Native actual 48-layer two-step
execution and reset/replay completed with array-equal logits. The complete Python
reference also executed both fixed tokens in a fresh subprocess. Owners execute
sequentially to bound GPU storage; comparison uses snapshots, not forced shared
activations. Reference preparation explicitly has cache=0, no decoded-host expert
cache and no promotion.

First violating boundary: **token 0, layer 7, hc_ffn_read**.

| Metric | Observed |
| --- | ---: |
| dtype, native and oracle | FP32 |
| rtol / atol | 2e-4 / 2e-4 |
| maximum absolute error | 0.00155842304 |
| relative L2 error | 0.000535675091 |
| violating elements | 719 / 2560 |
| max violating componentwise relative error | 0.70095104 |
| example component 1, native | -0.08726051449775696 |
| example component 1, Python | -0.086783766746521 |
| example absolute error | 0.00047674775123596 |

Useful preceding evidence:

| Boundary, token 0 | max abs | relative L2 |
| --- | ---: | ---: |
| Layer 0 attention HC read | 4.76837158e-7 | 6.93854184e-8 |
| Layer 0 post MoE | 4.47034836e-8 | 1.81326442e-7 |
| Layer 1 post mixer, including PLE | 4.47034836e-8 | 1.95817631e-7 |
| Layer 3 post mixer, first QSA | 8.94069672e-8 | 2.23523452e-7 |
| Layer 6 FFN HC read | 8.34465027e-7 | 2.27641651e-7 |
| Layer 6 post MoE | 7.90730119e-5 | 0.000198024536 |
| Layer 7 attention HC read | 9.44137573e-5 | 8.93516242e-6 |
| Layer 7 post mixer | 7.85700977e-5 | 0.000196367287 |

The first sharp increase is layer 6 MoE, not the first QSA or PLE boundary. The
available evidence does **not** distinguish CPU Q8 threshold sensitivity to the
slightly different upstream input from a native/reference MoE arithmetic mismatch.
No specific incorrect production operation has been demonstrated. In particular,
we did not change QSA, HC or expert math to conceal the downstream assertion.
Final full-logit numerical acceptance is not established because the boundary
assertion fails first; native reset equality is not reference parity.

Initial integration corrections: avoid passing native GGUFStore directly to the
Python-only index adapter (use already validated header shapes); supply one page
for a one-logical-page test capacity; admit actual Q6_K output projection. These
were concrete API/descriptor failures, not numerical tuning.

The first same-process native-then-Python oracle attempt aborted during Python
GGML DSO loading after native replay. An attempted DSO keepalive subsequently hit
CUDA OOM during native setup despite a later idle query showing 15825 MiB free.
Neither failure was interpreted as a decoder equation defect. The final test
instead uses `tests/native_decoder_reference.py` in a fresh interpreter; that
completed and exposed the numerical failure above. Root cause of DSO reload/OOM
is not independently established; no CPU pool/library lifetime production changes
were made speculatively.

### Baseline regression and timeout: report separately

Full opt-in environment additionally set:

```sh
export MINI_LLM_TEST_PLE_CUDA=1
export MINI_LLM_QSA_LAYER_CHECKPOINT=1
export MINI_LLM_MOE_CHECKPOINT=1
export MINI_LLM_QSA_HIDDEN_CHECKPOINT=all
export MINI_LLM_QSA_PROJECTION_CHECKPOINT=1
export MINI_LLM_QSA_ATTENTION_BENCH=1
```

Executed:

```sh
.venv/bin/python -m pytest -q -rs tests/test_native_engine*.py --ignore=tests/test_native_engine_decoder.py
```

**1 failed, 622 passed, 2 skipped in 126.05 s.** The shell expands the explicit
file list before pytest; `--ignore` did not exclude the explicitly supplied new
decoder file. Failure is the preserved synthetic comparison above; 622 passes
include the established 620 baseline cases and two new safety cases. Skips were
the new full-model opt-in and existing nonzero-device test (only one GPU visible).
This is not a green full-suite claim.

A final explicitly filtered regression command was attempted:

```sh
.venv/bin/python -m pytest -q -rs tests/test_native_engine*.py -k 'not test_synthetic_all48_reference_replay_lifetime and not test_actual_all48_two_steps_reset'
```

**Timed out at 180 seconds** with progress past 73% and no reported failure before
the timeout. It was not retried with a longer timeout. Therefore no successful
final filtered full-suite result is claimed. The focused safety gates were rerun
afterwards and passed as recorded above. No new sanitizer or throughput result.

## Remaining blocker and single next experiment

Do not close the numerical acceptance gates, remove the failing tests, weaken the
envelope, or replace the implementation on this evidence.

**Next experiment:** capture the layer-6 FFN HC input from one trajectory and feed
that identical FP32 array to the existing native standalone MoE owner and unchanged
all-CPU Python MoE oracle, comparing router IDs/weights and CPU GGML terms before
HC write. This isolates the first amplification boundary from upstream reduction
differences and tests whether CPU activation quantization thresholds explain it.
It is diagnostic-only, not a forced-input full-model parity claim. Only if that
experiment identifies a specific mismatching operation is a production correction
justified.

Additional unpassed/unclaimed gates: complete synthetic replay/lifetime assertion,
actual full-logit parity, final successful full regression within 180 s, sanitizer
of the new decoder, independent resource-peak/RSS measurement, long-context WP6
quality, multiple state-only sessions with immutable sharing, sampling/prefill,
concurrent execution, capture, serving integration and performance acceptance.
