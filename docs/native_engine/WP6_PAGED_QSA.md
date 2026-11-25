# WP6 — opt-in complete paged QSA composition

Date: 2026-11-24. **WP6 is not marked complete.** This accepts the described
component composition and tests, not a complete decoder, production backend,
quantized-model quality gate, or fully populated262K model inference.
Parent-owned `IMPLEMENTATION_STATUS.md` and `native_reference_manifest.json`
are unchanged. Checkpoint data, production reference code and training work
were not modified. No server or full model was launched.

## Added integration and preserved default

`Engine.qsa_layer(...)` still creates the original monolithic state unless an
explicit `kv_policy` is supplied. Its original capacity4096/16384 contract,
projection/frontend/attention arithmetic, import/reset/readback names and native
HC chaining entry point remain available. The native layer owner now branches
only when its optional paged state is present. The complete native regression,
including monolithic trajectories and checkpoint opt-ins, passed below.

```python
layer = engine.qsa_layer(
    262144, query_gate, key, value, index_query, index_key, output,
    gamma_q, gamma_k, gamma_iq, gamma_ik,
    policy="strata",                       # ID tie policy, not KV precision
    kv_policy="reference_fp32",            # or strata_int8_per64_fp16
    page_size=64, physical_pages=1,
    host_budget_bytes=16 * 2**20,
    raw_diagnostics=True,                   # optional host-backed full raw history
)
layer.step(hidden_upload, position)
layer.import_reference_state(length, raw_upload, keys_upload, values_upload, pooled_upload)
layer.reset()
```

The ten weights retain the original exact shape/format validation. Paged capacity
is positive, divisible by4, and at most262144. Page size is1..4096; there is at least
one fixed GPU slot, no more than the logical page count. Paged options without an
explicit KV policy are rejected. No precision policy is inferred from tie policy,
capacity, cache pressure or input format.

Added files:

- `mini_llm/native_cpp/qsa_paged.{h,cu}`: FP32 packing and per-page selected gather.
  INT8 packing reuses the validated `ml_kvi8_quantize`, unchanged.
- `mini_llm/native_cpp/qsa_paged_runtime.inc`: lazy bounded ordinary-RAM pages,
  device page table, completion-fenced chunked gather, diagnostics/accounting.
- `tests/test_native_engine_qsa_paged.py`:21 complete-layer cases.

Targeted integration in `qsa_layer_runtime.inc`, `runtime.{h,cpp}`, CMake and
`mini_llm/native/__init__.py`. Existing `qsa_attention.cu`, `qsa_frontend.cu`,
`qsa_selection.cu`, projection kernels, and standalone `kv_pages.*` /
`kv_streamed.*` implementations were not changed this turn. This composition
reuses their verified arithmetic and the device request-compaction primitive;
it does not replace their existing public contracts or silently remove the
standalone streamed store's minimum-working-set check.

## Complete native graph

```text
hidden (uploaded input or existing native HC hidden operand)
  -> Q/gate, K, V, IQ and raw-index projections
  -> norms/RoPE, dead key and conditional completed pool
  -> staged active pooled index + native GPU scores/ascending IDs
  -> pack current K/V under explicit storage policy
  -> scalar frontend/selection/current validation
  -> GPU IDs -> logical pages -> compact miss metadata -> native C++ scheduling
  -> gather resident hits; stage/gather misses through retired bounded slots
  -> selected scratch -> unchanged FP32 GQA/softmax/sigmoid gate
  -> output projection -> complete status validation
  -> host K/V/raw-debug commit + GPU first/raw-window/pool commit
  -> completed stream -> publish length and valid output
```

No Python IDs, Python selection, diagnostic array, CPU attention, or per-page
Python orchestration is a step input. The existing HC->QSA native call reaches
the same `qsa_layer_step_device` branch. New paged HC-specific trajectories were
not separately added; monolithic HC chaining regressions remain in the full run.

Both policies execute attention/output accumulation in the existing FP32 graph.
`reference_fp32` stores exact FP32 K/V. `strata_int8_per64_fp16` quantizes history
and the current K/V using the exact per64/FP16-scale policy documented in
`WP6_QUANTIZED_STREAMED_KV.md`: half-rounded scale before divide, nearest-even
integer conversion, clip[-127,127], zero rounded scale -> zero bytes, explicit
nonfinite/nonrepresentable-scale rejection. No Strata prompt-MMA/query arithmetic
or implicit BF16 conversion is introduced.

## Exact raw-history dependency and bounded GPU state

Inspected `qsa_frontend.cu`, in particular rows30/31 and the chronological pool
sum. At step position`t`, `n=t+1`:

1. The dead key uses the current raw projection only at n1, otherwise **raw[0]**.
   It is normalized with index-key gamma and is not position-rotated.
2. A new pool is produced only when `n%4==0`. It uses exactly
   `(((raw[n-4] + raw[n-3]) + raw[n-2]) + current_raw) * 0.25`, with the existing
   separately rounded, left-to-right additions, normalization and RoPE at n-4.
3. Future selection reads completed normalized/rotated pools, not their old raw
   terms. No other historical raw row is read by the frontend computational graph.

Therefore the paged GPU state has exactly **first_raw[128] plus raw_window[4,128]**:
640 floats,2560 bytes. Committed token`t` writes window slot`t%4`; only token0
initializes first_raw. On a completing block, slots0/1/2 hold precisely the three
previous terms. Slot3 is retained for explicit diagnostics, not reassociated into
the pool computation. A failed step never advances these committed slots.

Import restores raw[0], then writes the last four source rows to their `t%4`
slots. For an incomplete imported block, the needed slots0..tail-1 are current;
other slots may belong to the previous block but are not read before replacement.
For a complete imported block, subsequent commits replace slots0/1/2 before the
next pool. This handles imported lengths with every remainder, including2050 and
2051. Tests compare first_raw/window exactly against the latest monolithic row
for each residue after import, every step, reset and replay.

The paged layer allocates **no capacity-sized FP32 raw array and no full FP32
K/V device history**. Existing pooled and staged-pooled index arrays remain on
GPU, each `capacity/4 *128` floats, plus capacity scores. During step, only the
committed active pool prefix is copied. Score launch capacity is rounded-up active
`n`, not configured capacity; selection also reads device n. Short active steps at
configured262K therefore do not scan262K unused score blocks or raw/KV history.
Construction/reset still clear capacity-sized index metadata; no claim is made
that those operations are capacity-independent.

## Import, reset and explicit diagnostics

The existing import API remains an explicit F32 reference-state import into an
empty valid layer with `4<=length<=capacity`. Exact shapes, engine ownership and
all finite inputs are checked first. Caller source uploads can be large: this API
is not a new streamed-import transport. Tests use bounded active history uploads,
not a capacity-sized262K import.

Paged import reserves budgeted host pages, packs bounded page chunks, and restores
host K/V, optional full raw debug rows, first raw/window, and GPU completed pools.
Source state remains caller-provenance data, as before. INT8 import applies the
named quantization policy; FP32 import preserves K/V exactly. Source uploads are
not retained once consumed. A failed mutating import is poisoned until reset;
length is not partially published. Budget-exhausted import/recovery is tested.

Reset fences readers, clears device/host page mappings, frees lazy host pages,
clears first/window and GPU index/output state, advances the page generation,
and resets length/cache counters. Original outputs/IDs/scores remain unavailable
until a successful new step.

Existing diagnostic names retain their meanings:

- `output`, `ids`, `scores`: latest successful step snapshot.
- `keys`, `values`: all committed rows, decoded to caller-owned FP32 output. FP32
  storage is exact; INT8 returns the defined decoded storage, not prequantized data.
- `pooled`: committed GPU pool prefix.
- `raw`: full committed raw history **only if `raw_diagnostics=True`**. It lives
  in separately charged host-page suffixes, never a full device duplicate. Disabled
  mode returns an explicit error, not an incomplete raw history.
- Added paged-only `raw_first`, `raw_window`: direct fixed-size device diagnostics,
  including zeroed reset state.
- Added paged-only `queries`, `gates`: latest normalized/rotated query and separate
  gate operands, available only after a successful output. Tests use them to
  factor an independent decoded-INT8 attention oracle; production does not read them.

Readback is explicit and never feeds native step. Full raw diagnostics are off by
default. No unreported1.5GiB aggregate raw-device duplication is introduced.

## Chunked selected gather: legal with one GPU slot

Each host page is ordinary pageable RAM, lazily allocated on first reserve and
bounded by an explicit host payload budget. K/V is token-major within each page;
INT8 uses1056 bytes/token, FP32 uses4096. Optional raw-debug storage adds512 bytes/
token in a page suffix. Packed/FP32 source content remains authoritative in host
backing, independently of current GPU residency.

The device compactor receives native IDs/n and produces unique requested logical
pages plus a compact missing-page list, excluding the separate current token.
The same bounded32840-byte page-metadata envelope as the prior streamed slice is
copied to native C++. It contains page metadata, not a Python selected-token list.

For **any** selected historical working set with at least one configured slot:

1. Gather the separately packed current operand, if selected, and initialize
   compact device n/IDs. No extra current softmax entry is created.
2. Gather **all initially resident requested pages** into fixed selected scratch,
   then fence their last reads. These hits require no payload upload.
3. For each compact missing logical page, choose an empty slot or a retired slot;
   clear the old table entry, stage only populated bytes, publish the new mapping,
   gather that page's selected rows to their original selection-order locations,
   then fence its reader before the slot may be reused.
4. Run unchanged attention over the complete selected scratch. The last selected
   row acts as the baseline launcher's separate compact-current operand, even if
   the original current token was not selected. Width/order are unchanged.

Thus selected rows survive eviction of their source pages. There is no
`selected_pages<=slots` hard block in the complete paged layer, no selected-token
drop, no hidden additional slots, and no full-resident fallback. Tests use one
physical slot with selected tokens spanning many logical pages, including33-page
short/sparse fixtures. GPU arena addresses remain stable.

This implementation launches a filtered gather for each requested page and uses
synchronous native scheduling. It is correctness-first, not a claim of optimal
launch count, overlap, or constant cache-hit rate. Current appends invalidate only
an affected cached page; other cached pages remain valid.

## Completion, transactional publication and ownership

The boundary lock serializes calls. A completion event/pending epoch and same-
stream ordering protect stage reuse, table publication and physical-slot reuse.
Each page's selected reader retires before its slot is recycled; the selected
scratch outlives all page chunks. Host slot generation is checked against the
current generation. Reset retires all readers before mapping/page release.

All frontend/selection/quantization/attention/output-projection status is checked
before any K/V/raw/window/pool commit. Reserving a future host page before compute
is allocation only; it does not publish a token and may remain reserved after a
finite validation rejection. On successful output validation, the owner is marked
invalid before commit, packed current K/V and optional raw are copied to host,
GPU first/window/pool are updated, and only successful synchronization publishes
new length and output-valid flags. A CUDA error during compute poisons the layer;
an interrupted commit is already poisoned. Reset is the recovery path. No
resumable partial token state is exposed.

Tests verify finite input failure, host page refusal, position errors, unchanged
committed history, unavailable failed output, failed import poisoning/reset, and
lifetime after public weight/upload/engine close. **Actual CUDA fault injection
was not performed.** The failure ordering is source-enforced; no claim of tested
rollback after an injected driver/device fault is made. This is per-layer atomic
publication, not whole-model/HC transaction rollback or exported asynchronous leases.

## Requested-memory accounting and measured fixtures

`layer.paged_stats` returns the existing twelve-field statistics layout. Its
`device_bytes` includes the layer workspace and paged backing workspace, but not
retained weight uploads. Host bytes include page-request metadata, host table,
slot/generation arrays and all allocated host pages including padding and raw
suffixes. GPU bytes participate in the engine budget; host bytes use the explicit
per-layer host budget.16 pinned bytes remain charged by the layer. Dynamic host
bytes/counters are reset on reset; GPU allocations remain stable.

Let C=capacity, P=page_size, S=slots, L=ceil(C/P), R=32840, and B=4096 for FP32 or
1056 for INT8. Requested device payload is:

```text
layer workspace = 8 + 2051*8 + 8 + 4*(640 + 65*C + 42368)
paged workspace = 8*L + R + 8 + 2051*8 + 2051*4096 + (S+1)*P*B
```

Requested host metadata is `R + 8*(L+2*S)`; each allocated host page additionally
costs `P*(B + (raw_diagnostics ? 512 : 0))`. No full-capacity host K/V/raw reserve
is required. Host-budget exhaustion does not enlarge the budget or allocate a
full-resident device fallback.

Paired4096/262144 tests import the same31 active rows and step through length35,
using page_size64, one GPU slot, and a1MiB large-layer host budget. Outputs/IDs/
scores match the corresponding small configured layer exactly. FP32 also matches
the monolithic layer exactly. Raw-debug is disabled on the large short test.

| KV policy | Configured capacity | Active length | Requested layer+page GPU bytes | Requested host bytes | Host budget |
|---|---:|---:|---:|---:|---:|
| reference_fp32 | 262144 | 35 | 77353104 | 327768 | 1048576 |
| strata_int8_per64_fp16 | 262144 | 35 | 76963984 | 133208 | 1048576 |

The different debug setting on the small comparison owner does not affect GPU
size. Exact capacity-dependent device difference is asserted:
`(262144-4096)*260 + ((262144-4096)/64)*8`; it is index storage/table metadata, not
raw or K/V history. GPU bytes remain unchanged through import and steps. Host
bytes grow only by actually reserved bounded pages.

Actual checkpoint layer3 (converted original weights/gammas), configured262144,
page_size64, one slot, host budget16MiB, raw-debug enabled:

| Policy | Steps | Imported sparse start | Requested GPU bytes | Host bytes at length2053 | Page misses after sparse reset | Engine requested peak |
|---|---:|---:|---:|---:|---:|---:|
| reference_fp32 | 8 | 2050 | 77353104 | 9797720 | 99 | 138258136 |
| strata_int8_per64_fp16 | 8 | 2050 | 76963984 | 3377240 | 99 | 137869016 |

The eight steps are five from empty plus three from imported2050 history. The
source history is generated with the checkpoint projections, converted norms,
position-specific RoPE, chronological raw pooling, and deterministic11-period
hidden inputs. Bounded per-input projection memoization avoids re-dequantizing
identical checkpoint rows thousands of times. Imported data is identical for both
native owners. It is one attention-layer fixture, not a real prompt/full-model
trajectory or fully populated262K layer acceptance.

All memory numbers above are **requested payload accounting**, not total device
memory or process RSS. CUDA contexts, allocator overhead, events, Python/CuPy
reference arrays, mmap/page cache and CUDA-internal pageable staging are not
included. Host budgets are not an aggregate process/cgroup RAM governor.

The prior standalone store actually populated64K/128K/262K in
`WP6_QUANTIZED_STREAMED_KV.md`; those tests also ran in the full regression here.
That evidence must not be relabeled as a fully populated262K complete layer or
full-model test. This composition's actual active checkpoint context reaches2053.

## Gate coverage and numerical results

New21 cases:

-12 FP32 cases: both tie policies; starts0,63,2048,2050,2051,4092; chronological
  state/IDs/scores/output all **bitwise equal** to monolithic after each step;
  import, page/pool/selection boundaries, raw first/window, capacity end,
  reset/replay, one-slot many-page selected sets.
-4 INT8 cases: both tie policies, empty and2050-import starts. Exact decoded K/V
  bytes/values against CPU quantization of monolithic prequantized state; exact
  raw/pool/IDs/scores. An independent NumPy softmax/GQA/sigmoid graph uses those
  decoded rows and diagnostic native query/gate operands, followed by the original
  output projection callback. This factors the storage/attention check from the
  separately verified frontend; it is not an independent full-model oracle.
-2 paired configured262K short-active capacity/memory/raw-debug-off cases.
-1 finite/position/host-budget/import-poison/reset/ownership case.
-2 actual checkpoint layer3 short/sparse cases, one per precision policy.

INT8 output comparison retains rtol4e-5/atol3e-6 from the previous selected-attention
oracle; no tolerance was relaxed. Measured synthetic maxima:

| Start | Tie policy | Decoded oracle max absolute output error | Difference from FP32 output |
|---:|---|---:|---:|
| 0 | ggml_cpu | 2.561137080192566e-9 | 9.887339547276497e-5 |
| 0 | strata | 2.561137080192566e-9 | 9.887339547276497e-5 |
| 2050 | ggml_cpu | 6.05359673500061e-9 | 8.925143629312515e-5 |
| 2050 | strata | 6.05359673500061e-9 | 8.925143629312515e-5 |

Checkpoint FP32 outputs/state/IDs are bitwise equal. Checkpoint INT8 decoded-oracle
max output error is2.0116567611694336e-7; separate FP32-policy output difference is
0.0027400553226470947. These are synthetic-layer tolerances/results, not quantized
model quality acceptance or a claim that INT8 equals FP32.

## Exact executed commands and results

**Every shell command and GPU child used the required UUID mask.** Identity check:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 .venv/bin/python -c "import cupy as cp; p=cp.cuda.runtime.getDeviceProperties(0); print('visible_count=',cp.cuda.runtime.getDeviceCount()); print('visible_ordinal=',cp.cuda.Device().id); print('name=',p['name']); print('uuid=',p.get('uuid')); print('pci_bus=',cp.cuda.runtime.deviceGetPCIBusId(0))"
```

Observed one visible device, ordinal0, RTX5070Ti, matching UUID bytes, PCI
`0000:04:00.0`: authorized physical GPU1 only. Physical GPU0 remained hidden.

Build (executed initially and again after adding diagnostic operands):

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 cmake --build /tmp/mini-llm-wp1-build -j 2
```

Passed. Initial rebuild showed the three existing `gguf_loader.cpp`
misleading-indentation warnings; no unrelated warning cleanup was made.

Focused non-checkpoint command:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s -x tests/test_native_engine_qsa_paged.py
```

First invocation failed before native execution because the new Python option loop
shadowed the `value` weight parameter with a host-budget integer. One localized
binding correction renamed the loop variable; no algorithm or tolerance changed.
Retest: **19passed,2checkpoint-gated skips in4.39s**.

Actual checkpoint focused command:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_QSA_LAYER_CHECKPOINT=1 MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s -x tests/test_native_engine_qsa_paged.py -k checkpoint
```

**2passed,19deselected in1.39s**, with memory/numerical records above.

Memcheck logical subset A, short and2050-import FP32 trajectories:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q 'tests/test_native_engine_qsa_paged.py::test_fp32_import_boundaries_exact_state[0-9-strata]' 'tests/test_native_engine_qsa_paged.py::test_fp32_import_boundaries_exact_state[2050-5-strata]'
```

**2passed in5.79s; ERROR SUMMARY:0 errors**.

Memcheck logical subset B, all INT8/capacity/error/checkpoint cases:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_QSA_LAYER_CHECKPOINT=1 MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q tests/test_native_engine_qsa_paged.py -k 'not fp32_import_boundaries_exact_state'
```

**9passed,12deselected in19.37s; ERROR SUMMARY:0 errors**. Thus11 unique new cases
were memchecked; the remaining10 FP32 parameter combinations passed normal/full
regression, not a new claim of memchecking every parameter. Each command had a
180second bound. Neither timed out; no automatic long timeout rerun was used.

Full native suite with baseline GPU/checkpoint/PLE/MoE/timing opt-ins:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=cupy MINI_LLM_TEST_PLE_CUDA=1 MINI_LLM_QSA_LAYER_CHECKPOINT=1 MINI_LLM_MOE_CHECKPOINT=1 MINI_LLM_QSA_HIDDEN_CHECKPOINT=all MINI_LLM_QSA_PROJECTION_CHECKPOINT=1 MINI_LLM_QSA_ATTENTION_BENCH=1 MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 .venv/bin/python -m pytest -q -rs tests/test_native_engine*.py
```

**620passed,1skipped in123.99s**, within the180second bound. Sole skip:
`test_native_engine_upload.py:230`, nonzero-device test needs two visible devices.
GPU0 was not exposed to satisfy it. Relative to the supplied pre-WP6
570passed/1skip baseline, the new50 cases are20 FP32 store +9 streamed INT8 store
+21 complete paged-layer cases. Existing fully populated store tests and all
monolithic opt-ins passed in this run.

The baseline attention timing test ran serially as an opt-in regression. No new
paged-QSA throughput/latency benchmark or performance improvement is reported.
GPU tests/benchmarks did not overlap. Pytest elapsed times are validation records,
not inference performance numbers. `git diff --check` passed.

## Remaining gates / honest scope

- Whole-model quantized quality and multi-layer/decoder/serving composition.
- Fully populated262K **complete layer** and real long-prompt behavior; current
  complete-layer checkpoint gate is active2053 at configured262K.
- Additional paged HC chaining-specific acceptance beyond shared native-step
  reachability and existing monolithic chaining regressions.
- Real CUDA fault injection, nonthrowing teardown failure attribution and
  asynchronous/concurrent-stream reader/quarantine validation.
- A streamed/bounded import transport that does not require caller whole-prefix
  uploads, aggregate host admission/RSS enforcement, host eviction/SSD tier.
- Direct paged-load attention or a more efficient chunked gather, asynchronous
  overlap, measured hit/miss latency and matched performance benchmarks.

Default monolithic precision/behavior and both previously validated standalone
store policies are preserved. Parent milestone/status ownership remains unchanged.
