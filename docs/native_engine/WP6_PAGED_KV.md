# WP6 paged KV — first resident FP32 acceptance slice

Date: 2026-11-24. **WP6 is not complete.** This is independent experimental
engine-owned per-session/layer storage plus a selected-attention adapter, not
production dispatch, quantized KV, out-of-core storage, or a complete decoder.
Parent-owned `IMPLEMENTATION_STATUS.md` and `native_reference_manifest.json`
were deliberately not edited.

## Preserved baseline and inspected architecture

Inspected `qsa_layer_runtime.inc`, `qsa_attention.{h,cu}`, runtime ownership and
attention entry points, the native WP3/WP5 reports, `REUSE_AND_BASELINE.md`,
`IMPLEMENTATION_STATUS.md`, and `docs/qwen4exp/memory_architecture.md`.

The existing `QSALayer` retains its exact API and monolithic raw/K/V/pool state,
4096/16384 configuration restriction, transactional step, and baseline kernel.
Neither `qsa_layer_runtime.inc` nor `qsa_attention.cu` was edited. Existing
production Python computation, `train_model.py`, checkpoints, and pre-existing
user work were not modified. No model/server was launched or checkpoint rewritten.
The architecture warning that selected top-k does not bound persistent history
still applies; this slice refuses physical-page exhaustion rather than evicting
needed history or silently truncating context.

## Added API and state

- `kv_pages.h`, `kv_pages.cu`: GPU logical-token resolution and bounded selected
  gather, followed by the unchanged verified attention launcher.
- `kv_pages_runtime.inc`: private owner, device table, fixed physical pool,
  generation validation, append/remap/reset/diagnostic APIs.
- Additive declarations/includes/build registration in `runtime.h`, `runtime.cpp`,
  `CMakeLists.txt`; thin `Engine.kv_pages(...)` / `KVPages` ctypes binding.
- `tests/test_native_engine_kv_pages.py`: resident paging acceptance tests.

```python
with engine.kv_pages(262144, page_size=16, physical_pages=8) as store:
    length, generation = store.state
    store.append(keys_upload, values_upload, generation=generation)
    store.attention(attention, q_upload, gate_upload, current_k_upload,
                    current_v_upload, generation=generation)
    store.remap(logical_page=0, physical_slot=7, generation=generation)
    store.reset()  # length=0; generation increments; old generation rejected
```

Each owner is an explicit session/layer namespace; separate owners do not share
mutable history. It retains its native engine independently of the public engine
handle. There is no automatic association with `Session`, layer dispatch, or a
shared global KV cache. Caller chooses which session/layer owns each store.

- Logical capacity: 1..262144 tokens.
- Configurable `page_size`: 1..4096 tokens, including non-power-of-two sizes.
- Physical pages: 1..ceil(capacity/page_size), fixed at creation, budget-admitted.
- K/V storage and arithmetic: FP32 only, two heads of width256 each.
- Append input: two immutable same-engine uploaded F32 `[rows,2,256]` arrays;
  appends consecutive history, validating all rows before store mutation.
- Attention input: existing valid device selection with `n=history_length+1`,
  `n<=store.capacity`, q/gate `[24,256]`, separate current K/V `[2,256]`.
- The selection capacity can be a short active capacity; it need not equal the
  store's logical capacity. Selection remains authoritative for token IDs.
- Output is the existing attention object's device snapshot, usable by existing
  projection consumers. Explicit readback is diagnostic only.

Address relation for historical token `t`:

```text
logical_page = t / page_size
row_in_page = t % page_size
table[logical_page] = physical_slot + 1   (0 means nonresident)
K address = pages + ((physical_slot * page_size + row_in_page) * 1024)
V address = K address + 512
```

The device table contains slot indices, not host pointers. All physical-page
addresses, the table, and scratch stay stable until owner destruction. A host
map stores only populated logical-page/slot metadata; no host K/V mirror exists.
Physical slots form one stable preallocated arena. Remap moves into an empty slot
or swaps two occupied slots with one page of fixed scratch, preserves all logical
history, and publishes the corresponding table entries after ordered copies.

### Page-use lifetime and generations

This first API is deliberately synchronous. The runtime boundary lock serializes
calls. The run retains store, selection/attention and uploaded inputs through a
`StreamCompletion` guard; reset/remap/destruction drain the same engine stream
before modifying or freeing page/table storage. Thus no physical slot is reused
while an outstanding kernel can read it. This is a completion-fenced page-use
lease, **not** an asynchronous exported lease API or an event-driven eviction
scheduler. No public device pointers escape.

Store generation starts at1 and advances on reset; stale-generation append,
attention, remap and diagnostics fail before accessing history. Remap preserves
generation because logical contents are unchanged and no readers remain in flight.
Reset invalidates all table entries without reallocating or clearing full pages;
length/table bounds prevent stale payload reuse. Commit/remap failure leaves the
store invalid until reset; real CUDA fault injection is not tested. Failed
finite validation leaves committed history unchanged. Failed attention does not
publish a consumable output. Existing completed attention snapshots do not borrow
pages and survive store destruction.

## Selected attention computational graph

```text
device n + existing device IDs
  -> GPU table lookup for each selected historical token
  -> direct current operand when original ID == n-1
  -> fixed GPU selected K/V scratch, original ascending selection order
  -> compact IDs 0..m-1 and compact n=m, m=min(n,2051)
  -> unchanged selected FP32 GQA / online softmax / sigmoid gate
  -> scalar status synchronization -> valid device output snapshot
```

For the unchanged kernel, compact row `m-1` is supplied as its separate current
operand; that row is the last **selected** original token, whether or not the
original current token was selected. The softmax has exactly `m` entries: no
extra current row is introduced. Thus key/value order and arithmetic match the
monolithic same-ID graph. Original q/gate/current sources are finite-validated
even when the original current token is unselected. Historical sources were
validated at append and remain owned/immutable except ordered content-preserving
remaps. There is no selected-ID host readback or host selection in this adapter.

This is an intentionally bounded GPU gather adapter, not a zero-gather/direct
paged-load attention kernel. Diagnostic ID readbacks in tests never feed it.

## Capacity versus active work

Requested device payload bytes per store:

```text
8 * ceil(capacity/page_size)                  logical table
+ physical_pages * page_size * 4096           physical K/V arena
+ 2051 * 4096                                selected K/V scratch
+ 8 + 2051 * 8                               compact n / IDs
+ page_size * 4096                           remap scratch
```

No application-managed pinned allocation, full raw-index array, full logical
K/V history, or capacity-sized selected gather is created. Engine statistics
charge the entire formula; exact-budget success and one-byte-short refusal are
tested. Host map/registry overhead and CUDA context/allocator overhead are outside
requested-payload accounting, as in the existing runtime.

Paired tests use identical populated prefixes, pages and inputs with logical
capacities4096 and262144. They verify exact accounting, no allocation growth on
attention/remap/reset/repopulate, exact historical K/V, unchanged IDs, and
bitwise-equal context/gated outputs. For page_size16, table bytes are2048 versus
131072; all other allocation terms are identical for identical physical budgets.
The attention launcher receives active `n`, **not capacity**; gather dispatch is
fixed2051 blocks and exits beyond `m`, and baseline validation/attention sees
compact `m`. Reset/construction may touch capacity-sized table metadata; short
active attention does not scan the logical capacity. Append work follows actual
rows and populated pages, never unpopulated logical capacity. This establishes
bounded allocation and dispatch structure, **not a latency benchmark**.

Full actual populated262144-token context, all-model-layer memory feasibility,
and sparse late-page histories are not established here. Store append is still
sequential and refuses more populated pages than its physical budget. Existing
raw/pool/index-selection storage is not converted by this slice.

## Validation — exact commands and results

All GPU commands and their children used this exact prefix, exposing only the
user-authorized **physical GPU1** as CUDA visible0:

`CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631`

Identity command:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 .venv/bin/python -c "import cupy as cp; p=cp.cuda.runtime.getDeviceProperties(0); print('visible_count=',cp.cuda.runtime.getDeviceCount()); print('visible_ordinal=',cp.cuda.Device().id); print('name=',p['name']); print('uuid=',p.get('uuid')); print('pci_bus=',cp.cuda.runtime.deviceGetPCIBusId(0))"
```

Observed: visible_count1, ordinal0, NVIDIA GeForce RTX5070Ti, UUID bytes matching
`e961516f-d77d-813c-98e9-2f5403f71631`, PCI `0000:04:00.0`. Physical GPU0 was not
used. The two-device test is intentionally skipped rather than exposing GPU0.

Build:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 cmake --build /tmp/mini-llm-wp1-build -j 2
```

Passed. Three existing misleading-indentation warnings in unchanged
`gguf_loader.cpp` remain; they were not addressed as unrelated work.

Focused final state:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_kv_pages.py
```

**20 passed in1.08s**. Coverage: page sizes1/3/4/7/16/32/64, lengths1/2/4/5/8/33/
2053/4096, same-history/ID/output comparison against monolithic, both policies,
current selected/unselected, poisoned uncommitted/future monolithic slots,
remap to occupied/empty slots, reset/repopulate, stale generation, finite-source
failure, physical exhaustion, bad shapes/configuration, cross-engine rejection,
isolated stores, exact budget/release and owner lifetime after source/engine
public close. Added final incremental case checks17 consecutive single-row
appends across page_size5 tails, per-step outputs, and nonfinite-current failure
without committed-history mutation or readable failed output.

The first focused run was18passed/1failed: test expected NativeError for capacity
262145, but Python correctly raised ValueError at its integer bound. Only that
test expectation was corrected. Retest19passed in1.14s; later the incremental
case was added, producing the final20-pass result. Production code did not change
in response to that fixture failure.

CUDA memory checking of the final focused state:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q tests/test_native_engine_kv_pages.py
```

**20 passed in8.79s; ERROR SUMMARY:0 errors.** No sanitizer fault injection or
asynchronous concurrent-stream stress claim is made.

Broader native regression (same implementation, before adding the final
test-only incremental case):

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine*.py
```

**488 passed,102 skipped in34.53s.** NumPy does not enable CuPy-oracle tests and
checkpoint opt-ins were absent. This is not the full supplied baseline gate.

Then the broader opt-in CuPy/checkpoint configuration:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=cupy MINI_LLM_TEST_PLE_CUDA=1 MINI_LLM_QSA_LAYER_CHECKPOINT=1 MINI_LLM_MOE_CHECKPOINT=1 MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q -rs tests/test_native_engine*.py
```

**586 passed,4 skipped in106.08s.** Skips, exactly:

- `test_native_engine_qsa_attention.py:266`: opt-in CUDA timings.
- `test_native_engine_qsa_hidden_frontend.py:246`: optional actual checkpoint hidden frontend.
- `test_native_engine_qsa_projection.py:212`: optional actual checkpoint projection validation.
- `test_native_engine_upload.py:230`: nonzero-device test needs two visible devices.

Unlike the user-supplied prior570passed/1skip result, this invocation did not
enable those first three optional gates. No claim that they were rerun is made.
All commands were bounded to140seconds each and completed within that limit.
Final `git diff --check` passed. No full-model performance benchmark was run.

## Remaining WP6 work / explicit host seams

- Quantization, per-group scales, packed page layout and quantized decode parity
  are **not implemented**. No INT8/BF16 policy is silently selected.
- No host/SSD KV tier, eviction, miss prefetch, asynchronous transfer publication,
  overlapping readers/remappers, per-page exported lease API, or page scheduler.
- Append uses native host scheduling and D2D row copies, not host K/V arithmetic.
  Device table publication, host length/generation bookkeeping, upload waits,
  completion fences and scalar validation readbacks remain synchronous. Append
  currently synchronizes during new-page publication. No capture/overlap promise.
- Selection, frontend raw-index/pools and full `QSALayer` transactional composition
  remain the verified monolithic implementation. Callers explicitly append history
  and invoke this adapter; there is no new complete paged `QSALayer.step` API.
- Bounded selected gather costs about8MiB/store. A future direct paged-load kernel
  can remove it only with independent same-graph parity/lifetime validation.
- No full populated262144-token acceptance, complete-decoder integration,
  serving integration, quality acceptance or throughput improvement claim.

This document records only the first GPU-resident FP32 paging acceptance slice;
it does not update or close the parent WP6 milestone.
