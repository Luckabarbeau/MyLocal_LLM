# WP6 — explicit INT8/pageable-host KV paging slice

Date: 2026-11-24. **WP6 remains incomplete.** This adds a separate storage policy,
not a replacement for FP32, a complete paged QSA layer, production dispatch,
Strata prompt attention, or a whole-model long-context acceptance claim.
Parent-owned status/manifest files are unchanged. No checkpoint, training file,
monolithic QSA implementation, or prior FP32 paging source was changed this turn.

## Source contract and named policy

Required policy string: **`strata_int8_per64_fp16`**. No alias, default policy,
automatic FP32 replacement, or silent fallback is accepted.

Inspected the existing local `mini_llm/ops/native_qsa_prefill.cu:387-430` and pinned
`references/strata-712fec6b4297/src__prefill__kernels.cu:2530-2573`.
Source revision: `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`.
The freshly measured source SHA256 matches the read-only reference manifest:

```text
410f9436065015db76271bc3906b3e21dc1392b0d76a658860b69fd85296d590
```

Each group has 64 FP32 source values; each token has 8 K groups and 8 V groups
(two KV heads of width256 each). The exact storage equations are:

```text
amax = max(abs(x[0:64]))                         FP32
scale_bits = float_to_half_RNE(amax / 127.0f)     FP32 divide, then FP16 RNE
sf = half_to_float(scale_bits)
q = sf > 0 ? clamp(float_to_int_RNE(x / sf), -127, 127) : 0
decoded = float(q) * sf
```

Division uses the **already FP16-rounded scale**, not an unrounded FP32 scale or
an approximate reciprocal. Integers use round-to-nearest-even, not truncation or
round-away-from-zero. A rounded-zero scale produces zero bytes, including tiny
nonzero inputs. The GPU quantizer preserves the pinned two-warp max reduction.
No fast math/FMA policy changes were introduced. The port retains MIT attribution.

The new packed layout is local: per token, K-head0 groups0..3, K-head1 groups0..3,
then corresponding V groups. Each group is64 signed INT8 bytes followed by the
2-byte FP16 scale. **16*66 =1056 bytes/token**. This is not a binary-compatible
Strata pool layout; the scalar quantization/scale rules are the pinned contract.

Nonfinite sources and finite sources whose scale overflows FP16 are explicitly
rejected before commit. This finite-scale admission guard is stricter than blindly
storing an infinite scale and then exposing nonfinite decoded values. Finite
underflow-to-zero is preserved, not rejected or replaced with another scale.

**Current K/V is quantized with the same rule**, even before append. Q and gate
remain FP32. Attention uses the existing FP32 online-softmax/GQA/sigmoid graph over
decoded INT8 K/V. It does **not** use Strata prompt MMA/query splitting, alter gate
arithmetic, or claim bitwise Strata full-attention parity.

## Files and API

New:

- `mini_llm/native_cpp/kv_streamed.h`: packed format constants and private launch contracts.
- `mini_llm/native_cpp/kv_streamed.cu`: quantizer, GPU page-request compaction,
  decoded selected gather, adapter to the unchanged attention launcher.
- `mini_llm/native_cpp/kv_streamed_runtime.inc`: independent native owner and C APIs.
- `tests/test_native_engine_kv_streamed.py`: CPU quantization reference, paging,
  error/budget/lifetime tests, actual populated-context test.

Additive registration only in `runtime.h`, `runtime.cpp`, `CMakeLists.txt`, and
`mini_llm/native/__init__.py`. The previous `kv_pages.*`/`kv_pages_runtime.inc`,
`qsa_layer_runtime.inc`, and `qsa_attention.cu` remain unchanged.

```python
with engine.quantized_streamed_kv(
    262144, page_size=1024, physical_pages=4,
    host_budget_bytes=280_000_000,
    policy="strata_int8_per64_fp16",
) as kv:
    length, generation = kv.state
    kv.append(keys_upload, values_upload, generation=generation)
    kv.attention(attention, q_upload, gate_upload, current_k_upload,
                 current_v_upload, generation=generation)
    stats = kv.stats
    kv.remap(logical_page=0, physical_slot=1, generation=generation)
    kv.reset()
```

- Capacity1..262144; page_size1..4096; fixed physical pages1..logical page count.
- Independent mutable session/layer namespace; native owner retains its engine.
- Append takes finite same-engine F32 `[rows,2,256]` K/V uploads. Each call is
  bounded to `rows<=page_size` and remaining logical capacity. Cross-page partial
  appends are supported; larger imports require explicit bounded calls.
- Attention accepts an existing valid native device selection, with
  `n=committed_length+1<=capacity`. No alternate IDs are supplied from Python.
- Readback is diagnostic: packed bytes for an explicit start/row range.
- `remap(logical_page,physical_slot)` explicitly stages that backed page into a
  chosen slot, evicting its previous occupant. It never changes logical contents.
- Stats: length, generation, requested device_bytes/host_bytes/host_budget,
  populated_bytes, resident_pages, cumulative page hits/misses, actual H2D payload
  bytes, fixed metadata-transfer bytes, and pending completion flag.

## Bounded backing and stable slots

Host backing is **ordinary pageable RAM**, not pinned/mapped host storage or a
GPU-accessible host mirror. One zero-initialized packed backing covers rounded
logical capacity and is admitted against an explicit per-store host-byte budget.
Only quantized history is retained. No full FP32 KV/raw-history array exists in
this store. There is no SSD tier, compression of the backing itself, or host LRU.

The GPU owns a stable physical arena of `physical_pages * page_size *1056` bytes,
one page-sized quantization/upload staging buffer, one capacity-sized logical
slot table, bounded selected FP32 gather scratch, compact attention IDs/n, status,
and fixed page-request metadata. No new GPU payload allocation occurs on append,
cache hit/miss, remap, or reset. Logical pages can far exceed physical slots.

Append quantizes on GPU into the bounded stage and copies packed bytes to ordinary
RAM. The original uploaded F32 inputs are not retained after completion. Appending
to a cached partial tail invalidates only affected logical pages; unrelated hits
stay resident. The next reader reloads the changed page, rather than consuming a
stale cached tail. Backing is authoritative for all committed rows.

### Device selection to native miss scheduling

```text
existing device n/ascending selected IDs
  -> CUDA validates IDs and maps historical tokens to logical pages
  -> adjacent duplicate pages compacted (IDs are ascending)
  -> requested logical pages + compact missing-page list + counts
  -> bounded native C++ device-to-host metadata copy
  -> working-set preflight
  -> protect all requested resident pages
  -> choose only empty/nonrequested slots for misses
  -> ordinary RAM -> stable GPU slot, then publish device table entry
  -> GPU gather/decode selected rows, with separately quantized current K/V
  -> unchanged FP32 selected attention/gate
  -> completed scalar status -> valid device output snapshot
```

There is **no Python token-ID roundtrip**, host attention arithmetic, or Python
per-selected-page scheduler. The CUDA compactor scans at most2051 IDs, not the
configured capacity, and excludes the current token from history-page requests.
Native C++ copies one fixed **32840-byte metadata envelope** per attention call:
three uint64 counts/status fields plus two2051-entry page arrays. Valid entries
are compacted prefixes; padding is zeroed. It contains requested logical pages
and missing pages, not a host selected-token list. This host metadata fence and
native scheduling are explicit seams, not a fully GPU-resident page scheduler.

Repeated selections load no already-resident payload. Miss loading protects every
requested hit before eviction. With three slots, the mixed-window fixture verifies
exactly two page hits and one page miss when its window moves. Remapping and
returning to the same selection preserves outputs and reloads only absent pages.

### Explicit working-set requirement — no hidden fallback

**Every distinct selected historical page must fit in the configured GPU slots
for that call.** The current token does not require a history slot. Before any
host-to-GPU page payload transfer or cache eviction, native C++ checks:

```text
number_of_distinct_selected_historical_pages <= physical_pages
```

Otherwise attention returns ML_CAPACITY with `selected-page working set exceeds
GPU slots`, leaves committed history and cache payload unchanged, and makes the
attention output unavailable. Current-source quantization/validation and the
bounded metadata fence may already have run; no historical page staging occurs.
The caller must configure sufficient slots for its selection locality. To guarantee
arbitrary selections without a locality assumption, reserve at least
`min(2051, ceil((n-1)/page_size))` slots; this worst-case bound is conservative when
there are fewer than2051 selected history tokens. The actual request is checked
every call. A two-slot/three-required-page test verifies refusal, zero staging,
zero residency growth, and intact committed length. There is no full-resident
fallback, hidden extra slot allocation, or dropped selected token.

Chunked staging/gather for a selected working set larger than the cache is **not
implemented** in this slice. The explicit minimum-working-set contract is the
chosen acceptance option.

## Generations, pending completion, failure and ownership

All public calls still serialize through the native runtime boundary. A native
CUDA completion event and pending flag protect the store's outstanding copy and
kernel epoch. Enqueued work is recorded and retired before slot reuse, table
mutation, stage reuse, reset or successful return. Stream-completion guards retain
all source/store/attention owners on enqueue/status failure, and destruction drains
the stream before releasing physical pages, ordinary RAM, or the engine.

Per-slot generation metadata is checked against the active store generation.
Reset increments generation, clears device/host mapping metadata, retires slots,
resets logical length/stat counters and rejects old-generation append, attention,
remap and diagnostics. Backing bytes need not be cleared: bounds and generation
prevent old rows from being exposed. Explicit remap preserves generation because
logical contents do not change and all prior readers have retired.

Calls are synchronous; `stats.pending` is therefore normally0 at public observation.
This is internal completion-protected ownership, **not** an asynchronous exported
lease API, per-slot concurrent-stream service, or DMA/compute overlap claim. CUDA
fault injection and concurrently outstanding readers are not tested.

Finite/shape/generation/capacity/working-set admission failures do not advance
length. Current nonfinite input is rejected before page loading; failed output
readback is unavailable. An interrupted append/remap/load mutation leaves the
owner invalid until reset. Completed attention output snapshots own their values
and remain usable after the store and source handles close. Public engine close
does not invalidate the retained native store or completed attention owner.

## Accounting and actual populated-context evidence

Let `L=ceil(capacity/page_size)`, `P=page_size*1056`, `S=physical_pages`, and
`R=32840` bytes for page-request metadata. Requested store allocations are:

```text
host_bytes = L*P + R + 8*(L + 2*S)
GPU bytes  = 8*L + R + 8 + 2051*8 + 8 + 2051*4096 + (S+1)*P
```

The host formula charges packed backing, request metadata, host table and slot
page/generation arrays. The GPU formula includes padded status, compact n/IDs,
~8MiB selected FP32 gather, fixed arena and stage. No application-managed pinned
memory is allocated. GPU bytes participate in the existing aggregate engine
budget; ordinary-RAM bytes use an independent explicit per-store budget. Exact
budget success and one-byte-short rejection are tested for both budgets.

This is requested-payload accounting, **not total VRAM/RSS**: CUDA context,
allocator rounding, events, native owner/registry containers, Python arrays,
read-only file mappings/page cache and CUDA-internal pageable staging are outside
these counters. Per-store host budgets are not an aggregate process/cgroup RAM
governor; constructing many owners still requires parent memory admission.

The progressive test uses capacity262144, page_size1024, **four GPU slots**, a
280000000-byte ordinary-RAM budget, and a128MiB aggregate GPU-engine budget.
It actually commits and checks all packed chunks through each milestone:

| Committed rows | Populated packed bytes | Requested store GPU bytes | Requested host bytes | Resident pages after final append |
|---:|---:|---:|---:|---:|
| 65536 | 69206016 | 13858928 | 276859016 | 2 |
| 131072 | 138412032 | 13858928 | 276859016 | 2 |
| 262144 | 276824064 | 13858928 | 276859016 | 2 |

The backing is reserved/zero-initialized up front; populated_bytes counts actual
successful appends, not the reservation. Every written chunk is read back and
compared byte-for-byte to CPU quantization. One reusable random1024-row FP32 source
chunk is repeated across logical positions; this validates actual populated storage
and addressing, not distinct real-model content at every position. No full FP32
history is created on CPU or GPU. Only a selected2051-row decoded reference is
formed for attention checks.

Before committing the final row at each milestone, attention evaluates both a
prefix selection and a distant-tail selection at n65536/131072/262144, including
the separately quantized current token in the tail case. The selection is generated
on GPU from flat or increasing pooled scores. Diagnostic IDs are read only after
native staging, to construct the independent reference. Both windows are checked
against decoded INT8 references and immediately repeated to prove hits/zero new
payload transfer. Final-row append invalidates the cached tail page, explaining
the two remaining residents in the table above.

Cumulative observed counters at the three populated milestones:

| Rows | Hits | Misses | Actual host->GPU page bytes | Metadata bytes | Pending |
|---:|---:|---:|---:|---:|---:|
| 65536 | 6 | 6 | 6487008 | 131360 | 0 |
| 131072 | 12 | 12 | 12974016 | 262720 | 0 |
| 262144 | 18 | 18 | 19461024 | 394080 | 0 |

Capacity-dependent selection scores/pooled index inputs are existing separate
workspaces; the long test uploads two32MiB pooled-score fixtures plus bounded
query/current inputs. They are not FP32 KV history. Store GPU size and the steady
engine allocation are asserted unchanged as actual history grows. This proves one
bounded streamed store with prefix/tail selections, **not** all QSA layers/full
model feasibility or arbitrary fragmented selection feasibility with four slots.

## Numerical results: decoded reference versus FP32 policy

Packed INT8 bytes and raw FP16 scale bits match the NumPy CPU reference exactly.
Fixtures cover zero, nearest-even half ties, signs, saturation limits, zero-rounded
subnormal scales, minimum positive FP16 scale, maximum finite FP16 scale, and
nonrepresentable/nonfinite rejection. Explicit signed subnormal-scale cases
force the actual +127/-127 clipping branch after FP16 scale rounding. The tested
per-group reconstruction bound is `max(0.50001*sf, max(amax-127*sf, 0))`, plus
FP32 multiplication roundoff: the second term accounts for clipped saturation,
and gives `amax` when sf rounds to zero. A half-step-only bound is not valid for
all subnormal rounded scales. Maximum observed fixture reconstruction error is0.5
(the deliberate scale1 half-tie fixture), not an attention tolerance.

Attention comparisons use rtol4e-5/atol3e-6, as for the existing NumPy attention
reference. Quantization discrepancy is recorded separately and never used to
relax decoded-reference correctness:

| n | Max absolute error versus decoded-INT8 CPU graph | Max absolute difference versus original FP32 graph |
|---:|---:|---:|
| 1 | 2.384185791015625e-7 | 0.012390714138746262 |
| 9 | 7.152557373046875e-7 | 0.015823960304260254 |
| 33 | 7.152557373046875e-7 | 0.009536385536193848 |
| 2053 | 8.940696716308594e-8 | 0.0013262107968330383 |

These synthetic differences are not model-quality acceptance or FP32 equivalence.

## Exact commands and results

All GPU commands and children inherited the mandatory mask:

`CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631`

Identity check:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 .venv/bin/python -c "import cupy as cp; p=cp.cuda.runtime.getDeviceProperties(0); print('visible_count=',cp.cuda.runtime.getDeviceCount()); print('visible_ordinal=',cp.cuda.Device().id); print('name=',p['name']); print('uuid=',p.get('uuid')); print('pci_bus=',cp.cuda.runtime.deviceGetPCIBusId(0))"
```

Observed one visible device, visible0, RTX5070Ti, matching UUID bytes, PCI
`0000:04:00.0`: the authorized **physical GPU1**. Physical GPU0 was not exposed.
No GPU tests/benchmarks ran concurrently.

Build:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 cmake --build /tmp/mini-llm-wp1-build -j 2
```

Passed. Only the three already documented misleading-indentation warnings in
unchanged `gguf_loader.cpp` appeared.

First new-suite execution:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s tests/test_native_engine_kv_streamed.py
```

First result:7passed/1failed. The working-set-refusal fixture attempted to use an
attention owner after its context had closed (`object is closed`), so that call
never reached native admission. One localized fixture fix creates a live
selection/attention for the refusal case. The progressive test was extended to
exercise distant-tail misses as well as prefix hits. Retest: **8passed in1.85s**.
No production change was made in response to that fixture failure.

Final focused run, after adding cached-tail/nonfinite-current/full-capacity/stale-
generation coverage, including the previous FP32 paging baseline:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s tests/test_native_engine_kv_streamed.py tests/test_native_engine_kv_pages.py
```

**29passed in2.53s**:9 new policy tests and20 existing FP32 paging tests. No skips.
This final execution includes the explicit signed subnormal clipping cases; the
preceding29-test execution passed in2.52s before that additional fixture data.

Final CUDA memcheck:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q tests/test_native_engine_kv_streamed.py tests/test_native_engine_kv_pages.py
```

**29passed in77.06s; ERROR SUMMARY:0 errors**. The prior29-test state passed in76.96s
with0errors before the signed subnormal-clipping fixture addition. An earlier
28-test state also passed memcheck in77.19s with0errors.
Every GPU validation command was bounded to140seconds and completed within it.

CPU-only source identity command:

```sh
sha256sum references/strata-712fec6b4297/src__prefill__kernels.cu
```

Matched the pinned manifest hash above. `git diff --check` passed. Broad native
regression is intentionally deferred per the requested scope; prior broader
results in `WP6_PAGED_KV.md` are historical, not a rerun of this implementation.
No benchmark or throughput measurement was run. Pytest elapsed times above are
test execution records, not paging/attention performance claims.

## Remaining work

- Complete paged QSA layer composition: frontend/raw-index/pooling, selection,
  output projection, transactional append and full-layer API integration.
- Broad native regression and real-checkpoint quantized-layer/quality acceptance.
- Arbitrary selected working sets larger than GPU slots: chunked gather/staging
  or a separately accepted larger minimum-slot plan; current path refuses them.
- Asynchronous exported page leases, nonblocking miss scheduling, per-slot
  overlapping transfer/compute readers, allocation quarantine/fault injection.
- Host eviction/SSD paging, aggregate host admission and measured RSS/global VRAM.
- Direct paged-load attention without the ~8MiB bounded selected FP32 scratch.
- Full decoder/serving integration and any matched performance benchmark.

The original monolithic and explicit FP32 paged policies remain available and
unchanged. This document does not close or update the parent WP6 milestone.
