# WP2 budgeted native stored-byte uploads

Date: 2026-11-23. Isolated experimental upload consumer, **not native model inference or completion of WP2**. The parent owns `IMPLEMENTATION_STATUS.md`; it is intentionally untouched.

## API and computational invariant

`Engine.upload(lease) -> UploadedTensor` allocates exactly `lease.descriptor.nbytes` on the engine's configured CUDA device and copies the original stored representation:

```text
shared PayloadLease → immutable GGUF mmap byte range
                    → cudaMemcpyAsync(H2D, engine nonblocking stream)
                    → recorded CUDA event
                    → owned device byte allocation
```

There is no type-dependent upload branch, decoding, unpacking, casting, reshaping of storage, or precision change. F32 bit patterns (including special encodings), F16/BF16, quantization scales, and packed codes are all transported as uninterpreted bytes. Descriptor shape, type identity, row/block geometry, and source offsets remain unchanged. Python mutable detached descriptors are not allocation authorities: the native lease descriptor is authoritative.

The C ABI adds `ml_engine_upload`, `ml_uploaded_descriptor`, `ml_uploaded_query`, `ml_uploaded_wait`, `ml_uploaded_destroy`, and `ml_uploaded_readback_for_test`. Existing status codes, serialized exception boundaries, stale/wrong-kind handle rejection, zero destroy, and create-output zeroing are preserved. Python exports `UploadedTensor`, with context manager, idempotent close, and best-effort GC cleanup. A lease must belong to the same Python `Runtime` as its engine.

## Transfer and ownership

- The upload retains the **same `shared_ptr<PayloadLease>`**, not merely its mmap pointer or a reopened store handle. It also retains `shared_ptr<Engine>`; no Python store/lease/engine reference is required for safety.
- Closing public store, lease, or engine handles does not invalidate an uploaded consumer. Closed engine handles admit no new uploads/sessions. Existing uploads retain their stream and allocation. Closing other leases does not affect them.
- `query() -> bool` returns false only for `cudaErrorNotReady`. Successful event query or `wait()` retires the source reference. The device allocation and copied descriptor survive until upload destruction. Retirement of this consumer does not close independently live store or lease handles.
- A query/wait error raises `NativeError` **without retiring the source reference**. Destruction synchronizes the engine stream before device free and lease release; if stream synchronization reports an error, it attempts device synchronization as a cleanup fallback. Cleanup is nonthrowing/best-effort, consistent with existing runtime destructors. A poisoned/lost CUDA context is not recoverable by this API; hardware teardown errors cannot be returned by destructors.
- Failed allocation, enqueue, event record, or handle publication unwinds through the same native owner. This covers a copy enqueued successfully before event recording fails: teardown drains the stream instead of relying on that event. Device bytes are reclaimed; create output remains zero. Peak records successfully requested live device storage even if a subsequent enqueue fails.
- Native registry construction order ensures upload teardown precedes lease/store/engine registry teardown at process exit. Shared ownership independently preserves dependencies. All native operations remain under the existing global mutex. No throughput/concurrency claim is made; upload close can also drain unrelated work on the same engine stream.

**Pageable-source semantics:** the source is the exact immutable mmap range, passed directly to `cudaMemcpyAsync`. The engine stream is created with `cudaStreamNonBlocking`, but pageable H2D may stage/synchronize inside CUDA. **Asynchronous host return is not guaranteed**, and a small upload may physically complete before the API returns. There is no application-managed pinned staging, whole-model registration/pinning, additional whole-tensor host copy, or claim of transfer/compute overlap. Source shards must remain immutable until all consumers retire/release them.

## Shared engine admission and accounting

For live uploads U and sessions S:

```text
device_bytes = sum(U.descriptor.nbytes) + 4 * S
pinned_bytes = 8 * S
live_requested = device_bytes + pinned_bytes <= budget_bytes
peak_bytes = maximum observed live_requested
```

Uploads use zero application-managed pinned staging bytes. Session limits are independently enforced. Admission for **both** upload and session creation uses remaining budget, `budget - device_bytes - pinned_bytes`, instead of an overflow-prone sum with the requested allocation. Upload byte sizes must fit `SIZE_MAX`; input sizes/offsets/capacities are uint64-validated. Successful allocations and destruction preserve counter invariants under the native mutex. Upload query/wait does not lower live device bytes; close/GC does. Session reset does not lower peak. Peak never decreases.

Accounting covers requested payload allocations, **not total CUDA/process memory**. It excludes CUDA contexts, streams/events, allocator rounding, CUDA's internal pageable-copy staging, mmap/page cache, registry/shared-pointer objects, and readback's caller-owned host buffer. No application-owned unbounded pinned storage is hidden outside the budget. A budget does not guarantee the CUDA allocator can satisfy an admitted request; CUDA allocation errors still surface and unwind deterministically.

## Test-only bounded readback

`upload.readback_for_test(offset, size, destination) -> copied_count` requires an explicit tensor-relative byte range and a caller-owned writable contiguous Python buffer with sufficient capacity. The C ABI likewise requires explicit destination, capacity, and copied-count output; it does not allocate a tensor-sized readback buffer. The source device pointer is internal to native code for future native GPU operators and is never returned to Python.

Readback validates `offset <= nbytes` and `size <= nbytes - offset` before waiting/copying, avoiding uint64 addition overflow. Invalid ranges/capacity/null arguments leave the caller buffer unchanged and zero a supplied copied count. Null destination is allowed only for zero-byte copies. Once the upload event is successfully waited, D2H uses synchronous `cudaMemcpy`; no pending write to caller memory remains on successful return. Hardware copy errors may partially modify a valid destination; this is not transactional copying. Native callers must provide valid addresses and truthful capacities, as with payload-lease copies.

Example (after explicitly building and selecting the library):

```python
from mini_llm.native import Runtime

runtime = Runtime()
with runtime.gguf("weights.gguf") as store:
    lease = store.lease("tensor.name")
    engine = runtime.engine(budget_bytes=lease.descriptor.nbytes + 12)
    upload = engine.upload(lease)
    lease.close()
    engine.close()
    # Upload still owns its native source lease and engine.
    upload.wait()
    prefix = bytearray(min(16, upload.descriptor.nbytes))
    upload.readback_for_test(0, len(prefix), prefix)
    upload.close()
```

## Validation / verified state

Files involved: `mini_llm/native_cpp/runtime.h`, `mini_llm/native_cpp/runtime.cpp`, `mini_llm/native/__init__.py`, and new `tests/test_native_engine_upload.py`. Documentation: repository `README.md`, `mini_llm/native_cpp/README.md`, and this note. Existing lease work and unrelated working-tree changes were preserved; no production Qwen execution/training code, GGUF parser, or status document was changed.

Before GPU execution, `nvidia-smi` and its compute-app query showed two available NVIDIA devices (RTX 5080 at index 0, RTX 5070 Ti at index 1), with no training/serving compute workload reported; only Zed editor GPU use was listed.

Build from repository root:

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda && cmake --build /tmp/mini-llm-wp1-build
```

Outcome: succeeded, linked `/tmp/mini-llm-wp1-build/libmini_llm_native.so`. Three pre-existing `-Wmisleading-indentation` warnings remain in unchanged `gguf_loader.cpp` (lines 34, 59, 113); this is not a warning-free build claim.

Focused test:

```sh
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_upload.py
```

Outcome: **39 passed in 0.62s**, no failures/skips. The initial implementation passed; no failure-driven corrective edit was needed.

All native-engine tests after the build:

```sh
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_*.py
```

Outcome: **110 passed in 1.21s**, no failures/skips (the established 71 cases plus 39 uploads). This is the verified source state. `git --no-pager diff --check` also passed after this test command.

New coverage:

- Raw byte equality and unchanged descriptors across all 20 native-loader storage type IDs, including F32, F16, BF16, Q8_0, K/IQ types, and MXFP4. Unaligned partial, full, and zero/end-range readbacks.
- Store/lease/engine close and Python GC before explicit event retirement, native rather than hidden Python ownership, mmap release at successful wait, and uploaded bytes surviving source release.
- Upload GC without query/wait, repeated teardown/reclamation, idempotent close, and stale/wrong-kind handles.
- Shared session/upload budgets in both allocation orders, exact-fit admission, failed admission leaving stats unchanged, live-byte reclamation and persistent peak.
- Python integer/uint64/range and caller-buffer validation; C null outputs/destinations, overflow-safe ranges, zeroed failure outputs, and unchanged sentinels.
- Actual upload/readback on device **1**, with caller CUDA device restoration checked after query/wait/readback and destruction. This case skips only if fewer than two devices exist; it ran here.

Limits: these small pageable-copy tests do not deterministically force a physically in-flight DMA at public close, even though they do force close before explicit consumer retirement. No injected CUDA event/enqueue/allocator fault, poisoned-context recovery, sanitizer, full-checkpoint upload, native GPU weight operator, model parity, or performance claim is included. The error/lifetime paths are implemented but CUDA fault behavior is not experimentally exercised by this result.
