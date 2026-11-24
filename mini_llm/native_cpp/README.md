# Experimental native CUDA runtime: transport, uploads, and bounded QSA selection

This provides **synthetic token transport, exact stored-byte weight upload, and
bounded GPU QSA index scoring/selection, not Qwen or any model inference**. No
logits, sampling, tokenizer, KV cache, native weight projections, full QSA attention,
or production integration is implemented.
A step echoes a nonnegative int32 token using pinned host → device → pinned host
`cudaMemcpyAsync` operations on the engine-owned nonblocking CUDA stream, then
synchronizes and returns `(token, zero_based_position)`. Position advances only
on success. Capacity limits successful steps, not token values.

## Build and test (repository root)

Requires C++20 compiler, CMake >=3.24, Ninja, CUDA toolkit >=13.0 (tested with
nvcc 13.4), and a working NVIDIA CUDA device/driver. There is no CPU fallback and
GPU tests do not silently skip if CUDA or the library is missing.

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/mini-llm-wp1-build
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_*.py
```

CMake requires and links CUDA::cudart, enables CUDA, and invokes nvcc for
`qsa_selection.cu`. CUDA compilation uses no fast math, with `--fmad=false` and
`--ftz=false`. Host runtime sources remain C++20.
All build artifacts remain outside the source tree. Load the built shared library
explicitly via `Runtime(path)` or `MINI_LLM_NATIVE_LIBRARY`; importing does not build.
The repository's existing `mini_llm/__init__.py` still runs on normal package import.

## Ownership and ABI contract

See `runtime.h` for the C-compatible API. No C++ exceptions escape. Status codes:
0 success, 1 invalid argument/closed handle, 2 capacity, 3 CUDA error, 4 internal
exception. `ml_last_error()` returns thread-local diagnostic text until the next
ABI call. Outputs are committed on success; create outputs are zeroed on failure
when a non-null output pointer is supplied. Callers must supply valid writable
pointers (arbitrary invalid addresses cannot be validated).

Handles are monotonically assigned, process-local IDs, not raw pointers. Stale IDs
are rejected. Destroying zero is a no-op. C callers must destroy each nonzero handle
once; Python `close()` is idempotent and supports context managers and best-effort
GC cleanup. Sessions retain a native shared engine reference and a Python engine
reference. Closing the public engine handle prevents new sessions but existing
sessions remain usable; final session destruction releases the stream. All native
calls are serialized, including destruction, and restore the caller's CUDA device.
No concurrency throughput claim is made. Destructors synchronize and perform
best-effort nonthrowing CUDA cleanup; teardown CUDA errors cannot be returned.

Each session allocates exactly 4 requested device bytes and 8 requested pinned
host bytes. Each upload allocates exactly the descriptor's stored `nbytes` on the
engine's device, with **zero application-managed pinned staging**. Uploads and
sessions compete for one engine budget in either allocation order; sessions also
have an independent count limit. Admission uses subtraction bounds to avoid
uint64 overflow. `device_bytes` includes sessions and live uploads; `pinned_bytes`
includes session buffers; `peak_bytes` is the high-water sum and never decreases.
Completion retires source ownership, not device storage: upload close/GC reclaims
its device bytes. Reset/step allocate no buffers and session capacity allocates no
history. Reset synchronizes and resets position without reallocating.

Accounting covers requested payload, not CUDA context/stream/event overhead,
allocator rounding, CUDA-internal pageable transfer staging, registry/shared-pointer
objects, host mmap/page cache, or test readback caller buffers. It is **not** total
GPU/process memory accounting. No whole-model pin/register or decoded copy occurs.
CUDA errors are surfaced, not hidden; destructors use nonthrowing best-effort
cleanup and recovery from a poisoned CUDA context is not promised.

## Exact stored-byte uploads

```python
from mini_llm.native import Runtime

runtime = Runtime()  # MINI_LLM_NATIVE_LIBRARY must be set
with runtime.gguf("/path/to/immutable.gguf") as store:
    with store.lease("tensor_name") as lease:
        with runtime.engine(device=0, budget_bytes=lease.descriptor.nbytes + 12) as engine:
            with engine.upload(lease) as uploaded:
                # No FP cast, dequantization, reshape, or packing change.
                ready = uploaded.query()  # bool; a poll is not an implicit wait
                uploaded.wait()          # raises NativeError on CUDA failure
                descriptor = uploaded.descriptor
                # Tests only: explicit byte offset/size and caller-owned buffer.
                destination = bytearray(min(16, descriptor.nbytes))
                uploaded.readback_for_test(0, len(destination), destination)
```

`Engine.upload(PayloadLease) -> UploadedTensor` uses exact mmap → device
`cudaMemcpyAsync` on an engine-owned **nonblocking CUDA stream**, followed by a
completion event. The source is pageable: CUDA can synchronize/stage internally,
so **asynchronous host return is not guaranteed**. No overlap/throughput claim is
made. All accepted GGUF storage formats use the same uninterpreted byte path.

Native shared ownership retains the same `PayloadLease` and `Engine`, independent
of Python references. Public store, lease, and engine close prevents new operations
on those handles but does not invalidate an existing upload. Successful query,
wait, or test readback retires the upload's source lease; errors retain it. The
uploaded descriptor and device allocation remain valid until upload close.
Destruction drains the stream before freeing storage, including partial enqueue /
event-record failure. Closing an upload may wait for other work on its engine
stream. There is no cancellation. All native calls remain globally serialized.

The C ABI exposes descriptor/query/wait/destroy plus a test-only D2H copy with
explicit range, capacity, destination, and copied-count output. Python readback
requires a writable contiguous caller buffer; no raw mutable device address is
exposed. Range/capacity errors write no payload, failed copy counts are zero, and
readback waits before a synchronous D2H copy. CUDA copy errors may partially modify
a valid caller buffer; no transactional guarantee is made for hardware failures.
Future native operators can use the internal device allocation without exporting
pointers to Python. See [`WP2_WEIGHT_UPLOADS.md`](../../docs/native_engine/WP2_WEIGHT_UPLOADS.md)
for validation and scope. WP2 completion/status remains parent-owned.

## Bounded WP3 QSA selection

`Engine.qsa_selection(capacity)` allocates stable GPU n/score/ID/status storage.
`workspace.set_n(n)` explicitly uploads a host scalar. Call
`workspace.run(query_upload, pooled_upload, dead_upload, policy="ggml_cpu")` (or
explicit `"strata"`) with same-engine exact F32 uploads of Python shapes `[4,128]`,
`[capacity/4,128]`, `[128]`. Completed pools, including any newly completed pool,
must already be committed in the supplied pooled input. Runs synchronize and
return CUDA-event milliseconds, but never copy scores or IDs to host implicitly.
`readback_for_test(scores_f32, ids_int64)` is explicit diagnostics only. Reset and
n updates invalidate results without reallocating. This is a reference GPU heap
implementation, not an optimized selector or full attention. Full contract,
accounting, exact validation results and timings: [`WP3_SELECTION.md`](../../docs/native_engine/WP3_SELECTION.md).
WP3 is in progress; parent-owned status files are not updated here.
