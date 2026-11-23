# WP1: isolated synthetic CUDA runtime

This is **synthetic token transport, not Qwen or any model inference**. No weights,
logits, sampling, tokenizer, KV cache, or production integration is implemented.
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
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_runtime.py
```

CMake requires and links CUDA::cudart; this transport-only implementation contains
no kernels, so it compiles C++20 with the host compiler rather than invoking nvcc.
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
host bytes. The per-engine budget bounds their sum, with an independent session
count limit; reset and step allocate no buffers. Accounting reports live requested
payload and peak payload (reset does not lower the peak). It excludes CUDA context,
stream, allocator rounding, registry/shared-pointer overhead and driver resources;
it is **not** total GPU/process memory accounting. Session capacity does not allocate
history. Reset synchronizes and resets position without reallocating. CUDA errors
are surfaced, not hidden; recovery from a poisoned CUDA context is not promised.
