# WP2 payload-consumer leases

Date: 2026-11-23. Isolated native host-payload slice, **not GPU uploads, model inference, or completion of WP2**.

## Implemented contract

- `mini_llm/native_cpp/runtime.h` and `runtime.cpp` expose `ml_gguf_lease_create`, `ml_payload_lease_descriptor`, `ml_payload_lease_copy`, and `ml_payload_lease_destroy` through the existing serialized, noexcept exception boundary.
- Each native `PayloadLease` holds `shared_ptr<ml::GGUF>` and an immutable tensor index. Its registry entry is itself shared ownership, suitable for a future asynchronous consumer to retain. No mapped payload address escapes the ABI.
- Closing a store removes its public handle and prevents new leases. Already-created leases remain valid without a Python reference to the store. Each lease retains the whole GGUF (all shards); mappings disappear only after the store and the last lease release ownership. Closing one lease does not invalidate sibling leases.
- Handles are process-local, monotonically allocated IDs. Zero destroy is a no-op; stale or wrong-kind handles are rejected. Create outputs are zeroed before fallible work. Descriptor outputs are committed only on success. Copy requires a non-null byte-count output, zeroed on failure, and reports the exact copied count on success.
- Native copies use explicit tensor-relative byte offset/size, a caller-owned writable host buffer, and its explicit capacity. Bounds use `offset <= nbytes` and `size <= nbytes - offset`, avoiding uint64 addition overflow. Null destinations are accepted only for zero-byte copies. Invalid ranges/capacity/null arguments fail before any destination write; there are no partial successful copies.
- As with the existing C ABI, callers must provide valid pointer addresses and truthful writable buffer capacities. Arbitrary invalid non-null addresses cannot be validated. Source files must remain immutable until all mappings are released; concurrent mutation/truncation is unsupported.
- `mini_llm/native/__init__.py` exposes `store.lease(name) -> PayloadLease`, `.descriptor`, and `.read(offset, size) -> bytes`. Both read arguments are mandatory; bool, noninteger, negative, and out-of-uint64 inputs are rejected. Live native descriptor bounds and host address-space size are checked before payload allocation. Reads copy into a ctypes-owned host buffer and return detached bytes, never a raw pointer or mapped view. Returned create handles, descriptor rank, and copied byte counts are checked.
- Python holds its runtime RLock through descriptor lookup, range checks, host allocation, copy, and output checking, serializing against close. The native mutex independently serializes C calls and destruction. Leases inherit context-manager, idempotent close, and best-effort GC cleanup behavior.
- No production loader/model code, GGUF parser, CUDA upload path, engine budget, or status file was modified. A read temporarily owns the ctypes buffer and returned bytes; this is not a zero-copy Python interface or a new memory-admission budget.

## Validation / verified state

Files involved: `mini_llm/native_cpp/runtime.h`, `mini_llm/native_cpp/runtime.cpp`, `mini_llm/native/__init__.py`, new `tests/test_native_engine_payload.py`, and the three unchanged existing native-engine test suites below.

Exact build command, from repository root:

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda && cmake --build /tmp/mini-llm-wp1-build
```

Outcome: **succeeded**, linked `/tmp/mini-llm-wp1-build/libmini_llm_native.so`. Three existing `-Wmisleading-indentation` warnings were emitted in unchanged `gguf_loader.cpp`, at lines 34, 59, and 113. This is not a warning-free-build claim.

Exact test command:

```sh
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_payload.py tests/test_native_engine_runtime.py tests/test_native_engine_gguf.py tests/test_native_engine_qwen_contract.py
```

Exact outcome: **71 passed in 0.98s**, no failures or skips: 24 new payload cases plus the existing 47 native runtime/GGUF/Qwen-contract cases. The initial implementation passed; no corrective implementation edit was needed. This establishes the verified source state for the files above. `git --no-pager diff --check` also completed successfully after testing.

New coverage includes:

- Nonzero F32 and packed Q8_0 source-byte equality, descriptor identity, unaligned partial byte reads, nonzero data/relative offsets, zero-length and end-of-tensor ranges, and detached returned bytes after close.
- Explicit size, invalid names, missing tensors, integer/range/uint64 overflow rejection before ctypes payload allocation, and a sparse tensor-relative read above 4 GiB.
- Independent same-tensor and cross-shard leases, whole-store shard retention until the last lease, store close and store GC without Python strong ownership, lease GC, idempotent Python close, and stale native IDs.
- Native null pointers, missing outputs, wrong-kind handles, unchanged descriptor/destination outputs on failure, zeroed failed copy/create counts, exact successful copy counts, and caller-buffer sentinel preservation.
- Python rejection of simulated successful native calls with missing create/copy outputs; deterministic Python read-vs-store-close and read-vs-lease-close locking checks.
- Repeated mmap/file-descriptor cleanup and subprocess native registry teardown with intentionally live raw store/lease handles.

No full-checkpoint payload comparison, sanitizer run, GPU weight upload, asynchronous lifetime test, conversion/dequantization, or performance claim is included in this result. Existing synthetic CUDA runtime tests did run as part of the 71-test command.

## Next actionable step

Implement the next isolated consumer: a bounded native upload operation with explicit device-memory admission and completion ownership. It must retain the **same `shared_ptr<PayloadLease>`**, not just its source pointer or a re-opened store handle, until its CUDA event/stream completion is established. Test closing both public store and lease handles while that operation is in flight, then verify exact uploaded bytes and final release/accounting. Do not connect production inference before that lifetime test passes.

`docs/native_engine/IMPLEMENTATION_STATUS.md` is intentionally untouched; the parent owns status updates. Pre-existing changes to `native_reference_manifest.json`, `WP0_CHECKPOINT_HASHES.json`, and `strata_launch_bench_capture.sh` were preserved.
