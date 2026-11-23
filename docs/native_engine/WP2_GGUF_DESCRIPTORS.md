# WP2 isolated slice: native read-only GGUF descriptors

Date: 2026-11-23. This is metadata and mmap-backed descriptor indexing only.
**Not dense GPU loading, model inference, or completion of WP2.**

## Implemented boundary

- `mini_llm/native_cpp/gguf_loader.{h,cpp}` parses little-endian GGUF v2/v3 with checked uint64 arithmetic, bounded header reads, rank 1..4, nonzero dimensions, power-of-two alignment, packed row alignment, payload bounds and nonoverlap.
- One `PROT_READ`, `MAP_PRIVATE` mapping per shard; file descriptors close immediately after mapping. RAII releases mappings on successful close and partial-open failure. Source files must remain immutable while open; concurrent truncation/mutation is not supported.
- Tensor names must be unique across all shards. Repeated shard device/inode identities are rejected. Split ordinal/count/total and overlapping typed encoded metadata are validated. Metadata equality is conservative byte equality, not floating-point normalization.
- Supported tensor descriptor types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, repository Q2_0 (42: 64 elements / 18 bytes). Other types explicitly fail; a descriptor does not imply a decoder/kernel exists.
- Standard scalar metadata and flat scalar/string arrays are supported. Nested arrays, unknown metadata types, embedded NUL strings and other versions/endianness explicitly fail.
- `runtime.h`/`runtime.cpp` add independent process-local GGUF handles to the existing serialized exception boundary. Borrowed C names/encoded metadata remain valid only until destruction; C callers must serialize pointer use against destruction. No tensor payload pointer escapes.
- `Runtime.gguf(path)` reuses Python filename discovery only. Headers and descriptors are parsed natively, without calling the Python GGUF reader. Python copies metadata and descriptor records only, under its runtime lock; no full tensor duplication or decoding. Detached records remain inspectable after close, but all native mappings are released. No GPU allocation or runtime-engine attachment occurs.

## Validation / verified state

Command from repository root:

```sh
cmake --build /tmp/mini-llm-wp1-build && MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_gguf.py tests/test_native_engine_runtime.py
```

Exact result: build succeeded; **23 passed in 0.51s**, no skips (15 GGUF cases plus 8 existing CUDA runtime tests). No corrective implementation edit was needed after testing.

The build emitted three `-Wmisleading-indentation` warnings in `gguf_loader.cpp` at lines 34, 55 and 109. These are recorded, not claimed warning-free.

New fixtures cover Python `MultiShardTensorStore` parity for metadata, tensor identity, shapes, packed sizes, offsets and byte strides; read-only OS mappings and release; multishard order/identity; duplicate tensor names; conflicting shard metadata; invalid split identity; bounds, overflow, row/offset alignment, rank/dimensions, unsupported tensor types, overlapping payloads, alignment, header counts, versions and endian; stale/null ABI arguments; repeated cleanup and failed-open descriptor/mapping release. No full checkpoint or independent decoder/GPU loading test was run.

## Follow-up milestone

IQ descriptor coverage and actual 1224-tensor checkpoint header parity were subsequently validated; see `WP2_IQ_INVENTORY.md` and `WP2_GGUF_INVENTORY.json`. The initial 23-test result above remains historical; the follow-up ran 28 tests successfully. The initial format list and blockers 1/5 below describe the initial milestone, not the current coverage.

## Concrete remaining WP2 blockers (initial milestone)

1. Native format coverage does not yet include the production IQ formats (including IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS and IQ4_NL); those files currently fail explicitly. Add audited block-size descriptors and parity fixtures before production indexing.
2. No native architecture-specific metadata/configuration contract, tensor-name/shape validation, required tensor set, embedding/output alias policy or conversion/permutation plan exists in this slice. Reuse the established Python architecture index contracts rather than inventing new ones.
3. No dense device weight allocation/upload, dtype conversion, dequantization, RoPE row transformation, budget/admission planning or stream/event lifetime integration is implemented. Existing CUDA transport budgets do not account for mapped files or model weights.
4. No native bounded payload/row/expert access or retained-view lifetime API is exposed. Future consumers need an explicit mapping lease/ownership contract before asynchronous copies can safely outlive a store handle.
5. Full production shard/checkpoint identity, supported-format inventory and architecture-level parity against the existing Python index remain untested. Tiny fixtures establish this isolated contract only; they do not establish full-model correctness or performance.

Unrelated worktree changes were preserved. This slice does not change production loader/model paths.
