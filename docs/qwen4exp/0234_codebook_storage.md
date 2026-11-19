# Patch 0234 — IQ codebook storage experiment

## Objective and files

Measure whether divergent constant-memory codebook access is unnecessarily expensive, without changing a decoded weight or activation operation.

- `mini_llm/ops/iq_packed_gpu.py`: mode-keyed source/module caches.
- `tests/test_0234_iq_codebook_storage.py`: exact lane/output, stream, cache and bounded real-weight tests.
- `0234_iq_codebook_storage_v1.json`: preserved real microbenchmark.
- `0229_decode_performance_v6.json`: full-model extension, reusing successful v1–v5 measurements.

## Contract and architecture

`MINI_LLM_IQ_GPU_CODEBOOK_STORAGE=constant|global` selects only five CUDA table storage qualifiers. Default remains `constant`. Invalid values fail explicitly. Device/dtype/storage are separate module-cache keys; tables are compiled once per variant, not uploaded per call. CPU metadata/import paths do not initialize CUDA.

Packed bytes, lookup indices, FP32 arithmetic and reductions are unchanged. Existing tables retain their GGML MIT attribution; no additional third-party source or checkpoint was downloaded.

## Validation and measurement

Focused current run: **79 passed,1 skipped in0.83s**; the skipped real benchmark already has a retained successful artifact. Constant/global results are bit-exact across exhaustive lanes, mixed projections, grouped experts and all seven real expert-format triples.

The retained benchmark uses balanced warm ABBA/BAAB order,20 samples per mode/path, excludes JIT/weight upload and includes grouped pointer-table upload. It supports opting into global storage; timing alone does not prove a hardware serialization cause. Additional RSS871,804,928 bytes; checkpoint size/mtime unchanged.

Full-model storage-only extension: **8.566126→9.174106tokens/s** on forced IDs1–16, positions8–15. Same-prefix optimization errors versus the original FP32 production baseline remain ≤2.065315e-6 relative L2. This is not independent GGML full-model parity or a70–80tokens/s result.

Next dependency: reduce dispatch and kernel scheduling overhead. Keep both storage variants as independent oracles/switches.
