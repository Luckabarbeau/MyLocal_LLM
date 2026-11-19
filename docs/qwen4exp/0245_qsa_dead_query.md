# 0245 — Eliminate unused QSA index-query work

## Objective and mathematical invariant

The existing opt-in CuPy all-causal-cell QSA path retains every cell while `n <= index_topk + block_size - 1`. Selection therefore cannot depend on index scores. The query projection, query RMSNorm and query RoPE have no consumer on that path and do not update persistent state. Skip them there, but retain raw index-key projection, block pooling, normalization and positional rotation of pooled keys. Those updates are required when a later position enters sparse selection. Sparse and diagnostic paths still evaluate the query; CPU reference behavior is unchanged.

## Files and interface

- `mini_llm/external/qwen4exp_attention.py`: resolve the existing all-cell condition once and avoid the dead query branch. No new public option; controlled by the existing `fast_decode` flag. Projection callbacks are documented as pure; an unused callback is no longer evaluated or validated on the optimized path.
- `tests/test_0230_qsa_gpu_decode.py`: assert zero query calls for all-cell positions, restored calls for sparse positions and diagnostics, and preserve output-failure transaction coverage.
- `0245_qsa_dead_query_v1.json`: bounded RTX 5080 microbenchmark measurements.

No checkpoint encoding, dtype, arithmetic kernel, norm equation, routing or Qwen3-Next implementation changed. No new allocations or residency tiers were introduced. This patch does not implement Strata Q8_1 activation arithmetic or claim Strata numerical parity.

## Validation and verified state

- `MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_0230_qsa_gpu_decode.py`: **8 passed**. Tests cover exact persistent states, diagnostic output equality, sparse fallback output equality, and all-cell output tolerance versus the pre-existing reference path (`atol=2e-5`, `rtol=3e-5`). No tolerance changed.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_0208_qwen4exp_attention.py tests/test_0209_qwen4exp_model.py tests/test_0106n_qwen3next_attention.py`: **101 passed, 1 skipped**.

## Measured performance scope

Synthetic production QSA dimensions (D=2560, 24 query heads, two KV heads, head width 256, four index heads of width 128), FP32 random dense weights, repeated identical hidden vector, 128 sequential positions, one warmup per arm and four alternating-order pairs. The control reintroduces the exact unused projection/norm/RoPE ahead of the key projection on the same optimized module. Both arms synchronize at the run boundary; timings include Python validation and dispatch, not just device kernels.

| Pair | With unused query (ms/position) | Without (ms/position) |
|---|---:|---:|
|1|0.8710|0.7242|
|2|1.0746|0.9057|
|3|1.1297|0.9837|
|4|1.0370|1.0122|

Saved 0.025–0.169 ms/position in each pair. Final output was bit-exact between arms in every pair. Device memory-pool reservation was 426,479,104 bytes (~407 MiB), not total VRAM high-water. Host RSS was not measured. Variation is material; do not extrapolate this dense synthetic microbenchmark into packed full-model throughput or claim an independent-reference parity improvement.

## Remaining dependency

Measure the packed checkpoint hot path using a fixed workload, and independently quantify Q8_1 activation conversion versus retained FP32 arithmetic before adopting an opt-in quantized kernel. Long-context sparse selection and model-level throughput were not benchmarked here. Strata baseline rates also include suffix speculation and route-tail omissions, unlike MyLocal full top-10 execution.
