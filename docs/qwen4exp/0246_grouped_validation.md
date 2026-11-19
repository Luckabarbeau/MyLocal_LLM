# 0246 — Grouped IQ validation deduplication and fresh checkpoint baseline

## Objective

Work toward 50 emitted tokens/s without silently adopting Strata's route-tail omissions or suffix speculation. First measure current full-top-10 execution and remove demonstrated redundant CPU work. No claim that this localized change closes the throughput gap.

## Findings and implementation

The retained `diagnostics_0246_profile_v1/cpu_profile.prof` shows 2,304 `_unsupported_reason` calls for 768 grouped dispatches: executor eligibility, grouped entry validation, and repeated IQ entry validation. The grouped entry now calls a private `_grouped_validated` IQ launch seam after its own validation. Direct `iq.grouped` callers still validate; format and launch-limit guards remain. Executor eligibility checks remain deliberately intact. GPU kernels, inputs, FP32 reductions, routing and cache ownership are unchanged.

Modified: `mini_llm/ops/iq_packed_gpu.py`, `mini_llm/ops/grouped_quantized_experts.py`, and `tests/test_0237_warp_iq_gpu.py`. No activation quantization was introduced. The new private function requires successful metadata validation by its caller, with no intervening operand mutation.

## Validation

`MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_0231_grouped_iq_gpu.py tests/test_0237_warp_iq_gpu.py`: **286 passed, 2 skipped**. New test checks both public entry points validate once, produce identical output, and reject invalid activation dtype.

Real checkpoint forced IDs 1..16, sequential bounded processes using existing `probe_0229_decode_performance.run('slot_cache', output)`: pre-edit `diagnostics_0246_v1/slot_cache.json`, profiled control `diagnostics_0246_profile_v1/slot_cache.json`, post-edit `diagnostics_0246_v2/slot_cache.json`. Guard: 20 GiB sampled RSS / 120 s per child.

- Pre-edit unprofiled: **4.91794 tok/s**, sampled peak RSS 10,269,757,440 bytes.
- Pre-edit cProfile run: **8.54591 tok/s**; instrumentation and cache warmness differ.
- Post-edit unprofiled: **11.15820 tok/s**, sampled peak RSS 10,310,639,616 bytes.
- Pre/post logits and active-state snapshot hashes are **bit-exact at prefixes 1/2/4/8/16**.
- All 7,680 selected experts ran on GPU in the pre-edit trace; 1,952 promotions transferred 3,994,496,000 bytes. No CPU fallback in that trace. This short workload's resident set fits the 6 GiB cache; it is not a prolonged conversation/cache-eviction benchmark.

**Do not attribute the full-model timing difference to the edit:** cold/warm and runtime variability are substantial. A same-process, warmed, six-pair packed top-10 IQ3_S microbenchmark (`0246_validation_cost_v1.json`, D2560 F768, 200 calls per arm) reintroduced only the eliminated validation as control. Medians were approximately 0.28786 vs 0.28821 ms per call, i.e. **no convincing throughput benefit**. Outputs were bit-exact. One outlier favored the change; results do not support a speedup claim. Validation deduplication remains a small CPU work reduction, not a performance milestone.

## Gap and next experiment

50 tok/s requires roughly 20 ms per emitted token; the warm measured full-top-10 single-forward path is approximately 89.6 ms per forced token. These are different metrics from Strata with suffix drafts and route-tail omissions. The cProfile trace attributes much startup time to GGUF metadata parsing; optimizing that does not improve steady decode. Synchronous expert materialization consumed 0.689 s across the 16-token profiled trace, but cannot alone explain the remaining warm-token latency.

Next: obtain an exclusive warm-token critical-path breakdown (including promotion waits, Python launch/validation gaps, device packed GEMVs, and routing synchronization), then select the highest-cost verified path. Q8_1 native integer dot products and batched/asynchronous promotion are candidates, not yet numerically or performance validated here. Do not make a speculative arithmetic change to chase the target. Preserve the existing FP32 path and use identical-input operator tests before enabling changed precision.
