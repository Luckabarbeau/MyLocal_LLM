# 0280 — Verify QSA fence attribution without removing finite validation

## Objective and established critical path

Follow-up to0279: determine whether repeated QSA finite-check readbacks are avoidable overhead or exposed dependency time. **The experiment is mathematically validated within its tested scope but shows no performance gain. Default remains unchanged.**

`0279_node_critical_path_v1.nsys-rep` was captured with CUDA graph **node** tracing, 15workers+callback participation, native handoff/shared/pre graphs, QSA frontend off. It is a diagnostic window inside the longer response, not a throughput benchmark:

-16tokens:635.600326ms wall.
-Complete visible node kernel sum380.769236ms; operation union including kernel/copy/memset363.543719ms (~22.72ms/token).
-Runtime cudaMemcpyAsync API wall219.885ms, not DMA duration.
-`0279_node_copy_attribution_v2.json` correlates all8416copyAPIcalls to direct transfers and deepest NVTX boundaries. The dominant groups are **4-byte D2H reads inside QSA mixers**, not bulk expert-result payloads: e.g. L31/16calls23.386795ms, L23/21.702852ms, L35/18.229131ms. Layer/CPU/GPU/API spans overlap and cannot be added into an exclusive decomposition.

The initial attributionv1 matched an unversioned API name and returned no rows; v2 uses the actual exported `cudaMemcpyAsync_v3020` name and verifies8416calls. Both artifacts are preserved. This is a diagnostic-query correction, not a model/reference change.

## Implementation and contracts

Files:

-`mini_llm/ops/finite_checks_gpu.py`: new `finite_status_gpu(*arrays,out=None)` performs the same complete GPU finite check and returns a device uint32 scalar (0valid). Existing `all_finite_gpu` remains a synchronous host-bool wrapper.
-`mini_llm/ops/qsa_commit_gpu.py`: conditional GPU cache commit after that validation. Invalid status writes **none** of K,V,raw-index or newly pooled index history. Contiguous FP32 state/projections required explicitly.
-`mini_llm/external/qwen4exp_attention.py`: optional internal `validation_status` keyword. Default standalone behavior still validates synchronously before committing. Deferred callers must retire a failed request, because Python length advances speculatively even though invalid cache writes are suppressed.
-`mini_llm/external/qwen4exp_model.py`: opt-in `MINI_LLM_QSA_DEFER_VALIDATION=1`, requiring fast CuPy QSA. Owns48uint32statuses (~192bytes), resets per-token, provides per-layer views, checks them **once before a successful model step returns**, reports the first invalid layer, marks request failed, and prohibits resumption. New requests receive isolated state and cleared statuses. Diagnostics retain the synchronous path. Cleanup fences pending device status/state operations.
-Workload configuration records the flag; `tests/test_0280_deferred_qsa_validation.py` checks numerical and failure contracts.

No check is deleted. No projection, normalization, routing, RoPE, attention, pooling or expert equation/precision changes. Cache mutation remains conditional on the same original projection/result finite checks. Additional later layers can enqueue before a failed token is reported, but the entire request is then invalidated, never exposed as successful/resumable. This is **not** standalone transactional rollback of an entire model token.

## Validation

- Focused operator/model-lifecycle file: **9passed**. Every checked projection can independently be nonfinite; K/V/raw/pool caches remain unchanged. Complete pool-block writes are covered, not just single-token KV writes.
- Tests compare24positions across the dense/sparse width transition and verify exact output/cache equality to synchronous validation.
- Model-level synthetic test verifies first-invalid-layer error, failed-state rejection and clean subsequent request.
- `0280_deferred_validation_parity_v1.json`: actual checkpoint, two requests, fixedprefixes1/2/4/8/16; **bit-exact logits and all active-state hashes** against retained0276synchronous state. This is same-arithmetic replay parity, not new independent llama.cpp equivalence.
- Final NumPy inference regression: **1939passed,1025skipped**.
- Final actual-CuPy inference regression: **1211passed,18skipped**.

Commands are the prior0279broad groups with `tests/test_0280_deferred_qsa_validation.py` added to GPU selection; CPU `tests/test_02*.py` includes it automatically. Explicit pinned GGML library and single-thread unrelated BLAS/OpenMP settings retained. JUnit: `0280_cpu_regression_v1.xml`, `0280_gpu_regression_v1.xml`. Verified source hashes: `0268_0280_performance_v3.json`.

## Longer-run result and memory

`diagnostics_0280_deferred_validation_v1/result.json`: same captured Strata prompt/template/sampling settings,128warmup/512responsecap,23nativeworkers+participatingcallback, frozen6GiBexpertcache, mmapwarmup50GiB, time/memoryguards. The resulting413tokenIDs exactly match the prior0277workers23control:

-412decodeforwards/16.585260s = **24.84134/s**.
-Sampling call wall0.254327s versus control0.630282s: apparent waits moved between seams, but total decode **did not improve**.
-Prefill3.681023s; complete request20.267512s.
-Native CPU spans about14.18ms/forward including77prefillpositions, versus13.21ms in the best23-workercontrol. These are separate runs, not exclusive CPU critical-path differences or confidence intervals.
-Sampled RssAnon858517504bytes; RssFile54917722112bytes (separate category maxima). Default GPU pool snapshot12157393408used/12345443840reservedbytes, excluding private graphs/driver allocations. No full weight duplication or allocation of PLE/expert banks.

**Do not enable this merely to remove visible readback waits.** The result is compatible with those waits exposing required preceding computation and with run-to-run native scheduling variation. It does not prove all readback overhead is irreducible.

## Updated optimization direction and limitations

Best completed longer result remains **26.95159/s**, target50/s unresolved. Node-level profiling now reveals substantial packed projection and resident-expert work previously hidden by coarse graph tracing: ~90.35ms `_packed_linear_decode_kernel`,56.56ms `packed_linear`,84.67ms resident gate/up and24.73ms resident down across the16-token capture. HC read kernels add ~50.18ms. These are overlapping operation totals, not independent wall-time contributions.

Prioritize **integrated, independently validated packed Q8 GPU arithmetic**, native CPU quantization/row cost and measured CPU/PCIe splitting rather than deleting synchronization or assuming Python launch removal alone yields50/s. CPU currently uses GGML format-specificQ8_K/Q8_0; GPU experts still use FP32 activations. Existing standaloneIQ3_S×Q8_1 operators are not a model-wide mixed-format dispatch. All-format GPU Q8 remains a substantive missing optimization.

Strata still differs in suffix verification/multi-token groups, route-tail omission, INT8KV, long context capacity, sized-profile slots and pinned-complement PCIe execution. Its emitted-token rate is not automatically a full-top10 non-speculative forward rate. No new graph defect was established and no numerical tolerance was relaxed. Keep the current reference paths and explicit feature flags.
