# 0259 — Bounded native row-phase CPU expert pool

## Objective and authority

Implement the first adaptation from `0258_strata_pipeline_audit.md`: persistent native CPU workers, one quantized layer input shared across selected experts, and row-level scheduling. Strata authority is `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`, specifically executable `src/kernels/cpu/pool.cpp` and the native expert interface documented in0258. This is an original scheduler, not a port of Strata's runtime.

**Status:** CPU operator numerically validated against retained GGML execution; heterogeneous CuPy integration tested; real sampled workload completed. Not50tokens/s, not all-format GPU Q8 integration, and not full independent-model equivalence.

## Files and interfaces

- `mini_llm/ops/ggml_packed_cpu.cpp`: persistent `RowPool`, bounded workspace, C ABI create/submit/wait/destroy. Existing synchronous linear/expert arithmetic retained.
- `mini_llm/ops/ggml_packed_cpu.py`: `GGMLPackedCPURowPool`, owned batch handles, ctypes declarations and pthread build flag; obsolete0256 integration commentary corrected.
- `mini_llm/ops/expert_residency.py`: opt-in `cpu_packed_backend='ggml_pool'`; batch submit before resident GPU dispatch and drain before combine.
- `mini_llm/external/qwen4exp_model.py`: accepts `expert_cpu_backend='ggml_pool'`; default remainsnative.
- `scripts/probe_0254_strata_workload.py`: accepts new backend; bounded parent monitoring no longer opens cross-process ptrace-protected `/proc/io`; detects explicitly zombie status before reading RSS fields. The live child still records its own I/O. Memory and timeout checks are not disabled.
- `tests/test_0259_ggml_row_pool.py`, focused supervisor regressions in `tests/test_0254_strata_workload.py`.
- `0259_row_pool_comparison_v1.json`: machine-readable aggregation of retained result files.

`GGMLPackedCPURowPool(executor, d_model, d_ff, workers, capacity=10)` preallocates hidden and quantized workspace and starts persistent workers. Dimensions must be positive multiples of32, bounded by65536; capacity1–512, workers1–256. Integration uses capacity10, not a512-expert weight copy. Workspace is O(capacity×(D+F)); packed weight pointers refer to immutable canonical sources.

`submit(x, experts, out=None)` accepts one shared contiguous-or-copyable FP32 activation row and complete gate/up/down bundles with matching geometry. It prepares at most one Q8_K and one Q8_0 input for the entire batch. Different function pointers for the same activation format are rejected rather than presumed interchangeable. It returns while native row work executes. One outstanding batch is allowed; a second submission is rejected, never placed on an unbounded queue.

`batch.result()` drains and returns unweighted FP32 `[E,D]`. The handle retains sources and destination until workers finish. Supplied output must be writable, contiguous, correctly shaped and non-aliasing. Old results remain valid across epochs. `close()` drains, joins workers and is idempotent. Calls on a closed pool fail explicitly. The existing router applies weights once, outside CPU arithmetic.

The native phases are gate/up rows with the retained SiLU expression, per-expert hidden quantization, then down rows. Fixed16-row task chunks span all selected experts. All workers cross a mutex/condition-variable phase barrier; the final barrier retires a batch only after every worker finishes. Epoch publication, descriptor ownership and barriers prevent stale claims or descriptor reuse. There is no native per-layer vector allocation. Python descriptors and existing pinned D2H/H2D slabs are still dispatch-local; do not describe this as allocation-free decode.

The pool is used only by eligible singleton decode batches, including sequential prompt prefill that reaches that interface. Nonpacked or larger batches, direct single-expert calls and diagnostic paths retain existing execution. The general Python worker pool remains available for those fallbacks; it does not schedule pooled batches.

## Precision and correctness

No model equation, routing, state evolution or packed encoding changed. CPU arithmetic still calls the explicitly selected pinned GGML library:

`171e8846b4af9766c354064cb776cb34a50f053f`, `libggml-cpu.so` SHA256 `924ac13f3e949cb1b62eddff20221600a6efcb1a2b99eb4abd390e75c23df839`.

IQ2_S/IQ3_S/IQ3_XXS/IQ4_XS use Q8_K; IQ4_NL/Q2_0 use Q8_0. This is **not GPU Q8_1**. The original FP32 decoder and0256 per-expert GGML path remain available. Independent reference sources/builds and checkpoint files were not changed.

Tests compare pooled outputs bit-exactly with the existing synchronous GGML expert path for mixed triples,1/2/7/10 experts,1/4/15 workers and120 back-to-back epochs. Tests cover fewer experts than workers, delayed native dot callbacks, source lifetime, repeated result retrieval, output retention, close/drain, busy/closed rejection, shape/dtype/capacity/alias checks and shared per-format quantizer call counts. CuPy integration tests exercise packed CPU results returning to GPU and ascending weighted combination. This is not a thread-sanitizer proof or full-model logit comparison.

## Tests executed

Environment: project `.venv/bin/python`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; explicit GGML library path as above.

- Initial focused0259+0256: **47passed,3skipped**.
- Final CPU regression group0254/0259/0256/0229 batch/0209 model/0210 generation/0106d async/0106j ownership/0106p Qwen3-Next: **120passed,7skipped in6.86s**, including the zombie regression. The corrected supervisor's18tests also pass separately. Scoped `git diff --check` passes; an unrestricted check reports a pre-existing `.gitignore` trailing blank line, left untouched.
- Actual CuPy0259/0257/0229 batch/0106j group: **74passed,1skipped**.
- Optional warmed operator timing: **1passed**,15workers, syntheticD2048/F768:

| CPU experts | Whole-expert futures median | Row-pool median | Ratio old/new |
|---|---:|---:|---:|
|1|0.3854ms|0.5684ms|0.678× (regression)|
|3|0.9614ms|0.3617ms|2.658×|
|10|1.5615ms|0.5018ms|3.112×|

Small15-sample, warm synthetic operator measurements are not full-model gains. Keep the pool opt-in; one-expert wake/barrier overhead remains relevant. No affinity, host-drain or spin tuning has been inferred from these measurements.

## Long sampled workload

Same captured Strata75/77 prompt IDs, temperature1, top-p.95, top-k20, seed1, thinkingxhigh,128-token warmup/512-token response cap. Sequential processes,15CPUworkers,6GiB expert arena,14GiB allocator cap, heapvictim policy, mixedGPUgrouping off, fulltop10, no suffix speculation. Each process retains the180-second watchdog,8GiB anonymous-memory cap and4GiB MemAvailable floor. Requested50GiB source warm touches the50,292,326,400-byte mapped expert source without a private full-weight copy.

| Run | Warmup decode forwards/s |427-token request forwards/s|Supervisor|
|---|---:|---:|---|
|Old GGML controlv1|6.7715|7.5595|Child0/result complete; parent failed missingRSS exit race|
|New poolv1|8.8017|8.7239|Completed0, no stop reason|
|Old GGML warmed controlv2|8.2030|8.2097|Completed0, no stop reason|

Use **controlv2** as the better comparison: source residency after warmup preparation was50,292,326,400bytes vs pool50,258,399,232bytes. Controlv1 was only31,948,886,016bytes and had a slower cold startup. Do not advertise its15.4% difference as the isolated scheduler gain.

Observed warmed request improvement is **1.06264×,6.26%**;426forwards take51.88974s versus48.83114s. Pool emitted-token convention is427/decode_seconds=8.74442/s; including prefill,427/total_seconds=7.41829/s. These are different metrics. Both128-token and427-token sequences are exactly identical across all three runs; all final device-cache counters match. No full-model logits were captured, so generated-ID equality is not called mathematical equivalence.

Only one completed pair at near-full source residency was measured, not randomized interleaved trials or a confidence interval. Pool run performed more disk reads/major faults than warm control (process-final10.29GB/28,706 versus0.512GB/969; includes startup/prefill), so residency/page-fault variability is still a limitation.

Artifacts: `diagnostics_0259_long_control_v1/`, `diagnostics_0259_long_pool_v1/`, `diagnostics_0259_long_control_v2/`, including result, preflight, per-case, memory and supervisor journals. Previous reports were not overwritten.

Reproduce the bounded candidate run with an exclusive new output directory:

```sh
MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/probe_0254_strata_workload.py \
  --output docs/qwen4exp/diagnostics_0259_long_pool_v2 \
  --host-warm-gib 50 --mem-available-floor-gib 4 \
  --victim-policy heap --cpu-backend ggml_pool --cpu-workers 15
```

`pool_v2` is a suggested new run, not an existing result. Use `--cpu-backend ggml` for the retained whole-expert scheduler control. Omit `--group-mixed-decode` to match the measured pair.

## Memory

Pool run sampled RssAnon highwater824,041,472bytes, RssFile55,226,150,912bytes; warmed control825,638,912 and55,251,689,472bytes. These are separate sampled category maxima, not an exact simultaneous peak sum. The large file-backed mapping remains reclaimable, not pinned or privately duplicated. Both end snapshots: CuPyused12,148,603,904bytes, reserved12,334,209,024bytes. These are allocator snapshots, **not device-wide VRAM highwater**. Expert arena remains6,440,345,600bytes/2,419slots with4MiB staging. Explicit guards passed in both fully supervised runs.

## What still limits the pipeline

Across warmup plus request, both schedulers perform68,237promotions and139,100,006,400packed transferbytes. The427-token request alone adds46,591promotions and94,913,766,400transferbytes, **including its77-token prefill**. The final cache hit rate is61.31%. This is evidence of heavy synchronous movement, not a measured exclusive-critical-path attribution. Do not divide the case transfer count solely by426decode forwards and label it a decode-only rate.

Strata's profile/sized slots and bounded periodic adaptation differ materially from this on-demand synchronous churn. The next dependency is a validated placement/admission policy with stable canonical sources and bounded updates, followed by protected staging/shadow slots and event publication. Do not remove synchronization without protecting source and slot reuse.

Also still missing: Strata's specialized CPUQ2_0 path, hybrid-core affinity/host participation, reusable pinned activation/output scratch, all-formatGPUQ8, graph-compatible device metadata and captured pre/post segments. INT8KV, efficient prefill and suffix speculation require separate numerical gates. The50t/s target remains unmet. No claim is made that this patch implements every audit item or makes benchmark configuration identical to Strata.
