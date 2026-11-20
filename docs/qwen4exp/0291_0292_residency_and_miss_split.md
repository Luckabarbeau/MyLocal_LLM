# 0291–0292 — Post-incident benchmarks and CPU/PCIe split evidence

## Objective and disposition

Benchmark0289's compact resident source before increasing RAM use, then measure the dominant CPU-result join and the proposed heterogeneous miss split. **No production equation, arithmetic kernel or default was changed in these patches. Target50tokens/s remains unmet.**

0291 is implemented and validated as a diagnostic/residency benchmark.0289 is now actual-checkpoint end-to-end exercised with a **512MiB partial CPU source**, not a full/pinned complement or performance-optimized residency policy.0292 validates a standalone scheduling experiment, **not an integrated asynchronous model mover**.

## Modified/new interfaces

- `scripts/probe_0291_resident_host.py`: CPU-only actual-checkpoint packed-copy and registered row-pool A/B. Source32..512MiB, workers1..23, explicit isolation required; no CUDA/full-model initialization. Complete top10 groups per distinct format combination that fits the approved budget. Eight alternating measurement blocks per storage site; raw samples retained. Reports byte parity, identical-input CPU parity, copy time and before/after/close memory.
- `scripts/probe_0292_miss_split.py`: isolated, bounded actual expert operator experiment. At most32MiB owned pinned expert data per format group; separate reusable GPU staging arena; dedicated copy stream and event before consumption. Four or seven resident experts, zero..three staged misses, remaining experts on19-worker native CPU pool plus participating waiter. All ten outputs are evaluated. No router-weight combination, durable admission, service-thread mover or model integration is claimed. Timings include Python dispatch/completion fence, but not model graph, routing, final combination or transfer-service overhead.
- `scripts/probe_0266_critical_path.py`: diagnostic now requires0290 isolation before CUDA/headers/output creation. Added explicit cache/warmup/reserve/fixed-token options. Removed the implicit50GiB mmap warmup; default is0. Captures16tokens after64response decode steps, with node-level CUDA graph tracing.
- `tests/test_0291_residency_benchmarks.py`: refuses unisolated diagnostics before side effects; bounds source/workers.

Reference revisions/checkpoint unchanged. CPU execution uses pinned GGML per-format Q8_K/Q8_0; GPU execution uses the already validated native Q8_1 path. Moving execution site can change activation rounding; same-site oracles are kept separate, not forced into false cross-site bit equality.

## Safety conditions and memory scopes

All experiments ran sequentially through `scripts/run_0290_isolated_benchmark.py`, outside the desktop cgroup, with swap0 and minimum12GiB global reserve. CPU copy experiment:1.75GiB high/2GiB hard. Longer model experiments:32GiB high/36GiB hard, timeout200s plus existing180s inner guards; **no large source warmup**. Standalone staged split/GPU tests:3GiB high/4GiB hard. No reference model ran concurrently.

The cold baseline kernel cgroup peak32.003GiB briefly crossed MemoryHigh between polls and incurred reclaim; the hard limit was not reached. The guard did not abort and no OOM events occurred. This is not a claim that no reclaim occurred. Later runs reused shared file-cache pages charged outside the new service and therefore had much smaller **charged cgroup** peaks despite larger file-backed process RSS. MemoryMax is not a total shared-RSS limit. Host MemAvailable counts reclaimable clean cache and is not physical unused RAM.

No44GiB anonymous source, full pinned complement, checkpoint modification, cache dropping, unrelated process killing or VRAM-budget bypass. The old large-memory/pinning status remains unvalidated. Small CUDA registration was resumed only after the CPU source experiment passed; each split group was capped32MiB.

## 0291 — Copy and warm CPU operator benchmark

`diagnostics_0291_host_v1/result.json`:

-60actual experts across six fitting format combinations; budget128MiB, actual anonymous storage **115,834,880bytes** (~110.5MiB).
-Setup **0.143406s**; all packed bytes and all tested native outputs bit-exact.
-Input deterministic FP32 `[1,2560]`, counts1/2/5/10,19native workers plus participating waiter, tasks scheduler/50us spin, BLAS/OpenMP1.
-RssAnon360,292,352 ->476,135,424 ->361,566,208bytes before/copy/close. The source was released; residual process/library metadata is not a leaked whole source.
-Warm mmap/anonymous timings were effectively equivalent. Top10 mmap/anonymous ratios ranged0.9795..1.0656 across the six combinations. Single-threaded or cold-disk results are not inferred.

Conclusion: moving bytes into anonymous RAM does not by itself accelerate already resident packed arithmetic. It can still protect a useful working set from file-cache eviction; that separate under-pressure benefit is not measured here.

## Longer fixed-token model comparison

All four runs used the exact retained warmup/response tokens from `diagnostics_0284_bf16_hc_cache7_v1/result.json`:128warmup tokens,359response tokens,358post-prefill forwards, actual75/77prompt IDs. Full logits/head/sampler still execute, but tokens are forced: **controlled replay, not independently sampled natural generation**.

Common configuration: GPU expert+denseQ8, exactBF16HCstorage/FP32arithmetic,7GiB uniform fixed-slot cache frozen after warmup, native callback graphs, pre-MoE/shared graphs, deferred shared launch,19native workers/50us plus participating waiter, no manual affinity, no mmap bulk warmup, context1024/FP32KV, complete top10.

| Artifact | Partial anonymous source | Decode forwards/s | Decode wall s | Response+prefill/setup disk bytes | Major faults in same scope |
|---|---:|---:|---:|---:|---:|
|`diagnostics_0291_baseline_v1` cold|0|21.295015|16.811447|12,701,478,912|168385|
|`diagnostics_0291_resident512_v1` warmer|536,436,736B|33.905481|10.558765|10,268,672|224|
|`diagnostics_0291_baseline_v2` warm|0|33.870911|10.569542|0|0|
|`diagnostics_0292_deferred_v1` warm, deferredQSAcheck enabled|0|34.019848|10.523269|0|0|

Exact full precision results: `0291_0292_benchmark_summary_v1.json`. The last row's unrounded value in that artifact is authoritative.

The512MiB source copied263complete bundles from21753nonresident candidates (~1.2% by expert count); source setup's internal copy took0.223674s. All263 were registered in native plans and CPU/GPU aggregate expert counts matched the no-copy run. CPU expert arithmetic is unchanged. Copying all model experts was neither attempted nor implied.

**Do not advertise21.30 ->33.91 as a resident-copy speedup:** the cold run was I/O-confounded. Candidate/warm-control ratio **1.001021** (~0.10%), no material gain. The deferred-QSA-check trial ratio **1.004397** (~0.44%) is also not a convincing gain from one pair; defaults stay off. Prefix/state/logit hashes were not newly captured by these throughput runs, and forced token equality alone is not new numerical-equivalence evidence. Actual copied-source operator outputs and the existing tiny model parity tests remain the correctness evidence.

Warm baseline native pool accumulated4.355367s over435response model steps (77prefill+358decode), about10.01ms/step **not an exclusive decode critical-path attribution**. Decode process CPU88.902s /10.5695wall is ~8.41active-core equivalents;19native workers plus the participating host were enabled. Multithreading was not silently limited to one worker by BLAS/OpenMP1.

### Memory observations

| Run | Kernel charged cgroup peak GiB | Max sampled full PSI avg10 |
|---|---:|---:|
|Cold baseline|32.003|1.41%|
|512MiB candidate|9.348|0.01%|
|Warm baseline|1.671|0%|
|Deferred-check trial|1.667|0%|

Warm baseline process RSS40,468,934,656bytes is mostly shared mapped/file-backed data; RssAnon864,346,112bytes. Candidate RssAnon1,399,971,840bytes reflects the bounded source. This does not contradict smaller charged cgroup peaks. Pool used/reserved GPU snapshots and raw host memory are retained in the summary; neither is a device-wide VRAM high-water measurement.

## Fresh critical-path capture

Artifacts: `0291_timeline_v1.nsys-rep`, read-only-exported `0291_timeline_v1.sqlite`, `0291_critical_path_v1.json`, statsCSV, `diagnostics_0291_timeline_v1/capture_window.json`.

Across16instrumented tokens:

- GPU span **514.771118ms**.
- GPU operation union **210.119744ms**; summed kernels **206.438835ms**.
- Gaps grouped by next CPU-result H2D: **177.495885ms**,740gaps. These are observed idle gaps, **not a proof all time is CPU arithmetic**.
- CPU-result H2D operation time: **2.884592ms** over768copies of102400bytes; data movement itself is not the177ms wait.
- Runtime `cudaMemcpyAsync` spans **160.593753ms**; overlapping host waits, not another additive latency category.
- Largest kernel totals: `q8_linear`72.400246ms, `q8_gate_up`20.917625ms, HCdown14.304950ms, HCup13.936872ms, QSA13.536585ms.

The capture supports targeting native CPU completion/dispatch and measured overlap rather than replacing fast copies indiscriminately. CUDA graph scheduling, host submission and CPU work overlap; kernel sum and CPU spans must not simply be subtracted from wall time to promise50/s. Nsight could not collect CPU context switches; its instrumentation overhead means this is not an uninstrumented throughput measurement.

## 0292 — Bounded concurrent CPU/staged-GPU experiment

`diagnostics_0292_split_v1/result.json` tests **56configurations** over seven actual mixed-format triples, including layer47IQ4_XS. Every CPU row is bit-exact to native CPU execution; resident/stagedGPUrows are bit-exact to local-VRAMQ8_1. Complete packed bundles are transferred and consumed after their copy event; all top10terms are present. No cross-site FP32/Q8_K/Q8_1 equality claim.

Selected measured median latencies (whole scheduling envelope):

| Format / resident count | All misses CPU ms | Best staged count | Split ms | Ratio |
|---|---:|---:|---:|---:|
|IQ3_XXS/IQ3_XXS/IQ4_NL;4hits|0.156975|1|0.135906|1.155x|
|IQ2_S/IQ2_S/Q2_0;4hits|0.196412|1|0.172257|1.140x|
|IQ2_S/IQ2_S/IQ4_NL;4hits|0.156934|1|0.146010|1.075x|
|IQ3_XXS/IQ3_XXS/Q2_0;4hits|0.172204|2|0.144136|1.195x|
|IQ3_S/IQ3_S/IQ4_NL;4hits|0.198588|1|0.174078|1.141x|
|IQ3_S/IQ3_S/Q2_0;4hits|0.206439|2|0.196153|1.052x|
|IQ4_XS/IQ4_XS/IQ4_NL;4hits|0.123058|0|0.123058|1.000x|

At7resident/3misses, CPU-only won five of seven format groups. One staged miss helped the two Q2_0-down/IQ3_XXS orIQ3_S cases (~1.20x/~1.12x). Three staged misses generally lost. These are exploratory seven-block medians, tested in sequential stage-count order, not randomized confidence intervals; frequency/cache/order effects remain possible.

**Conclusion:** use format- and miss-count-aware admission, not a fixed blanket GPU-miss policy. Staging may save tens of microseconds per selected MoE call, not automatically provide the remaining~1.48x full-model gain to50/s. A native asynchronous mover remains justified as a bounded experiment, not a guaranteed solution.

Next implementation dependency: extend the native service **outside CUDA host callbacks** to publish one/two complete staged bundles from a bounded pinned tier while other misses execute on the native pool; explicitly reserve/lease staging slots, use ready events and failure fencing, and retain deterministic once-weighted combination. Keep cheap/low-miss cases CPU-only and compare identical prefixes/states before longer benchmarks. Do not make CUDA calls from the existing CUDA host function. The diagnostic's Python orchestration is not the final decode architecture.

## Tests, limitations and next action

Final isolated tests:

- **96passed,4skipped,1deselected** CPU: new diagnostic preconditions,0290guard,0289source, tiny model, budgets, row pool and workload contracts. Actual installed-tokenizer checkpoint test intentionally deselected.
- **20passed,1skipped** actualGPU:0289source registration/CPU-GPUhandoff integration,0275/0276handoff graphs,0280deferredQSAvalidation.
- Real diagnostic24CPU storage-count cases and56CPU/GPUsplit cases pass their exact same-site oracles.
- No production implementation changed this turn, so no new production arithmetic precision claim or full training-suite result.

Detailed artifacts/sourceSHA256: `0291_0292_benchmark_summary_v1.json`; JUnit `0292_cpu_tests_v1.xml`, `0292_gpu_tests_v1.xml`. Every run's0290guard log preserves its exact command/cgroup/limits/pressure and termination. Prior reports were not overwritten.

The pinned Strata workload remains a target rather than a fully equivalent configuration: it has different context/KV policy, suffix drafting and eligible tail omission. MyLocal retains complete top10, no drafting, FP32KV/context1024. No new Strata server benchmark was run in parallel with these experiments. A warm fixed-token34/s observation is not natural conversation50/s.
