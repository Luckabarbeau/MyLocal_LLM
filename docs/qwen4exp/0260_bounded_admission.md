# 0260 — Remove compulsory synchronous expert movement from decode

## Objective and architectural choice

Test a structural Strata-inspired change, not another resident-kernel tweak: separate routing/execution from admission, and bound synchronous placement work. The0259 response case performed46,591promotions/94.914GBtransfers including prefill. Pinned Strata authority and bounded placement design are documented in0258 (`712fec6b4297c9cfd32ab43daaa2979a9aa01edf`). This patch is not a Strata profile loader, sized-slot planner or asynchronous mover.

## Files and contract

- `mini_llm/ops/expert_residency.py`: `AdaptiveExpertCache.set_admission_budget(max_promotions=None, window_requests=480)`.
- `scripts/probe_0254_strata_workload.py`: `--promotion-limit`, requires leadingwarmup; applies after warmup and before each subsequent request, including its prefill. Configuration recorded inresult.json.
- `tests/test_0260_admission_budget.py`.

None preserves old admission; zero disables new admissions; positive limits bound complete-bundle materialization attempts per request-count window. Failed attempts also consume budget. Hits and misses still update routing heat. A deferred miss executes the complete expert on CPU; no selected expert or router weight is dropped. Configuration begins a new window, does not evict weights, and must occur on the serialized dispatcher thread. `clear()` resets counters/window but retains configured policy. Existing idle-eviction configuration is independent; benchmarkidle eviction isoff. Default admission remains unrestricted.

480requests corresponds to one48layer/top10token in this model, not a portable token-count API. Window limits are discrete, not rolling-window limits. Existing in-flight protection, complete bundle publication and canonical source ownership remain intact. No new weight buffers or prefetch queue.

## Tests and numerical scope

Initial focused failure was in the new dense test's weight orientation: dense API expects[D,F], packed API[F,D]. Corrected the test only; production arithmetic was not changed. Retest:55passed1skipped onNumPy;48passed onCuPy combined with0257. Tests verify no materialization/eviction for deferred misses, heat updates, bounds/reconfiguration, hits advancing request windows, allocation-failure accounting, in-flight victim protection and complete weighted top10 CPU fallback.

Final broader suite with0261enabled:1875CPUtests passed962skipped; selected actualCuPy147passed3skipped. No independent full-model equivalence claim.

## Long sampled experiment

Same captured75/77promptIDs,128/512caps, temperature1/top-p.95/top-k20/seed1,15CPUworkers,6GiBcache, mapped-source50GiBwarming,4GiBMemAvailablefloor,8GiBAnoncap and180s/processwatchdog. Each run completed supervised with code0. No concurrent model processes.

| Policy afterwarmup | Response tokens | Decode forwards/s | Responsecase promotions |
|---|---:|---:|---:|
|Prior0259unrestricted|427|8.72394|46,591|
|Frozen|324|10.31149|0|
|Oneattempt/480requests|352|10.29833|427|

Responsecase includes77tokenprefill. Eliminating synchronous movement does **not** deliver the expected several-fold gain. CPU misses increase; sampled sequences differ because changing residency changes which selected experts use existingCPUQ8 versusGPUFP32 arithmetic. This is not an identical-trajectory speedup ratio and does not establish precision parity across sites. No new quantizer or equation was introduced. Keep these policies opt-in rather than silently changing arithmetic placement globally.

Artifacts: `diagnostics_0260_frozen_v1/`, `diagnostics_0260_bounded_v1/`; comparison in `0260_0264_pipeline_comparison_v1.json`.

Frozen sampledRssAnon825,151,488bytes/RssFile55,135,801,344bytes; bounded823,713,792/55,181,402,112bytes. Category maxima are separate samples, not a simultaneous peak sum; mapped pages remain reclaimable. No expert payload duplication. Device budgets remain unchanged; exact allocator snapshots inartifacts, not device-wideVRAMhighwater.

## Conclusion / next dependency

Heavy synchronous movement is real but is not the sole cause of the50t/s gap. Generic scalarCPUQ2_0 and host/launch coordination remain plausible costs. Next0261 isolates SIMD CPU arithmetic without changing its Q8 contract. Profile/sized startup placement and asynchronous protected movement remain outstanding; this patch is not a replacement for them.
