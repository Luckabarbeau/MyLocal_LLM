# 0293–0294 — Native service miss split (opt-in)

## Current disposition

Implemented native bounded staged execution with exact per-site tests and actual-checkpoint execution. **Not performance optimized; no default enabled.50tokens/s remains unmet.** Work continues on the measured CPU-result join and residency policy; do not infer a speed gain from implementation alone.

## Interfaces and ownership

- `native_expert_staging.h/.py`: `MINI_LLM_NATIVE_MOE_STAGE_LIMIT=0|1|2` (default0), `MINI_LLM_NATIVE_MOE_STAGE_SOURCE=bounce|pinned` (defaultbounce).
- Requires native service, native handoff/device plans and packed GPUQ8. Eligible gate formatsIQ2_S/IQ3_S/IQ3_XXS, at least5misses. Other projections/expert contributions are not dropped.
- One shared16MiB pinned arena and16MiB VRAM arena per native pool, two fixed8MiB slots. No per-token allocation. Model preflight reserves32MiB plus per-layer Q8/term/metadata scratch before construction.
- CPU service submits remaining misses to the existing native row pool before staging. It publishes complete host bundles and a sorted-rank map with release/acquire epochs. GPU kernels pull complete original packed bytes into VRAM before Q8expert kernels; final combination includes residentGPU/stagedGPU/CPU terms in expert-ID order, weights once.
- CPU arithmetic remains GGML per-formatQ8_K/Q8_0; staged/residentGPU arithmetic remainsQ8_1. Execution-site rounding may change routing/logits; exact comparisons use the proper site oracle, not false CPU/GPU bit equality.
- `pinned` mode only stages eligible misses already in the bounded registered CPU source. Setup queries each complete bundle's actual CUDA alias. GPU-copy source aliases are published with the epoch and copied into VRAM metadata before the weight-copy kernel. No extra host weight copy. Uncached/unpinned misses execute on CPU.
- Failure/timeout suppresses reads of incomplete maps/host data, poisons output and fails the request. Two-second bounds unchanged. Plans fence readers and unregister from the service before stage descriptors/source/arenas retire. Registry/canonical mmap ownership remains independent of GPUcache; staging is not durable admission.
- Counters preserve actual cache misses even when executed by stagedGPU. CPU/GPUexecution counts subtract/add staged experts. Shared buffer sizes are exposed once in executor statistics; native elapsed time includes the CPU/staging join, not exclusiveCPUarithmetic. `native_stage_host_fill_ns` excludes GPUcopy time.

## Diagnosed CUDA submission deadlock and correction

Initial runtime-DMA service implementation passed uncaptured tests but intermittently timed out during captured replay. No timeout or output tolerance was relaxed.

`0293_timeout_trace_v1.nsys-rep/.sqlite` established: main-thread and service-thread `cudaMemcpyAsync` API calls both blocked for~2000ms while graph `await_stage` ran~2000ms. API calls returned only after bounded wait failure. The transfer submission and graph wait formed a runtime scheduling dependency. It is unsafe to assume the driver will accept dynamic DMA from a service while other host submissions block behind the waiting graph.

Localized correction removed all CUDA calls from the service and moved the packed transfer to a coalesced16-byte GPUcopy kernel after host publication. This is **staging once into VRAM**, not repeatedly dotting mapped-host weights. Final native thread enters neither CUDA nor Python. Initial failed results are preserved; final tests pass. Reference mathematics/checkpoint untouched.

## Verified tests and state

- Initial candidate:20passed2failed; capture diagnosis5passed3failed. Failures were bounded timeouts/poisoned results, not small numerical mismatches.
- GPU-pulled correction: **22passed** (`0293_gpu_tests_v2.xml`).
- Direct pinned-source extension plus native/service/source/graph regressions: **35passed1skipped** (`0294_gpu_tests_v1.xml`). Includes bounce-buffer canary proving direct mode did not copy host weights, and CPUfallback when no pinned bundle is eligible.
- Actual checkpoint, same staged arithmetic: captured vs uncaptured exact logits at1/2/4/8/16 over two isolated requests; **10exact logit comparisons**, all persistent-state hashes identical, request reset hashes identical. Artifacts `diagnostics_0293_parity_{graph,uncaptured}_v1`. No new independent llama.cpp parity claim.
- `probe_0265_gdn_graph_parity.py` now requires isolation and12GiBreserve, and reports staged routing/statistics accurately.

## Longer fixed-token measurements so far

All use same retained128warmup/359response IDs (358decode forwards),19workers+participatingwaiter,7GiBfrozen uniformGPUcache, Q8dense/expert and BF16HCstorage. Context1024/FP32KV, fulltop10, no suffix drafting. New exclusive artifact directories, no bulk mmap warmup. These are sequential single observations, not confidence intervals.

| Artifact | Change | Decode forwards/s |
|---|---|---:|
|`diagnostics_0293_service_control_v1`|Native service, no staging|34.463188|
|`diagnostics_0293_stage1_v2`|One bounce-staged miss when eligible|34.264559|
|`diagnostics_0294_pinned1_v1`|Direct pinned source1GiB, one eligible staged miss|35.537949|
|`diagnostics_0294_readonly_v1`|Above + input-only graph copy-back disabled|37.012342|
|`diagnostics_0294_pinned2_v1`|Above with2GiB source|35.974541|
|`diagnostics_0294_readonly_control_v1`|Input-only copies disabled,1GiB pinned CPU source, **no staging**|37.846813|

The first cold `diagnostics_0293_stage1_v1` attempt was **aborted by MemoryHigh** at~55s during the response, with no benchmark result inferred. Its partial progress is retained; limit was not raised. Subsequent runs reused warmer OS cache, so cold partial timing is not a speed claim.

Bounce mode filled~21.81GB host staging bytes/435response modelsteps, with2.383s host-fill time (~5.48ms/step); it did not improve the full model. Direct1GiB mode removed that copy (~0.790ms cumulative host-fill bookkeeping for4783staged experts) but only covered526of21753nonresident candidates. Enlarging the pinned tier to2GiB did not improve throughput. The strongest no-staging/input-only control was faster than the staged candidates: **do not promote staging or larger pinned sources to defaults**.

## Safety and comparison with Strata

All experiments sequential under0290isolated services,32GiBhigh/36GiBhard/swap0/12GiBhostreserve for actual model,3GiBhigh/4GiBhard for syntheticGPUtests. No unbounded overnight worker, large44GiBpinning, source checkpoint modification, GPUreset or guard weakening. Cold MemoryHigh abort worked as intended.

Pinned Strata reference remains712fec6b4297c9cfd32ab43daaa2979a9aa01edf. Fresh raw source retrieval timed out; no complete new snapshot claimed. Existing0258audit contracts describe CPU/PCIe separation and measured cost/admission, but no claim is made that this implementation reproduces its active mover. The historical~50emittedtokens/s includes suffix drafting/eligible tail omission and differentKV/context policy; MyLocal's complete-top10single-forward rate is not automatically equivalent.

Next dependency: evaluate routing heat/cache misses on the strongest low-copy path before adding more staging. CPU fallback/placement remains critical; blanket staging and increased pinned memory have not solved it. Any further admission change must be causal, bounded, protected and tested with unchanged complete routing.
