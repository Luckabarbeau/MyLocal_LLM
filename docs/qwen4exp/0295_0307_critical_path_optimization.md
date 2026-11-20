# 0295–0307 — Critical-path optimization and retained negative results

## Outcome and comparison scope

**50 tokens/s is not achieved.** Best earlier controlled single observation is 42.64032 decode forwards/s (`diagnostics_0302_qsa_v1`). The latest adjacent, same-token control/shared-grid pair is **40.05676 / 40.17001 forwards/s**, only 0.28% apart. Do not advertise the GPU operator gains as a demonstrated model gain or use the earlier best as the guaranteed current conversation rate.

The controlled workload retains 128 warmup output tokens and 359 response output tokens, including EOS, with 358 post-prefill decode forwards. Exact captured Strata prompts/template/sampling settings and full output head/sampler execute, but output IDs are forced. Natural generation is reported separately below. MyLocal uses full top-10, no suffix drafting, context 1024 and FP32 KV. Historical Strata emitted-token rates included suffix drafting, eligible expert-tail omission, context 262144 and INT8 KV. They are not an equivalent single-forward throughput denominator.

Machine-readable results, memory scopes, source SHA256s and JUnit summaries: `0295_0307_optimization_results_v2.json` (v1 retained). No checkpoint or pinned reference mathematics changed. No new optimization below was made the default.

## Per-patch ledger

All patches preserve canonical packed sources and complete routing. Numerical validation is scoped, not a claim of new independent full-model llama.cpp equivalence.

| Patch | Objective / principal files | Interface and architectural choice | Validation / measured disposition | Next dependency |
|---|---|---|---|---|
| 0295 | Routing heat: `expert_heat_policy.py`, native handoff/profile statistics, `analyze_0295_route_heat.py` | Optional bounded route counters; pure causal equal-stride proposals, hysteresis and movement cap. No transfers or residency mutation | 9 policy tests; 28 profile tests. Causal fixed-trajectory proposals save <0.7% CPU misses at up to ~3 GB transfers. **Numerically validated policy; mover not integrated** | Re-evaluate on natural conversations; do not apply equal-stride proposals to sized slots |
| 0296 | State/copy ownership: `inference_graph.py`, `qwen4exp_pre_moe_graph.py`, model memory ledger, budget tests | `bound_mutable` retains exact request buffers, recaptures on pointer-owner changes; `MINI_LLM_GRAPH_DEVICE_COPY=1` uses bit-preserving GPU copies | 25 GPU tests/1 skip; actual checkpoint logits/state exact at 1/2/4/8/16 over two requests. Bound-state model 38.56/s; device-copy 38.65/s, small difference. **Numerically validated in tested state scope** | Reduce remaining critical path, not copied state equations |
| 0297 | IQ3_S CPU gather: `iq3_cpu_simd.h`, CPU bridge/executor, operator probe/tests | `MINI_LLM_GGML_IQ3_GATHER=1`; same GGML Q8_K producer, integer sums, FMA and reduction order | Exact independent pinned-GGML tests; actual top-10 operator ~1.19–1.20x. No isolated whole-model gain established. Sized 7.25 GiB cache plus deferred QSA result 42.32847/s | IQ2_S/IQ3_XXS are larger measured CPU contributors |
| 0298 | CPU/GPU attribution: native timeline, `analyze_0285_critical_path.py`, profiler probe/tests | Diagnostic 2048-record ring; schema names distinguish old 5-column and new 9-column captures; align monotonic brackets, exclude wait kernels from useful GPU work | 29 phase tests. Verified 768 CPU batches / 768 GPU publish/wait pairs. **Implemented and validated instrumentation**, not an optimization | Target actual CPU work; publication was already prompt |
| 0299 | IQ4_NL CPU VNNI: `iq4_cpu_vnni.h`, bridge/tests | `MINI_LLM_GGML_IQ4_VNNI=1`; unchanged Q8_0 and paired-block accumulation | 37 combined CPU tests; real operators ~0.95–1.03x. **Numerically validated, no useful speedup; off**. 23-worker model 41.80778/s vs 19-worker 42.32847/s | Do not add workers or enable VNNI on assumption |
| 0300 | IQ3_XXS CPU grid gather: `iq3xxs_cpu_simd.h`, bridge, tests/probe | `MINI_LLM_GGML_IQ3XXS_GATHER=1`; Q8_K unchanged, final quarter scaling after hsum retained | Initial grid+sign gathers slower; localized scalar-sign variant ~0.89–1.03x on real experts. Exact uint32 outputs. 15 workers 41.55957/s. **Numerically validated, no gain; off** | Avoid speculative full-model run of losing operator |
| 0301 | Parallel HC stream projections: `hyperconnection_read_gpu.py`, tests/probe | `MINI_LLM_HC_PARALLEL_READ=1`; parallel per-stream up projections and warp completion of the same reduction tree | 26 focused GPU tests, exact bit patterns and graph replay. ~18.46 us/operator for both variants, no material gain. **Numerically validated, not performance optimized; off** | HC bandwidth rather than barrier count limits this fixture |
| 0302 | Reproducible settings and trace/telemetry: `probe_0302_decode_configuration.py` | Isolated caller required; explicit flags, workers/cache/task factor; optional trace and bounded telemetry; natural/replay distinction | Multiple successful long runs. QSA frontend single observation 42.64032/s. Unsupported sized-slot frequency-placement trial failed before response; removed unsupported wrapper option, no production bypass. **Implemented diagnostic runner** | Repeated matched measurements, not a flag stack |
| 0303 | Paired IQ3_XXS gate/up: `iq3xxs_cpu_pair.h`, bridge and 0300 tests | `MINI_LLM_GGML_IQ3XXS_PAIR=1`; shares Q8_K loads, independent exact row arithmetic; incompatible with gather flag | 67 combined CPU tests. Real operators ~0.93–0.99x; not a gain. **Numerically validated, off** | Fresh pinned Strata source confirms default singleton IQ2_S/IQ3_XXS uses GGML dots; multi-token gains cannot be assumed at one token |
| 0304 | Native task granularity: row pool, `probe_0304_task_grain.py`, tests | `tasks_per_worker` / `MINI_LLM_GGML_POOL_TASK_FACTOR`, 1..16; default remains 3. Native setter rejects busy pools and invalid values | 83 combined CPU tests. Small operator gains at larger factors, but factor 8 long model 38.02620/s. **Numerically validated; no default change** | Whole-model scheduling/overlap, not operator-only grain tuning |
| 0305 | Capture shared expert inside native MoE graph: model, device plan, handoff, executor, tests | Optional `gpu_branch` with pure `enqueue_gpu_branch(x)` and `combine_gpu_branch(routed, values)`. Branch owner latched per graph. CPU input published before shared work. `MINI_LLM_NATIVE_MOE_FUSED_SHARED=1` | 37 focused GPU tests. Actual combined state/logit parity below. Fused 37.41464/s vs adjacent control 37.29292/s: no material gain. **Numerically validated in tested scopes, off** | Profile scheduling rather than assume fewer launches imply rate gain |
| 0306 | GPU input publication: native service/handoff, tests | `MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH=1`; bit-preserving stores into existing pinned slabs, every writer system-fences before release publication. Requires native service; no new weight copy | 40 focused GPU tests including no-runtime-DMA trap, dynamic routes, arbitrary input bits and fail-closed publication. Trace scheduling gap reduced; long combined 37.47011/s remains neutral. **Numerically validated in tested scopes, off** | Understand profiler versus uninstrumented behavior |
| 0307 | Shared-memory IQ lookup: `mixed_q8_gpu.py`, tests and actual expert probe | `MINI_LLM_Q8_SHARED_GRID=1`; original codebooks uploaded per CTA, aligned shared 32-bit lookup, unchanged integer/scaling/reduction. Uniform inactive-expert blocks skip unused upload | Final real operators **1.085–1.857x**, bit-exact, six actual format triples at 2/5/10 hits. 117 initial GPU tests; final regressions below. Adjacent long 40.05676 -> 40.17001/s: **operator performance optimized, model gain not established; off** | CPU completion and natural-trajectory page faults now require attribution |

All controls remain available. Experimental switches that are neutral or slower are not silently enabled by model construction.

## New interface contracts and ownership

### Task factor

`GGMLPackedCPURowPool(..., tasks_per_worker=None)` resolves an integer 1..16, default 3. It controls only task chunk sizing. It does not change row order within a dot, Q8 preparation, hidden quantization, selected experts or final routing-weight application. `mylocal_ggml_pool_set_task_factor` is an idle setup seam and rejects a live batch. It is not a per-token scheduler knob.

### Captured GPU branch

`try_forward_device_plan(..., gpu_branch=owner)` can decline without executing the branch when native service/capture prerequisites are absent. An accepted owner supplies pure GPU enqueue/combine operations. The branch is owned by the graph; switching owner on a live graph is rejected. CPU output remains unweighted and the routed result is weighted exactly once before shared combination. Unsupported placement and non-frozen admission retain the existing path. Graph retirement fences readers and explicitly releases the branch owner before native handles/sources retire.

The Qwen adapter supplies the existing shared gate/up/SwiGLU/down/score and stable-sigmoid combine, not alternative equations. Shared work executes after CPU publication on the native graph's stream. No CUDA/Python call is introduced into the native service thread.

### Kernel publication

Activation and metadata aliases retain the original pinned owners. A single CTA copies their original bits; each writer executes `__threadfence_system`, the CTA synchronizes, then lane zero publishes the epoch with system-release semantics. The CPU's existing acquire and complete-output publication remain unchanged. A failure prevents new publication, preserves bounded waits and prohibits incomplete CPU payload reads. No per-token allocation or packed-weight DMA is introduced.

### Shared IQ codebooks

The original packed matrices remain authoritative. A module-cache key includes shared/global mode, so kernels with different storage do not alias. Shared tables cost at most 8 KiB per CTA and no persistent model-sized allocation. Initialization happens before any potentially divergent return; the inactive-block shortcut is CTA-uniform and used only with four-row-aligned expert geometry. Mixed gate/up formats fall back to the correct table rather than reusing an incompatible grid.

## Critical-path evidence

### CPU is doing real expert work

`0298_overlap_alignment_v1.json` verifies 768 batch associations, 48 per token:

- publication to CPU start: median ~0.6445 us/batch;
- native CPU duration: 10.574 ms/token in the instrumented window;
- useful GPU overlap before join: 2.511 ms/token;
- GPU wait: 8.469 ms/token;
- CPU completion to wait retirement: median ~1.852 us.

`diagnostics_0298_phases_v1` measures ~8.40 ms native CPU per model step without the GPU profiler: gate/up 5.10, down 2.71, hidden quantization 0.20, dispatch/input 0.13 and completion 0.26 ms. IQ2_S and IQ3_XXS gate/up are the largest groups. Increasing worker count and replacing singleton dots did not materially reduce these costs.

Fresh read of pinned Strata `src/kernels/cpu/native_expert.cpp` at `712fec6b4297c9cfd32ab43daaa2979a9aa01edf` confirms: default multi-token threshold is 2; default singleton IQ2_S/IQ3_XXS falls through to linked GGML dots; window kernels decode weights once across multiple token activations. IQ3_S gather-at-one is separately opt-in and ISA-dependent. That does not supply a missing universally faster singleton kernel on this AVX2-only CPU. Source was fetched for inspection; no new saved-source hash is fabricated.

### Copies and profiler effects

`0305_critical_path_v2.json` (7 GiB cache, fused shared branch) records 546.309 ms GPU span / 334.172 ms operation union across 16 tokens; 135.024 ms gaps immediately precede publication after input DMA nodes. `0306_critical_path_v1.json` at the same 7 GiB cache and fused branch plus kernel publication records 413.325 ms span / 342.340 ms union. The large gap disappears, while useful non-wait work is 219.662 -> 231.489 ms and GPU waiting 114.510 -> 110.851 ms.

These are instrumented observations, not an uninstrumented 1.32x model gain. Complete long runs did not reproduce a material benefit. The first trace terminated at capture completion (exit 143); its report is preserved and no completion/rate is inferred. The second uses `--capture-range-end=stop` and completes normally. A 7.25 GiB profiler trial failed the physical VRAM check; the trace used 7 GiB rather than weakening headroom.

The small final status D2H still includes waiting for queued model completion. Its API duration is not exclusive data-transfer latency and its validation must not be removed.

## Long measurements and natural-generation caveat

| Retained artifact | Mode / change | Forwards/s |
|---|---|---:|
| `diagnostics_0297_deferred_v1` | Controlled earlier best configuration | 42.32847 |
| `diagnostics_0299_workers23_v1` | Same, 23 workers | 41.80778 |
| `diagnostics_0300_workers15_v1` | Same, 15 workers | 41.55957 |
| `diagnostics_0302_qsa_v1` | Controlled QSA frontend | 42.64032 |
| `diagnostics_0304_factor8_v1` | Controlled, task factor 8 | 38.02620 |
| `diagnostics_0305_control_v1` | Recent control, factor 3 | 37.29292 |
| `diagnostics_0305_fused_v1` | Fused shared graph | 37.41464 |
| `diagnostics_0306_combined_v1` | Fused shared + kernel publication | 37.47011 |
| `diagnostics_0307_long_v1` | Shared grid, controlled | 39.91715 |
| `diagnostics_0307_control_v2` | Adjacent repeated control | 40.05676 |
| `diagnostics_0307_long_v2` | Adjacent repeated shared grid | 40.17001 |
| `diagnostics_0307_combined_v1` | Shared grid + QSA frontend + fused shared + kernel publication | 39.92709 |
| `diagnostics_0307_natural_v1` | Natural, 371 emitted / 370 forwards | 31.95859 |
| `diagnostics_0307_natural_v2` | Same settings, exactly the same natural output IDs | 35.15922 |

Natural response intervals (including prompt prefill) read **3,048,243,200 / 1,854,091,264 bytes** and incur **39,383 / 26,760 major faults** respectively. The second run is warmer, not fully hot. Controlled adjacent runs incur essentially no disk reads/major faults. Natural routing/expert counts differ from the retained forced trajectory, so the replay rate must not be presented as actual conversation throughput. Neither disk counters nor native spans isolate PLE versus expert-bank I/O yet. Larger anonymous/pinned residency is not justified without a bounded page-source experiment.

## Numerical and regression verification

- `0307_cpu_regression_v1.xml`: **471 passed, 43 skipped**.
- `0307_gpu_regression_v1.xml`: **201 passed, 4 skipped**, including Qwen3-Next mixer, attention, decoder and model.
- `0307_parity_comparison_v1.json`, repeated on final source in `0307_parity_comparison_v2.json`: actual checkpoint fixed IDs 1..16, two independent requests, 10 logit boundaries at prefixes 1/2/4/8/16. All logits bit-exact (maxabs 0), every persistent-state hash and reset-request comparison identical to retained uncaptured control.
- The full-model parity fixture freezes empty GPU expert placement: it validates dense projections, state, native CPU experts, fused shared branch and publication. Resident packed GPU arithmetic is independently checked against extracted Strata rows and real checkpoint operators; do not claim the CPU-placement fixture validates every resident GPU route.
- No new independent llama.cpp full-model numerical-equivalence claim, no relaxed tolerance and no training-suite claim. All original result files, initial indentation-error outputs and losing experiments are retained.

## Memory and safety

All model/CPU/GPU experiments execute sequentially under `run_0290_isolated_benchmark.py`, with verified separate app.slice cgroups, swap 0, finite high/hard limits and 12 GiB host reserve. Typical model limits remain 32/36 GiB. No bulk 50 GiB warming, full expert pinning, cache dropping, global power tuning or checkpoint changes occurred.

The collected 0300–0307 guard sample peaks are at most ~5.42 GiB charged memory in this warm shared-page environment, minimum host-available ~47.0 GiB, no OOM kills. These charged values do not equal process RSS: mapped shared file RSS is around 36–38 GiB, anonymous RSS <1 GiB, and pages may be charged to older/ancestor cgroups. Consult each guard for PSI and event details; isolation does not prove immunity to unrelated system or driver failures.

Latest CuPy pool used ~11.49 GiB with 7.25 GiB expert arena, not a device-wide high-water measurement. Sampled trace NVML reaches roughly 14 GiB including driver/display/profiler. No headroom checks were reduced. Exact budget/peak/IO fields are in the structured report.

## Next evidence-supported dependency

Do **not** declare performance optimization complete. The most useful next experiment is per-source attribution of the natural request's major faults/read bytes, followed by a small bounded causal host-residency/read-ahead experiment on those identified pages. Simultaneously preserve a hot controlled replay for CPU kernel/overlap work. This separates storage stalls from IQ2_S/IQ3_XXS compute instead of treating every wait as PCIe or assuming that loading an arbitrary RAM subset solves it.

Whole-token/window execution and correctly verified multi-token expert reuse remain architectural opportunities, but require dynamic QSA/PLE/state and numerical gates. Historical Strata speculation must not be substituted silently for full-top10 single-forward speed.
