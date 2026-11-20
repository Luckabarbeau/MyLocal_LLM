# Qwen4Exp performance and memory history

## Current — 0310–0317 fresh Strata comparison and bounded experiments

**50 complete decode forwards/s remains unmet.** Fresh Strata full-top10/no-draft clean repeats measure36.48/49.26 forwards/s (profiled32.04/47.63); hot second response is not the two-response sustained rate. Effective context1024 and retained prompt/continuation IDs match the local diagnostic, but INT8 versus FP32 KV, cache7.233 versus7GiB and expert placement remain distinct. No unchanged full `standard-v1`/262144-context equivalence claim.

| Local controlled experiment | Response1 / response2 forwards/s | Decision |
|---|---:|---|
| PLE control0312 | 39.81 / 39.79 | Baseline |
| Bounded batched PLE0312 | 39.44 / 40.19 | Cold fault reduction, no hot gain |
| Complete recurrent block0313 | 38.24 / 41.04 | Exact within-path parity; opt-in |
| Cooperative FP32 QSA0314 | 39.51 / 41.70 | Operator2.08–12.46x; model quality gate open |
| Recurrent chain0315 | 39.18 / 39.95 | No demonstrated hot gain |
| Complete QSA block0317 | 41.44 / 39.94 | Runtime positions/state exact; no demonstrated hot gain |

All responses emit359 forced IDs/358post-prefill forwards, after128output-token warmup, with full head/sampling cost. These are not natural-conversation rates. PLE diagnosis measured2.625ms/step and4282embedding-thread major faults; bounded reader measured0.526ms first/0.115ms repeat, zero embedding-thread major faults. This does not isolate every expert/storage stall.

CPU observation tested rather than inferred: 50us spin gives~9.52active-core equivalents/41.04hot forwards/s; 2000us gives18.21cores/34.88s. Strata's pinned pool source defaults20ms spin, not our50us. Captured19-worker affinity+500us gives16.50s; creating the native service before submitter pinning fixes the inherited-one-core setup mismatch but still gives21.15s, so neither affinity/spin setting is promoted. Higher CPU load is not equivalent to higher throughput.

0313's16-token trace:443.334msGPUspan,215.746msnon-wait operationunion,137.167mssummedwaitkernels,62graphlaunches/token. Categories overlap and gaps are not an exclusive host-overhead budget. Complete-block/chain experiments do not justify treating every launch or gap as removable critical latency.

Final selected regressions407CPU/284CuPupassed, skips scoped in `0317_final_{cpu,gpu}_v1.xml`. Full patch/provenance/guard/source ledger: `0310_0317_strata_comparison.md`, `0310_0317_results_v2.json`. All old results preserved; experimental switches remain off.

## Prior — 0308–0309 page residency and RAM-source negative result

Complete1224tensorpagecensus performs zero payload disk reads during the census, finds~14.88GBcanonical routed-expert payload nonresident; a26.82GiBPLEtable remains mostly disk-backed.10diagnostictests pass. This is a snapshot, not exact timed-decode fault attribution or CPU-complement residency.

Bounded4GiBanonymoushostsource/2111bundles uses no pinning or GPUstaging, retains identical371naturalIDs and measures32.51596forwards/s, **not an improvement** over earlier35.15922/snatural repeat. Setup1.9063s is outside decode; its I/O is included in the broader interval counters and must not be mislabeled decode-only. AnonymousRSS~4.81GiB, chargedservicepeak~8.56GiB, zeroOOMkills. No automatic budget growth or default change. Report `0308_0309_storage_residency.md`; latest results/sourceSHA256s `0295_0309_optimization_results_v1.json`. **50/s remains unmet.**

## Prior — 0293–0307; gains scoped to measured critical paths

**50/s remains unmet.** Earlier best controlled observation42.64032/s; latest adjacent control/shared-grid40.05676/40.17001forwards/s (358forwards, same359response IDs), not a robust gain. Real packed GPU shared-grid operators improve1.085–1.857x bit-exactly, but CPU completion hides much of that benefit. 15/23workers, larger task factors, IQ3_XXS gather/pair, IQ4VNNI and parallel HC did not establish full-model gains; none was silently promoted.

Native shared-branch capture and system-fenced kernel input publication reduce launch/copy-node structure. A16token traced span546.31 ->413.32ms at7GiBcache is **not** an uninstrumented throughput gain: longer fused/publication37.47/s remained neutral versus recent37.29/scontrol; the combined shared-grid/QSA/fused/publication trial39.93/s is not50/s. Failed/partial traces and VRAM admission rejections remain preserved.

Natural sampling emits exactly the same371IDs in two new runs: **31.95859 /35.15922forwards/s**, versus the largely hot forced replay. Response+prefill disk reads3.048/1.854GB and majorfaults39383/26760 show a storage-residency contributor still present on the second run. Different natural versus replay routes/token counts are not an identical-input speed ratio. Existing Strata suffix/tail/KV/context differences remain explicit.

Verification:471CPU passed43skipped;201GPU passed4skipped. New actual-model local-arithmetic two-request logits/state hashes remain exact. Detailed per-patch ledger, critical-path attribution, command and memory scopes: `0295_0307_critical_path_optimization.md`; structured observations/sourceSHA256s: `0295_0307_optimization_results_v2.json`. No full training-suite or new independent llama.cpp model-equivalence claim.

## Current — 0291–0292 bounded post-incident measurements

Warm fixed-token decode **33.87091/s**;512MiB partial CPU-source **33.90548/s** (0.10%, no material gain). Same128warmup/359response tokens,358decodeforwards,19nativeworkers+participatingwaiter,7GiBfrozenGPUcache, testedQ8/BF16HC, no bulk mmap warming. Cold baseline21.29501/s incurred12.70GBresponse+prefilldiskreads and must not be used to advertise source-copy acceleration. DeferredQSAcheck34.01985/s is only0.44% above control; no demonstrated improvement/default change.

New16tokennode-leveltimeline:GPUspan514.771ms/operationunion210.120ms;177.496msgaps precede CPU-result uploads totaling2.885ms. Standalone complete-top10split across56configurations shows selective1–2miss staging~5–20% operator gains but cheap/low-miss and blanket staging often lose; no integrated model speedup inferred. Native mover remains next dependency, not a50/s guarantee.

All runs isolated under0290hard/highlimits and12GiBreserve. CPU copy/storage exact outputs;56mixedCPU/GPUsame-siteoraclecases exact. Focused96CPU passed4skipped1deselected;20GPU passed1skipped. Report/artifacts: `0291_0292_residency_and_miss_split.md`, `0291_0292_benchmark_summary_v1.json`. **50/s unmet; no production mathematical changes.**

## Prior — 0290 safety pause; no new throughput result

A desktop OOM kill was confirmed on2026-11-19: systemd-oomd selected the50.6G GNOME Shell cgroup. The exact offending process is unresolved. Large-memory copies/pinning/GPU/full-model experiments are paused;0289's resident complement is only small-fixture validated, with no measured performance benefit.

0290 adds fail-closed app.slice service execution, verified hard/high/swap/group-OOM limits,12GiB reserve, PSI/ancestor monitoring and cleanup. Final CPU/no-model regression89passed4skipped1deselected; kernel cgroup peak643858432bytes, sampled PSI0. Safe timeout test aborted as expected. No inference speed or numerical result is inferred from these tests. **Target50/s remains unmet.** Report: `0290_memory_incident_and_isolation.md`.

## Prior — 0281–0288 mixed Q8 and bounded execution

**Best retained safe callback observation31.42545 decode forwards/s;50/s unmet.** Final callback/service reproductions29.29/29.56/s. Full report: `0281_0288_execution_optimization.md`; retained measurements/source hashes/parity: `0281_0288_performance_v3.json`.

| Configuration | Response tokens | Decode forwards/s |
|---|---:|---:|
|Expert+dense GPU Q8,23workers/500us|415|23.17058|
|Same Q8,19workers/50us|415|28.61418|
|Exact BF16 HC storage,6GiB expert cache|415|30.62288|
|BF16 HC,7GiB cache|359|31.42545|
|Initial service (historical timeout path later corrected)|359|31.71163|
|Intermediate safe mapped-status copy|359|19.60218|
|Final safe device-status service copy|359|29.55954|
|Sized fixed slots, natural sampling|351|29.96722|
|Uniform slots, controlled359-token trajectory|359|30.19602|
|Sized slots, same controlled trajectory|359|31.14912|
|Final natural7GiB reproduction, same359 IDs|359|29.29410|

Final natural reproduction29.29/s shows the29–32/s run-to-run range; do not advertise the small native-service difference as robust. An8GiB-cache probe failed safely at physical headroom validation; no result is inferred and no guard weakened. Full head/sampler remain in controlled replay; it is not natural generation. One run/configuration, not confidence intervals. No isolated speedup ratio across415/359/351-token trajectories. Hidden-phase fusion, cumulative warmup placement and extra graph flags show no useful measured gain and remain off. Sized slots increase capacity2823 ->3673 within7GiB but only modestly improve controlled throughput.

Q8 operator gains2.12–3.53x for resident top10 are not model gains. Node-level Q8/BF16 trace:214.856ms kernel sum/217.214ms GPU-operation union/602.011ms GPU span across16 tokens. Remaining gaps include CPU-result joins and host submission; they are not removable memcpy arithmetic. Direct mapped PCIe dots lose to the19-worker CPU pool; explicit staging is much better but integrated CPU/PCIe overlap is not yet measured. Native service31.71 versus31.43/s is too small to advertise as a meaningful gain. See all failed/neutral experiments in the numbered report.

Final selected regressions1969CPU passed/1124skipped;1367CuPy passed/14skipped. BF16 HC and native-service fixed-prefix logits/state hashes are exact against same-arithmetic controls, not new independent llama.cpp equivalence. Strata speculation/tail/KV/context differences remain explicit.

## Prior follow-up —0280 validation-dependency experiment

Best remains26.95159decode forwards/s. Deferred QSA finite validation with conditional GPU cache writes retains all checks and passes exact same-arithmetic logits/state parity, but same413-token longer workload **24.84134/s** shows no gain. Sampling-call wait decreased while total decode did not: do not mistake shifted wait attribution for eliminated critical work. Node-level graph capture reveals380.77mssummedGPUkernels/363.54msGPUoperationunion in635.60ms/16tokens; dominant copy waits are4-byte QSA status readbacks. Times overlap, profiling overhead applies. See `0280_qsa_validation_dependencies.md` and `0279_node_copy_attribution_v2.json`.

Latest full selected inference regressions1939CPU passed/1025skipped,1211actualCuPy passed/18skipped. Current source hashes/measurements: `0268_0280_performance_v3.json`. Integrated mixed-format GPUQ8 and nativeCPU/PCIesplit work remain pending;50/s not reached. No model equations/checkpoint/reference changed.

## Prior — 0268–0279 host/GPU execution pipeline

The 0268 block/warp oracle mismatch is resolved; same-warp outputs are bit-exact. Current best longer run **26.95159 decode forwards/s**, not 50/s. Complete benchmark records/source hashes: `0268_0279_performance_v2.json`; contracts/limits: `0269_0273_execution_pipeline.md` and `0274_0279_native_handoff.md`.

| Configuration | Response tokens | Decode forwards/s |
|---|---:|---:|
|0268 persistent device plan|512|17.57999|
|0269 unpinned host participation|512|17.72739|
|0270 pre-MoE recurrent graph|512|17.88268|
|0271 registration, old barrier scheduler|512|8.49598|
|0272 task-completion scheduler|512|18.96007|
|0273 shared expert graph|413|18.88463|
|0274 deferred shared launch|413|19.31426|
|0275 native asynchronous CPU handoff|413|24.29482|
|0276 captured native handoff|413|25.86397|
|0277 specialized resident kernels|413|24.62057|
|0277 23 workers + participating callback|413|**26.95159**|
|0278 QSA frontend capture|413|26.12946|
|0279 input-only graph operands|413|25.41335|

0273–0279 have identical output IDs. No ratio across the 512/413-token trajectories is an identical-input speedup. Neutral/regressing candidates remain opt-in. One sequential run per configuration, not randomized confidence intervals. CPU format-specific Q8_K/Q8_0 and packed GPU FP32 activation arithmetic are retained.

0275 sampled RssAnon 852914176 / RssFile 55074443264 bytes; default GPU pool used/reserved 12157392896 / 12345454080 bytes. These are category-specific sampled maxima/snapshots, not device-wide peak VRAM. Graph private pools are separately bounded/accounted (native graphs ~1.52 MB total; QSA frontend ~2.49 MB). Model planning now reserves their caps and credits only owned pools. No full packed/dequantized expert duplication or full PLE allocation.

Final tested source: **1939 CPU passed / 1016 skipped; 1202 actual GPU passed / 18 skipped**. Fixed-prefix same-arithmetic parity remains exact for new handoff/graph scopes. Latest 15-worker captured-handoff trace ~588 ms/16 tokens contains ~182 ms runtime memcpy API time; graph-node kernels were not exported in detail, so visible kernel totals are incomplete. Those times overlap CPU/native/GPU dependencies. Next measurement requires node-level graph tracing and call-site correlation. A mapped-host cuStreamWaitValue32 probe returned error 801, not a usable completion path.

Strata emitted-token throughput still includes differing suffix/tail/KV/context/host-placement behavior; no exact full-configuration equivalence claim. **50/s remains unmet.**

## Historical — recurrent graphs, critical-path measurement and worker latency (0265–0268)

0265captures the complete recurrent segment with Triton FP32 dense projections. Same-arithmetic control/capture logits and all active-state hashes are exact at1/2/4/8/16 over two requests. Long273-token response **12.60665decode forwards/s**; previous cuBLAS path sampled a different trajectory, so not an identical-input ratio.

0266current-path Nsight16-token window1.48257s: summedkernels396.473ms, GPUoperationunion405.990ms, CPUcompletionwait321.591ms,41,969driverkernel launches. Parent spans/CPU/GPU/API times overlap; no exclusive-critical-path claim. Existingmixedgrouping retest12.55698/s showed no demonstrated gain.

0267bounded500us worker spin reduces synthetic5-expert latency360.011->175.846us, and same273-token longresponse272forwards/17.88016s = **15.21239forwards/s**, versus12.60665sleeping. Both warmup and response IDs match. CPU utilization/power tradeoff, defaultsleep retained; one pair, not randomized confidence bounds. See numbered reports and `0265_0267_performance_v1.json`.

0268persistentdeviceplan is **unvalidated**: latest94passed1skipped2failed; tiny GPU differences under an exact assertion, possible block-vs-warp oracle mismatch. No fullmodel0268run, no speed claim, no default enablement. Last measured checkpoint state precedes these experimental source edits.50t/s remains unmet; next experiment is the explicit same-warp oracle comparison documented in `0268_device_plan_blocker.md`.

## Historical — 0260–0264 structural pipeline experiments

SamecapturedStrata75/77prompts,128/512caps andsamplingsettings,15CPUworkers,6GiBcache, mappedsourcewarming andboundedserialprocesses:

| Candidate | Response tokens | Decode forwards/s | Interpretation |
|---|---:|---:|---|
|0259pooladaptive|427|8.72394|Prior meaningfullongresult|
|0260frozenplacement|324|10.31149|Zeroresponsepromotions; changedCPU/GPUarithmeticplacement/trajectory|
|0260bounded1promotion/480requests|352|10.29833|Movementcapped, notlargegain|
|0261frozen+Q2SIMD|324|10.72100|SameIDs asfrozen; exactoperatorcontract|
|0262+sharedoverlap|324|10.79765|SameIDs; no convincingisolatedgain|
|0264+canonicalhandlecache|324|**11.42412**|SameIDs;323forwards/28.27350s|

These are separate sampledprocesses, not randomizedtrials; do not ratio427/352/324token differenttrajectories as identical-input speedups.0261Q2downoperator4.138x andcompleteexpert1.865x are syntheticwarmratios, notmodelgains.0263hostprofile includes77prefill+323decode andadds instrumentationoverhead; not a throughput result orGPUcriticalpathtimeline.

0264sampledRssAnon829,919,232/RssFile55,192,346,624bytes, separatecategorymaxima; GPUallocator12,148,603,904used/12,335,560,704reservedbytes, notglobalVRAMpeak. Allnormalbenchmarksupervisorscompleted0. See numbered0260–0264reports and `0260_0264_pipeline_comparison_v1.json`.

**50t/s remainsunmet.** Eliminatingresponsepromotiontraffic wasnot sufficient. Nextlargecandidate is validatednativeexecutionplans/capturedGPUsegments andpersistentdynamicrouting/statebuffers; remainingall-formatGPUQ8/profileplacement/asyncmover/efficientprefill/INT8KV/suffixspeculation are not claimedimplemented. Numericalreferences/defaults retained except semantics-preservingcanonicalmetadatareuse.

## Historical — 0259 persistent CPU row pool

Pinned Strata audit0258 led to opt-in native row-phase scheduling, sharedQ8 input and bounded scratch. Operator outputs are bit-exact against retained per-expert GGML; model equations unchanged. Synthetic15-worker warm operator ratios old/new:0.678x for1expert (regression),2.658x for3,3.112x for10. Not model throughput.

Captured longer workload, exact75/77prompt IDs and128/512caps, both actual output streams identical128/427tokens: near-fully-source-resident warmed control**8.20971** versus pool**8.72394decode forwards/s**, observed**6.26%** higher throughput;426forwards51.88974s versus48.83114s. Both supervisors completed0; one sequential pair, not randomized repeated trials. Earlier cold control7.55953 and failed parent exit-race are preserved as confounded, not the scheduler gain. Full Strata configuration parity remains false. See `0259_native_row_pool.md` and `0259_row_pool_comparison_v1.json`.

Pool sampledRssAnon824,041,472bytes/RssFile55,226,150,912bytes (separate sampled maxima); GPUallocator12,148,603,904used/12,334,209,024reservedbytes, not globalVRAM peak. Same2,419slot arena. Responsecase including77-tokenprefill adds46,591promotions/94,913,766,400transferbytes; measured synchronous churn motivates bounded placement/mover work. No exclusive critical-path proof from byte counts.50t/s remains unmet.

## Historical — longer captured Strata workload (0254–0257)

Exact prompt-token agreement75/77, same captured sampling settings and128/512 caps; CPU policy/precision/residency changes evaluated on generated text rather than16forced IDs. Original sort/native/no-warm run timed out180s before warmup. Heap/native/warming warmup1.77393forwards/s; combined heap+GGMLQ8 CPU+15workers completed128 warmup and427-token response: **426forwards/53.00188s=8.03745forwards/s**, emitted427/decode_seconds8.05632/s; supervisor completed. See `diagnostics_0256_long_v1/`. This supersedes any interpretation of20.5t/s as sustained chat speed, not the historical trace itself.

Mixed grouping candidate child completed352-token response at7.83378forwards/s; no demonstrated improvement, keep opt-in. Full-warm128-token child7.71648/s versus prior mixed warmup7.71750/s, showing incomplete host warming alone does not explain steady latency. These parent runs failed a /proc/io exit race despite successful child results; supervisor not validated. CPU profile indicates major future-wait and synchronous promotion costs but not an exclusive timeline proof.50t/s remains unmet.

0252byte-exact GPUQ8_1 quantizer and0253IQ3_S integerGEMV independently pass41/38 tests, respectively; neither establishes all-format model GPUQ8 integration. GGMLCPU quantization is per-formatQ8_K/Q8_0, not Q8_1.

## Historical — 0251 integrated elementwise fusion

Controlled real-checkpoint same-process warm A/B, native HC constant, IDs1..16 full top10 FP32: prior18.02873/17.97284 versus fused20.53097/20.23955 forced tokens/s (~13.25% higher mean rate). Final logits and active-state hashes bit-exact. Sampled peakRSS~9.59GiB, fixed cache/device budgets. Not a prolonged conversation or50t/s result. See `0251_elementwise_fusion.md` and `diagnostics_0251_ab_v1/`.

0252 native Q8_1 quantizer candidate is not integrated or performance validated. Independent oracle fixture compilation is blocked after a corrective attempt; latest19 passed22 failed. Precision changes remain gated on exact-format byte validation, then packed-dot comparisons.

## Historical — 0247–0251

Warm Nsight:465.503ms summed GPU kernels across16 tokens (~29.1ms/token) versus~1.20s wall, with1,920 D2H operations, including1,152 QSA finite checks. See0247 report and guard caveat.

0248 controlled same-process validation batching: old14.1985/14.3450 versus batched15.6075/15.8511 forced tokens/s; logits/state exact.0249 compile specialization:1.40–1.97x resident real-expert operator speedups, but whole-model~16.25–16.47 unchanged materially; opt-in only.

0250 three-launch HC: synthetic read~0.060ms to~0.021ms; real native-path warm17.8544/17.7630/17.8277 forced tokens/s (~9.6GiB sampled RSS). Final logits relativeL2=6.52e-7 against prior FP32 path, same argmax/top10; repeated native-path states exact. No precision reduction. HC mode remains opt-in; no full model A/B speed ratio from separate processes.

0251 has no performance result: test suite39 passed1 failed at an unsupported `.options` test attribute after permitted correction/retest, candidate not integrated. Goal50t/s remains unmet; numerical gates and the stop rule take precedence over throughput.

## 0246 grouped validation cleanup

Fresh bounded real-checkpoint full-top10 forced-token runs: pre-edit4.91794, profiled pre-edit8.54591, post-edit11.15820 tok/s. Prefix1/2/4/8/16 logits and state hashes bit-exact; ~9.6GiB sampled peak RSS. Cold/warm/runtime variability prevents attributing these differences to the edit. Six warmed same-process packed-operator pairs show **no convincing throughput benefit**; this removes a redundant validation scan only. GPU tests286 passed,2 skipped. Goal50 tok/s remains unmet. See `0246_grouped_validation.md` and preserved versioned artifacts.

## 0245 unused QSA query elimination

RTX 5080, synthetic FP32 production-dimension QSA, 128 positions, four interleaved pairs: removal of unused index-query projection/RMSNorm/RoPE saved 0.025–0.169 ms per position; final outputs bit-exact. Existing fast path only, no precision change. GPU pool reservation ~407 MiB; not a high-water measure. Eight CuPy tests pass; CPU QSA/model/Qwen3-Next regressions: 101 passed, 1 skipped. Not a packed/full-model throughput measurement. See `0245_qsa_dead_query.md` and `0245_qsa_dead_query_v1.json`.

0243 Strata baseline: short/4K/16K median emitted-token rates 50.63/51.31/55.28 tok/s, model drafter off but suffix lookup and route-tail omission active; 43.12 GiB pinned host experts and 5.31 GiB GPU expert cache under an 8 GiB RAM headroom preference. Standard suite stopped at its time cap after all nine performance cases. See `0243_strata_benchmark_results.md`. These rates are not comparable to unassisted full-top-10 single-forward benchmarks.

## Historical entries

Short-context full-model lifecycle smoke timings are now recorded below; no dedicated steady-state prefill/decode benchmark or optimized throughput claim has been established.

| Patch / workload | Time / memory | Interpretation |
|---|---|---|
|0200 initial header audit/export | ~5.24s before serialization; ~889MiB peak RSS | Metadata/tokenizer parsing; no weight payload reads/dequantization |
|0201 actual two-shard header indexing |5.447s; ~244MiB settled RSS, ~884MiB transient peak |1224 descriptors and full tokenizer metadata; instrumented reads stayed before payload regions |
|0201 synthetic store tests + existing row tests |46 passed in0.19s;57.4MiB peak RSS | Small fixtures, including sparse offsets beyond4GiB; not model inference |
|0202 CPU reference blocks/GEMV | Correctness only | No benchmark or speedup asserted |
|0203 native packed experts | Correctness only | Direct packed execution avoids dense matrices; no measured throughput comparison |
|0204 sparse CPU/GPU capability tests | Correctness only | Finite fixtures, residency and fallback semantics; no overlap-efficiency or throughput claim |

Audit hardware query: RTX5080,16303MiB total VRAM,14615MiB free at sampling; physical RAM66,843,660,288 bytes, available57,824,903,168 bytes;8GiB swap present and already partly used. Measurements are snapshots, not reserved capacity. Weight files were not downloaded/modified.

## Original-checkpoint lifecycle smoke

Both runs use max_length8, inputIDs[1,2], four CPU workers, two independent requests and six total token forwards; neither includes an independent numerical reference. Every forward executes full top-10 across48 layers.

| Backend | Whole test elapsed | Peak process RSS | Final GPU pool snapshot |
|---|---:|---:|---:|
| NumPy |40.68s (internal40.475s) |10,281,414,656 bytes (~9.58GiB) |0 |
| CuPy / RTX5080 |30.06s (internal29.782s) |8,632,389,632 bytes (~8.04GiB) |4,414,358,016 used /4,420,487,680 reserved bytes (~4.12GiB) |

Timings include construction, prefill, reset and incremental generation; do not divide by token count and advertise steady-state tokens/s. The GPU path retains a budgeted backbone, while all routed experts execute native CPU because IQ GPU kernels are not implemented. Counters show2880 native CPU experts and zero decoded-host-cache bytes. The GPU pool snapshot is not total process VRAM or a sampled high-water mark. RSS includes file-backed pages; it is not the owned allocation estimate.

Initial host-span/CUDA-event profiling is recorded below; exclusive critical-path profiling, global VRAM peaks, SSD bytes/page faults and overlap measurements remain outstanding in Patch0212. CUDA graph capture, fixed-slot transfers, grouped prefill and adaptive prediction are not enabled. The subsequent short-text validation gate permits measured performance work; complete independent full-model equivalence remains unclaimed. Reference implementations remain available.

## Independent safe-loading diagnostic

A separate CPU-only pinned-source copy disables only loader prefetch. The fixed[1,2] comparison ran sequentially in27.16s; reference peakRSS~5.72GiB and production~8.71GiB, below the12GiB diagnostic cap. This resolved the unsafe eager-prefetch prerequisite, **not** model numerical parity. Logit relativeL2 differences remain~5–6%. No production performance optimization followed that comparison alone.

## 0226–0227 short-text gate and initial0212 profile

Representative16-token independent DeltaNet/QSA state replay passes, including continuous local state. Full-model incremental/prefill logits and all active-state hashes are identical at prefixes1/2/4/8/16. No production mathematical defect was exposed; measured profiling has begun, without an implemented speedup.

Sequential CuPy processes, IDs1–16, four native expert workers, FP32, max_length16: incremental67.99s / prefill65.30s; sampled peak RSS10,369,241,088 /10,391,388,160 bytes. GPU pool4,414,818,816 used /4,420,900,352 reserved bytes; not global VRAM high-water. Each process executes7,680 native CPU experts with zero decoded-expert cache and zero GPU expert residents.

Positions8–15 mean instrumented token latency3.6965s. Principal host spans/token: routed experts1.55152s; complete GDN including projections1.36484s; complete QSA including projections/indexer0.55624s; parent MoE1.72397s. Spans include waiting and nested operations: **do not sum parent/child timings or interpret these as exclusive critical-path costs**. No per-operation synchronization is inserted; token boundaries synchronize.

Measured next priorities are packed GPU mixed-format expert execution, separating GDN projection fallback from recurrence/launch costs, and QSA launch/synchronization/indexing. See `0212_initial_profile.md` and `0227_short_sequences_profile_v1.json`. This is a short instrumented baseline, not steady-state throughput or an optimization comparison.

## Patch0213 — direct packed GPU experts and launch reduction

Source reference remains pinned Strata `c77572806996349b49995e4f98d41b5294d36a21`. No route-tail skipping, Q8 activation approximation or speculation is enabled. Exact checkpoint formats remain packed; activation/state arithmetic remains FP32.

Sequential separate-process ablations consume fixed IDs1–16, measuring synchronized decode positions8–15 without stage instrumentation. Cache admissions/uploads in those positions are included; construction/JIT warmup at earlier positions is not. These are short forced-token rates, not natural-language or long-context steady-state throughput.

| Cumulative configuration | Tokens/s |
|---|---:|
| Previous4GiB/legacy-GPU placement, CPU experts |0.272867 |
| All eligible backbone GPU projections |0.626706 |
| Batched CPU miss transfers |0.628522 |
|6GiB packed expert cache, admit after3 requests |1.274565 |
| Admit after1 request |4.757141 |
| Fused all-causal-cell short-context QSA |6.054848 |
| Grouped resident IQ dispatch |6.098446 |
| Fused FP32 HC stages |7.386917 |
| Existing full-softmax fused router enabled |7.740246 |
| Three-launch sigmoid-gated DeltaNet |**8.566126** |

Final speedup **31.39×** over the measured baseline; target70–80tokens/s is **not reached**. No confidence intervals/repeated-run speedup significance are claimed for small adjacent differences. The grouped kernel separately measures1.21–1.31× over sequential resident GPU top10; small end-to-end effect reflects all-selected-resident eligibility and fresh admissions.

Versioned artifacts: `0229_decode_performance_v1.json` preserves cache admission failure; v2 fixes double-counted resident backbone bytes; v3–v5 reuse successful prior variants and execute only newly introduced variants. Latest numerical difference versus previous FP32 baseline: relativeL2≤2.065315e-6, maxabs≤2.789498e-5; all checked argmax/top10 sets match. Latest decode process peakRSS10,318,860,288 bytes, pool9,819,179,008 used /9,829,571,072 reserved bytes.1,952 resident experts occupy3,994,496,000 bytes within the6GiB cap; no full expert bank expansion.

Optimized reset/prefill lifecycle also passes (`0233_fast_lifecycle_v1.json`):16-token logits/all-active-state hashes identical and repeated greedy smoke IDs `[220,100561,198]`. Its extra requests use4,778,188,800 cache bytes and10,616,590,848 pool-reserved bytes; these are distinct workloads, not comparable throughput measurements or global VRAM high-water marks.

Latest instrumented profile (`0231_fast_profile_v2.json`), mean host spans/token: routed experts0.05940s; complete GDN0.01208s; QSA0.02152s; HC attention+FFN reads0.00770s; router0.00172s. Profile spans include waiting and nested work and perturb timing; do not sum them or replace the uninstrumented throughput above. Remaining work is transfer/dispatch attribution, fixed-slot async movement, stable launch/graph reduction and longer realistic benchmarks. See `0213_packed_gpu_and_decode_optimizations.md` for exact controls and Strata comparison caveats.

## Follow-ups0234–0238 — codebooks, grouping, warp rows and fixed slots

The same bounded fixed-ID method now extends the preserved ablation through `0229_decode_performance_v10.json`, reusing successful previous variants rather than executing them again.

| New variant | Tokens/s | Decision |
|---|---:|---|
| Read-only global codebooks |9.174106 | Opt-in; arithmetic/output bits unchanged |
| Static HC graph replay |8.965350 | No demonstrated gain; keep disabled |
| Group after complete cache admission |10.197347 | Removes cold-route individual expert launches |
| Warp-per-row grouped IQ |**12.033481** | FP32 row reduction changes, numerically validated |
| Fixed-slot cache with same warp kernels |**11.976041** | Bounded ownership foundation;0.5% difference not significant from one run |

Best observed rate is44.10× the original0.272867tokens/s. This is still **not70–80tokens/s** and not a natural-language/long-context benchmark. Strata's exact launch/MTP/tail-skip/cache/context settings are needed for a like-for-like target.

The real-weight warp benchmark (`0237_iq_warp_scheduling_v1.json`) measures median1.472287× CUDA-event speedup over the retained block kernels across7 actual triples, with20 warm balanced samples/mode. Worst relativeL2=3.176314e-7; maxabs3.492460e-10. Additional RSS882,761,728 bytes; no checkpoint modification or full expert decode.

Full-model warp/fixed-slot results versus the original production baseline: max relativeL2=2.014602e-6 and maxabs=2.717972e-5; argmax/top10 match at prefixes1/2/4/8/16. These two cache variants' logits **and active-state hashes match bit-for-bit**. Slot GPU arena6,440,345,600 bytes, pinned staging4,194,304 bytes; forced-token process RSS10,302,685,184 bytes/default pool12,267,438,592 reserved bytes. These are not global GPU high-water measurements.

The extended lifecycle `0233_fast_lifecycle_v2.json` passes16-token prefill/incremental/logit/all-state equality and repeated greedy IDs `[220,100561,198]`. Elapsed11.1123s includes multiple requests and must not be divided into steady-state throughput. Sampled RSS11,180,748,800 bytes; default pool12,270,584,320 reserved bytes. It retains2,332/2,419 slots without quarantines.

The graph experiment owns17,153,024 private-pool bytes under a203,423,744-byte cap, separately from the default pool. No whole-decode graph, dynamic-position capture or asynchronous mover is claimed. Remaining priority is bounded admission/transfer overlap and longer realistic cache-pressure/steady-state tests. Individual patch contracts and failures are preserved in `0234_codebook_storage.md` through `0238_packed_expert_slots.md`.
