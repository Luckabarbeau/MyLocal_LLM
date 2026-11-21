# Qwen4Exp implementation status

## Current — 0335 identical-input CPU expert experiment

Diagnostic-only native comparison implemented and run twice on actual checkpoint slices/retained reference activations, 16/20participants, 3/6/10experts. All24whole-expert local-versus-pinned-Strata-GGML configurations bit-exact; no general IQ-backend upgrade established. Dedicated unchanged Strata Q2_0 rows are1.32–1.55x faster under the same neutral native scheduler even with exact local Q8_0 values/scales imported; relativeL2~2e-7/maxabs~3.35e-8, versus~1.8e-4relative for legacy FP32-scaleActQ. **Not a full-model gain; production unchanged;50/sremains unmet.** 77selectedCPUtests passed/4skipped; adapter synthetic checks pass. ProcesspeakRSS1165.875MiB, sampledcgrouppeak~0.976GiB,swap0; no full model/GPU run. Source provenance, artifacts, numeric baselines and next opt-in Q2row integration gate: `0335_identical_input_cpu_experts.md`.

## Prior audit — 0334 GPU expert capacity and native CPU backend

User-requested pre-optimization audit: local has 3673 complete GPU experts versus historical Strata75.2/s 2792 and controlled fulltop10 Strata3786; capacity is not the main shortfall. Current local backend is native `ggml_pool`, 19workers+participating service thread, SIMD/Q8 arithmetic—not the old Python expert executor. Fresh isolated feature query confirms AVX2/FMA/AVX-VNNI availability in the loaded GGML library; no model run. CPU MoE exposes ~5.13ms/token GPUwait, but current captures differ in CPU work (~2.99 versus6.02experts/layer), so backend superiority is unresolved. Higher Strata utilization is partly consistent with its20ms spin policy and more CPU-routed work; increasing local spin previously regressed performance. No production changes or fresh throughput/regression claim. Evidence and normalized same-input CPU A/B next dependency: `0334_gpu_cache_cpu_backend_audit.md`. Target50/s remains unmet.

## Prior — 0326–0333 measured placement gain; 50/s still unmet

**Latest completed four-response controlled replay:** static sized-slot profile + exact router **44.64/47.12/47.20/47.20forwards/s**, aggregate46.51275/s over1172decode forwards. Matched original-router/profile control45.69/46.55/46.53/46.16/s, aggregate46.22953/s. Hot last3:47.17321vs46.41283/s (~1.64%observed gain, no confidence claim). All responses have294forced outputs after128warmup; **not natural conversation or Strata configuration parity**. Earlier profile pair improved45.02->48.30hot/s with same7GiBarena; differing warm states/trajectories are not interchangeable baselines.

Implemented opt-in setup-only sized-slot frequency placement and prior-request profile input, cgroup/file-page attribution, exact warp router variants, and warmed-second-response tracing. Profile reduces observed CPU experts75864->56114/request; installation0.51s/1278replacements, no runtime mover or added GPU/host capacity. Router real-checkpoint two-request IDs1..16 comparisons preserve all10saved FP32logits and all saved persistent-state records **bit-exactly**. This is same-site local parity, not new independent llama.cpp or cross-placement equivalence.

**Final selected regressions:463CPU passed/12skipped/1deselected/166subtests passed;411actualCuPy passed/4collection-skipped/1deselected.** Qwen3-Next CPU coverage passes; GPU Qwen3-Next modules are collection-skipped. Existing NumPy-route-ID GPU fixture failure remains explicitly scoped/unmodified. No full training-suite claim.

Successful0330tracev2records16hotdecode steps:~4.99ms/token packed dense projections,~2.40msHC,~5.13msGPUdependency waits. Native-only phases total5.79ms/token,3.60msgate/up/1.64msdown; these overlap GPU work. IQ2_S AVX2gather is bit-exact but slower, **defaultoff/performance rejected**. Isolated HCdown~3%nominal/~0.033ms/token is not worth integrating. Original math/quantization encoding, fulltop10, CPUfallback and guards remain intact. No checkpoint downloads/writes or complete host-pinned arena.

0330tracev1and0332combinedv1VRAM-admission failures are preserved. After user-reported interference was removed, identical guarded relaunches completed; no budget/headroom relaxation. Next: measured shared-input packed dense projection reuse/fusion, guided by the new hot trace; natural conversation/profile-generalization validation remains open. Full contracts, tests, memory/performance scopes, failures and next dependency: `0326_0333_optimization_report.md`, `0326_0333_results_v1.json`, `0332_router_model_parity_v1.json`.

## Prior — 0324–0325 exact-launch audit and placement diagnostics

**50decode forwards/s remains unmet.** Historical installed-Strata logs corroborate75.2emittedtokens/s (12261tokens/163053ms), with46.84GiBpinnedexpertarena, profileGPUcache, adaptiveasyncplacement, INT8KV, route-tail skipping and suffixdrafting. This is not a new fulltop10benchmark. The fresh unchangedCLI selected a different mmap/32.82GiBcomplement path under40GiBhardlimit, then stopped at38GiBMemoryHigh before serving. No download was needed. Session-aware offline analysis separates23sessions/22historical/1current; no current generation summary.

0324adds bounded exact-launch capture/session analysis and pre-cleanup execution-outcome persistence.0325exposes opt-in native route counters and implements causal, size-class-compatible request-boundary placement analysis (max8bundles/32MiB; no actual movement). **74diagnostictests passed,61subtests passed.** New natural two-response profile attempt safely stopped at28GiBMemoryHigh after57s, last response progress32tokens; no completed response or placement-benefit/speedup claim. Partial actual counters validate48layers×512experts,108calls/layer and51840routes, with correct top10and slot occupancy.

Production mathematics, kernels, defaults, checkpoints and pinned references remain unchanged. No new logit/state parity claim. Latest completed local control remains0321**41.99128/43.70105forwards/s** forced replay. Next: cold file-page/cgroup working-set attribution under unchanged limits before another model run; complete causal placement measurement before implementing a mover. Do not increase caps, pin the full complement or import Strata tail skipping silently. Evidence, commands, memory scopes, limitations and next dependency: `0324_0325_strata_residency_report.md`, `0324_strata_sessions_v1.json`.

## Prior — 0323 restart checkpoint; optimization paused by user

**User requested a PC restart and handoff. No further model work is running. Target50decode forwards/s remains unmet.** Resume from `RESTART_HANDOFF_0323.md`, not the older historical sections. Latest complete controlled pipeline0321 measures **41.99128 / 43.70105 forwards/s** over two359-output/358-forward repeats. This is forced-trajectory replay, not natural conversation throughput. Fresh Strata clean subset36.48/49.26/s is also separately warm-dependent, with distinct KV/cache/placement.

0320 adds opt-in **bit-exact FP32** warp-dot/value-tile QSA (~2x standalone), including runtime-position graph dispatch. 0321 fixes orphaned request-graph retirement before replacement-state admission; all memory guards remain intact. 0322 whole-token capture covers all48layers/12QSA/PLE and passes same-prepared-arithmetic two-request parity, but39.32/42.22/s shows no gain. Preparing PLE with capture-capable FP32 matmul changes old cuBLAS-path logits2.3–6.7%; **whole-token/PREP are not quality accepted relative to the older path and remain off**. Cooperative0314 and slower exact0319 staging also remain off/unintegrated as documented.

0323 decode-only CPU profile:9.47of11.50instrumentedseconds in716CuPy.get calls; this is queued GPU/native-work completion waiting, not physical transfer bandwidth or CPU utilization. Nested validation timings overlap and are much smaller. Removing validation would shift the wait, not demonstrate a gain. Do not restart blind spin/affinity tuning or assume every graph/host gap is removable latency.

**Final source selected regressions:732CPU passed/116skipped/1deselected;586actualCuPy passed/24skipped/1deselected.** Includes selected Qwen3-Next coverage, with GPU collection skips explicitly recorded; no full training-suite claim. Latest all-state/logit parity is against the appropriate local-arithmetic controls, not a new independent llama.cpp equivalence proof.

Source/artifact hashes, failed probes, exact numerics, memory scopes, commands and next dependency: `0318_0323_execution_report.md`, `0318_0323_execution_results_v1.json` (read-only reproducible check via `scripts/analyze_0318_0323_results.py --check`). Earlier0310–0317v2 snapshots are historical source states; do not overwrite/refresh them. All changes remain uncommitted; no checkpoint/reference mathematics was modified.

## Prior — 0310–0317 comparative profiling; target 50/s still not met

Fresh installed Strata, full top10/no drafting/context1024, reaches **36.48283 / 49.25981 decode forwards/s** over two identical 359-token response repeats; the second response is hot, not the overall sustained rate. Its INT8 KV, 7.233GiB static profile cache and CPU placement differ from local FP32 KV/7GiB observed placement. `strata_launch_bench_capture.sh` remains the workload/launcher reference, not an unchanged newly completed launch.

Implemented, opt-in, scope-numerically-validated: bounded parallel packed PLE row I/O (0312), complete native-service recurrent block capture (0313), adjacent recurrent-chain capture (0315), and complete short-context QSA-block capture with runtime positions and request-bound caches (0317). Real-checkpoint recurrent/chain captures preserve ten saved logits and all state hashes exactly; all12 QSA blocks preserve these against a **same capture-ready arithmetic** control. The full-model parity fixture uses CPU experts; mixed/resident arithmetic is separately tested. These are not new llama.cpp equivalence claims.

Cooperative FP32 Triton QSA (0314) improves standalone operators 2.08–12.46x and agrees with FP64 at ~1.6–1.9e-7 relative L2, but the quantized full model changes 3.62–6.89% at saved prefixes. **Not quality accepted; remains opt-in.** No graph equation or checkpoint encoding was changed to force parity.

No robust hot full-model gain is established: matched PLE control39.81/39.79 versus candidate39.44/40.19; recurrent38.24/41.04; cooperative39.51/41.70; chain39.18/39.95; completeQSA41.44/39.94 forwards/s. Higher CPU spin raises active-core use but regresses throughput; copied Strata affinity also regresses. Defaults stay unchanged. Cold PLE fault/latency reduction is real, not a hot-throughput claim.

Final source selected regressions: **407 CPU passed/107 skipped/1 deselected; 284 actual CuPy passed/23 skipped/1 deselected**, including selected Qwen3-Next coverage (GPU modules also have collection skips). Prior evidence remains preserved. Numbered objectives/interfaces, numerical/performance/memory scopes, failed runs, source hashes and next dependency: `0310_0317_strata_comparison.md`, `0310_0317_results_v2.json`. Next: causal timeline attribution after complete-block capture; do not assume uncovered GPU gaps or higher CPU utilization are recoverable useful work.

## Prior — 0308–0309 storage diagnosis; target 50/s still not met

0308adds a payload-read-free Linuxmincorecensus over all1224checkpointtensors, with10passingtests. Latest canonical routed expert pages:50.292GBtotal/35.410GBresident/14.882GBnonresident; PLE28.866GBtotal/~0.697GBresident. This is file-page residency, not a CPU-complement or per-token attribution claim. 0309tests the existing4GiBanonymous CPUsource:2111bundles, no pinning/staging, identical371naturaloutputIDs,32.516forwards/s versus earlier35.159/s; **no demonstrated RAM-source gain**. No default/source-budget increase.

Latest complete results/sourcehashes: `0295_0309_optimization_results_v1.json`; diagnostic contracts, memory limits, failed initial census and next per-source/decode-only I/O experiment: `0308_0309_storage_residency.md`. Production verified state remains471CPU/201GPUpassing selected regressions plus exact local checkpoint logits/state comparisons; no production edits after those regressions. Additional0308diagnostictests10passed. No50/scompletion claim.

## Prior — 0293–0307 critical-path work; target 50/s not met

The retained best controlled single observation is **42.64032 decode forwards/s** (0302 QSA frontend); latest adjacent control/shared-grid pair **40.05676 / 40.17001/s** shows no material model gain. Natural conversation runs with identical generated IDs measured **31.95859 / 35.15922/s**, with substantial remaining file I/O/major faults. Do not present forced-token replay as natural-generation throughput.

Implemented and scope-numerically-validated: bounded native staged execution (0293–0294, experimental/off), causal heat proposals (0295, no integrated mover), request-bound recurrence and exact GPU graph copies (0296), CPU IQ3_S gather (0297), aligned CPU/GPU/phase instrumentation (0298), experimental IQ4 VNNI/IQ3_XXS gather/pair/HC/task-grain variants (0299–0304), native-graph shared branch (0305), fenced GPU publication of pinned input slabs (0306), and exact shared-memory IQ GPU codebooks (0307). **0307 is operator-performance-optimized (1.085–1.857x real operators), not model-performance-optimized.** Losing/neutral switches remain off and task factor remains 3.

Final selected regressions: **471 NumPy passed/43 skipped; 201 actual CuPy passed/4 skipped**, including Qwen3-Next components. Actual checkpoint two-request prefixes 1/2/4/8/16 preserve all 10 logit boundaries and persistent-state hashes exactly against retained uncaptured local arithmetic; the fixture uses CPU experts, while resident GPU arithmetic has independent Strata-source/real-slice tests. No new independent llama.cpp full-model equivalence or complete training-suite claim.

Every experiment used verified isolated services and unchanged memory/headroom guards. No complete host-pinned complement, discarded expert tail or checkpoint change. Full per-patch interfaces, status, tests, numerical/memory scopes and failed experiments: `0295_0307_critical_path_optimization.md`; results/source hashes: `0295_0307_optimization_results_v2.json`. Safe reproducible runner: `scripts/probe_0302_decode_configuration.py` under0290 isolation. Next dependency: attribute natural-trajectory page faults to expert/PLE sources before changing host residency; retain hot replay for CPU compute/overlap diagnosis.

The older sections below are historical, not current completion/default claims.

## Current — 0291–0292 isolated benchmarks; warm controlled decode ~33.9/s

After0290's safety gate, user authorized resumed benchmarking. All new experiments ran sequentially in verified isolated services; no50GiB bulk warming or44GiB source. **0291 actual-checkpoint partial-source validated:** 60-expert CPU copy test exact,128MiB cap; full model with512MiB anonymous source/263registered bundles33.90548forwards/s versus warm mmap control33.87091/s, same128warmup/359response IDs and CPU/GPUexpert counts. No material source-copy gain; cold21.295/s was disk-confounded. Full/pinned complement still unvalidated.

**0292 standalone split operator validated:**56actual mixed-format CPU/GPU configurations preserve complete top10 and exact same-site outputs. Staging1–2misses helps selected high-miss cases~5–20%; cheap/low-miss groups and blanket staging often lose. Not integrated into model decode. Fresh16tokenNsight:210.12msGPUoperationunion/514.77msspan,177.50msgaps preceding CPU-resultH2D versus2.88mscopies; waits/CPUcompletion remain the priority. DeferredQSAcheck trial34.01985/s is only0.44% above warm baseline, not a demonstrated gain; default remains unchanged.

Final focused tests **96CPU passed/4skipped/1deselected;20actualGPU passed/1skipped**. No production mathematics/kernel/default edits in0291–0292, no new independent model-parity claim. Full report/sourcehashes/results: `0291_0292_residency_and_miss_split.md`, `0291_0292_benchmark_summary_v1.json`. Next dependency: bounded native asynchronous service/mover with format/miss-count-aware staging, failure/lease/combination tests, then same-prefix/state and longer-workload comparison. **50/s remains unmet.**

## Prior — 0290 safety gate; experiments initially paused

**2026-11-19 desktop OOM incident confirmed:** systemd-oomd killed the GNOME Shell cgroup at50.6G; individual-process attribution remains unresolved. Earlier process-group/sample guards were not cgroup isolation. No44GiB resident-source model benchmark was launched. Do not resume full-model/pinning work based on previous successful small tests.

**0289 implemented/small-fixture validated, large-memory unvalidated:** optional complete packed anonymous CPU-source copies, canonical mmap ownership preserved, setup-only native registry replacement, explicit host budget and optional paced registration. No new throughput result. GPU-staged miss splitting is still not integrated.

**0290 implemented and CPU/no-model validated:** fail-closed separate systemd service, verified kernel memory/swap/group-OOM limits,12GiB reserve, PSI/ancestor monitoring, bounded runtime/cleanup and exclusive telemetry. Longer workload CLI now requires verified isolation. Final bounded regression **89passed,4skipped,1deselected**, cgroup peak643858432bytes (~614MiB), sampled PSI0. Timeout cleanup test returned expected125; no benchmark units remained. No CUDA initialized, model payload run, pinning or large allocation in0290. This is not proof against all system/GPU-driver OOM risks.

See `0290_memory_incident_and_isolation.md` and versioned journal/guard artifacts. Next dependency: small isolated CPU source memory/lifetime measurement before any increase in budget or GPU work. **50/s remains unmet; no mathematical/precision changes.**

## Prior — through 0288; safe callback observation 31.43/s, target 50/s unmet

Integrated opt-in mixed-format GPU Q8 covers IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS/Q2_0 and Q4_K/Q5_K/Q6_K/Q8_0. Extracted pinned-Strata parts/rows are bit-exact; independent FP64 decoded-Q8 baselines/truncation bounds are tested. CPU retains GGML per-format Q8_K/Q8_0. This is scoped arithmetic alignment, not whole-engine Strata equivalence.

Direct BF16 HC storage reconstructs the exact checkpoint FP32 values at loads, saves **1,279,262,720 VRAM bytes**, and preserves exact outputs/state. Same-arithmetic full-model logits/all active-state hashes match at prefixes1/2/4/8/16 over two requests for BF16 storage and the new native mapped-doorbell service. The service has bounded waits, sticky failure reporting and retired ownership; its measured gain is not material, so it remains opt-in.

Final natural7GiB reproduction measured29.29/s on the same359 IDs; the29–32/s spread makes small single-pair gains uncertain. The8GiB probe was safely rejected by physical memory headroom checks, with no benchmark result inferred. Final focused group:122passed.

Longer sampled runs: expert+dense Q8 with19 workers/50us spin28.61/s; exact BF16 HC30.62/s with the same415 IDs;7GiB cache31.43/s; historical initial service31.71/s with the same359 IDs. Its timeout path was corrected to avoid reading incomplete CPU output; final completion-gated/device-status service29.56/s preserves those IDs and exact state parity, with no material gain. Failed copies never read CPU payloads; repeated mapped-status system loads are eliminated by publishing a device-local status once. Different trajectories must not be used for identical-input ratios. Size-classed fixed slots retain3673 versus2823 experts in7GiB; controlled359-token replay30.20 ->31.15/s is modest, not50/s. Cumulative warmup placement, hidden-phase fusion and extra graph flags did not demonstrate useful speedups. Direct mapped PCIe weight dots are slower than native CPU and are **not integrated**; staged expert costs are measured, but integrated asynchronous splitting remains pending.

**Final verified inference tests:1969 NumPy passed/1124 skipped;1367 actual CuPy passed/14 skipped**, including Qwen3-Next GPU mixer/attention/decoder/model. Full training suite not run. No production graph equation or pinned reference mathematics corrected, no checkpoint changed, no expert tail omitted. Current defaults/reference paths remain available. Contracts, commands, per-patch status, memory scopes and next dependencies: `0281_0288_execution_optimization.md`, `0288_reproduce.md`, `0281_0288_performance_v3.json`.

## Prior — through0280; best longer decode26.95/s

**0280 implemented/numerically validated within path, not performance optimized:** node-level graph trace attributes the dominant runtime copy waits to4-byteQSAfinite-status reads. An opt-in device-status/conditional-cache-commit path preserves every check, reports the first bad layer once/modelstep and isolates failed requests. Focused9tests pass; actual checkpoint fixed-prefix logits/all-state hashes remain bit-exact over two requests. Longer same413-token run24.84134/s does not improve26.95159/s; keep default off.

Latest verified source: **1939CPU passed/1025skipped,1211actualCuPy passed/18skipped**. JUnit and source hashes: `0280_{cpu,gpu}_regression_v1.xml`, `0268_0280_performance_v3.json`. Node trace:635.60ms/16tokens,380.77mssummedkernels,363.54msGPUoperationunion; overlapping diagnostic spans, not throughput/exclusive attribution. Coarse earlier graph traces omitted captured kernel nodes. Current priority is integrated mixed-format packedGPUQ8/nativeCPUcost/CPU–PCIesplitting, not blind fence deletion. See `0280_qsa_validation_dependencies.md`. No production mathematical defect demonstrated; fullmodelGPUQ8 remains unintegrated.

## Validated history — 0268–0279 execution scopes; best longer decode 26.95/s

**50 tokens/s remains unmet. No new production graph defect was found or corrected.**

- **0268 resolved:** the 3.73e-9 FP32 difference was a block/warp test-oracle mismatch. Same-warp comparisons are bit-exact; measured block/warp relative L2 2.3721e-7. Longer run 17.58/s. The old blocker below is historical.
- **0269–0273:** host participation, optional affinity, pre-MoE/shared graphs, immutable native descriptor registration and generation-tagged task completion. Best 0272 long 18.96/s. Exact Strata affinity regressed here and is not the default. See `0269_0273_execution_pipeline.md`.
- **0274 implemented/numerically validated within path:** deferred shared launch after CPU work publication, once-only callback/error draining; long 19.31/s.
- **0275 implemented/numerically validated within path:** stream-ordered native CPU host function, fixed resident leases, bounded canonical registrations, no Python completion wait; long 24.29/s.
- **0276 implemented/numerically validated within path:** captured native handoff with runtime routes/activations, owned pinned buffers and ordered CPU/GPU join; long 25.86/s.
- **0277 operator validated:** existing specialized resident kernels wired to device plan, no demonstrated model gain (24.62/s). Native CPU timing added. Larger unpinned **23-worker + participating callback team reached 26.95159/s**, ~17.06 active-core equivalents.
- **0278 implemented/numerically validated within path:** short-context QSA frontend graph, dynamic RoPE inputs, unchanged validation/commit/sparse fallback; 26.13/s, no demonstrated gain.
- **0279 implemented/numerically validated within path:** input-only graph operands avoid unnecessary copy-back without changing default mutable-state semantics; 25.41/s, no demonstrated gain.

0273–0279 responses have the same 413 output IDs. These are single sequential runs, not confidence intervals. New 0274/0275/0276/0278/0279 fixed-token captures have bit-exact same-arithmetic logits and all active-state hashes at prefixes 1/2/4/8/16 across two requests. This is not a new independent llama.cpp equivalence claim. GPU experts still use FP32 activations; CPU experts use GGML per-format Q8_K/Q8_0. All-format GPU Q8_1 is not integrated.

**Verified final source state:** NumPy inference regressions **1939 passed, 1016 skipped**; actual CuPy inference regressions **1202 passed, 18 skipped**. Relevant Qwen3-Next models/mixers/decoders included; entire training suite not run. XML, source hashes, timing and memory: `0279_{cpu,gpu}_regression_v1.xml`, `0268_0279_performance_v2.json`.

Frozen native handoff intentionally pauses per-expert decayed heat, retains all canonical sources/resident leases, and collects exact aggregate counts at synchronized statistics/cleanup seams. It is not an asynchronous placement mover. Peak RAM/VRAM scopes, flags, failed/neutral experiments and current critical-path evidence are in `0274_0279_native_handoff.md`. Current next dependency is node-level graph timeline/copy-call attribution and a supported native completion mechanism; an isolated mapped-memory `cuStreamWaitValue32` probe returned CUDA error 801 (unsupported).

## Historical milestone — SHORT-TEXT GATE PASSED; ~44× MEASURED DECODE SPEEDUP

The original local two-shard IQ3_S checkpoint now constructs and executes all48 layers on NumPy and CuPy/RTX5080, using packed CPU/GPU routed experts backed by mapped canonical sources. No pruned model, full expert expansion, whole-PLE allocation, new checkpoint download, or all-resident GPU intermediate architecture is used.

**Complete model/state mathematical equivalence is not yet claimed.** The current numerical investigation fixed the diagnostic indentation and analyzed retained states successfully. Identical-input MoE routing/experts/shared/combination tests at early and late layers agree at expected arithmetic precision after accounting for activation quantization. A one-time1.31e-6 early input perturbation produces2.31–6.33% logit changes in the diagnostic quantized path, demonstrating sensitivity at the observed discrepancy scale. No production mathematical defect is established. Subsequent0226 replays pass236 comparisons each for identical reference pre-state and continuous local state across16 tokens. Full-model CuPy incremental/prefill logits and all active-state hashes match exactly at prefixes1/2/4/8/16 (0227). The short-text gate has passed. Patch0213 now adds direct packed GPU kernels, grouped experts and switchable QSA/HC/router/GDN fusion. Bounded forced-token decode now reaches12.033481tokens/s with dynamic packed allocation, or11.976041 with fixed slots (~44× the original0.272867); the requested70–80tokens/s remains unmet. New optimized lifecycle checks preserve exact within-path prefill/incremental state consistency and the retained three-token greedy smoke. Full independent model equivalence and long-context attribution remain outstanding. See numerical_root_cause.md. No model equation was corrected or approximated; new execution kernels alter FP32 reduction ordering within quantified bounds. The pinned reference mathematics remains unchanged.

Status vocabulary: Not started; In progress; Implemented but unvalidated; Numerically validated; End-to-end validated; Performance optimized. Validation scope is always stated; an operator claim is not a model claim.

## Historical — initial 0265–0268 report (0268 blocker resolved above)

- **0265:** opt-in whole recurrent segment graph, existing Triton FP32 GEMV plus fused DeltaNet;16MiB cap per recurrent layer, explicit preflight/cleanup. GPU focused41passed2skipped; CPU model/workload57passed4skipped. Real fixed-prefix same-GEMV control/capture over two requests has bit-exact logits and all active-state hashes at1/2/4/8/16. This is replay parity, not new llama.cpp equivalence. Long sampled273-token response12.60665forwards/s.
- **0266:** current-path Nsight capture inside the longer response:16tokens/1.48257s, GPU operation union405.990ms,41,969driverlaunches, CPU completion-wait spans321.591ms. Times overlap; not an exclusive decomposition. Retesting mixed grouping12.55698/s gives no demonstrated gain.
- **0267:** opt-in native bounded spin, sleeping default preserved.57tests passed2skipped including policy/timing groups. Synthetic5-expert batch360.011us ->175.846us; long **15.21239forwards/s**, same128/273tokenIDs as0265. One sequential pair, not confidence bounds. See `0265_0267_performance_v1.json`.
- **0268:** persistent device route/expert plan implemented, default-off; **latest retest94passed1skipped2failed**. GPU-containing exact-output tests fail (all-GPU maxabs3.7253e-9 FP32); all-CPU case and lifecycle tests pass. Candidate explicitly useswarp kernels while test oracle may use defaultblock reduction. No actual graph defect demonstrated, but parity is unresolved. One fixture correction/retest used; stopped, no tolerance relaxed and no real-model0268benchmark. **Final worktree is not fully verified.** See `0268_device_plan_blocker.md` for the single next experiment.

**50t/s remains unmet.** Last completed model benchmark is0267before the experimental0268source edits. Preserve this distinction; do not label0268 complete or infer final-tree/model success from imports, old test totals or tiny discrepancies.

## Historical — structural pipeline experiments (0260–0264)

- **0260 bounded admission:** explicit placement windows; deferred misses execute completeCPUexperts, no route-tail omission. Frozen placement eliminates response-case promotions, reaches10.31149forwards/s; onepromotion/480requests10.29833/s. Sampled trajectories differ from0259 due to existingCPUQ8/GPUFP32 site arithmetic; no identical-trajectory ratio claimed. Defaults unchanged.
- **0261 opt-in exactQ2_0SIMD:** AVX2/F16C CPUdot retainsGGMLQ8_0 and scalarFP32block order; mixedexperts/directdots bit-exact. Syntheticdown4.138x/wholeexpert1.865x; longfrozen10.72100/s, same324tokenIDs. NotStrataActQ orall-formatGPUQ8.
- **0262 opt-in shared/routed overlap:** explicit producer/completionevents, no cross-layer dependency violation; long10.79765/s, sameIDs, no convincinggain inonepair. Uncaptured seam only.
- **0263 freshprofile:** current frozen/SIMD/overlap response includingprefill,400forwards/49.904instrumentedseconds; repeated canonicalbundle construction5.904cumulativeseconds, nativecompletionwait6.212internalseconds. NotGPUcriticalpath oruninstrumentedthroughput.
- **0264 defaultmetadatareuse:** bounded canonicalreadonlymmapbundlehandles, no payloadcopy; cleared atmodelunload. Long**11.42412decode forwards/s**,323forwards/28.27350s, same324tokenIDs as0261/0262. Configurationstill differs fromStrata;50t/s not reached.
- FinalCPUregressions**1875passed962skipped**; selectedactualCuPy**147passed3skipped**. Allsixlong0259–0264comparisonruns completedsupervised;0263profiling ran directly withchild/terminalguards, notparent-supervised. Artifacts andscope in `0260_0264_pipeline_comparison_v1.json` andnumberedreports.

The big remaining candidate is native execution planning/capturedGPUsegments with persistentbuffers and dynamicdevice routing/state metadata. Do not infer a50t/s guarantee from the historical summedGPUkerneltime. No model equations/referencebuild/checkpoint modified; no tail omission orspeculation silently added.

## Historical — pinned Strata audit and persistent CPU pool (0258–0259)

- **0258 documented:** exact installed Strata source `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`; executable CPU scheduling audit, source/interface distinctions and staged placement/transfer/graph/precision ledger in `0258_strata_pipeline_audit.md`. Not all audit items are implemented.
- **0259 opt-in `expert_cpu_backend='ggml_pool'`:** bounded persistent native row workers share one per-format Q8 input across selected CPU experts; phases/ownership/draining preserve unweighted GGML arithmetic. Mixed-format CPU operator tests are bit-exact. Final CPU regression120passed7skipped; actual CuPy integration74passed1skipped. Final supervisor18tests pass after zombie-status handling; real pool and warmed control both fully supervised successfully.
- **Long sampled request:** identical128-token warmup and427-token response IDs; warmed old scheduler8.20971 versus pool**8.72394decode forwards/s**, observed**6.26%** higher rate. One sequential near-resident pair, not a repeated-trial confidence claim. Cold control7.55953 was confounded and its supervisor failed after child exit; preserved, not advertised as the isolated gain.
- **Remaining measured issue:**46,591promotions/94.914GB transferred during the response case including prefill; synchronous placement churn remains a candidate bottleneck. Next dependency is bounded profile/heat-driven admission and protected asynchronous publication, not blindly deleting waits. Specialized CPUQ2_0 and all-formatGPUQ8 remain outstanding.

See `0259_native_row_pool.md` and `0259_row_pool_comparison_v1.json` for contracts, memory, commands and limitations.50t/s remains unmet; model equations/defaults and pinned numerical references are preserved.

## Historical — longer sampled workload and Q8 progress (0252–0257)

- **0252 validated:**41GPUtests byte-exact pinned-Strata Q8_1 producer parity, native/IEEE modes. Historical blocked fixture report superseded by `0252_q8_1_validation.md`.
- **0253 operator validated:**IQ3_S×Q8_1 integer GPU GEMV38tests plus41quantizer tests pass; not integrated/all-format GPU inference.
- **0254 longer workload:**captured Strata prompts/template/sampling replayed with exact75/77 prompt token IDs,128 warmup/512 response cap. Config differences explicitly recorded. Old native CPU run timed out180s before warmup; no completed benchmark claim for it.
- **0255 opt-in heap policy:**bounded metadata index avoids full resident sorting, defaultsort retained; representative18000-request decisions tested, near-floating-tie equivalence not guaranteed. CPU policy~173–178x speedup is not model throughput.
- **0256 integrated optional GGML CPU:**format-specific Q8_K/Q8_0 dots, no dense experts; selected via explicitlibrary and `expert_cpu_backend='ggml'`. CPU integration/regressions64passed4skipped. Fully supervised sampled run completed128 warmup plus427-token short response: **8.03745 post-prefill forwards/s**, emitted convention8.05632/s. This is the current meaningful longer workload result; previous20.5 was a cache-fitting16-token trace, not equivalent.
- **0257 opt-in mixed grouping:**GPU/CPU terms retain ascending ordered combination; integratedGPU71passed. Child long result7.83378/s shows no demonstrated gain. Full source warming child7.71648/s versus prior mixed warmup7.71750/s did not explain the gap. Parent /proc/io exit-race fix failed its next run, though children exited0 and wrote results; further corrections stopped peruser rule. See `0254_strata_workload_comparison.md` / `0257_mixed_expert_decode.md`.

50t/s remains unmet. Strata suffix drafting, tail omissions,INT8 KV and large capacity are not silently claimed matched. CPU Q8 is enabled only explicitly; GPU inference is still FP32 outside the isolated0253 operator. Next is the supervisor race then real-workload CPU scheduling/promotion critical-path measurements.

## Historical — 0251 integrated; 0252 Q8_1 oracle blocked

0251 test API blocker resolved. Candidate43 passed1 skipped; GPU integration/regression group112 passed4 skipped; CPU model/Qwen3-Next group132 passed1 skipped. Opt-in `elementwise_fast_decode=True`, with native HC held constant, improves controlled warm fixed-prefix runs from17.97–18.03 to **20.24–20.53 forced tokens/s**, final logits and active-state hashes bit-exact. See `0251_elementwise_fusion.md`.

0252 source audit establishes the exact pinned Strata native Q8_1 contract and implements an isolated quantizer candidate, **not integrated**. Latest suite19 passed22 failed because the independent CUDA fixture's offsetof static_assert does not compile under NVRTC. One correction/retest used; candidate remains implemented but unvalidated. No byte discrepancy is known; no parity or speed claim. See `0252_q8_1_contract.md`. User authorizes Strata-aligned Q8 arithmetic, not silent substitution of Q8_0/Q8_K/Q8_1.

50t/s remains unmet. Existing precision-reference paths are retained. Subsequent entries below are historical, including the resolved0251 blocker.

## Current follow-ups — 0247–0251

- **0247:** warm Nsight timeline captured: ~29.1ms/token summed GPU kernels versus~75ms wall; discarded expert fallback zero buffer removed. GPU singleton/admission42 passed2 skipped; CPU23 passed21 skipped. Nsight wrapper RSS samples its profiler parent only, not the model descendant; do not mislabel it a model RSS guard.
- **0248:** batched QSA finite validation retains all checks before state commit. GPU49 passed. Controlled old14.20/14.34 versus batched15.61/15.85 forced tokens/s, repeated logits/state hashes exact.
- **0249:** opt-in warp specialization improves seven real resident expert triples1.40–1.97x with bit-exact outputs. GPU54 passed1 skipped. Controlled model A/B~16.25–16.47 shows no material whole-model gain; remains opt-in.
- **0250:** opt-in three-launch HC read, FP32 unchanged precision; GPU90 passed1 skipped. Real checkpoint warm17.76–17.85 forced tokens/s, final-logit relativeL2=6.52e-7 against prior path, same argmax/top10; native-path repeated logits/state hashes exact. Separate-process rates are not a controlled A/B improvement ratio. `hc_native_read=True` plus `hc_fast_decode=True`; chat default unchanged.
- **0251:** implemented but unvalidated and unintegrated. Latest candidate suite39 passed1 failed due unsupported test access to `ElementwiseKernel.options`. Stopped after permitted correction/retest; next action documented in `0251_elementwise_fusion_blocker.md`.

The50t/s target remains unmet. These are fixed-prefix warm-cache forced-token measurements, not prolonged-chat emitted-token benchmarks. No tail skipping, new activation quantization or speculative acceptance is hidden in these measurements. Each numbered report records scope and limits.

## Latest measurement and cleanup — 0246

Grouped IQ public/integration validation is deduplicated without changing arithmetic. Actual GPU tests: **286 passed, 2 skipped**. Real-checkpoint forced-token pre/post logits and state hashes are bit-exact at prefixes1/2/4/8/16. Fresh runs vary 4.92–11.16 tok/s (profiled control8.55); same-process packed operator A/B shows **no convincing speedup** from this cleanup. The50 tok/s goal remains unmet. See `0246_grouped_validation.md`; do not attribute cold/warm timing differences to the change.

## Latest incremental optimization — 0245

Removed unused index-query projection/norm/RoPE only on the existing CuPy all-cell QSA fast path. Sparse selection, index history and diagnostic captures remain intact. Actual CuPy QSA tests: **8 passed**; CPU QSA/model/Qwen3-Next attention regressions: **101 passed, 1 skipped**. Synthetic production-dimension microbenchmark saved 0.025–0.169 ms per QSA position in four pairs, final outputs bit-exact. No full-model throughput or Strata precision-equivalence claim. See `0245_qsa_dead_query.md` and versioned JSON. Prior 0244 removes discarded output heads during prefill (27 passed, 2 skipped); 0243 measured Strata with model drafter off, suffix drafts and route-tail skip still active.

## Numbered patch ledger

| Patch | Objective / principal files | Status and evidence | Limitations / next dependency |
|---|---|---|---|
|0200 | Source/model compatibility audit; `reference_contract.md`, `tensor_inventory.{md,json}`, `gguf_metadata.json`, `memory_architecture.md`, `tests/test_0200_qwen4exp_contract.py` | Implemented but unvalidated provenance; complete1224-tensor header/operator inventory tested | Immutable checkpoint revision not embedded; payload hash not verified. Full model numerical evidence is separate |
|0201 | `mini_llm/external/gguf_store.py`, `tests/test_0201_gguf_store.py` | Implemented; header/ownership/range validation passes, including actual original namespace and >4GiB sparse offsets | No payload integrity hashing; little-endian contiguous GGML layouts only |
|0202 | `mini_llm/ops/quantized_weight.py`, `gguf_codebooks.{py,h}`, `tests/test_0202_iq_formats.py`, package-data and MIT notices | Numerically validated **CPU block decoding**: IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS/Q2_0 exactly match compiled reference | Not GPU kernel support; native packed execution0203 |
|0203 | `mini_llm/ops/native_packed_cpu.{py,cpp}`, `tests/test_0203_native_iq_experts.py` | Numerically validated **native packed CPU operators**; independent GGML lanes exact, synthetic experts maxabs3.8147e-6 | FP32 arithmetic even when returned dtype differs; throughput not benchmarked |
|0204 | `expert_source.py`, `expert_residency.py`, `quantized_linear.py`, `external/qwen4exp_gguf.py`, sparse tests and `scripts/probe_0204_qwen4exp_experts.py` | Implemented with numerically validated **CPU/supported-GPU subpaths**; all7 actual expert-format triples maxabs5.6617e-11; complete publication/in-flight tests | IQ GPU/top-10 execution now validated in0213; fixed slots/asynchronous weight mover not implemented. Existing synchronous sparse materializer retained |
|0205 | `external/qwen4exp_hyperconnection.py`, standalone tests | Implemented but unvalidated against linked full reference; synthetic FP64 math and actual CuPy tests pass | Boundary identity verified; first real HC-read difference explained by BF16 rounding; not full-model parity |
|0206 | `external/qwen4exp_ple.py`, standalone tests | Implemented but unvalidated as a full linked-reference block; compiled Strata hash parity, bounded row decoding and synthetic activation/history tests pass | Fixed target geometry, one sequence, sequential history; independent checkpoint PLE activation parity pending |
|0207 | `external/qwen4exp_mixer.py`, standalone tests | Numerically validated **representative checkpoint state subpaths**, including16-token identical-input/continuous state replay against pinned GGML | All-layer independent state parity pending; sigmoidZ differs deliberately from Qwen3-Next default |
|0208 | `external/qwen4exp_attention.py`, standalone tests | Numerically validated **representative short-text QSA state/output subpaths** over16 reference tokens; synthetic2061-cell sparse selection/state and compiled textRoPE/tie checks pass | Text scalar positions only; local libstdc++ tie membership, not portable STL promise; no multimodal/long-context performance claim |
|0209 | `external/qwen4exp_weights.py`, `qwen4exp_model.py`, combined graph tests | Implemented but unvalidated; actual CPU/GPU original-model forward succeeds | Boundary association verified;5–6% baseline reference logit discrepancy not fully attributed; no demonstrated graph defect |
|0210 | Generation state/lifecycle in model; `tests/test_0210_qwen4exp_generation.py`, reference probe/harness/artifacts | In progress; actual3 greedy decisions, reset checks and16-token full-model incremental/prefill active-state consistency pass | Reference logits maxabs~1, relativeL2~5–6%; no end-to-end numerical validation |
|0211 | Existing OpenAI serving registration/tokenizer/template/streaming integration | Not started | Gated on numerically validated model; no new HTTP stack |
|0212 | Initial profiling; `scripts/probe_0227_short_sequences_and_profile.py`, `0212_initial_profile.md` | In progress; measured eight CuPy decode positions with host spans/CUDA events | Exclusive critical path, SSD/page faults, overlap efficiency and global VRAM high-water outstanding; measured0213 speedups are separate |
|0213 | Packed GPU operators and switchable decode fusion; follow-up0234/0236/0237 | Numerically validated **packed GPU/operator subpaths**, real7-triple warp tests and original-model12.033tokens/s | Not70–80tokens/s; full reference discrepancy and longer cache-pressure behavior unresolved |
|0214 | Fixed-slot asynchronous expert mover; synchronous foundation0238 | In progress; capped arena, projection-view leases, complete publication and CPU fallback validated | Copy completion still synchronous; no staging/shadow async lifecycle, overlap or bounded mover implemented |
|0215 | Independent efficient prefill policy | Not started | Sequential correct-path candidate currently used, model parity unresolved |
|0216 | Adaptive routing/prefetch | Not started | Reuse existing profiler/cache seams later; no prediction-based omissions |
|0217 | CUDA graphs/long context; static HC experiment0235 | In progress, **HC-only prototype** numerically tested; no measured throughput benefit | Disabled by default; dynamic QSA/router/cache excluded, no whole-decode/long-context capture claim |

Each implemented patch has its own numbered document in this directory, with interfaces, tests, architecture and limitations. Numbers denote independently reviewable change groups, **not git commits**. No branch or commit was created. Qwen3-Next modules/training implementations were not rewritten. The legacy dense Llama format acceptance boundary was kept separate from the expanded CPU registry; old unsupported-format test sentinels and GPU capability parametrization were updated where necessary.

## Latest verified state — follow-ups0234–0238

- CPU inference/diagnostic plus selected training-component regressions: **1691 passed,780 skipped in17.89s**. Command: `MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_020*.py tests/test_021*.py tests/test_022*.py tests/test_023*.py tests/test_0106*.py tests/test_linear.py tests/test_rmsnorm.py tests/test_swiglu.py tests/test_moe_experts.py tests/test_moe_router.py tests/test_checkpoint.py`.
- Actual selected CuPy suite: **1031 passed,22 skipped,2 deselected in8.21s**. The same two previously documented NumPy-only cases below remain excluded; no entire-repository GPU claim.
- Follow-up rates: global codebooks9.174106; HC graph8.965350 (not recommended); group after admission10.197347; warp rows12.033481; fixed slots11.976041tokens/s. Historical variants are reused, not rerun. See `0229_decode_performance_v6.json` through `v10.json` and individual numbered documents.
- Warp scheduling real top10: median1.472287× kernel speedup; worst error3.176314e-7 relative L2 versus block kernels across all7 actual triples. Full optimized logits versus original FP32 baseline: max relativeL2=2.014602e-6, maxabs=2.717972e-5, all checked argmax/top10 sets match.
- Fixed-slot and dynamic-warp logits **and all active-state hashes are bit-exact** at prefixes1/2/4/8/16. `0233_fast_lifecycle_v2.json` additionally passes16-token prefill/incremental equality and repeated fresh greedy IDs `[220,100561,198]`.
- Slot arena:2,419×2,662,400-byte slots,6,440,345,600 physical bytes,4MiB staging. Extended lifecycle retains2,332 slots, no quarantines, sampled RSS11,180,748,800 bytes; default pool12,270,584,320 reserved bytes, not global VRAM high-water.

Next dependency: bounded asynchronous staging/shadow slots and transfer publication/overlap, followed by realistic longer cache-pressure measurements. No Q8 activation approximation, tail skipping, speculation or full expert expansion was introduced. HC graph replay is opt-in and not recommended for throughput at this granularity.70–80tokens/s remains an open target, not a completed milestone.

## Historical Patch0213 verified state

- CPU regression: `MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_020*.py tests/test_021*.py tests/test_022*.py tests/test_023*.py tests/test_0106*.py` — **1650 passed,654 skipped in19.35s**.
- Selected actual CuPy operators/residency/fusion regressions — **894 passed,20 skipped,2 deselected in8.20s**. The two previously documented NumPy-only tests are excluded under CuPy: `test_packed_gguf_experts_cpu_fallback_and_cache_integration` and `test_full_packed_expert_and_opt_out_cpu_reference`. The exploratory run including the latter failed on its NumPy/CuPy input mismatch; it was not hidden or fixed as unrelated work. Both pass in the CPU suite. No whole-repository GPU green claim.
- Real packed GPU probe covers one expert in all48 layers/all7 actual format triples and a routed top10 case; worst relativeL2=1.622820435e-6. Grouped real top10 outputs are bit-exact versus sequential GPU across all7 triples.
- Latest uninstrumented original-model forced-token rate: **8.566126tokens/s**, positions8–15 of fixed IDs1–16, including admissions/transfers there; ~31.39× baseline. Same-prefix logit relativeL2≤2.065315e-6, maxabs≤2.789498e-5, argmax/top10 sets match at prefixes1/2/4/8/16. This is not natural-language steady-state throughput or the70–80 target.
- `0233_fast_lifecycle_v1.json`: optimized16-token incremental/prefill logits and all active-state hashes identical; two fresh greedy requests return `[220,100561,198]`.13.02s child, sampled RSS11,173,183,488 bytes; pool10,616,590,848 reserved bytes, not total VRAM high-water.
- Model-memory admission now credits already committed static/cache/live-state bytes on replanning, without weakening initial/logical budget checks. Completed residency fences are reaped nonblockingly even below cache capacity.

Next dependency: measure and reduce routed-expert admission/transfer and launch overhead with bounded fixed slots; preserve all current oracles. See `0213_packed_gpu_and_decode_optimizations.md`. The latest separate profile is `0231_fast_profile_v2.json`; nested spans are not exclusive costs. No route-tail skipping, Q8 activation approximation, speculative decoding or full expert expansion was introduced.

## Historical verified states and commands

### Consolidated CPU inference/training-component regressions

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_BUILD_DIR=.pytest_cache/native_0203 \
OPENBLAS_NUM_THREADS=1 \
QWEN4EXP_LLAMA_SOURCE=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp \
.venv/bin/python -m pytest -q tests/test_020*.py \
  tests/test_0210_qwen4exp_generation.py tests/test_0106*.py \
  tests/test_linear.py tests/test_rmsnorm.py tests/test_swiglu.py \
  tests/test_moe_experts.py tests/test_moe_router.py tests/test_checkpoint.py
```

**1488 passed,184 skipped in29.52s**. This establishes the current new modules plus existing0106 suite and listed training primitives/checkpoint tests on CPU. It is not the whole repository training suite. Optional GPU/benchmarks/real-model opt-in cases skip as declared. No production code was changed after this verified run.

Earlier selected foundation run:481 passed,86 skipped. Subsequent operator/runtime selected run:466 passed,7 skipped. These overlap and must not be added to the1488 total.

### Targeted actual GPU kernels

Supported-format GPU GEMV and finite mixed-fallback tests: **24 passed,43 deselected** (six mixed cases plus18 GEMV cases exercising FP16/FP32 legacy kernels). New model/reference activation arithmetic is FP32; this legacy FP16 regression coverage is not a preference change. No new IQ GPU format is advertised.

A broader exploratory CuPy run identified six pre-existing backend-array-assumption cases that fail when CPU-oriented tests run under CuPy; they are listed in `0204_sparse_execution.md` and were left untouched. **The entire CuPy suite is not green/claimed green.** The18 CPU-only-format GPU-parametrization failures were fixed by testing GPU capability rather than the CPU registry; CPU format coverage remains intact.

### Original-checkpoint lifecycle on CPU and GPU

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_BUILD_DIR=.pytest_cache/native_0203 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
QWEN4EXP_MODEL_PATH=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/ \
.venv/bin/python -m pytest -q -s tests/test_0210_qwen4exp_generation.py
```

CPU: **1 passed in40.68s**. Repeat with `MINI_LLM_BACKEND=cupy`: **1 passed in30.06s**. Fixed IDs[1,2]; greedy IDs[220,100561,198] on both. Within each backend, prefill logits and fresh-request incremental logits match exactly. Six token forwards/test,2880 native CPU expert calls, zero decoded-expert cache bytes, zero GPU expert residents. This is self-consistency, **not independent correctness**.

Peak process RSS: CPU10,281,414,656 bytes (~9.58GiB); CuPy8,632,389,632 bytes (~8.04GiB). GPU pool snapshot4,414,358,016 used /4,420,487,680 reserved bytes (~4.12GiB); not global VRAM high-water. Owned-allocation planning excludes OS-controlled mmap residency, so the8GiB host budget is not an8GiB RSS limit. Hard RSS/page-cache control and long-context stress validation remain limitations.

### Independent reference — unresolved numerical result

Original pinned build's eager prefetch initially blocked safe loading. A **separate CPU-only source copy**, baseSHA171e8846b4af9766c354064cb776cb34a50f053f, changes only `init_mappings(true,...)` to `false`; its3643-file source comparison and exact patch are recorded. Original pinned source is still clean. No quantization/graph mathematics changed. The no-prefetch reference and production ran sequentially under a12GiB RSS cap; no duplicate resident model processes.

| Fixed input prefix | Max absolute logit difference | RelativeL2 | Actual/reference argmax | Top10 overlap |
|---|---:|---:|---|---:|
|[1] |0.958172 |0.061643 |271/271 |9/10 |
|[1,2] |1.006777 |0.050675 |220/220 |9/10 |

Max absolute log-probability differences:1.07785 and0.94804. FP32 stored logits; production native packed FP32 activations differ from GGML's quantized activation arithmetic. No broad acceptance tolerance was applied, and these metrics are **not a passing numerical gate**. No finite-difference step is involved. Reference peak RSS5.72GiB, production8.71GiB; combined probe27.16s.386 boundary comparisons were collected.

The old HC-attention comparison is superseded by the verified0218–0220 captures: reshaped aliases overwrote occurrence-based artifacts. Corrected first HC-read error is0.00102728 relativeL2, explained by BF16 projection-input rounding. No production correction was needed.

Artifacts: `0210_noprefetch_reference_validation.json`, `0210_noprefetch_reference.patch`, `0210_reference_diagnostics.md`. The earlier blocked preflight result is preserved separately, not overwritten as a success.

## Investigation outcome and next measured work

Known: packed decoders/native operators pass; original model runs with bounded owned allocations, mapped experts/PLE, deterministic selected combination and isolated states. Diagnostic misassociation is resolved; BF16/Q8 arithmetic differences and amplification are demonstrated. Representative identical-input DeltaNet/QSA states pass over16 tokens, and full-model local state consistency passes. Complete attribution of the5–6% reference logit discrepancy remains open; no incorrect production operation has been demonstrated.

**Update through investigations0224–0225:** indentation is fixed; state reanalysis passes with35 comparisons and exact PLE row IDs, without model reloads. Six identical-input MoE cases have matching top10 sets; teacher-forced down/combination agree at roughly1e-7 precision. A shared-expert quantization threshold amplifies1.47e-7 input error into5.50e-4 output error. Controlled full-forward perturbations demonstrate several-percent sensitivity. Most recent focused diagnostics plus0106 inference regressions:1295 passed,176 skipped. No production/reference math changed. Those historical next checks are superseded by0226–0227 below; full mathematical equivalence remains unclaimed.

**Update through0226–0227:** both independent block-state policies pass236 comparisons; all full-model active-state hashes/logits match at prefixes1/2/4/8/16. Latest consolidated command:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -m pytest -q \
  tests/test_020*.py tests/test_021*.py tests/test_022*.py tests/test_0106*.py
```

**1492 passed,186 skipped in16.75s** on the final rerun (earlier run:16.54s). This overlaps earlier totals; do not add them. Separate real-checkpoint block replays and sequential CuPy runs passed. New diagnostic files: `tests/fixtures/qwen4exp_block_state_reference.cpp`, `scripts/probe_0226_identical_block_states.py`, `scripts/probe_0227_short_sequences_and_profile.py` and their focused tests. No production source changed.

The short-text validation gate is passed. Initial profile: mean instrumented token3.6965s; routed expert span1.55152s, complete GDN1.36484s, complete QSA0.55624s. These include waiting/nested work, not exclusive critical-path costs. Next: separate projection fallback from mixer core overhead and validate packed mixed-format GPU expert kernels against retained oracles. Fixed-slot asynchronous movement and long-context validation remain pending.
