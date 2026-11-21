# 0326–0333 — Measured placement gain, exact router optimization, hot critical path

## Current conclusion

**50 complete decode forwards/s remains unmet.** The strongest new improvement is setup-only frequency placement within the existing sized GPU slots. It reduces observed CPU expert executions by about26% on the retained repeated trajectory without increasing the7GiBcache. An exact router variant provides a smaller gain. New CPU IQ2_S gather and isolated HC projection experiments do not justify integration/default changes.

Latest four-response controlled replay:

| Configuration | Response1 | Response2 | Response3 | Response4 | Aggregate all4 | Aggregate hot last3 |
|---|---:|---:|---:|---:|---:|---:|
| Static profile, original router |45.68824|46.55048|46.52938|46.16071|46.22953|46.41283|
| Same profile, exact warp-reduction router |44.63786|47.12285|47.19762|47.19923|46.51275|47.17321|

Rates are **post-prefill decode forwards/s**, not emitted-N/decode-second. Each response has294forced outputs/293decode forwards, after128warmup outputs; all4responses total1172decode forwards. Candidate first response incurred storage I/O; both configurations have no additional process physical reads/major faults after their first response. The hot comparison shows about1.64%higher observed rate, not a repeated-trial confidence claim. These runs are **not natural conversation, not growing conversation history and not configuration-equivalent to Strata's262144-context/75.2emitted-token/s run**.

Structured timings, raw denominators, current source hashes, native phase totals and operator reports: `0326_0333_results_v1.json`. Existing artifacts were preserved. Current source hashes describe the final tree, not reconstructed historical source identities.

## 0326 — Attribute file-page memory before changing residency

**Objective:** distinguish mapped checkpoint page residency, cgroup charges and private allocations.

Files:
- `scripts/run_0290_isolated_benchmark.py`, `tests/test_0290_isolated_benchmark.py`.
- `scripts/qwen4exp_memory_census.py`, `tests/test_0326_memory_census.py`.
- Opt-in boundary hooks in `scripts/probe_0254_strata_workload.py`, tested in `tests/test_0254_strata_workload.py`.

Contracts:
- `snapshot()` adds bounded optional `memory_stat` and `memory_swap_current`, with explicit errors rather than fabricated zeros. Raw memory.stat mixes byte quantities and event counters. Missing optional telemetry does not mask failures in existing safety-critical reads.
- No limit, reserve, pressure threshold, stop condition or swap policy changed.
- `checkpoint_census(store)` inspects descriptors and `mincore` bits without dereferencing payloads or creating tensor views. It validates owned read-only mappings and bounds metadata at4096tensors/64shards. Groups distinguish routed experts, the exact PLE table, PLE operators and dense/other weights. File-page residency is not cgroup ownership, process PTE residency or unique physical RAM.
- `MINI_LLM_MEMORY_CENSUS=1` journals setup/completed-request snapshots outside timed replay. No warming, pinning, discard or cache-advice policy is applied.

Results:
- Initial focused isolated selection: **249passed,61subtests passed** (`0326_cpu_tests_v1.xml`).
- Actual instrumented natural run `diagnostics_0326_memory_v1`: identical294-output responses, **34.68188/45.94144forwards/s**. Second response added no physical reads or major faults. This is a diagnostic run, not the later clean control.
- At model-ready, about31.66GBof expert payload was already resident globally, while the service charged about1.49GB. At completion, expert residency was36.53GB, service charge about7.22GB; peak charged memory7,242,723,328bytes. Census physical read deltas were zero.
- Therefore the earlier cold28GiBguard abort did not establish a hidden private expert arena or a memory leak. Reclaimable/shared file pages and their charging history materially change the experiment. The successful run is not a matched cold reproduction and cannot prove that a fresh full working set fits28GiB.
- The enabled PLE reader already uses bounded owned packed rows and a separate random-advised descriptor; no accumulating mapped-row cache was found. Native registrations retain canonical pointers/owners, not a second packed payload copy.

**Status:** diagnostics implemented and fixture-validated; actual mapping/cgroup distinction measured. No hard canonical-page working-set governor or full pinned arena was added. Next dependency was completed causal routing data.

## 0327 and0329 — Exact router selection/reduction variants

**Objective:** reduce exposed router synchronization before CPU/GPU expert work can be published.

Files:
- `mini_llm/ops/router_decode.py`.
- `scripts/probe_0327_router_selection.py`.
- `tests/test_0327_router_warp_selection.py`, `tests/test_0329_router_reductions.py`.

Contracts:
- `MINI_LLM_ROUTER_WARP_SELECT=1`: warp argmax preserves descending score/lower-ID tie order, NaNs last, all top10experts, full-softmax arithmetic and selected-mass normalization. Selection-loop block barriers fall90->20for top10.
- `MINI_LLM_ROUTER_WARP_REDUCE=1`: additionally preserves shared reductions at strides64/32 and the original FP32 operand tree for strides16..1using warp shuffles. No exp/division, normalization floor or selected-mass summation is changed.
- Explicit old/candidate overrides make the original kernel an environment-independent oracle. Original CUDA source and selection-only source remain available; variants have distinct device-cache entries. Both flags default off.
- Operator probe preallocates48-call graphs, warms compilation/capture outside timing, alternates variants and checks input preservation plus bitwise output parity. `--warp-reductions` selects the full candidate.

Results:
- Original plus selection-only GPU suite: **188passed** (`0327_gpu_tests_v1.xml`).
- Selection operator median16.1713->12.8636us/call, **1.2571x**.
- Full-reduction operator median15.9043->12.6386us/call, **1.2584x**; removing the remaining barriers did not create a substantially larger gain than selection alone.
- Current final GPU selection: **411passed,4collection-skipped,1deselected** (`0332_gpu_regression_v1.xml`). Includes actual CuPy router/profile/slot tests; the Qwen3-Next modules in this command are collection-skipped, not falsely reported as GPU model passes.
- Real-checkpoint control/candidate over two requests with IDs1..16: **all10saved FP32logit arrays bit-exact (maxabs0) and all persistent-state records equal**, at prefixes1/2/4/8/16. See `0332_router_model_parity_v1.json`.
- Parity uses identical empty GPU expert placement/complete native CPUQ8experts. It proves same-arithmetic router replay, not equivalence across different expert placements or a new independent llama.cpp result.

**Status:** numerically validated within these scopes, operator accelerated, modest end-to-end observed gain. Defaults remain off; no50/sclaim.

## 0328 — Setup-only sized-slot profile placement

**Objective:** exploit the materially larger placement opportunity without implementing an unproven asynchronous mover.

Files:
- `mini_llm/ops/expert_profile.py`, `mini_llm/ops/expert_residency.py`.
- `scripts/qwen4exp_profile_input.py`.
- `scripts/probe_0254_strata_workload.py`, `scripts/probe_0302_decode_configuration.py`.
- `tests/test_0328_sized_profile.py`, `tests/test_0328_profile_input.py`, wrapper contracts in `test_0324_strata_exact_launch.py`.

Evidence and contracts:
- Earlier8-swap/32MiBcausal policy saved only627/75864CPUmisses (~0.83%). That did not rule out a **setup-time full profile**: fixed-class frequency placement trained on completed responseA predicted56103misses for repeated responseB, down26.05%, with1278replacements/2,586,521,600incoming bytes and unchanged3673slots.
- `sized_frequency_placement()` ranks within existing256-byte-aligned physical classes and capacities. Equal heat prefers existing residents, then deterministic keys. It neither reallocates the arena nor assumes equal strides.
- `install_observed_profile(*, counts=None)` retains the uniform API and adds sized slots. None uses warmup observations; supplied counts are validated snapshots. All selected packed source geometry/formats are preflighted before eviction. Complete synchronous copies use existing immutable canonical sources, staging and leases.
- Installation requires no dispatch, pending CPU work, protected/in-flight consumers, frozen device plans, native service plans or host-source substitution. It is not a between-request asynchronous refresh API. Incomplete installation raises; no incomplete bundle is published, and missing experts retain CPU fallback.
- New/zero-score entries receive ordinary unit admission scores because the existing heap requires positive scores; no route hits are fabricated. Existing positive scores are preserved.
- `load_profile_counts()` reads/hashes at most64MiBof result JSON and selects an explicit completed case. Later case counts cannot affect the selected profile. Canonical path/size/mtime provenance must match current checkpoint stat provenance; this is explicitly not cryptographic checkpoint verification.
- Wrapper options: `--expert-profile-from`, `--profile-case-index`, `--tokens-from`, and existing `--warmup-frequency-placement`. Installation is after benchmark warmup, before freezing plans. Setup is untimed for decode but its wall time is reported. This is **not automatically wired into serving startup**.

Measured controlled pair, same128/294token IDs and7GiBcache:

| Configuration | Response1 | Response2 | CPUexperts/request | Hot native CPU elapsed/request |
|---|---:|---:|---:|---:|
| Existing placement |44.63088/s|45.02055/s|75864|2.74760s|
| Supplied static profile |46.42426/s|48.30164/s|56114|2.18635s|

Profile selected3673complete experts, retained2395, replaced1278; setup0.51367s. Class occupancies exactly match the existing physical capacities and7,515,878,400reserved bytes. No new host arena or per-token weight allocation was introduced.

Actual56114misses versus predicted56103reflect changed CPU/GPU execution-site rounding and resulting routes: forcing the same token trajectory does not force identical hidden activations or router selections. Checkpoint bytes/operations remain intact, but cross-placement logits are **not claimed bit-exact or independently quality-accepted by this benchmark**. The profile was trained on an earlier occurrence of this same prompt/response, so broader conversation generalization is unmeasured.

A counts-only heat-per-byte layout experiment predicted only549additional saved misses beyond the fixed-class profile. That is insufficient evidence to redesign the physical slot layout.

Tests:
- Final actual mixed-size GPU profile/slot tests **41passed** (`0328_gpu_tests_v3.xml`); exact packed bytes, same-site outputs, canonical immutability, budgets, guarded idle seams and explicit failure handling.
- Final current CPU regression selection **463passed,12skipped,1deselected,166subtests passed** (`0332_cpu_regression_v1.xml`), including Qwen3-Next GGUF/mixer/attention/decoder/model coverage.

**Status:** setup placement implemented/operator-validated and benchmarked on controlled replay. Not a general adaptive mover, natural-conversation quality result or default serving policy. Next dependency: hot critical-path attribution and independent/generalizing profile evaluation.

## 0330 — Hot-response capture and native phase attribution

Files: `scripts/probe_0266_critical_path.py`, wrapper0302, `tests/test_0330_hot_trace.py`.

- `--capture-response 1|2` retains the original default; response2runs warmup and the first response untouched, then installs/restores wrappers only for the selected response.
- Profile/forced-token source options propagate to the existing runner; setup remains before frozen plans.
- `capture_metadata.json` records response/case index,16expected/captured anchors, completeness and `rate_claim=false`. NVTX names and monotonic anchors remain consistent. Incomplete captures are not benchmarks.
- Wrapper exposes `--trace-response 2` and `--native-timeline`; default native timestamps remain off. CPU backtrace sampling/context-switch capture were unavailable in Nsight; native timestamps are separately recorded.
- Harness selection/restoration/isolation contracts: **51passed,6subtests passed** including wrapper tests.

### Successful capture

`diagnostics_0330_trace_v2`: exact16-token window from warmed response2, positions64..79. No cache/host-limit reduction was used. Profiler execution is diagnostic, not clean throughput. Artifacts:
- `diagnostics_0330_trace_v2_trace.nsys-rep`
- `0330_trace_v2.sqlite`
- `0330_critical_path_v2.json`
- Native48-layer timelines, explicit private graph warmup ordinals, NVTX/perf-counter anchors in the capture directory.

GPU span343.892ms over16tokens; non-wait GPU operation union220.367ms; summed wait kernels82.060ms. Summed selected kernel times/token:

| Family | ms/token |
|---|---:|
| `q8_linear`,301calls |4.99245|
| HCnorm/down/up,97calls each |2.39708|
| Resident expert gate/up + down |1.84263|
| Gated DeltaNet update |0.75693|
| Original router |0.74185|
| Dense/control vector matmul |0.60179|
| QSAdecode + warp dots |0.37037|
| GPUwait for CPU,48calls |5.12874|

Categories overlap CPU work and are not additive CPU+GPU latency. GPU gaps include graph/host scheduling, copies and dependencies; they are not all removable downtime. The large wait-inclusive memcpy API durations do not establish a physical PCIe bandwidth bottleneck.

### Native-only hot phase run

`diagnostics_0330_native_v1`: exclude explicit private ordinals, validate740production calls/layer, select final293decode calls after the second77-token prompt.14064handoffs,1771with no CPU experts, all recorded error codes zero.

| Native phase | ms/decode step |
|---|---:|
| Preparation |0.11018|
| Gate/up |3.60233|
| Hidden quantization |0.18370|
| Down projection |1.64417|
| Completion tail |0.24915|
| Total |5.78952|

These are native wall-phase durations, including scheduling; not pure arithmetic or an additive GPU critical budget. IQ3_XXSgate/up contributes~1.313ms andIQ2_Sgate/up~1.273ms/token, more than IQ3_S~0.941ms. This motivated0331rather than another blind worker/spin sweep.

An exploratory timestamp join finds16rows/layer matching768publication/wait kernels. Native starts are typically within~1usof GPUpublication completion. This argues against a large ordinary dispatch-copy delay, but exceptional completion gaps and wide RangePush timing brackets require care. No untested wait/fence removal followed from this analysis.

**Status:** diagnostic harness validated, hot GPU/native captures obtained. Next high-value area is reducing actual packed projection/gate-up work while preserving arithmetic and measuring overlap, not assuming a PCIe bottleneck.

## 0331 — IQ2_S AVX2 gather: exact, performance-rejected

Files:
- `mini_llm/ops/iq2_cpu_simd.h`, included/hashed by `ggml_packed_cpu.cpp`/`.py`.
- `tests/test_0331_iq2_cpu_gather.py`; existing real-operator probe0297gains `--kernel iq2_gather`.

`MINI_LLM_GGML_IQ2_GATHER=1` / `GGMLPackedCPUExecutor(iq2_gather=True)` replaces scalar codebook assembly with AVX2four-way64-bit gathers. The canonical82-byte IQ2_Sblock, sign/nibble scales, GGMLQ8_Kproducer, integer lane grouping, block FMA, horizontal reduction and final1/8factor remain unchanged. No expert expansion occurs; the extra8KiBtable is a codebook, not weights. Original GGML path remains the default. CPU support is explicitly checked.

**60tests passed,2skipped** across new IQ2tests, existing IQ3tests and direct GGML-oracle tests (`0331_cpu_tests_v1.xml`). Finite outputs are bit-exact, including unaligned read-only weights, scale extremes, mixed down formats, ten-expert batches and parallel ownership.

Actual checkpoint IQ2_S/IQ2_S/{Q2_0,IQ4_NL}experts with1/2/5/10selected experts are all bit-exact, but old/new speedup ratios0.7285–0.9401mean the candidate is slower. **Leave off; no full-model benchmark or default promotion.** GGML MIT attribution/license is retained in the derivative header; pinned sources were not modified.

## 0332 — Longer combined replay and numerical gate

- `diagnostics_0332_combined_v1` failed device admission before a complete result: need2,217,778,688/free2,117,337,088bytes. User reported a competing process and requested relaunch.
- Fresh preflight found946MiBdevice use and only the editor in the compute-process list. The unchanged relaunch `combined_v2` completed all4responses. `control_v1` then repeated the same4response trajectory with original router. The table at the top records exact rates; latest source hashes and raw timing fields are retained in the structured ledger.
- Candidate aggregate46.51275/s versus control46.22953/s is only~0.61%; hot last3comparison47.17321vs46.41283/s is~1.64%. First-request I/O differed. These are sequential observations, not confidence intervals or a50/sresult.
- Candidate charged peak2,341,945,344bytes, host availability at least55,174,262,784bytes, high/max/OOMevents0 and service swap0. These charges do not include all globally shared page-cache ownership.
- Two32-token fixed-prefix router parity runs completed; ten saved logits and all recurrence/convolution/QSA/PLE state records are exact. The optional IQ2gather remained off.

## 0333 — Isolate existing HCdown projection before adding a flag

No production integration was made. A synthetic97-distinct-bundle BF16project benchmark uses643,645,440weight bytes and alternates captured old versus existing warp-down kernels. This isolates the down/injection component rather than repeating the neutral fullparallelHCmode.

Final test is bit-exact. Median11.16261->10.82638us/call, nominal1.0311x, with substantial variation. The projected saving over97calls is only~0.033ms/token; **not sufficient to justify another integration/full-model sweep**. First probe attempt used the legacy default stream and capture refused; the localized correction used an explicit nonblocking stream after retiring initial copies. Both artifacts are preserved.

## Preserved failures and safety boundaries

-0328initial GPU profile tests revealed zero admission scores reaching the heap's `log2(score)`. Final localized score initialization handles new/zero-score entries; final41tests pass. Earlier failures remain in v1/v2XMLs.
- The old0324“no model imports”test assumed a pristine shared test process. It now checks that `load_workload()`adds no forbidden imports, rather than failing because other suites previously imported a backend.
- One CPU regression selection exceeded its small1GiBtest MemoryHighwhile including real installed-tokenizer/header preflight. The subsequent selected suite explicitly excludes that test. It is not a hidden model failure; the complete0326selection had previously run it successfully.
- `test_0106a_expert_residency::test_executor_and_sparse_dispatch_match_existing_fully_resident_path` supplies NumPy route IDs to a CuPy bincount and fails in GPU mode. It remains unmodified and explicitly deselected in GPU commands; its CPU execution is included/passing. Four Qwen3-Next GPU modules are collection-skipped. No whole-training-suite claim.
-0330tracev1failed second-state VRAM admission (need2,236,387,328/free2,203,582,464bytes) and produced no report/capture. It also recorded transient cgroup high events; noOOMkill. This was not conclusively attributable to profiler overhead alone. After the competing process was removed and freeVRAMincreased, identical v2completed without lowering headroom.
- All full-model attempts retained28/32GiBhigh/hard,12GiBhostreserve and swap0. Tests/operators used smaller explicit isolated-service budgets. No global cache dropping, checkpoint writes, checkpoint download, full anonymous/pinned expert complement, expert-tail omission or concurrent model processes were introduced.

## Next dependency

1. Preserve the successful profile path as the current high-value candidate, not an automatically enabled serving default. Validate natural conversations and profile generalization; repeated forced trajectories do not establish them.
2. Use the successful0330hot trace, not the failedv1or an older cold trace. Rank exposed `q8_linear` projections and CPUgate/up work by format and dependency position. The IQ2gather result rules out that particular implementation; it does not rule out all native arithmetic improvements.
3. Investigate reuse/fusion of shared-input packed dense projections with a separately measurable operator prototype.301packed dense calls/~5msand397activation quantizers are a concrete target; prove savings before changing the recurrent/QSA graph. Preserve quantizer partitions/reduction order or explicitly validate any Strata-aligned arithmetic difference.
4. Keep rejected/neutral variants off. Neither a larger host copy, a few cache swaps, additional spinning nor deleting correctness checks is supported as the next high-return change.

No optimized50tokens/sor complete Strata-equivalence claim is made.
