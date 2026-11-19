# 0258 — Pinned Strata execution audit and adaptation ledger

## Authority and scope

Source: **Niko1221/Strata `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`**, the source SHA previously resolved from installed build `712fec6b4297+cuda13.0`. The old local checkout `c775728...` is not the authority. Source URLs below are immutable. Sources were retrieved through the fetch tool; a terminal attempt to save a larger snapshot timed out and produced no source files. Do not claim a complete local checkout or fabricated source hashes. Three delegated audits reached usage limits without writing reports; this consolidated audit records the directly inspected contracts instead.

Base URL for relative paths: `https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/`.

Directly inspected in this audit: `include/strata/kernels/cpu/pool.hpp`, executable `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/native_expert.hpp`, `include/strata/core/expert_cache.hpp`, `include/strata/core/expert_source.hpp`, `include/strata/core/session.hpp`. Prior0252/0253 audited native GPU Q8 and dot source against the same SHA; prior layer audit inspected `src/core/layer.cpp`. Interface contracts below are distinguished from executable-body verification. Some header histories describe obsolete stages: e.g. early expert-source commentary says all hits run on CPU, while later `TokenHits`/dispatch contracts explicitly execute resident experts on GPU. Do not infer the active path from an old comment or quote historical measurements as this workstation's results.

## 1. CPU pool — executable source inspected

[pool.hpp](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/include/strata/kernels/cpu/pool.hpp), [pool.cpp](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/src/kernels/cpu/pool.cpp): `ExpertPool::worker`, `claim`, `begin_batch`, `run_phase`, `drain`, `run_split_multi_native`.

- Persistent native threads; worker-local scratch created at setup, not per expert.
- One read-only quantized input per layer/token, shared by selected experts. Pointer identity is NOT a cache key: the same staging address carries different activations every layer.
- Native gate/up phase distributes ranges over **all selected experts' rows**, not one entire expert per worker. This keeps cores available when only1–3 CPU experts miss.
- After gate/up, quantize each expert's SwiGLU hidden; then a down-row phase uses all workers again. No router weighting inside CPU arithmetic.
- Epoch-tagged claims prevent a late worker from stealing a subsequent batch. Completion and parked-worker protocol protects description reuse. Release/acquire publication is a correctness requirement, not an optional micro-optimization.
- Cache-line-separated atomics; no locked diagnostic increment inside the spin loop. Workers spin briefly and then sleep; publication coordinates with sleepers to avoid lost wakeups. Close signals stop and joins threads.
- Worker placement is by physical cores with hybrid P-core awareness; host-core reservation is enforced, not just assumed. Host can drain tasks while synchronously waiting. These placement policies need local measurement, not blindly copying a worker count.

**MyLocal gap:** `ThreadPoolExecutor` futures per missed expert; `GGMLPackedCPUExecutor.expert` allocates per-call vectors and requantizes the same input for each expert; fewer misses than workers leaves cores idle. The existing shared FP32 activation D2H is useful but does not implement shared Q8 preparation or row-level scheduling.

**Adaptation0259:** a bounded persistent native pool, shared per-format input quantization, gate/up -> hidden quant -> down row phases, asynchronous submit + explicit completion, immutable weight ownership until completion. Keep generic GGML arithmetic and router weighting outside. First version need not replicate Strata's exact spin policy or host-draining implementation; document deviations and benchmark them.

## 2. CPU formats and a specific remaining kernel gap

[native_expert.hpp](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/include/strata/kernels/cpu/native_expert.hpp): `NativeFmt`, `native_quant_act`, `native_quant_h`, `native_gu_rows`, `native_down_rows`.

Per-weight `vec_dot_type` selects CPU activation format. It is not universally GPU Q8_1. Multiple-token arithmetic can differ from singleton kernels; `WindowInvariant` deliberately selects invariant paths where required.

Executable `ExpertPool::drain` mode5/6 specifically dispatches Q2_0 through `q2_rows_any`; the source explains that generic ggml-cpu Q2_0 is scalar on x86. Hidden preparation for that path uses `act_quant_any`, not an assumed generic Q8_0 interchange. **This is an explicit gap in0256**, which calls generic GGML Q2_0. Exact specialized quantization/scaling and vector arithmetic must be audited/tested before porting; do not label a generic integer dot as matching Strata's fastest kernel.

**Adaptation dependency:** independently validate specialized Q2_0 quantization/rows and CPU ISA dispatch, retaining the generic GGML oracle and supported fallback. Not implemented merely by adding a pool.

## 3. Expert storage, profile and sized slots — interface contracts inspected

[expert_cache.hpp](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/include/strata/core/expert_cache.hpp): `read_expert_profile`, `open_sized`, `slot_stride`, `replace`, `fill_slot_queued_parts`, `sync_queued`, `verify_slot`.

- Profile ranks `(layer,expert)` pairs for startup placement; compulsory arrival-order fill is not equivalent.
- `open_sized` packs slots with per-layer blob sizes,256-byte alignment and explicit offsets. Replacements stay in a compatible same-layer slot. MyLocal's single worst-case stride wastes capacity relative to this layout.
- Storage/residency table is separate from adaptation policy. `replace` publishes a same-layer replacement only after its copy completes.
- Startup can queue many blob-part copies then synchronize once; byte verification validates the bytes, not only the index.
- Device consumers and transfer sources require stable ownership. Fixed addresses enable graphs; transient views must not outlive reused staging.

**MyLocal gap:** worst-case-stride arena, on-demand synchronous promotion, one staging slab, no profile-driven prefill. A single `.synchronize()` cannot simply be deleted: it currently protects staging reuse and complete publication.

**Required staged adaptations:** validated STRP profile loader -> per-layer/sized slot planner -> startup profile fill -> bounded periodic heat adaptation -> multiple protected staging/shadow buffers -> event-driven publication. CPU fallback remains valid until a complete new bundle is published. Avoid implementing all these inside a routing callback.

## 4. Host source, complement and transfers — interface contracts inspected

[expert_source.hpp](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/include/strata/core/expert_source.hpp): `ExpertSource::blob_stable`, `blob_parts`, `note_async_read`, `pinned`, `device_alias`; `FileExpertSource::pin_cache_complement`, `stage_exchange`, `commit_copies`, `commit_flip`; `GpuPlanSink`, `PcieModel`.

- Stable canonical sources and transient staging are distinct. `note_async_read` prevents staging reuse while DMA still reads it.
- Compact resident host storage primarily holds the GPU cache complement, optionally adding prompt-lend slots, rather than duplicating every GPU-held expert in pinned RAM. Pacing and physical/cgroup/commit headroom guard large registrations.
- An exchange stages the evicted expert before overwriting its GPU slot; old/new host overrides remain valid until copies complete. Host-copy movement and residency-table publication are separate commit phases.
- GPU-resident hits and an optional PCIe-served share are planned separately from CPU misses. The share may balance measured CPU and GPU costs; transfer work competes for host DRAM bandwidth.
- File-tier zero-copy/prefetch preserves exact bytes and uses bounded queues. Prediction warms storage; it must not alter routing.

**MyLocal gap:** warmed reclaimable pages are not a pinned complement; every miss currently chooses promotion or CPU rather than a bounded transient GPU execution tier. Full warming alone did not improve the observed0257 warmup, so do not promise pinning alone solves latency.

**Required adaptations:** plan/measure host complement without an unverified second full copy; stable-source handles; bounded PCIe staging; events and quarantine; only then cost-based splitting. All must honor16GiB VRAM/64GiB RAM and preserve mapped low-memory mode.

## 5. Decode launch structure and state — contracts inspected, no new whole-window port yet

[session.hpp](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/include/strata/core/session.hpp): `SessionState`, `SessionGraphs`, `SessionLoopScratch`, `TokenGraph`, `TokenHits`, `session_capture_token`, `session_run_token`.

- Per-layer pre/post graphs are split at routing/combination: CPU expert output must be combined with **that layer's** weights, never the following layer's.
- Later whole-token graph interfaces support device-side resident-hit selection. Early comments saying no resident tier in the token graph are historical, not the later contract.
- Positions/counts arrive through fixed-address runtime buffers; capturing Python scalar positions would freeze state evolution.
- Mapped pinned doorbells publish activation/IDs before CPU work; readiness flags permit the captured GPU to wait for its actual dependencies. The serial residual chain limits overlap: one cannot run an unrelated next layer early merely to hide CPU work.
- Shared GPU expert and resident-hit work are the legitimate overlap window. Session-owned scratch, event reuse and stable pointers avoid token-path allocation/free and its implicit synchronizations.
- Diagnostic sync/timing modes are separate from throughput; event spans containing host gaps are not pure kernel cost. Prefix timings need restored state or identical inputs.

**MyLocal gap:** Python dispatch with host reads at routing and many ordinary launches; existing `CapturedArrayInference` copies mutable state in/out and is not automatically a safe whole-model capture. QSA dynamic positions, PLE host I/O, residency changes and callbacks require explicit seams.

**Required adaptations:** persistent input/output scratch -> explicit native CPU submit/wait -> dynamic device-side metadata and state counters -> capture compatible pre/post segments -> device routing plans -> whole-window capture only after independent state parity. No graph may capture transient expert addresses or stale token positions.

## 6. Precision, prefill, PLE, KV and speculation ledger

- GPU IQ path uses Q8_1 including half scale and specified rounding/integer dot chains;0252/0253 preserve that contract at isolated operators. CPU Q8_K/Q8_0 are distinct. Remaining GPU IQ formats and down projections need exact tests before model-wide enablement.
- Native BF16 projections may consume FP32 activations; BF16 weights do not imply blanket BF16 activation rounding. Preserve normalization/reduction precision at each operator, not a global FP16 default.
- Prefill groups tokens per expert and uses separate chunk/ring policies. Correct but sequential MyLocal prefill cannot be compared with Strata's chunked throughput.
- PLE indexed I/O and history, QSA selections/positions and recurrence updates remain model correctness boundaries. Disk-backed PLE must not become a whole-table RAM allocation.
- INT8 KV and long-context residency need their own numerical/memory gates before claiming capacity parity.
- Suffix/MTP/DFlash speculation and conditional tail omissions contribute to emitted-token throughput. Verified speculation can preserve a chosen target evaluator, while tail omission changes the target graph. Neither is silently equated to current full-top10 single-forward throughput.

## Adaptation progress — 0259

`0259_native_row_pool.md` records the first implemented adaptation: bounded persistent native row workers, one shared per-format quantized input, gate/up -> hidden quantization -> down phases and explicit submit/drain ownership. It preserves generic GGML arithmetic, not Strata's specializedQ2_0 kernel or spin/affinity policy. Opt-in CPU and CuPy integration tests pass; retained warm long response8.20971 ->8.72394forwards/s with identical427tokens.50t/s not reached.

The same captures show46,591promotions and94.914GB synchronous packed transfers in the response case including prefill. This supports prioritizing bounded profile/heat-driven placement and protected staging/publication next; it does not prove exclusive transfer latency or justify deleting fences. Other audit items remain staged dependencies, not completed features.

## Adaptation progress — 0260–0264

0260separates admission from routing andbounds synchronousmovement; frozenresponsepromotionszero,10.31/s, so traffic alone doesnot explain50/s gap.0261directlyrereadsStrataexpert/native/Q2AVX2source: ActQlegacyFP32scale/halfaway differs fromGGMLQ8_0. Originalopt-inAVX2dot retainscurrentGGMLcontractbit-exact instead of silentlyimportingthatprecision.0262sharedGPUbranch nowmay overlap routedCPUwork with explicitfences, butsinglelongpairshowedno materialgain.0263currentpathprofile identifies repeatedsourceconstruction;0264retainscanonicalreadonlybundlehandles andclears onclose. Longlatest11.42412forwards/s, not50/s. Seeindividualnumberedreports andaggregateJSON.

RemainingStratamechanisms are notclaimedimplemented: profile/sizedslots, protectedasynchronousmover/hostcomplement, device-nativeexecutionplan andcapturedpre/postgraphs,all-formatGPUQ8, efficientprefill,INT8KV andsuffixspeculation.50/s requires measuring these structuralchanges, not extrapolatingoperatorratios.

## Adaptation progress — 0265–0268

0265captures the recurrent GPU segment with the existing Triton FP32 GEMV and fused recurrence; same-arithmetic full-state replay parity established over two16-token requests.0266profiles the current long-response path, rather than extrapolating old operator timings: heavy driver launches and exposed CPU completion waits remain.0267adapts Strata's bounded spin-before-sleep idea while preserving all-worker barriers, improving an identical273-token run to15.21239forwards/s; default sleeping remains.

0268implements a frozen device residency/route plan with persistent pointer tables, pinned slabs and GPU ordered combination. It is **not validated**: retest94passed1skipped2failed, possible block/warp oracle mismatch; no real model benchmark or default enablement. Source/ownership design is implemented, not proven correct merely by architecture resemblance. See `0268_device_plan_blocker.md`.50t/s remains unmet and correction-cycle stop rules apply.

## Validation and rollout rule

Each adaptation gets a numbered patch and independently switchable tests. Report operator/layer/state/model scopes separately, with matched long sampled requests and the retained FP32/GGML references. Default paths are preserved until their replacement is supported by numerical, ownership and memory evidence. This audit is a reproducible implementation map, **not a claim that all listed elements are implemented**, nor evidence of50t/s.
