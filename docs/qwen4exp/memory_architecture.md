# Qwen4Exp memory and execution architecture

## Current — 0318–0323 request retirement and whole-token ownership

0321 fixes a demonstrated replacement-state admission edge: finished-request graphs retained bound arrays until the next replay. Only when no weakly registered request states remain live, under the model execution lock, graph readers retire and unused allocator blocks are returned **before** new-state memory planning. Live-state buffers, admission headroom and failure checks are not bypassed. The previously failed partial run and successful two-response rerun are retained.

Whole-token0322 optionally borrows frozen native plans and immutable packed sources, binds recurrent/QSA state/status, copies runtime positions/RoPE and read-only PLE history, and returns owned logits/newPLEhistory. Old subgraph pools retire before allocation; the private pool caps256MiB (latest long reserved179653120bytes). Bound orphaned allocations and token-pool reservations are included in planning and retired before native handles. No giant weight or PLE copy. Mode remains experimental/off and not quality accepted relative to older PLE arithmetic.

RAM preflight refusals occurred as desktop/editor usage grew. Subsequent model runs used **stricter30/34 then28/32GiBhigh/hard caps**, with unchanged12GiBhostreserve, swap0 and7GiBGPUexpertcache. Cold/pressure differences must be retained in comparisons. No model/API/benchmark server should be running at the restart checkpoint; verify read-only after reboot. Authoritative guards, memory scopes and final source hashes: `0318_0323_execution_results_v1.json`; handoff: `RESTART_HANDOFF_0323.md`.

## Prior — 0310–0317 bounded I/O and complete-block graph ownership

- Optional `PackedRowReader` keeps a separate read-only descriptor, at most8I/O workers/16pending unique row reads per serialized batch, and an LRU of at most65536immutable90-byte PLE rows:5,898,240payloadbytes plus bounded Python overhead. `POSIX_FADV_RANDOM` reduces read-ahead; this is **not O_DIRECT** and OS page caching remains possible. No table-wide copy or full unquantized expert expansion. Reader workers/descriptors retire before model/store cleanup. Original mapped decoding remains available/default.
- Complete recurrent block graphs borrow frozen native plans, canonical sources and protected GPU expert leases; warmup touches private state, request pointer changes recapture, and outside-capture retirement events protect graph teardown. Adjacent chains cap length3 and reserve16MiB/member. They do not silently change cache placement or expert arithmetic.
- Complete QSA block graphs use runtime position and RoPE-factor inputs, bound request-owned KV/raw/pool/status buffers, conditional finite-validated commits, and8MiBprivate graph pools. Model eligibility limits cache payload to6MiB and all-cell-equivalent contexts; larger/sparse contexts retain the original path. Private warmup never updates caller caches. Bound orphaned QSA allocations and graph pools are included in planning; close graphs before native plan/service retirement.
- Cooperative QSA uses bounded FP32 partial scratch (up to64parts at n8192), not a persistent KV copy. It remains opt-in/not quality accepted. No switch silently changes checkpoint quantization or native CPU Q8 contracts.
- Model benchmarks remain sequential under0290verified32/36GiBhigh/hard limits, swap0 and12GiBhostreserve. A7.25GiBlocal-cache attempt failed VRAM headroom before tokens and was retained; subsequent comparisons explicitly use7GiB. No safety margin was reduced, full complement pinned, or unrelated server left running. Fresh Strata uses bounded mmap mode rather than the older~43GiBpinned configuration.

ProcessfileRSS, chargedcgrouppeak, finalCuPupoolbytes and device-widepeak are separate scopes. Use `0310_0317_results_v2.json` guards rather than systemd's small launcher peak. Capture-time device peak is unavailable. Full interface/lifetime/benchmark limitations: `0310_0317_strata_comparison.md`.

## Prior — 0308–0309 page-source diagnosis

`probe_0308_page_residency.py` queries owned live read-only mappings without touching payloads, bounded8192page-statusbytes/syscall. It counts all1224storetensors, weights partial tensor-edge pages and recognizes `per_layer_token_embd.weight` asPLE. Latest canonical routed50.292GB/35.410GBresident; PLE28.866GB/~0.697GBresident. Nonresident canonical experts may already have GPUcopies; this is not the required CPUcomplement missing-byte count. Privacy-redacted/unowned-file census is rejected.

0309approved4GiBanonymous source contains2111complete warmup-selected bundles, keeps canonical mmapindependent and substitutes CPUregistry views only; no expert pinning, GPUstaging or PLEcopy. Chargedservicepeak8.56GiB/anonymousRSS5.16GB, unchanged32/36GiBmodellimits/12GiBhostreserve. Natural32.52/sdoes not demonstrate a gain, so no larger source is authorized by this result. Census initially hit768MiBMemoryHigh duringmetadata setup; successfulmetadata-onlyrun used1792/2048MiBbounds, not relaxed model guards. Full report/limitations: `0308_0309_storage_residency.md`.

## Prior — 0293–0307 ownership and bounded execution additions

- Canonical mmap bytes remain immutable and independent of optional staged/registered copies. Integrated staging owns one bounded16MiBhost+16MiBVRAM arena with two complete8MiBslots; service-thread CUDA DMA was rejected after a captured scheduling deadlock. GPU kernels pull complete published bundles instead. Source-pinned staging never silently stages an unregistered expert; CPU fallback remains complete. Staging remains experimental/off.
- Request-bound recurrent graph operands retain exact buffer owners and recapture on pointer changes. Warmup stays private. Explicit retirement fences capture/replay readers; the planner credits actually orphaned bound allocations without weakening reservations or headroom. Stable state binding removes repeated recurrence copies, not state updates.
- Optional native GPU shared branch lives inside the handoff graph. Branch identity is latched; graph and owner retire before service/native source handles. CPU publication precedes shared GPU work. No CUDA/Python enters the native service thread, and no per-token weight allocation is introduced.
- Optional kernel input publication aliases the **existing** pinned activation/metadata owners. Every GPU writer system-fences before a CTA barrier and system-release epoch publication. CPU acquire/completion/failure contracts are unchanged. Failure suppresses new publication and incomplete-output reads. Mapped aliases do not copy expert banks.
- Shared IQ GPU codebooks add at most8KiB/CTA shared memory, not persistent model weights; module-cache keys distinguish modes. Original packed bytes, Q8_1, integer truncation and FP32 reduction remain unchanged. Inactive complete expert blocks skip unused uploads. CPU Q8_K/Q8_0 remains a separate contract.
- 7.25GiB sized cache admits3804experts versus2823in earlier7GiBuniform slots. 7.5/8GiB/profiler admission failures are retained; no safety margin was reduced. Current CuPy pool used~11.49GiB is a pool snapshot, not total device peak.

All actual experiments remain sequential inside0290 app.slice services (typically32/36GiBhigh/hard, swap0,12GiBhostreserve), without bulk50GiBwarming/full-complement pinning. Warm shared-page process fileRSS~36–38GiB is **not** charged service memory; source report preserves both scopes. New natural runs still fault/read1.85–3.05GB per response interval despite abundant host-available memory; the exact page sources/cgroup/cache dynamics are unresolved, so a blind large RAM copy is not a justified correction.

Contracts, measured limits, tests, source hashes and next attribution experiment: `0295_0307_critical_path_optimization.md`, `0295_0307_optimization_results_v2.json`.

## 0291–0292 bounded residency measurement after the safety gate

User-authorized sequential isolated benchmarks resumed after0290. A128MiB-capped actual expert copy passes byte/CPUoutput parity and releases its anonymous source. A512MiB partial source (536436736bytes/263bundles) successfully runs the actual model; CPU registry uses all263copies while canonical mmap remains authoritative. No material warm throughput gain (~33.91 versus33.87/s), so do not increase residency merely assuming more anonymous RAM speeds arithmetic. Full44GiB complement remains unvalidated.

Standalone0292stage-split probe owns at most32MiBpinned complete expert data per format group, one reusable GPUarena, dedicated transferstream/event; CPUmisses use the existing19workerpool concurrently.56same-siteoraclecases pass. This is not yet an integrated model mover. Small bounded pinning has resumed; unbounded/full-source pinning has not.

Longer model runs use32GiBhigh/36GiBhard/swap0/12GiBhostreserve and no50GiBbulk warmup. Cold peak32.003GiBincurred reclaim; no OOM events or guard abort. Shared cached pages can yield~40GiBprocessfileRSS yet~1.67GiBchargedservicepeak on later warm runs: report these scopes separately, not as a bypass of the kernel limits. GPU pool snapshots remain~11.95GBused/~11.96GBreserved, not device-wide peaks. See `0291_0292_residency_and_miss_split.md` for full safety/measurement limits.

## 0290 incident and mandatory diagnostic isolation

Large-memory/GPU/pinning experiments were initially paused after the2026-11-19 desktop-session OOM kill; subsequent bounded resumption is recorded above. systemd-oomd reported the GNOME Shell cgroup at50.6G; the responsible individual process is unknown. Process groups and sampled RssAnon are not memory containment.

`scripts/run_0290_isolated_benchmark.py` creates a separate app.slice user service with verified MemoryHigh/MemoryMax, swap0, group OOM killing and whole-service cleanup before workload execution. Default high/max768MiB/1GiB, minimum12GiB host reserve; PSI, finite ancestor headroom, charged file cache/current/peak and OOM events are monitored. The0254 workload CLI refuses unisolated execution. These limits protect against uncontrolled workload growth, not arbitrary sibling pressure or GPU-driver failures; there is no VRAM governor in this launcher. See `0290_memory_incident_and_isolation.md` for exact limits, small-test measurements and remaining risks.

0289's optional compact anonymous host source is small-fixture validated only. Canonical GGUF mappings remain immutable; complete RAM copies shadow CPU reads and are retired after native plans. PLE is not copied. Copying a large complement may transiently coexist with source file-cache pages and is **not approved by a logical host budget alone**. No44GiB model/pinning experiment or0289speedup has been measured. Resume only with a small isolated CPU source, not a full complement.

## Invariants

Canonical weights live in independent read-only shard mappings. Storage tier (`mmap`/host) and execution mode (`cpu`/mixed/strict `gpu`) are distinct. `ExpertBundle` contains complete gate/up/down views plus source owner; it survives GPU cache eviction. A model block requests selected expert outputs and never chooses where the weights live. Native packed bytes remain authoritative, including mixed-format projections.

The initial path reuses `AdaptiveExpertCache` and `HeterogeneousExpertExecutor`; synchronous bundle materialization publishes only after all projections exist. In-flight events and selected-working-set protection prevent eviction while consumers read. CPU misses execute all selected experts and produce unweighted outputs; deterministic combination applies weights once. Unsupported GPU formats take CPU execution in mixed mode and explicit error in strict GPU mode.

No full-model dequantization, full PLE allocation, pinned copy of all experts, or per-token mandatory expert upload is allowed. Reference CPU execution with zero decoded-host budget expands only rows; native CPU kernels decode lanes directly. Metadata/indexing and sparse source construction allocate no weight payload copies.

## Current storage/execution additions (0281–0288)

- Frozen mixed GPU Q8 plans retain original packed expert weights and only O(D + selected*F) quantized activation scratch. Dense Q8 projections do not create dense GPU weight copies. CPU Q8_K/Q8_0 remains independent of GPU Q8_1.
- Exact BF16 HC down/up/inject storage saves1279262720 device bytes while reconstructing FP32 at kernel loads. The preflight planner accounts the saving explicitly; gamma, activations and arithmetic remain FP32. Unsupported HC batch/fallback layouts fail rather than expanding matrices silently.
- Optional size-classed fixed slots share one preallocated arena. Actual checkpoint7GiB layout holds3673 slots/7515878400bytes versus2823/7515955200 uniform slots. Slot compatibility participates in eviction even when logical global bytes remain; held projection leases/quarantine and complete publication remain authoritative. This is static class sizing, not a heat-driven asynchronous mover.
- A standalone bounded mapped pinned-host tier demonstrates host storage with GPU execution. It holds only explicitly selected complete expert copies and checks budget/headroom. Projection aliases retain the allocation. Direct readonly mmap registration is unsupported here; no checkpoint mapping was made writable. Direct mapped GPU execution is slower and is **not used by model decode**. A proper staged CPU/PCIe split remains a separate dependency.
- Optional native service owns one native thread per row pool, at most64 cache-line-separated128-byte handoff flags and one sticky status. No CUDA/Python calls execute inside its thread. GPU epoch publication/wait uses mapped system-release/acquire operations; waits are bounded to2seconds. Wait completion publishes a device-local status; the conditional copy reads no CPU payload on failure, avoiding an incomplete-slab DMA race. Failure poisons output, is checked once at model completion, marks the state failed and requires executor unload. All clients/CPU batches retire before flags, sources, service or pool are destroyed. GPU spin time is waiting, not useful kernel arithmetic.

Host warming remains reclaimable canonical mmap residency, not a second complete pinned expert bank. PLE remains disk mapped. New diagnostic native service and pinned operator tier have distinct ownership/contracts. Exact measurements, snapshots versus high-water limits, neutral experiments and tests are recorded in `0281_0288_execution_optimization.md` / `0281_0288_performance_v3.json`.

## Optional deferred QSA validation (0280)

A model-owned48-element uint32 status buffer (~192bytes) replaces per-layer host finite-check readbacks only when explicitly requested. All original finite checks still run; GPU commit kernels suppress every K/V/raw/pool write for an invalid layer. One model-step status read reports the first invalid layer before successful return, marks the request failed/non-resumable and prevents subsequent reuse. Statuses reset for isolated requests; close fences pending operations. Standalone/default QSA keeps synchronous validation. This is not token-wide rollback, and it demonstrated no throughput gain. See `0280_qsa_validation_dependencies.md`.

## Frozen native handoff / graph ownership (0275–0279)

The optional native handoff retains immutable descriptors/canonical mmap owners, not copied expert payloads. At most32768registered descriptors (~4MiB metadata) and64device plans are allowed. Each plan owns pinned input/metadata/result slabs and GPU scratch bounded by top-k<=16 and widths<=4096. All current layer residents have cache leases until their plan's retirement fence; admission changes invalidate and retire plans before release.

CUDA host functions call only native CPU-pool arithmetic, never Python/CUDA. Main/CPU/shared stream events order producer input, complete native output transfers, packed GPU readers and combination. Exception paths retain inputs and join pending native work; executor shutdown retires **all** plans before destroying the shared row pool. Statistics is an explicit synchronization seam and reports errors; native failures cannot silently reuse old rows. Frozen execution pauses per-expert decayed heat and reports aggregate counters, so this is not a general adaptive mover.

Captured native graphs reserve1MiB/layer; QSA frontend graphs2MiB/QSA layer, separately added to model preflight. Later memory planning credits only actual owned private pools, including native/QSA pools, and model close fences graph-bound streams before clearing owners. Observed total private native reservations1,523,712bytes; QSAfrontend2,488,320bytes. Mutable recurrent/convolution arrays still copy back; only explicitly input-only routing/RoPE operands may omit copy-back (0279).

0275 sampled RssAnon852,914,176bytes and RssFile55,074,443,264bytes are separate category observations, not a simultaneous total or guaranteed peak. Default GPU pool snapshots12,157,392,896used/12,345,454,080reservedbytes exclude private graph pools and driver allocations. No device-wide high-water claim. PLE remains disk-mapped; warming50GiB touches reclaimable expert pages without a second packed/private or pinned expert bank. See `0274_0279_native_handoff.md` for limitations.

## Measured checkpoint storage

| Category | Packed bytes | GiB |
|---|---:|---:|
| Routed experts | 50,292,326,400 | 46.838 |
| Nonrouted core (includes embeddings/head/shared/HC/mixers) | 4,448,432,640 | 4.143 |
| PLE, including projections | 28,865,879,040 | 26.883 |
| Of which disk-backed table | 28,800,138,240 | 26.822 |
| Total payload | 83,606,638,080 | 77.865 |

Full FP32 expansion would be 707,775,598,080 bytes (~659.17 GiB). Mapping 77.9 GiB is virtual address reservation, not host residency. The 64 GiB machine must rely on bounded working sets and OS file-backed page eviction. A fully pinned source is not safe here.

One token selecting ten distinct experts in all 48 layers touches 982,272,000 packed expert bytes before page/cache effects (~937 MiB). PLE selects sixteen 90-byte rows (1440 payload bytes/token), but storage reads occur at page granularity. No throughput prediction follows from these lower bounds.

## Explicit device reservation

`Qwen4ExpGGUFIndex.memory_plan` requires byte reservations for static weights, persistent state, activations, expert cache, transfer staging, and headroom, and rejects a sum exceeding `device_total` before construction. The caller must derive the state reservation from its configured context, not metadata maximum by default.

Illustrative **short-context planning envelope**, not measured inference peak: static 4.5 GiB, state 1 GiB, activations/scratch 1 GiB, cache 6 GiB, staging 0.25 GiB, headroom 2 GiB = 14.75 GiB. This must be reduced against actual available VRAM and measured allocator/runtime high-water marks. The observed RTX5080 reports 16303 MiB total and 14615 MiB free at audit, so this example does **not** pass an available-memory check unchanged. Existing applications may already own VRAM.

At 262144 tokens and one sequence: FP32 K/V alone is 12 GiB, indexer raw FP32 keys 1.5 GiB; recurrent matrices ~108 MiB plus conv histories ~4.34 MiB. Even F16 K/V is 6 GiB. Full advertised context is not an initial all-GPU FP32 promise. Context limits/offload or validated BF16 storage policies must precede such allocation. QSA top-k does not bound persistent KV length.

## Patch0213 measured packed GPU configuration

The tested short-context configuration uses6GiB static cap,6GiB packed-expert cap and14GiB aggregate device budget. Actual eligible backbone storage is5,704,430,080 bytes (~5.313GiB):3,066,007,040 FP32 dense bytes plus2,638,423,040 packed bytes, excluding mapped token/PLE tables. Initial free-VRAM and aggregate logical budget checks remain mandatory; these constants are not safe universal defaults for every device/context.

The16-token optimized decode run retains1,952 expert bundles /3,994,496,000 bytes and records9,829,571,072 pool-reserved bytes, with sampled process RSS10,318,860,288 bytes. A longer lifecycle containing repeat requests retains2,332 bundles /4,778,188,800 bytes and10,616,590,848 pool-reserved bytes, sampled RSS11,173,183,488 bytes. Pool snapshots are not global VRAM high-water measurements. Canonical expert and PLE storage stays read-only mapped; no complete packed expert duplication or full floating expert bank is created.

A demonstrated replanning defect previously compared the entire aggregate requirement against free memory **after** the backbone was allocated. Replanning now credits already committed static bytes, actual cached bytes and live-state reservations when checking the remaining physical requirement. Logical reservations are unchanged, initial admission remains strict, and unrelated/global-pool allocations are never credited. Retained free pool blocks can still make admission conservative. Aggregate live-state checks remain enforced.

Completed cache events are reaped nonblockingly on subsequent publication, even below cache capacity; pending/unqueryable fences remain protective. CPU singleton misses optionally share one read-only pinned activation and one result slab per dispatch, bounded by selected-expert count times width; worker tasks are drained before buffer release. Grouped IQ execution allocates only pointer metadata, selected hidden/weighted-result scratch and output. QSA/HC fusion uses bounded scratch and current-stream execution; QSA wider than the all-causal-cell window retains the existing sparse reference path.

Synchronous cache promotion is still the initial implementation, not the final allocator design. Expert cache misses still fall back safely when admission cannot fit. Patch0214's fixed-slot asynchronous mover below remains unimplemented.

## Fixed-slot foundation0238 and future asynchronous mover

The cache materializer now has an opt-in fixed-slot backend (`expert_cache_slots=True`) without changing expert mathematics or canonical sources. Largest complete target bundle rounds to2,662,400 bytes; the6GiB cap holds2,419 slots and reserves6,440,345,600 physical bytes even when empty. Padded stride, not payload size, is charged per resident expert. Owned physical arena bytes are credited separately from retained-handle counts during replanning.

One reusable4MiB pinned slab and dedicated copy stream materialize complete bundles. Copies are currently synchronous: publication waits for completion. `UnownedMemory` projection-view leases prevent reuse after eviction while old handles/views survive; existing in-flight fences separately protect consumers. Exhaustion safely falls back to CPU; unknown DMA completion quarantines slots. Close waits for arena consumers. No GPU weight allocation occurs per admission after arena construction.

The extended lifecycle records2,332 leased slots,6,208,716,800 charged padded bytes,11,180,748,800 sampled RSS and12,270,584,320 default-pool-reserved bytes. Dynamic-cache and slot-cache logits/active-state hashes match exactly at all checked16-token prefixes. Pool snapshots exclude CUDA context/other owners and are not global VRAM high-water measurements. The slot cache's padding can reduce useful capacity relative to dynamic allocation; performance under longer cache pressure is not yet established.

Full asynchronous movement is still unimplemented. Reserve staging/shadow slots explicitly. A future mover must:

1. Reserve a destination without overwriting a live source slot.
2. Copy complete projection bundles on a dedicated transfer stream, retaining canonical/staging owners.
3. Publish only after transfer-completion event (or an explicit consumer ready fence).
4. Protect each published generation's outstanding compute consumers.
5. Quarantine old storage until the last reader retires, independently of copy completion.
6. Bound moves, queues and live shadow allocations; CPU misses remain correct when admission fails.

Heat, hysteresis, warm-start profiles, prefill grouping and decode scheduling are distinct policies. Radiance's mapped/zero-copy GPU miss path must not be confused with the existing native CPU fallback. Fixed slots are the validated synchronous foundation of Patch0214; asynchronous transfers/publication and shadow-slot scheduling remain in progress.

## Observed memory, not estimates

Header audit peak RSS ~889 MiB. Production header-only multi-shard indexing peak ~884 MiB, settled RSS ~244 MiB (large tokenizer metadata parsing dominates). Synthetic store tests peak ~57.4 MiB. The six-forward real-checkpoint lifecycle smoke subsequently measured peak process RSS10,281,414,656 bytes on CPU and8,632,389,632 bytes on CuPy. The GPU run's final pool snapshot was4,414,358,016 used /4,420,487,680 reserved bytes (not a global VRAM high-water mark). Owned-allocation plans do not impose an RSS ceiling: touched canonical mmap pages and Python/CUDA metadata increase RSS. The default8GiB host-budget parameter is therefore **not** an8GiB process-RSS guarantee. No hard host-RSS/page-cache governor is implemented; this remains a limitation to address before long-context/out-of-core stress claims. See performance_history.md.
