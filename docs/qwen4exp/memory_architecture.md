# Qwen4Exp memory and execution architecture

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
