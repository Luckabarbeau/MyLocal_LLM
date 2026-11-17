# Qwen4Exp memory and execution architecture

## Invariants

Canonical weights live in independent read-only shard mappings. Storage tier (`mmap`/host) and execution mode (`cpu`/mixed/strict `gpu`) are distinct. `ExpertBundle` contains complete gate/up/down views plus source owner; it survives GPU cache eviction. A model block requests selected expert outputs and never chooses where the weights live. Native packed bytes remain authoritative, including mixed-format projections.

The initial path reuses `AdaptiveExpertCache` and `HeterogeneousExpertExecutor`; synchronous bundle materialization publishes only after all projections exist. In-flight events and selected-working-set protection prevent eviction while consumers read. CPU misses execute all selected experts and produce unweighted outputs; deterministic combination applies weights once. Unsupported GPU formats take CPU execution in mixed mode and explicit error in strict GPU mode.

No full-model dequantization, full PLE allocation, pinned copy of all experts, or per-token mandatory expert upload is allowed. Reference CPU execution with zero decoded-host budget expands only rows; native CPU kernels decode lanes directly. Metadata/indexing and sparse source construction allocate no weight payload copies.

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

## Future fixed-slot mover contract (not implemented)

The cache materializer seam admits a preallocated slot backend without changing expert mathematics or canonical sources. Charge padded slot stride and reserve staging/shadow slots explicitly. A future mover must:

1. Reserve a destination without overwriting a live source slot.
2. Copy complete projection bundles on a dedicated transfer stream, retaining canonical/staging owners.
3. Publish only after transfer-completion event (or an explicit consumer ready fence).
4. Protect each published generation's outstanding compute consumers.
5. Quarantine old storage until the last reader retires, independently of copy completion.
6. Bound moves, queues and live shadow allocations; CPU misses remain correct when admission fails.

Heat, hysteresis, warm-start profiles, prefill grouping and decode scheduling are distinct policies. Radiance's mapped/zero-copy GPU miss path must not be confused with the existing native CPU fallback. No per-token GPU allocation scheme is claimed as the final design; fixed slots and asynchronous weight transfers remain Patch0214.

## Observed memory, not estimates

Header audit peak RSS ~889 MiB. Production header-only multi-shard indexing peak ~884 MiB, settled RSS ~244 MiB (large tokenizer metadata parsing dominates). Synthetic store tests peak ~57.4 MiB. The six-forward real-checkpoint lifecycle smoke subsequently measured peak process RSS10,281,414,656 bytes on CPU and8,632,389,632 bytes on CuPy. The GPU run's final pool snapshot was4,414,358,016 used /4,420,487,680 reserved bytes (not a global VRAM high-water mark). Owned-allocation plans do not impose an RSS ceiling: touched canonical mmap pages and Python/CUDA metadata increase RSS. The default8GiB host-budget parameter is therefore **not** an8GiB process-RSS guarantee. No hard host-RSS/page-cache governor is implemented; this remains a limitation to address before long-context/out-of-core stress claims. See performance_history.md.
