# Qwen4Exp performance and memory history

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
