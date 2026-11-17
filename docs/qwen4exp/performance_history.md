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

Per-layer critical-path profiling, global VRAM peaks, SSD bytes/page faults, expert cache/transfer and overlap measurements remain Patch0212. CUDA graph capture, fixed-slot transfers, grouped prefill and adaptive prediction are not enabled. Independent-model numerical validation is required before performance optimization. Reference implementations remain available.

## Independent safe-loading diagnostic

A separate CPU-only pinned-source copy disables only loader prefetch. The fixed[1,2] comparison ran sequentially in27.16s; reference peakRSS~5.72GiB and production~8.71GiB, below the12GiB diagnostic cap. This resolved the unsafe eager-prefetch prerequisite, **not** model numerical parity. Logit relativeL2 differences remain~5–6%. No production performance optimization followed this unresolved discrepancy.
