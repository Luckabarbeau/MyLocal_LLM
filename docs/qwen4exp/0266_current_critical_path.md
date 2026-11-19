# 0266 — Current-path critical-path evidence

## Objective / method

Profile16consecutive decode tokens after64response decode steps, following the captured Strata warmup/prompt workload. Configuration:0265GDN capture,0261Q2 SIMD, native row pool, frozen post-warmup cache, shared overlap. Do not reuse the old cache-fitting16ID timeline as current evidence.

New `scripts/probe_0266_critical_path.py` adds diagnostic NVTX ranges around CPU submit/wait, routing, mixer and expert dispatch. Nsight2026.3.2: `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop`. Direct runner-child invocation retains the180s/Anon/MemAvailable guards and an outer195s terminal timeout; it does not use the normal parent supervisor. No production equations changed.

Artifacts: `0266_current_path_v1.nsys-rep`, `.sqlite`, `0266_stats_v1_*.csv`, `0266_critical_path_summary_v1.json`, and `diagnostics_0266_timeline_v1/`. No reference/checkpoint modifications. This is a diagnostic, not throughput validation or flame-graph sampling of native CPU kernels.

## Measured window

| Quantity across16tokens | Value |
|---|---:|
|Wall window|1.482568928s|
|Summed GPU kernels|396.473ms|
|Union of GPU kernel intervals|396.361ms|
|Union of GPU kernels, copies and memset|405.990ms|
|GPU operation span|1,480.909ms|
|`try_forward_decode_batch` NVTX spans|812.946ms|
|Nested CPU completion-wait spans|321.591ms|
|Driver `cuLaunchKernel` calls|41,969 /108.405ms API time|

GPU operation activity covers roughly27.4% of the wall window. The complement includes host dispatch, CPU work, synchronization and dependencies; it is **not** a single attributed CPU cost. Parent/child NVTX durations overlap. API call time can overlap GPU execution. Instrumentation overhead is not subtracted. This is evidence to select experiments, not a complete exclusive-critical-path decomposition.

GPU kernel totals: `packed_swiglu`90.130ms; `_packed_linear_decode_kernel`87.303ms; `packed_linear`86.312ms; HC up26.091ms/down25.274ms; QSA14.075ms. Packed projections dominate GPU work, while large wall gaps remain between submissions. CPU completion wait is about20.10ms/token and is directly exposed in the dispatch path.

## Two evidence-driven follow-ups

1. Retest existing mixed GPU grouping under this now-frozen/captured path. `diagnostics_0266_grouped_v1/` completed supervised:328response tokens,327forwards/26.04129s = **12.55698/s**, versus0265's12.60665/s on a different continuation. No demonstrated gain; leave opt-in. Do not blindly enable it because it reduces launch count.
2.0267 tests Strata-style bounded worker spinning to reduce repeated phase sleep/wake latency.0268 designs a persistent device-side route/execution plan to remove repeated per-expert Python submission and pointer-table uploads. Both retain CPU fallback and immutable sources;0268's validation blocker is separately documented.

Strata authority remains the pinned0258audit (`712fec6b4297c9cfd32ab43daaa2979a9aa01edf`): persistent workers, bounded spin/sleep, stable runtime buffers, device hit planning and native handoffs. No claim all these mechanisms are implemented, nor that operation timing alone predicts50t/s.
