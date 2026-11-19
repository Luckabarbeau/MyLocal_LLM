# 0243 — Measured Strata IQ3_S baseline (model drafter off)

Capture: `diagnostics_0243_off_v1/`, including `benchmark_console.log` and the isolated `engine.log`. Used `strata_launch_bench_capture.sh` with `DRAFTING=off`, `STRATA_RAM_KEEP_GIB=8`, and a 300 s time cap. The explicit `auto` plan selected DFlash2 and required a 1.2 GB download; `off` did not download anything. The model was ready after roughly a minute. All nine standard-v1 performance rows returned; the cap interrupted sanity case 18/22, so no complete suite result or post-benchmark API capture. The server was stopped and no Strata GPU process remained. No inference code or checkpoint changed.

RTX 5080 CUDA engine 0.1.59, original two shards, 262144 context capacity, one request, INT8 KV, 15 CPU workers. Medians across three repetitions, from the console:

| Workload | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---:|
| Short | 139.59 | 50.63 | 0.573 |
| 4K | 2052.25 | 51.31 | 2.018 |
| 16K | 2649.77 | 55.28 | 6.211 |

## Memory, execution, and caveats

- The full 46.84 GiB arena plus headroom would need 54.84 GiB; the engine saw 52.10 GiB available after waiting. It used **43.12 GiB of page-locked host expert storage** instead, plus a file-backed fallback. The GPU expert cache used **5.31 GiB / 2,768 slots**. The engine reported 525 MiB VRAM free at ready. These are subsystem allocations and a ready snapshot, not process high-water marks.
- The engine confirmed `STRATA_ROUTE_TAIL_SKIP=7`: eligible low-ranked misses were omitted. The logs show suffix drafts accepted even though the model drafter was off. Therefore these are emitted-token rates, not single-target-forward rates, and this is not a full-top-10 numerical oracle.
- Historical CUDA entries in the local `iq3_s-256k.engine.log` independently show **69.6–90.7 tok/s** across sampled turns with roughly **82K–98K prompt tokens** and suffix drafting. Example: 1,479 output tokens in 20.310 s (72.8 tok/s), with 92,577 of 93,274 prompt tokens reused and 107 of 198 suffix proposals accepted. That is a different workload and memory configuration, not evidence that dialing up context alone improves speed.
- This capture did not enable per-operation profiling or establish arithmetic parity. Next experiment: use one fixed prompt in separate uninstrumented and `STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1` passes, and compare suffix/lookup-disabled full routing against the default execution policy. Verify from live counters that no proposals were offered in the unassisted arm. Keep INT8 KV and Q8_1 projection comparisons separate from scheduling changes.
