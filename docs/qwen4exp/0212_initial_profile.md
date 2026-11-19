# Patch 0212 — initial bounded decode profile (in progress)

## Objective and files

Identify measured bottlenecks after the short-text numerical gate, without changing production mathematics or implementing speculative optimizations.

- Diagnostic runner: `scripts/probe_0227_short_sequences_and_profile.py`.
- Focused tests: `tests/test_0227_short_sequence_profile.py`.
- Versioned result: `0227_short_sequences_profile_v1.json`; captures in `diagnostics_0227_v1/`.
- Prerequisite state oracles: `0226_identical_block_states_v2.json`, `0226_continuous_block_states_v1.json`.

No production source, checkpoint weights or pinned-reference mathematics changed. No new kernel, cache policy or speedup is claimed.

## Execution and ownership contract

Two sequential separate CuPy processes consume fixed IDs1–16 on the original mapped two-shard checkpoint. Runtime configuration: max_length16, four native CPU expert workers, FP32 compute/state, bounded default backbone placement, no GPU expert cache. Packed experts retain canonical mmap ownership; no full expert expansion or PLE allocation occurs.

Each child has a120-second deadline and sampled12GiB RSS guard; these are not hard cgroup limits. CUDA virtual-address reservations make the CPU diagnostic address-space cap inappropriate, so this runner uses RSS/time guards and the runtime VRAM planner instead.

The diagnostic `Profile` wrapper records host spans and CUDA events only for selected positions; it leaves unprofiled warmup untouched. There is no per-operation synchronization; token boundaries synchronize. `digest_state` hashes every active state array plus history/position metadata and excludes unwritten cache capacity. These are diagnostic interfaces, not production execution APIs.

## Correctness and memory

At prefixes1/2/4/8/16, incremental and chunked-prefill logits are bit-exact and hashes of all active recurrent, convolution, PLE, KV and indexer state match. This is whole-model within-backend self-consistency, not independent16-token full-model GGML parity. Representative independent DeltaNet/QSA state validation is recorded separately in `state_validation.md`.

| Measurement | Incremental | Prefill |
|---|---:|---:|
| Whole child elapsed |67.99s |65.30s |
| Construction, approximately |6.60s |6.36s |
| Sampled peak RSS bytes |10,369,241,088 |10,391,388,160 |
| Native CPU expert executions |7,680 |7,680 |
| Decoded-expert cache bytes |0 |0 |
| GPU expert residents |0 |0 |

CuPy pool snapshots:4,414,818,816 used /4,420,900,352 reserved bytes (~4.12GiB). These are not global VRAM high-water measurements. RSS includes file-backed mapped pages, not just owned allocations.

## Initial timing results

Instrumented incremental positions8–15: mean synchronized token latency **3.6965s**.

| Stage | Mean host span/token |
|---|---:|
| MoE total (nested parent) |1.72397s |
| Routed expert dispatch/compute/transfers |1.55152s |
| Complete Gated Delta mixer, including projections |1.36484s |
| Complete QSA, including projections/indexing |0.55624s |
| Shared down projection |0.04837s |
| Shared up projection |0.04627s |
| Shared gate projection |0.04568s |
| Attention HC read |0.02237s |
| FFN HC read |0.02065s |
| Router |0.01454s |
| PLE total |0.00105s |
| PLE lookup (nested) |0.00045s |

**Do not sum parent and child spans.** Host spans include implicit synchronization; CUDA event spans can include CPU idle gaps. This is an instrumented short fixed-token baseline, not an exclusive critical-path decomposition or optimized throughput benchmark. Whole-process times include setup and validation overhead and must not be presented as steady-state tokens/s.

## Tests and next dependency

Consolidated command:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -m pytest -q \
  tests/test_020*.py tests/test_021*.py tests/test_022*.py tests/test_0106*.py
```

Result: **1492 passed,186 skipped in16.75s** on the final rerun (earlier run:16.54s). Independent block replays and full-model CuPy processes also pass their scoped comparisons; no performance improvement was attempted.

Measured priorities:

1. Native packed GPU expert execution for actual mixed checkpoint formats, retaining bounded residency and CPU fallback.
2. Separate GDN dense projection/fallback costs from recurrent-core and launch overhead before selecting a kernel optimization.
3. Investigate QSA launch, synchronization and indexing costs.

Patch0212 remains **in progress**: exclusive overlap/critical-path accounting, transfer wait attribution, SSD bytes/major page faults, global RAM/VRAM high-water and robust prefill/decode throughput are incomplete. Fixed slots, asynchronous movement, grouped prefill and CUDA graphs are not implemented by this patch. Preserve all numerical oracles while progressing to independently switchable optimizations.
