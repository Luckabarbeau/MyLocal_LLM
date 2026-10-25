# PATCH 0089S — Generic historical block kernel and startup prewarm

## Purpose

Avoid repeated Triton compilation pauses when testing the experimental
deterministic historical block forward:

```bash
export MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1
```

## Cause

The first 0089R kernel declared source geometry and tensor strides as
`tl.constexpr`. Triton therefore specialized separate binaries for combinations
of:

- cropped historical length and K/V batch stride;
- deterministic phase / first selected source position;
- selected-token count;
- compact query length and Q/output strides;
- deterministic period.

Historical source cropping can produce many aligned lengths, so precompiling
all literal combinations is not bounded to a small useful set.

## Change

Geometry and strides are now runtime scalar arguments. Only the actual tile
sizes and fixed D=64 scale remain compile-time constants. One pointer-dtype
signature therefore covers the aligned crop, phase, period, and compact-length
cases used by production.

When the experimental forward is enabled, model construction performs a
compile-only warmup by default:

```bash
export MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM=1
```

The warmup uses tiny dummy device allocations and Triton's `JITFunction.warmup`
API. It compiles but does not execute the historical attention workload. The
flag defaults to `1` when the experimental forward is enabled. Set it to `0`
only when explicitly diagnosing lazy compilation.

Triton's persistent disk cache is reused by later processes. The default cache
location is normally sufficient. For containers or jobs with ephemeral home
directories, select a persistent writable location before launch:

```bash
export TRITON_CACHE_DIR=/path/to/persistent/triton-cache
```

## Validation

Fresh-cache compile/reuse smoke test:

```text
compile-only prewarm:             1.835 s
first real small execution:       1.61 ms
different phase/period execution: 0.30 ms
relative L2:                      3.84e-7
finite:                           yes
```

The different geometry did not incur another visible compilation pause.

Production-shape result:

```text
artifact: benchmarks/results/0089s_block_accuracy_dynamic_prewarm.json
B=3, Q=4096, selected K=5120, D=64
```

| Metric | Specialized 0089R | Generic/prewarmed 0089S |
|---|---:|---:|
| Median kernel latency | 2.997 ms | 3.175 ms |
| Speedup over CuPy reference | 1.946x | 1.884x |
| FP32 relative L2 | 1.0848e-6 | 1.0718e-6 |
| BF16 relative L2 | 5.2894e-5 | 4.7066e-5 |

The generic signature costs approximately `0.18 ms` in this isolated call but
prevents repeated compilation for crop/phase/period values. Accuracy remains
consistent with the earlier candidate.

Default-path regressions:

```text
30 passed, 1 skipped
```

## Scope and limitation

This improves startup behavior only. It does not change the 0089R numerical
acceptance decision: the experimental historical block forward remains disabled
by default because complete 20-layer gradient parity failed. Enabling it is
still an explicit research choice.
