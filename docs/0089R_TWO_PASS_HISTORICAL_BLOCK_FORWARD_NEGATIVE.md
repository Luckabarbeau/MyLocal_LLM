# PATCH 0089R — Two-pass historical block forward accuracy result

## Decision

Do not enable the streaming deterministic historical block forward for training.

The corrected production kernel is retained only behind the disabled opt-in flag:

```text
MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1
```

The default remains the validated CuPy reference path. The candidate is fast and
accurate as an isolated FP32 operation, but its small output differences cross
BF16 rounding boundaries and amplify through the complete 20-layer graph. It
fails the existing native-to-native body-output and gradient gates, so the
observed discrepancy is not accepted as harmless floating-point variation.

## Workload and controls

Both tests used the production checkpoint and real document-first seed-48
fixture:

```text
checkpoint: ./checkpoints/0083_intermediate_20L_1024_fullmix_100k
dataset:    ./token_shards_ensemble_32k_mixed
batch:      3
working:    4096
history:    [0, 6162, 61440]
seed:       48
```

All required fairness flags were fixed, including the Triton current-attention
backend, fused Q/K RoPE forward, history recomputation, 1024-row query chunks,
and the requested optimizer-offload, compact-target, grouped-GEMM, prepared-RoPE,
and memory-trim settings. The optimizer was not constructed or executed in the
parity benchmark.

## Corrected kernel

The 0089R candidate:

- handles deterministic `block` slots only;
- reads the original compact historical K/V stores directly;
- generates the exact 32-of-384 source coordinates in-kernel;
- preserves ragged validity and all-invalid rows;
- performs a first pass for the global row maximum;
- performs a second pass for the softmax denominator and `P @ V`;
- uses `tf32x3` dot products and FP32 accumulators;
- returns FP32 and leaves the established generic CuPy backward unchanged;
- does not materialize gathered K/V, scores, or probabilities.

Sparse, learned, decode, backward, and default forward paths are unchanged.

## Isolated production-shape result

Artifact:

```text
benchmarks/results/0089q_block_accuracy_production.json
```

Geometry:

```text
B=3, Q=4096, selected K=5120, D=64
valid historical rows=[0, 6162, 61440]
```

| Metric | CuPy reference | TF32x3 two-pass |
|---|---:|---:|
| Median GPU latency | 5.831 ms | 2.997 ms |
| Speedup | 1.000x | 1.946x |
| FP32 relative L2 | — | 1.0848e-6 |
| FP32 max absolute | — | 8.5831e-6 |
| BF16 relative L2 | — | 5.2894e-5 |
| BF16 mismatches | — | 73 / 786,432 |
| BF16 mismatch fraction | — | 9.2824e-5 |

This confirms that the production module reproduces the arithmetic sweep and
that the kernel itself has a large isolated performance opportunity.

The full 0089Q sweep also showed that neither tested IEEE variant is a justified
accuracy correction:

| Variant | FP32 relative L2 | Median GPU latency |
|---|---:|---:|
| IEEE online | 1.4720e-6 | 4.944 ms |
| TF32x3 two-pass | **1.0848e-6** | **2.997 ms** |
| IEEE two-pass | 1.4741e-6 | 4.861 ms |

TF32x3 two-pass was both the closest and fastest measured implementation.

## Complete body and gradient parity

Artifact:

```text
benchmarks/results/0089r_seed48_block_fwd_parity.json
```

The same loaded model and fixture were run baseline first and candidate second
in one process. The comparison covered the complete body output, loss, and all
217 parameter gradients.

| Metric | Observed | Required | Result |
|---|---:|---:|---|
| Body-output relative L2 | 5.5648e-3 | <= 1e-4 | Fail |
| Body-output cosine | 0.9998955 | diagnostic | — |
| Body-output max absolute | 0.375 | diagnostic | — |
| Global gradient relative L2 | 1.3625e-2 | <= 5e-4 | Fail |
| Worst parameter cosine | 0.9923513 | >= 0.99999 | Fail |
| Loss absolute difference | 6.6996e-5 | diagnostic | — |
| Finite outputs and gradients | yes | required | Pass |

Worst relative parameter result:

```text
historical_read.layer16.gates
relative L2: 0.1513692
cosine:      0.9923513
max abs:     4.0543e-5
```

The global gradient discrepancy is `27.25x` the allowed `5e-4` limit. The body
output discrepancy is `55.65x` the allowed `1e-4` limit. This is broad enough to
reject the candidate even though the scalar loss difference is small.

## Floating-point assessment

The evidence supports this dependency chain:

```text
~1e-6 isolated FP32 read difference
→ 73 BF16 rounding-boundary changes in one captured block read
→ repeated block-read substitutions across six layers / twelve slots
→ different BF16 values entering downstream projections
→ 5.56e-3 final body difference
→ 1.36e-2 global gradient difference
```

The original online candidate and corrected two-pass candidate produce similar
complete-graph divergence (`1.3895e-2` versus `1.3625e-2` global gradient
relative L2). Therefore online softmax was not the sole cause. The demonstrated
problem is downstream sensitivity to any non-bitwise-equivalent historical read
at BF16 storage boundaries.

The discrepancy may originate from legitimate floating-point reassociation, but
"floating-point" does not by itself establish training equivalence. Accepting
this path would require a separate optimizer-update and short-trajectory study
showing equivalent updates and loss behavior across repeated microbatch
accumulation. The current strict native parity evidence does not justify that
acceptance experiment as a production promotion.

## Validation

```text
python -m py_compile \
  mini_llm/ops/triton_historical_block.py \
  mini_llm/ops/historical_read.py \
  benchmarks/benchmark_0089q_block_accuracy.py \
  benchmarks/benchmark_0089r_block_forward_parity.py \
  benchmarks/profile_0087a_training.py

PYTHONPATH=. pytest -q \
  tests/test_0083b_historical_read.py \
  tests/test_0089o_historical_nsys.py \
  tests/test_0087a_profile_ranges.py
```

Result:

```text
30 passed, 1 skipped
```

`git diff --check` passed.

## Next measured opportunity

The deterministic block backward remains the largest measured historical target:

```text
192.060 ms / complete seed-48 F+B
20.28% of 947.242 ms
```

A backward-only streaming kernel is not yet justified because it must reconstruct
probabilities consistently with the accepted forward arithmetic. The safer next
avenue is to profile and fuse historical encoder recomputation subgraphs that do
not feed a BF16 replacement boundary with changed arithmetic, or to investigate
a bitwise-compatible forward that preserves the reference cuBLAS/CuPy reduction
results while eliminating only gather/materialization overhead.
