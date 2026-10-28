# Patch 0102N — Attention backward profile and BF16 output-projection dX

Patch 0102N profiles the complete 20-layer dual-resolution backward and adds an
**opt-in** BF16 tensor-core path for the attention output projection's input
gradient. The architecture, masks, summaries, supervision, and serialized
parameters are unchanged.

## Change implemented

`DualResolutionAttention.backward()` previously evaluated both output-projection
backward GEMMs in FP32:

- `dWo = merged.T @ dy`
- `dmerged = dy @ Wo.T`

The optimized 338M `GQAAttention` already evaluates the analogous `dmerged` GEMM
with BF16 branch tensors. Patch 0102N applies that established precision boundary
to the dual-resolution path while deliberately retaining the existing FP32 `dWo`
calculation and FP32 downstream attention-gradient contract.

Enable it with:

```bash
export MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX=1
```

The default is `0`. This switch is independent of packed projection, pooling,
compressor RMSNorm, and gate switches so that it can be A/B tested separately.

A paired benchmark alternates A/B order while reusing exactly the same initialized
model, inputs, target rows, and process:

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=1 \
MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS=1 \
MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV=1 \
MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=1 \
MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM=1 \
MINI_LLM_DUAL_RESOLUTION_FUSED_GATE=1 \
python benchmarks/benchmark_dual_resolution_output_dx_0102n.py \
  --context 4096 --layers 20 --batch-size 1 --compression-dim 512 \
  --score-last-k 128 --head-chunk-tokens 64 \
  --warmup-pairs 3 --repeat-pairs 7 \
  --output benchmark_0102n_paired_4k.json
```

## Baseline and profile

Environment used locally:

- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB
- Driver: 580.178.04
- Python: 3.12.3
- CuPy: 14.2.0
- CUDA runtime: 13.2
- Triton: 3.8.0
- Git `HEAD`: `7b64c7745fbe449e09f03dfc37d219fa7332372c`
- Working-tree baseline: patches 0102A–0102M applied but uncommitted

The initial warmed 4K reference reproduced at 100.42 ms forward, 0.47 ms loss,
407.50 ms backward, and 8,052 processed tokens/s. This session was slower than
the historical 9,068 tokens/s result, so conclusions use paired measurements,
not the historical absolute rate.

The repository's synchronized scopes reported the following 20-layer totals.
Parent and child regions overlap and must not be summed.

| Region | Time | Share/interpretation |
|---|---:|---|
| SwiGLU backward | 192.46 ms | Largest complete backward region |
| Attention backward | 100.53 ms | Immediate audit target |
| Compressor backward | 25.40 ms | Child of attention backward |
| Global attention backward | 24.53 ms | Child of attention backward |
| Local attention backward | 10.62 ms | Child of attention backward |
| Gate backward | 4.36 ms | Child of attention backward |
| Pooling backward | 2.63 ms | Child of compressor backward |
| Norm backward, both norms | 16.47 ms | Outside attention |

Nsight Systems attributed about 37% of captured GPU kernel time to three FP32
SIMT GEMM families. Global `_dkdv` represented 4.6%. CUDA kernel-launch API time
was small relative to GPU execution. The measured workload is therefore primarily
arithmetic/GEMM limited, with substantial secondary memory traffic from casts,
adds, and packing; it is not primarily a Python launch-overhead workload at 4K.

The isolated `4096×1024 @ 1024×1024` output-gradient operation measured:

| Operation | GPU time |
|---|---:|
| FP32 GEMM | 304.86 µs |
| BF16 GEMM | 96.65 µs |
| BF16 GEMM plus FP32 cast | 114.63 µs |

With synchronized full-model scopes, the 20 output-projection backward regions
dropped from 12.83 ms to 8.41 ms. Complete attention backward dropped from
100.14 ms to 93.17 ms in that diagnostic run.

## Correctness

CPU/reference validation:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_dual_resolution_output_dx_0102n.py \
  tests/test_dual_resolution_packed_projections_0102h.py \
  tests/test_dual_resolution_0102.py \
  tests/test_dual_resolution_multibatch_rope_0102g.py
```

Result: **41 passed, 5 skipped**.

CUDA validation:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q \
  tests/test_dual_resolution_output_dx_0102n.py \
  tests/test_dual_resolution_packed_projections_0102h.py \
  tests/test_dual_resolution_fused_gate_0102m.py
```

Result: **18 passed, 5 skipped**. One Triton warning reports that an integer
condition passed to `tl.where` is deprecated; it is pre-existing and unrelated
to this patch.

The new tests cover:

- Explicit default-off and opt-in behavior.
- Production-width BF16 execution.
- Batch sizes 1 and 2.
- Unequal document lengths and EOS boundaries.
- Invalid summary slots and partial final blocks.
- All three Triton attention kernels.
- Fused pooling, compressor RMSNorm, and local/global gate paths.
- Input-gradient and all parameter-gradient parity.
- Exact preservation of forward output and FP32 `Wo` weight gradient.
- Gradient accumulation over repeated backward passes.

The optimization is backward-only, has no inference path, creates no persistent
weights, and does not affect optimizer weight refresh.

## Performance

The table uses the paired same-model benchmark. The displayed before/after values
are per-mode medians; improvement is the median of each A/B pair, which is more
robust than taking the ratio of independent medians.

### 4K, 20 layers, 128 scored rows

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Forward time | 108.90 ms | 109.32 ms | No expected change; observed noise |
| Backward time | 423.59 ms | 419.60 ms | 4.83 ms paired median / 1.15% |
| Total time | 533.84 ms | 529.62 ms | 4.22 ms paired median / 0.80% |
| Processed tokens/s | 7,673 | 7,734 | 0.80% from independent medians |
| Supervised tokens/s | Not measured | Not measured | Synthetic tail-scored benchmark |
| GPU memory | 3,945 MiB pool reserved | Same process/pool | No persistent increase measured |

A longer seven-pair 4K run was noisier but still positive: median paired savings
were 1.16 ms backward and 4.98 ms total (0.29% and 0.94%).

### 16K, 20 layers, 128 scored rows

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Forward time | 362.29 ms | 360.82 ms | No expected change; observed noise |
| Backward time | 1,594.93 ms | 1,579.67 ms | 17.13 ms paired median / 1.09% |
| Total time | 1,958.39 ms | 1,940.13 ms | 16.82 ms paired median / 0.86% |
| Processed tokens/s | 8,366 | 8,445 | 0.94% from independent medians |
| Supervised tokens/s | Not measured | Not measured | Synthetic tail-scored benchmark |
| GPU memory | 7,993 MiB pool reserved | Same process/pool | No persistent increase measured |

Every 16K A/B pair favored the BF16 path. Paired total improvements ranged from
0.65% to 1.12%.

No real document-first optimizer-step benchmark was run for this patch. The
reported throughput must not be interpreted as supervised production throughput.

## FLOP accounting and normalized target

Counting convention: one multiply-add is two FLOPs. The audit counts matrix
multiplications, attention score/value products, projection gradients, checkpoint
replay, attention probability reconstruction, historical-encoder replay, and
chunked vocabulary-head replay. It excludes pointwise functions, reductions,
masking, indexing, casts, optimizer work, memory traffic, and launch overhead.

### Implemented dual-resolution path, full AR supervision

Executed GFLOPs per source/supervised token:

| Component | 4K | 16K | 32K |
|---|---:|---:|---:|
| Main QKV projections | 0.251658 | 0.251658 | 0.251658 |
| Local attention | 0.221184 | 0.232243 | 0.234086 |
| Attention output projection | 0.167772 | 0.167772 | 0.167772 |
| SwiGLU | 2.013266 | 2.013266 | 2.013266 |
| Compressor projections | 0.336527 | 0.336527 | 0.336527 |
| Compressor self-attention | 0.026214 | 0.026214 | 0.026214 |
| Learned pooling | 0.000164 | 0.000164 | 0.000164 |
| Global projections | 0.000655 | 0.000655 | 0.000655 |
| Global attention | 0.026214 | 0.052429 | 0.104858 |
| Gate projection | 0.002621 | 0.002621 | 0.002621 |
| Vocabulary head, full supervision | 0.268435 | 0.268435 | 0.268435 |
| **Total** | **3.314713** | **3.351986** | **3.406258** |

The forward/recompute/gradient split for the existing 128-target benchmark is:

| Context | Original forward | Recomputation | Gradient work | Total |
|---:|---:|---:|---:|---:|
| 4K | 0.749986 | 0.804708 | 1.499972 | 3.054666 |
| 16K | 0.755868 | 0.818045 | 1.511735 | 3.085648 |
| 32K | 0.766460 | 0.839492 | 1.532920 | 3.138871 |

Full supervision changes only the vocabulary-head term in this accounting,
raising totals to the full-AR values above.

### Optimized 338M historical-hybrid path

The 338M trunk always executes 4,096 current rows. The 16K and 32K rows below
represent 12K/28K valid historical positions, not a 16K/32K deep Transformer
trunk. FLOPs are normalized per 4,096 supervised current token.

| Component | 4K horizon | 16K horizon | 32K horizon |
|---|---:|---:|---:|
| Main QKV | 0.188744 | 0.188744 | 0.188744 |
| Current attention | 0.236596 | 0.214272 | 0.214272 |
| Attention output | 0.125829 | 0.125829 | 0.125829 |
| SwiGLU | 1.509949 | 1.509949 | 1.509949 |
| Historical projections/encoder | 0.000007 | 0.010324 | 0.024096 |
| Historical self-attention | <0.000001 | 0.000196 | 0.000456 |
| Historical reads/routes | 0 | 0.016916 | 0.035272 |
| Vocabulary head | 0.268435 | 0.268435 | 0.268435 |
| **Total per supervised token** | **2.329560** | **2.334667** | **2.367057** |

The small category values differ from source-token values because 16K/32K work
is divided by the fixed 4,096 supervised rows.

At the historical 27,000 supervised tokens/s baseline, same-effective-FLOP/s
normalization gives:

| Effective horizon/context | 338M GF/token | Dual GF/token | Dual target |
|---:|---:|---:|---:|
| 4K | 2.329560 | 3.314713 | **18,975 tokens/s** |
| 16K | 2.334667 | 3.351986 | **18,806 tokens/s** |
| 32K | 2.367057 | 3.406258 | **18,763 tokens/s** |

Thus the earlier 24K estimate was optimistic under the currently implemented
training paths. Approximately 18.8K–19.0K supervised tokens/s is the more
consistent arithmetic-efficiency target for these assumptions. This remains a
normalization, not a throughput prediction: the models have different trunk
lengths, checkpoint policies, memory traffic, kernel shapes, and optimizer/data
workloads.

## Remaining bottlenecks

Ranked by measured 4K synchronized time and credible savings:

1. **SwiGLU backward (192.46 ms).** This is the largest complete region, but it
   is outside the immediate attention assignment. Its large FP32 SIMT GEMMs
   should be compared against the 338M fused/low-precision backward contract.
2. **Global attention backward (24.53 ms).** `_dkdv` dominates the global Triton
   pair and grows with summary count. Better query/key blocking or a two-stage
   deterministic reduction is the highest-priority attention-kernel experiment.
3. **Compressor backward (25.40 ms).** The attention kernel emits separate FP32
   dQ/dK/dV tensors which are cast and concatenated before packed BF16 GEMMs.
   A direct packed-gradient output can remove three casts plus concatenation,
   but requires a larger kernel/API change and careful parity validation.
4. **Main QKV integration.** Local/global dQ addition, separate inverse RoPE,
   three casts, and concatenation remain separate operations. The optimized 338M
   path fuses inverse RoPE and QKV packing.
5. **Attention backward zero fills.** Dense Triton backward allocates zeroed dQ,
   dK, and dV although nonempty kernels overwrite valid outputs. Replacing these
   with `empty` after preserving the zero-key case is low risk but expected to be
   a memory/launch-sized gain rather than a major step.

The first compressor layer still has no compressor-internal positional encoding
before bidirectional self-attention and learned pooling. For token embeddings,
that path is permutation equivariant followed by permutation-invariant pooling,
so its first-layer summaries do not explicitly preserve within-block order. This
is a representation-quality concern, not part of patch 0102N; adding compressor
RoPE would be an architectural experiment and must remain separate.

## Next action

Profile and optimize global-attention `_dkdv` at 4K, 16K, and 32K. The single
highest-value next experiment is to compare the current key-owned deterministic
kernel with a two-stage partial dK/dV reduction that preserves exact masks and
GQA accumulation while increasing parallelism across query tiles.
