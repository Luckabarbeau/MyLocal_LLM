# PATCH 0089P — Streaming historical block forward negative result

## Decision

Reject and remove the first Triton deterministic block-read forward candidate.

The candidate delivered a large complete F+B speedup and reduced peak CuPy pool reservation, but it failed complete native gradient parity. It was not promoted and its production dispatch, kernel module, and candidate-specific test/harness were removed.

PATCH 0089O instrumentation and benchmark evidence remain.

## Candidate

The opt-in candidate replaced deterministic block-read forward with one Triton online-softmax kernel per slot. It:

- read Q and the original full-history K/V stores directly;
- generated deterministic 32-of-384 source indices in-kernel;
- avoided gathered K/V tensors;
- avoided full FP32 score and probability tensors;
- applied ragged validity in-kernel;
- accumulated FP32 output;
- used `tf32x3` Triton dot products and online `exp2` softmax;
- retained the established generic CuPy backward unchanged.

The experiment was controlled by:

```text
MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD
```

and was disabled by default.

## Direct forward test

Three GPU cases passed the initial direct limit:

```text
uncropped block selection
cropped block selection
phase-offset selection
ragged validity
all-invalid sample row
BF16 query, FP32 K/V and output
relative L2 <= 1e-4
```

Result:

```text
3 passed
```

This short-shape test was insufficient to predict accumulated error over the production 5,120-key reduction and the subsequent 20-layer BF16 graph.

## Complete seed-48 screening performance

Identical B3/T4096 seed-48 runs used three warmups and ten measured samples. Every required fairness flag was fixed; only the experimental flag changed.

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| GPU median | 933.759 ms | 860.429 ms | **-73.330 ms (-7.85%)** |
| GPU p10 | 932.619 ms | 858.785 ms | -73.834 ms |
| GPU p90 | 934.531 ms | 863.817 ms | -70.714 ms |
| GPU MAD | 0.497 ms | 1.332 ms | +0.835 ms |
| Host median | 933.996 ms | 860.551 ms | **-73.445 ms (-7.86%)** |
| Host p10 | 932.797 ms | 858.807 ms | -73.990 ms |
| Host p90 | 934.997 ms | 863.963 ms | -71.034 ms |
| Host MAD | 0.573 ms | 1.435 ms | +0.862 ms |
| Peak CuPy pool | 14,397,997,056 B | 14,079,229,952 B | **-318,767,104 B** |
| Loss | 2.0288021564 | 2.0288832188 | +8.1062e-5 |

The p10/p90 ranges do not overlap. The performance mechanism and memory reduction are real.

The `73.330 ms` complete saving closely matches the `76.893 ms` forward block-read cost measured by PATCH 0089O, showing that the kernel removed nearly the entire forward bottleneck without shifting substantial work elsewhere.

## Complete forward and gradient parity

The same loaded checkpoint and exact seed-48 fixture were run baseline-first and candidate-second in one process. Baseline gradients for all 217 parameters were copied to CPU, then compared against candidate gradients.

| Metric | Observed | Required | Result |
|---|---:|---:|---|
| Body output relative L2 | 5.611e-3 | <= 1e-4 | **Fail** |
| Body output cosine | 1.0 as FP32-reported scalar | diagnostic only | — |
| Body output max absolute | 0.25 BF16 | diagnostic only | — |
| Global parameter-gradient relative L2 | 1.3895e-2 | <= 5e-4 | **Fail** |
| Worst parameter cosine | 0.998038 | >= 0.99999 | **Fail** |
| Bitwise-equal parameter gradients | 8 / 217 | diagnostic only | — |
| Finite outputs and gradients | yes | required | Pass |
| Loss absolute difference | 8.1062e-5 | diagnostic only | — |

Worst relative-gradient result:

```text
historical_read.layer1.gates
relative L2: 0.0642483
cosine:     0.9982910
max abs:    7.686e-5
```

Other affected parameters included learned historical K/V, historical gates, ordinary attention Q/K projections, and FFN weights downstream of the modified reads. This is broad graph-level divergence, not an isolated reduction-order difference in one unused tensor.

## Diagnosis

The baseline forward graph uses:

```text
FP32 Q/K/V
separate FP32 cuBLAS/MAGMA GEMMs
CuPy max/exp/sum/divide softmax operations
materialized FP32 probabilities
```

The rejected candidate uses:

```text
FP32 values loaded into Triton
TF32x3 tiled dot products
online softmax rescaling
exp2-based exponentiation
new key-tile and reduction orders
```

The candidate's local direct errors were small, but the production operation reduces over approximately 5,120 keys for 12 block slots. Those changes are injected into six layers and then pass through BF16 projections and all downstream layers. The complete parity result demonstrates that their accumulated effect is too large for native-to-native acceptance.

No production-code correction is justified without separating these possible sources:

1. `tf32x3` dot-product differences versus the baseline FP32 GEMMs.
2. Online-softmax reduction order over 5,120 keys.
3. `exp2` approximation versus CuPy's exponential path.
4. Combined downstream BF16 threshold amplification.

## Single next experiment

Before another production implementation, run a production-shape isolated block-read numerical sweep with fixed tensors and no full model:

```text
B=3
Q=4096
selected K=5120
D=64
valid history rows=[0, short, full]
```

Compare these kernel arithmetic variants against the exact CuPy reference:

```text
A. input_precision="ieee", online exp2 softmax
B. input_precision="tf32x3", two-pass softmax
C. input_precision="ieee", two-pass softmax
```

Measure forward relative L2/max absolute and isolated latency for each variant. This one experiment distinguishes dot precision from online-softmax/reduction error.

A new kernel attempt is justified only if one variant achieves production-shape forward relative L2 comfortably below `1e-4` while retaining enough isolated speed to plausibly save at least 5% complete F+B. If only the IEEE/two-pass variant passes but loses most of the measured 73 ms benefit, move to the encoder FFN/SwiGLU/RMSNorm targets instead.

## Retained evidence

```text
benchmarks/results/0089p_seed48_block_fwd_baseline_screen.json
benchmarks/results/0089p_seed48_block_fwd_candidate_screen.json
benchmarks/results/0089p_seed48_block_fwd_parity.json
```

## Final state

- Historical computation is restored to the validated CuPy reference path.
- No failed candidate is enabled or retained in production code.
- PATCH 0089O ranges and the correlation analyzer remain.
- The measured block-read bottleneck remains the largest opportunity.
- Another implementation must begin with the production-shape arithmetic sweep above, not a speculative full integration.
