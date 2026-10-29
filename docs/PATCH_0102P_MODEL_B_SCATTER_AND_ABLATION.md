# 0102P — Complete Model B local-only ablation and optional backward scatter reuse

This patch continues directly from the Model B (0102O) implementation and 0102P
full-supervision runtime-profile work recorded in the supplied development transcript.
It does not change the model architecture, checkpoint format, or training objective.

## Baseline on RTX 5080 (previously measured, NOT measured by this patch)

Identical 4K, 20-layer, full-supervision, BF16, block-checkpoint benchmark,
with the repository's measured GPU execution defaults and 512-row LM-head tiles:

| Configuration | Supervised tokens/s |
|---|---:|
| Model B (4 summaries/block, 2 global heads) | 14,925 |
| Local-only mixed-window trunk | 18,354 |

Historical measurements imply about 50 ms extra full-model time for matrix-pooling
and global retrieval combined. This cost must be reduced without detaching history,
changing causality, or turning off complete-sequence supervision.

## A. Correct the benchmark-only local-only ablation

`benchmarks/benchmark_dual_resolution_full_model_0102d.py::build_config()` now
applies `--disable-global` to the `--tiny` configuration too. Previously, the
production-width branch honored the flag but the tiny branch returned early.
The failing `test_matrix_pool_local_only_ablation_reports_disabled_global_branch`
now passes, as does the corresponding production-width configuration check.

## B. Experimental in-place gradient scatter

Existing Model B pooling backward writes into an entire `[batch, length, 1024]`
FP32 zero tensor, adds it to global query gradients, and allocates a separate
zero residual-gradient tensor. The new optional path adds the nonoverlapping
completed-block gradients directly into the already existing FP32 global-query
input-gradient buffer. The causal block metadata guarantees that each token
belongs to at most one complete block.

Enable with:

```bash
MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE=1
```

It stays **disabled by default** until CUDA gradient parity and a warmed GPU
benchmark confirm a benefit. Explicit `0` restores the unmodified accumulation
path. It changes neither computed attention nor the compressor's trainable
parameters. It does not affect the original `dual-resolution-1024-20l` model.

The benchmark JSON records the new switch.

## C. Tests

CPU/reference:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
    tests/test_runtime_defaults.py \
    tests/test_dual_resolution_full_model_0102d.py \
    tests/test_matrix_pool_dual_resolution_0102o.py \
    tests/test_dual_resolution_0102.py
```

GPU parity on the user's workstation:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
    tests/test_matrix_pool_dual_resolution_0102o.py \
    tests/test_triton_dual_resolution_local_0102b.py \
    tests/test_triton_dual_resolution_remaining_0102c.py
```

## D. Matched warmed full-model A/B

Set the 512-row runtime LM-head chunk; all 4096 positions are scored. Both
runs must use the same seed, runtime profile, and optimized kernels.

```bash
for mode in 0 1; do
    MINI_LLM_BACKEND=cupy \
    MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=1 \
    MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS=1 \
    MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE=$mode \
    python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
        --architecture matrix-pool \
        --runtime-profile auto \
        --context 4096 \
        --layers 20 \
        --batch-size 1 \
        --score-last-k 4096 \
        --head-chunk-tokens 512 \
        --activation-checkpoint block \
        --local-kernel triton \
        --global-kernel triton \
        --compression-kernel triton \
        --warmup-iterations 3 \
        --repeat-iterations 7 \
        --output "0102p_matrix_pool_scatter_${mode}.json"
done
```

Use A/B/A repetition if the change is small. The historical 14,925 tok/s
number is not a substitute for a matched remeasurement.

## E. Next major optimization

This scatter change is narrowly scoped. The full-model pool/global branch still
accounts for a substantial fraction of runtime. The next profiler-driven work
should isolate matrix-pooling scores + reductions versus narrow global-attention
forward/backward, and prioritize a fused matrix-pool kernel if the pool accounts
for sufficient GPU time. Do not extrapolate standalone kernel speedups directly
to the complete 20-layer model.
