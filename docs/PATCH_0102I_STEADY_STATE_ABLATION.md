# Patch 0102I — Diagnose the 0102H GPU performance regression

## Why

The first RTX 5080 full-model run of the new 0102H fast path measured 1,148
input tokens/s versus 4,156 tokens/s with the reference (4K, 20 layers, BF16,
all three Triton kernels, block checkpointing, last 128 scored). This is a
**3.62× slowdown**, dominated by 2,576 ms backward versus 455 ms.

This is a real *observed single-run* result, **not yet proven steady-state**.
The 0102D smoke benchmark measures a single pass with no warm-up, and the
0102H compressor switches its Triton QKV inputs from FP32 to BF16. Triton JIT
compiles distinct kernels for those dtypes. CuPy kernel setup and lazy packed
weight creation can also contaminate cold measurements.

## Changes

- The existing full-decoder benchmark accepts `--warmup-iterations 1
  --repeat-iterations 3` (defaults retain the old one-shot behavior).
  Each iteration is a full forward, chunked loss, complete 20-layer backward;
  gradients are zeroed between passes and all CUDA work is synchronized before
  timing is recorded. Both per-iteration times and medians are output. The
  optional optimizer step still occurs *once* after the measured passes.
- Derived packed weights are prepared *before* the warm-up and timed passes.
- New `MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV=0` switch allows testing
  packed projections while keeping FP32 QKV at the Triton compressor. It is
  only meaningful when `MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=1`. Defaults
  to `1` to preserve the 0102H behavior. The independent existing
  `MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS=0` switch keeps projection
  backward GEMM operands FP32. Neither flag changes topology or tensor shapes.

## RTX 5080 A/B matrix

Use the same common command for every case, changing **only** the environment
variables and output file:

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=0 \
python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --context 4096 --layers 20 --batch-size 1 --compression-dim 512 \
  --score-last-k 128 --head-chunk-tokens 64 --activation-checkpoint block \
  --local-kernel triton --global-kernel triton --compression-kernel triton \
  --query-chunk 512 --warmup-iterations 1 --repeat-iterations 3 \
  --output 0102i_ref.json
```

Run the same command with the following prefix and unique `--output` file:

| Case | FAST | BF16_PROJECTION_GRADS | BF16_COMPRESSOR_QKV |
| --- | ---: | ---: | ---: |
| Reference | 0 | irrelevant | irrelevant |
| Packed only, FP32 grads and compressor | 1 | 0 | 0 |
| Packed + BF16 gradient operands | 1 | 1 | 0 |
| Packed + BF16 compressor QKV | 1 | 0 | 1 |
| Complete 0102H | 1 | 1 | 1 |

Example (packed only):

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=1 \
MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS=0 \
MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV=0 \
python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --context 4096 --layers 20 --batch-size 1 --compression-dim 512 \
  --score-last-k 128 --head-chunk-tokens 64 --activation-checkpoint block \
  --local-kernel triton --global-kernel triton --compression-kernel triton \
  --query-chunk 512 --warmup-iterations 1 --repeat-iterations 3 \
  --output 0102i_packed_only.json
```

**Interpretation:** If the warmed complete 0102H path is fast, compilation
was the major contributor. If packed-only is slower, inspect packed GEMM
layout, grouped backward, and gradient-buffer strides. If only BF16 gradients
slow down, profile CuPy's BF16 `@` path instead of assuming it uses the
optimized 338M cuBLAS kernels. If only BF16 compressor QKV slow down, profile
the Triton dense compressor forward/backward BF16 specializations.

The benchmark scores 128 targets only; it is intended for isolated
full-model execution A/B, not directly comparable with the older 338M
**supervised**-tokens/s production training result. GPU peak memory is also
not established by the benchmark's synchronized samples.

## Tests

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_dual_resolution_packed_projections_0102h.py \
  tests/test_dual_resolution_0102.py

MINI_LLM_BACKEND=cupy python -m pytest -v \
  tests/test_dual_resolution_packed_projections_0102h.py \
  tests/test_triton_dual_resolution_remaining_0102c.py
```

The GPU tests are **not run** in the patch-authoring environment. Until
hardware parity and warmed speed are verified, use FAST_PROJECTIONS=0 for
production training.
