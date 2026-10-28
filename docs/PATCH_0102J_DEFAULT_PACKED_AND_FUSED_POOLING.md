# Patch 0102J — default packed/BF16 projections, experimental fused pooling

This incremental patch applies **after 0102I**. It is intentionally restricted
to the `dual-resolution-1024-20l` training/inference implementation. No new
parameters, layers, attention equations, block-visibility rules, positional
encoding, or checkpoint fields are introduced.

## 1. Default projection path

The 0102I warmed RTX 5080 results for the 4K full-model forward/backward
benchmark, **128 scored tokens, without optimizer**, were:

| Projection configuration | Median wall time | Throughput |
| --- | ---: | ---: |
| Original separate projections | 602.1 ms | 6,803 processed tok/s |
| Packed only, FP32 backward | 527.9 ms | 7,759 processed tok/s |
| Packed + BF16 backward/compressor QKV | 482.5 ms | 8,489 processed tok/s |

The optimized packed path is now **enabled by default on the CuPy backend**.
The NumPy reference defaults to unpacked, preserving exact finite-difference
gradient and in-place weight-perturbation tests. Explicit switches override
either backend:

```
MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=1           # GPU default; 0 = separate reference GEMMs
MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS=1       # default
MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV=1         # default
```

Packed weights are derived buffers, not new model parameters. After manual
in-place weight edits, **explicitly call** `model.refresh_compute_buffers()`;
the production trainer does this after optimizer updates. An FP32 central
finite-difference perturbation should use the reference mode or refresh packed
weights after each perturbation. The optimizer's FP32 gradient accumulators
remain intact. Native BF16 backward GEMMs are *not* bitwise equivalent to FP32.

## 2. Optional learned-pooling fusion

```
MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=0  # DEFAULT, CuPy reference
MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=1  # EXPERIMENTAL, Triton fused pooling
```

At each completed block and compressor head, the learned pooling query
performs softmax attention over exactly the 128 in-block token states. The
new Triton fused forward computes dot scores, stable FP32 softmax, and the
weighted value in a single program. Its backward computes dK/dV and a
per-block dQuery, with deterministic FP32 reduction across blocks outside
the kernel. Only completed valid blocks participate. The existing compressor
QKV projection and full self-attention modes are unchanged.

This is **not yet validated on RTX 5080**, so fused pooling remains opt-in. It
may be slower for small document lengths or due to kernel register pressure.
Passing numerical parity tests alone does not justify enabling it by default.

### GPU parity first

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
    tests/test_dual_resolution_fused_pooling_0102j.py \
    tests/test_triton_dual_resolution_remaining_0102c.py \
    tests/test_dual_resolution_packed_projections_0102h.py
```

### Isolated pooling timings

```bash
MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_pooling_0102j.py \
    --blocks 1,16,64,256 --warmup 3 --repeats 20
```

### Full 20-layer A/B: fixed workload, matched warm-up

Both commands retain Triton local, global, compression attention and the
packed/BF16 projection defaults.  The only difference is learned pooling.

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=0 \
python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
    --context 4096 --layers 20 --batch-size 1 --compression-dim 512 \
    --score-last-k 128 --head-chunk-tokens 64 --activation-checkpoint block \
    --local-kernel triton --global-kernel triton --compression-kernel triton \
    --query-chunk 512 --warmup-iterations 2 --repeat-iterations 5 \
    --output 0102j_pool_reference.json

MINI_LLM_BACKEND=cupy MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=1 \
python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
    --context 4096 --layers 20 --batch-size 1 --compression-dim 512 \
    --score-last-k 128 --head-chunk-tokens 64 --activation-checkpoint block \
    --local-kernel triton --global-kernel triton --compression-kernel triton \
    --query-chunk 512 --warmup-iterations 2 --repeat-iterations 5 \
    --output 0102j_pool_triton.json
```

The full-model benchmark measures **processed** tokens; only the final 128
token predictions are scored. It does not time optimizer work. For a real
supervised-token throughput comparison, run matched document-first training
steps after a favorable warmed result.

## 3. Profiling, not to be confused with throughput measurement

The existing `--profile-steps 1` training switch now reports:

- `dual_resolution.local.forward/backward`
- `dual_resolution.global.forward/backward`
- `dual_resolution.compressor.forward/backward`
- `dual_resolution.pooling.forward/backward` (nested within compressor)

The profiler synchronizes GPU operations and therefore **distorts throughput**.
Use it only to select the next target; use `--profile-steps 0` for speed.
Parent and nested region times overlap. Do not add them together.

## 4. Validation and limitations

The NumPy CPU regression suite verifies default-reference behavior, explicit
packed parity, document/EOS handling, multi-batch training and compressor
gradients. GPU-only kernel compilation and numerical checks must be completed
on an RTX 5080 before treating this fusion as correct or faster. The 65K VRAM
limit and missing internal compressor positional encoding are separate issues,
not addressed in this patch.
