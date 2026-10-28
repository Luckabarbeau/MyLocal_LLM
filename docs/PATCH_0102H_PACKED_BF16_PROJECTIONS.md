# Patch 0102H — Dual-resolution packed BF16 projection optimization

This **opt-in** optimization reuses the 338M model's general projection principles without changing the 406M architecture or serialized parameter names.

## What changed

- **Persistent packed QKV/KV weight buffers**, refreshed after optimizer steps (including after a checkpoint resumes). Groups: main QKV, each compressor layer QKV, pooling KV, global-summary KV. Avoid 3/2 independent matrix multiplications per forward and per backward. Existing `Parameter` objects, names, and checkpoint format are unchanged.
- **Contiguous FP32 gradient views** for packed parameter groups. The packed backward uses one QKV/KV weight-gradient and input-gradient GEMM. Gradient views remain independent slices in the existing optimizer interface. `model.refresh_compute_buffers()` rebuilds the derived weights after updates.
- **Native BF16 projection backward GEMMs** by default within the opt-in path, like the mature 338M attention implementation. FP32 gradient accumulation remains mandatory; BF16 operand rounding makes the mixed-precision path *not bit-identical* to the FP32 reference. FP32 projections/tests use their existing dtype.
- **BF16 Q/K/V inside Triton compressor attention**, avoiding the unconditional FP32 promotion and slower `tf32x3` dot path. In reference mode, compressor Q/K/V still use the original FP32 path.
- **Nonoverlapping compressor gradient scatter**: precompute valid block indices once per document batch on the CPU, reuse across 20 layers and checkpoint replay, and assign completed-block gradients directly. Invalid/padded summary slots do not participate.
- Print selected attention and packed-projection modes at `ExtendedTrainer` startup.

No position encodings were added inside the compressor. That is a *separate architectural choice* and needs distinct order-sensitivity and learning-quality experiments; adding RoPE there now would confound performance comparisons.

## How to enable

```bash
export MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=1
export MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS=1
```

Disable the fast path with `MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS=0` for an A/B baseline. To isolate packed GEMMs from BF16 gradient operand rounding, set `MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS=0` (packed FP32 gradient GEMMs, BF16 compressor QKV still applies when Triton compression is selected).

Use the existing RTX 5080 production command with **all three Triton kernel environment variables**. Avoid mixing checkpoint states in benchmark comparisons; run separate checkpoint/log directories for fast and reference. For memory-safe initial full-model comparisons use `--context 4096 --layers 20 --batch-size 1 --activation-checkpoint block` with `benchmarks/benchmark_dual_resolution_full_model_0102d.py` and `--local-kernel triton --global-kernel triton --compression-kernel triton`.

## Tests

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_dual_resolution_packed_projections_0102h.py \
  tests/test_dual_resolution_0102.py \
  tests/test_dual_resolution_multibatch_rope_0102g.py

MINI_LLM_BACKEND=cupy python -m pytest -v \
  tests/test_dual_resolution_packed_projections_0102h.py \
  tests/test_triton_dual_resolution_remaining_0102c.py
```

The CPU tests check exact FP32/FP64 numerical parity, gradient accumulation, actual weight refresh and valid-block scatter without overlap. The CUDA test is explicitly *not yet run* in the patch authoring environment. Its BF16 parity tolerances account for changed rounding. Run and inspect GPU tests before extended training.

## Known limitations and next measurement

- Local/global/compressor Triton kernels and their exact causal masks are unchanged.
- Optimized weights are additional BF16 buffers; they increase persistent device memory somewhat. CPU/GPU memory should be measured on full 20-layer runs.
- This patch does **not** fix full-65K training OOM, nor guarantee any particular throughput. Compare steady-state complete optimizer-step times and supervised tokens/sec (not attention-only token counts), using matched document samples and compilation/warm-up conditions.
- Profiling in 0102F previously suggested attention backward dominates; after this patch, profile to determine whether QKV projection, pooling/normalization, or Triton global backward dominates next.
