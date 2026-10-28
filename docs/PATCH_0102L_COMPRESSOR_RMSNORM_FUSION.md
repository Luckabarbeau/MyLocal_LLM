# Patch 0102L: optional fused Triton compressor RMSNorm

This is an additive performance experiment atop 0102K. No architectural or
checkpoint changes. Both the compressor's contextual-attention RMSNorm and
its learned-pooling RMSNorm can use one Triton forward and one Triton backward
kernel each. The default remains the existing CuPy reference path until CUDA
numerical-parity and end-to-end speed measurements have passed.

`MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM=1` enables the new implementation on
CuPy only; `=0` disables it. It does **not** change the Transformer block's
own normalization (that already uses the existing optimized training path).

Implementation details:

- Forward uses a row-tiled Triton RMSNorm and retains just the per-row FP32
  inverse RMS and input. BF16 inputs return BF16 outputs.
- Backward computes FP32 `dX` and per-tile gamma gradient contributions,
  with a final GPU sum. FP32 parameter gradients continue accumulating in
  their existing buffers. The formula and epsilon match the reference.
- No FP32 `normalized` tensor is retained. The old CuPy implementation is
  untouched for comparison, and its cached format still works in backward.
- The path can handle unequal document lengths through the existing valid
  block compaction. No changes to causal masks or summary gating.

## GPU tests

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
  tests/test_dual_resolution_fused_rmsnorm_0102l.py \
  tests/test_dual_resolution_fused_pooling_0102j.py
```

## Microbenchmark

```bash
MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_rmsnorm_0102l.py \
  --blocks 1,16,64,256,512 --warmup 3 --repeats 12
```

## Matched warmed full-model benchmark

Run the same command for `mode=0` and `mode=1`; keep fused pooling enabled
in both so the new RMSNorm switch is the only changed variable:

```bash
for mode in 0 1; do
  MINI_LLM_BACKEND=cupy \
  MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=1 \
  MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM=$mode \
  python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
    --context 4096 --layers 20 --batch-size 1 --compression-dim 512 \
    --score-last-k 128 --head-chunk-tokens 64 --activation-checkpoint block \
    --local-kernel triton --global-kernel triton \
    --compression-kernel triton --query-chunk 512 \
    --warmup-iterations 2 --repeat-iterations 5 \
    --output "0102l_rmsnorm_${mode}.json"
done
```

**Important:** these are processed-context tokens without optimizer updates.
The actual production target is supervised tokens/s from `train_model.py`.
The 32K context cap and 65K long-context OOM remain separate questions.
