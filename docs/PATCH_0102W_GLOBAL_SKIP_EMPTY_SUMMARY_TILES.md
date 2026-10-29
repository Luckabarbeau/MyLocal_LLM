# Patch 0102W — Skip fully masked global attention tiles

The 0102C global Triton implementation previously performed QK/PV, dQ, and
K/V-gradient tensor-core operations for every query/summary tile, even when
**no query in the tile can see any of its summary keys** due to causal completed
block visibility, EOS document isolation, or an invalid summary slot.

This patch adds the experimental switch:

```bash
MINI_LLM_DUAL_RESOLUTION_GLOBAL_SKIP_EMPTY_TILES=1
```

When enabled, a CTA checks the existing exact validity mask before expensive
GEMMs. Entirely invisible tiles contribute nothing to softmax normalization or
gradients, so the K/V loads and dot products may be omitted exactly. The
forward and backward paths read the same setting from the stored forward cache,
protecting against an environment change between them. The default remains **0**
until CUDA parity tests and warmed full-model A/B results are available.

This does not eliminate CPU/CuPy launches, the learned-pooling computations,
global Q/K/V projections, or the metadata/mask checks for empty tiles. It also
does **not** eliminate all of the short-context overhead; the first 128 queries
represent only 3.125% of a 4096-token sequence. Longer-term improvements can
bound the iteration ranges or directly skip empty query tiles, after separate
correctness and performance validation. No change to model parameters, masks,
checkpoint format, or supervision.

## Validation

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
  tests/test_global_skip_empty_tiles_0102w.py \
  tests/test_triton_dual_resolution_remaining_0102c.py
```

## Matched benchmark (one process per mode, same full supervision)

```bash
for context in 4096 16384 32768; do
  for mode in 0 1 0; do
    MINI_LLM_BACKEND=cupy \
    MINI_LLM_DUAL_RESOLUTION_GLOBAL_SKIP_EMPTY_TILES=$mode \
    MINI_LLM_MATRIX_POOL_FUSED_FORWARD=0 \
    MINI_LLM_MATRIX_POOL_FUSED_BACKWARD=0 \
    python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
      --architecture matrix-pool --context "$context" --layers 20 \
      --batch-size 1 --score-last-k "$context" \
      --head-chunk-tokens 512 --activation-checkpoint block \
      --local-kernel triton --global-kernel triton \
      --warmup-iterations 3 --repeat-iterations 7 \
      --output "0102w_${context}_skip_${mode}.json"
  done
done
```

The repeated zero-mode run is an A/B/A thermal and clock drift control. Note
that the benchmark does not include an optimizer update, and its full-sequence
target count is only directly comparable to other full-sequence runs.
