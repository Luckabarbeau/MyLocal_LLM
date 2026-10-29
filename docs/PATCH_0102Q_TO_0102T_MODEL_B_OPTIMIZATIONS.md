# Model B sequence of optimization experiments: 0102Q–0102T

Applies **after 0102P**, using `git apply` in Q, R, S, T order. No
parameter layout, architecture, optimizer, causal mask, or checkpoint changes.

- **0102Q** (default on): when metadata proves every complete 128-token
  block occupies the initial contiguous token range, view existing FP32
  hidden memory instead of indexed-gathering a new `[B,blocks,128,D]` tensor.
  Backward also uses a contiguous gradient scatter. For batch >1 with
  unaligned tails or irregular EOS block indices, keep indexed fallback.
  A 65K FP32 gather can be 256 MiB **per layer invocation**; actual savings
  must be measured and include checkpoint replay.
- **0102R** (default on): share four expanded summary-metadata arrays across
  layers and checkpoint replay. Avoid allocating an all-zero global result
  when the global branch is absent. Minor fixed overhead reduction.
- **0102S** (OFF until GPU parity): experimental fused Triton scoring, position
  bias, token softmax, and weighted pooling **forward only**. Existing
  reference backward consumes its probabilities. The fused kernel loads each
  block twice; measure, do not assume it beats the original score GEMM.
- **0102T**: add isolation/rollback environment flags for Q and R.

## Environment controls

| Environment variable | Default | Meaning |
|---|---:|---|
| `MINI_LLM_MATRIX_POOL_CONTIGUOUS_VIEW` | `1` | 0102Q guarded view/scatter |
| `MINI_LLM_MATRIX_POOL_REUSE_SUMMARY_METADATA` | `1` | 0102R cached expanded metadata |
| `MINI_LLM_MATRIX_POOL_FUSED_FORWARD` | `0` | 0102S opt-in Triton pooling |
| `MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE` | `0` | 0102P opt-in additive gradient scatter |

For matched comparisons run an A/B/A warmed **full-model** benchmark, with the
same context, scored token count, dtype, optimizer configuration, and activation
checkpoint policy. Ensure all shapes have compiled before timing. Synthetic
benchmark processed tok/s is distinct from real supervised tok/s.

Example full-supervision 4K comparison (RTX 5080, CUDA/CuPy/Triton):

```bash
for mode in 0 1; do
  MINI_LLM_BACKEND=cupy \
  MINI_LLM_MATRIX_POOL_CONTIGUOUS_VIEW=1 \
  MINI_LLM_MATRIX_POOL_REUSE_SUMMARY_METADATA=1 \
  MINI_LLM_MATRIX_POOL_FUSED_FORWARD="$mode" \
  python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
    --architecture matrix-pool --context 4096 --layers 20 \
    --batch-size 1 --score-last-k 4096 --head-chunk-tokens 512 \
    --activation-checkpoint block --local-kernel triton \
    --global-kernel triton --warmup-iterations 3 \
    --repeat-iterations 7 --output "0102s_fused_pool_${mode}.json"
done
```

Run the GPU numerical tests **before** enabling the fused path:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
  tests/test_matrix_pool_fused_0102s.py \
  tests/test_matrix_pool_dual_resolution_0102o.py
```

All CUDA speedups are **unmeasured** in the assistant's CPU-only environment.
Do not default-enable `MINI_LLM_MATRIX_POOL_FUSED_FORWARD` until tests
pass and full-model speed is favorable. If no speedup is seen, leave disabled.
