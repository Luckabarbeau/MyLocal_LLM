# Patch 0102U: optional exact matrix-pool backward

## Why

In the matched 4K Model B full-supervision benchmark, fused forward pooling
changed 16,572 to 16,629 processed tokens/s (+0.34%, within normal measurement
variation), while backward remained ~187–188 ms. Therefore **do not** make the
0102S fused forward path a default based on those measurements.

0102U adds an experimental two-stage Triton matrix-pooling backward, replacing
several separate CuPy GEMMs, softmax-backward intermediates, and the full
per-token dBlocks allocation for fully contiguous completed blocks.

The kernel computes the same derivatives:

- `dP[t,r] = dot(H[t,:],dS[r,:])`
- `dZ[t,r] = P[t,r] (dP[t,r] - sum_u P[u,r] dP[u,r])`
- `dH[t,d] = sum_r P[t,r] dS[r,d] + dZ[t,r] W[d,r]`
- `dW[d,r] = sum_t H[t,d] dZ[t,r]`; positional-bias gradients sum dZ.

The implementation uses FP32 accumulation and computes per-block contributions
before reducing score/bias gradients on the GPU. It can add dH directly into an
existing full-sequence FP32 gradient buffer when 0102P is enabled, without
allocating a separate dBlocks array. The reference fallback remains for NumPy,
non-contiguous input, irregular packed documents, and batch tails.

## How to test

Default remains **off** pending RTX 5080 compilation, numerical parity, and
full-model benchmark. For GPU correctness, first run:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
  tests/test_matrix_pool_fused_backward_0102u.py \
  tests/test_matrix_pool_dual_resolution_0102o.py
```

Matched full supervision A/B (keeps forward reference and enables 0102P so
kernel can write directly into the global query gradient):

```bash
for mode in 0 1 0; do
  MINI_LLM_BACKEND=cupy \
  MINI_LLM_MATRIX_POOL_FUSED_FORWARD=0 \
  MINI_LLM_MATRIX_POOL_CONTIGUOUS_VIEW=1 \
  MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE=1 \
  MINI_LLM_MATRIX_POOL_FUSED_BACKWARD="$mode" \
  python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
    --architecture matrix-pool --context 4096 --layers 20 \
    --batch-size 1 --score-last-k 4096 --head-chunk-tokens 512 \
    --activation-checkpoint block --local-kernel triton \
    --global-kernel triton --warmup-iterations 3 \
    --repeat-iterations 7 --output "0102u_pool_backward_${mode}.json"
done
```

The final `0` repeats the baseline; the JSON file is overwritten for that final
run, so copy an earlier file before executing if all three raw records are desired.
Test at 16K/32K after confirming that the 4K integration is correct.

Do not claim an unmeasured speedup, and keep this behind
`MINI_LLM_MATRIX_POOL_FUSED_BACKWARD=1` until GPU results warrant enabling it.
