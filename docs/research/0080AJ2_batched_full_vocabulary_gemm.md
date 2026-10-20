# 0080AJ2 — batched full-vocabulary GEMM (opt-in)

## Goal and constraints

The existing 0080AJ source-mixture implementation is the numerical reference.
This patch adds `all_source_algorithm='batched_mixture'` in the **tiled** all-source
read. It does not alter the default (`source_mixture`), teacher-forced decoder
integration, v1 checkpoint contract, production CLI or generation path.
It is a GPU performance experiment, not a production-mode switch.

## Equivalent mathematics

As in 0080AJ, the prior is

    log_prior[g,r,v] = L[g,r,v]/T - logsumexp_u(L[g,r,u]/T)
    log_mix[m,g,j,v] = logsumexp_r(log_prior[g,r,v] + bias[m,j,r])
    score[m,g,j,v] = log_mix[m,g,j,v] + beta/sqrt(Dh) * Q[m,g,j] @ E[m,v]

Scores are normalized over vocabulary, and output is the probability-weighted
sum of the **shared** embedding in each head. The query projection,
output projection, source bias, proposal logits, and shared embedding have
exact manual VJPs, including the source log-softmax correction.

The batched implementation moves the head dimension outside GEMMs and performs
batched `[H, block_tile*K, Dh] @ [H, Dh, vocab_tile]` and batched
`[H, block_tile*K, vocab_tile] @ [H, vocab_tile, Dh]` rather than independently
issuing one GEMM per attention head. All proposal sources share a query GEMM.
All heads are updated in a single FP32 online-softmax loop; backward recomputes
the same bounded vocabulary tiles and multiplies `weight.T @ upstream` and
`dscore.T @ query` with batched GEMM. The score/attention scratch tensor is
bounded by `O(H * block_tile * K * vocab_tile)`; no `[H,G,K,V]` attention
cache or `[V,Dh]` per-token embedding expansion is retained.

The source mixture still has an explicit `K` loop, and the vocabulary remains
tiled. This patch does **not** implement fused CUDA softmax/reduction kernels.
GEMM efficiency and CUDA-launch reduction are hypotheses that must be
measured, not assumed.

## CPU tests and GPU verification

CPU mathematical suite:

```bash
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_block_proposal_0080aj2_batched_mixture.py \
  tests/test_block_proposal_0080aj_source_mixture.py
```

On the RTX 5080:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q \
  tests/test_block_proposal_0080aj2_cupy.py \
  tests/test_block_proposal_0080aj_cupy.py -x -vv
```

Paired same-input microbenchmark of the **original exact** 0080AJ mixture vs
batched GEMM (scratch-only, *not* 500M training throughput):

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.benchmark_block_proposal_vocab_streaming \
  --source-scope all --all-source-algorithm batched_pair \
  --block-groups 4 8 16 32 64 --repeat 5
```

Both implementations use the same weights, logits and upstream derivatives.
`speedup_over_source_mixture`, forward/backward times, query positions/s and
CuPy pool usage are measured. The benchmark checks differences in corrected
output, proposal-logit gradient and decoder-state gradient. It does *not*
certify complete optimizer-step timing, all parameter gradients, total GPU
VRAM, real training throughput or inference speed.

For tile tuning, hold other options fixed and compare `--block-tile 8`,
`--block-tile 16`, `--block-tile 32` and `--vocab-tile 1024/2048/4096`.
Do not conclude faster performance from this patch without the actual CuPy
results; if `batched_mixture` is slower, retain `source_mixture` as default.

## Correctness

The numerical comparison tests cover nonzero positional source bias,
K=1/4, nontrivial query scale and temperature, multiple block/vocab tile
sizes, all gradients (including E in its key and value roles), cross-horizon
proposal-gradient coupling, finite differences, and extreme logits.
The new algorithm is included in the forward tape and cannot be switched
between forward and backward. Source/bias normalizers are recomputed exactly
from the canonical parameters without a stale packed projection cache.

## Remaining engineering gates

No CuPy device is available in the patch-construction environment, so GPU
results must be collected on the workstation. After measuring this microkernel,
inspect per-operation CUDA timing/kernel launch counts; fusion is a separate
optimization step if bottlenecks remain. Before any 500M training run, finish
causal boundary conditioning, teacher-forced/cached-generation parity, strict
v2 metadata and full-model warm-start/resume tests. The existing CLI correctly
refuses experimental v2 training until those gates are satisfied.
