# 0080AJ — Exact source-mixture factorization, all-proposal full-vocabulary read

Parent: **0080AH+AI**, after 0080AE, AF, AG. This patch changes *only* the
experimental all-source tiled vocabulary-read implementation and its benchmark;
it does **not** add direct boundary-h conditioning, production CLI training,
cached-generation parity, v2 checkpoint loading/saving, or fused CUDA kernels.

## Algebraic identity

With source log-probabilities `l[r,v] = logsoftmax(L[r,:]/T)[v]`,
target j, head m and bias `B[m,j,r]`, the previous all-source path used

`score[j,m,r,v] = l[r,v] + B[m,j,r] + scale * dot(Q[j,m],E[v,m])`,

and normalized across `(r,v)`. Because the query-dependent score and the
value embedding are independent of source r, its output is **exactly equal** to

`log_mix[j,m,v] = logsumexp_r(l[r,v] + B[m,j,r])`

`score_mixed[j,m,v] = log_mix[j,m,v] + scale * dot(Q[j,m],E[v,m])`

`R[j,m] = softmax_v(score_mixed[j,m,:]) @ E[:,m]`.

The proof is simple: marginalize the unnormalized numerator over r, and note
that summing the joint partition function over r and v is the same as summing
the marginalized unnormalized weights over v. It preserves every vocabulary
entry, the original source bias, source-specific normalization, and the
conditional query sensitivity of the read. It does **not** convert marginals
into a joint conditional distribution or add missing boundary-state context.

The reverse mode computes `dscore_mixed[j,m,v] = A_mixed[j,m,v] *
(upstream dot (E[v,m]-R[j,m]))`. For each source,

`gamma[j,m,r,v] = exp(l[r,v] + B[m,j,r] - log_mix[j,m,v])`,

`dlog_prior[r,v] += sum_(j,m) gamma[j,m,r,v] * dscore_mixed[j,m,v]`

`dB[m,j,r] += sum_v gamma[j,m,r,v] * dscore_mixed[j,m,v]`.

Then each source's proposal logits receive the required logsoftmax Jacobian:

`dL[r,v] = (dlog_prior[r,v] - P[r,v] * sum_u dlog_prior[r,u])/T`.

This is exact, including gradients into other proposal horizons. The shared E
receives its original value-path and score/key-path contributions exactly once
per (target,head,v), rather than separately per source.

## Memory and compatibility

- Online FP32 logsumexp over *all* proposal logits and vocabulary entries.
- Bounded `[block_tile*K,vocab_tile]` query-dependent score tiles. Temporary
  source log priors `[block_tile,K,vocab_tile]`, log mixtures
  `[block_tile,K,vocab_tile]`, and small per-source responsibility tensors.
- No full `[G,K,H,K,V]` attention maps; no `[G,K,V,d_head]` products; no top-k
  approximations. Backward recomputes tiles instead of caching probabilities.
- `all_source_algorithm='source_mixture'` is the new default for `tiled` and
  all source. `'source_loop'` is the previous 0080AH+AI implementation,
  preserved as an exact comparison oracle. `own` is mathematically unchanged.
- The algorithm selection is not yet an official production checkpoint or CLI
  architecture field. A mismatch between a forward cache and backward
  algorithm is rejected rather than silently mixing incompatible tapes.
- Legacy expectation baseline and v1 checkpoint metadata remain unchanged.
  Production use is still blocked until 0080AK–AL generation/checkpoint work.
- The source-axis factorization removes 3/4 of K=4 query-to-embedding GEMM
  repetitions, **not necessarily 3/4 of execution time**. Source mixing and
  manual-gradient reduction still cost multiple CuPy launches. On-GPU speedup
  is unknown until the paired comparison is executed on the RTX 5080.

## CPU tests (run from repository root)

```bash
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_block_proposal_0080aj_source_mixture.py \
  tests/test_block_proposal_0080ah_ai_streamed_full_vocab.py
```

CPU verifies full source-loop parity (outputs, attention statistics and all
parameters/inputs) across K=1/4, beta=0/nonzero, source biases and
non-unit temperatures, plus finite differences of E, input decoder states,
proposal logits, query/output projection and source bias, extreme logits,
source-shift invariance, cross-horizon gradient and one cache-consistency test.

## CuPy correctness (run on Ubuntu RTX 5080)

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q \
  tests/test_block_proposal_0080aj_cupy.py \
  tests/test_block_proposal_0080ah_ai_cupy.py -x -vv
```

CuPy tests compare old versus factored all-source modes on identical BF16 and
FP32 inputs, with the source bias and every gradient branch included, plus
full 65,280-vocabulary BF16 smoke and memory bound. **Not run on the CPU host:**
CuPy/CUDA is unavailable here.

## Controlled paired RTX 5080 benchmark

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.benchmark_block_proposal_vocab_streaming \
  --source-scope all \
  --all-source-algorithm both \
  --block-groups 4 8 16 32 64 \
  --repeat 5
```

Each group uses identical source logits, inputs, embeddings, projections and
upstream gradients for the original `source_loop` and optimized
`source_mixture` implementations. Reports per-algorithm mean forward/backward
and p95 combined latency, query positions/sec, module CuPy-pool allocations,
measured speedup over source_loop and largest absolute output/dL differences.
The reported `scored_source_vocab_entries_per_second` represents the number
of **logical** `(target,head,source,vocab)` entries covered, even though the
new method reduces their source axis algebraically before the expensive GEMM.

The paired comparison temporarily retains a small corrected-logit result and
one full `[G,K,V]` FP32 dL from the old method to check parity. The reported
pool occupancy for the second method may therefore include these comparison
buffers, and is **not a fair standalone peak memory comparison**. For this
reason, also run the default **unpaired** measurement:

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.benchmark_block_proposal_vocab_streaming \
  --source-scope all --block-groups 4 8 16 32 64 --repeat 5
```

These are module-only measurements. Do not extrapolate a final 500M training
speed or declare the experiment production-ready from this benchmark alone.

## Scientific next steps

1. Confirm CuPy numerical parity and benchmark **measured** speedup and VRAM.
2. If still dominated by the many Python/CuPy tile launches, consider a
   fused CUDA kernel that performs log-mixture, softmax and VJP with the same
   mathematical identity; benchmark representative real `G`, BF16 and V.
3. Add independently switchable `h -> DEC` boundary conditioning (0080AJ
   follow-on), cached-generation parity (0080AK) and safe strict v2 checkpoint
   transfer/resume (0080AL) before a 500M-from-scratch training run.
