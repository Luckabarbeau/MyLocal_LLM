# 0080AJ3 — Experimental CUDA fusion of exact all-source vocabulary retrieval

## Objective

Reduce the number of CuPy launches and intermediate per-source arrays around
0080AJ2's batched QK/AV GEMMs. Maintain full-vocabulary, source-normalized,
learnable-bias attention, all-source cross-horizon gradients, shared embedding
key/value derivatives, and block-local memory. This is **opt-in** and still
**not authorized for 500M production training** until cached decoding and v2
checkpoints have passed the separately required parity tests.

## Algorithm switch

`SoftVocabularyRead(..., source_scope='all', execution='tiled',
all_source_algorithm='fused_mixture')`. On NumPy this calls the verified
`batched_mixture` implementation as a mathematical oracle. On CuPy it selects
`mini_llm/block_proposal/fused_vocab_cuda.py` (RawKernel CUDA). The legacy
`source_loop`, `source_mixture` and `batched_mixture` implementations remain
unchanged and available; `source_mixture` remains the default.

The GPU implementation **does not replace GEMM with scalar per-element CUDA**.
It retains batched cuBLAS QK, AV, backward dQ and dE matrix multiplies. The
forward RawKernel fuses the per-source normalized mixture, query-score
addition, tile maximum and online softmax-mass reduction into one launch per
vocabulary/block tile. The input score scratch is overwritten in-place with
unnormalized softmax weights for the AV GEMM. In backward, another kernel
recomputes mixture, weights, score VJP and normalized per-source score
responsibilities without global atomics. The resulting per-tile source VJP
is reduced into proposal-logit and source-bias gradients before applying
per-source log-softmax Jacobians.

For `K=4`, the backward transient derivative tensor is bounded by
`[H, block_tile, K_target, K_source, vocab_tile]`. At H=4,
block_tile=8, vocabulary_tile=2048 this buffer is 4 MiB. It is **not** cached
across vocabulary tiles or training groups. This trades a small bounded
scratch allocation for fewer separate CuPy source-loop operations. Full-
vocabulary logits and shared embedding gradients already exist outside that
scratch. No top-k, downsampling, detached branches or approximation of source
probabilities is used.

CUDA arithmetic is FP32 with stable max/sum reductions and online update;
minor floating-point differences from the batched reference are expected
because reduction order changes. The code has no packed/cached trainable
projection weights, uses parameters at every call, and does not accumulate
source gradients with nondeterministic atomics. This patch does **not**
enable BF16 GEMMs (only BF16 source/embedding inputs cast to FP32 before GEMMs),
as that is a separate numerical experiment.

## Tests

CPU (the new selector is routed to the batched oracle):

```bash
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_block_proposal_0080aj3_fused_contract.py \
  tests/test_block_proposal_0080aj2_batched_mixture.py
```

RTX 5080 (must pass before making any speed claim):

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q \
  tests/test_block_proposal_0080aj3_cupy.py \
  tests/test_block_proposal_0080aj2_cupy.py -x -vv
```

Same-data synchronized benchmark:

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.benchmark_block_proposal_vocab_streaming \
  --source-scope all --all-source-algorithm fused_pair \
  --block-groups 8 16 32 --repeat 5
```

This prints `batched_mixture` versus `fused_mixture` forward/backward time,
throughput, pool usage, and numerical differences in correction, proposal
logits, and decoder states. For stricter parity inspect full per-parameter
and embedding gradients through the GPU tests. The benchmark measures only
retrieval scratch path, **not** full 500M training, total GPU VRAM, or an
inference speedup. The environment used to author this patch has no CuPy/
CUDA runtime, and CUDA compilation and actual GPU speed have **not** been
verified here. If the fused implementation is slower or misses tolerance,
keep `batched_mixture` as the performance reference.

## Engineering caveats

- NVRTC compilation occurs lazily at first CUDA use. The benchmark warms up
  each implementation before measuring repeated passes.
- Tile sizes may need tuning; compare `--vocab-tile 1024/2048/4096`
  and `--block-tile 8/16` on the user's 16 GiB RTX 5080.
- Forward softmax and backward responsibilities are fused, but source logit
  normalization, GEMMs, and compact VJP reductions still use CuPy operations.
  End-to-end profiling is needed before designing any further kernels.
- Current training CLI and cached-generation implementation still reject
  experimental full-vocab v2; checkpoint migration, dense/routed cache parity
  and optional causal boundary h conditioning remain separate next patches.
