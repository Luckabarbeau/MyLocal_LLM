# 0080AH+AI (combined) — full-vocabulary streamed read, own + all proposal sources

Parent patch: **0080AG**. This is **one** revertible patch implementing both planned 0080AH and 0080AI scopes. It does **not** activate the production training CLI or v2 checkpointing, cached generation, boundary conditioning, or claim an RTX 5080 result.

## Contract and mathematics

For each target j, head m, source r and vocab v, with `E` split into head slices, define

- `own`: `S[j,m,v] = L[j,v]/T + beta * (Q[j,m] @ E[v,m]) / sqrt(d)`; normalize across all V.
- `all`: `S[j,m,r,v] = logsoftmax(L[r,:]/T)[v] + beta * (Q[j,m] @ E[v,m]) / sqrt(d) + B[m,j,r]`; normalize across **K * V**, **not** individually over each source after applying the query. `B` is a new learned `[heads,K,K]` table initially zero. This is necessary to prevent the independent horizon logit partition functions from changing source weights.

Return `R[j,m]=sum_{r,v} A[j,m,r,v] * E[v,m]`, concatenate heads, apply the zero-initialized output projection, and add to decoder DEC states only after the selected layer. For own-source with beta=0, `R=P_j@E`. For all-source with beta=0, zero `B` gives the equal mixture of all source expected embeddings.

Manual VJP uses `dS = A * (g·E_v - g·R)` and accumulates the **value term** `A * g`, **key term** `(beta/sqrt(d))*dS*q` to E, `dq`, source-bias gradients and output/query projection gradients. Own-source returns `dL=dS/T`. All-source **first** sums `dS` over all target queries and heads into each source r, then applies the source logsoftmax Jacobian:

`dL[r,v] = (dS_source[r,v] - softmax(L[r,:]/T)[v] * sum_u dS_source[r,u]) / T`.

Thus DEC1 gradients genuinely reach other horizons' logits. The existing decoder still adds the residual-prior and legacy PLAN-expectation derivatives; `TwoStageBlockModel` still adds proposal CE. There is only one main body forward/backward per training window.

## Memory and kernels

`mini_llm/block_proposal/streamed_vocab_read.py` is a shared NumPy/CuPy tiled implementation (not a RawKernel). The CUDA code uses FP32 2-D matrix multiplications and reductions and BF16-to-FP32 tile conversions; model parameters may remain BF16, gradients accumulate FP32. Tile sizes are explicit (`vocab_tile=2048`, `block_tile=8` by default). Its extra attention workspace is `[block_tile*K, vocab_tile]`, plus similarly bounded scratch and `[block_tile*K,d_head]` accumulators. Neither the forward nor its per-call cache stores a full `[G,K,H,K,V]` attention tensor or a `[*,V,d]` array. The backward **recomputes** scores/weights and accumulates keys and values through bounded GEMMs. It does allocate **one full `[G,K,V]` FP32 returned gradient** as required by the existing proposal-head API, **one BF16 proposal-logit copy** in its cache to protect against in-place CE, and a `[G,K,V]` original logits tensor owned by the caller. These costs must be included in realistic training VRAM accounting.

The all-source version first obtains stable per-source `logsumexp(L/T)` by streaming vocabulary tiles, then streams attention across all source/vocab tiles using the stable online-softmax recurrence for max, mass and weighted value. Backward regenerates attention probabilities from saved small attention log-normalizers. For all-source, a final streamed pass applies the logsoftmax VJP. It does not ever truncate vocabulary support. The source-bias table is trainable and kept separate from the original decoder RNG initialization. `parameters()` registers E only once through the main model.

**Important performance limitation:** CuPy's FP32 GEMMs and Python loop over heads, source and vocabulary are a **correctness-first bounded implementation**, not yet fused BF16 tensor-core kernels. The exact GPU execution time, TFLOPS, synchronization behavior and full-model group-size headroom are unknown until tested. The original desired <=2x slowdown is a **target**, not an achieved measurement. If this implementation is too slow, profile the tile kernels and fuse score/softmax/value/VJP without changing the normalized math or using top-k approximations.

## Safety and compatibility

- Baseline default `expectation` and v1 RNG/parameters/metadata remain unchanged.
- `experimental_cpu_full_vocab=True` supports tiny `own|all` NumPy reference (V<=512).
- `experimental_gpu_full_vocab=True` explicitly enables tiled `own|all` CuPy teacher forcing. It is **direct Python only**, *not* an exposed production CLI mode.
- Both experimental settings require `mode=full_vocab`, `boundary_condition=off`. Cached generation continues to reject new decoder variants (0080AK) and v1 architecture checkpoint serialization continues to reject them (0080AL). **Do not resume or overwrite any existing production checkpoint.**
- 0080AJ direct `h` conditioning remains future work. The new interface does not reconstruct the joint distribution from marginals in general.

## CPU verification (run after applying 0080AE, AF, AG, then this combined patch)

```bash
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_block_proposal_0080af_soft_vocab_read.py \
  tests/test_block_proposal_0080ag_decoder_integration.py \
  tests/test_block_proposal_0080ah_ai_streamed_full_vocab.py
```

The new CPU suite runs reference-vs-tiled own/all, K=1/4, beta=0/nonzero, source-bias/logit/embedding/projection finite differences, source-shift invariance, future-token causality, main-body single backward, and a **real V=65,280, K=4, all-source** forward/backward with small embedding width. The original CPU baseline suites must continue to pass.

## CuPy CUDA smoke (run on Ubuntu RTX 5080)

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q \
  tests/test_block_proposal_0080ah_ai_cupy.py -x -vv
```

This tests CuPy vs NumPy tiny-V output, all-source cross-horizon gradients, BF16 large vocabulary own/all scratch sizes, and BF16 actual decoder-layer forward/backward for both source modes. **Not executed in the patch-building environment**, which has no installed CuPy/CUDA GPU. Treat CUDA compatibility as unverified until this command completes on the workstation.

## Standalone realistic-shape CuPy throughput, not full model speed

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.benchmark_block_proposal_vocab_streaming \
  --source-scope own --block-groups 4 8 16 32 64 --repeat 3
MINI_LLM_BACKEND=cupy python -m benchmarks.benchmark_block_proposal_vocab_streaming \
  --source-scope all --block-groups 4 8 16 32 64 --repeat 3
```

The benchmark explicitly labels module-only throughput, reports synchronized forward/backward timing and allocated/reserved CuPy pool bytes; it cannot certify peak VRAM during the full 500M training backward. Choose a safe production group size only after profiling the entire model on 16 GiB and verifying full-run numerical parity, cached generation and v2 checkpoint warm-start/round-trip.
