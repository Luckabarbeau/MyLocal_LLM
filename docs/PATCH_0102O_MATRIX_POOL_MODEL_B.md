# Patch 0102O — Model B matrix-pooled dual-resolution Transformer

Patch 0102O adds Model B as a separate architecture:

```text
dual-resolution-matrix-pool-1024-20l
```

The existing `dual-resolution-1024-20l` architecture and its block Transformer
compressor remain available and retain their prior defaults.

## A. Change implemented

Model B keeps all 65,536 original tokens in all 20 Transformer layers, but replaces
the secondary 128-token compressor Transformer with four content-dependent matrix
pooled summaries per completed block:

\[
Z=H W_s+B_{pos},\qquad
A=\operatorname{softmax}_{tokens}(Z),\qquad
S=A^T H.
\]

The default preset uses:

- 15 local query heads with a 512-token causal window.
- One leading local query head with a 4,096-token causal window.
- Four learned 1,024-wide summaries per completed 128-token block.
- Two global query heads of width 64.
- Independent global Q/K/V/O projections.
- Strict access only to summaries whose block end precedes the query position.
- Full-sequence supervision and the existing document-first metadata path.
- FP32 residual state and pooling accumulation with BF16 projection GEMMs.

The per-layer graph is:

```text
z = Norm(X)
local_delta = MixedWindowLocalAttention(z)
H_local = X + local_delta
summaries = MatrixPool(H_local)
global_delta = GlobalAttention(H_local, summaries)
attention branch output = local_delta + global_delta
```

The outer Transformer block adds `X` once. Backward separately returns the
normalized-input gradient and the direct `H_local <- X` residual gradient, avoiding
a duplicate local residual.

New configuration fields are backward-compatible and default to the original
compressor behavior:

- `compression_mode`
- `long_local_window`
- `long_local_heads`
- `summaries_per_block`
- `global_heads`
- `global_head_dim`
- `global_enabled`
- `pool_position_bias`

The Triton local attention operator now accepts leading long-window query heads
without extending the other heads' windows. The existing EOS/document mask remains
active. The dense global Triton path is reused by repeating each completed block's
metadata once per summary.

### Frozen-model inference

Incremental inference now supports Model B rather than routing it through the old
one-summary compressor assumptions:

- The local cache capacity is 4,096 when long heads are enabled.
- Each short query head still reads only its final 512 keys.
- Global cache shape is `[global_heads, max_blocks * summaries_per_block,
  global_head_dim]` per layer and batch row.
- Incomplete blocks retain post-local residual states, matching training pooling.
- A completed block inserts all four summary K/V rows after its final token's query,
  so the current block is never visible to itself.
- EOS resets local, incomplete-block, and summary rings before the next document.

CuPy BF16 projections are flattened to 2D GEMMs before restoring logical shapes;
this avoids CuPy's unsupported generic batched BF16 `matmul` path.

## Parameters

| Architecture | Parameters |
|---|---:|
| Existing dual-resolution model | 406,199,296 |
| Model B | 348,261,376 |
| Reduction | 57,937,920 (14.26%) |

The estimator exactly matches the arrays registered by a constructed tiny Model B.

## B. Correctness

CPU/reference suite:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q \
  tests/test_matrix_pool_dual_resolution_0102o.py \
  tests/test_dual_resolution_0102.py \
  tests/test_dual_resolution_full_model_0102d.py \
  tests/test_dual_resolution_multibatch_rope_0102g.py
```

Result: **43 passed, 2 skipped**.

CUDA/Triton suite:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q \
  tests/test_matrix_pool_dual_resolution_0102o.py \
  tests/test_triton_dual_resolution_local_0102b.py \
  tests/test_triton_dual_resolution_remaining_0102c.py
```

Result: **11 passed, 5 skipped**.

A direct CuPy BF16 frozen-inference run over two complete four-token test blocks
also completed with finite logits and eight summary entries.

Coverage includes:

- Matrix-pooling forward and directional derivatives.
- Token, score-matrix, and positional-bias gradients.
- Independent normalization and nonzero gradients for all summary channels.
- Mixed local windows and strict future-token causality.
- EOS/document isolation and expanded summary metadata.
- Partial blocks and invalid summary slots.
- Checkpoint replay parity.
- Production-width BF16 CuPy/Triton output, input-gradient, and parameter-gradient
  parity.
- Incremental frozen inference versus full training forward at every token.
- Original dual-resolution model regressions.
- A real tiny AdamW update and packed-buffer refresh.

## C. Performance

These are warmed synthetic full-model measurements with block checkpointing and
only the final 128 token rows scored. They are **processed-token rates**, not real
full-supervision document-first training throughput. Runs at 16K and 32K used only
one warmup and few repeats, so their variance is not characterized as rigorously as
the 4K run.

| Context | Forward | Backward | Total | Processed tok/s | Pool reserved | Device-use sample |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 101.35 ms | 381.47 ms | 483.26 ms | 8,476 | 3,394 MiB | 5,169 MiB |
| 16,384 | 358.29 ms | 1,473.86 ms | 1,832.63 ms | 8,940 | 7,154 MiB | 8,929 MiB |
| 32,768 | 717.68 ms | 2,941.10 ms | 3,659.28 ms | 8,955 | 12,210 MiB | 13,985 MiB |

The historical 406M 4K result was 88.48 ms forward, 361.96 ms backward, 451.69 ms
total, and 9,068 processed tok/s. This was not a same-session A/B comparison and
must not be interpreted as a precise Model B regression. Model B also performs the
requested one-head 4,096 local window and four summaries per block, which differ
from that model's workload.

| Metric | Existing 406M historical run | Model B local run | Difference |
|---|---:|---:|---:|
| Forward time | 88.48 ms | 101.35 ms | +12.87 ms |
| Backward time | 361.96 ms | 381.47 ms | +19.51 ms |
| Total time | 451.69 ms | 483.26 ms | +31.57 ms |
| Processed tokens/s | 9,068 | 8,476 | -6.5% |
| Supervised tokens/s | Not measured | Not measured | Synthetic tail-scored runs |
| GPU memory | Not matched | 3,394 MiB reserved | Not comparable |

No real document-first optimizer-step throughput run was performed. The tiny
optimizer integration benchmark is a correctness check only; it is not a production
throughput measurement.

### Memory limit

A 65K full-model run was not attempted. Reserved memory grew by about 5.06 GiB from
16K to 32K, while the 32K device-use sample already reached 13.99 GiB. The existing
FP32 layer-input checkpoint policy therefore predicts that 65K training will exceed
16 GiB. No sequence truncation, gradient detachment, or CPU activation offload was
introduced to conceal this limitation.

## D. FLOP efficiency

Counting uses the Patch 0102N convention: one multiply-add is two FLOPs; block
forward replay, projection gradients, attention probability reconstruction, and
chunked LM-head replay are included. Pointwise functions, masks, indexing, casts,
optimizer work, memory traffic, and kernel-launch overhead are excluded. Local and
global attention use the same tiled-execution convention as the 0102N comparison,
so these are executed-arithmetic estimates rather than a claim about achieved
tensor-core FLOP/s.

Estimated Model B training GFLOPs per source/supervised token:

| Component | 4K | 16K | 32K | 65K |
|---|---:|---:|---:|---:|
| Main QKV projections | 0.251658 | 0.251658 | 0.251658 | 0.251658 |
| 15×512 + 1×4096 local attention | 0.266384 | 0.320958 | 0.330053 | 0.334601 |
| Local output projection | 0.167772 | 0.167772 | 0.167772 | 0.167772 |
| SwiGLU4096 | 2.013266 | 2.013266 | 2.013266 | 2.013266 |
| Pool score and weighted sum | 0.001311 | 0.001311 | 0.001311 | 0.001311 |
| Global query projection | 0.020972 | 0.020972 | 0.020972 | 0.020972 |
| Summary K/V projections | 0.001311 | 0.001311 | 0.001311 | 0.001311 |
| Global output projection | 0.020972 | 0.020972 | 0.020972 | 0.020972 |
| Narrow global attention | 0.006554 | 0.026214 | 0.052429 | 0.104858 |
| Vocabulary head, full supervision | 0.268435 | 0.268435 | 0.268435 | 0.268435 |
| **Total** | **3.018634** | **3.092868** | **3.128178** | **3.185155** |

For a checkpointed linear projection, this convention assigns `2*d_in*d_out` to
the original forward, the same amount to block replay, and
`4*d_in*d_out` to input/weight gradients. Matrix pooling similarly assigns
`4*d_model*summaries` each to original forward and replay and
`8*d_model*summaries` to backward per source token. Attention entries include
original forward, replay, probability reconstruction, and Q/K/V gradients.

Using the audited 338M values from Patch 0102N and its measured 27,000 supervised
tok/s baseline:

\[
R_B=27000\frac{F_{338}}{F_B}.
\]

| Context/horizon | 338M GF/token | Model B GF/token | Same-effective-FLOP/s target |
|---:|---:|---:|---:|
| 4K | 2.329560 | 3.018634 | **20,837 tok/s** |
| 16K | 2.334667 | 3.092868 | **20,381 tok/s** |
| 32K | 2.367057 | 3.128178 | **20,431 tok/s** |
| 65K | Not established | 3.185155 | Not calculated |

This normalization supports roughly 20.4K–20.8K as a comparable-efficiency target
at 4K–32K. It is not a throughput prediction: the 338M model uses a fixed 4K deep
trunk plus historical memory, whereas Model B executes every source token through
all 20 layers. Memory traffic, tile occupancy, launch overhead, optimizer/offload,
data sampling, and target fill are not represented by the arithmetic ratio.

## E. Remaining bottlenecks

1. **SwiGLU backward and projection gradients.** These remain the majority of the
   checkpointed 20-layer backward, as established by Patch 0102N.
2. **Mixed-window local attention.** The one 4,096-window head adds meaningful work
   at 4K and increasingly approaches its full window at longer contexts.
3. **Global dK/dV reduction.** Four summaries per block make this narrow branch grow
   linearly with context despite only two heads.
4. **Checkpoint memory.** FP32 layer-input checkpoints prevent 65K training on the
   current 16 GiB policy before summary storage becomes significant.
5. **Inference performance.** Model B inference is correct but uses a simple
   per-head/per-row CuPy implementation; no fused decode kernel is claimed.

## F. Next action

The single highest-value next step is a real document-first 4K/16K optimizer-step
A/B against the existing 406M model, with full target scoring and identical sampled
documents. That measurement will separate arithmetic gains from document-length
fill, LM-head cost, optimizer offload, and host/GPU synchronization before another
kernel is optimized.

## Changed files

- `mini_llm/config.py`
- `train_model.py`
- `mini_llm/blocks/transformer_block.py`
- `mini_llm/blocks/transformer_block_inference.py`
- `mini_llm/inference_model.py`
- `mini_llm/ops/matrix_pool_dual_resolution_attention.py`
- `mini_llm/ops/dual_resolution_attention_inference.py`
- `mini_llm/ops/triton_dual_resolution_local.py`
- `mini_llm/ops/triton_dual_resolution_dense.py`
- `benchmarks/benchmark_dual_resolution_full_model_0102d.py`
- `tests/test_matrix_pool_dual_resolution_0102o.py`
