# 0102Z — Deterministic stride-16 original-token head for Model B

## Preset

`dual-resolution-matrix-pool-sparse-1024-20l`

The previous `dual-resolution-matrix-pool-1024-20l` preset is unchanged.
The new preset keeps the same 20×1024 FFN4096 transformer and *exactly the same
learned tensors*. All 20 layers retain their compressed historical attention
(4 learned summaries per 128-token block, two 64-dimensional global heads)
and their original 4096-token sliding head (query head 0).

Layers **3–18** (1-based) replace query head **1** of the 15 local-512
heads with one stride-16 original-token historical head. Sparse phases are:

`0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15`

In a given layer, query at absolute tensor index `q` may attend original
K/V tokens at indexes `phase + 16*j <= q` **only in the same EOS-delimited
document**. These are the existing layer's shared GQA key/value projections
and carry ordinary end-to-end gradients; the phase is not a learned router.
At 65,536 tokens, each sparse head has at most **4,096 keys**.

A query with no eligible sparse keys produces **zero for that one head**.
This can happen in the first 0–15 tokens of a document. No fallback is
performed; its other fifteen local/long heads remain available.
The phase is defined over the **absolute sequence tensor index**, not the
within-document RoPE position. This is necessary for the constant-stride K/V
view; document-ID equality prevents any cross-document leakage.

## Computation and memory

The Triton sparse path reuses existing local Q/K/V projections. Key/value
views use `[:, phase::16]` (no full 65K gather). The preexisting
matrix-pool global tiled-attention kernels perform exact causal attention,
with new sparse-specific forward/backward bounds. Backward scatters its
FP32 K/V gradients into the corresponding existing GQA group after the
other 15 local query heads have contributed, without floating-point atomics.

Incremental inference stores only the selected original K/V entries for
this head: at most ~4096 keys/values per layer for the 65K horizon, as well
as the existing 4096-token local KV ring and compressed summary cache.
Document EOS resets both compressed and sparse caches.

## Starting from an existing Model B checkpoint

Use `--init-from` to initialize matching **weights** from Model B. Do not
use `--resume-from` when switching variants: that option intentionally
restores the **old model configuration** from its `config.json`, along with
the optimizer and step state. `--init-from` creates a fresh optimization run.

Example short smoke training (adapt paths to your repository):

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL=triton \
MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL=triton \
.venv/bin/python -u train_model.py \
  --model dual-resolution-matrix-pool-sparse-1024-20l \
  --init-from ./checkpoints/0102_model_b_matrix_pool_32k_100k \
  --objective ar --precision bf16-mixed --runtime-profile consumer-gpu \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --context-length 4096 --batch-size 2 --grad-accum-steps 32 \
  --optimizer hybrid-muon --gradient-accumulation-weighting supervised_tokens \
  --warmup-steps 750 --total-steps 100000 --max-run-steps 2 \
  --peak-lr 3e-4 --grad-clip 1 --weight-decay 0.1 \
  --checkpoint-dir ./checkpoints/model_b_sparse_0102z_test \
  --log-file ./logs/model_b_sparse_0102z_test.csv
```

## Validation

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_sparse_stride_history_0102z.py \
  tests/test_matrix_pool_dual_resolution_0102o.py
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q \
  tests/test_sparse_stride_history_0102z.py
```

GPU parity **must** be run on an RTX 5080 before production use. Compare
supervised-token throughput, active memory, end-to-end gradients and EOS-safe
long-range retrieval against the unchanged Model B checkpoint. Speedup or
quality improvement is not assumed. Triton compilation time should be
excluded from benchmark timings.
