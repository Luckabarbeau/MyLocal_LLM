# Patch 0102Y: prepare Model B's local RoPE once per microbatch

## Problem

The optimized `rope_forward` already supports `PreparedRoPEPositions`, which
contains per-token cosine/sine factors and can be reused for Q/K across layers.
But Model B passed the raw `metadata["position_ids"]` into both local RoPE calls.
On the CuPy backend, each raw call reduces the GPU position array with
`xp.any()` and `xp.max().item()`, synchronizing with the CPU. There are two
local calls per layer, and 20 layers, plus up to 20 checkpoint replays.

Model B already caches the expanded summary positions and prepared *global*
RoPE factors. Local Q/K did not reuse these facilities.

## Change

The per-microbatch immutable device metadata now caches a single
`PreparedRoPEPositions` object for local token positions. Both local Q and K
reuse it; the global token query reuses the **same** prepared factors (global
head dimension equals local head dimension). The summary key positions retain
their own separate prepared table. Cache keys include the head dimension,
RoPE base, compute dtype, and position shape. Metadata is scoped to one
microbatch; there is no global cache retaining document tensors.

For ordinary production CPU-produced document metadata, the existing 0102F
cache persists across all layers and their checkpoint replays. The original
position values are untouched, including EOS restarts and packed documents.
This does not change learned parameters, the local window pattern, or the
attention causal masks. The existing general RoPE implementation serves as
the A/B reference and fallback.

## A/B switch

- `MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE=1` (default): reuse prepared
  local RoPE positions across Q, K, layers, and checkpoint replay.
- `MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE=0`: original raw-position local
  RoPE calls (use for matched baseline).

`MINI_LLM_MATRIX_POOL_PREPARED_GLOBAL_ROPE` remains independent. To measure
**only** the new local-branch optimization, leave the global switch at its
existing default of `1` in both A and B runs.

## Test

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_matrix_pool_prepared_local_rope_0102y.py \
  tests/test_matrix_pool_dual_resolution_0102o.py \
  tests/test_0080bu_rope_prepared.py
```

On an RTX 5080, test GPU numerical parity and compare a *fixed-length*
full-model training benchmark with the local switch set to `0`, `1`, then `0`
(A/B/A). Record forward and backward time, executed and supervised tokens/s,
and peak allocated memory. In the production trainer, use identical document
samples and checkpoint state for the comparison; metadata reuse can improve
CPU/GPU synchronization costs without reducing theoretical FLOPs.

No GPU throughput gain is claimed until measured in production.
