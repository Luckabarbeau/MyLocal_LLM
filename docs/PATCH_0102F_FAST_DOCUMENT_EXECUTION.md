# Patch 0102F — efficient document-first execution (experimental GPU performance)

Applies **after 0102E**. Architecture, weights, loss objective, document selection,
65,536-token maximum, optimizer, and checkpoint layout are unchanged.

## Problems isolated in the six-step 65K-cap trial

* Document-first sampling reduced the real target count but created new exact
  lengths on almost every microbatch. The 0102B/0102C Triton programs specialize
  on `n_ctx` / `NQ` / `NK` as compile-time constants. Arbitrary new sequence lengths
  thus create kernel variants rather than reuse the compiled 4K kernels.
* The same CPU document metadata was being reuploaded to GPU and checked with
  synchronizing `xp.any` reductions for **every** layer and checkpoint replay.
* Documents shorter than or equal to a compression block can have **no visible
  earlier complete block summary**. The old path still constructed unused
  compressor, global and gate operations, including backward work.
* Setting `--context-length 65536` can allow sampled documents larger than the
  current 16GB memory budget. The trial OOM'd in compressor backward. This
  patch **does not** solve long-sequence VRAM limits, and it does not truncate
  genuine documents or silently cap the 65K objective.

## Changes

1. **CPU document-metadata validation and per-batch GPU reuse.** On immutable
   CPU-produced metadata, check valid indices on CPU and cache the converted GPU
   buffers in the per-microbatch metadata dict. All layers and checkpoint replays
   reuse the same validated buffers. An explicit `position_ids` override bypasses
   reuse. Raw device-provided metadata still undergoes existing bounds checks.
2. **No-visible-summary short path.** If physical sequence length is <= 128,
   the model executes only causal local attention and normal Transformer FFN.
   The compressor, global K/V projection, global attention and gate do no work;
   their gradients are correctly zero. For longer sequences the original exact
   architecture is untouched. This also works for custom compression block sizes.
3. **Optional geometric execution-shape bucketing.** Set
   `MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET=geometric` to pad the physical length
   to one of a small number of geometric buckets (16, 32, 48, 64, 96, 128,
   192, 256, 384, 512, 768, 1024, ...). No genuine tokens are dropped;
   padding has zero loss mask; padded compression blocks are marked invalid;
   batch=1 still uses one document only. This trades some bounded padding for
   fewer Triton recompilations and steadier throughput. With the environment
   variable unset or `none`, 0102E's exact physical lengths remain the default.

No GPU speedup or 65K backward-capacity improvement is claimed until the actual
CUDA tests are run. CPU tests cover short path backward finite differences,
block-summary visibility, index validation, shape buckets, real trainer loss
and step token accounting.

## Recommended A/B trial

Repeat the same document-first 4K trial on independent checkpoints with:

```
MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET=none
# compare to
MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET=geometric
```

Keep `--context-length 4096`, `--batch-size 1`, block checkpointing and
three Triton switches. Add `--profile-steps 2` if available to inspect model
forward/backward, loss head, optimizer and data time. Compare warm iterations
and actual `supervised_tok/s`, not nominal `capacity_tok/s`.

The short path and cached metadata are always enabled for dual-resolution;
shape bucketing is deliberately opt-in until GPU parity/performance validation.

## GPU regression tests

```
MINI_LLM_BACKEND=numpy python -m pytest -q \
 tests/test_dual_resolution_fast_document_0102f.py
MINI_LLM_BACKEND=cupy python -m pytest -q \
 tests/test_triton_dual_resolution_local_0102b.py \
 tests/test_triton_dual_resolution_remaining_0102c.py
```

## Long-context constraint

Training on **all** positions in documents up to 65,536 remains the goal, but
on the RTX 5080 16GB the current 20-layer backward is OOM for some sampled
long documents. The next stage requires genuine peak-memory attribution and
sequence-chunk recomputation / CPU activation offloading (while keeping exact
full-sequence causal/global gradient flow). Do not simply skip long documents,
shrink them without explicit consent, or treat a 4K/8K training run as evidence
of 65K training correctness.
