# Patch 0102G — unequal-length batch summary RoPE

## Symptom

`--model dual-resolution-1024-20l --batch-size 2`, document-first sampling,
checkpointing, and either CuPy or Triton attention can fail in
`DualResolutionAttention.forward -> rope_forward(summary_k_pre, ..., block_end_positions)`
with `ValueError: position_ids must be non-negative`.

## Root cause

Document-first training with `batch_size > 1` pads the shorter document to the
longest sampled document in the microbatch. `build_dual_resolution_metadata`
allocates the same summary count `M` for all rows. A row with fewer complete
128-token blocks uses `block_valid=False` and `block_end_positions=-1` for its
absent summary slots. These are valid *masked* sentinels, but the summary-key
RoPE calculation was incorrectly receiving the raw negative positions **before**
the global-attention valid-mask was applied.

## Fix

`_prepare_metadata` now creates a separate cached `block_rope_positions` array
with zero in invalid slots and the original end position in valid slots.
Only summary-key RoPE consumes the sanitized positions. Global-attention causal
visibility keeps the original positions and unchanged `block_valid` mask.
There is no new trainable parameter, architectural change, or altered
attention visibility. The prepared device metadata is reused across layers and
block checkpoint replay, so sanitization is not repeated for every layer.

## Tests

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
    tests/test_dual_resolution_multibatch_rope_0102g.py
```

CUDA/Triton, on the RTX 5080:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
    tests/test_dual_resolution_multibatch_rope_0102g.py \
    tests/test_triton_dual_resolution_local_0102b.py \
    tests/test_triton_dual_resolution_remaining_0102c.py
```

The new tests explicitly check unequal valid-summary counts, original metadata
preservation, row-wise forward/backward and gradient equivalence, and a tiny
batch-2 trainer update. CUDA test selects all three Triton kernels and executes
forward and backward. GPU verification must be performed on a CUDA host.

## Training guidance

First test with `--batch-size 2 --grad-accum-steps 2 --max-run-steps 2` before
committing to `--grad-accum-steps 128` (256 documents per update). Unequal
document lengths still pad to the longest row and may waste GPU execution;
0102G is a correctness fix, not a batch packing or speed optimization.
