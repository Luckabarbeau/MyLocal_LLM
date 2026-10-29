# Patch 0102X — Source-preserving length-aware dual-resolution batches

Model B's document-first sampler remains unchanged. With batch size 2 or 3,
independently drawn documents have widely different lengths; the shorter rows
are EOS-padded to the longest row, and the Transformer still executes padding.

This patch optionally **pre-samples one optimizer step and re-pairs the already
sampled document windows in descending length order within each source**. It
rebuilds the dual-resolution document IDs, positions, completed-block indices,
block-valid flags, and final training masks for the new microbatches. No source
sampling weights, document windows, or supervision labels are discarded.

Enable before launching production training:

```bash
export MINI_LLM_DUAL_RESOLUTION_ACCUM_LENGTH_BUCKETS=1
```

Requires `--gradient-accumulation-weighting supervised_tokens`,
`--grad-accum-steps >= 2`, and the dual-resolution AR objective. The existing
legacy `MINI_LLM_AR_ACCUM_LENGTH_BUCKETS` is **not** used. Default is **off**
so old training configurations behave exactly as before.

The step log now includes `dual_bucket_rows=OLD->NEW`, `dual_bucket_saved=...%`,
and `dual_bucket_strata=...`, and `executed_tok/s` measures the actual repacked
shapes rather than the original sampler padding. One caveat: pre-sampling
changes the order of documents within an optimizer step, so training is
mathematically target-equivalent but not bitwise resume-equivalent. Batch-source
stratification preserves existing per-source CE accounting. The achieved
padding reduction depends on how many microbatches each source contributes per
optimizer step; rare sources with only one microbatch cannot benefit.

Run focused tests:

```bash
python -m pytest -q tests/test_dual_resolution_accum_length_buckets_0102x.py \
  tests/test_dual_resolution_document_sampling_0102e.py
```

**Performance tradeoff:** prefetching an accumulation of microbatches uses
additional host memory, and different GPU shape transitions may affect CuPy
workspace reuse. Check supervised tokens/s, executed tokens/s, and actual peak
GPU memory in matched A/B/A production runs before choosing batch 2 versus 3.
