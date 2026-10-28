# Patch 0102E — EOS-bounded full-sequence dual-resolution training

The dual-resolution model previously used the legacy fixed-length packed-window
AR sampler. This allowed a window to contain unrelated documents; while
EOS-aware attention prevented cross-document leakage, it did **not** sample the
same distribution as the 338M historical-memory training configuration.

## Updated data policy

- Select one document/physical-shard fragment **first**, using the same
  length-weighted packed-document selection function as the original
  document-first 338M sampler. No length threshold or long-document retry.
- For a document with `N` next-token pairs and a configured `--context-length C`,
  physically execute `min(N, C)` positions. With batch size 1, do not pad the
  short sequence to `C`, and never concatenate unrelated documents to fill it.
- For `N <= C`, include the **whole document**. For `N > C`, draw a uniformly
  random contiguous `C+1`-token window *inside the selected document*.
- Supervise **every genuine next-token pair** in the selected window, including
  the earliest tokens. Unlike the old historical model's `4K target + historical
  prefix` split, there is no prefix-only unsupervised region.
- The next-token label may be EOS, but an EOS input must never predict the next
  document's first token. No such crossing is produced by the sampler.
- For batch size greater than 1, pad shorter rows up to the longest sampled
  row *within that microbatch*, and mask padding loss. For maximum compute
  efficiency at long context use batch size 1 with gradient accumulation.
- Global block summaries remain causal and limited to completed prior blocks;
  no current block summary becomes visible. The data change does not modify
  attention, training/inference parity, or the parameter layout.

`MINI_LLM_AR_COMPACT_TARGET` and `MINI_LLM_AR_ACCUM_LENGTH_BUCKETS` are legacy
optimizations tied to the older historical-memory target/prefix layout.
For dual-resolution only, these settings are automatically disabled with a
console notice: physical variable-length sampling already removes padding at
batch size 1. The older models retain their previous behavior.

## Full 65K support vs actual training cost

`--context-length` is the maximum **actual** window size, not a separate
historical cache capacity. `--context-length 4096` cannot train 65K dependencies;
it trains all valid pairs inside at most 4096 tokens. Use
`--context-length 65536` to permit full 65,536-token documents/windows.
At 65K the model processes all executed positions through 20 full-resolution
layers and scores all target tokens with the chunked LM head. This may exceed
16 GB device memory even using activation recomputation; validate at smaller
contexts first.

A physical packed-shard boundary truncates any original document that spans
multiple files. The existing index treats such boundaries as separate
fragments to preserve causality. This patch does not reconstruct cross-shard
fragments.

## Throughput definitions

For the dual-resolution model, step logs now report:

- `supervised_tok/s`: total genuine, supervised next-token targets divided by
  the completed optimizer-step duration (the metric for comparison to 338M).
- `executed_tok/s`: token positions actually processed in the Transformer,
  including masked padding when batch size >1.
- `capacity_tok/s`: configured upper-bound sequence capacity divided by step
  duration, *not* actual training throughput when samples are short.
- `target_fill`: supervised next-token targets relative to configured context
  capacity; low fill is expected from short EOS-bounded documents.

The train/validation samplers now both obey the same document-first policy.
Existing datasets and their `.docidx.npy` sidecars are reused as-is.

Run the focused CPU tests:

```bash
python -m pytest -q tests/test_dual_resolution_document_sampling_0102e.py
```

For a training smoke test, use the existing dual-resolution training command
with `--context-length 4096 --batch-size 1 --grad-accum-steps 4` and verify
that the console prints `Dual-resolution sampling: document-first` and actual
`supervised_tok/s` and `executed_tok/s` metrics. Compare performance after
warm-up. The `dual_resolution` architecture and all Triton kernels remain
unchanged.
