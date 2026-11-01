# Patch 0103K — Adaptive fine sequence-length buckets (through 65,536)

This is an **opt-in execution-shape optimization** for Model B's document-first
training. It changes only the amount of EOS padding, never the sampled
training windows, their targets, or the attention architecture.

Enable before launching training:

```bash
export MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET=fine
```

Omit the variable to retain Model B's validated `geometric` runtime default.
The existing modes `none` and `geometric` are unchanged. Checkpoint formats
and optimizer settings are unchanged. The unrelated accumulation regrouping
flag `MINI_LLM_DUAL_RESOLUTION_ACCUM_LENGTH_BUCKETS` remains off by default.

| Genuine length | Execution width | Buckets |
| --- | --- | ---: |
| 1–128 | Original geometric ladder: 1, 2, 4, 8, 16, 32, 48, 64, 96, 128 | 10 |
| 129–2,048 | 64-token steps, 192 through 2,048 | 30 |
| 2,049–4,096 | 128-token steps, 2,176 through 4,096 | 16 |
| 4,097–8,192 | 256-token steps, 4,352 through 8,192 | 16 |
| 8,193–16,384 | 512-token steps, 8,704 through 16,384 | 16 |
| 16,385–32,768 | 1,024-token steps, 17,408 through 32,768 | 16 |
| 32,769–65,536 | 2,048-token steps, 34,816 through 65,536 | 16 |
| **Total** | | **120** |

Widths are clipped to the training context cap, including non-aligned caps.
This may introduce one nonstandard final shape for a nonstandard context cap,
just as the existing geometric mode does. A sampled length above 65,536 is
explicitly unsupported in this mode. For Model B's current 8K training cap,
only 72 of the above widths can occur.

Example for **6,200** genuine positions at an 8K cap:

- `geometric`: 8,192 executed positions;
- `fine`: 6,400 executed positions;
- the same 6,200 targets are supervised with the same EOS and loss masks.

Implementation points:

- `mini_llm/data/token_shards.py`: new `_dual_resolution_fine_length` and `fine`
  dispatch in `create_dual_resolution_document_minibatch`.
- `mini_llm/train_extended.py`: allow `fine` as a shape-bucket policy.
- `mini_llm/data/dual_resolution_length_buckets.py`: support `fine` when the
  separate accumulation-regrouping feature is explicitly enabled.
- `mini_llm/runtime_defaults.py`: **unchanged**, maintaining `geometric` as
  the Model B default.

CPU regression command:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_dual_resolution_fine_shape_buckets_0103k.py \
  tests/test_dual_resolution_fast_document_0102f.py \
  tests/test_dual_resolution_accum_length_buckets_0102x.py \
  tests/test_dual_resolution_document_sampling_0102e.py
```

**Performance caveat:** 120 potential execution shapes are a maximum, not a
promise of a throughput improvement. More shapes can increase Triton compilation
and cache cost. Compare cold *total* wall time and warm supervised tokens/s from
the same checkpoint/seed on the production GPU before adopting this mode.
No RTX 5080 throughput claim is made by this patch.
