# 0080AD — Block-proposal checkpoint round-trip invariant

**Purpose.** Ensure an optimizer update cannot silently leave the production
block decoder using non-serialized packed-QKV weights. This is a regression for
0080AB, not a change to the two-stage architecture, optimizer, or checkpoint
format.

## Two complementary tests

### (1) Actual production optimizer update, including optimizer restore

```bash
python -m pytest -q \
  tests/test_block_proposal_0080ad_roundtrip.py \
  tests/test_block_proposal_0080ab_decoder_qkv_refresh.py
```

This CPU regression instantiates `ExtendedTrainer` and an **independent fresh
model**, runs one full production `train_step()` (gradient accumulation,
backward, clipping and real optimizer update), and checks packed QKV before
serialization. It captures proposal logits, decoder PLAN embeddings, delta
correction logits, final logits, main boundary states and per-horizon CE from
a fixed batch. It calls the original `ExtendedTrainer.save()`, restores **both
model and optimizer** with `load_two_stage_checkpoint`, then compares each
captured tensor and each horizon. Another test deliberately creates a stale
packed-QKV cache to prove the guard fails when synchronization is broken.

The regression fails against the unpatched version of 0080AB with
`stale packed QKV`, and passes after 0080AB.

### (2) Production 500M BF16/CuPy cold-model roundtrip

**Prerequisites:** 0080W, 0080AB, and 0080AD applied. Stop production training
before auditing its checkpoint. Set your existing *forward* CuPy optimization
environment (including `MINI_LLM_PACKED_QKV=1`).

```bash
mkdir -p ./benchmarks/results
MINI_LLM_BACKEND=cupy MINI_LLM_PACKED_QKV=1 \
python -m benchmarks.audit_block_proposal_checkpoint_roundtrip \
  --checkpoint ./checkpoints/0080_500m_memory64k_k4_large \
  --mixed-shard-root ./token_shards_pretraining_65280 \
  --source fineweb \
  --batch-size 1 \
  --group-size 16 \
  --seed 78078 \
  --expect-step 5500 \
  --output ./benchmarks/results/0080ad_roundtrip_500m.json
```

**IMPORTANT**: `--expect-step 5500` is an example assertion based on the
original checkpoint. Change it to the actual completed step in
`training_state.json` or omit the flag. The input checkpoint is NOT modified.
The benchmark serializes a **temporary model-only checkpoint** using the
production serializer, releases the first GPU model, constructs a **new**
model, restores the temporary checkpoint, and verifies the same fixed dataset
sample and all K horizons. It checks each of the decoder's three packed-QKV
buffers against the canonical trainable matrices and checks SHA-256 hashes
of every decoder and proposal tensor.

The benchmark compares six output categories: main boundary hidden states,
proposal logits, PLAN soft-token embeddings, decoder correction logits,
final logits and horizon CE. Intermediate vocabulary tensors are captured
only for the first K-block to bound VRAM. It fails loudly for any mismatch
above `--ce-tolerance 0.01` or `--logit-tolerance 0.04` (both adjustable).

This GPU smoke test runs on a checkpoint that **already resulted from real
optimizer steps**. It does not itself run `train_step()` on the 500M model;
that is tested in the production optimizer CPU regression above. Since it
only starts from a previously saved checkpoint, it does not independently
measure the live model **before** the original checkpoint was written. That
would require a separately enabled live pre-save capture. A clean GPU smoke
pass plus the CPU regression verifies this patch's intended invariants, but
is not proof of all 500M runtime states across the original save event.

The temporary checkpoint uses additional disk space comparable to the
uncompressed model weights. Use `--temporary-root /some/large/local/disk`
if necessary. It is deleted automatically on exit. No second copy of the
500M model should remain allocated in VRAM concurrently.

## Interpretation

- Pass: checkpoint serialization and reconstruction preserve these outputs,
  and the packed-QKV cache matches decoder canonical weights at both sides.
- Packed-QKV failure after live `train_step()`: regression of the 0080AB fix.
- Hash mismatch: decoder or proposal serialization/loading differs from saved
  values.
- Stable weights but changed PLAN/final logits: runtime-state or derived-
  buffer inconsistency.
- Stable logits but changed per-horizon CE: scoring or mask inconsistency.

**Do not** restore initialization-era packed-QKV matrices for production
training. Those were a diagnostic historical approximation, not an
architecturally correct model state.
