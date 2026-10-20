# 0080AC — decoder checkpoint discontinuity investigation

This is a **read-only forensic benchmark**, not a training modification. It
requires the production benchmark from 0080W (and works with 0080X–0080AB).

## Why

The original 500M continuous training showed much larger decoder gains than
running that same checkpoint after reload. 0080AB repaired a real stale
packed-QKV cache in the decoder, but it cannot restore already lost pre-restart
behavior. A step-5500 checkpoint's saved canonical weights are not sufficient
to reconstruct a stale, non-serialized GPU cache automatically.

Production checkpoints **already include** all proposal/decoder trainable
parameters and exact optimizer master weights; a strict name-set check occurs
at load. The audit independently checks their values (on disk vs loaded) and
checks the BF16-rounded model weights against saved FP32 master weights.

## Run

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_PACKED_QKV=1 python -m benchmarks.audit_decoder_checkpoint_discontinuity \
  --checkpoint ./checkpoints/0080_500m_memory64k_k4_step5500_preserved \
  --mixed-shard-root ./token_shards_pretraining_65280 \
  --source fineweb --split val --batch-size 1 --group-size 16 \
  --seed 78078 \
  --output ./benchmarks/results/0080_500m_decoder_checkpoint_probe.json
```

The checkpoint path is a **preserved copy** of the relevant original checkpoint,
not a new path the script creates. It never writes model/optimizer parameters.

## Conditions

- `checkpoint/current`: evaluate the regular restored decoder.
- `initialization-era packed QKV`: temporarily substitute **only** the packed
  attention matrices in the decoder with matrices rebuilt from its seeded
  initialization (`rng_seed=81`), leaving every saved trainable weight (including
  the trained decoder) unchanged.
- `recovered_checkpoint_packs`: restore the live packed matrices and check the
  same predictions are recovered.

If the initialization-era condition recovers substantially higher decoder gains
and changes no proposal CE, stale packed QKV provides a strong explanation.
The original continuous run may have loaded an earlier checkpoint; in that case
its stale matrices were from that earlier load, not necessarily initial seed 81,
and this reconstruction is not conclusive. A negative result also cannot prove
that every trained parameter's on-disk value is correct relative to its in-RAM
value *at the moment of save*, because the latter is no longer available.

An additional definitive test on a future run is a **before-save vs after-load
same-batch forward comparison**, performed in the original training process
before discarding its derived buffers, including comparing all decoder
parameter bytes, packed QKV, PLAN expectation, and corrections. Keep 0080AB
active for continued training.
