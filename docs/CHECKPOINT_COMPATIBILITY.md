# Checkpoint compatibility and safe continuation

This document defines how native foundation and specialist checkpoints should be
loaded after the model-centric refactor. It distinguishes exact continuation,
weights-only initialization, inference loading, and specialist attachment.

## Current protected run boundary

The user reported that the protected external Model B run was at **step 165** on
**2026-11-04**. That checkpoint is not available in this checkout and cannot be
verified here. No command below names it, and no test should read, hash, resume,
convert, or overwrite it.

Before operating on any valuable checkpoint:

1. identify the process that owns it;
2. stop using the active path as a test fixture;
3. make an immutable, separately named copy when inspection is required;
4. verify the copy contains `config.json`, `tokenizer.json`, `model_params/`,
   `optimizer_state/`, and `training_state.json` as appropriate;
5. direct experiments to a distinct destination.

The model-centric refactor and this documentation update changed no foundation
training code.

## Compatibility modes

| Operation | Weights | Optimizer/schedule | RNG/cursors | Config authority | Intended use |
|---|---|---|---|---|---|
| Strict `--resume-from` | exact complete schema | restored and required | restored when saved; required for deterministic continuation | saved checkpoint | continue the same run |
| `--init-from` | matching names only; same-name shape mismatch fails | fresh | fresh | selected target model | intentional weights-only start |
| Native inference / `Model.load` | complete canonical names and shapes | not loaded | new session state | saved checkpoint | evaluation or generation |
| Specialist load | complete specialist schema | optional for inference, required for training resume | specialist training resume restores sampler state | specialist config plus verified foundation digest | attach one residual to one frozen foundation |

`--resume-from`, `--init-from`, and block-proposal warm-start options are mutually
exclusive. They express different scientific operations and must not be used as
aliases for one another.

## Strict resume

Strict resume means continuation of one training state, not merely obtaining the
same initial logits.

### Saved config is authoritative

`train_model.py` reads `<checkpoint>/config.json` before model construction. A
saved config determines architecture and parameter layout; `--model` is not a
migration override. Current Model B should deserialize as:

- registry identity `model_b_348m_sparse16`;
- CLI preset `dual-resolution-matrix-pool-sparse-1024-20l`;
- 20 layers, `d_model=1024`, `d_ff=4096`, vocabulary 32,768;
- 16 query heads and 4 KV heads;
- matrix-pool summaries and sparse stride-16 layers `2..17`.

A missing `config.json` triggers a legacy CLI fallback in `train_model.py`.
Treat that as a recovery case, not normal strict resume. Reconstruct and validate
an immutable copy before allocating a production model.

The saved architecture does not make every CLI setting irrelevant. Precision,
run context, optimizer/schedule arguments, and data locations still need to
match the intended run contract. The pinned Model B values are recorded in
[`CURRENT_BASELINE.md`](CURRENT_BASELINE.md) and
`mini_llm/production_profile_0105.py`.

### Tensor schema must be exact

Foundation resume constructs the current model and compares its canonical
parameter names with every `.npy` under `model_params/`:

- no missing tensor names;
- no unexpected tensor names;
- no partial architecture loading;
- same-name tensor shapes must fit the constructed parameter;
- canonical parameter names must be unique.

The native inference loader independently rejects missing arrays and shape
mismatches before copying weights. This prevents a partly randomized model from
being reported as loaded.

A sparse Model B checkpoint has the same learned tensor count as its matrix-pool
base because the sparse branch reuses existing projections. That fact does not
weaken strict resume: exact names and the saved sparse config are both required.

### Optimizer and training state are part of the checkpoint

A modern strict resume requires both `optimizer_state/` and
`training_state.json`. The launcher validates the saved optimizer kind and the
complete state schema.

For `hybrid-muon`, continuation includes:

- FP32 master weights for every optimized parameter;
- Adam `m` and `v` for Adam-group parameters;
- Muon momentum `muon_m` for Muon-group parameters;
- optimizer step index;
- the exact Muon parameter-group name set;
- optimizer hyperparameters and parameter birth-step metadata where applicable.

Model weights alone are not a resume checkpoint. Recreating Hybrid Muon from
weights changes the optimization trajectory even if the next forward pass is
identical.

Training state also carries step and token counters, shard/source indices,
source metrics, progressive-depth state when applicable, data provenance,
mixture history, and sample-trace metadata.

### RNG and sampler state define deterministic continuation

The foundation trainer saves train and validation NumPy RNG states. Modern exact
continuation restores them together with shard/source state. Some legacy
checkpoints may lack RNG fields and can still be loaded, but they cannot claim
sample-exact deterministic continuation.

Specialist training saves its document sampler state, including per-split RNG,
cursors, counts, source weights, manifest digest, context, block size, and shape
bucket. Specialist resume validates the run config, restores the sampler before
constructing the run, requires optimizer state, and restores step/token counters.

### Tokenizer, checkpoint, and data are one pairing

Do not pair artifacts by vocabulary size alone. Two tokenizers can both contain
32,768 entries while assigning different IDs.

The mixed-shard loader requires:

- `mixture_manifest.json` with the supported format;
- a complete set of declared train and validation shards;
- `tokenizer.json` whose SHA256 matches the manifest when a hash is present;
- manifest vocabulary size consistent with the loaded tokenizer.

The trainer records tokenizer SHA256 in checkpoint data provenance and rejects a
resume tokenizer mismatch. It also copies the exact tokenizer and mixture
manifests beside a checkpoint for reproducible loading. Native model loading uses
the checkpoint tokenizer, falling back to the checkpoint's parent only for
legacy layouts; prefer self-contained copies.

Data provenance is more nuanced than tokenizer identity. If source revisions,
weights, local manifest, or stream-segment policy deliberately change, the
trainer records an auditable mixture/data fork and may reset affected cursors
while preserving model and optimizer state. This is valid continued training,
but it is **not** exact continuation of the original sample stream.

## Weights-only `--init-from`

`--init-from` streams matching checkpoint tensors into a newly constructed
target model. It deliberately does **not** restore:

- optimizer masters or moments;
- optimizer step;
- LR schedule position;
- training/token counters;
- train or validation RNG;
- shard and streaming cursors;
- prior data provenance as continuation state.

Same-name shape mismatches fail immediately. Missing parameters are reported;
the current generic training path permits only its explicitly recognized new
memory tensors to remain newly initialized and rejects missing backbone tensors.

Use weights-only initialization only when a fresh scientific run is intended.
Use a new, empty checkpoint destination and record the source checkpoint digest
or immutable copy in the experiment metadata.

Safe placeholder shape:

```bash
source scripts/0105_model_b_profile.env
python -u train_model.py \
  --model dual-resolution-matrix-pool-sparse-1024-20l \
  --init-from /path/to/approved-immutable-source-copy \
  --checkpoint-dir /path/to/new-empty-destination \
  --mixed-shard-root /path/to/paired-mixed-shards \
  ...
```

The ellipsis is intentional: use the reviewed profile, not this abbreviated
snippet, as the full launch contract.

## Strict resume placeholder

Only use this pattern after verifying an approved checkpoint copy and its paired
data. Do not substitute an active or protected checkpoint path.

```bash
source scripts/0105_model_b_profile.env
python -u train_model.py \
  --resume-from /path/to/approved-checkpoint-copy \
  --checkpoint-dir /path/to/approved-output-destination \
  --mixed-shard-root /path/to/paired-mixed-shards \
  ...
```

Resuming into the same directory is supported by the transaction mechanism, but
it replaces the active snapshot at the next save. For inspection, recovery,
profile testing, migration, or specialist work, use a separate immutable copy
and separate output instead.

## Transactional checkpoint publication

`mini_llm.checkpoint.save_checkpoint` publishes a checkpoint as a directory
transaction:

1. create a sibling `<checkpoint>.pending-<uuid>` directory;
2. preserve unmanaged supporting files such as config/tokenizer/manifests;
3. write all managed model, optimizer, and training payloads into staging;
4. remove an older `<checkpoint>.previous` only after staging succeeds;
5. rename the current checkpoint to `<checkpoint>.previous`;
6. rename the complete pending directory to the active path;
7. on a failed final publish, restore the previous active directory;
8. loaders fall back to `.previous` if the active path is absent after an
   interrupted rename sequence.

This gives atomic snapshot replacement on the same filesystem and keeps the
last complete predecessor. It also implies important limits:

- staging and destination must share a filesystem for rename semantics;
- enough disk is needed for staging plus the retained predecessor;
- writers of ancillary files must be synchronized with checkpoint publication;
- there is no general cross-process lock;
- large optimizer files are not individually `fsync`ed;
- this is atomic publication, not a guarantee against full power loss or storage
  failure.

A failed payload write must leave the active checkpoint untouched, and a failed
publish must restore the previous version. These behaviors are covered by
`tests/test_checkpoint_transactions_0104a.py`.

## Active checkpoint safety

Never use an actively written checkpoint as:

- a regression-test input;
- a conversion source and output at the same time;
- a specialist foundation to hash repeatedly;
- a target for weights-only initialization;
- a scratch location for benchmark artifacts;
- a directory manually edited while saves may occur.

Specialist foundation hashing reads every foundation weight file. If a training
save rotates the directory during hashing, the resulting read is not a stable
scientific identity. Freeze or copy the foundation first.

The production profile intentionally excludes `--checkpoint-dir`, `--log-file`,
and `--resume-from`. Regression commands must supply safe locations rather than
inherit production paths.

## Specialist foundation compatibility

A specialist checkpoint serializes only residual-specialist parameters plus its
own optional optimizer and training state. It binds to one frozen foundation by
an exact SHA256 over:

- `config.json` path and bytes;
- every sorted `model_params/*.npy` relative path and bytes.

On load, the foundation digest is recomputed and must match. Even a byte-only
config change, such as added whitespace, changes the digest and is rejected.
The recorded foundation path is provenance; the content digest is the identity.

The specialist loader also validates:

- provenance format version;
- specialist architecture config;
- exact specialist parameter-name set;
- saved parameter order;
- every specialist tensor shape.

After loading canonical specialist weights, derived compute buffers are
refreshed. For training resume, `SpecialistTrainingRun` additionally requires the
same run config, complete optimizer state, restored sampler state, and restored
step/token/source counters.

The foundation digest intentionally covers foundation architecture and weights,
not the tokenizer or training dataset. Tokenizer and data pairing remain separate
requirements and should be recorded with the specialist experiment.

## Canonical tensors and stale derived caches

Optimized execution may derive packed projections or other compute buffers from
canonical parameter arrays. These derived caches are not authoritative
checkpoint tensors.

A historical round-trip regression exposed the failure mode: a live process used
a stale packed-QKV cache after canonical QKV weights changed. Saving persisted the
correct canonical arrays, so a fresh model rebuilt different derived buffers and
produced different output from the still-running process. File equality alone
would not have caught the live-state discontinuity.

The durable lesson is:

1. update canonical parameters;
2. invalidate or refresh every derived compute buffer after updates and loads;
3. guard cache integrity before save when a derived cache exists;
4. compare a fixed-batch fingerprint before save with a **fresh model** after
   load;
5. include optimizer state in the round trip when claiming resume parity.

Current foundation resume calls `refresh_compute_buffers` after loading model
weights. Specialist loading does the same for specialist buffers. The historical
regressions in `tests/test_block_proposal_0080ad_roundtrip.py` verify both fresh
model parity and detection of a simulated stale packed-QKV cache.

A checkpoint round trip is therefore a behavioral test, not only a serialization
test.

## Compatibility checklist

Before strict resume:

- [ ] operate on an approved stable checkpoint, not the protected active path;
- [ ] verify `config.json` is present and identifies the intended architecture;
- [ ] verify tokenizer bytes/hash against checkpoint provenance and data manifest;
- [ ] verify exact model tensor names and shapes;
- [ ] verify complete optimizer components and saved optimizer kind/groups;
- [ ] verify training state, counters, and modern train/validation RNG state;
- [ ] decide whether data provenance is unchanged or an explicit fork;
- [ ] confirm runtime profile and schedule arguments match the reviewed command;
- [ ] ensure enough disk for pending and `.previous` snapshots;
- [ ] use a distinct location for tests, migrations, and specialist experiments.

Before claiming compatibility:

- [ ] cold-load into a fresh process/model;
- [ ] refresh or verify derived buffers;
- [ ] compare fixed inputs, logits/loss, and relevant intermediate fingerprints;
- [ ] for training, compare the next update and optimizer step, not only forward
      output;
- [ ] record what was actually verified and what remains user-reported.

## Focused validation

```bash
MINI_LLM_BACKEND=numpy pytest -q \
  tests/test_checkpoint.py \
  tests/test_checkpoint_transactions_0104a.py \
  tests/test_model_api_0103_0.py \
  tests/test_specialist_checkpoint_0104f.py \
  tests/test_specialist_training_run_0105e.py
```

These tests validate small/reference contracts. They do not verify the
unavailable protected step-165 checkpoint and do not establish full-size Model B
GPU throughput.
