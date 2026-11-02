# Model-centered repository guide (0104)

## Separation of responsibilities

A `Model` holds architecture, weights, tokenizer and capability reporting.
`GenerationSession` owns prompt-local KV/history and sampling state.
`TrainingRun` owns optimizer, RNG, data sampler and schedule state. The
production `ExtendedTrainer` remains the current native implementation.
`mini_llm.model_execution.KernelRegistry` describes compatible implementations,
but it is **not** yet a fully active operator planner.

Use `Model.wrap_native(native_model)` when training has already constructed a
native model and must reuse its parameter arrays. This does not copy weights.

```python
from mini_llm import Model

model = Model.wrap_native(existing_decoder)
run = model.train(
    train_shard_paths=["/path/to/train.bin"],
    val_shard_paths=["/path/to/val.bin"],
    batch_size=1,
    seq_length=8192,
    grad_accum_steps=512,
    checkpoint_dir="/path/to/new-checkpoint",
    optimizer_kind="hybrid-muon",
    eos_token_id=32767,
)
run.train_steps(10)
run.save_checkpoint()
```

The snippet demonstrates the interface only; adapt EOS ID, paths and options
to the existing run's config. Do not use the wrapper to resume an active
training run; existing resume plumbing loads optimizer and RNG state.

### Scoring

`Model.score(inputs, shifted_targets, loss_mask=..., dual_resolution_metadata=...)`
returns mean supervised cross entropy as a Python float. It does not modify
weights, allocate gradient tape or materialize full-vocabulary logits for all
positions. The caller must prepare identical masks and metadata to training.

### Specialist composition

```python
from mini_llm.specialists import (
    FrozenFoundation, ResidualSpecialist, SpecialistConfig,
    FrozenResidualComposition, save_specialist_checkpoint,
)

frozen = FrozenFoundation.from_checkpoint("/path/to/foundation")
specialist = ResidualSpecialist(SpecialistConfig(
    specialist_id="general", n_layers=12, d_model=1024, d_ff=4096,
))
combined = FrozenResidualComposition(frozen, specialist)
# combined.loss(inputs, targets, dual_resolution_metadata=metadata)
# combined.backward(cache)  # only specialist gradients
save_specialist_checkpoint("/path/to/specialist", combined)
```

This is a **CPU/reference composition and checkpointing interface**. It is not
an end-to-end GPU specialist trainer. Benchmark actual peak VRAM and runtime
before allocating a 12-layer specialist alongside Model B on a 16GB GPU.

### Checkpoint publication

`save_checkpoint` stages complete payloads beside the destination, then
replaces the active directory by renaming. The previous complete snapshot is
kept at `<checkpoint>.previous`. It consumes additional disk space; ensure
sufficient free space. During the very short rename gap, a loader can recover
from `.previous`. On systems where multiple processes write files inside the
same checkpoint directory concurrently, synchronize those writers with saves.
Do not use this to overwrite the running training checkpoint during patch
verification. This is an atomic publication protocol, not a claim of complete
power-loss durability or cross-process locking.

### Tests

```bash
MINI_LLM_BACKEND=numpy python3 -m pytest -q \
  tests/test_checkpoint_transactions_0104a.py \
  tests/test_model_b_config_roundtrip_0104b.py \
  tests/test_model_api_0103_0.py tests/test_training_run_0104d.py \
  tests/test_specialist_composition_0104e.py \
  tests/test_specialist_checkpoint_0104f.py
```

Before GPU parity, run the built-in preflight and check training process
resource ownership. GPU correctness must be evaluated in an isolated worktree.
