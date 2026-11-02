# Production baseline and model-centered integration (0104)

## Mathematical model (must not change during repository refactoring)

Model B, `dual_resolution_matrix_pool_sparse_1024_20l`, is the current
348,261,376-parameter research foundation. Its architecture has 20 layers,
1024 hidden dimensions, 4096-wide FFN, 128-token completed blocks with four
learned matrix-pool summaries, 512-token local attention, one 4096-token
long-local head, and the **stride-16 sparse historical head** in its established
16-layer phase schedule. Block summaries must be causal and document-isolated;
full-sequence and incremental inference must agree. Fine length bucketing is
an execution/data policy, not an architectural parameter.

Do not change model mathematics or optimizer settings as part of 0104. Keep the
live training checkout and checkpoint directory untouched while testing in a
separate worktree.

## Model-centered operations available after 0104A–G

- `Model.load`, `.inspect`, `.execution_info`: canonical native model loading.
- `Model.session` and `.generate`: one request-scoped generation session path,
  backed by existing numerical inference kernels.
- `Model.score(ids, targets, loss_mask=..., dual_resolution_metadata=...)`:
  teacher-forced average CE using the existing chunked native head. Callers
  provide already shifted targets and correct document masks.
- `Model.train(train_shard_paths=..., val_shard_paths=..., ...)`: thin
  `TrainingRun` adapter to **the existing `ExtendedTrainer`**, not a new
  training algorithm. `.train_steps`, `.save_checkpoint` and `.inspect` are
  available on the returned run. This is a new training start; **resuming
  an existing optimizer requires the legacy resume loader/launcher**.
- `Model.save_weights(path)`: specialist-incompatible native weights-only
  serialization; the caller must supply config/tokenizer for a standalone
  reload. Use a fresh destination, not an active training checkpoint.
- `FrozenResidualComposition`: a reference forward/backward for a frozen
  foundation plus hidden-only residual specialist. Includes strict Model B
  metadata forwarding and refuses multi-document specialist rows rather
  than risking cross-document leakage.
- Specialist-only save/load: records an exact SHA256 of foundation config and
  parameter files; writes specialist weights, optional optimizer state and
  optional training state without duplicating foundation parameters.

## Remaining production gates (do not claim these are complete)

1. **GPU Model B parity:** full-model numerical comparison and BF16 specialist
   forward/backward at 2 layers and 12 layers. Compare document and sparse
   attention paths, and verify no foundation weight/gradient mutation.
2. **Specialist data-source integration:** route the existing document-first
   mixture sampler through a shared training interface without instantiating
   foundation optimizer states. The initial specialist requires single-document
   rows until document-isolating attention is implemented.
3. **Specialist TrainingRun:** add efficient GPU accumulation, scheduling,
   checkpoint/resume and logging for independent trainable components using
   the shared training interfaces. The reference composition is *not* a
   production specialist-training launcher yet.
4. **Performance parity:** on the RTX 5080 compare the old trainer and the
   model facade for supervised tok/s, allocated VRAM, and model output. CUDA
   Graph execution reporting and strict fallback checks must remain truthful.
5. **Legacy de-duplication:** migrate remaining benchmark wrappers and training
   launchers incrementally, preserving their CLI behavior, then archive or
   remove dead experimental paths only after matching golden results.
6. **Model execution framework:** the existing `mini_llm.backend` is still
   process-global. Per-model simultaneous CuPy/NumPy/Torch execution and a
   full shared operator dispatcher remain future backend work.

## CPU acceptance

```bash
MINI_LLM_BACKEND=numpy python3 scripts/run_0103a_regressions.py --mode cpu
```

## Historical reference

Earlier 338M models and retired branches remain documented under `docs/research`
and existing experiment notes. Their results are valuable but they are not
production Model B configurations.
