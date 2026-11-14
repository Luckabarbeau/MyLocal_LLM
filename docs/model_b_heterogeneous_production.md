# Model B: verified native CUDA + HIP production trial

## Result

A real-data, full-size Model B trial completed one shared optimizer update using
`train_model.py`, the existing production mixture, the saved production optimizer,
and the native CuPy/Triton implementations on both GPUs. No PyTorch training path
was used. The bounded invocation exited successfully; no training job remains
running from this trial.

| Measurement | Observed |
|---|---:|
| Parent trainer step | 15,700 |
| Saved trainer step | 15,701 |
| NVIDIA owner microbatches | 1,024 |
| AMD peer microbatches | 696 |
| Context / batch on both workers | 8,192 / 1 |
| Global supervised tokens | 5,660,340 |
| AMD supervised tokens | 2,332,924 |
| Training loss | 2.310407886 |
| Global gradient norm | 0.292156160 |
| Learning rate | 0.0001425820513 |
| End-to-end training-step time | approximately 153.98 seconds |
| Global supervised tokens/second | approximately 36,760 |

Both workers use document-first variable-length sampling, so the context cap is
not the number of supervised tokens in every sampled document. The AMD worker
kept accumulating until the NVIDIA owner finished; it did not reach its 2,048
microbatch safety cap.

This is a successful production-data training/update/save trial, not a sustained
stability, convergence, or single-GPU speedup benchmark. It intentionally changes
the effective batch and reduction order relative to the parent single-GPU run.

## Preserved production lifecycle

The owner retains the original CLI settings: 100,000-step schedule, 750-step LR
warmup, peak LR 1.5e-4, Hybrid Muon with momentum 0.95 / five NS steps / BF16 matmul,
supervised-token weighting, clipping at 1, weight decay 0.1, unembedding ratio 1,
production mixture, benchmark suite, validation intervals, and component monitor.

The owner alone restores and applies the optimizer. The peer uses an
optimizer-disabled `ExtendedTrainer`, preserving the same mixture sampler and
handwritten backward implementation. Gradients are combined after each local
supervised-token normalization and before one global clipping/update. Peer weights
are refreshed before another round. Peer RNG, shard positions, counters, and the
heterogeneous topology contract are saved with the owner's checkpoint.

The production integration currently keeps the owner's 1,024-microbatch budget
fixed; the experimental runner's adaptive anchor controller is not enabled here.
Warmup is discarded on both devices and restores sampling RNG/counters.

## Checkpoint and logs

Original checkpoint, used read-only:

```text
checkpoints/model_b_batch1_8k_norecompute_long
```

Verified output and continuation checkpoint:

```text
checkpoints/model_b_8k_heterogeneous_trial_15700
```

The directory suffix identifies the **parent** step; the saved state inside is
step **15,701**. It includes config, tokenizer, mixture manifest, 302 model
parameter arrays, 302 FP32 masters, 82 Adam first moments, 82 Adam second moments,
and 220 Muon momentum arrays.

The parent already had an optimizer/trainer step offset of 50. The trial preserved
it rather than renumbering either counter:

```text
trainer:   15,700 -> 15,701
optimizer: 15,750 -> 15,751
```

Post-run checks verified these increments, the preserved offset, peer sampling
state, topology metadata, expected checkpoint assets/group counts, and unchanged
SHA256 hashes for the parent's training state and config. The initial small GPU
probe also verified saved RNG advancement and equality with the live idle peer
sampling state. No claim is made that every full-size saved tensor was scanned.

Logs:

- `logs/model_b_8k_heterogeneous_trial.console.log`
- `logs/model_b_8k_heterogeneous_trial.peer.log`
- `logs/model_b_8k_heterogeneous_trial.csv`
- `logs/model_b_8k_heterogeneous_trial.audit.json`

The component CSV is configured as
`logs/model_b_8k_heterogeneous_trial.components.csv`. No row/file was due at step
15,701: the configured interval is 10, so the next due step is 15,710. Validation
(interval 100) and live benchmark execution were not exercised by this one-update
trial. The final checkpoint was saved explicitly at the bounded invocation's end.

## Commands

The initial, one-shot launch was:

```sh
bash -o pipefail -c 'sh scripts/run_model_b_heterogeneous_trial.sh 2>&1 | tee logs/model_b_8k_heterogeneous_trial.console.log'
```

The trial script deliberately refuses to overwrite an existing trial checkpoint
or its training/peer/component logs. Do not rerun it to continue training.

Continue the verified fork for **10 more updates**:

```sh
sh scripts/resume_model_b_heterogeneous.sh
```

Or choose a larger bounded invocation, for example:

```sh
sh scripts/resume_model_b_heterogeneous.sh --max-run-steps 100
```

The resume helper points both `--resume-from` and `--checkpoint-dir` at the new
heterogeneous checkpoint, retains the production schedule/monitoring options,
and passes the identical peer config. It has been syntax-checked but the
continuation command itself has not yet been executed against the full-size
checkpoint. The original single-GPU parent is never its save destination.

The peer contract is in
`configs/model_b_heterogeneous_production_peer.json`. It includes this machine's
literal ROCm library paths and uses approximately 2.6 GiB of shared host tensor
buffers in `/dev/shm` for this model. Do not edit the peer context, batch, seed,
cap, environment, or contract while calling a continuation exact. Timing-based
peer microbatch membership still prevents a claim of bitwise trajectory
reproducibility; CPU fixed-membership resume tests cover state continuity.

## Validation performed

- Corrected and reran `scripts/probe_heterogeneous_production.py`: **PASS**, real
  BF16 CUDA+HIP shared update and temporary checkpoint/sampling-state checks.
- Ran `scripts/run_model_b_heterogeneous_trial.sh`: **exit 0**, real checkpoint,
  real mixture, full model, 1,024 owner + 696 peer microbatches, update and save.
- Post-save metadata/inventory/parent-hash audit: **passed**, results in the audit
  JSON above.

The first post-save inspection incorrectly assumed a component CSV must already
exist. Inspection of the monitor's completed-step cadence established that it is
not due at 15,701; the corrected audit passed without changing the monitor or
training code.
