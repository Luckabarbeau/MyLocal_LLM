# MyLocal_LLM User Guide

This is the operational guide for the current **Model B** workflow. The older
`wide-500m-*` and 338M historical-hybrid examples elsewhere in the repository
are research history, not the production launch recipe documented here.

Model B is the native preset
`dual-resolution-matrix-pool-sparse-1024-20l`: 20 layers, width 1024, dense
FFN 4096, matrix-pooled dual-resolution attention, and a stride-16 original-token
head across layers 3–18. The implementation has 348,261,376 parameters. The
current production training profile uses an 8,192-token training context even
though the architecture preset can represent a larger context.

> **Protected external run:** an externally managed Model B run is reported at
> **step 165**. Its checkpoint is not available in this workspace.
> Do not inspect, load, resume, overwrite, signal, benchmark against, or otherwise
> touch the protected run. Wait for an explicitly published immutable checkpoint
> copy or use a new, independent run directory.

## 1. Install and choose a backend

Run commands from the repository root, where `train_model.py` is located.

### CPU/reference development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
MINI_LLM_BACKEND=numpy python -c 'from mini_llm.backend import BACKEND_NAME; print(BACKEND_NAME)'
```

NumPy is suitable for unit tests, API work, profile audits, and small numerical
checks. It is not a practical backend for full Model B training.

### NVIDIA GPU development and training

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,dev,streaming,external,triton-attention]'
nvidia-smi
MINI_LLM_BACKEND=cupy python -c 'from mini_llm.backend import BACKEND_NAME; print(BACKEND_NAME)'
```

The `gpu` extra installs the CUDA 13 CuPy package selected by `pyproject.toml`;
the Triton extra is pinned by the project. Confirm that those choices match the
machine's driver before launching training.

`MINI_LLM_BACKEND` is process-global and is read during Python import. A Python
process cannot switch from NumPy to CuPy after importing `mini_llm`; start a new
process with the intended backend.

### Runtime profiles

Select a profile with `--runtime-profile` or `MINI_LLM_RUNTIME_PROFILE`:

| Profile | Behavior |
|---|---|
| `auto` | NumPy resolves to `reference`; CuPy inference resolves to `gpu-fast`; a large CuPy training model resolves to `consumer-gpu`. |
| `reference` | Applies no profile performance defaults. Use for correctness comparisons. |
| `gpu-fast` | Enables validated CuPy/BF16 fast paths without forcing optimizer offload or the consumer LM-head policy. |
| `consumer-gpu` | Adds memory-conscious training defaults, including full optimizer offload and chunked LM-head execution. It also applies Model B-specific matrix-pool defaults when the configuration matches. |

Profiles use `os.environ.setdefault`: an explicit `MINI_LLM_*` value always
wins. That is why the production run sources the exact environment file rather
than relying only on evolving profile defaults.

## 2. GPU ownership and safe operations

Treat the GPU as an exclusively owned resource for Model B work. Before any
CuPy training, inference, serving, GPU regression, or specialist run:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

If another training process owns the device, do not launch a competing process,
do not free its memory pool, and do not kill or signal it. Profile audits and
CPU regressions are safe because they do not allocate the GPU:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python scripts/run_0103a_regressions.py \
  --mode cpu --report 0105_cpu_report.json
```

The GPU regression gate is only for an otherwise idle GPU with enough free VRAM:

```bash
.venv/bin/python scripts/run_0103a_regressions.py \
  --mode gpu --model-b-profile --report 0105_gpu_report.json
```

Passing selected GPU gates is not proof that a full production run has the
expected throughput or memory headroom.

## 3. Required Model B data contract

Production commands below expect the current mixed-shard root at
`./token_shards_ensemble_32k_mixed`. It must contain:

- `tokenizer.json`;
- a complete `mixture_manifest.json` with format
  `weighted_packed_mixture_v1`;
- each registered source's train and validation shards; and
- tokenizer vocabulary/hash values consistent with the manifest.

The trainer derives EOS from the loaded tokenizer. Never assume that EOS is the
last vocabulary entry or paste a numeric EOS from another model family. See
[`DATA_PIPELINE.md`](DATA_PIPELINE.md) for tokenizer, shard, identity,
provenance, deduplication, and Common Pile procedures.

## 4. Exact production profile and read-only audit

`scripts/0105_model_b_profile.env` contains the explicit environment from the
current Model B production command. Source it in a **new shell**. Sourcing the
file does not attach to or restart any existing process.

The audit compares the exact environment and scientific CLI flags. It does not
load a model, open a checkpoint, or use the GPU. Run it from the repository root
so the expected relative mixture and benchmark paths match exactly:

```bash
source scripts/0105_model_b_profile.env

MODEL_B_FLAGS=(
  --model dual-resolution-matrix-pool-sparse-1024-20l
  --objective ar
  --precision bf16-mixed
  --runtime-profile consumer-gpu
  --mixed-shard-root ./token_shards_ensemble_32k_mixed
  --context-length 8192
  --batch-size 1
  --grad-accum-steps 512
  --gradient-accumulation-weighting supervised_tokens
  --optimizer hybrid-muon
  --muon-momentum 0.95
  --muon-ns-steps 5
  --muon-matmul-precision bf16
  --total-steps 100000
  --warmup-steps 750
  --peak-lr 1.5e-4
  --grad-clip 1.0
  --weight-decay 0.1
  --val-interval 100
  --val-steps 4
  --val-per-source-steps 1
  --val-fixed-mixed-steps 4
  --val-fixed-mixed-seed 78078
  --save-interval 50
  --log-interval 1
  --seed 42
  --benchmark-suite ./benchmarks/configs/foundation_capabilities_0097_progressive.json
)

.venv/bin/python scripts/audit_0105_training_profile.py \
  --report ./logs/model_b_profile_audit.json -- "${MODEL_B_FLAGS[@]}"
```

A zero exit status and `"matches": true` mean the shell and flags match the
frozen comparison contract. The audit deliberately excludes
`--checkpoint-dir`, `--log-file`, and `--resume-from`: those are run locations
and must never be copied from a protected run.

## 5. Safe launch of a new Model B run

There is no allocation-free dry-run mode in `train_model.py`. The safe launch
pattern is therefore:

1. use a new shell and source the profile;
2. pass the profile audit;
3. verify exclusive GPU ownership;
4. verify that the destination checkpoint path does not exist;
5. launch with a new log and checkpoint location; and
6. optionally bound the first invocation with `--max-run-steps 1`, which keeps
   the 100,000-step schedule horizon and publishes a resumable checkpoint.

Continuing the shell block from the previous section:

```bash
CHECKPOINT_DIR=/absolute/path/to/new_model_b_run
LOG_FILE=/absolute/path/to/logs/new_model_b_run.csv

# Both checks must succeed. Never point either variable at the protected run.
test ! -e "$CHECKPOINT_DIR"
test ! -e "$LOG_FILE"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

.venv/bin/python train_model.py \
  "${MODEL_B_FLAGS[@]}" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --log-file "$LOG_FILE" \
  --max-run-steps 1
```

The bounded invocation is a real optimizer update, not a dry run. If it passes,
continue only through the strict resume procedure below. For a deliberately
uninterrupted launch, omit `--max-run-steps 1`; all other flags remain identical.

A new run copies `config.json`, the exact tokenizer, and mixture metadata into
the checkpoint directory. Checkpoint saves include model arrays, complete
optimizer state, RNG state, sampler/source counters, schedule state, supervised
token accounting, and data provenance.

## 6. Strict Model B resume

A strict resume means more than supplying `--resume-from`. Use:

- the same `scripts/0105_model_b_profile.env`;
- the original scientific flags, unchanged;
- the same tokenizer and mixed-shard root;
- the same `mixture_manifest.json` bytes and source files;
- the original checkpoint directory as both input and output; and
- the original log location and operational policy.

`--total-steps 100000` remains the absolute schedule target, not “100,000 more
steps.” Do not substitute `--init-from`: that is a weights-only warm start with
a fresh optimizer and schedule.

```bash
source scripts/0105_model_b_profile.env

MODEL_B_FLAGS=(
  --model dual-resolution-matrix-pool-sparse-1024-20l
  --objective ar
  --precision bf16-mixed
  --runtime-profile consumer-gpu
  --mixed-shard-root ./token_shards_ensemble_32k_mixed
  --context-length 8192
  --batch-size 1
  --grad-accum-steps 512
  --gradient-accumulation-weighting supervised_tokens
  --optimizer hybrid-muon
  --muon-momentum 0.95
  --muon-ns-steps 5
  --muon-matmul-precision bf16
  --total-steps 100000
  --warmup-steps 750
  --peak-lr 1.5e-4
  --grad-clip 1.0
  --weight-decay 0.1
  --val-interval 100
  --val-steps 4
  --val-per-source-steps 1
  --val-fixed-mixed-steps 4
  --val-fixed-mixed-seed 78078
  --save-interval 50
  --log-interval 1
  --seed 42
  --benchmark-suite ./benchmarks/configs/foundation_capabilities_0097_progressive.json
)

.venv/bin/python scripts/audit_0105_training_profile.py -- "${MODEL_B_FLAGS[@]}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

CHECKPOINT_DIR=/absolute/path/to/the_original_model_b_checkpoint
LOG_FILE=/absolute/path/to/the_original_model_b_log.csv

.venv/bin/python train_model.py \
  "${MODEL_B_FLAGS[@]}" \
  --resume-from "$CHECKPOINT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --log-file "$LOG_FILE"
```

The resume loader rejects incomplete parameter sets, shape mismatches, missing
optimizer/training state, incomplete optimizer components, and a tokenizer hash
that differs from checkpoint provenance. The code can record a new data phase
when a mixture changes, but that is **not** a strict resume; do not use that
feature when exact continuation is required.

The protected step-165 run cannot be resumed from this checkout because its
checkpoint is unavailable here. Do not fabricate a checkpoint path or start a
replacement under the protected run's name.

## 7. Native inference

Inference requires a complete native checkpoint with its matching tokenizer.
Use a completed checkpoint or an immutable copy, never a path actively being
published by another process.

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python inference.py \
  --checkpoint /absolute/path/to/model_b_checkpoint \
  --prompt 'The role of numerical stability in optimization is' \
  --max-new-tokens 128 \
  --strategy top-p \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42 \
  --backend kv-cache \
  --runtime-profile gpu-fast \
  --execution-policy auto
```

For a long prompt, use `--prompt-file`. The cache horizon defaults to the
checkpoint configuration; do not increase it merely because the architecture
preset has a larger theoretical capacity. `--execution-policy strict-fast`
fails when the requested verified fast path cannot execute; use it as an
acceptance gate, not as a universal compatibility setting. `--backend reference`
recomputes the full prefix and is intended for verification rather than speed.

The loader is strict about canonical parameter completeness and shapes, but
operators must still preserve the checkpoint/tokenizer pair as an artifact.

## 8. Serving

Create a separate registry JSON. Relative model paths are resolved relative to
the registry file:

```json
{
  "models": [
    {
      "id": "model-b",
      "format": "native",
      "path": "./checkpoints/model_b_published_copy",
      "chat_template": "plain",
      "tool_format": "none",
      "context_length": 8192,
      "execution_policy": "auto",
      "sampling": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.0,
        "repetition_window": 256,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
      }
    }
  ]
}
```

Start a localhost-only server:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python serve.py \
  --registry /absolute/path/to/serving_models.json \
  --host 127.0.0.1 --port 8000
```

Check it with:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-b","prompt":"A stable optimizer","max_tokens":64}'
```

Binding beyond localhost requires `--api-key` or `MYLOCAL_LLM_API_KEY`. The
service is a small compatibility surface, not a hardened public deployment. It
keeps at most one local model GPU-resident and serializes local inference
requests. See the developer guide for API limitations.

## 9. Frozen-foundation specialist training

The guarded specialist launcher added by `eb61df0` is executable on GPU, but it
requires a **completed, immutable copy** of a sparse Model B foundation. It may
never use the active protected run or a checkpoint directory another process is
replacing. No suitable foundation checkpoint is available in this workspace at
present.

The foundation remains frozen. Only independent residual-specialist parameters
are optimized; the specialist checkpoint stores a content fingerprint of the
foundation and resume rejects a different foundation.

### Dry run

The dry run validates paths and prints the plan. It does not import the heavy
model code, allocate a model, train, or write a checkpoint. The foundation and
mixture paths must nevertheless exist, and the specialist destination must be
new.

```bash
.venv/bin/python scripts/train_0105_specialist.py \
  --foundation-checkpoint /absolute/path/to/immutable_model_b_copy \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --specialist-checkpoint /absolute/path/to/new_specialist_smoke \
  --specialist-id general \
  --specialist-layers 2 \
  --context-length 512 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --total-steps 10000 \
  --warmup-steps 100 \
  --peak-lr 3e-4 \
  --grad-clip 1.0 \
  --weight-decay 0.1 \
  --muon-momentum 0.95 \
  --muon-ns-steps 5 \
  --muon-matmul-precision bf16 \
  --head-chunk-tokens 512 \
  --shape-bucket fine \
  --seed 42 \
  --steps 1 \
  --validation-batches 2
```

### GPU smoke run

Run only on an idle GPU after the dry run succeeds:

```bash
source scripts/0105_model_b_profile.env

.venv/bin/python scripts/train_0105_specialist.py \
  --foundation-checkpoint /absolute/path/to/immutable_model_b_copy \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --specialist-checkpoint /absolute/path/to/new_specialist_smoke \
  --specialist-id general \
  --specialist-layers 2 \
  --context-length 512 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --total-steps 10000 \
  --warmup-steps 100 \
  --peak-lr 3e-4 \
  --grad-clip 1.0 \
  --weight-decay 0.1 \
  --muon-momentum 0.95 \
  --muon-ns-steps 5 \
  --muon-matmul-precision bf16 \
  --head-chunk-tokens 512 \
  --shape-bucket fine \
  --seed 42 \
  --steps 1 \
  --validation-batches 2 \
  --log-interval 1 \
  --profile-steps 1 \
  --log-text-file ./logs/specialist_general_smoke.log \
  --log-jsonl ./logs/specialist_general_smoke.jsonl \
  --run
```

By default the current trainer enables whole-specialist activation checkpoint
replay, matrix-pool attention-probability recomputation, and source-preserving
length regrouping when both batch size and accumulation are greater than one.
The `--no-*` switches disable those mechanisms and therefore become part of the
resume configuration. `--train-foundation-ce` adds an exact frozen-foundation CE
for each training minibatch but costs an additional frozen LM-head evaluation.

A 12-layer, 8,192-context run is a separate scientific experiment. Validate VRAM,
finite gradients, throughput, and checkpoint/resume first; do not infer safe
settings from the foundation's accumulation count.

### Specialist resume

Resume uses `--resume --run`, not `--resume-from`. Repeat every option that
constructs `SpecialistTrainingConfig` and every sampler/geometry option exactly.
`--steps` is the number of optimizer updates in this invocation and may change;
`--total-steps` remains the original schedule horizon.

```bash
source scripts/0105_model_b_profile.env

.venv/bin/python scripts/train_0105_specialist.py \
  --foundation-checkpoint /absolute/path/to/the_same_immutable_model_b_copy \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --specialist-checkpoint /absolute/path/to/existing_specialist_smoke \
  --specialist-id general \
  --specialist-layers 2 \
  --context-length 512 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --total-steps 10000 \
  --warmup-steps 100 \
  --peak-lr 3e-4 \
  --grad-clip 1.0 \
  --weight-decay 0.1 \
  --muon-momentum 0.95 \
  --muon-ns-steps 5 \
  --muon-matmul-precision bf16 \
  --head-chunk-tokens 512 \
  --shape-bucket fine \
  --seed 42 \
  --steps 10 \
  --validation-batches 2 \
  --log-interval 1 \
  --log-text-file ./logs/specialist_general_smoke.log \
  --log-jsonl ./logs/specialist_general_smoke.jsonl \
  --resume --run
```

Resume verifies the specialist ID/layer count, exact training configuration,
sampler manifest hash, source weights, context, batch size, EOS, shape bucket,
RNG/cursors, optimizer update count, parameter schema, and foundation digest.

## 10. Troubleshooting

### Profile audit reports mismatches

- Start a new shell and source `scripts/0105_model_b_profile.env` again.
- Run from the repository root.
- Use `--peak-lr 1.5e-4`; older notes showing `3e-4` are stale for the foundation.
- Pass flags after the auditor's standalone `--`.
- Remember that omitted options are mismatches; defaults are not treated as
  equivalent by the audit.

### Backend is NumPy when CuPy was expected

`MINI_LLM_BACKEND` was set after importing `mini_llm`, or it was omitted. Exit
that Python process and start a new one with `MINI_LLM_BACKEND=cupy` already set.

### Out of memory

First confirm that no other process owns the GPU. Do not change the protected
run. For a new experimental run, reduce specialist batch/context/layers or use
checkpointing/recomputation; changing foundation production flags means it is no
longer the exact production profile. GPU pool endpoint logging is not a lifetime
peak measurement.

### Resume rejects tokenizer or data

A tokenizer mismatch is fatal. For strict resume, restore the exact tokenizer,
mixture manifest, and shards. A changed mixture may be recorded as a new data
phase by the trainer, but it is not an exact continuation.

### Resume reports incomplete parameters or optimizer state

Use a complete published checkpoint directory. Do not reconstruct missing files
by hand. If the active directory is absent but `<checkpoint>.previous` is
complete, the checkpoint loader can recover through that publication fallback.
Do not manipulate `.previous` while another writer is active.

### Inference `strict-fast` fails

The checkpoint/backend/shape did not execute a verified fast path. Retry with
`--execution-policy auto` for compatibility, then investigate the execution
report separately. Do not describe an automatic fallback as strict-fast success.

### Specialist resume fails

Repeat the original configuration exactly and use the same immutable foundation
copy. Common causes are a changed `--batch-size`, context, shape bucket,
activation/probability recomputation switch, mixture manifest, source weights,
or foundation files.

### A `.partial` data source remains after interruption

Inspect it, then remove only that source's `.partial` directory before restarting
that source. Never register partial output or delete completed source directories.
See [`DATA_PIPELINE.md`](DATA_PIPELINE.md).
