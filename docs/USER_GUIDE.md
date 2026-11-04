# MyLocal_LLM User Guide

This is the operational guide for the current **Model B** workflow. The older
`wide-500m-*` and 338M historical-hybrid examples elsewhere in the repository
are research history, not the production launch recipe documented here.

Model B is the native preset
`dual-resolution-matrix-pool-sparse-1024-20l`. Its sparse branch covers
zero-based layers `2..17` (human-numbered layers 3–18). Exact geometry, parameter
count, pinned profile, protected external-run status, and the verification
boundary are canonical in [`CURRENT_BASELINE.md`](CURRENT_BASELINE.md).

Do not inspect, load, resume, overwrite, signal, benchmark against, or otherwise
touch the protected external run. Wait for an explicitly published immutable
checkpoint copy or use a new, independent run directory.

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

Continue in the shell where section 4 defined and audited `MODEL_B_FLAGS`. If
this is a new shell, source the profile and define/audit that array once using
section 4; do not maintain a second copy of the flags.

```bash
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

The protected external run cannot be resumed from this checkout. See the
verification boundary in [`CURRENT_BASELINE.md`](CURRENT_BASELINE.md); do not
fabricate a checkpoint path or start a replacement under the protected run's
name.

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

## 8. Pretrained models and local serving

MyLocal_LLM can expose native checkpoints and compatible pretrained models
through one local OpenAI-style API:

| Registry format | Source contract |
|---|---|
| `native` | Complete native checkpoint with `config.json`, matching tokenizer, and every canonical tensor |
| `gguf` | One compatible Llama-family GGUF plus embedded tokenizer metadata or adjacent `tokenizer.json` |
| `safetensors` | Compatible Hugging Face directory with `config.json`, `tokenizer.json`, and all referenced shards |
| `remote` | OpenAI-compatible base URL, upstream model ID, and optional API-key environment variable |

`inference.py` accepts native checkpoints only. External local models use the
external loader, unified benchmark runner, or `serve.py`.

### Install external-model support

```bash
source .venv/bin/activate
python -m pip install -e ".[gpu,external,dev]"
```

The `external` extra installs `safetensors`, `huggingface_hub`, and `gguf`.
Downloads require network access, disk space, and acceptance of the source
model's license/access terms. The server never downloads weights automatically.

### Download official SmolLM2 Safetensors

Choose a directory outside active checkpoints:

```bash
mkdir -p pretrained_models
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    local_dir="./pretrained_models/SmolLM2-360M-Instruct",
)
PY
```

The directory must contain model configuration, tokenizer, and every referenced
Safetensors shard. The in-process external engine targets compatible
Llama-family layouts; registration does not convert unsupported architectures.

### Inspect or run a compatible SmolLM2 GGUF

Supply a GGUF from a trusted source or compatible conversion workflow. The
current loader accepts F32, F16, and Q8_0 tensors. Q8_0 is dequantized at load;
common Q4_K/Q6_K files and arbitrary `llama.cpp` architectures are unsupported.

Inspect metadata on CPU without allocating the full model on GPU:

```bash
MINI_LLM_BACKEND=numpy python -m benchmarks.try_external_gguf \
  --model ./pretrained_models/SmolLM2-360M-Instruct-F16.gguf \
  --inspect
```

On an idle GPU, run a direct smoke test:

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.try_external_gguf \
  --model ./pretrained_models/SmolLM2-360M-Instruct-F16.gguf \
  --chat \
  --prompt "Explain why numerical stability matters." \
  --max-new-tokens 128 \
  --dtype float16
```

`--chat` applies this tool's SmolLM2-style ChatML wrapper. Verify each model's
actual prompt template and tokenizer rather than assuming ChatML.

### Configure and validate the local registry

The checked-in example is the canonical mixed local registry. It includes
placeholder entries for a published native Model B copy, SmolLM2 GGUF, and
SmolLM2 Safetensors:

```bash
cp serving_models.example.json serving_models.local.json
```

Keep only entries whose files exist and edit every `path`. Relative paths resolve
from the registry JSON's directory, not necessarily the shell's current working
directory. JSON does not expand `$HOME` or other shell variables.

`context_length` is a server upper bound and may not exceed loaded-model
capacity. It does not extend RoPE or establish quality at that length. Request
sampling fields override per-model defaults individually.

Validate the schema without loading model weights:

```bash
MINI_LLM_BACKEND=numpy python - <<'PY'
from mini_llm.model_sources import ModelCatalog

catalog = ModelCatalog.from_manifest("serving_models.local.json")
print(catalog.names())
for name in catalog.names():
    print(name, catalog.get(name))
PY
```

### Start the server and make requests

Check GPU ownership first. Local models load lazily on first request:

```bash
nvidia-smi
MINI_LLM_BACKEND=cupy python -u serve.py \
  --registry ./serving_models.local.json \
  --host 127.0.0.1 \
  --port 8000
```

The development server keeps at most one local model GPU-resident and serializes
local inference. Requesting another local model evicts the previous model and
releases the CuPy memory pool. Remote entries can coexist because their weights
execute upstream.

Endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

In another shell:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "smollm2-360m-safetensors",
    "messages": [{"role": "user", "content": "Write a Python primality test."}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

For native Model B completion, select its registry ID:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model-b","prompt":"A compact language model can","max_tokens":64}'
```

For server-sent-event streaming:

```bash
curl -Ns http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "smollm2-360m-gguf",
    "messages": [{"role": "user", "content": "Explain matrix multiplication briefly."}],
    "max_tokens": 128,
    "stream": true
  }'
```

IDE clients such as Zed can use `http://127.0.0.1:8000/v1` as a custom
OpenAI-compatible base URL and an ID returned by `/v1/models`.

### Add an existing remote provider

A remote registry entry names an upstream model and keeps credentials in an
environment variable rather than JSON. Copy
`serving_models.remote.example.json`, then set its optional `api_key_env` field
if the provider requires authentication. Remote providers own tokenization,
execution, and exact sampling semantics; they do not become native models with
accessible weights, logits, or training state.

### Security and compatibility limits

- Keep the default `127.0.0.1` bind. Beyond localhost, require `--api-key` or
  `MYLOCAL_LLM_API_KEY`, use TLS through a suitable reverse proxy, and restrict
  network access.
- Never serve or repeatedly hash an actively written training checkpoint.
  Publish or copy an immutable checkpoint first.
- Registry validation proves only JSON schema validity. File completeness,
  tensor names, architecture, tokenizer metadata, and quantization are checked
  during lazy loading.
- Native loading requires exact canonical names/shapes. Safetensors needs all
  shards. GGUF tokenizer reconstruction fails closed without compatible embedded
  metadata or adjacent `tokenizer.json`.
- `chat_template` supports `chatml` and `plain`, not arbitrary Hugging Face Jinja
  templates. Confirm generated prompts for every instruction model.
- Tool envelopes are compatibility plumbing, not constrained decoding or local
  tool execution. Use `tool_format: none` unless the complete path is tested.
- CUDA Graph and strict-fast modes are model/shape dependent. Establish output
  parity before enabling fail-closed fast-path settings.

See [`MODEL_CENTERED_DEVELOPER_GUIDE.md`](MODEL_CENTERED_DEVELOPER_GUIDE.md) for
HTTP semantics and API limitations, and [`../benchmarks/README.md`](../benchmarks/README.md)
for evaluation workflows.

## 9. Frozen-foundation specialist training

The guarded specialist launcher added by `eb61df0` is executable on GPU, but it
requires a **completed, immutable copy** of a sparse Model B foundation. It may
never use the active protected run or a checkpoint directory another process is
replacing. No suitable foundation checkpoint is available in this workspace at
present.

The foundation remains frozen. Only independent residual-specialist parameters
are optimized; the specialist checkpoint stores a content fingerprint of the
foundation and resume rejects a different foundation.

### Define specialist flags once

The dry run validates paths and prints the plan. It does not import heavy model
code, allocate a model, train, or write a checkpoint. The foundation and mixture
paths must exist, and a new run's specialist destination must not exist.

Define resume-relevant scientific flags once in the shell:

```bash
FOUNDATION=/absolute/path/to/immutable_model_b_copy
MIXTURE=./token_shards_ensemble_32k_mixed
SPECIALIST=/absolute/path/to/new_specialist_smoke

SPECIALIST_FLAGS=(
  --specialist-id general
  --specialist-layers 2
  --context-length 512
  --batch-size 2
  --grad-accum-steps 2
  --total-steps 10000
  --warmup-steps 100
  --peak-lr 3e-4
  --grad-clip 1.0
  --weight-decay 0.1
  --muon-momentum 0.95
  --muon-ns-steps 5
  --muon-matmul-precision bf16
  --head-chunk-tokens 512
  --shape-bucket fine
  --seed 42
  --validation-batches 2
)
```

### Dry run

```bash
.venv/bin/python scripts/train_0105_specialist.py \
  --foundation-checkpoint "$FOUNDATION" \
  --mixed-shard-root "$MIXTURE" \
  --specialist-checkpoint "$SPECIALIST" \
  "${SPECIALIST_FLAGS[@]}" \
  --steps 1
```

### GPU smoke run

Run only on an idle GPU after the dry run succeeds, continuing in the same shell:

```bash
source scripts/0105_model_b_profile.env

.venv/bin/python scripts/train_0105_specialist.py \
  --foundation-checkpoint "$FOUNDATION" \
  --mixed-shard-root "$MIXTURE" \
  --specialist-checkpoint "$SPECIALIST" \
  "${SPECIALIST_FLAGS[@]}" \
  --steps 1 \
  --log-interval 1 \
  --profile-steps 1 \
  --log-text-file ./logs/specialist_general_smoke.log \
  --log-jsonl ./logs/specialist_general_smoke.jsonl \
  --run
```

By default the trainer enables whole-specialist activation checkpoint replay,
matrix-pool attention-probability recomputation, and source-preserving length
regrouping when batch size and accumulation are both greater than one. The
`--no-*` switches alter resume configuration. `--train-foundation-ce` adds exact
same-minibatch frozen-foundation CE at the cost of another frozen LM-head pass.

A 12-layer, 8,192-context run is a separate scientific experiment. Validate
VRAM, finite gradients, throughput, and checkpoint/resume first; do not infer
safe settings from the foundation's accumulation count.

### Specialist resume

Resume uses `--resume --run`, not `--resume-from`. In a new shell, redefine
`FOUNDATION`, `MIXTURE`, and `SPECIALIST_FLAGS` exactly as above, and point
`SPECIALIST` at the existing specialist directory. `--steps` is invocation-local
and may change; `--total-steps` remains the original schedule horizon.

```bash
SPECIALIST=/absolute/path/to/existing_specialist_smoke

.venv/bin/python scripts/train_0105_specialist.py \
  --foundation-checkpoint "$FOUNDATION" \
  --mixed-shard-root "$MIXTURE" \
  --specialist-checkpoint "$SPECIALIST" \
  "${SPECIALIST_FLAGS[@]}" \
  --steps 10 \
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
