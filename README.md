# MyLocal_LLM

MyLocal_LLM is a local language-model research stack built around explicit
NumPy/CuPy tensors, handwritten forward and backward passes, visible optimizer
state, and architecture experiments designed for consumer hardware.

The project asks a practical question:

> How capable can a language model become when memory, bandwidth, compute, and
> training time are hard constraints?

Those constraints are design inputs. The repository favors inspectable model
mathematics, bounded-memory execution, reproducible checkpoints, and measured
behavior over hidden framework machinery or scale for its own sake.

## Current status

The current foundation is **Model B**: registry ID
`model_b_348m_sparse16`, training preset
`dual-resolution-matrix-pool-sparse-1024-20l`, and **348,261,376 parameters**.
It is a 20-layer, width-1,024 decoder with a 4,096-wide dense FFN,
matrix-pooled history, and deterministic stride-16 sparse retrieval.

[`docs/CURRENT_BASELINE.md`](docs/CURRENT_BASELINE.md) is the sole authority for
its exact architecture, pinned production profile, protected external-run
status, and verification boundary. Older 338M and 500M families are historical
research, not the current model.

## Philosophy

- **Own the numerics.** Parameters, gradients, attention, loss, optimizer state,
  and inference caches remain visible. NumPy is the readable reference;
  CuPy/Triton executes the same architecture on NVIDIA GPUs.
- **Separate model from execution.** `ModelConfig` defines geometry. Kernels,
  recomputation, bucketing, offload, and chunk sizes may alter performance, not
  model mathematics or checkpoint schema.
- **Measure behavior.** Use numerical tests, gradient checks, cold-load round
  trips, memory/throughput measurements, and held-out loss. CPU correctness does
  not prove GPU speed.
- **Protect durable state.** Exact continuation includes architecture, canonical
  tensors, optimizer, counters, RNG, tokenizer, and data provenance. Active
  checkpoints are operational assets, not test fixtures.

## What the repository provides

- a NumPy reference backend and a CuPy/Triton GPU backend;
- native training with explicit optimizer and checkpoint state;
- strict native checkpoint loading and request-scoped generation;
- GGUF and Hugging Face Safetensors loading for compatible Llama-family models;
- an OpenAI-compatible local server for native, GGUF, Safetensors, and remote
  model sources;
- frozen-foundation residual-specialist training infrastructure; and
- numerical, checkpoint, serving, data, benchmark, and GPU regression suites.

Compatibility is explicit rather than assumed. A recognized file format does
not imply support for every architecture, tokenizer, quantization, prompt
template, or execution policy.

## Setup

Requirements:

- Python 3.10 or newer;
- Linux for the primary development workflow;
- NumPy for CPU/reference work;
- a CUDA-compatible NVIDIA GPU and matching CuPy package for GPU work; and
- enough host RAM and disk for optimizer offload and transactional saves.

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CPU/reference development:

```bash
pip install -e ".[dev]"
MINI_LLM_BACKEND=numpy pytest -q -m "unit and not gpu and not slow"
```

CUDA 13 + Triton development:

```bash
pip install -e ".[gpu,triton-attention,dev]"
```

If the local CUDA runtime differs, install the compatible CuPy distribution
instead of assuming `cupy-cuda13x` will work.

## Native quickstart

### Inspect Model B without allocating its weights

```bash
MINI_LLM_BACKEND=numpy python - <<'PY'
from mini_llm.model_registry import registry
from mini_llm.model_contracts import ArchitectureSpec

config = registry.config("model_b_348m_sparse16")
print(config.estimated_parameter_count())
print(ArchitectureSpec.from_config(config))
PY
```

Expected parameter count: `348261376`.

Audit the pinned profile safely on CPU:

```bash
MINI_LLM_BACKEND=numpy pytest -q tests/test_production_profile_0105a.py
```

### Generate from a native checkpoint

Use a completed, approved, immutable checkpoint copy with its matching
`config.json` and tokenizer:

```bash
MINI_LLM_BACKEND=cupy python inference.py \
  --checkpoint /path/to/approved-checkpoint-copy \
  --prompt "A compact language model can" \
  --max-new-tokens 128 \
  --strategy top-p --temperature 0.8 --top-p 0.95
```

`inference.py` accepts native checkpoints. GGUF and Safetensors models use the
external loader, unified benchmark runner, or `serve.py`.

## Training safety

The complete launch and resume workflow lives in
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md). The exact profile lives in
`mini_llm/production_profile_0105.py` and is mirrored for shell use by
`scripts/0105_model_b_profile.env`.

Before any GPU operation:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Do not compete with, signal, inspect, hash, relocate, or reuse paths from an
active training run. New runs need new checkpoint and log destinations.
`--resume-from` is strict continuation; `--init-from` is a fresh run initialized
from compatible weights. Read
[`docs/CHECKPOINT_COMPATIBILITY.md`](docs/CHECKPOINT_COMPATIBILITY.md) before
either operation.

## Serve native and pretrained models locally

The local OpenAI-compatible server accepts these registry formats:

| Format | Source |
|---|---|
| `native` | Complete MyLocal_LLM checkpoint directory |
| `gguf` | Compatible Llama-family F32, F16, or Q8_0 GGUF |
| `safetensors` | Compatible Hugging Face directory with config, tokenizer, and all shards |
| `remote` | Existing OpenAI-compatible HTTP service |

Install external-model support, copy the canonical mixed registry example, and
edit its placeholder paths:

```bash
source .venv/bin/activate
python -m pip install -e ".[gpu,external,dev]"
cp serving_models.example.json serving_models.local.json
```

The example contains a native Model B copy and SmolLM2 GGUF/Safetensors entries.
The server does not download weights. For the official SmolLM2 download,
GGUF inspection, registry validation, remote providers, streaming, security,
and compatibility limits, see
[`docs/USER_GUIDE.md#8-pretrained-models-and-local-serving`](docs/USER_GUIDE.md#8-pretrained-models-and-local-serving).

Start the localhost-only server on the documented default port:

```bash
MINI_LLM_BACKEND=cupy python -u serve.py \
  --registry ./serving_models.local.json \
  --host 127.0.0.1 \
  --port 8000
```

Models load lazily. In another shell:

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "smollm2-360m-safetensors",
    "messages": [{"role": "user", "content": "Explain numerical stability briefly."}],
    "max_tokens": 128
  }'
```

IDE clients such as Zed can use `http://127.0.0.1:8000/v1` as a custom
OpenAI-compatible base URL and an ID returned by `/v1/models`.

Keep the default localhost bind. If access beyond localhost is intentional,
require `--api-key` or `MYLOCAL_LLM_API_KEY`, put TLS and network controls in
front of the service, and treat it as a development server rather than a
hardened multi-tenant deployment.

## Repository layout

```text
mini_llm/       model, ops, optimizer, data, checkpoint, inference, specialists
train_model.py  production foundation training entry point
inference.py    native autoregressive inference CLI
serve.py        OpenAI-compatible serving entry point
scripts/        profile auditors, regression runners, guarded workflows
benchmarks/     performance and evaluation programs
configs/        checked-in configuration artifacts
docs/           current guides, development log, and retained research artifacts
tests/          numerical, integration, checkpoint, and GPU regressions
```

Generated checkpoints, logs, token shards, downloaded weights, and benchmark
results are runtime state. Do not delete, relocate, repeatedly hash, or reuse
them as test outputs while a process may own them.

## Validation

Run the permanent bounded CPU gate:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python \
  scripts/run_0103a_regressions.py \
  --mode cpu \
  --report /tmp/model_b_cpu_regressions.json
```

Run GPU tests only when no important training, inference, or serving process
owns the device. CPU correctness does not prove full-model GPU parity,
throughput, or memory headroom.

## Documentation

Each topic has one canonical owner:

- [`docs/CURRENT_BASELINE.md`](docs/CURRENT_BASELINE.md) — exact current model and pinned profile;
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — installation and operational workflows;
- [`docs/MODEL_CENTERED_DEVELOPER_GUIDE.md`](docs/MODEL_CENTERED_DEVELOPER_GUIDE.md) — APIs, ownership, loading, execution, and HTTP semantics;
- [`docs/CHECKPOINT_COMPATIBILITY.md`](docs/CHECKPOINT_COMPATIBILITY.md) — resume, initialization, publication, and specialist binding;
- [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) — tokenizer, shards, manifests, and Common Pile procedures;
- [`benchmarks/README.md`](benchmarks/README.md) — benchmark commands and result semantics;
- [`tests/README.md`](tests/README.md) — test selection and CPU/GPU safety;
- [`docs/FUTURE_DEVELOPMENT.md`](docs/FUTURE_DEVELOPMENT.md) — prioritized open cleanup and acceptance work; and
- [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md) — completed changes, validation, and negative results.

Start with [`docs/README.md`](docs/README.md) for the complete documentation map.
Do not add standalone patch-report Markdown files; record durable changes in the
development log.

## License

This repository is licensed under the GNU Lesser General Public License v3.0.
See [`LICENCE.TXT`](LICENCE.TXT).
