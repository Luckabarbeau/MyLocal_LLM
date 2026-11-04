# MyLocal_LLM

MyLocal_LLM is a local language-model research stack built around explicit
NumPy/CuPy tensors, handwritten forward and backward passes, visible optimizer
state, and architecture experiments designed for consumer hardware.

The project asks a practical question:

> How capable can a language model become when memory, bandwidth, compute, and
> training time are hard constraints?

Those constraints are treated as design inputs. The repository favors inspectable
model mathematics, bounded-memory execution, reproducible checkpoints, and
measured behavior over hidden framework machinery or scale for its own sake.

## Current status

The current foundation is **Model B**, registry ID
`model_b_348m_sparse16`, exposed by the training CLI as
`dual-resolution-matrix-pool-sparse-1024-20l`.

An external protected training run was at **step 165**, as reported by the user on
**2026-11-04**. Its checkpoint is not present in this checkout, so its files,
optimizer state, data provenance, and numerical behavior have **not** been
verified here. Do not infer more than that user-reported status.

The model-centric refactor added unified model loading, generation, scoring,
training adapters, checkpoint safety, and specialist interfaces. **No foundation
training code or Model B mathematics changed as part of this documentation
update.**

Older 338M and 500M families are historical research, not the current model;
see [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md).

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

## Current Model B

| Field | Current value |
|---|---:|
| Registry model | `model_b_348m_sparse16` |
| Training CLI model | `dual-resolution-matrix-pool-sparse-1024-20l` |
| Parameters | **348,261,376** |
| Transformer layers | 20 |
| Model width | 1,024 |
| Dense SwiGLU width | 4,096 |
| Vocabulary | 32,768 |
| Query heads | 16 |
| KV heads | 4 |
| Head dimension | 64 |
| Addressable context | 65,536 tokens |
| Matrix-pool block | 128 completed tokens |
| Summaries per completed block | 4 |
| Standard local window | 512 tokens |
| Long-local heads | 1 head with a 4,096-token window |
| Sparse historical stride | 16 original tokens |
| Sparse layers | zero-based layers 2 through 17 |
| Maximum sparse keys/head at 65,536 | 4,096 |

Model B is a dense-FFN decoder with grouped-query attention. It does not run
dense token-to-token attention over all 65,536 positions.

Each layer retains exact local attention and compressed historical attention.
Every completed, same-document 128-token block contributes four learned
matrix-pool summaries. A summary becomes visible only after its block is
complete, preserving causality and document isolation. One query head keeps a
4,096-token exact local window; ordinary local heads use 512 tokens.

In zero-based layers `2..17`, query head 1 is a deterministic original-token
sparse-history head. Its layer phases are:

```text
layer:  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
phase:  0  8  4 12  2 10  6 14  1  9  5 13  3 11  7 15
```

A sparse layer attends same-document keys whose absolute token index matches its
phase modulo 16. The schedule covers all 16 phases without adding learned
parameters. At the 65,536-token horizon, one phase contains at most 4,096 keys.

## Pinned foundation profile

The current pinned profile is defined by
`mini_llm/production_profile_0105.py`. Its environment is mirrored by
`scripts/0105_model_b_profile.env`.

| Training setting | Pinned value |
|---|---:|
| Training context | 8,192 |
| Microbatch | 1 |
| Gradient accumulation | 512 |
| Accumulation weighting | `supervised_tokens` |
| Optimizer | `hybrid-muon` |
| Muon momentum | 0.95 |
| Muon Newton–Schulz steps | 5 |
| Muon matmul precision | `bf16` |
| Precision | `bf16-mixed` |
| Total optimizer steps | 100,000 |
| Warmup | 750 steps |
| Peak learning rate | `1.5e-4` |
| Gradient clip | 1.0 |
| Weight decay | 0.1 |
| Save interval | 50 steps |

The frozen profile also pins validation, seed, benchmark-suite, optimizer
offload, bucketing, Triton, prepared-RoPE, fused-backward, and memory-trim
settings. Treat the Python profile as the comparison authority; do not reconstruct
it from old command snippets.

## Setup

Requirements:

- Python 3.10 or newer;
- Linux for the primary development workflow;
- NumPy for CPU/reference work;
- a CUDA-compatible NVIDIA GPU and matching CuPy package for GPU work;
- enough host RAM and disk for full optimizer offload and transactional saves.

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

## Safe quickstart

### 1. Inspect the registered architecture without allocating 348M weights

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

### 2. Audit the pinned profile safely

```bash
MINI_LLM_BACKEND=numpy pytest -q tests/test_production_profile_0105a.py
```

For a new shell intended for Model B GPU work:

```bash
source scripts/0105_model_b_profile.env
```

Sourcing the file changes only that shell. It does not restart or modify an
existing process.

### 3. Review safety before training

Use `mini_llm/production_profile_0105.py` as the full launch contract and always
choose a new destination for a new run. Strict resume and weights-only
initialization are intentionally omitted here. Read
[`docs/CHECKPOINT_COMPATIBILITY.md`](docs/CHECKPOINT_COMPATIBILITY.md) before
using either operation; never substitute an active or protected checkpoint into
a test or example command.

### 4. Inference from an approved immutable checkpoint copy

```bash
MINI_LLM_BACKEND=cupy python inference.py \
  --checkpoint /path/to/approved-checkpoint-copy \
  --prompt "A compact language model can" \
  --max-new-tokens 128 --strategy top-p --temperature 0.8 --top-p 0.95
```

The checkpoint's `config.json` and tokenizer define the model. Do not mix a
checkpoint with a convenient tokenizer from another dataset.

## Checkpoint rules in one minute

`--resume-from` is strict continuation; `--init-from` is a fresh run initialized
from matching weights. Saved config, tokenizer/data pairing, exact tensor schema,
optimizer/RNG state, transactions, active-path safety, specialist foundation
SHA256, and derived-cache round trips are covered in
[`docs/CHECKPOINT_COMPATIBILITY.md`](docs/CHECKPOINT_COMPATIBILITY.md).

## Repository layout

```text
mini_llm/       model, ops, optimizer, data, checkpoint, inference, specialists
train_model.py  production foundation training entry point
inference.py    native autoregressive inference CLI
serve.py        serving entry point
scripts/        profile auditors, regression runners, guarded workflows
benchmarks/     performance and evaluation programs
configs/        checked-in configuration artifacts
docs/           current guides, development log, and retained research artifacts
tests/          numerical, integration, checkpoint, and GPU regressions
```

Generated `checkpoints/`, `logs/`, token shards, and benchmark results are runtime
state. Do not delete, relocate, hash repeatedly, or reuse them as test outputs
while a process may own them.

## Validation

Focused CPU checkpoint and profile checks:

```bash
MINI_LLM_BACKEND=numpy pytest -q \
  tests/test_baseline_contract_0103a.py \
  tests/test_model_b_config_roundtrip_0104b.py \
  tests/test_checkpoint_transactions_0104a.py \
  tests/test_production_profile_0105a.py \
  tests/test_specialist_checkpoint_0104f.py
```

Run GPU tests only when no important training or inference process owns the
GPU. Check `nvidia-smi` first. CPU correctness does not prove full-model GPU
parity or throughput.

## Documentation

- [`docs/CURRENT_BASELINE.md`](docs/CURRENT_BASELINE.md) — current Model B snapshot;
- [`docs/CHECKPOINT_COMPATIBILITY.md`](docs/CHECKPOINT_COMPATIBILITY.md) — checkpoint contract and safe operations;
- [`docs/README.md`](docs/README.md) — documentation map;
- [`docs/MODEL_CENTERED_DEVELOPER_GUIDE.md`](docs/MODEL_CENTERED_DEVELOPER_GUIDE.md) — current APIs;
- [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) — tokenizer, shards, manifests, and provenance;
- [`docs/FUTURE_DEVELOPMENT.md`](docs/FUTURE_DEVELOPMENT.md) — current roadmap and open gates;
- [`tests/README.md`](tests/README.md) — test markers and safe workflows;
- [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md) — historical models and development record.

## License

This repository is licensed under the GNU Lesser General Public License v3.0.
See [`LICENCE.TXT`](LICENCE.TXT).
