# Current baseline

This page is the concise source of truth for the current foundation model and its
pinned training profile after the model-centric refactor.

## Status

- **Current foundation:** Model B.
- **Registry ID:** `model_b_348m_sparse16`.
- **Training CLI model:** `dual-resolution-matrix-pool-sparse-1024-20l`.
- **Exact parameter count:** **348,261,376**.
- **Protected external run:** step **165**, reported by the user on **2026-11-04**.
- **Verification boundary:** that external checkpoint is not available in this
  checkout. Its weights, optimizer, RNG, tokenizer, provenance, and outputs
  cannot be inspected or verified here.
- **Code-change boundary:** the model-centric refactor did not change Model B
  mathematics or the foundation training algorithm.

Do not describe older 338M or 500M experiments as current. Historical details
belong in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).

## Model B architecture

| Contract field | Value |
|---|---:|
| Layers | 20 |
| `d_model` | 1,024 |
| Dense SwiGLU `d_ff` | 4,096 |
| Vocabulary | 32,768 |
| Query heads | 16 |
| KV heads | 4 |
| Head dimension | 64 |
| Architectural context horizon | 65,536 |
| Matrix-pool block size | 128 completed tokens |
| Learned summaries per completed block | 4 |
| Ordinary local window | 512 |
| Long-local branch | one 4,096-token head |
| Sparse original-token branch | stride 16, query head 1 |
| Sparse layers | zero-based layers `2..17` (human-numbered layers 3–18) |
| Maximum sparse keys at 65,536 | 4,096 per sparse head |

Every layer has exact local attention and compressed historical attention. Only
complete same-document blocks produce matrix-pool summaries, and a summary is
visible only after the represented block has completed. This preserves causal
and document-isolated behavior.

The 4,096-token head is retained. In zero-based layers `2..17`, query head 1 is
assigned deterministic sparse original-token history. The fixed phase schedule
is:

```text
0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15
```

Mapped to layers:

```text
layer:  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
phase:  0  8  4 12  2 10  6 14  1  9  5 13  3 11  7 15
```

The sparse branch uses existing GQA projections and adds no learned tensors.
Same-document masking prevents cross-document attention. At the full horizon,
each phase selects at most `65,536 / 16 = 4,096` original-token keys.

Code authorities:

- `mini_llm/config.py` — architecture preset;
- `mini_llm/model_registry.py` — `model_b_348m_sparse16` registration;
- `mini_llm/model/decoder_lm.py` and `mini_llm/inference_model.py` — phase wiring;
- `tests/test_baseline_contract_0103a.py` — framework-neutral architecture contract;
- `tests/test_model_b_config_roundtrip_0104b.py` — canonical config serialization.

## Pinned training profile

The comparison authority is `mini_llm/production_profile_0105.py`. Source
`scripts/0105_model_b_profile.env` in a **new shell** for the matching environment.
The environment file does not start, stop, or alter another process.

| Setting | Pinned value |
|---|---:|
| Backend | `cupy` |
| Objective | `ar` |
| Precision | `bf16-mixed` |
| Runtime profile | `consumer-gpu` |
| Training context | 8,192 |
| Batch size | 1 |
| Gradient accumulation | 512 |
| Accumulation weighting | `supervised_tokens` |
| Optimizer | `hybrid-muon` |
| Muon momentum | 0.95 |
| Muon Newton–Schulz steps | 5 |
| Muon matmul precision | `bf16` |
| Total steps | 100,000 |
| Warmup | 750 |
| Peak LR | `1.5e-4` |
| Gradient clip | 1.0 |
| Weight decay | 0.1 |
| Save interval | 50 |

The frozen profile additionally pins validation cadence, seeds, benchmark suite,
full optimizer offload with 16 MiB staging, fine dual-resolution shape bucketing,
Triton local/global/compression kernels, prepared RoPE, fused matrix-pool
backward, projection-gradient precision, LM-head chunking, and memory-trim
behavior. Runtime flags are execution policy; they are not permission to change
the architecture or checkpoint schema.

`--checkpoint-dir`, `--log-file`, and `--resume-from` are intentionally excluded
from the reusable profile. They are operational locations and must never be
copied from a protected run into a regression command.

## Operational boundaries

The exact model and profile above are intentionally separated from operational
tutorials:

- [`MODEL_CENTERED_DEVELOPER_GUIDE.md`](MODEL_CENTERED_DEVELOPER_GUIDE.md) owns
  model/session/training ownership, loading, scoring, execution, and HTTP API
  semantics.
- [`CHECKPOINT_COMPATIBILITY.md`](CHECKPOINT_COMPATIBILITY.md) owns strict resume,
  weights-only initialization, transactional publication, tokenizer/data
  identity, and specialist foundation binding.
- [`USER_GUIDE.md`](USER_GUIDE.md) owns launch, resume, inference, serving, and
  specialist workflows.

Specialist infrastructure exists, but CPU contracts do not prove full-size
composed-model GPU performance or end-to-end generation readiness.

## Validation

Safe CPU contract checks:

```bash
MINI_LLM_BACKEND=numpy pytest -q \
  tests/test_baseline_contract_0103a.py \
  tests/test_model_b_config_roundtrip_0104b.py \
  tests/test_checkpoint_transactions_0104a.py \
  tests/test_production_profile_0105a.py
```

Broader bounded CPU gate:

```bash
MINI_LLM_BACKEND=numpy python scripts/run_0103a_regressions.py \
  --mode cpu --report /tmp/model_b_cpu.json
```

GPU gates must run only on an idle device with enough memory. Passing these
checks protects selected contracts; it does not verify the unavailable external
step-165 checkpoint or establish full-run throughput.
