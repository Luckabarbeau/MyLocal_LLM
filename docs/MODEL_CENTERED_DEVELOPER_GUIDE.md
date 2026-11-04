# Model-centered developer guide

This guide describes the current model-centered implementation at `eb61df0`
and its operational boundaries. It replaces older 500M/338M assumptions with
the current sparse Model B and frozen-foundation specialist reality.

## 1. Current Model B identity

The production native architecture is registered as
`model_b_348m_sparse16` and constructed by
`ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()`:

- 348,261,376 parameters;
- 20 decoder blocks, width 1024, dense FFN width 4096;
- 16 query heads, 4 KV heads, head width 64;
- matrix-pool dual-resolution attention with 128-token blocks;
- local window 512, one long-local 4096 head, and four summaries per block;
- stride-16 sparse original-token retrieval in foundation layers 3–18; and
- a 32,768-token vocabulary for the current run family.

The preset's maximum context and the selected training context are distinct. The
current production profile trains at 8,192 tokens and saves that selected value
in checkpoint configuration.

The exact production environment and scientific CLI contract live in
`mini_llm/production_profile_0105.py` and
`scripts/0105_model_b_profile.env`. The externally managed production run is
reported at step 165, but its checkpoint is not available in this workspace.
It is protected and must not be loaded, mutated, resumed, benchmarked, or used
as a specialist foundation from this checkout.

## 2. Ownership boundaries

The central design rule is that long-lived weights, request-local inference
state, and mutable training state have different owners.

| Owner | Owns | Does not own |
|---|---|---|
| `Model` | Native architecture/configuration, parameter arrays, tokenizer reference, checkpoint identity, model inspection, one-shot generation facade, exact native scoring entry point. | Request-local KV state, sampler history, optimizer moments, training RNG/cursors, specialist-composition training. |
| `GenerationSession` | One request's generation state, KV/cache engine state, generated token IDs, stopping state, repetition window/counts, request timing, and observed execution report. | Model weights or persistent multi-turn conversation state. |
| `TrainingRun` | One existing `ExtendedTrainer`: optimizer, scheduler, data samplers/cursors, RNG, counters, validation/checkpoint lifecycle, and step/token state. | A second training implementation or copied model weights. |
| `SpecialistTrainingRun` | Independent specialist optimizer/schedule, specialist sampler state, source counters, paired evaluation, and specialist checkpoint lifecycle. | Foundation gradients, foundation optimizer state, foundation checkpoint publication, or the generic `Model.train()` adapter. |

### `Model`

`Model.wrap_native(native_model)` reuses the exact native object and its parameter
arrays; it does not copy weights or gradients. `Model.load(...)` delegates to the
canonical native checkpoint loader. Backend choice remains process-global:
requesting a backend/device different from the already imported NumPy/CuPy
backend raises, and the advertised Torch adapter is not implemented.

`Model.capabilities` currently reports generation/streaming only when a tokenizer
is attached, exact native scoring, native training through `TrainingRun`, no
persistent sessions, and accessible weights. `Model.train()` supports the
native `ExtendedTrainer` path only. It does not construct a
`FrozenResidualComposition` or a `SpecialistTrainingRun`.

`Model.save_weights(path)` is weights-only. It does not publish optimizer state
and deliberately does not fabricate `config.json` or tokenizer metadata. A
caller that uses it must already own a complete, correctly paired artifact.

### `GenerationSession`

Every call to `Model.session()` creates request-local state over shared model
weights. Reusing a session starts a fresh/reset request; this is not append-only
multi-turn KV continuation. Do not drive one session concurrently, and consume a
stream fully before beginning another request on that session.

The shared session owns sampling and stopping. It seeds both NumPy and the active
backend RNG when a seed is supplied, enforces a non-empty prompt and available
context, and stops on adapter-provided token IDs or requested stop strings. The
adapter derives the native stop ID from the tokenizer; application code must not
assume a numeric EOS value.

### `TrainingRun`

`TrainingRun.from_native()` constructs the existing `ExtendedTrainer` around the
same native model instance and requires train and validation sources. It does
not implement another sampler, optimizer, forward pass, or backward pass.
`TrainingRun.wrap_native()` rejects a trainer attached to a different native
model object.

The run exposes the trainer's step, supervised-token counter, accumulation size,
training loop, full-state checkpoint save, and inspection data. For a true
resume, use the established `train_model.py --resume-from` path, which restores
model, optimizer, RNG, data, and schedule state. Wrapping a loaded model in a new
`TrainingRun` is not a substitute for resume.

## 3. Strict native loading

`load_native_checkpoint()` is shared by the CLI, `Model`, benchmarks, and local
serving. Its guarantees are:

1. `config.json` must exist and deserialize as `ModelConfig`.
2. A tokenizer must exist at `<checkpoint>/tokenizer.json` or, as a legacy
   fallback, in the checkpoint's parent directory.
3. The model is instantiated from the saved configuration and canonical
   parameter names.
4. Duplicate canonical names are rejected.
5. Every canonical parameter must be present.
6. Every loaded parameter shape must equal the instantiated shape.
7. Progressive active depth is restored from training state when applicable.
8. The process-global backend/device contract is never silently changed.

The loader does **not** compare an inference tokenizer's bytes against
`training_state.json` provenance. Training resume does perform that tokenizer
SHA check. Therefore a published native inference artifact must keep the exact
checkpoint and tokenizer together; the parent-directory fallback is a legacy
convenience, not an identity guarantee.

Native inference loading skips optimizer state. Production resume is stricter:
`train_model.py` requires the on-disk parameter set to equal the expected set,
requires optimizer and training state, verifies all required optimizer
components, and rejects tokenizer provenance changes. A changed local mixture
can be recorded as a new data phase, but it is not an exact resume.

External GGUF and Safetensors models use a separate Llama-family loader. GGUF
architectures/quantizations and tokenizer metadata must be supported; otherwise
loading fails or requires an adjacent `tokenizer.json`. Safetensors directories
require their configuration, weights, and tokenizer. Registration never implies
compatibility.

## 4. Exact scoring semantics

`Model.score(token_ids, target_ids, ...)` computes teacher-forced mean cross
entropy through the native model body and chunked LM head. It does not create an
optimizer, run backward, or retain full-vocabulary logits for all positions.

The contract is intentionally strict and low-level:

- inputs and targets must both be rank-2 `[batch, sequence]` arrays;
- their shapes must match;
- the **caller** owns next-token shifting;
- the caller owns EOS/document masks and any padding mask;
- the caller must pass training-equivalent `position_ids`, memory metadata, and
  dual-resolution metadata;
- target slices returned by the native body are applied to targets and masks;
- the result is a Python `float`; and
- non-native/remote providers must reject exact scoring rather than silently
  emulate a different objective.

Do not hardcode EOS. Derive it from the selected tokenizer or validated manifest.
For training-family code, the existing helper is:

```python
from pathlib import Path
from train_model import load_tokenizer_file, tokenizer_eos_id

tokenizer = load_tokenizer_file(Path(mixed_shard_root) / "tokenizer.json")
eos_token_id = tokenizer_eos_id(tokenizer)
```

A manifest EOS value is an identity assertion that must be checked against that
result, not a universal model constant.

## 5. Registries and execution limitations

Three registry/catalog concepts exist and must not be conflated.

### Architecture `ModelRegistry`

`mini_llm.model_registry.registry` maps a model ID to a configuration factory.
It currently registers `model_b_348m_sparse16`. This registry describes
architecture only. It does not locate checkpoints, load weights, select a
backend, prove kernel compatibility, or launch execution.

### Serving `ModelCatalog` / model-source registry

`ModelCatalog` and `load_specs()` parse JSON model records for native, GGUF,
Safetensors, or remote completion providers. They validate fields, resolve local
paths relative to the registry file, and preserve per-model sampling defaults.
A valid record is still only a source description; actual loading can fail on
missing files, unsupported architecture/quantization/tokenizer, context limits,
or execution policy.

### `KernelRegistry`

`KernelRegistry` stores exact-match `KernelImplementation` descriptors by
operation name/version, backend, mode, and implementation name. `eligible()`
returns sorted descriptors and can require `verified=True`. It does not benchmark,
rank, select, dispatch, or patch model operators. Registration itself is not a
validation result. Current runtime selection still occurs in the established
model/kernel code and inference policy.

`strict-fast` is the fail-closed execution policy. It checks eligibility before
expensive generation and requires the observed completed request to report a
verified fast path. `auto`, `balanced`, and `fast` can have different fallback
behavior; a fallback must not be reported as strict-fast success.

## 6. Production specialist reality after `eb61df0`

The older description of specialists as CPU-only is stale. The repository now
contains a guarded GPU specialist trainer at
`scripts/train_0105_specialist.py`, backed by
`SpecialistTrainingRun`. Its current production-facing behavior is:

- `--run` is mandatory before heavyweight imports, allocation, training, or
  checkpoint writes; without it the command is a path-validating dry run;
- actual execution requires `MINI_LLM_BACKEND=cupy`;
- the foundation must match sparse Model B geometry, including width/depth,
  matrix pooling, block/window settings, and stride-16 layers;
- the specialist is built from the foundation's exact block geometry;
- any positive microbatch size is supported subject to device memory;
- each microbatch selects one mixture source and samples independent
  document-first rows from that source;
- short rows are EOS-padded while tail targets and invalid historical blocks
  remain masked;
- batch sizes greater than one have explicit metadata/masking coverage;
- whole-specialist activation checkpoint replay is enabled by default;
- matrix-pool attention probabilities are recomputed by default, independently
  of whole-specialist activation checkpointing;
- source-preserving accumulation-length bucketing is enabled by default when
  both microbatch size and accumulation exceed one;
- supervised-token weighting is exact across variable-length microbatches;
- interval logs include foundation-style loss/rate fields and per-source CE;
- optional JSONL, text logs, synchronized stage profiling, and exact
  same-minibatch frozen-foundation CE are available;
- optimizer, schedule, sampler RNG/cursors, source counters, supervised-token
  counters, specialist schema, and foundation identity are resumable; and
- the specialist checkpoint is independent of the foundation checkpoint.

### Foundation freezing

`FrozenResidualComposition` obtains hidden states from the foundation, runs the
specialist correction, and uses the shared frozen LM head. Backward propagates
through the frozen head only to obtain `dH` and then through the specialist. The
foundation has no optimizer parameters or gradients. Construction rejects any
intersection between frozen parameter names and specialist optimizer names.

The correction projection is initialized to exact zero, so the initial composed
hidden state and CE match the foundation. The first implementation inherits
historical information through the foundation hidden states; it has no private
specialist history reader and no lexical reinjection.

### Activation and probability recomputation

Whole-specialist activation checkpointing stores each block input and replays
block forward operations in reverse during backward. Attention-probability
recomputation is encoded in the specialist's copied dual-resolution config and
can be controlled separately. Disabling either changes memory/performance
behavior and is part of the resume-relevant configuration.

Prepared RoPE positions are reused across specialist blocks/replays when the
corresponding runtime setting is enabled. The specialist carries the
foundation's matrix-pool geometry rather than the legacy GQA-only prototype.

### Bucketing and batch sizes above one

The sampler supports arbitrary positive batch sizes. When accumulation bucketing
is active and both batch and accumulation are greater than one, sampled batches
are regrouped by source-preserving document lengths. Dual-resolution metadata is
rebuilt for the new shapes, original rowwise target masks/provenance are retained,
and padded historical blocks remain invalid.

### Logging and resume

Each step can report total and per-source supervised tokens, combined CE,
gradient norm, learning rate, step time, target fill, and supervised/executed/
capacity throughput. `--train-foundation-ce` reports exact pre-update foundation
CE and gain on the same training minibatches at additional compute cost.
Endpoint CuPy pool values from profiling are not lifetime peak allocations.

Resume is fail-closed:

- serialized `SpecialistTrainingConfig` must exactly equal the requested config;
- specialist ID and layer count must match explicit CLI values;
- sampler manifest SHA, source weights, context, block size, batch size, EOS,
  shape bucket, source set, cursors, and RNG state must match;
- optimizer update count must match training state;
- specialist names/order/shapes must match; and
- the supplied foundation config/weight content digest must match.

`--steps` controls updates in one invocation and is not serialized in the run
configuration. The schedule's `--total-steps` is serialized and must not change.

### Current specialist limitations

The guarded trainer exists and is GPU-capable, but important limitations remain:

- no suitable completed Model B foundation checkpoint is available in this
  workspace;
- the external step-165 foundation run is protected and cannot be used;
- full 12-layer, 8K memory/performance acceptance remains hardware-dependent;
- there is no specialist-aware `GenerationSession` or composed incremental
  generation path;
- the generic `Model.train()` adapter does not own specialist composition;
- there is no private specialist historical reader, lexical reinjection,
  instruction-tuning layer, combiner, or tool-use training; and
- document-first rows avoid cross-document specialist attention rather than
  implementing packed-document specialist masks.

## 7. Checkpoint identity and publication

### Native checkpoint publication

`mini_llm.checkpoint.save_checkpoint()` writes a complete staged sibling
`<checkpoint>.pending-<uuid>` and then publishes by directory rename. The prior
complete snapshot is retained at `<checkpoint>.previous`. If interruption leaves
the active path absent, reads can fall back to `.previous`.

Important limits:

- staging and destination must share a filesystem;
- publication needs additional disk for the staged and previous snapshots;
- supporting files such as config/tokenizer are preserved into the staged
  directory, using hard links where applicable;
- writers must be serialized; there is no cross-process checkpoint lock;
- ancillary files must not be written concurrently inside the checkpoint;
- regular files are not individually `fsync`ed, so this is atomic publication,
  not complete power-loss durability; and
- no process may use publication to overwrite a protected or actively trained
  checkpoint during validation.

### Specialist checkpoints

Specialist checkpoints serialize only residual-specialist parameters and their
optimizer/training state, not another copy of the 348M foundation. They include:

- specialist architecture;
- ordered trainable parameter names;
- the resolved foundation path for provenance; and
- a SHA-256 over foundation `config.json` plus every frozen parameter file.

Fingerprinting is intentionally expensive and occurs at specialist save/resume
boundaries. Use a completed immutable foundation copy so hashing cannot race a
foundation publisher.

## 8. HTTP and model API caveats

The OpenAI-style server is intentionally small:

- supported local endpoints are `GET /health`, `GET /v1/models`,
  `POST /v1/completions`, and `POST /v1/chat/completions`;
- request bodies are limited to 16 MiB;
- only `n=1` is supported;
- response formats are text only; constrained/JSON-schema output is not
  implemented;
- only text content parts are accepted; images are not supported;
- local chat rendering supports explicit `chatml` and `plain` templates, not
  arbitrary foreign Jinja templates;
- local tool calling is a prompt/JSON-envelope compatibility mechanism, not
  native constrained decoding or server-side tool execution;
- forced named `tool_choice` is not implemented;
- local inference is serialized and at most one local model is kept resident on
  the GPU; switching local models unloads the prior one and frees the CuPy pool;
- remote completion engines can remain alongside the local model and own their
  upstream prompt/sampling/tool protocol;
- binding beyond localhost requires an API key, but the server is not a hardened
  internet-facing deployment; and
- persistent multi-turn KV sessions are not exposed through `Model` or HTTP.

Remote providers may not support exact seeds, `top_k`, raw text completions, or
identical chat semantics. The remote adapter forwards completion-level requests;
it does not make a remote provider eligible for exact native scoring or weight
inspection.

## 9. Validation entry points

CPU model-centered and specialist tests can run while the GPU is occupied:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q \
  tests/test_checkpoint_transactions_0104a.py \
  tests/test_model_b_config_roundtrip_0104b.py \
  tests/test_model_api_0103_0.py \
  tests/test_training_run_0104d.py \
  tests/test_specialist_composition_0104e.py \
  tests/test_specialist_checkpoint_0104f.py \
  tests/test_specialist_production_contract_0107a.py \
  tests/test_specialist_multibatch_0107d.py \
  tests/test_specialist_model_b_shape_0108.py \
  tests/test_specialist_prepared_rope_0108.py
```

GPU gates require exclusive ownership and enough free VRAM. Run them only in an
isolated environment and never concurrently with the protected foundation run.
