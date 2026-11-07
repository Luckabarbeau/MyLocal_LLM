# Future development and cleanup roadmap

This file owns forward-looking work. Completed changes, validation results, and
negative findings belong in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md). The exact
current architecture and profile belong in
[`CURRENT_BASELINE.md`](CURRENT_BASELINE.md).

## Compatibility boundary

The external Model B checkpoint is unavailable in this workspace. Until an
immutable copy is available, do not refactor model mathematics, parameter
names/order, checkpoint serialization, Hybrid Muon grouping/restoration, resume
counters, RNG/data cursors, dual-resolution sampling/metadata, sparse phase
values, or the training hot path.

Any change across that boundary requires all of the following:

1. an immutable copied checkpoint and separate output destinations;
2. exact **348,261,376-parameter** identity under the pinned profile;
3. one-step loss and parameter-update comparison;
4. optimizer-state, RNG-state, data-cursor, and counter comparison;
5. save/reload and strict-resume parity; and
6. idle-GPU throughput and memory comparison.

A CPU contract pass is necessary but is not sufficient evidence for these
changes.

## Priority 1 — safe readability cleanup

These items should not change model execution or durable artifact formats.

### 1. Enforce canonical documentation ownership

- **Affected files:** `README.md`, canonical guides under `docs/`,
  `benchmarks/README.md`, `tests/README.md`.
- **Benefit:** keeps the landing page concise and prevents architecture,
  checkpoint, serving, and data instructions from drifting.
- **Risk:** low; inaccurate links or deleted unique instructions are the main
  failure modes.
- **Required gates:** local Markdown-link check, command/path review, JSON parse
  of both serving examples, and `git diff --check`.
- **Prerequisites:** move unique content to its canonical owner before deleting
  duplicate prose. Keep exact architecture/profile values in
  `CURRENT_BASELINE.md` and completed-change evidence in `DEVELOPMENT_LOG.md`.

### 2. Extract shared benchmark harness utilities

- **Affected files:** the 0087–0089 benchmark/profile scripts, especially their
  repeated environment capture, subprocess timing, result serialization, and
  checkpoint/model setup.
- **Benefit:** reduces copied CLI, timing, JSON, and failure-reporting code while
  keeping individual experiments readable.
- **Risk:** low to medium because measurement setup or result schemas can drift
  during extraction.
- **Required gates:** existing 0087A/0087B/0089B/0089D/0089E contract tests,
  dry-run command snapshots, and before/after JSON-schema comparison.
- **Prerequisites:** inventory exact duplicated blocks and preserve existing CLI
  flags, warmup policy, synchronization points, output keys, and exit behavior.
  Do not merge experiments merely because their harness code is similar.

### 3. Extract reusable test helpers

- **Affected files:** the 15 test modules imported by other `test_*.py` files,
  plus future helper modules under a non-test package such as `tests/helpers/`.
- **Benefit:** removes brittle bare imports and makes later test reorganization
  possible without changing coverage.
- **Risk:** low to medium; collection order and import-path changes can hide or
  duplicate tests.
- **Required gates:** full `pytest --collect-only`, permanent CPU regression
  gate, and focused tests for every moved helper.
- **Prerequisites:** map import edges first, move only non-test helpers/fixtures,
  and verify identical collected test IDs before renaming historical modules.
  Do not bulk-move numbered tests before this work is complete.

### 4. Normalize optional-dependency handling

- **Affected files:** external GGUF/Safetensors/Hugging Face tests and any module
  importing optional packages at collection time.
- **Benefit:** a minimal developer environment collects cleanly and records
  unavailable integrations as explicit skips.
- **Risk:** low, provided skips are restricted to genuinely optional features.
- **Required gates:** collection with base/dev dependencies, focused execution
  with each optional dependency installed, and assertions that mandatory
  dependency failures still fail.
- **Prerequisites:** document each optional extra in `pyproject.toml`; use
  `pytest.importorskip` at module/fixture scope with a reason rather than broad
  exception handling.

### 5. Require explicit GPU opt-in at item level

- **Affected files:** `tests/conftest.py` and Torch/CUDA smoke tests whose file
  names do not reliably express GPU allocation.
- **Benefit:** prevents an isolated Torch environment from allocating CUDA while
  the NumPy suite is expected to be CPU-only.
- **Risk:** medium if markers accidentally exclude CPU branches or leave GPU
  coverage unselected.
- **Required gates:** collection/marker assertions in CPU-only, Torch-with-CUDA,
  and CuPy environments; permanent CPU runner must allocate no GPU.
- **Prerequisites:** inventory all direct/indirect CUDA checks and add explicit
  markers or an opt-in fixture before changing marker selection rules.

### 6. Test the pinned environment mirror

- **Affected files:** `mini_llm/production_profile_0105.py`,
  `scripts/0105_model_b_profile.env`, and profile tests.
- **Benefit:** catches drift between the Python comparison authority and shell
  launch environment.
- **Risk:** low; parsing shell syntax incorrectly could create false confidence.
- **Required gates:** direct mirror test for every represented key plus the
  existing production-profile audit.
- **Prerequisites:** define which values are intentionally Python-only or
  shell-only before asserting equality.

## Priority 2 — medium-risk code consolidation

These changes should be performed independently, with narrow diffs and focused
parity tests. They must not alter Model B tensors or training behavior.

### 1. Centralize the sparse phase schedule

- **Affected symbols:** phase tuples in `mini_llm/model/decoder_lm.py`,
  `mini_llm/inference_model.py`, `mini_llm/native_dual_cuda_graph_decode.py`, and
  `mini_llm/specialists/residual_specialist.py`.
- **Benefit:** one named immutable architecture constant prevents train,
  inference, graph decode, and specialists from drifting.
- **Risk:** medium/high because phase order is protected model mathematics.
- **Required gates:** baseline contract, sparse full/incremental parity, CUDA
  graph parity, specialist phase/shape tests, and copied-checkpoint GPU parity.
- **Prerequisites:** preserve exact values and indexing; import the constant
  without introducing hot-loop work or circular imports.

### 2. Replace duplicated preset choices/dispatch with one preset map

- **Affected symbols:** CLI `--model` choices and dispatch in `train_model.py`,
  `ModelConfig` factories, and `mini_llm/model_registry.py`.
- **Benefit:** adding or retiring a preset updates one declarative source.
- **Risk:** medium because names are user-facing and checkpoint configs must keep
  deserializing.
- **Required gates:** CLI parser tests, registry/config round trips, exact Model B
  parameter identity, and legacy preset compatibility tests.
- **Prerequisites:** distinguish public stable names, aliases, and historical
  factories. Do not remove a name until repository and external migration are
  documented.

### 3. Consolidate tokenizer loading and EOS resolution

- **Affected symbols:** tokenizer/config loading in `train_model.py`, native and
  external model loading, generation, benchmark, and serving paths.
- **Benefit:** one fail-closed policy for tokenizer identity, EOS derivation,
  missing metadata, and error messages.
- **Risk:** medium/high because training targets, stopping, and checkpoint/data
  compatibility depend on exact behavior.
- **Required gates:** tokenizer hash/provenance tests, native and external loader
  tests, generation stopping tests, data-manifest tests, and strict-resume tests.
- **Prerequisites:** write down context-specific differences first. Inference's
  legacy parent-directory fallback must not accidentally weaken strict training
  resume.

### 4. Simplify the native inference CLI

- **Affected files/symbols:** `inference.py` one-shot and interactive loops,
  unused fields, and their overlap with `GenerationSession`.
- **Benefit:** less duplicated sampling/generation code and clearer execution
  reports.
- **Risk:** medium because output text, seeding, stop conditions, cache policy,
  and interactive UX are public behavior.
- **Required gates:** generation-session tests, deterministic CLI snapshots,
  reference/KV-cache parity, stop-token/string behavior, and execution-policy
  tests.
- **Prerequisites:** identify which CLI behaviors are intentionally different
  from server generation; preserve flags and output format before removing old
  code.

### 5. Migrate internal compatibility imports

- **Affected symbols:** internal callers importing reexports from
  `mini_llm/serving_models.py` rather than canonical model-source/serving
  modules.
- **Benefit:** clarifies module ownership while retaining a stable compatibility
  surface for external callers.
- **Risk:** low to medium due to import cycles and downstream imports not visible
  in this repository.
- **Required gates:** serving, remote-provider, model-catalog, and import smoke
  tests.
- **Prerequisites:** migrate internal imports first; keep documented reexports
  with deprecation policy before considering deletion.

### 6. Reduce generation-adapter duplication

- **Affected symbols:** native, external, and remote generation adapters plus
  benchmark/serving request translation.
- **Benefit:** shared validation and stopping/sampling semantics with fewer
  divergent implementations.
- **Risk:** medium/high because providers intentionally differ in tokenization,
  logits access, seeds, streaming, and tool envelopes.
- **Required gates:** native generation parity, GGUF/Safetensors tests, remote
  request snapshots, streaming tests, and benchmark golden results.
- **Prerequisites:** extract only behavior proven identical. Keep provider-owned
  semantics explicit rather than forcing false uniformity.

## Priority 3 — deferred/high-risk refactors

Do not begin these while the protected run lacks an immutable copied checkpoint
and full parity environment.

### 1. Decompose foundation training orchestration

- **Affected symbols:** `train_model.main`, `ExtendedTrainer.__init__`, and
  `ExtendedTrainer.train_step`.
- **Benefit:** smaller ownership units, clearer configuration validation, and
  easier focused testing.
- **Risk:** very high: accumulation weighting, optimizer ordering, offload,
  logging, validation cadence, memory lifetime, and throughput can change.
- **Required gates:** full copied-checkpoint checklist above plus deterministic
  tiny trajectories and production-profile idle-GPU comparison.
- **Prerequisites:** first extract pure parsing/reporting code. Do not split or
  reorder numerical work merely to reduce function length.

### 2. Refactor checkpoint/resume internals

- **Affected files:** `mini_llm/checkpoint.py`, optimizer-state serialization,
  trainer restore, transaction publication, and specialist foundation binding.
- **Benefit:** clearer schemas and less repeated validation/publication logic.
- **Risk:** critical; a subtle change can make an existing checkpoint unloadable
  or silently non-equivalent.
- **Required gates:** legacy/current fixture matrix, interrupted-publication
  recovery, strict resume, optimizer component/state placement, RNG/cursor
  parity, save/reload, and copied Model B continuation.
- **Prerequisites:** version schemas explicitly and preserve read compatibility.
  Never test against an active checkpoint directory.

### 3. Restructure `ModelConfig` and `InferenceModel`

- **Affected symbols:** the large `ModelConfig` schema/factories and
  `InferenceModel` construction/cache/execution code.
- **Benefit:** separates architecture data from runtime policy and makes
  inference components easier to reason about.
- **Risk:** very high because config serialization, parameter construction/order,
  cache layout, and fast-path selection are compatibility boundaries.
- **Required gates:** every config round trip, exact parameter names/order/count,
  native load, full/incremental parity, sparse/dual-resolution tests, and GPU
  graph/throughput gates.
- **Prerequisites:** preserve serialized field names/defaults and canonical
  parameter traversal; use additive adapters before structural migration.

### 4. Remove process-global backend/runtime state

- **Affected files:** backend import selection, environment-driven runtime flags,
  kernel registries/caches, model loading, tests, and serving model switching.
- **Benefit:** safer multi-model embedding and more isolated tests.
- **Risk:** very high: array types, compiled kernels, memory pools, and module
  initialization currently assume process-global state.
- **Required gates:** NumPy/CuPy isolation, local-model eviction, kernel cache,
  external-loader, generation, training, and performance tests in fresh and
  mixed processes.
- **Prerequisites:** design explicit backend/context ownership and migration
  semantics. Do not emulate per-instance isolation while hidden globals remain.

## Scientific and operational acceptance still open

- Verify any published Model B artifact with checkpoint/config/tokenizer hashes,
  training state, logs, finite metrics, and a read-only load/resume audit.
- Run Model B long-retrieval, held-out LM, and capability suites from immutable
  checkpoint copies.
- Validate full-size specialist training on an idle target GPU: two-layer smoke,
  then 12-layer context/memory/throughput and deterministic resume.
- Compare specialist and frozen foundation on matched supervised tokens and
  compute before adding composed incremental generation.
- Run the SmolLM2 control only within its native context and record exact model
  and tokenizer hashes.
- Report negative/null results, unsupported cases, confusion/answer bias, and
  counterfactual sensitivity—not aggregate accuracy alone.
