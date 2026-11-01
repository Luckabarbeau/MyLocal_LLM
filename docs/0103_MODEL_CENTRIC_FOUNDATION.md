# 0103-0 — Model-centric API foundation (first implementation)

Status: **additive compatibility foundation**, not the completed 0103 refactor.
This changes no model mathematics, numerical kernels, optimizer behavior,
checkpoint format, runtime defaults, existing CLI, serving code or benchmarks.

## Public entry point (implemented)

```python
from mini_llm import Model
model = Model.load("checkpoints/my_native_run", backend="auto", device="auto")
info = model.inspect()
print(info.architecture.family, info.parameter_count)
print(model.execution_info())
response = model.generate("Hello", max_new_tokens=32)
print(response.text, response.token_ids)
```

`Model.load` currently accepts *native* checkpoints, reproducing the existing
config, tokenizer and checkpoint loading behavior without importing the
benchmark package. It verifies that all expected parameter names and shapes
are present; unlike some legacy partial-load workflows, missing weights fail.
`Model.wrap_native(existing, tokenizer=...)` wraps a current
`DecoderLanguageModel` without copying parameters or allocating another model.
`Model.generate` delegates to the **existing** `inference.TextGenerator` and
always requests its KV-cached path. It has not been benchmarked or proven to
select a faster path than legacy CLI serving. A fresh generator is created for
each call: persistent sessions are *not* implemented yet. Results are
structured `GenerationResult` records, including observed execution metadata.

### Critical transitional limitations

* `mini_llm.backend` still binds NumPy/CuPy globally at import time. Set
  `MINI_LLM_BACKEND=numpy` or `MINI_LLM_BACKEND=cupy` **before launching** the
  Python process. `backend='auto'` selects only this bound backend. Asking for
  an unavailable backend/device fails; it does not switch silently.
* The performance-report infrastructure is present but cannot yet verify
  active kernels or CUDA Graph capture. Accordingly `require_fast=True`
  **fails explicitly** until patch 0103D establishes verification. Normal
  generation does not assert that it is accelerated.
* Unified `Model.train`, `Model.score`, `Model.session`, and weights-only export
  are **not yet connected**; they raise `UnsupportedModelOperation`. Existing
  training/scoring/session tools remain available unchanged.
* Internal GGUF/safetensors loading, remote OpenAI-compatible providers,
  simultaneous multi-backend models, PyTorch execution, and real operation
  dispatch will be introduced in subsequent patches. They are **not** claimed
  to work through the new `Model` API yet.
* The current model facade is intentionally a tiny adapter. It does not own
  separate optimizer/KV state and introduces no new forward/backward code.

## Framework-neutral and shared operation contracts (implemented)

- `model_contracts.py` defines immutable `ArchitectureSpec`, `OperatorSpec`,
  `Capabilities`, `ModelInfo`, `ExecutionReport` and `GenerationResult`.
- `model_registry.py` registers architecture *configuration factories*;
  `model_b_348m_sparse16` resolves to the **actual 348,261,376-parameter
  Model B baseline specification**, including stride-16 historical retrieval.
  The registry does not allocate its 348M weights merely to inspect the spec.
- `model_execution.py` defines independent protocols for bound engines,
  generation and training, plus a `KernelRegistry` describing reusable
  operation implementations by semantic operation, backend and mode.
  Registry entries are *descriptors*, not compiled kernel handles and not
  proof of optimization. Only verified entries qualify for fast path selection
  once full dispatcher and profiler are implemented.
- The model's parameter schema is read from existing canonical `Parameter`
  names and shapes; neither its ordering nor its checkpoint format is changed.

## Invariants to preserve through later patches

Model B must keep 20 layers, hidden size 1024, FFN 4096, its 128-token
completed-block pooling with four learned summaries, 512-token local window,
4096-token long-local head, and stride-16 sparse original-token attention
with its existing 16-layer phase assignment. No new kernel or model identity
may replace this production baseline with the non-strided Model B variant.

A future CuPy/PyTorch implementation must share this mathematical architecture
and parameter schema while remaining free to use different backward engines,
GPU kernels and tensor layouts. Fused kernels can cover several semantic
operators. Training optimizer states and inference KV/history caches remain
outside the persistent architecture object and are isolated per run/session.

## Validate

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q tests/test_model_api_0103_0.py
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_sparse_stride_history_0102z.py \
  -k test_architecture_separate_no_new_parameters_and_all_phases
```

The first test exercises a tiny actual decoder model, checkpoint loading,
capabilities, strict acceleration reporting, registry validation, and a mock
of the **existing** generator. The second asserts the sparse historical head
is preserved in the core Model B configuration. GPU benchmarks are intentionally
not run while the 348M baseline is training.

**Next:** 0103A numeric/runtime regressions, 0103B transactional checkpoints,
then 0103C/0103D full provider loading and verified fast execution selection.
