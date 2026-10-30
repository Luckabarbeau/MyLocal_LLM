# 0103C — Unified model loading and model source registration

**Baseline:** Model B, 348,261,376 parameters, completed-block matrix pooling with the stride-16 sparse historical head. This patch does not change architecture, kernels, optimizer, checkpoint format, generation mathematics, or training. It is additive/modular infrastructure for the agreed model-centric design.

## Canonical loading boundary

`mini_llm.model_loading` is the sole new library location for loading checkpoint configuration, tokenizer, canonical parameter arrays, and pretrained Llama-family GGUF/Safetensors weights.

- `load_native_checkpoint(path, requested_context=None, runtime_profile=None, apply_profile=False)` returns a `LoadedNative` record containing the model, tokenizer, resolved context, checkpoint training metadata, tokenizer path, and runtime-profile result. Missing/shape-incompatible tensors are **errors**, never silently retained from initialization. Progressive depth is restored from the checkpoint state. The selected checkpoint reader is the same versioned/legacy-compatible 0103B implementation.
- `resolve_native_context(config, requested)` defines one context policy. Memory-routing models default to their addressable history horizon and reject contexts beyond it; nonmemory architectures default to their architectural context, while explicit longer cache requests remain available for the existing inference implementation.
- `load_pretrained_weights(path, format=..., dtype=...)` reuses existing in-process **Llama-family** GGUF and Safetensors loaders, without recompiling a new engine or duplicating model kernels. It does not imply arbitrary GGUF architectures are supported.
- `mini_llm.model_sources.ModelSpec`, `SamplingConfig`, `ModelCatalog` and `load_specs` define model source configuration and a JSON catalogue independently of the HTTP server. Existing imports from `mini_llm.serving_models` remain valid compatibility aliases. Model IDs, tool templates, sampling defaults, and relative path resolution are preserved.
- Remote OpenAI-compatible model *descriptors* can be included in the catalogue (`format: remote`, `api_url`, `remote_model`, optional `api_key_env`). Actual remote generation/proxying is **not** implemented in 0103C; it fails explicitly until 0103F. The registry never embeds an API key or makes network calls.

## Current entry-point mappings

| Entry point | Loading | Generation |
|---|---|---|
| `Model.load` public facade | canonical `load_native_checkpoint` | existing `TextGenerator` via `legacy_generation` |
| `inference.py` CLI | canonical `load_native_checkpoint` | same `TextGenerator`, including its reference and KV-cache modes |
| `benchmarks.evaluate_generation.load_generator` | canonical `load_native_checkpoint` | compatibility wrapper around `TextGenerator` |
| `mini_llm.serving_models.load_model` | canonical native or pretrained loader | existing in-process `ServingModel` engine bridge |

`TextGenerator` and legacy sampling functions were moved **without mathematical changes** into `mini_llm.legacy_generation`. `inference.py` reexports identical objects so existing imports/tests still work. The experimental block-proposal adapter now imports that library module instead of the CLI. No module within `mini_llm` imports `benchmarks` or the top-level `inference.py`.

### Sample usage

```python
from mini_llm import Model
from mini_llm.model_sources import ModelCatalog
from mini_llm.model_loading import load_native_checkpoint

model = Model.load("./checkpoints/model_b_348m", backend="auto")
print(model.inspect())

sources = ModelCatalog.from_manifest("./serving_models.json")
print(sources.names())

# Native compatibility loading without GPU allocation for optimizer state:
bundle = load_native_checkpoint("./checkpoints/model_b_348m")
print(bundle.context_length, bundle.training_state.get("step"))
```

For an existing local HTTP service, the model registry and serving CLI are unchanged. Native serving now uses the shared loader and no longer depends on the benchmark package. External GGUF and Safetensors serving uses the same shared pretrained loader. No in-process LFM3 GGUF support is claimed (unsupported architectures require a separate provider such as llama.cpp).

### Runtime profile expectations

`apply_profile=False` is the low-level loader default and preserves 0103-0 facade behavior. The CLI, benchmark and server opt into runtime-profile initialization **before** model construction, matching their existing intended fast-profile behavior. 0103C does **not** certify which GPU fast paths execute. Verified fast-path selection and strict diagnostics remain the scope of **0103D**; a fast-path benchmark must not infer activation from environment flags alone.

### Validation

CPU smoke tests including the production CLI, model façade, benchmark and HTTP serving adapters, incomplete checkpoint failure, provider registration, external-loader delegation, existing external Llama/GGUF/Safetensors tests, and the 0103A CPU regression gate:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
    tests/test_canonical_model_loading_0103c.py \
    tests/test_model_api_0103_0.py \
    tests/test_0088a_openai_server.py \
    tests/test_0088c_serving_sampling.py \
    tests/test_generation_benchmark.py \
    tests/test_0085b_external_llama.py \
    tests/test_0085c_external_safetensors.py \
    tests/test_0085f_external_gguf.py \
    tests/test_0085g_gguf_tokenizer.py

python scripts/run_0103a_regressions.py --mode cpu \
    --report /tmp/model_b_cpu_regression.json
```

No GPU regression, full-scale native model load, or pretrained SmolLM2 GGUF load was executed in the patch-development environment. Previous 0103A GPU verification remains separate and pending on the RTX 5080.

## Remaining work

- **0103D:** explicit fast-path policies, observed execution diagnostics, CUDA Graph activation, strict performance contract.
- **0103E:** one shared token-generation session, sampler, stop logic, and streaming API for native and external models. `legacy_generation` is a compatibility step, not the long-term generation design.
- **0103F:** use the shared runtime for HTTP serving, add optional external OpenAI-compatible provider execution, keep agent compatibility.
- **0103G:** migrate remaining benchmarks/teacher-forced scorers to the public model and execution interfaces.
- **PyTorch:** remains a future backend using the same model/source contracts; this patch neither registers nor simulates a PyTorch execution backend.
