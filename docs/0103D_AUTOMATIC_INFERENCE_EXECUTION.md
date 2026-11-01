# 0103D — Inference execution policies and observed CUDA Graph reporting

## Motivation

The production Model B (20 layers, width 1024, FFN 4096, matrix pooling, 128-token blocks with four summaries, stride-16 sparse historical head) already has a narrowly specialized native CUDA Graph decoder. The old paths required environment variables to activate it, and performance benchmarks could silently run slower fallbacks without reporting which implementation actually executed.

This patch adds **one model-aware policy** to the existing library loading and generation paths. It does not introduce new CUDA kernels, change attention mathematics, or change training defaults.

## Public execution policies

| Policy | CPU | GPU / supported Model B | Other GPU architectures |
| --- | --- | --- | --- |
| `auto` | NumPy reference | Fast kernel profile + request native CUDA Graph | Existing fast kernel profile; graph not assumed |
| `reference` | Reference (with clean process environment) | No fast defaults applied; conflicting global graph option rejected | No fast defaults applied |
| `balanced` | Reference | Alias of current GPU fast profile | Alias of current GPU fast profile |
| `fast` | Explicitly rejected | Fast profile + request native CUDA Graph, with observed fallback reporting | Existing fast profile; graph not assumed |
| `strict-fast` | Explicitly rejected | Requires supported Model B and actually exercised CUDA Graph | Explicitly rejected until verified acceleration exists |

`strict-fast` means **observed CUDA Graph execution**, not a particular tokens/sec threshold or proof of bitwise numerical parity. Correctness and performance benchmarks remain independent gates. Inference paths for other models can still be accelerated; until their execution is observed and reported they are not labeled `strict-fast`.

## Usage

Native public API:

```python
from mini_llm import Model
model = Model.load("./checkpoints/model_b", execution_policy="auto")
result = model.generate("Hello", max_new_tokens=128)
print(result.execution)
# A strict benchmark can require a captured-and-exercised graph:
result = model.generate("Hello", max_new_tokens=128, require_fast=True)
```

Interactive CLI:

```bash
MINI_LLM_BACKEND=cupy python inference.py --checkpoint ./checkpoints/model_b \
  --prompt "Hello" --max-new-tokens 64 --execution-policy auto
```

Generation benchmarks (existing sampling and scoring unchanged):

```bash
MINI_LLM_BACKEND=cupy python benchmarks/evaluate_generation.py \
  --checkpoint ./checkpoints/model_b --execution-policy strict-fast \
  --output-dir /tmp/model-b-generation
```

The benchmark records an `execution` object per generation containing the observed backend, active graph implementation, and fallback reasons. `strict-fast` raises before persisting a false success when acceleration was not observed. The existing `--runtime-profile` option remains available for legacy comparisons.

HTTP serving JSON accepts an optional per-model `execution_policy` with the same names. The server's generation state now has `execution_report` after prefill. On strict-fast models, a failed capture raises instead of serving through an unreported slow path. Actual graph activation is tracked on the state rather than inferred from configuration.

## What is actually verified

- The configured policy is exposed separately from runtime graph observation.
- A graph attached to a generation state is **not** considered executed until an instrumented graph call has completed. In the specialized Model B implementation, its graph-based prefill counts as execution. Ordinary native/external graphs captured after prefill require a decode call first.
- Explicit `MINI_LLM_NATIVE_CUDA_GRAPH_DECODE=0` is respected. `strict-fast` rejects the disabled graph for supported Model B.
- On unsupported architectures, fast keeps previously validated kernels where available; no unsupported graph is automatically assumed or falsely reported.
- Tests use a CPU-hosted synthetic graph object and real tiny-model generation to verify reporting and rejection; production GPU capture and throughput **must still be tested on the RTX 5080**.

## Important limitations / next steps

1. **Execution flags are still process-global environment variables.** `setdefault` values are sticky across models in a single server process. Different policies should run in separate processes until 0103E introduces session-local dispatch; this patch does not claim per-model kernel isolation.
2. Not every GPU backend or model has a verified graph. In particular, the internal GGUF/Safetensors loader currently supports only compatible Llama-family formats; this patch does not implement new model families or remote execution.
3. CUDA Graph capability is constrained to the exact Model B geometry and BF16. The graph constructor remains the authoritative last check for memory, device, weights and layout. Capture may fail on a particular workload; `fast` reports the fallback, `strict-fast` rejects it.
4. `reference` selects no performance defaults; if another call has already set fast flags in the same process, they cannot safely be reset while other models are running. A fresh process is required for a trustworthy CPU/reference comparison.
5. The server's `execution_report` is stored on each request state, not in a global report. Exposing it through HTTP diagnostics is deferred to 0103F.
6. Graph capture and reuse can have significant costs; first-token latency, long-prefill behavior, and decode throughput must be measured separately. This patch does not assert that graph decoding is always fastest.
7. Training runtime paths and offloading choices are intentionally unchanged. PyTorch remains a future backend.

## Validation

CPU regression: `python scripts/run_0103a_regressions.py --mode cpu`.
GPU regression: `python scripts/run_0103a_regressions.py --mode gpu` on idle GPU, then check captured decode in interactive CLI and strict-fast benchmark. Unmeasured GPU performance must not be advertised as validated.
