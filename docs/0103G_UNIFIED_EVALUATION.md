# Patch 0103G — Unified evaluation and benchmarking (incremental)

## Purpose

Move benchmark **execution** onto the same session and model-loading abstractions
used by interactive inference and serving. Do not replace verified specialist
scoring algorithms with unvalidated generic substitutes.

## Entry point

`python -m benchmarks.run_unified` uses the model catalog introduced in 0103C,
the shared generation loop in 0103E, and the external provider adapter in 0103F.

### Native Model B / local checkpoints

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.run_unified \
  --checkpoint ./checkpoints/model_b_348m \
  --prompt-ids math_general_short,code_python_short \
  --tokens 128 --seeds 42,123 \
  --output-dir ./benchmarks/results/model_b_348m_0103g
```

### Registered native, SmolLM2 or remote model

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.run_unified \
  --registry ./serving_models.json --model smollm2-360m \
  --tokens 128 --output-dir ./benchmarks/results/smollm2_0103g
```

For an OpenAI-compatible external model, text completions are the default for
**document-continuation prompts**. Use `--remote-mode chat` only for intentionally
chat-formatted evaluation. Upstreams may not implement text completions; this
fails explicitly, rather than silently comparing chat against raw continuation.
External models cannot be assumed to support `top_k` or exact seed determinism.

### Existing quality benchmarks, including WikiText / code / retrieval

Existing evaluation is preserved and can be launched from the same entry point:

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.run_unified \
  --checkpoint ./checkpoints/model_b_348m \
  --legacy-suite benchmarks/configs/foundation_capabilities_0093.json \
  --benchmark-ids wikitext_1024,wikitext_4096,generation_quick \
  --output-dir ./benchmarks/results/model_b_live_0103g
```

This is an adapter to `benchmarks.run_checkpoint_suite.run_checkpoints` and the
existing `LiveBenchmarkScheduler`, not a rewrite of WikiText scoring, code
sandboxes or retrieval metrics. Its old result files retain their formats. A
`unified_legacy_index.json` sidecar records the source and suite hash.
`--resume` only covers the new JSONL generation suite; the existing scheduler
retains its own repeat/skip behavior.

## Standard new results

- `manifest.json`: model source metadata identity, prompt suite SHA256,
  selected prompt IDs, RNG seeds, max tokens, sampling, model execution policy,
  and source/backend declarations.
- `results.jsonl`: one durable row per `(prompt_id,seed)`; sample text,
  locally observed token IDs (when available), execution report, latency,
  token usage provenance, and token-level repetition metrics when possible.
- `summary.json`: cases complete/expected and domain-level mean end-to-end
  tokens/sec; **not** a kernel throughput claim.

Results are append-only and may be resumed with `--resume`. The resume guard
rejects changes in suite bytes, checkpoint file sizes/mtime, context, policy,
seed, or sampling. Source metadata is **not a cryptographic hash of weights**;
independent checkpoint integrity checks remain necessary. For remote models,
upstream weight revisions cannot be independently detected.

The CLI holds an advisory POSIX file lock to prevent two benchmark jobs from
writing the same output simultaneously.

## Measurement semantics

- Local: time includes tokenization, sampling, host overhead, and decoding.
  Prefill is measured separately with device synchronization. Decode time is
  derived from total wall time less prefill, *not* isolated CUDA-kernel time.
- Remote: wall time includes HTTP transport and provider execution. Prefill,
  per-token decode rate and exact token IDs are unavailable. Provider token
  counts are explicitly marked `provider_reported`; absent usage stays `null`.
- Fast path: execution reports reflect observed CUDA Graph usage, not requested
  environment flags. `strict-fast` is only available where verified native
  GPU acceleration is supported; remote models cannot satisfy it.
- Cross-model evaluation: identical raw prompts are necessary but not
  sufficient. Tokenizers, context capabilities and service prompt templates
  differ. Compare quality and throughput accordingly.
- Generation repetition metrics are not equivalent to code correctness,
  factuality or narrative coherence. Keep the existing specialist scorers.

## Scope and future work

- Does not replace `live_suite`, `run_checkpoint_suite`, `capability_suite`,
  reference continuation or WikiText evaluators. The CLI provides a shared
  entrypoint; deeper migration is subsequent work.
- Does not change model mathematics, sampling, tokenizer, existing benchmark
  definitions or CUDA kernels.
- Neither per-model GPU runtime isolation nor direct PyTorch execution is
  implemented by 0103G.
- Long-context and GPU decode throughput validation still require a free GPU
  on the target machine; CPU tests are not a performance proof.
