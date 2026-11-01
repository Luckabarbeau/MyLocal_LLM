# 0103E — Unified generation sessions (2026-10-30)

## Scope

This is a **control-flow refactor, not an architecture/kernel change**. It
introduces `mini_llm.generation` as the single shared implementation of token
sampling, bounded repetition penalties, text stop handling, streamed content
and per-request generation state. The following entry points now use the same
core:

- Native interactive/one-shot generation via `TextGenerator` (CLI and benchmarks).
- Public `Model.generate()` (delegates through `TextGenerator`).
- Public `Model.session()` for explicit request-scoped generation.
- In-process HTTP streaming and non-streaming chat/completions for native and
  supported local pretrained models through `ServingModel`.

The existing `sample_token` export in `serving_api` and legacy CLI sampling
helper names remain for compatibility. Their numerical sampling algorithm is
implemented once in `generation.sample_token`. No new CUDA kernels are added.

## Simple usage

```python
from mini_llm import Model

model = Model.load('checkpoints/model_348m', execution_policy='auto')
session = model.session()
result = session.generate('hello', max_tokens=64, temperature=0)
print(result.text, result.token_ids)
for event in session.stream('hello', max_tokens=16, temperature=0):
    print(event)  # prefill / content / done
```

`model.generate(prompt, max_new_tokens=...)` remains compatible with the
previous high-level API. `TextGenerator.generate(...)` retains its tuple
`(text, timing)` result and `return_token_ids=True` convention. The CLI
reference (`--backend reference`) retains a full-prefix numerical path for
comparison, but the default KV-cache path shares sampling/session behavior.

## Ownership and memory

The model retains shared parameters; a `GenerationSession` owns its current
request state, generated token IDs, sampling window and metrics. `ServingModel`
allocates request-local state. The native `TextGenerator` retains its
`InferenceModel` and, where supported, its captured CUDA Graph, and resets the
underlying generation state between requests. This avoids re-capturing graphs
unnecessarily.

**`Model.session()` does not yet mean persistent multi-turn KV continuation.**
Each call to `session.generate` or `session.stream` starts a new prompt/request.
This is intentionally reflected by `capabilities.persistent_sessions=False`.
Do not use the same `GenerationSession` concurrently from multiple threads.
The server serializes access through the existing model manager lock.

## Sampling, EOS and stopping

- One sampler for top-k, top-p, temperature, and greedy decoding.
- Request-local recent-token frequency/presence/repetition penalties.
- Backend logits are not overwritten by sampling penalties.
- Streaming suppresses incomplete UTF-8 and withheld stop-string prefixes;
  withheld text is flushed when EOS is reached without a matching stop string.
- Server stops omit the EOS token from its reported count; native CLI metrics
  continue to include it, preserving the old CLI accounting convention.
- One-shot native generation **and non-streamed HTTP responses** without stop
  strings decode generated IDs only once, avoiding repeated whole-output
  decoding on every token (important for speed). SSE streaming, or any request
  with explicit text-stop strings, still decodes incrementally as required.

**Stochastic compatibility:** top-k/top-p and temperature selection is now
shared with the server sampler. With a fixed seed, generated token streams can
change compared with earlier CLI versions. Quality benchmark reports should
record patch version and re-establish stochastic baselines. Greedy decoding
is unchanged in intent. No claim of exact old seeded stochastic sequences.

## Boundaries and future work

- No changes to training, Model B block compression, stride-16 sparse history,
  numerical forward/backward passes, or GPU kernel selection.
- Runtime optimization settings still involve process-global environment
  variables. Fully per-instance backend settings are **not** implemented here.
- True append-only conversational KV continuation (including document resets,
  block-compression boundaries and sparse historical states) remains a separate
  future feature; never simulate this by silently reusing unrelated caches.
- Remote OpenAI-compatible providers are planned for 0103F.
- Backend-specific GPU numerical parity and throughput tests are still required
  on the actual target GPU. The earlier 0103A2 GPU gate is a separate open
  validation requirement.

## CPU regression gate

```bash
python scripts/run_0103a_regressions.py --mode cpu \
  --report /tmp/model_b_cpu_regression.json
```

The gate now includes session, sampling, HTTP, and benchmark tests as a
separate suite. For GPU checks, run the existing `--mode gpu` gate when the
GPU is not occupied by training.
