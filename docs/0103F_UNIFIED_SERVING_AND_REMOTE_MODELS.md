# 0103F — Unified serving and remote OpenAI-compatible models

## Summary

One `serve.py` instance can expose a single `/v1/models` catalogue containing
native MyLocal_LLM checkpoints, internally supported GGUF/Safetensors models,
and models already hosted by a separate OpenAI-compatible HTTP server (e.g.
llama.cpp `llama-server`). Clients use the same local `/v1/chat/completions`
and `/v1/completions` endpoints and the local registry model ID.

This patch reuses the 0103E shared `GenerationSession` for **in-process** models.
A remote provider cannot provide token logits, KV cache, training or a native
in-process session, so `RemoteCompletionEngine` implements the same **HTTP
completion contract**, without duplicating decoding or tokenization. The remote
provider handles its own chat templates, tools, tokenization and generation.

## Mixed registry

The existing local model entries are unchanged. Add another entry:

```json
{
  "id": "lfm-remote",
  "format": "remote",
  "api_url": "http://127.0.0.1:18080/v1",
  "remote_model": "lfm3",
  "api_key_env": "LOCAL_LFM_API_KEY",
  "chat_template": "chatml",
  "tool_format": "native",
  "context_length": 32768
}
```

`api_url` points to the **already-running** upstream API root, typically
ending in `/v1` (do not put `/chat/completions` in it). `remote_model` is the
model ID recognized by the upstream provider. `api_key_env` is optional;
when present the environment variable must be set in the **MyLocal_LLM
server** process. Do not place API keys in JSON model registries or URLs.
For a local unauthenticated llama.cpp process, omit `api_key_env`.

Set `tool_format` to `native` for remote servers that implement OpenAI-style
function/tool calling. For native in-process models, `json_envelope` remains
supported, using 0103E's existing conservative tool-call parsing.

Start the upstream provider separately, and then:

```bash
MINI_LLM_BACKEND=cupy python serve.py --registry ./serving_models.example.json \
    --host 127.0.0.1 --port 8080
```

The local process exposes `GET /v1/models`, `POST /v1/completions` and
`POST /v1/chat/completions`. Zed and OpenAI-compatible coding agents may
connect to `http://127.0.0.1:8080/v1`, using the public ID `lfm-remote`.
No change is required in the client when switching between a native model
and a remote one, although not all models support the same tool protocol.

## Execution and resource isolation

- In-process native and supported pretrained models use **the same existing
  shared session/token sampler** and a single active GPU-loaded model at a time.
- Remote engines are lazily constructed and cached independently. They neither
  load local model weights nor touch the process-global CuPy execution flags.
- A remote call never evicts an already resident local model. Remote requests
  do not hold the GPU generation lock and may execute concurrently.
- Upstream `model` fields are mapped to/from the public registry ID. Native
  upstream `tool_calls`, `tool_choice`, streaming deltas, prompt multimodal
  data (when accepted upstream), and usage fields are forwarded unchanged.
- Credential forwarding is **one way**: the local server's incoming Bearer key
  is not sent upstream. The provider key comes only from `api_key_env`.
- The transport rejects redirects to avoid forwarding provider credentials to
  a different URL, and it limits response size. Upstream HTTP errors, missing
  credentials and invalid JSON become explicit HTTP errors.

For remote models, the local server forwards per-request sampling parameters
as supplied. It does not synthesize nonstandard `top_k`, repetition window or
other defaults that the remote provider may not support. The upstream server
owns its own generation defaults. Registry `sampling` defaults continue to
apply to local in-process models.

## Limitations and intentionally deferred work

- This patch **connects to** an existing provider; it does not download models
  or launch/manage llama.cpp processes.
- Only OpenAI-compatible chat and text completion endpoints are supported.
  `/v1/responses`, embeddings, vision execution for local native models,
  reasoning annotations, and other provider-specific routes are not proxied.
- A remote model cannot expose local model weights, low-level logits, training,
  or native `GenerationSession` states through this adapter.
- Fully independent per-model CuPy optimization flags remain deferred; local
  in-process GPU models are still switched under the manager lock.
- Remote provider throughput and correctness are delegated upstream; 0103D
  local CUDA Graph activation reports do not apply to remote providers.
- There is no claim of live-model GPU performance validation. The 0103A2 GPU
  numerical gate must still be run on the target RTX 5080.

## Testing

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
    tests/test_remote_serving_0103f.py \
    tests/test_0088a_openai_server.py \
    tests/test_0088c_serving_sampling.py

python scripts/run_0103a_regressions.py --mode cpu \
    --report /tmp/model_b_cpu_regression.json
```

The new tests launch two real loopback HTTP servers and validate chat and text
completion forwarding, native tool calls in streamed and non-streamed responses,
public model aliases, independent authentication, upstream errors, registry
validation, and preservation of the local resident model.

**No Model B architecture, kernels, checkpoints, training or optimizer state
were changed.**
