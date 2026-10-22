# 0088A — Native model-serving foundation (experimental)

This patch adds an OpenAI-style HTTP frontend to MyLocal_LLM's **existing**
KV-cached NumPy/CuPy inference engine. It does not invoke llama.cpp or PyTorch
and does not change training, checkpoint data, or attention kernels.

## Supported model backends

- `gguf`: **Llama architecture only**, F32/F16/Q8_0 source tensors. Q8_0 is
  **dequantized** to runtime FP16/FP32; neither Q4_K nor native low-bit GEMM
  is implemented. Embedded GGUF tokenizer or adjacent `tokenizer.json`.
- `safetensors`: strict Llama-compatible HF format and `tokenizer.json`.
- `native`: native autoregressive checkpoints with `config.json`, `model_params/`
  and `tokenizer.json` (or parent tokenizer). Loads weights via the established
  benchmark loader and shares references with `InferenceModel`; no optimizer
  state is loaded. The historical-read mode is limited to its existing KV
  inference horizon, currently the working window.
- Block-proposal/refinement inference is **not** wired into this serving adapter.
  Future formats can use an additional adapter without changing HTTP routes.

The example registry is `serving_models.example.json`. Change its paths and
contexts to locally verified files; JSON paths are relative to the registry,
not the terminal working directory. `format: "auto"` can recognize GGUF files,
HF Safetensors model folders, and native checkpoint folders. Each model ID is
stable and user-chosen. The registry is **explicit**; it does not scan remote
sources, automatically download weights, or imply that every GGUF architecture
is supported. A **single model** is resident on GPU at a time. Switching
between models discards the old resident model and releases the CuPy pool.
Requests are serialized across model loading and generation on one GPU.

## Run

From the repository root, with dependencies for external formats installed:

```bash
python -m pip install -e '.[external]'
MINI_LLM_BACKEND=cupy python serve.py \
    --registry ./serving_models.example.json \
    --host 127.0.0.1 --port 8000
```

Verify basic chat (replace the model ID by a valid registered model):

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"smollm2-360m","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Verify streaming:

```bash
curl -Ns http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"smollm2-360m","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

Also exposed: `/health`, `/v1/completions` (basic text prompt).

## Zed setting

Set your custom Zed OpenAI-compatible provider to the same base URL.
The configuration below is a **protocol-only** sample. Enable tools only
for model IDs verified to produce tool calls, and specify a valid context.

```json
{
  "language_models": {
    "openai_compatible": {
      "MyLocalLLM": {
        "api_url": "http://127.0.0.1:8000/v1",
        "available_models": [
          {
            "name": "smollm2-360m",
            "max_tokens": 4096,
            "max_output_tokens": 512,
            "capabilities": {
              "tools": false,
              "images": false,
              "parallel_tool_calls": false,
              "chat_completions": true,
              "max_tokens_parameter": true
            }
          }
        ]
      }
    }
  }
}
```

The server accepts both `max_tokens` and `max_completion_tokens`. Do not set
`capabilities.tools` true solely because a GGUF or Safetensors checkpoint
loads; tool reliability is a model/tokenizer/template property, not a server
endpoint property.

## Tool calls — experimental `json_envelope`

A registry entry may opt into `"tool_format": "json_envelope"` to exercise
an **end-to-end protocol**: incoming function tools are placed into an explicit
system instruction and the output must be exactly:

```json
{"tool_calls":[{"name":"read_file","arguments":{"path":"main.py"}}]}
```

Only known tool names and JSON argument objects are accepted. These are returned
as standard OpenAI `tool_calls` with generated IDs; **the client (Zed) executes
them** and sends subsequent `role: tool` messages. The server never directly
executes arbitrary shell or filesystem tools. Streaming tool requests are
currently **buffered until full model generation** to allow parsing; SSE then
reports the function delta and `finish_reason: "tool_calls"`. Normal text
responses stream token-by-token. No constrained decoding or grammar engine
exists in this patch, so small models may not follow the JSON instruction.
This is not yet proof of usable agentic coding with SmolLM2.

`chat_template` is explicitly `chatml` or `plain`. The generic ChatML format
matches the initial SmolLM2 experiment; other model-specific chat templates,
Jinja templates, proprietary tool control tokens, special-token boundaries,
multimodal messages, parallel tool syntax, and long-context truncation are not
silently approximated. Prompt+generation must fit the configured limit.

## Tests and acceptance

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q tests/test_0088a_openai_server.py
MINI_LLM_BACKEND=numpy python -m pytest -q tests/test_0085f_external_gguf.py tests/test_0085g_gguf_tokenizer.py
```

The CPU protocol tests include a deterministic fake inference engine and an
end-to-end **real tiny native checkpoint** load followed by an HTTP request.
External-loader tests cover offline format/tokenizer behavior. For deployment,
first test real local FP16/Q8 SmolLM2 using `benchmarks.try_external_gguf` and
its tokenizer-parity check, then start the server and measure prefill, decode,
first-byte streaming latency, and valid Zed tool calls. **Downloaded model
weights and RTX 5080 operation were not available during patch authoring.**

Known further work: native inference-only checkpoint deserialization (avoid
constructing a full training model during load), per-architecture chat templates
and tool parsers, streaming tool argument validation, true Q4_K/Q6_K kernels,
CPU/GPU offload and continuous batching, request cancellation, robust SSE error
handling, JSON-schema-constrained decoding, `/v1/embeddings` or `/v1/responses`
if clients require them, high-context production validation, and optional Lab
watchdog/knowledge-proxy compatibility tests.

**Security:** default bind is localhost. Binding to a network host without an
API key is refused. Optional `--api-key` (or environment
`MYLOCAL_LLM_API_KEY`) provides a bearer token; use TLS/reverse proxy for
nonlocal access. This prototype does not execute tool calls itself.
