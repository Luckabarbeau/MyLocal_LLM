# Patch 0088B — Native `historical_read` inference for the OpenAI server

**Applies after:** PATCH_0088A_NATIVE_OPENAI_SERVER.patch. No changes to GGUF / Safetensors inference.

## What changed

- Native `historical_read` checkpoints now advertise the checkpoint's **full trained memory horizon** (normally 65,536), rather than being silently capped at the 4,096-token working window.
- Historical checkpoint weights are shared into the inference attention layers. Historical block/sparse/learned heads replace designated current-window heads for **prefill and decode**; original head gates, learned K/V projections and fixed sweep phases are preserved.
- Long prompts are split into the last 4,096-token working window and the preceding right-aligned historical source (up to 61,440 tokens). History is encoded with the trained 32-token block encoder; invalid left padding is masked and cropped before expensive block encoding. **No 65K deep-transformer KV cache or giant 65K masks** are allocated.
- Decode uses a 4K ring KV cache. Evicted working tokens accumulate in the external token store. Learned 128-token routes are cached for the decode path and invalidated when history is rebuilt.
- On a history refresh, **both** the historical read index and the current working-window Transformer cache are rebuilt using the checkpoint's original RoPE coordinates. This rebase is needed for parity; exchanging historical K/V without recomputing the downstream 4K current hidden states would not be equivalent.
- `MINI_LLM_HISTORICAL_REFRESH_TOKENS` controls refresh frequency (default 32; `1` for correctness-first sliding-window parity). First crossing of a 4K window refreshes immediately.

## Correctness and limits

**Verified on CPU:** full-window and right-aligned partial-history prefill logits agree with the training forward path; refresh-at-every-token rebase agrees with a shifted training window. Tests cover learned route reuse, working cache shape and prior API regressions.

**Not yet verified on RTX 5080/CuPy:** 64K real-checkpoint prefill, VRAM peak, tokens/s, BF16 numeric parity, or Zed end-to-end agent performance.

**Important approximation:** with refresh interval >1, tokens evicted since the previous refresh are not visible to the historical reader until the next refresh; cached current hidden states between refreshes also reflect the previous historical read state. Learned route refresh within that interval is approximate, especially around 128-token query boundaries. For exact sliding-window comparison, set refresh=1, which can be **very expensive** because the entire current window and encoded history are recomputed for every generated token. Refresh 32 is a practical experiment, not a parity claim or validated performance optimum.

**Not implemented:** optimized *incremental* historical block K/V updates with exact per-token parity, streaming shared-memory concurrent requests, actual multi-user KV scheduling. This patch supports the `historical_read` mode, not `routed_prefix` or `terminal_landmark` changes.

## Apply

From the repository root (with 0088A already applied):

```bash
git apply --check PATCH_0088B_HISTORICAL_READ_INFERENCE.patch
git apply PATCH_0088B_HISTORICAL_READ_INFERENCE.patch
```

Set the **native checkpoint directory** in `serving_models.example.json` (not the placeholder) and select e.g.:

```json
{"id":"native-338m", "format":"native", "path":"./checkpoints/<your-real-checkpoint>",
 "chat_template":"plain", "tool_format":"none", "context_length":64000}
```

`path` must point to a valid trained checkpoint. In Zed, set the native model's `max_tokens` to `64000` and `supports_tools` to `false` until tool calling has been validated. The configured server context is **prompt + completion**, so leave room for generation.

Restart your server:

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_HISTORICAL_REFRESH_TOKENS=32 python serve.py \
  --registry ./serving_models.example.json --host 127.0.0.1 --port 8080
```

`/v1/models` should show `native-338m` with context 64000 if the registry entry is set accordingly. Loading is lazy: actual weight validation happens on the first chat request.

First try a tiny 16-token completion, then Zed. An initial 12K-token coding-agent prompt should now be accepted if prompt + requested completion fits within the server's configured context and the checkpoint actually uses historical-read.

## Tests

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_0083b_historical_read.py \
  tests/test_sparse_kv_inference.py \
  tests/test_inference_rope_base.py \
  tests/test_0088a_openai_server.py
```

On the RTX workstation, separately measure: 4K/8K/16K/32K/60K prefill time, tokens/s over 128 tokens, VRAM peak, and correctness for refresh intervals `1` and `32`. Keep the original 4K implementation as a performance baseline. Do not enable `supports_tools` merely because context now works.
