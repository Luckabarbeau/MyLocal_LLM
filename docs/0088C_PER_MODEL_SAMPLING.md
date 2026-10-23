# 0088C — Per-model sampling and bounded repetition penalties

This incremental patch is intended to apply **after 0088A and 0088B**.

## Registry

Each entry in `serving_models.example.json` (or whichever registry is passed
using `serve.py --registry`) may include a `sampling` object:

```json
{
  "id": "native-338m",
  "format": "native",
  "path": "./checkpoints/YOUR_CHECKPOINT",
  "chat_template": "plain",
  "tool_format": "none",
  "context_length": 64000,
  "sampling": {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "repetition_window": 256,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

The `sampling` block is optional. Absent settings preserve existing server
behavior: greedy (`temperature=0`), `top_p=1`, unrestricted `top_k`, and
no repetition/frequency/presence penalties. Other models may have their own
sampling blocks; the current resident model uses its own settings.

Incoming API requests can override **individual** defaults using the same
fields at the top level (OpenAI style). Omitted fields, including explicit
JSON `null`, inherit the per-model default. Thus, if a Zed client sends a
`temperature` value, it overrides that value from the registry. If Zed is
always submitting its own temperature, configure its model-specific parameter
as well or change the client setting.

## Sampling semantics

- `temperature = 0`: greedy after penalties. Positive temperatures enable
  sampling with top-k and top-p filtering.
- `repetition_penalty`: sign-aware adjustment of logits for tokens present in
  the recent-token buffer. `1.0` disables it. Positive logits are divided by
  the penalty and nonpositive logits multiplied by it (for positive penalties).
  Every distinct repeated token is adjusted once, regardless of its frequency.
- `repetition_window`: history length of the last N **prompt + generated**
  tokens, maintained in a bounded deque and Counter. `0` disables all three
  token-history penalties. No 64K token scan or long-context GPU mask is
  created at each decoding step.
- `frequency_penalty`: subtract coefficient × occurrence count in window.
- `presence_penalty`: subtract coefficient once if token appears in window.
- Order: repetition, frequency, presence, temperature, top-k/top-p.
- When penalization is enabled, only distinct recent token IDs and counts
  are sent to the inference backend (NumPy/CuPy). Input logits are not mutated.

The model's chat template, context horizon and KV cache are **unchanged**.
Repetition controls may reduce short loops, but cannot correct a broken
inference/training parity or turn a base pretrained model into an instruction
model. The historical-read cache refresh behavior remains as in 0088B.

## Apply and run

```bash
git apply --check PATCH_0088C_PER_MODEL_SAMPLING.patch
git apply PATCH_0088C_PER_MODEL_SAMPLING.patch
MINI_LLM_BACKEND=cupy python serve.py --registry ./serving_models.example.json --host 127.0.0.1 --port 8080
```

Edit your actual registry to include `sampling` for each model; the patch does
not replace a potentially customized local registry. Restart `serve.py` to
reload configuration. No changes in Zed are necessary if it uses this server.

### Tests

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_0088a_openai_server.py tests/test_0088c_serving_sampling.py
```

A CPU-only test checks each setting, registry validation, per-model override
precedence, bounded window updates, sign-aware greedy repetition and live HTTP.
CuPy GPU performance and model quality still need direct measurement.
