# 0085H — WikiText-2 for GGUF checkpoints

The existing `benchmarks.evaluate_wikitext` implements WikiText-2 raw-text
splitting, left-to-right sliding windows, and stabilized FP32 next-token NLL.
The additional `benchmarks.evaluate_external_wikitext` **reuses** those
functions rather than introducing a different loss definition.

GGUF model/tokenizer loading is native: only the GGUF metadata reader and
`tokenizers` are needed for file parsing and BPE; all model inference executes
through `ExternalLlamaModel` on the selected MyLocal_LLM NumPy/CuPy backend.
The evaluator does NOT read `tokenizer.json`, connect to Hugging Face, or
insert instruction/chat messages into WikiText.

Quick smoke run (BPB intentionally not reported for a token-truncated run):

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.evaluate_external_wikitext \
    --model ../external_models/smollm2-360m-instruct-q8_0.gguf \
    --data ../benchmarks/wikitext-2-raw/wiki.test.raw \
    --dtype float16 --context-length 256 --stride 256 \
    --max-tokens 2048 --progress-every 2
```

Full raw-text run (same 512-token context and stride for the native benchmark):

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.evaluate_external_wikitext \
    --model ../external_models/smollm2-360m-instruct-q8_0.gguf \
    --data ../benchmarks/wikitext-2-raw/wiki.test.raw \
    --dtype float16 --context-length 512 --stride 512 \
    --progress-every 20 \
    --output-json ./benchmarks/results/smollm2_360m_instruct_q8_0_wikitext2.json
```

Use precisely the same original text, `--document-mode`, context length and
stride in the existing native evaluation:

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.evaluate_wikitext \
    --checkpoint YOUR_NATIVE_CHECKPOINT_DIRECTORY \
    --data ../benchmarks/wikitext-2-raw/wiki.test.raw \
    --memory-mode plain --context-length 512 --stride 512 \
    --document-mode continuous
```

For `--document-mode article`, both evaluators reset context at the article
heading. For `continuous`, both concatenate all original WikiText-2 text into
one token stream. The external model uses its embedded BPE, so native-token
perplexities are not directly comparable; compare bits per original UTF-8
byte (BPB). To match the established native metric, BPB divides total NLL by
*all* original text bytes, although the first token of each stream is unscored
when no BOS is present. Treat this as the same benchmark convention, not a
strict arithmetic-code length for every single byte.

The first validation should be a small run; 32-layer full-causal inference can
be slow on the custom CuPy engine, and runtime depends on context length. The
GGUF weights are dequantized from Q8_0 and used as FP16 or FP32, **not** Q8
GPU matrix multiplications. This is an **instruction-tuned** model, whereas
for quality comparisons against native base pretraining models an untuned
SmolLM2 base checkpoint would be preferable.

Offline unit tests:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
    tests/test_0085h_gguf_wikitext.py tests/test_wikitext_routed_eval.py
```

Acceptance limitation: no local GGUF weights and no CuPy were available during
patch authoring, so the full real-model WikiText result is pending execution on
the RTX 5080. Verify embedded tokenizer versus HF first if that comparison has
not yet been run with the original `--validate-tokenizer` option.
