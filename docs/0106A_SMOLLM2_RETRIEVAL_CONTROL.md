# 0106A — SmolLM2 long-retrieval control (native context)

The standalone runner `benchmarks/evaluate_external_long_retrieval.py` scores **the same 0097 synthetic fact-retrieval task bank** and uses the **same prompt-building and 4-choice/full-vocabulary rank functions** as `benchmarks.long_retrieval_0097`. It loads an existing **supported Llama-style SmolLM2 GGUF** through the current MyLocal_LLM external CuPy inference engine, with the GGUF's own tokenizer. The Model B benchmark is not modified.

## Run on the RTX 5080, separately from Model B training

```bash
# First: compare the 32 common native-context cases (16 each at 1K, 4K).
MINI_LLM_BACKEND=cupy .venv/bin/python -m benchmarks.evaluate_external_long_retrieval \
  --model ./pretrained_models/SmolLM2-360M-Instruct-F16.gguf \
  --distances 1000,4000 \
  --repetitions 2 \
  --output-dir ./benchmarks/results/smollm2_1k_4k

# Then: request the exact 96-case task grid.
# Cases beyond the GGUF's *native* context are explicitly skipped, never
# silently truncated. Expect up to 48/96 evaluated for an 8192 context.
MINI_LLM_BACKEND=cupy .venv/bin/python -m benchmarks.evaluate_external_long_retrieval \
  --model ./pretrained_models/SmolLM2-360M-Instruct-F16.gguf \
  --distances 1000,4000,8000,16000,24000,32000 \
  --repetitions 2 \
  --output-dir ./benchmarks/results/smollm2_native_context

# Optional harder diagnostic: genuinely test sensitivity to changing the fact.
# Doubles inference work; start with 1K/4K only.
MINI_LLM_BACKEND=cupy .venv/bin/python -m benchmarks.evaluate_external_long_retrieval \
  --model ./pretrained_models/SmolLM2-360M-Instruct-F16.gguf \
  --distances 1000,4000 \
  --include-counterfactual \
  --output-dir ./benchmarks/results/smollm2_counterfactual
```

Replace the GGUF path with the model file you actually use. **Run on a free GPU**, not concurrently with the live Model B trainer. The external full-causal prefill may allocate O(T²) attention workspaces, unlike Model B's specialized long-context path.

## Outputs

- `result.json`: the same `long_retrieval` family, cases, per-distance summary schema, and explicit `unsupported_cases` with reasons.
- `retrieval_by_case.csv`: existing 0097 CSV column layout.
- `report.html`: existing 0097 report generator.

For 8K distances, the **assembled prompt** (fact + filler + query + phase padding + optional BOS) must fit the model limit; the program checks this for every case.

## Comparison limitations

- **Context:** SmolLM2-360M's native trained context is typically 8,192 tokens. This runner deliberately does not change RoPE, extend KV length or truncate prompts. A 16K–32K experiment would require explicit, labeled context extrapolation and a validated efficient long-prefill backend such as llama.cpp. It would not be the same in-distribution comparison.
- **Tokenizer:** each model must use its *own* tokenizer and native token positions. Identical 0097 text/fact/question templates and requested token distances do not imply byte-for-byte identical filler sequences or token IDs.
- **Four candidates, eight reference labels:** 0097 has 8 balanced target colors, but `_one_token_choices` scores 4 candidates on each individual case. Uniform-choice chance is 25%; always choosing a fixed color can yield 12.5% across the balanced bank. Report the actual confusion matrix and answer bias, not only aggregate accuracy.
- **Inference:** reference Model B scores with its bounded full-sequence path, while GGUF uses a native Llama-family full-causal cached prefill and scores the last position. Numerical paths differ; the task and scoring definitions are shared.
- **Inference only:** this adapter does not implement the entire mixed foundation capabilities suite. Generation/scoring other than 0097 may need separate external adapters. Choose base versus instruction-tuned SmolLM2 deliberately, and record which was used.
- **Validation:** CPU contract tests cover suite construction, exact distances, explicit exclusions, 4-choice scoring and counterfactual metadata. GPU GGUF inference was not executed during development.
