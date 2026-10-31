# Benchmarks

This directory contains benchmarking and evaluation scripts for the MyLocal_LLM project.

## Evaluation Scripts

These scripts enable evaluation of trained models on standard language modeling benchmarks:

| Script | Purpose |
|--------|---------|
| `evaluate_wikitext.py` | Evaluate on WikiText-2 raw dataset |
| `evaluate_lambada.py` | Evaluate on LAMBADA OpenAI English test |
| `evaluate_corpus.py` | Evaluate on custom text corpus |
| `prepare_external_benchmarks.py` | Prepare external benchmarks for evaluation |

## Usage

Run all commands from the repository root. Generated output is consolidated in
`benchmarks/results/`.

Checkpoint evaluation launchers:

```bash
DRY_RUN=1 bash benchmarks/run_checkpoint_evaluations.sh
DRY_RUN=1 bash benchmarks/run_fusion_evaluations.sh
```

### Preparing Benchmarks

First, prepare the benchmark datasets:

```bash
python prepare_external_benchmarks.py \
    --root ./benchmarks/data \
    --datasets wikitext-2,lambada,c4,pg19
```

### Evaluating a Checkpoint

Evaluate on WikiText-2:

```bash
python evaluate_wikitext.py \
    --checkpoint ./checkpoints/my_model \
    --data ./benchmarks/wikitext-2-raw/wiki.test.raw \
    --stride 256
```

Evaluate on LAMBADA:

```bash
python evaluate_lambada.py \
    --checkpoint ./checkpoints/my_model \
    --data ./benchmarks/lambada-test.jsonl
```

### Custom Corpus Evaluation

```bash
python evaluate_corpus.py \
    --checkpoint ./checkpoints/my_model \
    --data ./my_corpus.txt \
    --context-length 1024
```

## Benchmark Metrics

All evaluation scripts report:
- **Token cross-entropy**: Average negative log-likelihood per token
- **Perplexity**: Exponential of cross-entropy
- **Bits per byte**: Compression efficiency metric
- **Token/byte ratio**: Encoding efficiency
- **Throughput**: Tokens processed per second

## Benchmark Scripts (Performance)

The following scripts measure performance characteristics:

| Script | Purpose |
|--------|---------|
| `benchmark_performance.py` | Comprehensive training/inference benchmark |
| `benchmark_tokenizer.py` | Tokenizer performance comparison |
| `benchmark_inference_decode_components.py` | Inference decode component analysis |
| `benchmark_terminal_memory_scaling.py` | Memory usage scaling analysis |

## Notes

- Generated benchmark results (`.csv`, `.txt`) should be git-ignored
- The `benchmarks/data/` directory contains prepared datasets and should also be ignored
- Use `--checkpoint` to specify your trained model checkpoint
- Use `--backend cupy` for GPU-accelerated evaluation

## Fixed-prompt generation and repetition benchmark (Patches 0078A–0078B)

`evaluate_generation.py` evaluates the same 24 prompts (mathematics, science,
Python/C++, and stories; short/medium/long openings) with three fixed seeds.
It uses `inference.TextGenerator` and the same KV-cache generation path as
`inference.py`. It loads each checkpoint **once** per benchmark run.

From the repository root, using the usual validated GPU environment:

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=1536 \
python benchmarks/evaluate_generation.py \
    --checkpoint ./checkpoints/wide-500m-memory-64k_0060_30b_muon_run_baseline \
    --output-dir ./benchmarks/results/generation_45600
```

Default (0078B): **two decoding strategies on every prompt/seed**: top-k=40
and top-p=1.0 (full-distribution temperature sampling), both temperature=0.8,
512 new tokens, seeds 42/123/2026, KV-cache, and the checkpoint's trained
context horizon. This is 24 x 3 x 2 = 144 generations. For a quick smoke test:

```bash
MINI_LLM_BACKEND=cupy python benchmarks/evaluate_generation.py \
    --checkpoint ./checkpoints/wide-500m-memory-64k_0060_30b_muon_run_baseline \
    --output-dir ./benchmarks/results/generation_smoke \
    --prompt-ids math_general_short,code_python_short \
    --seeds 42 --max-new-tokens 64
```

To benchmark only one strategy, use `--strategy top-k` or `--strategy top-p`.
To run other combinations, use `--strategies top-k,top-p --top-p 0.9`.
For matched comparisons, give each distinct configuration its own output directory;
`--resume` will reject any configuration change. The 0078A one-strategy results
are not migrated to the 0078B two-strategy format.

An interrupted run can be continued by repeating the **same** command with
`--resume` (a complete run can also be resumed harmlessly). Results:

- `manifest.json`: checkpoint step, complete configuration, prompt-suite SHA256,
  and effective `MINI_LLM_*` runtime environment.
- `generations.jsonl`: full prompt, seed, **decoding strategy / temperature / k / p**,
  generated text, exact generated token IDs, EOS status, per-sample metrics
  and summary timings; appended after each generation for safe restart.
- `summary.json` / `summary.csv`: overall, **by decoding strategy**, by
  strategy x prompt length, by strategy x domain, and existing domain/length
  statistics. Partial summaries are updated after each completed sample.
  Compare top-k and top-p using the *strategy-specific* rows; overall combines both.

`distinct-3/4/5/8`, repeated n-gram rates, first repeated 8-gram position, and
longest exact repeated span are calculated from **generated token IDs only**,
excluding the prompt and terminal EOS. Longest repeated span allows overlapping
occurrences. Missing (too-short) n-gram metrics use JSON `null`.

These are automatic **exact-repetition** metrics. They cannot determine whether
a story remains coherent, a claim is true, code executes correctly, or two
paraphrases repeat the same idea. Inspect `generations.jsonl` for those
questions. Compare checkpoints using identical prompts, seeds, decoding
parameters, and runtime settings.

CPU-only checks: `python -m pytest -q tests/test_generation_benchmark.py`.

### Model B long-retrieval evaluation (0097)

The progressive 0097 suite now evaluates synthetic factual recall at distances
of **1K, 4K, 6K, 8K, 12K, 16K, 24K, and 32K tokens**. Each main-suite
length uses all eight base project facts with two deterministic variations
(**16 probes per distance**), covering all 16 original-token sparse-attention
phases for the stride-16 Model B. The short smoke suite retains fewer cases.
A separate, less frequent, smaller counterfactual run tests whether changing
an historical fact changes the model's token probabilities.

Model B is identified by `dual_resolution.enabled` and
`compression_mode="matrix_pool"`. It evaluates complete contexts beyond the
*training* `context_length` where the configured `max_blocks * block_size`
permits, without changing weights or the training configuration. Its forward
pass uses `forward_body(return_cache=False)` plus the vocabulary projection
**only for the final hidden state**. This avoids materializing `[T, vocab]`
logits, but still performs full-sequence attention, which can be expensive at
32K. The historical-read model retains its existing native prefill path.

The `result.json`, `report.html` and `retrieval_by_case.csv` contain exact
fact-end-to-query token distance, correct-token **full-vocabulary rank**, its
log-probability and probability, four-candidate rank and margin, phase modulo
16 and (when enabled) counterfactual score shifts. `by_distance` also records
median/geometric-mean rank and top-128 fraction. The benchmark is deterministic
and never truncates unsupported long prompts. Interpretation: better target
rank alone can reflect language-model progress; causal sensitivity to changing
the fact is a stronger retrieval diagnostic. The filler text is intentionally
repeated; success here does not establish general long-document retrieval.

To reduce the number of full-model evaluations, adjust the `task_ids`,
`repetitions`, or `distances` fields in the chosen
`benchmarks/configs/foundation_capabilities_0097*.json` file. Each distinct
configuration gets a different suite hash. Full-model 32K validation on your
RTX 5080 still needs to be run against a trained checkpoint; CPU tests cover
logic, phase coverage and last-token-only projection, not GPU performance.
