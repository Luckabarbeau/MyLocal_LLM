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
