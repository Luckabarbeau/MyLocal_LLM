# Tokenizer Implementation Changes

## Overview

This document describes the tokenizer improvements made to support Cosmopedia preprocessing.

### Problem Statement

The original `SimpleBPETokenizer` was designed as an educational reference implementation. While transparent and easy to understand, it was far too slow for production Cosmopedia preprocessing - training a 16K vocabulary tokenizer could take hours.

### Solution

We implemented a two-tier approach:

1. **Preserve SimpleBPETokenizer** as a reference/educational implementation
2. **Add FastBPETokenizer** using Hugging Face tokenizers for production use

## Files Changed

### New Files
- `benchmarks/benchmark_tokenizer.py` - Comprehensive benchmark suite
- `mini_llm/tokenizer/config.py` - Tokenizer configuration classes

### Modified Files
- `mini_llm/tokenizer/tokenizer.py` - Added FastBPETokenizer, TokenizerProtocol
- `mini_llm/tokenizer/__init__.py` - Updated exports
- `mini_llm/config.py` - Split tokenizer_vocab_size from model vocab
- `train_tokenizer.py` - Support both backends with CLI options
- `generate_token_shards.py` - Use FastBPETokenizer by default
- `train_model.py` - Configurable tokenizer settings

## Performance Results

### Training Speed (512 vocab, 5MB corpus)
| Backend | Time |
|---------|------|
| SimpleBPETokenizer | ~0.5s |
| FastBPETokenizer | <0.01s |

**~50x faster training**

### Encoding Speed (single-thread)
| Backend | Throughput |
|---------|------------|
| SimpleBPETokenizer | ~370 KB/s |
| FastBPETokenizer | ~4760 KB/s |

**~13x faster encoding**

### Batch Encoding (4 threads)
| Backend | Throughput |
|---------|------------|
| N/A | ~12,400 KB/s |

**~33x faster than single-threaded SimpleBPETokenizer**

## Usage

### Training a Tokenizer

```bash
# Quick smoke test (5MB sample, 512 vocab)
python train_tokenizer.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --vocab-size 512 \
    --output data/tokenizer_smoke.json \
    --smoke-test

# Production training (750MB sample, 16K vocab)
python train_tokenizer.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --vocab-size 16384 \
    --output data/tokenizer.json \
    --backend fast \
    --threads 8

# Reuse existing tokenizer
python train_tokenizer.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --tokenizer-path data/tokenizer.json \
    --output data/tokenizer_new.json
```

### Generating Token Shards

```bash
# Generate shards using pre-trained tokenizer
python generate_token_shards.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --tokenizer data/tokenizer.json \
    --num-shards 10 \
    --documents-per-shard 10_000 \
    --num-workers 8
```

### Training a Model

```bash
# Use default tokenizer settings
python train_model.py \
    --model mini \
    --num-parquet-shards 10 \
    --context-length 512 \
    --batch-size 16 \
    --total-steps 10000

# Or use pre-trained tokenizer
python train_model.py \
    --model mini \
    --tokenizer-path data/tokenizer.json \
    ...
```

## API

### Tokenizer Backends

#### SimpleBPETokenizer
- Educational reference implementation
- Pure Python BPE algorithm
- Transparent merge order
- Slower but suitable for small datasets and learning

#### FastBPETokenizer
- Production implementation using Hugging Face tokenizers
- Rust-based BPE backend
- Multi-threaded processing
- Byte-level BPE for robust Unicode/code handling
- Significantly faster training and encoding

### Configuration

```python
from mini_llm.tokenizer.config import TokenizerConfig, TokenizerManager

# Using config class
config = TokenizerConfig(
    vocab_size=16384,
    backend="fast",
    threads=8,
    sample_mb=750.0,
)

manager = TokenizerManager(config)
tokenizer = manager.load_or_train()
```

### Factory Function

```python
from mini_llm.tokenizer import get_tokenizer

# Create tokenizer by backend
simple_tok = get_tokenizer("simple", vocab_size=256)
fast_tok = get_tokenizer("fast", vocab_size=16384, threads=8)
```

## Special Tokens

Consistent special token IDs across all backends:
- `<|pad|>` → ID 0
- `<|eos|>` → ID 1  
- `<|unk|>` → ID 2

## Byte-Level BPE

The FastBPETokenizer uses byte-level BPE which provides:
- Arbitrary Unicode coverage
- No large unknown-token problem
- Robust source code handling
- Robust punctuation handling
- No dependence on language-specific whitespace splitting

This is especially important for mixed natural language + code training.

## Vocabulary Size Recommendations

| Use Case | Recommended Vocab |
|----------|------------------|
| Smoke test/debugging | 256-512 |
| Development/testing | 8192 |
| Production (coding/scientific) | 16384-32768 |
| Large production | 32768+ |

**Note**: Keep vocab size ≤ 65536 for uint16 token storage.

## Benchmarking

```bash
# Quick smoke test
python benchmarks/benchmark_tokenizer.py --smoke-test

# Full benchmark (may take minutes)
python benchmarks/benchmark_tokenizer.py
```

## Migration Guide

### From SimpleBPETokenizer Only

Old code:
```python
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
tokenizer = SimpleBPETokenizer(vocab_size=16384)
tokenizer.train(texts)
```

New code (fast):
```python
from mini_llm.tokenizer import FastBPETokenizer
tokenizer = FastBPETokenizer(vocab_size=16384, threads=8)
tokenizer.train(texts)
```

Or use factory:
```python
from mini_llm.tokenizer import get_tokenizer
tokenizer = get_tokenizer("fast", vocab_size=16384, threads=8)
```

### Training Workflow

**Old approach** (slow):
1. Train tokenizer every model run (~hours for 16K vocab)
2. Generate token shards
3. Train model

**New approach** (fast):
1. Train tokenizer ONCE (~10 seconds for 16K vocab)
2. Save tokenizer to disk
3. Reuse tokenizer for all experiments
4. Generate token shards ONCE
5. Run many model experiments

## Dependencies

### Optional: Hugging Face tokenizers
```bash
pip install tokenizers
```

Required for FastBPETokenizer. SimpleBPETokenizer works without it.

## Testing

```bash
# Run tokenizer tests
python test_tokenizer.py

# Run benchmarks
python benchmarks/benchmark_tokenizer.py --smoke-test
```
