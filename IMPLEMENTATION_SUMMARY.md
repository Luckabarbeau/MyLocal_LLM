# Extended Training Implementation Summary

## Overview

This document summarizes the extended training infrastructure built for Cosmopedia-v2 training, including fixes for multiprocess parallelization and correct BPE merge ranking.

## Files Created/Modified

### Core Infrastructure

| File | Purpose | Lines |
|------|---------|-------|
| `mini_llm/config.py` | Model configurations (mini/small/medium) | ~100 |
| `mini_llm/checkpoint/__init__.py` | Save/load model/optimizer state | ~80 |
| `mini_llm/data/token_shards.py` | Token shard generation and I/O | ~400 |
| `mini_llm/data/__init__.py` | Module exports | ~10 |
| `mini_llm/train_extended.py` | ExtendedTrainer with accumulation/checkpointing | ~350 |
| `mini_llm/blocks/transformer_block.py` | Fixed parameter naming for unique per-block names | ~10 |
| `mini_llm/tokenizer/tokenizer.py` | BPE tokenizer with merge ranks and multiprocess support | ~280 |

### Training Scripts

| File | Purpose | Lines |
|------|---------|-------|
| `train_tokenizer.py` | Train tokenizer once, save to JSON | ~230 |
| `generate_token_shards.py` | Encode corpus using multiprocessing workers | ~450 |
| `inference.py` | Text generation with multiple decoding strategies | ~250 |

### Tests

| File | Purpose |
|------|---------|
| `test_tokenizer.py` | Comprehensive tokenizer tests |

### Documentation

| File | Purpose |
|------|---------|
| `EXTENDED_TRAINING.md` | Comprehensive training guide |
| `IMPLEMENTATION_SUMMARY.md` | This summary |

## Key Fixes Implemented

### 1. BPE Merge Rank Support

**Problem**: Original encoder used "first valid pair" approach, not respecting learned merge order.

**Solution**: 
- Added `merge_ranks` dictionary tracking when each merge was learned
- Encoding now chooses merges by lowest rank (learned first = highest priority)
- Save/load preserves merge ranks exactly

**Files**: `mini_llm/tokenizer/tokenizer.py`

### 2. Separate Tokenizer Training from Shard Generation

**Problem**: Tokenizer was retrained every time shard generation ran, wasting time and making training unpredictable.

**Solution**: 
- Created `train_tokenizer.py` to train tokenizer once
- Tokenizer saved to JSON with vocab, merges, and merge_ranks
- `generate_token_shards.py` loads pre-trained tokenizer
- Training sample is bounded (50k documents default)

**Files**: `train_tokenizer.py`, `generate_token_shards.py`

### 3. Multiprocess Encoding

**Problem**: Original `--num-workers` flag existed but encoding was still single-core due to:
- Worker function receiving pre-built batches
- No actual multiprocessing in the worker
- Each document reloading tokenizer from disk

**Solution**:
- Used `ProcessPoolExecutor` with initializer pattern
- Workers load tokenizer ONCE at startup via `init_worker()`
- Documents encoded in parallel, order preserved by `executor.map()`
- Chunksize parameter (default 32) for efficient IPC

**Files**: `generate_token_shards.py`

### 4. Memory-Efficient Pipeline

**Problem**: Dataset couldn't be fully loaded (~100GB).

**Solution**:
- Process one Parquet shard at a time
- Use PyArrow streaming via `iter_records()`
- Convert batches to text and encode without materializing entire corpus
- Write token shards incrementally as they're generated

**Files**: `generate_token_shards.py`

### 5. Deterministic Output

**Problem**: Parallel encoding could reorder documents.

**Solution**:
- Use `executor.map()` which preserves input order
- Same input + tokenizer always produces same output
- Verified across 1/2/4/8 worker counts

## Model Configurations

### Predefined Configs

```python
from mini_llm.config import ModelConfig

# Mini (~29M params): d_model=256, d_ff=768, n_layers=8
config = ModelConfig.mini()

# Small (~53M params): d_model=384, d_ff=1024, n_layers=8  
config = ModelConfig.small()

# Medium (~98M params): d_model=512, d_ff=1536, n_layers=8
config = ModelConfig.medium()
```

### VRAM Estimates

| Model | Params | Approx VRAM | Batch Size (RTX 5080) |
|-------|--------|-------------|----------------------|
| Mini | ~29M | 2-3 GB | 32-48 |
| Small | ~53M | 6-8 GB | 16-24 |
| Medium | ~98M | 12-15 GB | 8-12 |

## Features Implemented

### 1. ExtendedTrainer

Production-ready training with:
- Gradient accumulation (effective batch = batch_size × accum_steps)
- Validation monitoring with held-out data
- Periodic checkpointing
- CSV logging for TensorBoard-compatible analysis
- Learning rate warmup + cosine decay schedule

```python
from mini_llm.train_extended import ExtendedTrainer

trainer = ExtendedTrainer(
    model=model,
    train_shard_paths=["shard1.bin", "shard2.bin"],
    val_shard_paths=["val1.bin"],
    batch_size=8,
    seq_length=512,
    grad_accum_steps=2,  # Effective batch: 16
    warmup_steps=1000,
    total_steps=10000,
    peak_lr=3e-4,
)

losses = trainer.train(num_steps=10000)
```

### 2. Checkpoint System

Save/load model, optimizer, and training state:

```python
from mini_llm.checkpoint import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    path="./checkpoints/my_model",
    model_params={p.name: p.data.copy() for p in model.parameters()},
)

# Load
params, _, _ = load_checkpoint("./checkpoints/my_model")
for p in model.parameters():
    if p.name in loaded:
        p.data[...] = loaded[p.name]
```

### 3. Tokenizer Training

Train tokenizer on bounded sample:

```bash
python train_tokenizer.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --vocab-size 16384 \
    --max-documents 50000 \
    --output ./tokenizer.json
```

### 4. Parallel Shard Generation

Encode corpus using multiple workers:

```bash
python generate_token_shards.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --tokenizer ./tokenizer.json \
    --num-shards 104 \
    --documents-per-shard 10000 \
    --num-workers 8 \
    --chunksize 32
```

### 5. Text Generation

Multiple decoding strategies:

```bash
# Greedy decoding
python inference.py \
    --checkpoint ./checkpoints/mini_10shards \
    --prompt "The sky is" \
    --max-new-tokens 50

# Temperature sampling
python inference.py \
    --checkpoint ./checkpoints/mini_10shards \
    --strategy temperature \
    --temperature 0.7 \
    --prompt "Once upon a time"

# Top-p sampling (better quality)
python inference.py \
    --checkpoint ./checkpoints/mini_10shards \
    --strategy top-p \
    --top-p 0.9 \
    --prompt "In the future"
```

## Training Workflow

### Phase 1: Train Tokenizer (One-Time, ~5-10 min)

```bash
python train_tokenizer.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --vocab-size 16384 \
    --max-documents 50000 \
    --output ./tokenizer.json
```

### Phase 2: Quick Test (1 hour)

```bash
# Generate token shards from 5 Parquet shards
python generate_token_shards.py \
    --dataset-path cosmopedia-v2 \
    --tokenizer ./tokenizer.json \
    --num-shards 5 \
    --documents-per-shard 1000 \
    --num-workers 8

# Train mini model
python train_model.py \
    --model mini \
    --total-steps 5000 \
    --batch-size 32 \
    --checkpoint-dir ./checkpoints/test

# Generate text
python inference.py \
    --checkpoint ./checkpoints/test \
    --prompt "The sky is"
```

### Phase 3: Production Run (1-2 days)

```bash
# Generate all token shards
python generate_token_shards.py \
    --dataset-path cosmopedia-v2 \
    --tokenizer ./tokenizer.json \
    --num-shards 104 \
    --documents-per-shard 10000 \
    --num-workers $(nproc)

# Train small model
python train_model.py \
    --model small \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --total-steps 50000 \
    --checkpoint-dir ./checkpoints/small_full \
    --log-file ./logs/small_full.csv

# Resume if needed
python train_model.py \
    --resume-from ./checkpoints/small_full \
    --total-steps 100000
```

## Testing

Run tokenizer tests:

```bash
python test_tokenizer.py
```

Tests verify:
- Basic encoding/decoding
- Merge ranks are assigned correctly
- Encoding respects merge ranks
- Save/load roundtrip preserves state
- Deterministic output across load/save
- Large vocabularies with many merges

## API Reference Quick Start

### Model Creation

```python
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel

config = ModelConfig.mini()
model = DecoderLanguageModel(config, dtype="float16")
```

### Training

```python
from mini_llm.train_extended import ExtendedTrainer

trainer = ExtendedTrainer(
    model=model,
    train_shard_paths=["shard1.bin"],
    val_shard_paths=["val1.bin"],
    batch_size=8,
    seq_length=512,
    grad_accum_steps=2,
)
losses = trainer.train(num_steps=10000)
```

### Inference

```python
from mini_llm import TextGenerator
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer

tokenizer = SimpleBPETokenizer()
tokenizer.load("./tokenizer.json")
generator = TextGenerator(model, tokenizer)

output = generator.generate("Hello", max_new_tokens=50, strategy="temperature", temperature=0.7)
```

### Tokenizer Training

```bash
python train_tokenizer.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --vocab-size 16384 \
    --max-documents 50000 \
    --output ./tokenizer.json
```

### Parallel Shard Generation

```bash
python generate_token_shards.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2/ \
    --tokenizer ./tokenizer.json \
    --num-shards 104 \
    --documents-per-shard 10000 \
    --num-workers 8
```

## Troubleshooting

### Tokenizer Not Found

```bash
# Train tokenizer first
python train_tokenizer.py --dataset-path cosmopedia-v2 --output ./tokenizer.json

# Then generate shards
python generate_token_shards.py --tokenizer ./tokenizer.json ...
```

### Low CPU Utilization

```bash
# Increase worker count (check your CPU cores first)
nproc  # See available cores
python generate_token_shards.py --num-workers $(nproc)

# Adjust chunksize if too much IPC overhead
python generate_token_shards.py --chunksize 64
```

### Out of VRAM

```bash
# Reduce batch size
python train_model.py --batch-size 8

# Increase gradient accumulation
python train_model.py --grad-accum-steps 4

# Use smaller model
python train_model.py --model mini
```

### Slow Training

```bash
# Profile data loading (should be < 10% overhead)
# Check GPU utilization
# Consider mixed precision if not already using
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Loss decrease in 10K steps | 1-2 nats |
| Final perplexity | < 100 (simple text) |
| No NaN/Inf | 100% training stability |
| Coherent output | Makes sense for domain |

## Remaining Algorithmic Bottlenecks

### Tokenizer Training (Intentionally Serial)

- BPE merge ranking requires global pair statistics
- Each merge affects all documents
- Would need incremental statistics approach for true parallelization
- **Not prioritized** - training sample is bounded, one-time cost

### Encoding (Parallelized)

- Now uses multiprocessing with ProcessPoolExecutor
- Each worker loads tokenizer once at startup
- Document order preserved by executor.map()
- Scales linearly up to CPU core count

## Files Changed

### New Files
- `train_tokenizer.py` - Tokenizer training script
- `test_tokenizer.py` - Comprehensive tokenizer tests

### Modified Files
- `mini_llm/tokenizer/tokenizer.py` - Merge ranks, optimized encoding, save/load
- `mini_llm/data/parquet_reader.py` - Added `_get_table_for_shard()` method
- `mini_llm/blocks/transformer_block.py` - Fixed parameter naming
- `generate_token_shards.py` - Full multiprocessing rewrite
- `README.md` - Updated workflow documentation
- `IMPLEMENTATION_SUMMARY.md` - This file
