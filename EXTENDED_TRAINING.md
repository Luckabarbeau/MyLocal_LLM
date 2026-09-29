# Extended Training Guide

This document describes the extended training infrastructure for Cosmopedia-v2 training.

## Architecture Overview

```
Model Configurations
├── Mini (~12M params):   d_model=256,  d_ff=768,   n_layers=8
├── Small (~53M params):  d_model=384,  d_ff=1024,  n_layers=8
└── Medium (~120M params): d_model=512, d_ff=1536, n_layers=8

Training Features
├── Gradient Accumulation: Effective batch size = batch_size × accum_steps
├── Mixed Precision: FP16 model weights with FP32 optimizer state
├── Checkpointing: Save/load model, optimizer, and training state
├── Logging: CSV format for TensorBoard-compatible analysis
├── Validation: Hold-out 1% for overfitting monitoring
└── LR Schedule: Warmup + cosine decay

Data Pipeline
├── Parquet Reader: Streaming from Cosmopedia-v2
├── Token Shards: Binary format for efficient loading
└── Multi-shard: Process multiple Parquet files
```

## Quick Start

### 1. Generate Token Shards

```bash
# Process 104 Parquet shards with default settings
python generate_token_shards.py \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2 \
    --output-dir ./token_shards \
    --num-shards 104 \
    --documents-per-shard 10000 \
    --context-length 512
```

Output:
- `token_shards/shard_*.bin` - Binary token shards
- `token_shards/tokenizer.json` - Trained tokenizer

### 2. Train a Model

```bash
# Train mini model on 10 Parquet shards
python train_model.py \
    --model mini \
    --num-parquet-shards 10 \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --seq-length 512 \
    --total-steps 10000 \
    --checkpoint-dir ./checkpoints/mini_10shards \
    --log-file ./logs/mini_10shards.csv
```

Key flags:
- `--model`: Model configuration (mini/small/medium)
- `--num-parquet-shards`: Number of Parquet files to use
- `--batch-size`: Per-step batch size
- `--grad-accum-steps`: Gradient accumulation factor
- `--total-steps`: Total training steps
- `--checkpoint-dir`: Directory for saving checkpoints
- `--log-file`: CSV log file path

### 3. Resume Training

```bash
python train_model.py \
    --resume-from ./checkpoints/mini_10shards \
    --total-steps 20000
```

### 4. Generate Text

```bash
# Greedy decoding
python inference.py \
    --checkpoint ./checkpoints/mini_10shards \
    --prompt "The sky is" \
    --max-new-tokens 50

# Temperature sampling
python inference.py \
    --checkpoint ./checkpoints/mini_10shards \
    --prompt "Once upon a time" \
    --temperature 0.7 \
    --max-new-tokens 100 \
    --strategy temperature

# Top-p sampling (better quality)
python inference.py \
    --checkpoint ./checkpoints/mini_10shards \
    --prompt "In the future" \
    --top-p 0.9 \
    --max-new-tokens 75 \
    --strategy top-p
```

## Training Configuration

### Model Sizes

| Size | d_model | d_ff | n_heads | Parameters | VRAM (approx) |
|------|---------|------|---------|------------|---------------|
| Mini | 256 | 768 | 4 | ~12M | 2-3 GB |
| Small | 384 | 1024 | 6 | ~53M | 6-8 GB |
| Medium | 512 | 1536 | 8 | ~120M | 12-15 GB |

### Recommended Settings

**Mini Model (RTX 5080):**
```bash
--batch-size 32 --grad-accum-steps 1  # Effective: 32
```

**Small Model (RTX 5080):**
```bash
--batch-size 16 --grad-accum-steps 2  # Effective: 32
```

**Medium Model (RTX 5080):**
```bash
--batch-size 8 --grad-accum-steds 4  # Effective: 32
```

### Learning Rate Schedule

- Warmup: 1000 steps with linear ramp-up
- Peak LR: 3e-4 (can go up to 1e-3 for smaller models)
- Decay: Cosine decay to 0

## Data Pipeline

### Token Shard Format

Binary format with header + data:
```
Header (16 bytes):
  - num_documents: int64
  - context_length: int64

Data:
  - uint16 token IDs (num_documents × context_length)
```

### Expected Performance

- Loading shard: ~10-50ms (depends on disk speed)
- Batch extraction: ~1-5ms
- Total pipeline overhead: < 10% of training time

## Evaluation Metrics

### Perplexity Targets

| Training Steps | Expected Loss | Expected Perplexity |
|----------------|---------------|---------------------|
| 1K | 10-12 | ~20000-100000 |
| 5K | 6-8 | ~400-3000 |
| 10K | 4-6 | ~55-400 |
| 50K | 2-4 | ~7-55 |
| 100K+ | 1-3 | ~3-20 |

### Success Criteria

1. **Loss decrease**: Should decrease by 1-2 nats in first 10K steps
2. **Perplexity**: Should drop below 100 after sufficient training
3. **No NaN/Inf**: Training should complete without numerical issues
4. **Coherent output**: Generated text should make sense for the domain

## Checkpointing

### What Gets Saved

- Model parameters (all trainable weights)
- Optimizer state (momentum, variance)
- Training state (step count, shard indices)

### Checkpoint Frequency

Default: Every 2000 steps
Adjust with: `--save-interval N`

### Manual Save/Load

```python
from mini_llm.checkpoint import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    path="./checkpoints/my_model",
    model_params={p.name: p.data for p in model.parameters()},
    optimizer_state=optimizer.state_dict(),
    training_state={"step": step},
)

# Load
params, opt_state, train_state = load_checkpoint("./checkpoints/my_model")
```

## Logging

### CSV Format

```csv
step,train_loss,lr,grad_norm,val_loss,steps_per_sec
1000,5.234,0.000100,0.876,5.412,15.23
2000,4.876,0.000200,0.912,5.123,15.45
...
```

### Visualization

Import CSV into:
- Excel/Sheets for basic analysis
- Python (pandas) for advanced analysis
- TensorBoard (using plugin or conversion)

## Troubleshooting

### NaN Loss
1. Check gradients are being computed correctly
2. Verify learning rate isn't too high
3. Check for division by zero in attention
4. Try lower learning rate

### Out of VRAM
1. Reduce batch size
2. Increase gradient accumulation
3. Use smaller model (mini < small < medium)
4. Reduce sequence length

### Slow Training
1. Profile data loading (should be < 10% overhead)
2. Check GPU utilization (should be > 70%)
3. Consider mixed precision training
4. Increase batch size if VRAM allows

### Poor Convergence
1. Verify loss is actually decreasing
2. Check learning rate schedule
3. Ensure sufficient training steps (10K+ minimum)
4. Try different learning rate (increase or decrease)
5. Check data quality (tokenizer, shard generation)

## Next Steps

1. **Start with Mini model**: Test the full pipeline quickly
2. **Monitor validation loss**: Catch overfitting early
3. **Generate text periodically**: Visual check of quality
4. **Scale up gradually**: Move to Small/Medium after successful mini run
5. **Longer training**: 50K-100K steps for good quality

## Example Workflows

### Quick Test (1 hour)
```bash
# 1. Generate shards
python generate_token_shards.py --num-shards 10

# 2. Train mini model
python train_model.py \
    --model mini \
    --num-parquet-shards 10 \
    --total-steps 5000 \
    --batch-size 32 \
    --checkpoint-dir ./checkpoints/test

# 3. Generate text
python inference.py \
    --checkpoint ./checkpoints/test \
    --prompt "The sky is"
```

### Production Run (1-2 days)
```bash
# 1. Generate all shards
python generate_token_shards.py --num-shards 104

# 2. Train small model
python train_model.py \
    --model small \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --total-steps 50000 \
    --checkpoint-dir ./checkpoints/small_full \
    --log-file ./logs/small_full.csv

# 3. Resume and continue if needed
python train_model.py \
    --resume-from ./checkpoints/small_full \
    --total-steps 100000
```

## API Reference

### ModelConfig

```python
from mini_llm.config import ModelConfig

# Predefined configs
config = ModelConfig.mini()
config = ModelConfig.small()
config = ModelConfig.medium()

# Custom config
config = ModelConfig(
    vocab_size=16384,
    context_length=512,
    n_layers=8,
    d_model=384,
    n_q_heads=6,
    n_kv_heads=2,
    d_ff=1024,
)
```

### ExtendedTrainer

```python
from mini_llm.train_extended import ExtendedTrainer

trainer = ExtendedTrainer(
    model=model,
    train_shard_paths=["shard1.bin", "shard2.bin"],
    val_shard_paths=["val1.bin"],
    batch_size=8,
    seq_length=512,
    grad_accum_steps=2,
    warmup_steps=1000,
    total_steps=10000,
    peak_lr=3e-4,
)

losses = trainer.train(num_steps=10000)
```

### TextGenerator

```python
from mini_llm.inference import TextGenerator

generator = TextGenerator(model, tokenizer)

# Greedy
output = generator.generate("Hello", max_new_tokens=50)

# Temperature
output = generator.generate("Hello", max_new_tokens=50, strategy="temperature", temperature=0.7)

# Top-p
output = generator.generate("Hello", max_new_tokens=50, strategy="top-p", top_p=0.9)
```
