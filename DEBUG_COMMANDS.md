# Debug Commands for MyLocal_LLM

## Quick Start - Micro Model (47K params)

### Run Debug Training (50 steps)
```bash
cd /home/lucka/Documents/my_local_LLM/MyLocal_LLM

python debug_train.py \
    --model micro_debug \
    --total-steps 50 \
    --batch-size 4 \
    --context-length 32 \
    --peak-lr 1e-3
```

**Expected output:**
- Creates synthetic dataset automatically in `./debug_shards`
- Trains for 50 steps (~4 seconds)
- Loss should decrease from ~5.5 to ~4.9
- Checkpoint saves every 25 steps

### Run Extended Debug Training (100 steps with validation)
```bash
python debug_train.py \
    --model micro_debug \
    --total-steps 100 \
    --batch-size 4 \
    --context-length 32 \
    --peak-lr 1e-3 \
    --checkpoint-dir ./checkpoints/debug_extended \
    --log-file ./logs/debug_extended.csv
```

### Resume from Checkpoint
```bash
# First, train for some steps
python debug_train.py \
    --model micro_debug \
    --total-steps 50 \
    --batch-size 4 \
    --context-length 32 \
    --checkpoint-dir ./checkpoints/resume_test

# Then resume (edit the script or modify arguments)
# Note: Current resume requires modifying the script to pass --resume-from
# This is a limitation that should be fixed
```

### Test Checkpoint/Resume Manually
```bash
python -c "
from pathlib import Path
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.checkpoint import save_checkpoint, load_checkpoint

config = ModelConfig.micro_debug()
model = DecoderLanguageModel(config, rng_seed=42, dtype='float16')

# Save
save_checkpoint(
    path='./checkpoints/resume_test',
    model_params={p.name: p.data for p in model.parameters()},
    optimizer_state=None,
    training_state={'step': 0},
)

# Load
param_names = [p.name for p in model.parameters()]
loaded_params, _, _ = load_checkpoint('./checkpoints/resume_test', param_names=param_names)
for p in model.parameters():
    if p.name in loaded_params:
        p.data[...] = loaded_params[p.name]

print('Checkpoint save/load verified!')
"
```

## GPU Mode (CuPy)

To run with GPU acceleration:

```bash
export MINI_LLM_BACKEND=cupy

python debug_train.py \
    --model micro_debug \
    --total-steps 50
```

Or in one line:
```bash
MINI_LLM_BACKEND=cupy python debug_train.py --model micro_debug --total-steps 50
```

## Full Cosmopedia Training (After Debug Passes)

### Small Model (53M params, ~100M tokens)
```bash
cd /home/lucka/Documents/my_local_LLM/MyLocal_LLM

python train_model.py \
    --model small \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2 \
    --num-parquet-shards 104 \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --context-length 512 \
    --total-steps 25000 \
    --warmup-steps 1000 \
    --peak-lr 3e-4 \
    --grad-clip 1.0 \
    --weight-decay 0.1 \
    --val-ratio 0.01 \
    --checkpoint-dir ./checkpoints/cosmopedia_small_100M \
    --log-file ./logs/cosmopedia_small_100M.csv \
    --save-interval 5000 \
    --val-interval 500 \
    --seed 42
```

### Mini Model (12M params, faster for initial runs)
```bash
python train_model.py \
    --model mini \
    --dataset-path ../cosmopedia-v2/cosmopedia-v2 \
    --num-parquet-shards 50 \
    --batch-size 8 \
    --context-length 512 \
    --total-steps 10000 \
    --checkpoint-dir ./checkpoints/cosmopedia_mini_10M \
    --log-file ./logs/cosmopedia_mini_10M.csv
```

## Test Commands

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Module
```bash
python -m pytest tests/test_packed_dataset.py -v
python -m pytest tests/test_train_extended.py -v
python -m pytest tests/test_moe_block.py -v
```

## Troubleshooting

### If training fails with "tokenizer not found":
The debug script creates its own tokenizer. For Cosmopedia, ensure:
```bash
# Check dataset exists
ls ../cosmopedia-v2/cosmopedia-v2/train-*.parquet | head -5
```

### If out of memory:
- Reduce batch size: `--batch-size 2`
- Reduce context length: `--context-length 256`
- Use micro model for debugging: `--model micro_debug`

### If no GPU available:
```bash
export MINI_LLM_BACKEND=numpy
# or just run without setting anything (numpy is default)
```

## Files Created During Debug

| File/Dir | Description |
|----------|-------------|
| `./debug_shards/` | Synthetic dataset for testing |
| `./checkpoints/debug_micro/` | Training checkpoints |
| `./logs/debug_micro.csv` | Training logs |
