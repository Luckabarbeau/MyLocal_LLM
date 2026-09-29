# mini-llm-lab

A deliberately explicit decoder-LLM implementation for studying the numerical
structure of modern language models.

The core design rule is:

> Learned quantities are visible matrices, forward operations are explicit tensor
> contractions, and backward operations are hand-derived.

No PyTorch, JAX, TensorFlow, Hugging Face Transformers, or automatic differentiation
is used in the numerical core.

## Milestone 1

Implemented:

- NumPy / CuPy backend switch
- explicit `Parameter(data, grad)` objects
- deterministic clipped-normal initialization
- explicit Linear layer + backward
- token Embedding + scatter-add backward
- RMSNorm + backward
- SwiGLU + backward
- RoPE + transpose-rotation backward
- grouped-query causal self-attention (GQA) + full handwritten backward
- numerically stable cross entropy + backward
- global gradient clipping
- AdamW
- warmup + cosine learning-rate schedule
- directional derivative checks
- CPU unit tests
- an inspection script that prints actual attention matrices/tensors

## Backend

CPU/reference mode is the default:

```bash
export MINI_LLM_BACKEND=numpy
```

For an NVIDIA GPU with CuPy installed:

```bash
export MINI_LLM_BACKEND=cupy
```

The exact same model code is used in both cases.

## Install

CPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

GPU (CUDA 12.x example):

```bash
pip install -e ".[gpu,dev]"
export MINI_LLM_BACKEND=cupy
```

Choose the CuPy package matching the CUDA version installed on the machine.

## Inspect a tiny attention block

```bash
python examples/inspect_attention.py
```

The example uses a deliberately tiny model:

- `d_model = 16`
- `n_q_heads = 2`
- `n_kv_heads = 1`
- `d_head = 8`
- `sequence_length = 5`

and prints `Wq`, `Wk`, `Wv`, `Wo`, Q/K/V, the causal score matrix,
softmax probabilities, and the output.

## Cosmopedia-v2 Dataset Setup

### Step 1: Download the Dataset

The full Cosmopedia-v2 dataset is ~100GB with 104 Parquet shards:

```bash
# Activate your virtual environment first
source .venv/bin/activate

mkdir -p cosmopedia-v2
cd cosmopedia-v2

# Download all 104 Parquet shards
for i in $(seq 0 103); do
    f=$(printf "train-%05d-of-00104.parquet" "$i")
    echo "Downloading $f"
    wget -c \
      "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/${f}?download=true" \
      -O "$f"
done
```

### Step 2: Generate Token Shards (Test Mode)

**Important**: Don't process all 104 shards at once! Start with just 5-10 Parquet shards for testing:

```bash
cd ..  # Go back to project root
python generate_token_shards.py \
    --dataset-path cosmopedia-v2 \
    --output-dir ./token_shards \
    --num-shards 5 \            # Only process 5 Parquet shards
    --documents-per-shard 1000  # Generate smaller token shards
```

This will:
- Train tokenizer on 500 documents (sample from first Parquet shard)
- Tokenize ~50,000 documents from 5 Parquet shards using multiprocessing
- Generate token shards with ~1000 documents each
- Output: ~50 token shards in `./token_shards/`

### Step 2b: Accelerating Token Generation (Optional)

Token generation uses multiprocessing by default. Adjust worker count based on your CPU:

```bash
# Use all available cores (recommended)
python generate_token_shards.py --num-workers $(nproc)

# Use a specific number of workers
python generate_token_shards.py --num-workers 8

# Single-threaded (for debugging)
python generate_token_shards.py --num-workers 1
```

### Step 2c: Full Dataset (Optional)

Once testing works, process the full dataset:

```bash
python generate_token_shards.py \
    --dataset-path cosmopedia-v2 \
    --num-shards 104 \
    --documents-per-shard 10000 \
    --num-workers $(nproc)  # Use all CPU cores

### Step 3: Full Dataset (Optional)

Once testing works, you can process the full dataset:

```bash
python generate_token_shards.py \
    --dataset-path cosmopedia-v2 \
    --num-shards 104 \
    --documents-per-shard 10000
```

## Extended Training

All training commands require the virtual environment to be activated:

```bash
source .venv/bin/activate
```

### Backend Selection (CPU vs GPU)

By default, training runs on CPU. For GPU acceleration with CuPy:

```bash
export MINI_LLM_BACKEND=cupy  # Enable GPU mode
```

For CPU mode (default):

```bash
export MINI_LLM_BACKEND=numpy  # Or unset the variable
```

### Quick Start

```bash
# Generate token shards from downloaded Cosmopedia data
python generate_token_shards.py --dataset-path cosmopedia-v2

# Train mini model (1 hour)
python train_model.py \
    --model mini \
    --batch-size 32 \
    --total-steps 5000 \
    --checkpoint-dir ./checkpoints/mini_test

# Generate text
python inference.py \
    --checkpoint ./checkpoints/mini_test \
    --prompt "The sky is"
```

### Full Training (1-2 days)

```bash
# Generate all shards from full dataset
python generate_token_shards.py \
    --dataset-path cosmopedia-v2 \
    --num-shards 104

# Train small model with full dataset
python train_model.py \
    --model small \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --total-steps 50000 \
    --checkpoint-dir ./checkpoints/small_full \
    --log-file ./logs/small_full.csv

# Resume training
python train_model.py \
    --resume-from ./checkpoints/small_full \
    --total-steps 100000
```

## Milestones 5-8 Completed

- ✅ Dataset/tokenizer pipeline and binary token shards
- ✅ Full training loop with checkpointing, logging, validation
- ✅ KV cache and autoregressive inference (inference.py)
- ✅ AdamW optimizer with mixed precision support

See `EXTENDED_TRAINING.md` for complete documentation.
