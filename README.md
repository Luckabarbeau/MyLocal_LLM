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

## Planned milestones

1. Numerical primitives — current milestone.
2. Transformer block with residual streams and explicit block backward.
3. Dense decoder LM and small-text overfit test.
4. Sparse MoE, router diagnostics, load balancing.
5. Dataset/tokenizer pipeline and binary token shards.
6. Full training loop/checkpointing.
7. KV cache and autoregressive inference.
8. AdamW vs Muon experiments.
