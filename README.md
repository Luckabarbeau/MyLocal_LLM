# MyLocal_LLM

**A research and development platform for building, training, inspecting, and experimenting with small language models on consumer hardware.**

The purpose of this repository is not to reproduce a frontier model by scaling compute until the problem disappears. It is to explore the opposite question:

> **How capable can a language model become when memory, compute, bandwidth, and training time are hard constraints?**

Those constraints are intentional. They force architectural and numerical questions that are easy to avoid when the default answer is simply *make the model larger, make the context longer, and add more GPUs*.

This project therefore treats a consumer workstation as a useful research environment rather than merely a limitation. The goal is to develop tools, training infrastructure, and model architectures that make local LLM research practical, understandable, and modifiable.

The numerical core is deliberately explicit: NumPy/CuPy tensors, visible parameter arrays, handwritten forward and backward passes, custom attention kernels, explicit optimizer state, and no automatic differentiation framework.

---

## Project philosophy

Several principles guide development.

### 1. Constraints are a design tool

A fixed GPU memory budget, limited compute, and local execution force useful questions:

- Can long context be made cheap instead of simply made dense?
- Can useful memory be selected before expensive Transformer processing?
- Can sparse attention preserve the important interactions while removing waste?
- Can optimizer state live in system RAM without destroying throughput?
- Can recomputation replace large temporary tensors when VRAM is scarce?
- Can a smaller model compensate for limited capacity with better context construction and better training data?

The project should prefer finding a better formulation over assuming that scaling alone will solve the problem.

### 2. Make the model understandable

The repository intentionally exposes operations that large frameworks usually hide:

- parameter arrays and gradients;
- Q/K/V projections;
- RoPE;
- sparse-attention plans;
- MoE routing;
- context routing;
- cross entropy;
- optimizer updates;
- mixed-precision behavior;
- inference KV-cache layout.

This makes the repository useful not only for training a model, but for studying and changing one.

### 3. Architecture should remain modular

Attention policies, context routing, MoE structure, memory integration, inference caching, and training behavior are separate components where practical. New ideas should be testable as controlled alternatives rather than requiring a complete rewrite.

### 4. Measure instead of assuming

Changes are expected to be validated through some combination of:

- reference NumPy tests;
- numerical equivalence tests;
- gradient checks;
- memory measurements;
- throughput measurements;
- held-out language-model loss;
- ablations;
- task benchmarks.

A theoretically cheaper architecture is not useful if irregular kernels make it slower in practice.

---

# Current state

The repository has grown from a small educational decoder implementation into a complete local pretraining and inference stack.

Implemented capabilities include:

- NumPy reference backend and CuPy CUDA backend;
- explicit `Parameter(data, grad)` objects;
- handwritten backward passes throughout the numerical core;
- BF16 mixed-precision training;
- RMSNorm, RoPE, SwiGLU, GQA, MoE and sparse attention;
- local, dilated, deterministic-global, and learned-retrieval attention heads;
- packed token shards and mixed-corpus pretraining;
- custom BPE tokenizer tooling;
- gradient accumulation;
- AdamW with optimizer-state offload to pinned system RAM;
- chunked/recomputed LM head for low-VRAM training and validation;
- checkpoint/resume including optimizer state and RNG state;
- KV-cached autoregressive inference;
- learned long-context routed-prefix memory;
- bounded long-context inference with deterministic router refreshes;
- extensive NumPy/reference tests and numerical validation tools.

The current research focus is the **routed sparse-context architecture** described below.

---

# Current reference model

The main experimental family is `wide-500m-*`.

The current 64k routed-memory preset is:

| Component | Configuration |
|---|---:|
| Parameters | ~501M total |
| Transformer layers | 12 |
| Model width | 768 |
| Query heads | 12 |
| KV heads | 3 |
| Head dimension | 64 |
| MoE experts | 6 |
| Active experts/token | 2 |
| Expert FFN width | 2304 |
| Tokenizer vocabulary | 65,280 |
| Dense current window | 4,096 tokens |
| Addressable causal horizon | 65,536 tokens |
| Searchable external history | up to 61,440 tokens |
| History block size | 128 tokens |
| Routed blocks | up to 16 |
| Routed prefix | up to 2,048 tokens |
| Maximum deep active sequence | 6,144 tokens |

The model is intentionally not a conventional dense-attention 64k Transformer. The long history is reduced **before** the expensive Transformer trunk.

---

## Sparse attention inside the Transformer

Each layer of the current wide model uses 12 query heads with complementary roles:

```text
6 local heads
2 dilated heads
2 deterministic global-sparse heads
2 learned retrieval heads
```

The 4k reference geometry uses bounded local/dilated work, deterministic broad coverage, and content-dependent retrieval. These mechanisms are complementary:

- **local attention** preserves dense recent information;
- **dilated attention** provides inexpensive structured medium-range coverage;
- **global sparse anchors** guarantee deterministic long-range connectivity;
- **learned retrieval attention** provides content-dependent distant access.

This sparse-attention system is distinct from the top-level sparse-context mechanism.

---

# Routed sparse context

The current long-context experiment asks a simple question:

> Instead of passing the entire long context through every Transformer layer, can a cheap learned router construct a smaller context containing the historical information that matters now?

For the 64k preset:

```text
same-document causal horizon (~64k)
              │
              ├── current continuous 4k
              │         │
              │         └── router query
              │
              └── searchable history (~60k)
                        │
                  128-token blocks
                        │
                  router scores
                        │
                     top-16
                        │
              chronological ordering
                        │
             up to 2,048 exact old tokens
                        │
                        ▼
             [routed history][current 4k]
                        │
                        ▼
                all 12 Transformer layers
                        │
                        ▼
               LM loss on current tokens
```

### Important properties

**The retrieved history is made of exact historical tokens.** The Transformer does not receive block scores or router rank as an information channel.

**Historical ordering is preserved.** Selected blocks are sorted by their original location before being reopened.

**True positions are preserved.** RoPE uses the original historical positions rather than pretending the sparse reconstruction was contiguous in the source document.

**The router is trained from the LM objective.** There is no externally labeled relevance target. A block is useful if selecting it helps reduce language-model loss on the current window.

**Training uses stochastic hard routing.** Gumbel perturbation provides exploration while a straight-through surrogate gives the router a handwritten gradient path.

**Inference uses deterministic routing.** The highest-scoring historical blocks are selected directly.

### Partial history

The router is not forced to fill a 2k budget unnecessarily:

```text
no history              -> ordinary current window
1..2048 history tokens  -> use all available history directly
more history            -> bounded/routed historical prefix
```

If all available history fits into the active prefix, there is no selection problem and therefore no need to train the router on that example.

---

# Long-context inference

Long-context inference does not allocate a 64k deep Transformer KV cache.

Instead it maintains two levels of state:

```text
external history store
    token IDs + cheap router projections
    no deep Transformer K/V

            ↓ deterministic routing

bounded deep KV cache
    [historical prefix | current 4k]
    maximum ~6k active tokens
```

The current working window is decoded normally through the KV cache. When the configured route-refresh boundary is reached, the router is evaluated again and the bounded active cache is rebuilt.

Useful controls include:

```bash
MINI_LLM_ROUTED_PREFIX_REFRESH_TOKENS=128
MINI_LLM_ROUTED_PREFIX_PREFILL_CHUNK=4096
```

A refresh interval of `1` is useful as a strict reference mode. A larger interval amortizes the cost of rebuilding the active context.

---

# Why NumPy/CuPy instead of PyTorch?

This repository is intentionally not trying to compete with PyTorch as a general deep-learning framework.

The explicit implementation is useful because architectural experiments often require changing exactly the operations normally hidden inside framework primitives. Examples already explored in this project include:

- custom attention layouts;
- explicit routing plans;
- fused BF16 kernels;
- memory-aware inference layouts;
- optimizer-state offload;
- manual recomputation strategies;
- custom gradient estimators for discrete context selection.

The NumPy backend provides a readable reference implementation. The CuPy backend makes the same architecture practical on NVIDIA GPUs.

---

# Repository layout

```text
mini_llm/
├── blocks/          Transformer and MoE blocks
├── checkpoint/      checkpoint save/load infrastructure
├── data/            token shards, packed data, corpus readers
├── diagnostics/     gradient/numerical debugging tools
├── model/           decoder language model
├── ops/             attention, routing, memory, kernels, norms, loss
├── optim/           AdamW, schedules, gradient clipping
└── tokenizer/       custom tokenizer implementations

train_model.py        main pretraining entry point
inference.py          autoregressive inference
benchmarks/           focused performance benchmarks
tests/                reference and regression tests
```

Important design/history documents include:

- `EXTENDED_TRAINING.md`
- `HIERARCHICAL_MEMORY_0058.md`
- `PATCH_0060A_ROUTED_PREFIX.md`
- `PATCH_0060B_ROUTED_PREFIX_INFERENCE.md`
- `FUTURE_DEVELOPMENT.md`
- `USER_GUIDE.md` — launch, resume, inference, runtime profiles, and complete tuning/override reference

The patch/history documents describe how specific mechanisms evolved; this README describes the current project direction.

---

# Installation and dependencies

## System requirements

The numerical reference backend runs on CPU through NumPy. Practical training of the current wide models requires an NVIDIA CUDA GPU.

Recommended baseline:

- Python 3.10 or newer;
- Linux for the main development/training workflow;
- an NVIDIA GPU and a CUDA installation compatible with the selected CuPy package for GPU execution;
- sufficient system RAM and disk space for optimizer offload, token shards, logs, and checkpoints.

The current Python dependencies declared by the project are:

```text
numpy>=1.24
ml-dtypes>=0.5
pyarrow>=16.0
tokenizers>=0.20

# optional GPU dependency
cupy-cuda13x>=14.0

# development/test dependency
pytest>=8.0
```

Create and activate a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CPU/reference development:

```bash
pip install -e ".[dev]"
export MINI_LLM_BACKEND=numpy
pytest -q
```

GPU development:

```bash
pip install -e ".[gpu,dev]"
export MINI_LLM_BACKEND=cupy
```

The `gpu` extra currently installs the CUDA-13 CuPy package used by the main development environment. If your CUDA installation differs, install the appropriate CuPy build for that environment instead.

---

# Data pipeline

The current training pipeline supports a mixed pretraining corpus stored as packed token shards. The active research mixture has included sources such as:

- GitHub/code;
- FineWeb;
- FineWeb-Edu;
- Cosmopedia-v2;
- Wikipedia;
- scientific PDFs;
- PES2O.

The exact mixture is a training decision, not part of the model architecture.

The repository contains utilities for:

- training/loading the BPE tokenizer;
- generating packed token shards;
- multiprocessing tokenization;
- document-aware EOS-bounded sampling;
- mixed-corpus weighted sampling;
- validation splits.

See `TOKENIZER_README.md`, `PREPARING_FOR_COSMOPEDIA.md`, and `EXTENDED_TRAINING.md` for pipeline details.

---

# First launch

Training data is not bundled with the repository. Before the first training launch, prepare packed token shards as described in the [Data pipeline](#data-pipeline) section and the associated preprocessing documents. The example below assumes the current mixed-tokenized corpus exists at `./token_shards_pretraining_65280`.

Create output directories once:

```bash
mkdir -p checkpoints logs
```

A short GPU smoke run of the current 64k routed-prefix model is:

```bash
MINI_LLM_BACKEND=cupy \
python train_model.py \
    --model wide-500m-memory-64k \
    --precision bf16-mixed \
    --mixed-shard-root ./token_shards_pretraining_65280 \
    --batch-size 1 \
    --grad-accum-steps 4 \
    --total-steps 10 \
    --warmup-steps 2 \
    --peak-lr 3e-4 \
    --grad-clip 1.0 \
    --weight-decay 0.1 \
    --val-interval 10 \
    --val-steps 2 \
    --save-interval 10 \
    --log-interval 1 \
    --checkpoint-dir ./checkpoints/wide-500m-memory-64k_smoke \
    --log-file ./logs/wide-500m-memory-64k_smoke.csv
```

This is a correctness/installation check, not a recommended full-training schedule. On CuPy, the default `auto` runtime profile enables the validated fast paths; large local training models additionally receive the memory-saving `consumer-gpu` defaults. Explicit `MINI_LLM_*` overrides still take precedence. See `USER_GUIDE.md` for the full behavior and tuning reference.

After the checkpoint is written, a first inference run is:

```bash
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint ./checkpoints/wide-500m-memory-64k_smoke \
    --prompt "The purpose of numerical simulation is" \
    --max-new-tokens 64 \
    --backend kv-cache \
    --strategy top-p \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42
```

A short prompt verifies the inference path but does not exercise long-history routing. To test routed-prefix memory, provide a prompt longer than the 4k working window, preferably through `--prompt-file`:

```bash
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint ./checkpoints/wide-500m-memory-64k_smoke \
    --prompt-file ./long_prompt.txt \
    --max-new-tokens 128 \
    --backend kv-cache
```

---

# Training

The trainer supports:

- FP32, FP16, and BF16-mixed execution;
- gradient accumulation;
- warmup and cosine LR scheduling;
- gradient clipping;
- checkpoint/resume;
- validation;
- full optimizer-state offload;
- chunked LM-head computation;
- routed-prefix joint training;
- router diagnostics;
- optional profiling and memory diagnostics.

Normal launches are intentionally concise. Hardware/runtime policy is supplied by `--runtime-profile auto`, while every low-level experimental switch remains available as an explicit override. See `USER_GUIDE.md` for runtime profiles, the full `MINI_LLM_*` reference, memory-tuning recipes, and resume/inference examples.

For a quick architecture smoke test, reduce total steps and gradient accumulation rather than changing the model preset.

---

# Inference

Basic generation:

```bash
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint ./checkpoints/<checkpoint> \
    --prompt "The problem can be approached by" \
    --max-new-tokens 128 \
    --backend kv-cache
```

Long routed-prefix checkpoints can use the full trained addressable horizon while keeping only the bounded active context in the deep KV cache:

```bash
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint ./checkpoints/<64k-checkpoint> \
    --prompt-file ./long_prompt.txt \
    --max-new-tokens 512 \
    --backend kv-cache
```

---

# Evaluation strategy

A small model should not be judged by loss alone. Current and planned evaluation includes:

1. held-out language-model loss/perplexity;
2. WikiText-2 evaluation;
3. fixed inference prompts for qualitative regression testing;
4. code and scientific-text completion;
5. long-context synthetic dependency tests;
6. router inspection and retrieval-distance statistics;
7. routed-context ablations.

For the context router, particularly important comparisons are:

```text
current 4k only
vs.
recent historical prefix + current 4k
vs.
random historical prefix + current 4k
vs.
learned routed historical prefix + current 4k
```

The objective is to demonstrate that learned sparse context provides value beyond simply giving the model more tokens.

---

# What this project is — and is not

This project **is**:

- a local LLM research platform;
- a transparent implementation of modern LLM components;
- a place to prototype architectures that are difficult to express cleanly in standard stacks;
- a testbed for compute- and memory-efficient model design;
- an attempt to understand how far careful architecture and implementation can push consumer hardware.

This project is **not** currently:

- a production inference server;
- a drop-in replacement for PyTorch/Transformers;
- a claim that a 500M model can replace frontier models;
- a project whose primary strategy is to scale parameter count as quickly as possible.

Scaling remains useful, but it should follow understanding rather than replace it.

---

# Research direction

The current routed-prefix experiment is only one point in a larger design space. Future work includes better router representations, stronger evaluation, residual sparse access to history, memory representations, training-memory optimizations, larger models when justified, and longer addressable horizons.

These ideas are tracked separately in:

**[`FUTURE_DEVELOPMENT.md`](FUTURE_DEVELOPMENT.md)**

That document deliberately separates implemented behavior from hypotheses and future experiments.

---

# Guiding question

The long-term question behind this repository is intentionally broader than any single model:

> **Given a fixed amount of local hardware, how much capability can we obtain by improving architecture, numerical implementation, context use, training strategy, and data — before resorting to scale?**

That constraint is not merely something to work around. It is one of the main reasons to build the project.
---

# License

This project is licensed under the **GNU Lesser General Public License, Version 3 (LGPL-3.0)**.

The full license text is provided in [`LICENSE`](LICENSE). Any use, modification, redistribution, or combination with other software must comply with the terms of the LGPL-3.0.
