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

The current reference training effort is the **338M-parameter, 20-layer, 1024-wide, dense FFN-4096 historical-hybrid model**. Its compact `historical_read` mechanism provides access to prior context without evaluating a full 64k-token dense Transformer. Research also explores independently trained specialists, model combination, and block proposals. The earlier ~500M routed-prefix architecture remains documented below as a distinct, valuable experiment; it is **not** the reference model currently being trained.

---

# Current reference model: 338M historical-hybrid Transformer

The main reference training configuration is `ensemble-1024-20l-dense-historical-hybrid-64k`, corresponding to `ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()`. Its architecture is:

| Component | Configuration |
|---|---:|
| Parameters | 338,134,308 |
| Transformer layers | 20 |
| Model width | 1,024 |
| Dense FFN width | 4,096 |
| Query heads | 16 |
| KV heads | 4 |
| Head dimension | 64 |
| FFN design | Dense SwiGLU (no MoE) |
| Tokenizer vocabulary | 32,768 |
| Current causal working window | 4,096 tokens |
| Addressable causal horizon | 65,536 tokens |
| Earlier searchable history | up to 61,440 tokens |
| Historical integration | `historical_read` (read-only causal history) |
| Historical retrieval | 32-token blocks, up to 8 selected blocks per configured read |
| Precision in the reference training workflow | BF16 mixed precision |

**Attention design.** Each of the 20 layers reserves one full 4,096-token causal head for the current working window. Other current-window heads use causal sliding windows of 512 tokens unless assigned to the historical reader according to the layer/head schedule. The historical reader accesses selected earlier context through the dedicated memory system; it does **not** create the 2k routed prefix used by the previous 500M model. The expensive Transformer current-window sequence remains 4,096 tokens rather than a dense 65,536-token sequence.

The current model is being trained with the CuPy backend and the mixed-shard pretraining pipeline. Its production-oriented training recipe, evaluation, native inference, external-model inference, and serving commands are in [Current pipeline capabilities](#current-pipeline-capabilities--practical-end-to-end-workflows), especially [Step 4B](#4b-338m-historical-hybrid-model-production-oriented-launch). The runtime tuning values are **run settings, not model architecture**; preserve the tokenizer/checkpoint pairing on resume.

---

# Earlier research model: ~500M routed-prefix Transformer

The previous reference research family was `wide-500m-*`. It remains available for experiments on sparse attention and routed-prefix memory, but should not be described as the currently trained reference model.

The earlier 64k routed-memory preset is:

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

Each layer of the earlier `wide-500m` model uses 12 query heads with complementary roles:

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

# Earlier routed sparse context (500M research)

The earlier routed-prefix experiment asks a simple question:

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

# Routed-prefix inference in the earlier 500M model

The following describes the *earlier* `routed_prefix` design, not the 338M reference model's `historical_read` execution path. In this routed-prefix mode, long-context inference does not allocate a 64k deep Transformer KV cache.

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
tests/                reference and regression tests (see tests/README.md)
```

Important documentation includes:

- `docs/USER_GUIDE.md` — launch, resume, inference, runtime profiles, and tuning reference
- `docs/EXTENDED_TRAINING.md` — extended training workflow
- `docs/research/HIERARCHICAL_MEMORY_0058.md` — routed-memory design background
- `docs/FUTURE_DEVELOPMENT.md` — planned research directions
- `docs/README.md` — documentation and operational-entry-point index
- `docs/research/` — active predictive and probability-combination research reports

Research reports describe active experimental directions; this README describes the current project direction.

---

# Installation and dependencies

## System requirements

The numerical reference backend runs on CPU through NumPy. Practical training of the current 338M reference model requires an NVIDIA CUDA GPU.

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
pytest -q -m "unit and not gpu and not slow"
```

The test suite is categorized with `unit`, `integration`, `gpu`, and `slow`
markers. See `tests/README.md` for focused commands and the safe testing workflow
when a training process is already using the GPU.

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

See `docs/TOKENIZER_README.md` and `docs/EXTENDED_TRAINING.md` for pipeline details.

---

# First launch

Training data is not bundled with the repository. The active 338M model uses an **existing 32,768-token tokenizer and matching packed shards**. For a brand-new run, follow [Current pipeline capabilities](#current-pipeline-capabilities--practical-end-to-end-workflows) to download example source data (Cosmopedia or another corpus), create or select a tokenizer, pack held-out train/validation shards, and decide where checkpoints should live. **Do not retrain or substitute the tokenizer for an existing checkpoint.**

For a short GPU **smoke test of the current 338M reference architecture**, first set paths to directories you chose. This deliberately does not prescribe a personal directory layout:

```bash
export MODEL_SHARDS="/path/to/your/compatible_32k_mixture_shards"
export RUN_DIR="/path/to/your/test_run"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs"

MINI_LLM_BACKEND=cupy \
python -u train_model.py \
    --model ensemble-1024-20l-dense-historical-hybrid-64k \
    --objective ar --precision bf16-mixed \
    --runtime-profile consumer-gpu \
    --mixed-shard-root "$MODEL_SHARDS" \
    --batch-size 1 \
    --grad-accum-steps 2 \
    --gradient-accumulation-weighting supervised_tokens \
    --optimizer hybrid-muon \
    --total-steps 10 \
    --warmup-steps 2 \
    --peak-lr 2.5e-4 \
    --grad-clip 1.0 \
    --weight-decay 0.1 \
    --val-interval 10 \
    --val-steps 2 \
    --save-interval 10 \
    --log-interval 1 \
    --checkpoint-dir "$RUN_DIR/checkpoints/338m_smoke" \
    --log-file "$RUN_DIR/logs/338m_smoke.csv"
```

Replace both example paths before running. This is a correctness and installation check, **not** a recommended training schedule or a claim that the model will learn useful behavior in ten steps. The explicit `consumer-gpu` runtime profile enables the documented optimized training paths; some newer historical kernels are numerical opt-ins. For the full production-oriented BF16 training command and performance caveats, see [Step 4B](#4b-338m-historical-hybrid-model-production-oriented-launch) in the pipeline guide below.

Once the smoke checkpoint has been written, test native generation:

```bash
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint "$RUN_DIR/checkpoints/338m_smoke" \
    --prompt "The purpose of numerical simulation is" \
    --max-new-tokens 64 \
    --backend kv-cache \
    --strategy top-p \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42
```

A short prompt verifies the inference code path but not historical memory. To test the 338M model's historical-reader path, supply a same-document prompt longer than its 4,096-token working window (for example, with `--prompt-file`):

```bash
export LONG_PROMPT_FILE="/path/to/your/long_prompt.txt"
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint "$RUN_DIR/checkpoints/338m_smoke" \
    --prompt-file "$LONG_PROMPT_FILE" \
    --max-new-tokens 128 \
    --backend kv-cache
```

The earlier 500M `routed_prefix` model remains available through its own `wide-500m-*` presets; its routing behavior and refresh controls are described in the historical sections above. Do not assume the two memory modes have interchangeable inference flags.

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
- historical-read training for the current 338M family;
- routed-prefix joint training for the earlier 500M family;
- router and historical-memory diagnostics;
- optional profiling and memory diagnostics.

Normal launches are intentionally concise. Hardware/runtime policy is supplied by `--runtime-profile auto`, while every low-level experimental switch remains available as an explicit override. See `docs/USER_GUIDE.md` for runtime profiles, the full `MINI_LLM_*` reference, memory-tuning recipes, and resume/inference examples.

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

For the earlier **`routed_prefix`** checkpoints, the trained addressable horizon can be used with a bounded active context in the deep KV cache. For the current 338M **`historical_read`** model, see the native-generation and server commands in the pipeline section below; the memory modes must not be conflated:

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

For the current 338M `historical_read` family, prioritize held-out cross-entropy, supervised-token efficiency, generation quality, historical-reader ablations, and comparisons to the same checkpoint with history disabled where supported. For the **earlier routed-prefix context router**, particularly important comparisons were:

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
- a claim that the 338M reference model (or the earlier 500M experiment) can replace frontier models;
- a project whose primary strategy is to scale parameter count as quickly as possible.

Scaling remains useful, but it should follow understanding rather than replace it.

---

# Research direction

The current 338M dense historical-read model is the main ongoing foundation-training reference. Active research directions also include independently trainable residual specialists (with configurable historical memory), learned specialist combination, probability fusion, and continuous block proposals. The earlier 500M routed-prefix experiment remains one useful point in this larger design space. Future work includes stronger evaluation, improved historical retrieval, memory representations, training-memory optimizations, larger models when justified, and longer addressable horizons.

These ideas are tracked separately in:

**[`docs/FUTURE_DEVELOPMENT.md`](docs/FUTURE_DEVELOPMENT.md)**

That document deliberately separates implemented behavior from hypotheses and future experiments.

## Performance TODO: startup attention autotuning

Replace the calibrated local-attention chunk work budget with an optional startup
benchmark that measures forward and backward candidates on the active GPU. Cache
the winning chunk by device architecture, precision, batch size, sequence length,
window, active query-head count, head dimension, and relevant kernel flags. The
autotuner must enforce the configured VRAM limit, avoid mutating model or optimizer
state, and fall back to the current geometry-based policy when tuning is disabled or
no cached result is available.

---

# Operational TODO: runtime directory migration

After the current training run has completed, consolidate generated runtime
artifacts under a single `runtime/` hierarchy:

```text
runtime/
├── checkpoints/
├── logs/
├── caches/
├── tokenizers/
└── shards/
```

Do **not** move runtime paths while training or inference is active. Before this
migration:

1. confirm that no model process is using the repository;
2. verify the final checkpoint and its `training_state.json`;
3. checksum or back up the final checkpoint;
4. inventory active versus obsolete runs;
5. update path defaults and documentation before moving data;
6. migrate one category at a time and validate checkpoint loading/resume; and
7. delete old copies only after the new locations are verified.

Until then, `checkpoints/`, `logs/`, and active token-shard directories must
remain at their current paths.

---

# Guiding question

The long-term question behind this repository is intentionally broader than any single model:

> **Given a fixed amount of local hardware, how much capability can we obtain by improving architecture, numerical implementation, context use, training strategy, and data — before resorting to scale?**

That constraint is not merely something to work around. It is one of the main reasons to build the project.
---

# Current pipeline capabilities — practical end-to-end workflows

> **Additive operations guide (0090C, October 2026).** This section supplements rather than replaces the project philosophy, architecture discussion, original model examples, and research roadmap above. The **current reference is the 338M 20L × 1024 × FFN4096 historical-read model**; the original `wide-500m` routed-prefix family remains documented as an earlier architecture experiment. The pipeline also supports other native checkpoints and compatible external Llama-family inference. Some modes remain experimental, and the server is not a hardened public production service.
>
> All commands below are examples for a **new** run, not instructions to overwrite existing checkpoints or move an active data layout. Choose your own paths. The Cosmopedia source is illustrative, not a prescribed dataset or corpus mixture.

## Path policy and optional experiment layout

**No user data paths are prescribed by this repository.** The following is a **recommended, optional layout for a new experiment**. All later commands use these shell variables; change `RUN_ROOT` to any directory you control (including a mounted disk), or set every subpath independently. This does **not** move or rename paths used by existing training runs. Run the commands from the repository root.

```text
<your chosen RUN_ROOT>/
├── raw/                     raw downloaded datasets (e.g. cosmopedia-v2/)
├── shards32k/               tokenizer.json, manifest, packed train/val shards
├── mixture_shards32k/       optional multi-source mixture (independent of pilot)
├── standalone_shards/       optional single-source packer output
├── checkpoints/             native checkpoints
├── logs/                    training logs
├── benchmarks/              held-out evaluation and generated samples
├── external_models/         optional GGUF / Safetensors models
└── serving_models.json      optional OpenAI-compatible serving registry
```

```bash
# First, choose where THIS experiment should live. Change it before running.
export PROJECT_ROOT="$PWD"    # current directory must contain train_model.py
export RUN_ROOT="${RUN_ROOT:-$HOME/MyLocal_LLM_experiments/cosmopedia_pilot}"
export DATA_ROOT="${DATA_ROOT:-$RUN_ROOT/raw}"
export SHARD_ROOT="${SHARD_ROOT:-$RUN_ROOT/shards32k}"
export MIX_SHARD_ROOT="${MIX_SHARD_ROOT:-$RUN_ROOT/mixture_shards32k}"
export STANDALONE_SHARD_ROOT="${STANDALONE_SHARD_ROOT:-$RUN_ROOT/standalone_shards}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints}"
export LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
export BENCH_ROOT="${BENCH_ROOT:-$RUN_ROOT/benchmarks}"
export MODEL_STORE="${MODEL_STORE:-$RUN_ROOT/external_models}"
export REGISTRY_PATH="${REGISTRY_PATH:-$RUN_ROOT/serving_models.json}"
export COSMO_CONFIG="$PROJECT_ROOT/configs/cosmopedia_quickstart.json"
export MIX_CONFIG_MULTI="$PROJECT_ROOT/configs/pretraining_coding_mix.json"
export SMOKE_CKPT="$CHECKPOINT_ROOT/cosmopedia_mini_smoke"
export NATIVE_CKPT="$CHECKPOINT_ROOT/ensemble_338m_new"
export SMOKE_LOG="$LOG_DIR/cosmopedia_mini_smoke.csv"
export NATIVE_LOG="$LOG_DIR/ensemble_338m_new.csv"
export WIKITEXT_TEST="$BENCH_ROOT/wiki.test.raw"   # supply a real held-out file
export HELDOUT_TEXT="$BENCH_ROOT/sample_eval.txt" # supply a real held-out file
export GGUF_PATH="$MODEL_STORE/SmolLM2-360M-Instruct-F16.gguf"  # supply model file
mkdir -p "$DATA_ROOT" "$CHECKPOINT_ROOT" "$LOG_DIR" "$BENCH_ROOT" "$MODEL_STORE"
```

The suggested default is outside the Git checkout to avoid filling the source tree with training data. `RUN_ROOT` is a convenience, **not a project requirement**. If you already maintain your own paths, set `DATA_ROOT`, `SHARD_ROOT`, `NATIVE_CKPT`, `GGUF_PATH`, etc. to those existing absolute paths. Keep the variables in the same terminal, or persist them in your own environment configuration. The dataset configuration's `relative_path` refers to a subdirectory *inside* `DATA_ROOT`; this is required by the mixture packer's source-mapping contract, not an assumption about your home directory.

The examples below use **Cosmopedia-v2 only to demonstrate the process**. You can use other supported Parquet/JSONL/text datasets by adjusting the mixture configuration (`sources`, `relative_path`, and `text_column`) and supplying the corresponding files. The example is not the project's prescribed production mixture.

## Step 0 — Install and check the runtime

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,dev,streaming,external,triton-attention]'
# Optional: check that the CUDA driver and CuPy agree with your system.
nvidia-smi
MINI_LLM_BACKEND=cupy python -c 'from mini_llm.backend import BACKEND_NAME; print(BACKEND_NAME)'
```

The optional Triton dependency is needed only for Triton-based optimizations; remove `triton-attention` if not compatible with your environment. `cupy-cuda13x` in `pyproject.toml` targets CUDA 13.x; install the CuPy wheel matching your CUDA runtime. `MINI_LLM_BACKEND` **must be set before Python starts**; it cannot be changed inside an already imported process. CPU correctness checks can use `MINI_LLM_BACKEND=numpy`.

**Profiles** (`--runtime-profile` for `train_model.py`/`inference.py`, or `MINI_LLM_RUNTIME_PROFILE` where supported): `reference` (no defaults), `gpu-fast` (optimized kernels), `consumer-gpu` (GPU fast paths plus CPU optimizer-state offload and chunked LM head during training), and `auto` (default: NumPy reference; CuPy inference gpu-fast; large CuPy training consumer-gpu). **All explicit `MINI_LLM_*` environment overrides take priority.** Note that the server does not currently have a `--runtime-profile` argument: configure server fast paths using environment variables as shown below.

## Step 1 — Download Cosmopedia-v2 raw Parquet

Cosmopedia-v2 is stored inside the Hugging Face [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/cosmopedia-v2) repository `HuggingFaceTB/smollm-corpus`, in its `cosmopedia-v2/` directory. The following downloads **one source Parquet file** (already about 1.2 GB), suitable for a first end-to-end run, into the *optional example* `DATA_ROOT` layout described above:

```bash
mkdir -p "$DATA_ROOT"
hf download HuggingFaceTB/smollm-corpus \
    cosmopedia-v2/train-00000-of-00104.parquet \
    --repo-type dataset --local-dir "$DATA_ROOT"
ls -lh "$DATA_ROOT/cosmopedia-v2"/*.parquet
```

To obtain the complete corpus (substantially larger; check disk budget and dataset access requirements):

```bash
hf download HuggingFaceTB/smollm-corpus \
    --repo-type dataset --include 'cosmopedia-v2/*.parquet' \
    --local-dir "$DATA_ROOT"
```

`hf` is provided by `huggingface_hub` (included in the `external` extra). For reproducibility, record the dataset repository revision (and preferably add `--revision <commit>` to your real run). Inspect the license and disk footprint before downloading. Other sources may be Parquet, JSON(L), text or supported code files; see `mini_llm/data/text_corpus.py`.

## Step 2 — Create a shared tokenizer (training documents only)

**Recommended:** use the `prepare_ensemble_mixture.py` pipeline for both a single source and a full mixture. It creates a deterministic document-hash train/validation split, trains the tokenizer **only from training documents**, and packs every complete document with an EOS boundary. It does **not** cut documents at 4,096 tokens. Do **not** use legacy `generate_token_shards.py` for new long-context experiments: that older path truncates texts to a configured context length.

For a reproducible Cosmopedia-only example, this repository includes [`configs/cosmopedia_quickstart.json`](configs/cosmopedia_quickstart.json):

```json
{
  "name": "cosmopedia_quickstart",
  "sources": {
    "cosmopedia_v2": {
      "relative_path": "cosmopedia-v2",
      "text_column": "text",
      "tokenizer_weight": 1.0,
      "pretraining_weight": 1.0
    }
  }
}
```

Train a 32,768-token BPE for this *new* experiment (increase `--sample-documents` if the tokenizer reports insufficient documents). For the single-Parquet pilot, 10,000 source documents may be enough to exercise the pipeline; a larger, diverse training-only tokenizer sample is preferable for a real model:

```bash
RAYON_NUM_THREADS=16 python prepare_ensemble_mixture.py tokenizer \
    --data-root "$DATA_ROOT" \
    --mix-config "$COSMO_CONFIG" \
    --output-root "$SHARD_ROOT" \
    --vocab-size 32768 --sample-documents 10000 --threads 16 \
    --seed 42 --val-ratio 0.01
```

Outputs: `$SHARD_ROOT/tokenizer.json` and `tokenizer_training.json`. **A trained checkpoint and its shards must use the exact same tokenizer.** In particular, the `ensemble-*` model presets require a **32,768-token** vocabulary; you cannot swap their tokenizer for another 32k tokenizer when resuming the same checkpoint.

**Existing production 32k tokenizer:** to continue training an existing model, reuse the immutable tokenizer that produced its current shards/checkpoint. Do not retrain it from scratch; copy it to a *new* mixture output root only when building a compatible dataset, and verify its SHA256 and special-token IDs. Keep the original checkpoint and shard roots unchanged while a training process is running.

## Step 3 — Tokenize full documents and create train/validation shards

```bash
python prepare_ensemble_mixture.py shards \
    --data-root "$DATA_ROOT" \
    --mix-config "$COSMO_CONFIG" \
    --output-root "$SHARD_ROOT" \
    --workers 8 --batch-documents 256 --shard-size-mb 256 \
    --seed 42 --val-ratio 0.01

python -m json.tool "$SHARD_ROOT/mixture_manifest.json" | head -65
find "$SHARD_ROOT" -name 'train_shard_*.bin' | head
find "$SHARD_ROOT" -name 'val_shard_*.bin' | head
```

The shards are contiguous `uint16` token IDs with per-document EOS, *not* padded examples. The manifest specifies independently weighted sources, tokenizer hash, source paths, and held-out split. `train_model.py --mixed-shard-root` consumes this manifest. If preprocessing is interrupted, use the same arguments to resume its partial manifest; a complete output root refuses an overwrite. Reweighting an existing **complete multi-source** mixture can use `prepare_ensemble_mixture.py reweight` with a compatible source set and a changed configuration, without re-encoding all files.

**Alternative standalone Cosmopedia packer** when you already have a tokenizer:

```bash
python generate_packed_cosmopedia_shards.py \
    --dataset-path "$DATA_ROOT/cosmopedia-v2" \
    --tokenizer "$SHARD_ROOT/tokenizer.json" \
    --output-dir "$STANDALONE_SHARD_ROOT" \
    --num-shards 1 --val-ratio 0.01 --shard-size-mb 256
# For this standalone shard directory, supply --shard-dir and --tokenizer-path
# to train_model.py, not --mixed-shard-root.
```

Do not mix a standalone directory with the weighted manifest format. Choose one preprocessing route per experiment.

### A multi-corpus training mixture (separate example)

The example `configs/pretraining_coding_mix.json` refers to source directories **relative to** `DATA_ROOT` for GitHub, FineWeb, Cosmopedia, etc. **Download/populate every configured relative source path beneath your selected `DATA_ROOT` first** (or create a smaller config with only the sources actually present). Then run:

```bash
RAYON_NUM_THREADS=16 python prepare_ensemble_mixture.py tokenizer \
    --data-root "$DATA_ROOT" \
    --mix-config "$MIX_CONFIG_MULTI" \
    --output-root "$MIX_SHARD_ROOT" \
    --vocab-size 32768 --sample-documents 200000 --threads 16

python prepare_ensemble_mixture.py shards \
    --data-root "$DATA_ROOT" \
    --mix-config "$MIX_CONFIG_MULTI" \
    --output-root "$MIX_SHARD_ROOT" \
    --workers 8 --batch-documents 256 --shard-size-mb 256
```

The bundled mixture config is **an example and may not match your latest custom source weights**. For a continued run, retain the actual manifest/tokenizer that belongs to that run. `--source <name>` can limit the pack operation to selected sources; it leaves an intentionally incomplete mixture until the other configured sources are packed.

## Step 4 — Train a model from scratch

### 4A. Short GPU smoke test (small model, new shards)

```bash
mkdir -p "$CHECKPOINT_ROOT" "$LOG_DIR"
MINI_LLM_BACKEND=cupy python -u train_model.py \
    --model mini --objective ar --precision bf16-mixed \
    --runtime-profile gpu-fast \
    --mixed-shard-root "$SHARD_ROOT" \
    --context-length 512 --batch-size 2 --grad-accum-steps 2 \
    --gradient-accumulation-weighting supervised_tokens \
    --optimizer hybrid-muon \
    --total-steps 1000 --max-run-steps 5 --warmup-steps 50 \
    --peak-lr 3e-4 --grad-clip 1.0 --weight-decay 0.1 \
    --val-interval 5 --val-steps 2 --save-interval 5 --log-interval 1 \
    --checkpoint-dir "$SMOKE_CKPT" \
    --log-file "$SMOKE_LOG"
```

`--max-run-steps 5` executes just five optimizer updates while retaining the requested 1,000-step learning-rate horizon. Remove it for a real run. For meaningful training, allocate a realistic token budget and increase validation/intervals appropriately.

### 4B. 338M historical-hybrid model, production-oriented launch

This preset implements **20 layers × width 1024 × FFN 4096**, vocabulary 32,768, 4k current window and bounded historical reads. The command uses `MIX_SHARD_ROOT` from the optional **multi-source** packing step above; if experimenting on the single-source pilot, explicitly substitute `SHARD_ROOT` (while acknowledging it is not a representative general-purpose pretraining mixture). Use a **32k tokenizer-compatible shard root** with enough data and document history. This is a starting configuration for the RTX 5080 (16 GiB), not a universal performance optimum:

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=auto \
MINI_LLM_HISTORY_QUERY_CHUNK=1024 \
MINI_LLM_HISTORY_RECOMPUTE=1 \
MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=1 \
MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1 \
MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM=1 \
MINI_LLM_ROPE_PREPARED_POSITIONS=1 \
MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1 \
MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1 \
MINI_LLM_AR_ACCUM_LENGTH_BUCKETS=1 \
python -u train_model.py \
    --model ensemble-1024-20l-dense-historical-hybrid-64k \
    --objective ar --precision bf16-mixed \
    --runtime-profile consumer-gpu \
    --mixed-shard-root "$MIX_SHARD_ROOT" \
    --batch-size 3 --grad-accum-steps 128 \
    --gradient-accumulation-weighting supervised_tokens \
    --optimizer hybrid-muon --muon-momentum 0.95 \
    --muon-ns-steps 5 --muon-matmul-precision bf16 \
    --total-steps 100000 --warmup-steps 750 --peak-lr 2.5e-4 \
    --grad-clip 1.0 --weight-decay 0.1 \
    --val-interval 500 --val-steps 10 --val-per-source-steps 2 \
    --save-interval 2000 --log-interval 10 \
    --checkpoint-dir "$NATIVE_CKPT" \
    --log-file "$NATIVE_LOG"
```

The **explicit** `consumer-gpu` profile in this patch now sets the 0089M prepared-RoPE/grouped-cuBLAS flags, 0089M autoregressive length buckets, and for the **validated 20L/1024/BF16 historical-read geometry only**, history chunk=1024, history recomputation, local probability recomputation, and the Triton block forward/prewarm flags. The explicit variables above are shown for readability and cross-version launch reproducibility; they may be removed when running this patched version. `--runtime-profile auto` **does not silently enable the numerically non-bitwise Triton historical forward**, protecting old auto-profile checkpoint resumption. All other fast paths (packed QKV, fused QKV/RoPE, vectorized SwiGLU, BF16 pipelines, grouped GEMMs, fused CE, residual RMSNorm) are supplied by the profiles.

**Numerical caveat:** the 0089T historical-block Triton implementation passed local operation/indexing audits but produced non-bitwise differences in full-model BF16 activations/gradients; see [0089T acceptance audit](docs/0089T_HISTORICAL_BLOCK_ACCEPTANCE_AUDIT.md). Its throughput benefit was user-observed (~25k to slightly over 26k supervised tokens/s), **not** a controlled production benchmark. For exact previous-path behavior, use `MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=0` even when selecting `consumer-gpu` explicitly. First verify validation continuity on a copied checkpoint. Set `MINI_LLM_HISTORY_QUERY_CHUNK=512` if the 1024-row configuration exceeds GPU memory. Avoid changing optimizer and numerical kernel policies in the middle of an experiment without recording a new phase.

The runtime profile does **not** universally guarantee maximum throughput: measure supervised tokens/s and VRAM at fixed geometry. Checkpoint, validation, gradient accumulation, historical occupancy and memory pressure change effective throughput. `--profile-steps 1` is a diagnostic and introduces synchronization overhead; leave it disabled for steady-state production. Keep `MINI_LLM_LOCAL_DETAIL_PROFILE=0`, `MINI_LLM_MOE_DETAIL_PROFILE=0`, and `MINI_LLM_GPU_MEMORY_AUDIT=0` for ordinary throughput tests. If fragmentation causes pressure, try `MINI_LLM_GPU_MEMORY_TRIM_TRAIN_INTERVAL=5` / `MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION=1`, but assess the extra synchronization overhead rather than assuming it is faster.

**Memory/speed tuning:** `MINI_LLM_OPTIMIZER_OFFLOAD=full` and `MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB=16` are built into `consumer-gpu`; `MINI_LLM_LM_HEAD_CHUNK_TOKENS` is configuration-dependent (for the 20L/1024/32k BF16 historical preset the profile selects 8192; use `1024` when a smaller tile is needed for VRAM). Smaller attention/history chunks and activation recomputation reduce VRAM but can hurt speed. Microbatch size and gradient accumulation are **different**: increase the former only when it fits; tune the latter for optimization/token efficiency.

### Resume, validate, and compare

```bash
MINI_LLM_BACKEND=cupy python -u train_model.py \
    --resume-from "$NATIVE_CKPT" \
    --mixed-shard-root "$MIX_SHARD_ROOT" \
    --precision bf16-mixed --runtime-profile consumer-gpu \
    --batch-size 3 --grad-accum-steps 128 \
    --gradient-accumulation-weighting supervised_tokens \
    --total-steps 100000 --val-interval 500 --val-steps 10 \
    --checkpoint-dir "$NATIVE_CKPT" \
    --log-file "$NATIVE_LOG"
```

On resume, checkpoint architecture/optimizer state remains authoritative where supported. **Use the same performance environment variables as the original training run** (especially the Triton historical opt-in) to avoid an accidental numerical-path change; the short command above assumes the intended explicit profile. Never overwrite an active checkpoint with experimental training.

## Step 5 — Check model quality (loss and generation)

### 5A. Quantitative language-model evaluation

Evaluate an existing raw WikiText-2 **test** file with the checkpoint tokenizer (fetch the test text separately; it is not bundled here):

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.evaluate_wikitext \
    --checkpoint "$NATIVE_CKPT" \
    --data "$WIKITEXT_TEST" \
    --document-mode article --memory-mode auto \
    --context-length 4096 --stride 4096
```

This reports cross-entropy, perplexity, BPB and evaluated throughput. Use identical document mode, context, stride, tokenizer and scoring policy for comparisons. For a normal `mini` model, replace checkpoint and use its own context length (e.g. `--context-length 512`). WikiText is a **general-language regression test**, not proof of code quality; compare held-out per-source CE from training logs for domain-specific improvements.

For any local text file (or JSONL document file), the generic benchmark accepts `--format text` or `--format jsonl --text-field text`:

```bash
PYTHONPATH=. MINI_LLM_BACKEND=cupy python benchmarks/evaluate_corpus.py \
    --checkpoint "$SMOKE_CKPT" \
    --data "$HELDOUT_TEXT" --format text --context-length 512
```

Create `sample_eval.txt` from **held-out** text first; do not claim meaningful generalization from a five-step smoke run. For `routed_prefix` checkpoints only, the evaluator additionally provides `--memory-mode routed` and `--eval-mode autoregressive` teacher-forced testing. For `historical_read` checkpoints use the documented supported windowed/auto mode; do not assume routed-prefix-only benchmarking modes apply.

### 5B. Native generation: sanity check and qualitative benchmarks

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_NATIVE_CUDA_GRAPH_DECODE=1 \
MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT=0 \
python inference.py \
    --checkpoint "$NATIVE_CKPT" \
    --runtime-profile gpu-fast --backend kv-cache \
    --prompt 'Write a Python function that reverses a list:' \
    --strategy top-k --top-k 40 --temperature 0.8 --max-new-tokens 160
```

**Native CUDA graph decode is a distinct feature** from external GGUF graph decode. It is currently specialized for compatible **native BF16, batch-one, fixed-shape** inference. With `STRICT=0`, unsupported configurations can fall back; turn on `MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT=1` only once capture and output parity work for your *particular checkpoint and context*. Strict mode makes graph unavailability an error rather than silently reverting. Both graph modes optimize **decode**, not the initial variable-length prompt prefill, and may have capture/memory overhead. To compare fairly, benchmark warmed generation tokens/s with graphs on/off and keep the same prompt and sampling.

The project also has a reproducible 24-prompt/seed repetition suite:

```bash
MINI_LLM_BACKEND=cupy python benchmarks/evaluate_generation.py \
    --checkpoint "$NATIVE_CKPT" \
    --output-dir "$BENCH_ROOT/native_generation" \
    --prompt-ids code_python_short,math_general_short \
    --seeds 42 --max-new-tokens 128 --backend kv-cache
```

Outputs include `generations.jsonl`, `summary.json`, and `summary.csv`. Inspect the text as well as repeated-n-gram statistics; these cannot prove code correctness. For robust coding evaluation, execute generated code against isolated unit tests.

## Step 6 — Run external Llama-compatible models (GGUF / Safetensors)

Native `inference.py` accepts **native checkpoints only**. External Llama models use the repository's separate **experimental external loader**, and can be generated directly with `benchmarks.try_external_gguf` or served through the common OpenAI-style API.

Supported GGUF tensor types in this implementation: **F32, F16 and Q8_0** (Q8_0 is **dequantized** at runtime); **Q4_K, Q6_K and arbitrary GGUF model architectures are not supported**. Safetensors support requires a compatible Llama architecture and tokenizer. Do not confuse compatibility here with broader `llama.cpp` format support.

Place a compatible GGUF at the **user-selected** `GGUF_PATH` (or set `GGUF_PATH` to an existing file) and test:

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.try_external_gguf \
    --model "$GGUF_PATH" --inspect

MINI_LLM_BACKEND=cupy \
MINI_LLM_CUDA_GRAPH_DECODE=1 \
MINI_LLM_CUDA_GRAPH_DECODE_STRICT=1 \
python -m benchmarks.try_external_gguf \
    --model "$GGUF_PATH" \
    --chat --prompt 'Explain the finite element method.' \
    --max-new-tokens 128 --dtype float16
```

`MINI_LLM_CUDA_GRAPH_DECODE=1` and `MINI_LLM_CUDA_GRAPH_DECODE_STRICT=1` select the **external FP16 Llama static batch-one CUDA graph**. **Strict mode intentionally errors** when the decoder cannot support the model, e.g. unsupported dimensions, cache shape or dtype. Use `STRICT=0` for fallbacks when mixing model families. A successful strict launch is useful evidence of the graph fast path; validate decoded text against a non-graph control.

The repo also has a held-out GGUF benchmark for WikiText:

```bash
MINI_LLM_BACKEND=cupy python -m benchmarks.evaluate_external_wikitext \
    --model "$GGUF_PATH" \
    --data "$WIKITEXT_TEST" \
    --context-length 1024 --stride 1024 --document-mode article
```

You can also download a supported Llama-compatible **Safetensors** repository directly, without GGUF conversion:

```bash
hf download HuggingFaceTB/SmolLM2-360M-Instruct \
    --local-dir "$MODEL_STORE/SmolLM2-360M-Instruct"
```

To serve the downloaded Safetensors model, add an entry to the generated registry using `format: safetensors`, `dtype: float16`, and an **actual absolute path** to the downloaded model directory (for example `$MODEL_STORE/SmolLM2-360M-Instruct`). The registry is JSON and does *not* expand shell variables inside quoted strings.

The server infers compatible model dimensions from the external weights; Safetensors model folders need the appropriate `config.json`, weights and `tokenizer.json`. **Not every Hugging Face repository is compatible with this loader.** A GGUF file must have a supported architecture and tensor quantization. Check licensing and model-specific chat templates.

## Step 7 — Serve native and external models in one API

Generate a registry using the model paths selected above. This avoids relying on the relative location of either the registry or the shell working directory. Model paths in the registry can also be relative to the registry JSON file if you prefer.

```bash
python - <<'PYREG'
import json, os
from pathlib import Path
registry = {
    "models": [
        {"id": "native-338m", "format": "native", "path": os.environ["NATIVE_CKPT"],
         "chat_template": "plain", "tool_format": "none", "context_length": 64000,
         "sampling": {"temperature": 0.8, "top_k": 40,
                      "repetition_penalty": 1.1}},
        {"id": "smollm2-360m", "format": "gguf", "dtype": "float16",
         "path": os.environ["GGUF_PATH"], "chat_template": "chatml",
         "tool_format": "none", "context_length": 8192,
         "sampling": {"temperature": 0.7, "top_k": 40,
                      "repetition_penalty": 1.05}},
    ]
}
path = Path(os.environ["REGISTRY_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
print("Wrote", path)
PYREG
```

The two referenced models must exist; change the models array to include only those you intend to serve. For external Safetensors models, set `format` to `safetensors` and `path` to the appropriate local directory. `context_length` is an example upper bound, not evidence that every checkpoint supports it.

The generated native checkpoint path is an **example**, not a checkpoint shipped with the repository; set `context_length` to a horizon actually supported and tested by that checkpoint and leave room for the completion. The server lazily loads **one GPU-resident model at a time**; switching model IDs evicts the previous model. The OpenAI-style interface exposes `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`. It does **not** automatically download weights, execute tools, or guarantee an arbitrary GGUF chat template.

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_RUNTIME_PROFILE=gpu-fast \
MINI_LLM_ROPE_PREPARED_POSITIONS=1 \
MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1 \
MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1 \
MINI_LLM_CUDA_GRAPH_DECODE=1 \
MINI_LLM_CUDA_GRAPH_DECODE_STRICT=1 \
MINI_LLM_NATIVE_CUDA_GRAPH_DECODE=1 \
MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT=0 \
MINI_LLM_HISTORICAL_REFRESH_TOKENS=32 \
python -u serve.py --registry "$REGISTRY_PATH" \
    --host 127.0.0.1 --port 8080
```

The **external strict** setting is appropriate only if *every external model you intend to call* supports the graph path. To serve mixed, potentially incompatible external formats, change `MINI_LLM_CUDA_GRAPH_DECODE_STRICT=0`. The separate native strict setting defaults to fallback here because native historical read / refresh may not meet all static graph constraints. For long native history, `MINI_LLM_HISTORICAL_REFRESH_TOKENS=32` is an approximate performance setting; `1` rebuilds on every token for correctness comparison and can be much slower. The graph cannot remove the cost of historical rebase or long-prompt prefill. Explicit environment flags, not the `MINI_LLM_RUNTIME_PROFILE` value, are what make graph behavior verifiable in serving.

In another shell:

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"native-338m","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
curl -Ns http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"smollm2-360m","messages":[{"role":"user","content":"Write a short Python function"}],"max_tokens":128,"stream":true}'
```

The default bind is localhost. To expose the API beyond your machine, configure `--api-key`/`MYLOCAL_LLM_API_KEY`, use an appropriate TLS reverse proxy, and restrict access. For Zed/other IDEs use `http://127.0.0.1:8080/v1` as the OpenAI-compatible base URL; enable model tool capabilities only if tested with that model's actual tool-call format. See [serving protocol documentation](docs/0088A_NATIVE_OPENAI_SERVER.md).

## Step 8 — Production flag reference (what the profile really does)

| Scope | Flag(s) | Why / applicability |
|---|---|---|
| All optimized CuPy models | `MINI_LLM_PACKED_QKV=1`, `MINI_LLM_FUSED_QKV_ROPE_FORWARD=1`, `MINI_LLM_ROPE_PREPARED_POSITIONS=1` | Shared QKV work; reduced RoPE overhead. |
| CUDA grouped kernels | `MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM=1`, `MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1`, `MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1` | Reduce grouped-operation setup overhead. |
| General BF16 fast paths | `MINI_LLM_FUSED_BF16_LOCAL_PIPELINE=1`, `MINI_LLM_FUSED_BF16_INDEXED_PIPELINE=1`, `MINI_LLM_FUSED_BF16_CE=1`, `MINI_LLM_FUSED_BF16_RMSNORM=1`, `MINI_LLM_FUSED_SWIGLU=1`, `MINI_LLM_SWIGLU_VEC2=1` | Existing profiled kernels; supplied by gpu-fast. |
| Training | `MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM=1`, `MINI_LLM_AR_ACCUM_LENGTH_BUCKETS=1` | Measured fused RMSNorm path / accumulation bucketing. |
| 20L/1024/BF16 historical training | `MINI_LLM_HISTORY_QUERY_CHUNK=1024`, `MINI_LLM_HISTORY_RECOMPUTE=1`, `MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=1` | Geometry-specific settings; measure VRAM. |
| 20L/1024/BF16 historical block sweep | `MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1`, `MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM=1` | **Explicit gpu-fast/consumer-gpu opt-in only**; 0089T numerical caveat. |
| Local 16GiB training | `MINI_LLM_OPTIMIZER_OFFLOAD=full`, `MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB=16` | Consumer-gpu training only. |
| Native decoding | `MINI_LLM_NATIVE_CUDA_GRAPH_DECODE=1`, `MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT=1` | Native graph, strict only after supported shape verified. |
| External decoding | `MINI_LLM_CUDA_GRAPH_DECODE=1`, `MINI_LLM_CUDA_GRAPH_DECODE_STRICT=1` | FP16 external Llama graph, strict compatibility check. |

Not every flag applies to every architecture. `MINI_LLM_FUSED_EXPERT_GEMM=0` and `MINI_LLM_CONCURRENT_EXPERTS=0` remain the current profiled defaults; switching them on is **not** universally faster. For the actual effective flags, inspect the startup `Runtime profile:` / `GPU effective fast paths:` lines, record `MINI_LLM_*` settings alongside benchmarks, and do a controlled A/B before declaring a new setting faster.

## Step 9 — Troubleshooting / reproducibility

- **Tokenizer or vocabulary mismatch:** verify manifest `tokenizer_sha256`, checkpoint tokenizer and `--tokenizer-path`; ensemble presets require 32k vocabulary. Never retokenize an existing model's history with a new tokenizer without starting a distinct run.
- **No shards or incomplete manifest:** verify raw download is under `$DATA_ROOT/<relative_path>`, check `--data-root` and `--mix-config`, finish every configured source, and rerun the same packing command on partial output.
- **Out of VRAM on RTX 5080:** reduce `--batch-size` before altering effective accumulation, lower history/current query chunks, use `consumer-gpu`, recompute and smaller `MINI_LLM_LM_HEAD_CHUNK_TOKENS`; observe actual peak VRAM rather than relying on estimates.
- **CUDA Graph strict error:** graph path is model/precision/cache specific; compare with the matching `*_STRICT=0`, then graph disabled. Graph benefit is decode-only; first-run capture and warmup affect timing.
- **Different generation quality or repeating output:** a pretrained base LM may lack instruction following. Experiment with top-k/temperature and repetition penalties in the server registry, but judge quality on held-out or executable tasks; graph flags are not sampling parameters.
- **Long-context latency:** historical access is not free. Check the history refresh interval, current-window length and route occupancy; record both prefill and decode speed.
- **Benchmark fairness:** compare *supervised* tokens/s during training (not source tokens/s), warm GPU kernels, use identical inputs/seeds/sampling and avoid measuring compilation, validation or checkpoint writes as steady-state throughput.

## Further reading

[Extended training](docs/EXTENDED_TRAINING.md) · [Tokenizer guide](docs/TOKENIZER_README.md) · [User guide](docs/USER_GUIDE.md) · [Historical optimization acceptance](docs/0089T_HISTORICAL_BLOCK_ACCEPTANCE_AUDIT.md) · [Native server](docs/0088A_NATIVE_OPENAI_SERVER.md) · [Native historical inference](docs/0088B_HISTORICAL_READ_NATIVE_INFERENCE.md) · [Generation benchmark](benchmarks/README.md) · [Research index](docs/README.md).

---

# License

This project is licensed under the **GNU Lesser General Public License, Version 3 (LGPL-3.0)**.

The full license text is provided in [`LICENSE`](LICENSE). Any use, modification, redistribution, or combination with other software must comply with the terms of the LGPL-3.0.

---

# 0102: dual-resolution long-context attention

0102 adds the opt-in `dual-resolution-1024-20l` research preset: 20 layers, width 1,024, dense SwiGLU-4,096, 16 query heads / 4 KV heads / 64 dimensions, exact causal local-512 token attention, and learned summaries of complete 128-token same-document blocks. Summary visibility is strictly causal (`block_end_position < query_position`); EOS terminates its current document, resets positions/blocks for the following token, and masks the shifted cross-document target during training.

This is currently a **correctness-first, unfused NumPy/CuPy implementation**, not a production-ready GPU claim. It does not allocate dense token `T × T` attention, but handwritten backward retains substantial chunked local/global/compressor probability state. Measure the target hardware before drawing performance conclusions.

- Architecture, defaults, estimator counts (`dc=256/512/1024`), causality/EOS semantics, formulas, caveats, and commands: [`docs/DUAL_RESOLUTION_0102.md`](docs/DUAL_RESOLUTION_0102.md)
- Batch-1 context sweep (`1,024` through `65,536`) with score-workload, timing, memory estimate, tokens/s, and optional native inference metrics: [`benchmarks/benchmark_dual_resolution_0102.py`](benchmarks/benchmark_dual_resolution_0102.py)

```bash
MINI_LLM_BACKEND=numpy python benchmarks/benchmark_dual_resolution_0102.py \
    --validate-only --skip-backward --skip-inference
```
