# 0105 — Model-centered training parity and residual specialist readiness

**Scope:** additive patches after 0104A–G, against merged `MyLocal_LLM-main(20)`.
No Model B parameter schema, sparse-stride attention semantics, optimized
attention implementation, or active foundation training command is changed.

**Status:** The CPU reference implementation and smoke/launch plumbing exist.
Full-model CuPy/Triton GPU parity, full-size 12-layer specialist memory/performance,
long-range accuracy, and an integrated composed-model generation session are **not
validated**. Do not interpret passing CPU tests as production specialist readiness.

## Changes

| Patch | Changes | What is and is not established |
|---|---|---|
| A | `production_profile_0105.py`, read-only profile auditor | Pins all current env/CLI scientific and performance flags without touching checkpoints. |
| B | GPU contract regression + optional `--model-b-profile` | CUDA device, configured real paths, existing sparse/triton parity and CUDA-graph tests. The machine running it must have free GPU memory. |
| C | Public `TrainingRun` versus legacy optimizer trajectory parity | One tiny dual-resolution CPU scenario checks identical samples, updates and counters; no full-size GPU performance claim. |
| D | Specialist document-first mixed sampler | Exact same mixture weights, sorted source selection, EOS boundaries, mask/padding semantics and checkpointed RNG; includes legacy sampler comparison. |
| E | `SpecialistTrainingRun`, frozen foundation, paired CE, resume | Optimizes only residual parameters via existing Hybrid Muon and chunked frozen head. Independent specialist snapshot contains optimizer, sampler RNG, token counters and foundation hash. CPU-tested. |
| F | Guarded CLI + regression integration | Safe dry run by default, requires `--run` to allocate/train and a distinct specialist checkpoint directory. |
| G | Documentation and final integrated gates | Explicit next steps and non-goals. |

## Existing production Model B training — unchanged

Your current Model B run uses **CuPy, BF16 mixed precision, context 8192,
batch 1, gradient accumulation 512, supervised-token weighting, hybrid Muon,
full optimizer offload and fine dual-resolution bucketing**. The exact 36
`MINI_LLM_` variables and the stable scientific CLI arguments are recorded in
`mini_llm/production_profile_0105.py`. `scripts/0105_model_b_profile.env` can be
sourced in a *new shell* to reproduce the environment, without changing the
running process or checkpoint files.

This patch does **not** replace `train_model.py`, its optimizer/sampler, or the
foundation training schedule. Model B has **348,261,376 parameters** and the
stride-16 sparse historical head is part of its architecture contract.

### CPU gate (safe while GPU is busy)

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python scripts/run_0103a_regressions.py \
    --mode cpu --report 0105_cpu_report.json
```

### GPU gate (run only when the RTX GPU has sufficient free VRAM)

```bash
.venv/bin/python scripts/run_0103a_regressions.py \
    --mode gpu --model-b-profile --report 0105_gpu_report.json
```

The GPU runner checks available device memory before each suite, fails when
CuPy/Triton is unavailable, and does **not** report skipped GPU assertions as
passed. **Do not run GPU tests concurrently with the active 16GB training
process**. These tests verify selected GPU mathematical paths but not complete
production-training equivalence or throughput.

### Optional read-only profile audit

The production-profile auditor compares the *environment and explicit CLI
flags*, not the training process itself. Omitted options report a mismatch:

```bash
source scripts/0105_model_b_profile.env
.venv/bin/python scripts/audit_0105_training_profile.py -- \
    --model dual-resolution-matrix-pool-sparse-1024-20l \
    --objective ar --precision bf16-mixed --runtime-profile consumer-gpu \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --context-length 8192 --batch-size 1 --grad-accum-steps 512 \
    --gradient-accumulation-weighting supervised_tokens \
    --optimizer hybrid-muon --muon-momentum 0.95 --muon-ns-steps 5 \
    --muon-matmul-precision bf16 --total-steps 100000 --warmup-steps 750 \
    --peak-lr 3e-4 --grad-clip 1.0 --weight-decay 0.1 \
    --val-interval 100 --val-steps 4 --val-per-source-steps 1 \
    --val-fixed-mixed-steps 4 --val-fixed-mixed-seed 78078 \
    --save-interval 50 --log-interval 1 --seed 42 \
    --benchmark-suite ./benchmarks/configs/foundation_capabilities_0097_progressive.json
```

`--checkpoint-dir`, `--log-file`, and `--resume-from` are *deliberately excluded*
from the scientific profile: tests must never use the active checkpoint output.

## Frozen-foundation specialist experiment

**Foundation training target:** a 20-token/parameter reference is about **6.97B
supervised tokens** for Model B. That Chinchilla ratio is a comparison milestone,
not an assertion that training saturates at 20 tokens/parameter.

**Initial specialist:** foundation frozen, only normalized full-context
hidden `H` passed to a residual specialist. No lexical reinjection and no
specialist-owned history. One correction at the shared frozen LM head. Start
with **two layers** for GPU smoke validation and then test the intended
**12-layer** configuration separately. All token weights come from the same
mixed corpus as the foundation. The sampler uses document-first EOS-bounded
windows with unchanged fine-shape bucketing. It does not do short-document
packing or cross-document specialist attention.

**Use a completed, immutable COPY of the foundation checkpoint**, not the
running `--resume-from` path. Checkpoint hashing reads all frozen weight files
at save/resume, so avoid sharing files that a trainer is actively replacing.

### Safe dry-run (no model allocation, no writes)

```bash
.venv/bin/python scripts/train_0105_specialist.py \
    --foundation-checkpoint ./checkpoints/model_b_7b_frozen_copy \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --specialist-checkpoint ./checkpoints/specialist_general_smoke \
    --specialist-layers 2 --context-length 512 --grad-accum-steps 1 \
    --steps 1
```

### **Experimental GPU smoke only**, when GPU is free

```bash
source scripts/0105_model_b_profile.env
.venv/bin/python scripts/train_0105_specialist.py \
    --foundation-checkpoint ./checkpoints/model_b_7b_frozen_copy \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --specialist-checkpoint ./checkpoints/specialist_general_smoke \
    --specialist-id general --specialist-layers 2 --context-length 512 \
    --grad-accum-steps 1 --muon-matmul-precision bf16 \
    --steps 1 --validation-batches 2 --run
```

The smoke run performs paired foundation/specialist CE on identical held-out
tokens, one residual-only optimizer update and a separately saved specialist
checkpoint. A zero-initialized correction must yield the same initial CE as the
frozen foundation. The optimizer/checkpoint restores the full specialist state.

For an **actual 12-layer experiment**, explicitly pass
`--specialist-layers 12 --context-length 8192`, first with small accumulation
and a copied checkpoint, and measure peak VRAM and supervised-token throughput
before committing to long runs. Accumulation 512 from the foundation is **not**
a default for specialist training; determine its optimum independently.

### Remaining acceptance gates

1. Run the 0105 GPU gate in an otherwise idle RTX 5080 environment.
2. Run a **two-layer CuPy smoke**, compare frozen vs corrected logits/loss,
   verify nonzero specialist gradients, unchanged foundation SHA256, full
   optimizer resume and absence of cross-document influence.
3. Repeat a *single* update with 12 layers and progressively larger context
   lengths, including 8192, auditing VRAM and numerical finite checks.
4. Compare specialist against foundation continuation using matched supervised
   token budgets **and matched compute**, not just optimization steps.
5. Implement production-friendly composed-model incremental generation and
   specialist-aware benchmark loading; the existing reference CE comparison
   is not a replacement for full qualitative-generation benchmarks.
6. Extend the common `Model.train()`/`TrainingRun` interface to cover
   `FrozenResidualComposition` after the optimized specialist GPU training
   path is validated. The current `SpecialistTrainingRun` is a *separate
   composition-level adapter* that reuses shared numerical components.

### Known limitations

- **No GPU hardware in the 0105 authoring environment**; no new CUDA kernel
  speed or correctness claims are made.
- No EOS-packed multi-document specialist attention; the sampler keeps each
  microbatch row in one document and ignores only masked tail padding.
- No independent historical readers, lexical reinjection, instruction-tuning,
  combiner, tool use, or specialist generation session yet.
- Validation samples are generated deterministically from an independent RNG
  stream; `evaluate_paired` restores validation sampler state, so comparisons
  are matched across checkpoints when starting from the same sampler seed.
- A frozen foundation is forward-only but remains expensive; 12-layer
  specialist speed and VRAM must be measured on the target 16GB GPU.
