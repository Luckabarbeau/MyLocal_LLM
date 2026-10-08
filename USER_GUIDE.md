# MyLocal_LLM User Guide

This guide describes how to **run, train, resume, evaluate, tune, and debug** the
current MyLocal_LLM implementation. The main `README.md` explains the project
philosophy and architecture; this file is the operational reference.

The project intentionally keeps low-level controls available because it is a
research platform. Normal use, however, should not require a wall of environment
variables. Starting with the 0062 runtime-profile cleanup, validated settings are
applied automatically and the individual `MINI_LLM_*` variables are primarily
**overrides for experiments and debugging**.

---

## 1. Runtime profiles

The array backend is selected before Python imports the model implementation:

```bash
MINI_LLM_BACKEND=cupy   # GPU
MINI_LLM_BACKEND=numpy  # CPU/reference
```

Training and inference then apply a runtime profile. Select it with
`--runtime-profile` or `MINI_LLM_RUNTIME_PROFILE`.

| Profile | Purpose |
|---|---|
| `auto` | Default. NumPy -> `reference`; CuPy inference -> `gpu-fast`; large CuPy training -> `consumer-gpu`. |
| `reference` | No implicit optimization overrides. Useful for correctness tests and A/B comparisons. |
| `gpu-fast` | Enables the validated CuPy/BF16 fused and grouped fast paths but does not move optimizer state or force LM-head chunking. |
| `consumer-gpu` | `gpu-fast` plus memory-saving training defaults for large local models: full optimizer offload, 16 MiB staging, and a 1024-row chunked LM head. |

**Explicit environment variables always win.** Profiles use `setdefault`, so a
launch such as

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=768 \
python train_model.py ...
```

keeps the explicit `768` even when `auto` resolves to `consumer-gpu`.

For the current 768d wide models, the CuPy profile also defaults the local
attention query chunk to `1536`, which is the best measured setting on the main
RTX 5080 development system. Override it when VRAM headroom or another GPU makes
a different tile preferable.

---

## 2. First training launch

Assuming the mixed packed shards are in `./token_shards_pretraining_65280`:

```bash
mkdir -p checkpoints logs

MINI_LLM_BACKEND=cupy \
python train_model.py \
    --model wide-500m-memory-64k \
    --precision bf16-mixed \
    --mixed-shard-root ./token_shards_pretraining_65280 \
    --batch-size 1 \
    --grad-accum-steps 4 \
    --total-steps 20 \
    --warmup-steps 2 \
    --val-interval 10 \
    --val-steps 2 \
    --save-interval 20 \
    --log-interval 1 \
    --checkpoint-dir ./checkpoints/wide-500m-memory-64k_smoke \
    --log-file ./logs/wide-500m-memory-64k_smoke.csv
```

With `MINI_LLM_BACKEND=cupy`, `auto` recognizes this as a large local training
configuration and supplies the validated fast paths and memory-saving defaults.
The startup banner prints the resolved runtime profile.

For the current long run, a representative command is therefore only:

```bash
MINI_LLM_BACKEND=cupy \
python train_model.py \
    --model wide-500m-memory-64k \
    --precision bf16-mixed \
    --mixed-shard-root ./token_shards_pretraining_65280 \
    --batch-size 2 \
    --grad-accum-steps 32 \
    --total-steps 160000 \
    --warmup-steps 5000 \
    --peak-lr 3e-4 \
    --grad-clip 1.0 \
    --weight-decay 0.1 \
    --val-interval 1000 \
    --val-steps 10 \
    --save-interval 2500 \
    --log-interval 10 \
    --checkpoint-dir ./checkpoints/wide-500m-memory-64k_30b \
    --log-file ./logs/wide-500m-memory-64k_30b.csv
```

The `wide-500m-memory-*` presets now carry the current routed-prefix defaults:
4k current window, 128-token historical blocks, top-16 / 2k routed prefix,
50% retrieval-batch probability, Gumbel exploration, router temperature
`1.0 -> 0.25`, and a 50k-step router-temperature anneal.

---

## 3. Resume training

Resume from the checkpoint directory rather than starting a new optimizer:

```bash
MINI_LLM_BACKEND=cupy \
python train_model.py \
    --resume-from ./checkpoints/wide-500m-memory-64k_30b \
    --precision bf16-mixed \
    --mixed-shard-root ./token_shards_pretraining_65280 \
    --batch-size 2 \
    --grad-accum-steps 32 \
    --total-steps 160000 \
    --warmup-steps 5000 \
    --peak-lr 3e-4 \
    --grad-clip 1.0 \
    --weight-decay 0.1 \
    --val-interval 1000 \
    --val-steps 10 \
    --save-interval 2500 \
    --log-interval 10 \
    --checkpoint-dir ./checkpoints/wide-500m-memory-64k_30b \
    --log-file ./logs/wide-500m-memory-64k_30b.csv
```

The checkpoint restores model weights, optimizer state, step count, shard state,
and RNG state. Keep schedule-defining arguments consistent with the original run.

`--init-from` is different: it initializes matching model weights from another
checkpoint but starts a **fresh optimizer and training schedule**.

---

## 4. Inference

### Short prompt

```bash
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint ./checkpoints/wide-500m-memory-64k_30b \
    --prompt "The purpose of numerical simulation is" \
    --max-new-tokens 128 \
    --strategy top-p \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42
```

`kv-cache` is already the default inference backend.

### Long prompt / routed memory

```bash
MINI_LLM_BACKEND=cupy \
python inference.py \
    --checkpoint ./checkpoints/wide-500m-memory-64k_30b \
    --prompt-file ./long_prompt.txt \
    --max-new-tokens 512 \
    --strategy top-p \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42
```

For a routed-prefix checkpoint, inference automatically uses the trained memory
horizon from `config.json`; `--max-context 65536` is therefore optional for the
64k preset.

The routed prefix is deterministically refreshed every 128 generated tokens by
default. For an exact but slower validation mode:

```bash
MINI_LLM_ROUTED_PREFIX_REFRESH_TOKENS=1 \
MINI_LLM_BACKEND=cupy \
python inference.py ...
```

---

## 5. Important training CLI arguments

| Argument | Meaning |
|---|---|
| `--model` | Architecture preset. The current main model is `wide-500m-memory-64k`. |
| `--precision` | `float32`, `float16`, or `bf16-mixed`. Current GPU training uses `bf16-mixed`. |
| `--mixed-shard-root` | Root containing the mixed packed-shard manifest/tokenizer/source folders. |
| `--batch-size` | Sequences in one GPU microbatch. |
| `--grad-accum-steps` | Number of microbatches accumulated before one optimizer update. Effective batch = batch size × accumulation. |
| `--total-steps` | Total optimizer-step target, including the already-completed steps after resume. |
| `--warmup-steps` | Linear LR warmup length. |
| `--peak-lr` | Peak learning rate before decay. |
| `--grad-clip` | Global gradient-norm clipping threshold. |
| `--weight-decay` | AdamW weight decay. |
| `--val-interval` | Optimizer steps between validation runs. |
| `--val-steps` | Validation batches per validation run. |
| `--save-interval` | Optimizer steps between checkpoints. |
| `--log-interval` | Optimizer steps between console/CSV summaries. |
| `--checkpoint-dir` | Output checkpoint directory. |
| `--resume-from` | Continue model + optimizer + schedule state from a checkpoint. |
| `--init-from` | Load matching weights only; start a new optimizer/run. |
| `--profile-steps` | Synchronized performance profiling for the first N executed optimizer steps. Keep `0` for long runs. |
| `--numerical-debug` | Expensive NaN/Inf tracing. Use only for diagnosis. |
| `--runtime-profile` | `auto`, `reference`, `gpu-fast`, or `consumer-gpu`. |
| `--progressive-depth` | Enable 0064B progressive residual depth for a new run. The final/max depth remains the preset's `n_layers`. |
| `--initial-active-layers` | Number of Transformer blocks executed at the start of a new progressive run. Default: 1. |
| `--progressive-growth-steps` | Comma-separated completed optimizer steps after which one additional block becomes active, e.g. `1000,3000,7000`. Omit for manual-only growth. |

Run `python train_model.py --help` for the complete CLI list.

### Progressive-depth experiment

0064B implements deterministic/manual growth only. It deliberately does **not**
yet implement the future automatic plateau detector or adaptive token selection.
A new block is independently random-initialized, but both learned residual gates
start at exact zero when that block becomes active:

```text
y = x + alpha_attn * Attention(Norm(x))
z = y + alpha_mlp  * MoE(Norm(y))

new layer: alpha_attn = alpha_mlp = 0
```

The insertion therefore preserves the current network function exactly. All
previously active blocks remain trainable. The new block receives fresh Adam
moments/master state and a local Adam age of one on its first update; existing
optimizer history is not rebuilt or reset.

Example deterministic research run:

```bash
MINI_LLM_BACKEND=cupy \
python train_model.py \
    --model wide-500m-memory-64k \
    --precision bf16-mixed \
    --mixed-shard-root ./token_shards_pretraining_65280 \
    --progressive-depth \
    --initial-active-layers 1 \
    --progressive-growth-steps 1000,3000,7000,12000 \
    --batch-size 2 \
    --grad-accum-steps 32 \
    --total-steps 20000 \
    --warmup-steps 1000 \
    --checkpoint-dir ./checkpoints/progressive_test
```

A growth step `N` means that `N` optimizer updates are completed at the previous
depth; the next block is activated before update `N+1`. Checkpoints store the
current active depth, the explicit growth schedule, and the per-parameter Adam
birth steps. If `--progressive-growth-steps` is omitted on resume, the saved
schedule is reused.

For the first scientific comparison, use an explicit schedule or call the
trainer's growth method manually. Automatic growth based on validation
improvement per FLOP belongs to the next patch once the base mechanism has been
benchmarked.

---

## 6. Important inference CLI arguments

| Argument | Meaning |
|---|---|
| `--checkpoint` | Checkpoint directory containing `config.json`, model arrays, and tokenizer. |
| `--prompt` | Prompt text on the command line. |
| `--prompt-file` | Read a long prompt from a UTF-8 file. |
| `--max-new-tokens` | Generation budget. |
| `--backend` | `kv-cache` (default) or slow `reference` full-prefix recomputation. |
| `--strategy` | `greedy`, `temperature`, `top-k`, or `top-p`. |
| `--temperature` | Sampling temperature. |
| `--top-k` / `--top-p` | Sampling cutoff for the selected strategy. |
| `--max-context` | Override the addressable inference horizon. Memory checkpoints already default to their trained horizon. |
| `--runtime-profile` | Runtime optimization profile. CuPy inference `auto` resolves to `gpu-fast`. |

---

# 7. Environment-variable reference

The following variables are intentionally retained as research controls. In
normal use, prefer a runtime profile and override only the setting you are
actively testing.

## 7.1 Backend and profile

| Variable | Normal behavior | Purpose |
|---|---|---|
| `MINI_LLM_BACKEND` | `numpy` if omitted | Select `numpy` or `cupy`. GPU use currently requires setting `cupy` before process start. |
| `MINI_LLM_RUNTIME_PROFILE` | `auto` | Same selection as `--runtime-profile`; the CLI option takes precedence when supplied. |
| `MINI_LLM_BF16_MIXED_CONTEXT` | off | Experimental mixed BF16 context-storage path used by attention internals. Leave off unless benchmarking it explicitly. |

## 7.2 Attention and QKV fast paths

| Variable | Auto CuPy profile | Purpose |
|---|---:|---|
| `MINI_LLM_PACKED_QKV` | `1` | Compute Q/K/V from one packed projection path. |
| `MINI_LLM_COMPACT_PACKED_V` | `1` | Avoid unnecessarily expanded V storage in the packed path. |
| `MINI_LLM_FUSED_LOCAL_SOFTMAX` | `1` | CUDA local causal softmax path. |
| `MINI_LLM_BF16_LOCAL_GEMM` | `1` | Keep local-attention GEMMs in BF16/Tensor-Core-compatible form. |
| `MINI_LLM_FUSED_BF16_LOCAL_PIPELINE` | `1` | Fused BF16 local-attention pipeline. |
| `MINI_LLM_FUSED_LOCAL_MULTI_CHUNK_SOFTMAX` | `1` | Fused local softmax for multi-chunk execution. |
| `MINI_LLM_FUSED_INDEXED_SOFTMAX` | `1` | Fused softmax for indexed sparse attention. |
| `MINI_LLM_FUSED_BF16_INDEXED_PIPELINE` | `1` | BF16 indexed-attention fused path. |
| `MINI_LLM_FUSED_DILATED_MULTI_CHUNK_SOFTMAX` | `1` | Fused softmax for dilated multi-chunk attention. |
| `MINI_LLM_LOCAL_ATTN_QUERY_CHUNK` | `1536` for current wide models | Local-attention query tile. Lower values usually reduce transient VRAM; higher values may improve throughput. |
| `MINI_LLM_INDEXED_ATTN_QUERY_CHUNK` | module default `128` | Query chunk size for generic indexed attention. |
| `MINI_LLM_DILATED_ATTN_QUERY_CHUNK` | module default `1024` | Query chunk size for dilated attention. |
| `MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM` | `1` | Use direct-pointer grouped cuBLAS for the validated local path. |
| `MINI_LLM_LOCAL_CUBLAS_MODE` | `grouped` | Choose grouped or experimental batched cuBLAS dispatcher. |
| `MINI_LLM_CUBLAS_AUTOTUNE` | `0` | Enable/disable runtime cuBLAS autotuning. Current validated profile leaves it off. |
| `MINI_LLM_CUBLAS_GROUPED_DILATED_GEMM` | `1` | Extend grouped cuBLAS execution to the dilated path. |

## 7.3 Retrieval and context-router kernels

| Variable | Auto CuPy profile | Purpose |
|---|---:|---|
| `MINI_LLM_DIRECT_RETRIEVAL_SCATTER` | `1` | Direct CUDA scatter for retrieval queries/KV gradients instead of generic indexing. |
| `MINI_LLM_FUSED_BF16_RETRIEVAL_PIPELINE` | `1` | BF16 Tensor-Core/fused-softmax retrieval-attention path. |
| `MINI_LLM_DIRECT_QUERY_POOL` | `1` | Direct CUDA pooling for the top-level context-router query. |
| `MINI_LLM_DIRECT_QUERY_POOL_BACKWARD` | `1` | Matching direct backward path for query pooling. |

## 7.4 MoE / expert kernels

| Variable | Auto CuPy profile | Purpose |
|---|---:|---|
| `MINI_LLM_FUSED_SWIGLU` | `1` | Fused BF16 SwiGLU activation/backward kernels. |
| `MINI_LLM_FUSED_EXPERT_GEMM` | `0` | Experimental packed gate/up expert GEMM. Current measurements favor the normal expert GEMMs. |
| `MINI_LLM_CONCURRENT_EXPERTS` | `0` | Run expert work on concurrent CUDA streams. Currently disabled by default. |
| `MINI_LLM_EXPERT_STREAMS` | `2` | Number of expert streams when concurrent experts are enabled. |

## 7.5 RMSNorm, loss, and LM head

| Variable | Auto/consumer behavior | Purpose |
|---|---:|---|
| `MINI_LLM_FUSED_BF16_RMSNORM` | `1` | Fused training RMSNorm for FP32 residual -> BF16 branch activations. |
| `MINI_LLM_DISABLE_FUSED_RMSNORM` | `0` | Inference/debug escape hatch for fused RMSNorm. |
| `MINI_LLM_FUSED_BF16_CE` | `1` | Fused BF16 cross-entropy path. |
| `MINI_LLM_INPLACE_BF16_CE` | `1` | Reuse compatible CE buffers to reduce memory traffic/allocation. |
| `MINI_LLM_LOSS_TOKEN_CHUNK` | `512` | Token chunk used by the fallback/chunked loss implementation. |
| `MINI_LLM_LM_HEAD_CHUNK_TOKENS` | `1024` in `consumer-gpu` training | Chunk/recompute the vocabulary projection and validation loss. `0` materializes the full logits tensor. |

## 7.6 Optimizer and activation memory

| Variable | Consumer profile | Purpose |
|---|---:|---|
| `MINI_LLM_OPTIMIZER_OFFLOAD` | `full` | `none`, `moments`, or `full`. Full offload keeps Adam history/master state in host memory with GPU staging. |
| `MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB` | `16` | GPU staging chunk size for optimizer offload. Smaller saves VRAM but can add overhead. |
| `MINI_LLM_ACTIVATION_CHECKPOINT` | `none` | `none` or `block`. Recompute block activations in backward to save VRAM at substantial compute cost. |

## 7.7 Routed-prefix memory architecture

Most of these are **architecture/curriculum overrides**. The
`wide-500m-memory-*` presets already contain the current defaults, so they are
not required in normal launch commands.

| Variable | 64k preset default | Purpose |
|---|---:|---|
| `MINI_LLM_HIERARCHICAL_MEMORY` | enabled by preset | Force hierarchical-memory enable/disable. |
| `MINI_LLM_MEMORY_INTEGRATION` | `routed_prefix` | `routed_prefix`, legacy `terminal_landmark`, or legacy `pretransformer_read`. |
| `MINI_LLM_MEMORY_TRAINING` | `joint` | Memory training mode; routed-prefix uses joint LM training. |
| `MINI_LLM_MEMORY_LENGTH` | `65536` | Total causal/addressable horizon. |
| `MINI_LLM_MEMORY_RECENT_TOKENS` | `4096` | Recent/working context length. |
| `MINI_LLM_MEMORY_TARGET_TOKENS` | `4096` | Continuous current-window target length. |
| `MINI_LLM_MEMORY_BLOCK_SIZE` | `128` | Historical routing block size. |
| `MINI_LLM_MEMORY_TOP_K_BLOCKS` | `16` | Maximum selected historical blocks; 16 × 128 = 2048-token routed prefix. |
| `MINI_LLM_MEMORY_QUERY_TOKENS` | `4096` | Tokens pooled to construct the context-router query. |
| `MINI_LLM_MEMORY_ROUTER_DIM` | `64` | Router projection dimension. |
| `MINI_LLM_MEMORY_ROUTER_TEMPERATURE` | `1.0` | Initial soft surrogate/Gumbel routing temperature. |
| `MINI_LLM_MEMORY_ROUTER_TEMPERATURE_MIN` | `0.25` | Final router temperature. |
| `MINI_LLM_MEMORY_ROUTER_TEMPERATURE_ANNEAL_STEPS` | `50000` | Steps used to anneal router temperature. |
| `MINI_LLM_MEMORY_ROUTER_SURROGATE_SCALE` | `1.0` | Scale of the straight-through router surrogate gradient. |
| `MINI_LLM_MEMORY_ROUTER_GUMBEL_NOISE` | enabled | Stochastic Gumbel exploration during training. Inference remains deterministic. |
| `MINI_LLM_MEMORY_RETRIEVAL_BATCH_PROBABILITY` | `0.5` | Requested fraction of training samples/batches using retrieval; short histories naturally use direct prefixes/no routing. |
| `MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS` | `1` under CuPy profile | Print periodic router-selection diagnostics. |
| `MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS_INTERVAL` | `100` | Optimizer-step interval for router diagnostics. |

Legacy external-reader / terminal-Landmark controls are kept for ablations:

| Variable | Purpose |
|---|---|
| `MINI_LLM_MEMORY_READ_HEADS` | Number of external-memory read query heads. |
| `MINI_LLM_MEMORY_READ_KV_HEADS` | K/V heads for the legacy memory reader. |
| `MINI_LLM_MEMORY_READ_QUERY_CHUNK` | Query chunk size for that reader. |
| `MINI_LLM_MEMORY_ATTENTION_LAYER` | Transformer layer used by terminal-Landmark memory. |
| `MINI_LLM_MEMORY_MIN_ROUTER_HISTORY_BLOCKS` | Minimum eligible history blocks for legacy routing. |
| `MINI_LLM_MEMORY_READER_RESIDUAL_SCALE` | Residual scaling for the legacy external-reader path. |

## 7.8 Routed-prefix inference controls

| Variable | Default | Purpose |
|---|---:|---|
| `MINI_LLM_ROUTED_PREFIX_PREFILL_CHUNK` | `4096` | Chunk size used to project/store long prompt history during prefill. |
| `MINI_LLM_ROUTED_PREFIX_REFRESH_TOKENS` | checkpoint default `128` | Number of generated tokens between deterministic top-level reroutes/cache rebuilds. Use `1` for strict reference semantics. |
| `MINI_LLM_TERMINAL_MEMORY_PREFILL_CHUNK` | `4096` | Legacy terminal-Landmark prefill chunk. |
| `MINI_LLM_FUSED_DECODE_QKV_ROPE_STORE` | `1` | Fused single-token QKV unpack + RoPE + KV-cache insertion. |
| `MINI_LLM_FUSED_SPARSE_DECODE` | `0` | Experimental fused one-token sparse-attention kernel; still opt-in. |
| `MINI_LLM_FUSED_TERMINAL_MEMORY_DECODE` | `1` | Legacy terminal-memory decode fusion. |

## 7.9 Profiling and numerical debugging

| Variable | Default | Purpose |
|---|---:|---|
| `MINI_LLM_MEMORY_PROFILE` | `0` | Record GPU-memory boundaries for early optimizer steps. |
| `MINI_LLM_MEMORY_PROFILE_STEPS` | `1` when memory profiling is enabled | Number of profiled optimizer steps. |
| `MINI_LLM_LOCAL_DETAIL_PROFILE` | `0` | Fine-grained local-attention profiler scopes. |
| `MINI_LLM_RETRIEVAL_ROUTER_DETAIL_PROFILE` | `0` | Fine-grained retrieval/router profiler scopes. |
| `MINI_LLM_MOE_DETAIL_PROFILE` | `0` | Fine-grained MoE profiler scopes. |
| `MINI_LLM_RMSNORM_DETAIL_PROFILE` | `0` | Fine-grained RMSNorm profiler scopes. |
| `MINI_LLM_FINITE_TRACE_START` | `-1` | Begin expensive finite-value tracing at an optimizer step. |
| `MINI_LLM_FINITE_TRACE_END` | same as start | Final finite-trace step. |

### Strict flags

Most optimized CUDA paths have a matching `*_STRICT` flag. Examples include:

```text
MINI_LLM_PACKED_QKV_STRICT
MINI_LLM_FUSED_LOCAL_SOFTMAX_STRICT
MINI_LLM_FUSED_BF16_LOCAL_PIPELINE_STRICT
MINI_LLM_FUSED_INDEXED_SOFTMAX_STRICT
MINI_LLM_FUSED_BF16_INDEXED_PIPELINE_STRICT
MINI_LLM_DIRECT_RETRIEVAL_SCATTER_STRICT
MINI_LLM_FUSED_BF16_RETRIEVAL_PIPELINE_STRICT
MINI_LLM_DIRECT_QUERY_POOL_STRICT
MINI_LLM_FUSED_SWIGLU_STRICT
MINI_LLM_FUSED_BF16_CE_STRICT
MINI_LLM_FUSED_BF16_RMSNORM_STRICT
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT
MINI_LLM_FUSED_DECODE_QKV_ROPE_STORE_STRICT
MINI_LLM_FUSED_SPARSE_DECODE_STRICT
MINI_LLM_FUSED_TERMINAL_MEMORY_DECODE_STRICT
```

Normal optimized paths are allowed to fall back if a kernel cannot compile or a
layout is unsupported. Setting the matching strict flag to `1` converts that
fallback into an error. Strict mode is useful for kernel validation and
benchmarking; it is intentionally **not** part of the normal runtime profile.

---

# 8. Memory tuning recipes

## Need a little more VRAM headroom

Try, in order:

```bash
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=1024
```

then:

```bash
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=768
```

and/or reduce optimizer staging:

```bash
MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB=8
```

The chunked LM head is already enabled by `consumer-gpu` for the wide model.
Reduce it further if the vocabulary projection itself is the pressure point:

```bash
MINI_LLM_LM_HEAD_CHUNK_TOKENS=512
```

Use block activation checkpointing only when these lighter options are
insufficient; it trades substantial extra compute for memory.

## Debug a suspected optimized-kernel problem

Run the same case with:

```bash
MINI_LLM_BACKEND=cupy \
python train_model.py --runtime-profile reference ...
```

Then re-enable one optimization at a time. This is the preferred A/B workflow.

---

# 9. What should usually remain untouched

For normal training of a named model preset, do not repeat architecture values
on the command line/environment unless you are intentionally running an
ablation. In particular, the current memory preset already defines:

```text
integration mode
memory horizon
block size
number of routed blocks
router dimensionality
router temperature schedule
Gumbel exploration
retrieval probability
inference refresh interval
```

Keeping architecture in `ModelConfig` and hardware/runtime policy in the
runtime profile makes checkpoints easier to reproduce and commands much easier
to read.

---

# 10. Recommended workflow

1. Use `auto` for normal work.
2. Change one explicit `MINI_LLM_*` variable only when testing a hypothesis.
3. Record that override with benchmark results.
4. If the new behavior becomes consistently better and robust, promote it into
   the appropriate runtime profile rather than permanently lengthening launch
   commands.
5. Keep architecture changes in model presets/configuration rather than runtime
   performance profiles.

This keeps the project both **easy to run** and **fully inspectable**.
