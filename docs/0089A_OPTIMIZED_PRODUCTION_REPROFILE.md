# 0089A — Re-profile the optimized production training graph

## Objective

0089A measures whether the 0088A Triton mixed-attention speedup survives the complete production forward/loss/backward graph and establishes the new bottleneck distribution before broader CuPy-versus-compiled-PyTorch work.

This patch does not change model mathematics or production defaults. The bounded probe:

- loads the real checkpoint with `skip_optimizer=True` and `skip_training=True`;
- samples the existing document-first, EOS-bounded mixed-shard path;
- executes `forward_body`, chunked LM-head/loss forward, numerical LM-head backward, and `backward_body`;
- never constructs an optimizer, updates a parameter, or writes a checkpoint;
- retains grouped attention as its default;
- permits Triton only when explicitly selected, with strict mode required.

The primary score is complete no-optimizer model F+B. It includes all 20 blocks, historical work activated by the sampled batch, final norm, and chunked LM-head/loss. It excludes checkpoint/data setup, warmup, `zero_grad`, optimizer/offload/update, validation, and checkpoint writes.

## Implementation

### Backend-neutral bounded probe

`benchmarks/profile_0087a_training.py` now accepts:

```text
--attention-backend grouped|triton
```

If omitted, the prior behavior is preserved: an explicit environment setting wins, otherwise grouped is selected by the existing production defaults. Triton measurements require:

```bash
MINI_LLM_TRAIN_ATTN_STRICT=1
```

so an unsupported geometry or missing optional dependency cannot silently fall back to grouped attention.

The JSON schema is `0089A-v1` and records:

- selected attention backend and strictness;
- Triton version when active;
- raw GPU-event and synchronized host-wall samples;
- exact input, target, loss-mask, and history-start SHA-256 hashes;
- router seed;
- actual history validity and executed current-head pattern;
- scalar loss after the measured numerical backward;
- the existing no-update safety assertions.

### Correct executed-head accounting

The previous report called a scheduled historical head active if any batch row had a valid source. Production dispatch has a different rule: current heads are globally skipped only when `historical_state.all_rows_have_valid` is true. Mixed-validity batches retain all current fallback heads and replace outputs row-wise only where historical reads are valid.

0089A therefore reports separately:

- scheduled historical heads;
- heads with any valid historical source;
- whether all batch rows have valid history;
- exact `executed_historical_skip_heads`;
- exact `executed_active_local_heads` and count.

For example, the measured T4096 batch had history lengths `[0, 5657, 61440]`. It contains historical work, but because one row has no history, every layer executes all 15 current local heads. The old any-row accounting would have incorrectly labeled some layers as 13/11/9-head current-attention patterns.

### Fail-closed report comparator

`benchmarks/compare_0089a_training.py` refuses a performance verdict unless:

- both reports use schema `0089A-v1`;
- grouped and Triton backends are identified correctly;
- Triton strict mode was active;
- all no-update safety fields are false;
- checkpoint/model metadata match;
- exact batch hashes and router seed match;
- every effective runtime flag other than backend/strict mode matches;
- final scalar loss is finite and within the declared tolerances.

Declared scalar-loss gates:

```text
absolute error <= 5e-3
relative error <= 2e-3
```

This full-model scalar gate supplements, but does not replace, the existing tensor-level output/dQ/dK/dV and production-dispatch gradient tests from 0088A.

## Environment

| Component | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, SM120, 16 GiB |
| Driver | 580.178.04 |
| Python | 3.12.3 |
| CuPy | 14.2.0 |
| CUDA runtime reported by CuPy | 13.2 |
| Triton | 3.8.0 |
| Checkpoint | `checkpoints/0083_intermediate_20L_1024_fullmix_100k` |
| Dataset | `token_shards_ensemble_32k_mixed` |

The compared production geometry is 20 layers, width 1024, FFN width 4096, 16 query heads, four KV heads, head dimension 64, BF16 branch compute, and FP32 residual/gradient paths.

## Methodology

For each shape:

- B=3;
- one deterministic real document-first batch selected with seed 42;
- identical checkpoint and all non-attention runtime flags;
- two untimed warmups;
- 20 synchronized CUDA-event and host-wall samples;
- separate grouped and Triton processes;
- no profiler for authoritative timing;
- compile/JIT work excluded by warmup;
- raw samples retained in JSON.

The fixture hashes were compared before speedup calculation. Independent medians are not added.

## Full-model numerical results

All four matched comparisons passed the declared scalar-loss gates.

| Shape | Computed rows | Supervised rows | Valid history tokens/sample | Loss abs. error | Loss relative error |
|---|---:|---:|---|---:|---:|
| B3/T1024 | 3,072 | 1,842 | `[0, 0, 0]` | 0.000216 | 0.000127 |
| B3/T2048 | 6,144 | 4,028 | `[0, 0, 0]` | 0.000745 | 0.000485 |
| B3/T2304 | 6,912 | 3,617 | `[0, 0, 0]` | 0.000576 | 0.000197 |
| B3/T4096 | 12,288 | 9,078 | `[0, 5657, 61440]` | 0.000110 | 0.000043 |

These differences are consistent with the already-qualified BF16 reduction-order differences. This patch does not claim complete per-parameter full-model gradient parity; the attention tensor/gradient contract remains covered by 0088A, and complete block parameter-gradient parity is the next patch gate.

## Authoritative unprofiled performance

All times are warmed CUDA GPU-event medians from 20 samples. Host-wall medians closely matched because synchronization occurs only at the outer timing boundary.

| Shape | Grouped F+B | Triton F+B | Speedup | Latency reduction |
|---|---:|---:|---:|---:|
| B3/T1024 | 145.327 ms | 111.185 ms | 1.307x | 23.49% |
| B3/T2048 | 293.814 ms | 215.425 ms | 1.364x | 26.68% |
| B3/T2304 | 333.877 ms | 240.404 ms | **1.389x** | **28.00%** |
| B3/T4096 | 1042.760 ms | 855.311 ms | 1.219x | 17.98% |

### Distribution details

| Shape/backend | p10 | Median | p90 |
|---|---:|---:|---:|
| T1024 grouped | 143.943 ms | 145.327 ms | 146.229 ms |
| T1024 Triton | 110.274 ms | 111.185 ms | 112.582 ms |
| T2048 grouped | 291.520 ms | 293.814 ms | 294.905 ms |
| T2048 Triton | 214.073 ms | 215.425 ms | 220.290 ms |
| T2304 grouped | 330.608 ms | 333.877 ms | 345.872 ms |
| T2304 Triton | 238.652 ms | 240.404 ms | 243.117 ms |
| T4096 grouped | 1037.851 ms | 1042.760 ms | 1045.162 ms |
| T4096 Triton | 853.646 ms | 855.311 ms | 857.406 ms |

The intervals are clearly separated; the result is larger than observed sample noise.

### Interpretation

The isolated 0088A component win survives full production integration. On history-free B3/T1024–2304 batches, replacing current grouped attention reduces complete model F+B latency by 23–28%. This is the direct full-model evidence that was missing from 0088A.

The history-heavy T4096 batch still improves by 18%, but its smaller relative gain shows that current attention is no longer the dominant cost when substantial historical state is active. The T4096 total is not directly comparable to the older 0087A T4096 result because the sampled batch has much more history (`[0, 5657, 61440]` here versus `[0, 0, 9284]` in the earlier report).

These no-optimizer microbatch measurements must not be presented as full optimizer-step throughput. The production log provides encouraging independent evidence, but it has variable rows/history and includes optimizer work, so it is not substituted for this controlled comparison.

## Nsight Systems: optimized T2304 without history

One diagnostic sample measured:

- analysis window: 251.177 ms;
- union of GPU kernel/memory execution: 242.296 ms;
- execution-gap proxy: 3.536%;
- candidate avoidable host starvation: 1.674 ms, 0.666%;
- fixed-GPU-work host-starvation upper bound: 1.0067x;
- internal gap median/p90/max: 1.824/2.615/778.731 microseconds.

This again rejects ordinary Python starvation as the principal remaining limit.

Correlation-aware ranges included:

| Range | GPU execution union | Instances |
|---|---:|---:|
| LM head/loss | 52.227 ms | 2 |
| Backward envelope | 48.582 ms | 1 |
| Attention backward/recompute envelope | 30.651 ms | 20 |
| Current Triton attention | 12.368 ms | 40 |
| RMSNorm/residual | 9.983 ms | 82 |
| FFN backward scope | 6.998 ms | 20 |
| FFN forward scope | 1.589 ms | 20 |
| Historical scopes | 0.224 ms | 22 |

Ranges overlap and queued work may execute after a Python scope closes, so these values are diagnostic and not additive. Kernel ranking is more useful for target selection. Major families included:

- BF16 tensor-operation GEMMs: 45.48, 41.26, 26.40, and 13.39 ms across leading shapes;
- BF16 SwiGLU forward: 9.62 ms;
- BF16 SwiGLU backward: 7.89 ms;
- Triton dK/dV: 8.75 ms;
- Triton dQ: 6.86 ms;
- Triton forward: 4.09 ms;
- large LM-head-related GEMMs near 5 ms each.

For ordinary history-free batches, complete-block GEMMs/SwiGLU and LM-head/loss are now the highest-priority independent comparisons.

## Nsight Systems: optimized T4096 with history

The history-heavy sample measured:

- analysis window: 862.150 ms;
- union of GPU kernel/memory execution: 833.015 ms;
- execution-gap proxy: 3.379%;
- candidate avoidable host starvation: 0.712 ms, 0.083%;
- fixed-GPU-work host-starvation upper bound: 1.0008x;
- internal gap median/p90/max: 1.472/2.368/838.764 microseconds.

Correlation-aware ranges included:

| Range | GPU execution union | Instances |
|---|---:|---:|
| Historical retrieval/recompute | **511.745 ms** | 34 |
| Backward envelope | 499.846 ms | 1 |
| Attention backward/recompute envelope | 465.041 ms | 20 |
| Forward envelope | 143.124 ms | 1 |
| LM head/loss | 90.332 ms | 2 |
| Current Triton attention | 24.727 ms | 40 |
| FFN backward scope | 8.611 ms | 20 |
| RMSNorm/residual | 7.791 ms | 82 |
| FFN forward scope | 3.935 ms | 20 |

The historical range overlaps parent forward/backward envelopes and is not an exclusive 59% phase budget. Nevertheless, both its magnitude and kernel composition establish that historical processing—not current attention or host launch starvation—is the principal next target for this specific heavy-history batch.

The leading history-heavy kernel families include numerous FP32 copies, multiplies, where/add/subtract/exp operations, FP32 SIMT GEMMs, BF16 GEMMs, and large projection GEMMs. This points to data movement, temporary materialization, and historical encoder/read arithmetic rather than a single missing launch fusion.

## Decision

0089A establishes two workload-dependent next priorities:

1. **History-free/common current-window batches:** build the exact complete Transformer-block CuPy-versus-compiled-PyTorch F+B benchmark. This covers RMSNorm, packed QKV, RoPE, optimized attention, output projection, residuals, dense SwiGLU, and every parameter gradient. It will determine whether FFN/projection fusion offers another material local optimization.
2. **History-heavy batches:** separately benchmark the historical encoder, deterministic/sparse/learned reads, shared gradient fan-in, and recomputation. Standard full-block PyTorch does not represent this architecture, so historical work requires its own exact fixture and comparison.

LM-head/loss is also material in both traces and should follow the block/historical comparisons. Optimizer/offload remains outside this patch and still needs an isolated full-step measurement.

## Reproduction

Run from the repository root. The production virtual environment must contain Triton 3.8.0 for the Triton candidate.

### Tests

```bash
.venv/bin/python -m py_compile \
  benchmarks/profile_0087a_training.py \
  benchmarks/compare_0089a_training.py \
  tests/test_0089a_optimized_training.py

.venv/bin/python -m pytest -q \
  tests/test_0087a_profile_ranges.py \
  tests/test_0089a_optimized_training.py
```

### Matched unprofiled runs

Replace `2304` consistently with `1024`, `2048`, or `4096` for the other shapes.

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_TRAIN_ATTN_STRICT=0 \
MINI_LLM_BF16_MIXED_CONTEXT=0 \
.venv/bin/python benchmarks/profile_0087a_training.py \
  --attention-backend grouped \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --batch-size 3 --microbatches 1 --working-length 2304 \
  --seed 42 --warmup 2 --repeats 20 --no-optimizer \
  --out benchmarks/results/0089a_grouped_b3_t2304.json

MINI_LLM_BACKEND=cupy \
MINI_LLM_TRAIN_ATTN_STRICT=1 \
MINI_LLM_BF16_MIXED_CONTEXT=0 \
.venv/bin/python benchmarks/profile_0087a_training.py \
  --attention-backend triton \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --batch-size 3 --microbatches 1 --working-length 2304 \
  --seed 42 --warmup 2 --repeats 20 --no-optimizer \
  --out benchmarks/results/0089a_triton_b3_t2304.json

.venv/bin/python benchmarks/compare_0089a_training.py \
  --grouped benchmarks/results/0089a_grouped_b3_t2304.json \
  --triton benchmarks/results/0089a_triton_b3_t2304.json \
  --out benchmarks/results/0089a_comparison_b3_t2304.json
```

### Nsight capture

Use `--capture`; the probe changes repeats to one and enables coarse annotations. `stop-shutdown` normally makes Nsight return 143 after a successful capture and report generation.

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_TRAIN_ATTN_STRICT=1 \
MINI_LLM_BF16_MIXED_CONTEXT=0 \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=PROFILE_0087A_MICROBATCH@MyLocal_LLM_0087A \
  --capture-range-end=stop-shutdown --force-overwrite=true \
  --output=benchmarks/results/0089a_triton_t4096_profile \
  .venv/bin/python benchmarks/profile_0087a_training.py \
    --attention-backend triton \
    --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --batch-size 3 --microbatches 1 --working-length 4096 \
    --seed 42 --warmup 2 --no-optimizer --capture \
    --out benchmarks/results/0089a_triton_t4096_profile.json

nsys export --type sqlite --force-overwrite=true \
  --output=benchmarks/results/0089a_triton_t4096_profile.sqlite \
  benchmarks/results/0089a_triton_t4096_profile.nsys-rep

.venv/bin/python benchmarks/analyze_0087a_nsys.py \
  benchmarks/results/0089a_triton_t4096_profile.sqlite \
  --output-json benchmarks/results/0089a_triton_t4096_analysis.json \
  --output-prefix benchmarks/results/0089a_triton_t4096_analysis
```

## Raw artifact paths

Generated benchmark and Nsight files are intentionally excluded from the patch and remain under `benchmarks/results/`:

```text
0089a_grouped_b3_t{1024,2048,2304,4096}.json
0089a_triton_b3_t{1024,2048,2304,4096}.json
0089a_comparison_b3_t{1024,2048,2304,4096}.json
0089a_triton_t2304_profile.nsys-rep
0089a_triton_t2304_profile.sqlite
0089a_triton_t2304_analysis.json
0089a_triton_t4096_profile.nsys-rep
0089a_triton_t4096_profile.sqlite
0089a_triton_t4096_analysis.json
```

No Nsight Compute hardware-counter claim is made.
