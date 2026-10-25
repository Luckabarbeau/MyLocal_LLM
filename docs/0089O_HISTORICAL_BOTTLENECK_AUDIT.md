# PATCH 0089O — Correlation-correct historical GPU bottleneck audit

## Result

The current seed-48 B3/T4096 profile establishes that deterministic block-sweep historical read attention is the dominant historical bottleneck.

```text
Complete profiled F+B GPU event:       947.242 ms
Correlation-attributed history union:  426.235 ms
Historical share of complete F+B:       45.00%
Historical kernels:                     7,953
Unclassified historical duration:       1.605 ms (0.376%)
```

Deterministic block reads alone consume:

```text
Forward:   76.893 ms /   768 kernels
Backward: 192.060 ms / 1,968 kernels
Total:    268.953 ms / 2,736 kernels
```

That is `28.39%` of complete profiled F+B and approximately `63.1%` of correlation-attributed historical GPU duration.

## Repository and workload

```text
Git revision:       96ed2c3
Checkpoint:         ./checkpoints/0083_intermediate_20L_1024_fullmix_100k
Mixed data:         ./token_shards_ensemble_32k_mixed
Batch:              3
Working length:     4096
History lengths:    [0, 6162, 61440]
Document spans:     [1033, 11236, 1170271]
Target lengths:     [1032, 4096, 4096]
Supervised rows:    9224
Source:             finepdfs/file_00009/train_shard_00004.bin
Sampling attempts:  2
Input SHA256:       8bed9e8e4902ccf10b0635d43d6f05856ab62e9e43ede41a027c36d5e434d16d
Router seed:        293357562
```

The fixture is document-first and uses real preceding source tokens. No padded history was treated as valid and no dense 64K mask was introduced.

## Effective benchmark contract

The capture report records the required flags:

```text
MINI_LLM_BACKEND=cupy
MINI_LLM_LOCAL_DETAIL_PROFILE=0
MINI_LLM_MOE_DETAIL_PROFILE=0
MINI_LLM_OPTIMIZER_OFFLOAD=full
MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB=16
MINI_LLM_FUSED_EXPERT_GEMM=0
MINI_LLM_AR_COMPACT_TARGET=1
MINI_LLM_AR_ACCUM_LENGTH_BUCKETS=1
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM=1
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT=0
MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1
MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1
MINI_LLM_ROPE_PREPARED_POSITIONS=1
MINI_LLM_GPU_MEMORY_TRIM_TRAIN_INTERVAL=5
MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION=1
MINI_LLM_GPU_MEMORY_TRIM_AFTER_CHECKPOINT=1
MINI_LLM_GPU_MEMORY_AUDIT=0
```

The promoted computational path also used:

```text
MINI_LLM_TRAIN_ATTN_BACKEND=triton
MINI_LLM_TRAIN_ATTN_STRICT=1
MINI_LLM_FUSED_QKV_ROPE_FORWARD=1
MINI_LLM_HISTORY_RECOMPUTE=1
MINI_LLM_HISTORY_QUERY_CHUNK=1024
MINI_LLM_ACTIVATION_CHECKPOINT=none
MINI_LLM_SWIGLU_VEC2=1
```

The no-optimizer profiler never constructs an optimizer, so offload settings do not execute during this capture; they remain fixed and reported for parity with production configuration.

## Capture command

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=PROFILE_0087A_MICROBATCH@MyLocal_LLM_0087A \
  --capture-range-end=stop-shutdown \
  --force-overwrite=true \
  --output=benchmarks/results/0089o_seed48_history \
  python benchmarks/profile_0087a_training.py \
    --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --batch-size 3 --microbatches 1 --working-length 4096 \
    --seed 48 --warmup 2 --no-optimizer --capture \
    --runtime-profile gpu-fast --attention-backend triton \
    --out benchmarks/results/0089o_seed48_history_capture.json
```

Nsight exits with status 143 because `stop-shutdown` terminates the target after the registered range. The report was generated successfully.

Analysis command:

```bash
python benchmarks/analyze_0089o_historical_nsys.py \
  benchmarks/results/0089o_seed48_history.nsys-rep \
  --sqlite benchmarks/results/0089o_seed48_history.sqlite \
  --output-json benchmarks/results/0089o_seed48_history_analysis.json \
  --output-prefix benchmarks/results/0089o_seed48_history
```

## Accounting method

Raw CPU-NVTX/GPU timestamp overlap is not used for source attribution.

For each GPU event:

1. Find its CUDA API record through the CUPTI correlation ID.
2. Assign the CPU submission to the deepest containing historical NVTX range.
3. Attribute the complete asynchronous GPU event to that source range.
4. Preserve enclosing ranges as ancestry, allowing encoder forward and encoder recomputation to be separated.
5. Assign work submitted only inside a coarse historical parent to `historical.unclassified`.

GPU execution union and summed duration are reported separately. Summed durations are not treated as wall-clock latency because streams may overlap.

The old timestamp-overlap profile assigned `511.745 ms` to historical CPU windows. Correlation attribution measures `426.235 ms` for the current capture and leaves only `1.605 ms` of correlated historical activity unclassified. This confirms that timestamp overlap materially overstates source ownership when previously queued work executes inside historical CPU ranges.

## Phase breakdown

| Phase | GPU execution union | Summed duration | Kernels |
|---|---:|---:|---:|
| Forward | 119.754 ms | 120.023 ms | 2,727 |
| Encoder recomputation | 26.291 ms | 26.291 ms | 61 |
| Backward/close | 280.190 ms | 280.190 ms | 5,165 |
| **Historical total** | **426.235 ms** | **426.504 ms** | **7,953** |

## Ranked source regions

| Source region | Phase | GPU duration | Kernels | Complete-F+B share |
|---|---|---:|---:|---:|
| `historical.backward.block_read` | Backward | 192.060 ms | 1,968 | 20.28% |
| `historical.layer.block_read` | Forward | 76.893 ms | 768 | 8.12% |
| `historical.backward.sparse_read` | Backward | 17.783 ms | 1,968 | 1.88% |
| `historical.encoder.backward.ffn` | Backward | 17.504 ms | 23 | 1.85% |
| `historical.encoder.swiglu` | Forward | 8.379 ms | 6 | 0.88% |
| `historical.encoder.swiglu` | Recompute | 8.379 ms | 6 | 0.88% |
| `historical.encoder.backward.rmsnorm2` | Backward | 7.968 ms | 13 | 0.84% |
| `historical.encoder.backward.rmsnorm1` | Backward | 7.968 ms | 13 | 0.84% |
| `historical.encoder.backward.block_attention` | Backward | 6.626 ms | 20 | 0.70% |
| `historical.backward.learned_read` | Backward | 6.538 ms | 564 | 0.69% |
| `historical.encoder.backward.input_projection` | Backward | 5.339 ms | 6 | 0.56% |
| `historical.encoder.block_attention` | Recompute | 4.954 ms | 16 | 0.52% |
| `historical.encoder.block_attention` | Forward | 4.660 ms | 16 | 0.49% |
| `historical.layer.sparse_read` | Forward | 4.295 ms | 768 | 0.45% |
| `historical.close.learned_projection_backward` | Backward | 3.445 ms | 24 | 0.36% |
| `historical.layer.learned_read` | Forward | 3.340 ms | 180 | 0.35% |

### Aggregated attention modes

```text
Deterministic block reads:  268.953 ms
Deterministic sparse reads:  22.078 ms
Learned compact reads:        9.878 ms
```

The block-sweep path is therefore the only read mode with a double-digit complete-run optimization bound.

## Block-read kernel composition

Leading block-read families include:

| Operation | Phase | Duration | Launches |
|---|---|---:|---:|
| FP32 SIMT GEMM | Backward | 33.413 ms | 192 |
| FP32 copy | Backward | 29.762 ms | 384 |
| FP32 multiply | Backward | 26.210 ms | 192 |
| FP32 mask/where | Backward | 19.629 ms | 96 |
| FP32 scalar multiply | Backward | 18.835 ms | 96 |
| FP32 subtract | Backward | 17.298 ms | 96 |
| FP32 exponential | Forward | 15.997 ms | 48 |
| MAGMA batched SGEMM | Backward | 15.153 ms | 96 |
| FP32 mask/where | Forward | 14.415 ms | 96 |
| FP32 scalar multiply | Forward | 11.716 ms | 48 |
| FP32 SIMT GEMM | Forward | 9.513 ms | 48 |
| FP32 division | Backward | 9.257 ms | 48 |
| FP32 exponential | Backward | 8.313 ms | 48 |
| FP32 division | Forward | 6.935 ms | 48 |
| FP32 subtract | Forward | 6.691 ms | 48 |

The current graph repeatedly materializes selected K/V, scores, probabilities, softmax intermediates, and gradient preparation arrays. This is a data-movement and launch-fragmentation problem in addition to GEMM execution.

## Amdahl bound

Using the current measured `268.953 ms` block-read total:

| Hypothetical block-read reduction | Complete saving | Complete F+B reduction |
|---:|---:|---:|
| 25% | 67.238 ms | 7.10% |
| 50% | 134.477 ms | 14.20% |
| 67% | 180.199 ms | 19.02% |
| 100% theoretical ceiling | 268.953 ms | 28.39% |

A custom block-read kernel is justified by Amdahl's law. It must, however, preserve the FP32 reference closely enough that downstream BF16 layers and all parameter gradients remain within native parity limits.

## Memory

The profiled process reported:

```text
Peak CuPy pool reservation: 13,643,022,336 bytes
Device used at boundary:    15,364,587,520 bytes
Physical device memory:     16,600,465,408 bytes
```

These values are session-sensitive and should not be directly compared with a different process's allocator high-water. The memory constraint remains strict: no full score/probability cache and no full encoder activation cache.

## Optimization priority

1. Deterministic block-read streaming attention, with production-length numerical testing before integration.
2. Historical encoder FFN/SwiGLU and RMSNorm backward fusion.
3. Fixed 32-token encoder attention.
4. Sparse read only as a low-cost extension of a proven block kernel.
5. Learned compact read only after larger targets; its measured bound is about 1% of complete F+B.
6. Historical RoPE/projection grouping only if subsequent profiles expose a larger remaining leaf.

## Validation

```text
Profiler/analyzer tests: 7 passed, 1 skipped
Historical correctness tests: 32 passed
Analyzer diagnostics: clean
Unclassified historical duration: 0.376%
Fixture hashes: matched seed-48 reference
```

Artifacts:

```text
benchmarks/results/0089o_seed48_history.nsys-rep
benchmarks/results/0089o_seed48_history.sqlite
benchmarks/results/0089o_seed48_history_capture.json
benchmarks/results/0089o_seed48_history_analysis.json
benchmarks/results/0089o_seed48_history_ranges.csv
benchmarks/results/0089o_seed48_history_phases.csv
benchmarks/results/0089o_seed48_history_kernels.csv
benchmarks/results/0089o_seed48_history_events.csv
```
