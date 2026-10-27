# PATCH 0089V — Streaming deterministic historical block backward

## Decision

Retain the implementation as an explicit opt-in:

```bash
export MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1
export MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM=1
export MINI_LLM_TRITON_HISTORICAL_BLOCK_BACKWARD=1
```

On the production seed-48 history-heavy workload, the candidate reduces complete no-optimizer forward/backward GPU latency by **83.9–88.3 ms** across two independent process pairs. The median-of-medians reduction is **86.1 ms / 10.46%**, corresponding to a **1.117x** throughput increase for this fixed batch.

The implementation remains opt-in rather than a silent default because its accumulation order differs from the generic CuPy path. The direct local VJP agrees near `1e-7`, while the complete BF16 model amplifies those differences to global gradient relative L2 `0.003442`. Forward output and loss are bitwise identical when only the backward path is varied.

## Workload and fairness contract

All measurements use:

```text
checkpoint: checkpoints/0083_intermediate_20L_1024_fullmix_100k
dataset: token_shards_ensemble_32k_mixed
GPU: NVIDIA GeForce RTX 5080, 16 GB
CuPy: 14.2.0
Triton: 3.8.0
CUDA runtime: 13.2
B=3, T=4096, layers=20
seed=48
history tokens per sample=[0, 6162, 61440]
supervised rows=9224
computed rows=12288
input SHA-256=8bed9e8e4902ccf10b0635d43d6f05856ab62e9e43ede41a027c36d5e434d16d
warmup=3, measured repeats=10
```

The benchmark reports confirmed the following effective flags in both arms:

```bash
export MINI_LLM_BACKEND=cupy
export MINI_LLM_LOCAL_DETAIL_PROFILE=0
export MINI_LLM_MOE_DETAIL_PROFILE=0
export MINI_LLM_OPTIMIZER_OFFLOAD=full
export MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB=16
export MINI_LLM_FUSED_EXPERT_GEMM=0
export MINI_LLM_AR_COMPACT_TARGET=1
export MINI_LLM_AR_ACCUM_LENGTH_BUCKETS=1
export MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM=1
export MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT=0
export MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1
export MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1
export MINI_LLM_ROPE_PREPARED_POSITIONS=1
export MINI_LLM_GPU_MEMORY_TRIM_TRAIN_INTERVAL=5
export MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION=1
export MINI_LLM_GPU_MEMORY_TRIM_AFTER_CHECKPOINT=1
export MINI_LLM_GPU_MEMORY_AUDIT=0
export MINI_LLM_TRAIN_ATTN_BACKEND=triton
export MINI_LLM_TRAIN_ATTN_STRICT=1
export MINI_LLM_FUSED_QKV_ROPE_FORWARD=1
export MINI_LLM_HISTORY_RECOMPUTE=1
export MINI_LLM_HISTORY_QUERY_CHUNK=1024
export MINI_LLM_ACTIVATION_CHECKPOINT=none
export MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1
export MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM=1
```

Only `MINI_LLM_TRITON_HISTORICAL_BLOCK_BACKWARD` changed between baseline (`0`) and candidate (`1`). Fixture hashes are identical in each pair.

## Implementation

Files:

```text
mini_llm/ops/triton_historical_block.py
mini_llm/ops/historical_read.py
benchmarks/profile_0087a_training.py
benchmarks/benchmark_0089r_block_forward_parity.py
```

The accepted streaming forward optionally retains the minimum state required by the VJP:

```text
FP32 output: [B,Q,64]
FP32 base-2 LSE: [B,Q]
first selected local block
selected token count
selection period
```

The backward replaces each generic CuPy block-read graph with three Triton launches:

1. Compute row delta and atomically accumulate the scalar gate gradient.
2. Traverse queries key-major and add `dK`/`dV` into the shared FP32 destinations.
3. Traverse selected keys query-major, add `dQ`, and clear active `dcontext` rows.

Working tile geometry on RTX 5080:

```text
dK/dV: BLOCK_M=32, BLOCK_N=64
dQ:    BLOCK_M=64, BLOCK_N=32
```

The smaller tiles are required by the device's `101,376 B` shared-memory limit. Earlier `64x64` variants required `123,392 B` for `dK/dV` and `114,688 B` for `dQ`.

Sparse, learned, router, encoder, shared fan-in, compact selected-context, validity, EOS, and causal-history paths are unchanged.

## Complete F+B timing

GPU-event and outer synchronized host-wall intervals include complete numerical forward and backward. They exclude setup, checkpoint loading, compilation, warmup, `zero_grad`, and optimizer work.

### Independent process pair 1: baseline then candidate

| Metric | Generic CuPy backward | Triton backward | Change |
|---|---:|---:|---:|
| GPU median | 824.536 ms | 736.268 ms | **-88.268 ms / -10.71%** |
| GPU MAD | 4.475 ms | 2.863 ms | -1.612 ms |
| GPU p10 | 820.508 ms | 733.226 ms | -87.283 ms |
| GPU p90 | 834.859 ms | 747.996 ms | -86.863 ms |
| Host-wall median | 824.865 ms | 736.367 ms | **-88.498 ms** |
| Supervised rows/s | 11,187 | 12,528 | **+11.99%** |

### Independent process pair 2: candidate then baseline

| Metric | Generic CuPy backward | Triton backward | Change |
|---|---:|---:|---:|
| GPU median | 821.144 ms | 737.215 ms | **-83.928 ms / -10.22%** |
| GPU MAD | 1.772 ms | 3.313 ms | +1.541 ms |
| GPU p10 | 819.748 ms | 733.783 ms | -85.965 ms |
| GPU p90 | 831.679 ms | 746.775 ms | -84.904 ms |
| Host-wall median | 821.411 ms | 737.389 ms | **-84.022 ms** |
| Supervised rows/s | 11,233 | 12,512 | **+11.38%** |

Median of the two process medians:

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Complete 20-layer F+B | 822.840 ms | 736.741 ms | **-86.098 ms / -10.46%** |
| Fixed-batch supervised rows/s | 11,210 | 12,520 | **+11.69%** |
| Speedup | 1.000x | **1.117x** | |

This is no-optimizer model throughput, not a measurement of a Hybrid-Muon optimizer step. Actual production training throughput remains a separate validation item.

## Nsight mechanism audit

Artifacts:

```text
benchmarks/results/0089u_seed48_triton_block.nsys-rep
benchmarks/results/0089u_seed48_triton_block_analysis.json
benchmarks/results/0089v_seed48_backward_candidate_profile.nsys-rep
benchmarks/results/0089v_seed48_backward_candidate_profile.sqlite
benchmarks/results/0089v_seed48_backward_candidate_profile_analysis.json
benchmarks/results/0089v_seed48_backward_candidate_profile_{ranges,phases,kernels,events}.csv
```

The profile confirms that the measured complete-F+B gain is caused by eliminating the generic block-read backward graph:

| Deterministic block backward | Generic CuPy | Streaming Triton | Change |
|---|---:|---:|---:|
| GPU execution union | 172.393 ms | 77.914 ms | **-94.479 ms / -54.80%** |
| Kernel launches | 1,968 | 36 | **-1,932 / -98.17%** |
| Launches per 12 block slots | 164 | 3 | |

Candidate kernel composition:

| Kernel | Launches | Total GPU time | Median per launch |
|---|---:|---:|---:|
| `_backward_dq_clear_kernel` | 12 | 39.924 ms | 3.257 ms |
| `_backward_dkdv_kernel` | 12 | 37.894 ms | 3.158 ms |
| `_backward_delta_kernel` | 12 | 0.097 ms | 0.007 ms |

The baseline range spent `128.093 ms` in non-GEMM work and launched hundreds of materialized score/probability elementwise, copy/indexing, and reduction kernels. Those operations disappear from the candidate block-read range. Nsight capture durations are profiler-perturbed and are not used as steady-state latency measurements.

Historical execution across all subsystems fell from `351.576 ms / 7,197 kernels` to `238.479 ms / 5,157 kernels` in the two captures. Some of that cross-capture difference is unrelated run variation; the directly attributed block-read range is the defensible mechanism comparison.

## Correctness

### Direct one-slot VJP

The local test uses non-contiguous BF16 queries, prefilled additive gradient destinations, and an all-invalid sample:

| Tensor | Relative L2 |
|---|---:|
| Forward | `2.80e-7` |
| `dQ` | `4.83e-7` |
| `dK` | `2.11e-7` |
| `dV` | `9.49e-8` |
| `dcontext` | exact |
| Gate gradient | exact |

All outputs and gradients are finite.

### Complete production-shape backward-only comparison

Artifact:

```text
benchmarks/results/0089v_seed48_backward_parity.json
```

The Triton forward is fixed in both arms; only the backward implementation changes.

| Metric | Result |
|---|---:|
| Forward body | bitwise identical |
| Loss | bitwise identical (`1.999375581741333`) |
| Global gradient relative L2 | `0.00344196` |
| Worst parameter cosine | `0.99994223` |
| Worst per-tensor relative L2 | `0.00640546` (`blocks.3.attention.Wk`) |
| Worst absolute element difference | `4.768e-6` for the worst-relative tensor |
| Bitwise-equal parameter gradients | `102 / 217` |
| NaN/Inf | none |

The old strict native threshold (`5e-4` global relative L2 and `0.99999` worst cosine) therefore reports `failed`. This is documented rather than hidden.

Evidence against an indexing or computational-graph defect:

- the isolated VJP agrees near `1e-7` for every derivative destination;
- complete forward and loss are bitwise identical;
- all complete-model gradients remain finite and strongly aligned;
- affected tensors are the expected downstream attention projections;
- the candidate changes reduction/tensor-core accumulation order relative to the many-kernel CuPy implementation;
- the backward-only discrepancy is substantially smaller than the already accepted streaming-forward global gradient difference (`0.003442` versus `0.013625`).

The current evidence supports floating-point accumulation-order amplification through BF16 layers, not a missing gradient path.

## Memory

The retained cache is approximately `36.6 MiB` across all 12 deterministic block slots. CuPy pool high-water is allocator-sensitive:

| Run | Baseline | Candidate | Candidate minus baseline |
|---|---:|---:|---:|
| Short smoke | 13.672 GB | 14.075 GB | +0.403 GB |
| Pair 1 | 14.427 GB | 14.075 GB | -0.352 GB |
| Pair 2 reversed | 14.427 GB | 14.130 GB | -0.297 GB |

The longer fresh-process pairs do not reproduce the smoke-run increase. The implementation has a real `~36.6 MiB` live-cache requirement, but the observed pool reservation difference is not stable enough to attribute hundreds of megabytes to the candidate.

## Reproduction

After exporting the fairness environment shown above, run each arm in a fresh process:

```bash
MINI_LLM_TRITON_HISTORICAL_BLOCK_BACKWARD=0 python benchmarks/profile_0087a_training.py \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --batch-size 3 --microbatches 1 --working-length 4096 \
  --seed 48 --warmup 3 --repeats 10 \
  --runtime-profile gpu-fast --attention-backend triton --no-optimizer \
  --out benchmarks/results/0089v_seed48_backward_baseline.json

MINI_LLM_TRITON_HISTORICAL_BLOCK_BACKWARD=1 python benchmarks/profile_0087a_training.py \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --batch-size 3 --microbatches 1 --working-length 4096 \
  --seed 48 --warmup 3 --repeats 10 \
  --runtime-profile gpu-fast --attention-backend triton --no-optimizer \
  --out benchmarks/results/0089v_seed48_backward_candidate.json
```

Backward-only full-model parity:

```bash
python benchmarks/benchmark_0089r_block_forward_parity.py \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --seed 48 --working-length 4096 --target backward \
  --out benchmarks/results/0089v_seed48_backward_parity.json
```

Focused validation:

```bash
python -m py_compile \
  mini_llm/ops/triton_historical_block.py \
  mini_llm/ops/historical_read.py \
  benchmarks/benchmark_0089r_block_forward_parity.py \
  benchmarks/profile_0087a_training.py \
  benchmarks/analyze_0089o_historical_nsys.py

PYTHONPATH=. pytest -q \
  tests/test_0083b_historical_read.py \
  tests/test_0089o_historical_nsys.py \
  tests/test_0087a_profile_ranges.py
```

Result: `30 passed, 1 skipped`.

## Remaining opportunity

The deterministic backward has been reduced by `54.8%`, exceeding the pre-implementation 40–50% target. The two remaining streaming kernels account for `77.818 ms` of its `77.914 ms` profile, so further work must improve their traversal rather than fuse miscellaneous launches.

The next largest unchanged historical range is sparse backward (`~13.2 ms`, `1,968` launches). Eliminating that entire range would bound complete no-optimizer improvement near `1.8%`; a realistic partial reduction is smaller. The larger possible target is further optimization of the deterministic `dQ` and `dK/dV` kernels, but any redesign must preserve additive shared-gradient fan-in and the compact selected-context representation. Nsight Compute should be used next to determine whether those kernels are limited by tensor-core throughput, memory traffic, or occupancy before changing their tile geometry.
