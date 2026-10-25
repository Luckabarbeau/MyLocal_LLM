# PATCH 0089O/P — Historical bottleneck audit and custom-kernel plan

## Decision

Continue optimizing the native CuPy historical-read implementation. Do not begin with a monolithic historical kernel and do not begin with the learned top-8 path alone.

The first implementation target, subject to a current seed-48 correlation-correct profile, is the deterministic block-sweep historical read attention. The existing graph materializes gathered K/V, full FP32 score matrices, softmax probabilities, and several backward intermediates for attention over approximately 5,120 selected historical tokens per slot. Six layers execute these block-sweep heads. This has a much larger measured bound than the 256-token learned read.

No historical computational path is changed by this plan.

## Preserved constraints

The work must preserve:

- the existing 32-token historical encoder;
- deterministic block and sparse source-coordinate schedules;
- learned top-8 block retrieval and its route-score bias;
- compact selected context, with no dense 64K masks or giant padded contexts;
- document/EOS validity and no future-token leakage;
- FP32 historical attention and shared gradient accumulation semantics unless a separately gated mixed-precision experiment is justified;
- BF16 projection boundaries;
- one shared historical encoder backward after all per-layer gradient fan-in;
- recomputation required by the 16 GB memory budget;
- packed QKV and checkpoint compatibility;
- Hybrid-Muon full CPU offload behavior.

## Established production geometry

For the 20-layer, 16-head checkpoint:

```text
Historical block size:            32
Searchable history:               up to 65,536 tokens
Seed-48 cropped heavy history:    61,440 tokens
Historical encoder blocks:        1,920 per sample
Historical encoder heads:         4
Historical encoder head width:    32
Current query length:              4,096
Route stride:                      128
Routes per sample:                 32
Learned selected blocks:           8
Learned selected tokens:           256
Historical read head width:        64
```

The schedule contains 36 historical slots:

```text
12 deterministic block-sweep slots
12 deterministic sparse slots
12 learned slots in six layers, using heads 14 and 15
```

A deterministic block-sweep slot selects approximately:

```text
61,440 / 384 * 32 = 5,120 historical tokens
```

A sparse slot selects approximately:

```text
61,440 / 128 = 480 historical tokens
```

A learned slot selects exactly 256 token positions before validity masking.

## Existing evidence and accounting correction

### Complete measurements

The current promoted seed-48 baseline is:

```text
History lengths:                  [0, 6162, 61440]
Complete F+B GPU median:          928.372 ms
Complete F+B host median:         928.545 ms
CuPy pool high-water:             14,091,812,864 bytes
Loss:                             2.0288021564483643
```

The archived authoritative heavy-history capture used `[0, 5657, 61440]` and measured:

```text
Complete F+B median:              855.311 ms
Complete GPU execution union:     833.015 ms
Historical CPU-window GPU union:  511.745 ms
Historical CPU elapsed union:     529.930 ms
```

The `511.745 ms` value is not an exclusive source attribution. CUDA work is asynchronous: work submitted before a CPU NVTX range can execute inside that range, and work submitted inside it can execute after the range closes.

### Correlation-ID attribution of the archived capture

A planning-only reanalysis assigned each CUDA API call to the historical NVTX range containing its CPU submission, then mapped the associated GPU operation through its CUPTI correlation ID. This is more defensible than raw timestamp overlap, but the old labels remain coarse and their semantic identity is inferred from call order.

| Coarse source group | Correlated GPU union | Kernel launches | Qualification |
|---|---:|---:|---|
| Historical `open` | 27.332 ms | 157 | Encoder forward plus history-wide projections and summaries |
| Sparse-only forward layers | 1.723 ms | 276 | Two sparse heads per layer |
| Block-containing forward layers | 76.912 ms | 1,600 | Includes sparse/learned companions in some layers |
| Learned-only forward layers | 3.472 ms | 464 | Two learned heads per layer |
| Learned-only backward layers | 6.071 ms | 695 | Includes router/read backward |
| Block-containing backward layers | 189.012 ms | 3,812 | Includes sparse/learned companions in some layers |
| Sparse-only backward layers | 4.947 ms | 694 | Two sparse heads per layer |
| Historical `close` | 77.153 ms | 255 | Projection fan-in, recompute, encoder backward |
| **Total correlated historical work** | **386.622 ms** | — | Does not equal the old timestamp-overlap value by design |

The six block-containing layers account for `265.924 ms` before subtracting their sparse and learned companions. Using pure sparse/learned layers as an approximate subtraction puts deterministic block-sweep read attention near `~248 ms` in the archived run. This is an estimate, not a current seed-48 measurement.

The historical-window kernel mix also establishes that the path is dominated by fragmented FP32 work and materialization:

```text
FP32 SIMT 64x64 GEMM:              41.023 ms / 401 launches
FP32 copies:                        37.450 ms / 908 launches
FP32 multiply(array,array):         28.102 ms / 380 launches
FP32 where:                         27.007 ms / 312 launches
FP32 multiply(array,scalar):        24.620 ms / 243 launches
FP32 subtract:                      21.340 ms / 248 launches
MAGMA batched SGEMM:                19.624 ms / 242 launches
FP32 exp:                           17.948 ms / 157 launches
FP32 add:                           12.791 ms / 476 launches
FP32 divide:                        12.501 ms / 157 launches
```

These family totals span multiple historical operations. They establish a fusion opportunity but do not identify which source region owns each family.

## Stage 1 — PATCH 0089O correlation-correct historical audit

### Instrumentation

Add asynchronous NVTX-only ranges around these regions. The normal training path must not synchronize at range boundaries.

```text
historical.open.validity_crop
historical.encoder.forward
historical.encoder.embedding_gather
historical.encoder.input_projection
historical.encoder.rmsnorm1
historical.encoder.qkv_projection
historical.encoder.block_attention
historical.encoder.output_projection_residual
historical.encoder.rmsnorm2
historical.encoder.ffn_projections
historical.encoder.swiglu
historical.encoder.down_projection_residual
historical.encoder.validity_mask
historical.shared_kv_projection
historical.shared_rope
historical.router_summary
historical.router_key_projection
historical.learned_kv_projection
historical.learned_rope
historical.layer.route
historical.layer.block_read
historical.layer.sparse_read
historical.layer.learned_gather
historical.layer.learned_read
historical.layer.context_replace
historical.backward.block_read
historical.backward.sparse_read
historical.backward.learned_read
historical.backward.router
historical.close.shared_projection_backward
historical.close.learned_projection_backward
historical.close.router_summary_backward
historical.encoder.recompute
historical.encoder.backward
historical.encoder.backward.ffn
historical.encoder.backward.block_attention
historical.encoder.backward.qkv
historical.encoder.backward.input_projection
historical.close.embedding_scatter
```

Parent ranges may be retained for readability, but the report must distinguish parent-inclusive and leaf-exclusive accounting.

### Analyzer contract

Create `benchmarks/analyze_0089o_historical_nsys.py` by reusing the schema-discovery and interval utilities from `benchmarks/analyze_0087a_nsys.py`.

For each GPU event:

1. Find the CUDA runtime/driver API record with the matching correlation ID.
2. Assign that API record to the deepest historical NVTX range containing its CPU submission.
3. Attribute the complete GPU event to that source range even if execution occurs after the CPU range closes.
4. Keep events without a usable correlation or range as `historical.unclassified`; do not guess.
5. Report both summed GPU duration and the union of GPU intervals. Summed duration can exceed wall time across streams and must not be described as latency.
6. Report memcpy and memset separately from kernels.
7. Report range submission time separately from GPU execution time.
8. Verify that leaf-attributed events plus the unclassified remainder reconcile with the parent historical attribution.

The report must include:

- execution union, summed duration, launch count, and memcpy/memset bytes per leaf range;
- the top 20 kernel families per leaf range;
- median launch duration per kernel family;
- per-layer and per-slot-kind totals;
- forward, recompute, and backward separation;
- unclassified duration and percentage;
- complete-microbatch wall/GPU-event latency and peak memory;
- exact fixture hashes and retrieved learned block identities.

### Fixed workload

Use the real document-first seed-48 fixture:

```text
Batch:                             3
Working length:                    4096
History lengths:                   [0, 6162, 61440]
Document spans:                    [1033, 11236, 1170271]
Sampling attempts:                 2
Source:                            finepdfs/file_00009/train_shard_00004.bin
Input SHA256:                      8bed9e8e4902ccf10b0635d43d6f05856ab62e9e43ede41a027c36d5e434d16d
```

The checkpoint and mixed-data paths were verified present on 2026-10-25.

### Fair-run environment

Every result must record and enforce:

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
```

For the current promoted computational baseline also use:

```bash
export MINI_LLM_TRAIN_ATTN_BACKEND=triton
export MINI_LLM_TRAIN_ATTN_STRICT=1
export MINI_LLM_FUSED_QKV_ROPE_FORWARD=1
export MINI_LLM_HISTORY_RECOMPUTE=1
export MINI_LLM_HISTORY_QUERY_CHUNK=1024
export MINI_LLM_ACTIVATION_CHECKPOINT=none
```

Capture command from the repository root:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=PROFILE_0087A_MICROBATCH@MyLocal_LLM_0087A \
  --capture-range-end=stop-shutdown \
  --output=benchmarks/results/0089o_seed48_history \
  python benchmarks/profile_0087a_training.py \
    --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --batch-size 3 --microbatches 1 --working-length 4096 \
    --seed 48 --warmup 2 --no-optimizer --capture \
    --runtime-profile gpu-fast --attention-backend triton \
    --out benchmarks/results/0089o_seed48_history_capture.json
```

## Stage 2 — PATCH 0089P deterministic block-read streaming attention

Proceed only if the current 0089O audit confirms that deterministic block-read forward/backward remains the largest removable leaf cost.

### Current graph to replace

For each block-sweep slot, the current path performs:

```text
Materialize selected K/V using a 32-of-384 source-coordinate pattern
For each query chunk:
    FP32 QK GEMM against approximately 5,120 keys
    Materialize full score matrix
    Apply validity mask
    max / exp / sum / divide softmax sequence
    Materialize probability matrix
    FP32 P@V GEMM
Backward:
    Recreate scores and probabilities
    Materialize output for gate gradient
    DP, DV, softmax VJP, DQ, DK GEMMs
    Add/scatter gradients into shared stores
```

At B3/T4096, a single 1,024-query score tile against 5,120 keys contains 15,728,640 FP32 elements, approximately 60 MiB. Scores and probabilities are transient and repeatedly materialized.

### Specialized forward kernel

Implement an opt-in Triton path, initially for `slot.kind == "block"` only.

The kernel should:

- consume Q and the original full-history K/V stores directly;
- compute deterministic source indices from `phase`, `period`, block size, and cropped offset instead of materializing index arrays or gathered K/V;
- use tiled online softmax, never materializing the full score or probability matrix;
- apply each sample's historical validity in the kernel;
- accumulate in FP32;
- produce the FP32 read output;
- optionally apply the scalar head gate and write the designated context head only after aliasing safety is established;
- optionally retain one FP32 log-sum-exp value per query row for backward.

Caching LSE for all 12 block slots costs approximately:

```text
3 * 4096 * 12 * 4 bytes = 589,824 bytes
```

This is acceptable if it measurably improves backward. No score/probability cache is acceptable.

### Specialized backward

Implement matching backward without materializing full scores/probabilities:

- recompute attention tiles from Q/K and optional LSE;
- compute gate gradient;
- compute DQ in FP32;
- accumulate DK/DV into the existing shared FP32 history stores;
- preserve accumulation across layers and slots;
- use atomic FP32 fan-in or a bounded compact partial buffer, selected by measurement;
- do not change the final single encoder backward.

The first implementation should preserve the existing context replacement as a separate operation. Fuse gate/context replacement only if 0089O measures it as material and ownership analysis proves that in-place writes cannot destroy data needed by backward.

### Reuse

Reuse the tiling and numerical lessons from the production Triton mixed-attention kernels, but do not force the current-attention interface onto historical reads. Historical selection is source-coordinate based and has different K lengths, masks, and gradient fan-in.

### Amdahl gate

The archived planning estimate for deterministic block reads is approximately `248 ms` out of an 855 ms complete run:

| Hypothetical block-read reduction | Estimated complete saving | Estimated complete reduction |
|---:|---:|---:|
| 25% | ~62 ms | ~7.2% |
| 50% | ~124 ms | ~14.5% |
| 67% | ~166 ms | ~19.4% |

These are not predictions. Recompute the table from the current 0089O seed-48 measurement before implementation.

Implementation gate:

- proceed if the measured block-read leaf cost is at least 10% of complete F+B and a streaming implementation can plausibly remove at least half of its materialization/kernel cost;
- retain the kernel only if alternating complete seed-48 runs improve median GPU and host latency by at least 5%, or by at least 3% with a clear memory reduction and non-overlapping robust variability;
- reject it if it only improves an isolated read benchmark.

## Stage 3 — Extend or move to the next measured target

### 3A. Sparse historical reads

The sparse path uses an implicit stride-128 index pattern and is structurally compatible with streaming attention. Add it only after the block mode passes. The archived pure sparse layers account for only `6.670 ms` combined forward/backward, so it is not independently worth a complex kernel unless the current audit contradicts that result.

### 3B. Learned top-8 reads

The learned path has fixed production geometry:

```text
B=3, routes=32, query rows per route=128,
selected blocks=8, selected tokens=256, head width=64
```

A learned-mode kernel can fuse:

```text
rank-based K/V gather
validity
0.25 * selected route-score bias
QK
online softmax
P@V
```

Backward must produce:

```text
DQ
DK/DV scatter into learned stores
selected route-score bias gradient
head-gate gradient
```

The archived learned-only ranges account for about `9.543 ms` combined forward/backward. Implement this mode only if current profiling shows a larger cost, or if it is a low-complexity extension of the proven block kernel.

Do not fuse top-k routing into read attention initially. Routing has a different full-candidate reduction and a surrogate backward; keeping that boundary simplifies correctness.

### 3C. Historical encoder attention

`open + close` account for `104.485 ms` in the archived correlation analysis, but this includes history-wide projections, recomputation, encoder attention, FFN work, and parameter gradients.

If 0089O identifies the fixed 32-token encoder attention as the largest leaf inside this group, implement dedicated forward/backward kernels for:

```text
B * 1,920 independent blocks
4 heads
32 queries
32 keys
head width 32
```

Forward should fuse QK, validity mask, stable softmax, P@V, and output layout conversion. Backward should recompute probabilities and write packed DQKV directly, avoiding `stack + transpose + reshape` materialization.

Keep the surrounding BF16 cuBLAS projections unless profiling proves they are slower than a replacement.

### 3D. Encoder elementwise fusion

If RMSNorm, residual, validity, SiLU×up, dtype conversion, or layout materialization dominate, introduce narrow kernels around the existing GEMMs. Candidate sequences are:

```text
RMSNorm + validity + BF16 projection-input preparation
SiLU(gate) * up
residual add + validity mask
softmax VJP + scale
packed DQKV layout write
```

Each fusion must have a measured complete-run bound. Do not repeat an isolated residual/norm experiment without historical-specific evidence.

### 3E. Historical RoPE and learned projection grouping

Prepare historical sine/cosine factors once for the cropped position range and share them across shared K, both learned K heads, and inverse rotations only if historical RoPE is a measured leaf cost.

Packing learned-head K projections and learned-head V projections may reduce repeated full-history reads. This is distinct from the failed FFN gate/up packing experiment, but it still requires an isolated and complete-model benchmark.

### 3F. BF16 historical backward GEMMs

Revisit PATCH 0089F only if the current 61K audit shows FP32 history projection/backward GEMMs dominate. The short-history result was negative and is not evidence for promotion. Any 61K experiment must retain FP32 parameter-gradient accumulation and pass complete optimizer-step parity.

## Correctness gates

### Direct kernel parity

Cover:

- block periods and phases;
- cropped and uncropped history;
- one row with no history, one short row, and one 61K row;
- partially valid first blocks;
- sequence lengths that do and do not fill query tiles;
- duplicate K/V destinations across layer gradient fan-in;
- recomputation enabled and disabled.

Start with these native-to-native targets:

```text
Forward relative L2:              <= 1e-4
Input/DQ relative L2:             <= 1e-4
DK/DV and parameter grad rel L2:  <= 5e-4
Cosine similarity:                >= 0.99999
No unexpected NaN/Inf
```

These are initial upper limits, not permission to accept unexplained discrepancies. Tighten them when the arithmetic order is intended to be identical. If atomics introduce run-to-run variation, measure and document it across repeated executions.

### Full historical parity

Compare all of:

- forward body output and loss;
- working-input/embedding gradient;
- historical encoder input projection, both RMSNorm gammas, QKV, output, gate, up, and down gradients;
- shared historical key/value gradients;
- both learned-head key/value gradients;
- router-key gradient;
- every historical head-gate gradient;
- retrieved learned block identities and validity;
- route scores and route-gradient fan-in;
- two-microbatch accumulation;
- one real Hybrid-Muon optimizer step with full CPU offload;
- checkpoint save/resume after the optimizer step.

No scalar-loss-only acceptance is allowed.

## Performance and memory protocol

For every candidate:

1. Run a direct operation microbenchmark.
2. Run a tiny historical model complete F+B parity test.
3. Run the exact seed-48 checkpoint workload.
4. Alternate baseline and candidate across at least three independent processes.
5. Use at least 10 warmups and 30 measured samples per unprofiled process where memory and runtime permit.
6. Report GPU-event median, host-wall median, p10/p90, MAD, kernel count, and peak pool reservation.
7. Run one Nsight capture per mode only after unprofiled timing establishes a real difference.
8. Run a real fixed-configuration Hybrid-Muon training step and report supervised tokens/s.

Memory gate:

- target no increase in the current `14.092 GB` CuPy pool high-water;
- reject an unexplained increase above 128 MiB or 1%, whichever is smaller;
- never cache full score/probability matrices or full encoder intermediates;
- any bounded cache must include an explicit byte calculation.

## Stop/go order

```text
1. Implement 0089O fine-grained ranges and correlation analyzer.
2. Capture current seed-48 baseline and publish exclusive leaf accounting.
3. If block-sweep read attention remains dominant, implement block-mode streaming FWD.
4. Validate FWD directly, then implement matching BWD once.
5. Run full parity and alternating seed-48 benchmarks.
6. Promote only if the complete-run and memory gates pass.
7. Extend to sparse/learned modes only if their measured remaining bound justifies it.
8. Otherwise move to fixed 32-token encoder attention.
9. Apply elementwise/RoPE/projection changes only in descending measured order.
```

If the first block-read implementation plus one localized correction fails correctness or does not clear the performance gate, stop and report the exact remaining failure rather than cycling through alternate rewrites.

## Expected next deliverables

```text
benchmarks/analyze_0089o_historical_nsys.py
tests/test_0089o_historical_nsys.py
docs/0089O_HISTORICAL_BOTTLENECK_AUDIT.md
benchmarks/results/0089o_seed48_history_capture.json
benchmarks/results/0089o_seed48_history_analysis.json
benchmarks/results/0089o_seed48_history_ranges.csv
benchmarks/results/0089o_seed48_history_top_kernels.csv
```

Only after those measurements identify and bound the first production target:

```text
mini_llm/ops/historical_read.py
focused Triton/CUDA kernel module if separation is justified
tests for direct and complete historical gradient parity
docs/0089P_STREAMING_HISTORICAL_READ_ATTENTION.md
```
