# PATCH 0089K — Comparative depth-20 GPU bottleneck audit

## Scope and repository state

This audit compares the existing PATCH 0089E B3/T2304 fixed-depth Transformer stack on the RTX 5080. The graph is 20 complete production blocks with numerical forward and backward, FP32 residuals/parameter-gradient accumulation, BF16 branch compute, strict custom Triton mixed attention, and no optimizer, LM head, historical reads, or activation checkpointing.

Initial repository audit at `96ed2c3` found:

- 0089F, 0089G, and 0089I are unapplied patch files.
- 0089H is not active in production; its opt-in environment dispatch is absent. Shared low-level BF16 GEMM support has diverged from the patch.
- 0089J comparison infrastructure is applied in the working tree; its benchmark/doc/test files remain untracked.
- Production source was therefore equivalent to the validated pre-0089F behavior before 0089L.
- The checkpoint and exact fixture exist.
- `token_shards_ensemble_32k_mixed` contains the real mixed shards; no substitute data was generated.

Environment:

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB |
| Driver | 580.178.04 |
| Python | 3.12.3 |
| CuPy | 14.2.0 |
| CuPy CUDA runtime | 13.2 |
| Torch | 2.14.1+cu130 |
| Triton | 3.8.0 |
| Nsight Systems / Compute | installed under `/usr/local/cuda/bin` |

Nsight Compute hardware counters were unavailable to the user account (`ERR_NVGPUCTRPERM`). No register-pressure, achieved-occupancy, Tensor Core utilization, or DRAM-bandwidth percentage is claimed.

## Implementation

Benchmark-only changes add:

- `--capture` to `benchmarks/benchmark_0089e_block_stack.py`;
- one warmed `cudaProfilerApi` capture range named `0089K_STACK_FWD_BWD`;
- separate forward, backward, and Torch gradient-export NVTX ranges;
- non-synchronizing NVTX translation of existing CuPy block, attention, RMSNorm, and FFN detail scopes;
- `benchmarks/analyze_0089k_block_stack_nsys.py`, which reuses the correlation-aware SQLite machinery in `analyze_0087a_nsys.py`.

The analyzer:

1. selects CUDA activities by runtime correlation ID inside the F+B capture;
2. attributes CuPy kernels to the innermost active source NVTX range;
3. classifies exact GPU activities, not inclusive NVTX duration;
4. reports global interval union separately from summed operation durations;
5. keeps ambiguous symbols as `Unknown`;
6. emits per-category CSV and top-20 kernel-family CSV files.

Nested NVTX ranges are never added together. Category duration is the sum of mutually classified GPU activity durations. Category union is also retained because streams could overlap. In these captures, summed all-event duration and global union differ by only `0.014 ms` for CuPy and are identical at displayed precision for Torch.

## Reproduction

Authoritative unprofiled timing:

```bash
.venv/bin/python benchmarks/benchmark_0089e_block_stack.py \
  --backend cupy \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 20 --warmup 5 --repeats 20 \
  --output-json benchmarks/results/0089k_depth20_cupy_baseline.json

.venv-0089b-torch/bin/python benchmarks/benchmark_0089e_block_stack.py \
  --backend torch-compile \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 20 --warmup 5 --repeats 20 \
  --output-json benchmarks/results/0089k_depth20_torch_baseline.json
```

Matched depth-20 capture, replacing the backend/environment for the Torch command:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true \
  --output=benchmarks/results/0089k_depth20_cupy_profile \
  .venv/bin/python benchmarks/benchmark_0089e_block_stack.py \
    --backend cupy \
    --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
    --depth 20 --warmup 5 --repeats 1 --capture \
    --output-json benchmarks/results/0089k_depth20_cupy_capture.json

nsys export --type sqlite --force-overwrite=true \
  --output=benchmarks/results/0089k_depth20_cupy_profile.sqlite \
  benchmarks/results/0089k_depth20_cupy_profile.nsys-rep
```

Comparative analysis:

```bash
.venv/bin/python benchmarks/analyze_0089k_block_stack_nsys.py \
  --cupy benchmarks/results/0089k_depth20_cupy_profile.sqlite \
  --torch benchmarks/results/0089k_depth20_torch_profile.sqlite \
  --output-json benchmarks/results/0089k_depth20_comparative_analysis.json \
  --output-prefix benchmarks/results/0089k_depth20_comparative
```

Profiler-perturbed CUDA-event values in the capture JSON include profiler start/stop overhead and are not latency evidence. The SQLite F+B boundaries and the separate unprofiled measurements are the relevant values.

## Authoritative latency reproduction

| Depth | Backend | GPU median | p10 | p90 | MAD | Host median |
|---:|---|---:|---:|---:|---:|---:|
| 8 | CuPy | 78.785 ms | 78.506 | 80.668 | 0.243 | 79.216 ms |
| 8 | compiled Torch | 70.289 ms | 68.479 | 85.240 | 1.818 | 70.318 ms |
| 20 | CuPy | **197.696 ms** | 196.752 | 201.129 | 0.899 | 198.645 ms |
| 20 | compiled Torch | **171.891 ms** | 171.204 | 174.686 | 0.640 | 171.962 ms |

Depth-20 gap:

```text
absolute:                   25.805 ms
CuPy excess versus Torch:   15.01%
required CuPy reduction:    13.05%
```

The reproduced gap is within about 1 ms of the prior `24.750 ms` result.

## Depth-20 GPU accounting

| Metric | CuPy | compiled Torch | Difference |
|---|---:|---:|---:|
| Correlation-projected F+B boundary | 198.976 ms | 174.363 ms | +24.613 ms |
| GPU kernel/memcpy/memset union | 187.332 ms | 172.022 ms | +15.310 ms |
| Summed kernel duration | 186.610 ms | 171.978 ms | +14.632 ms |
| GPU execution-gap proxy | 11.644 ms | 2.341 ms | +9.303 ms |
| Kernels | 1,540 | 860 | +680 |
| Memcopies | 100 | 0 | +100 |
| Memsets | 100 | 60 | +40 |
| All GPU events | 1,740 | 920 | +820 |
| Candidate host-starved time | 6.674 ms | 0.432 ms | +6.242 ms |

The execution-gap proxy is `analysis boundary - GPU operation union`; it is not SM occupancy. The correlation-aware host-starvation heuristic gives CuPy a `1.0347x` upper bound from removing only measured submission starvation in this profiler capture. Profiler overhead can enlarge submission gaps, so CUDA Graphs remain a measured secondary opportunity, not a promoted optimization.

## Classified GPU activity

Times below are summed exclusive GPU activity durations, not wall time or inclusive NVTX duration.

| Component | CuPy | Torch | Difference | CuPy kernels | Torch kernels |
|---|---:|---:|---:|---:|---:|
| GEMM | 129.229 ms | 131.213 ms | **-1.984 ms** | 300 | 300 |
| Attention | 20.379 ms | 19.873 ms | +0.505 ms | 120 | 60 |
| SwiGLU forward/recompute | 8.598 ms | 3.398 ms | **+5.200 ms** | 40 | 20 |
| SwiGLU backward | 6.921 ms | 6.873 ms | +0.048 ms | 20 | 20 |
| RMSNorm | 6.270 ms | 5.552 ms | +0.718 ms | 120 | 80 |
| Residual | 4.530 ms | 1.494 ms | **+3.036 ms** | 80 | 60 |
| BF16/FP32 conversion | 3.464 ms | 0.651 ms | **+2.813 ms** | 120 | 100 |
| QKV/RoPE processing | 4.412 ms | 0 independently identifiable | **+4.412 ms** | 400 | 0 |
| Parameter-gradient accumulation | 1.578 ms | 0 independently identifiable | **+1.578 ms** | 60 | 0 |
| Memory/data movement | 1.739 ms | 2.662 ms | -0.923 ms | 180 | 180 |
| Other elementwise | 0.148 ms | 0 | +0.148 ms | 20 | 0 |
| Unknown | 0.078 ms | 0.306 ms | -0.228 ms | 80 | 40 |

“0 independently identifiable” means Torch folded the operation into another generated kernel; it does not mean the mathematics disappeared. Classification accounts for all but `0.078 ms` of CuPy kernel time and `0.306 ms` of Torch kernel time.

### Diagnosis

- **GEMMs are not the problem.** CuPy launches the same number and spends `1.984 ms` less GPU time.
- **Attention is not the problem.** The complete 20-layer difference is only `0.505 ms` in summed activity.
- **SwiGLU backward is already matched.** The `0.048 ms` difference is negligible.
- The explained non-GEMM excess is dominated by SwiGLU forward/recompute, QKV/RoPE materialization, residual kernels, BF16/FP32 conversions, and explicit FP32 parameter-gradient additions.
- CuPy also has approximately `9.3 ms` more execution gaps in the profiled boundary and 680 more kernel launches. Only `6.24 ms` of the gap difference meets the analyzer's conservative host-starvation attribution; the rest may include dependencies, allocator/library behavior, and profiling overhead.

## Top 20 CuPy CUDA kernel families

The JSON/CSV artifact retains exact full symbols. Dimensions are source-attributed where one geometry is unambiguous.

| Rank | Kernel family | Class | Launches | Total | Median | GPU kernel % | Dtype | Dimensions/source | Torch equivalent | Opportunity |
|---:|---|---|---:|---:|---:|---:|---|---|---|---|
| 1 | CUTLASS BF16 64x64 NN | GEMM | 100 | 41.081 ms | 515.2 µs | 22.01% | BF16 | mixed QKV/Wo/FFN forward | yes | already competitive |
| 2 | `nvjet...80x128...NTNN` | GEMM | 80 | 38.522 | 523.1 | 20.64% | BF16 compute | QKV/FFN wgrad | yes | already competitive |
| 3 | `nvjet...64x112...TNNN` | GEMM | 40 | 22.356 | 533.2 | 11.98% | BF16 compute | gate/up dx | yes | preserve cuBLAS |
| 4 | CUTLASS BF16 128x128 TN | GEMM | 20 | 11.932 | 551.5 | 6.39% | BF16 | down dx, `[6912,1024]x[1024,4096]` | yes | preserve cuBLAS |
| 5 | `_dkdv_kernel` | Attention | 20 | 8.914 | 385.3 | 4.78% | mixed | B3/T2304/H16:4/D64 | FlexAttention bwd | low |
| 6 | `bf16_swiglu_fwd` | SwiGLU F/recompute | 40 | 8.598 | 208.9 | 4.61% | BF16 | `[6912,4096]` | generated SiLU | **selected 0089L target** |
| 7 | CUTLASS FP32 SIMT SGEMM NT | GEMM | 20 | 7.977 | 360.2 | 4.27% | FP32 | Wo backward | yes | inspect precision contract before change |
| 8 | `nvjet...128x176...TNNN` | GEMM | 40 | 7.360 | 177.1 | 3.94% | BF16 compute | Wo/QKV dx | yes | already competitive total |
| 9 | `bf16_swiglu_bwd` | SwiGLU B | 20 | 6.921 | 318.9 | 3.71% | BF16 | `[6912,4096]` | generated SiLU bwd | matched |
| 10 | `_dq_kernel` | Attention | 40 | 6.130 | 214.0 | 3.28% | mixed | B3/T2304/H16:4/D64 | FlexAttention bwd | low |
| 11 | `_fwd_kernel` | Attention | 40 | 4.113 | 102.8 | 2.20% | mixed | B3/T2304/H16:4/D64 | FlexAttention fwd | low |
| 12 | `cupy_add__float32_float32_float32` | Residual | 60 | 3.864 | 65.0 | 2.07% | FP32 | `[3,2304,1024]` | fused generated add | graph-fusion candidate |
| 13 | `rmsnorm_bwd_wgrad_bf16dy` | RMSNorm | 40 | 2.592 | 58.2 | 1.39% | BF16/FP32 | `[3,2304,1024]` | generated reduction | modest bound |
| 14 | `rmsnorm_bwd_dx_bf16dy` | RMSNorm | 40 | 1.977 | 49.9 | 1.06% | BF16/FP32 | `[3,2304,1024]` | generated norm bwd | modest bound |
| 15 | `cupy_copy__bfloat16_float32` | Conversion | 60 | 1.777 | 27.5 | 0.95% | BF16→FP32 | projection/backward boundaries | folded conversion | remove/fold where safe |
| 16 | BF16 RoPE multiply | QKV/RoPE | 160 | 1.732 | 12.1 | 0.93% | BF16 | RoPE forward | fused generated prep | materialization target |
| 17 | `cupy_copy__float32_bfloat16` | Conversion | 60 | 1.687 | 27.7 | 0.90% | FP32→BF16 | branch boundaries | folded conversion | remove/fold where safe |
| 18 | `inverse_rope_pack_qkv_bf16` | QKV/RoPE | 20 | 1.627 | 80.6 | 0.87% | BF16 | packed QKV backward | fused generated prep | inspect packing |
| 19 | `cupy_add__float32_bfloat16_float32` | Grad accumulation | 60 | 1.578 | 26.3 | 0.85% | BF16/FP32 | three FFN wgrads/layer | folded epilogue | prior 0089H gain was noise-sized |
| 20 | `_delta_kernel` | Attention | 20 | 1.222 | 60.9 | 0.65% | mixed | B3/T2304/H16:4/D64 | FlexAttention bwd | low |

Torch's top-20 exact list is in `0089k_depth20_comparative_torch_top20.csv`; the dominant families are the corresponding 300 CUTLASS GEMMs (`131.213 ms`), FlexAttention (`19.873 ms`), generated SwiGLU backward (`6.873 ms`), generated SwiGLU forward (`3.398 ms`), generated RMSNorm kernels, and timed BF16-gradient FP32 export copies (`2.618 ms`).

## Amdahl bounds and ranked opportunities

1. **SwiGLU forward/recompute:** CuPy excess `5.200 ms`; absolute category-elimination ceiling `4.35%` of baseline wall time. Caching `h` would eliminate only the 20 backward recomputations (`~4.30 ms`, ideal `1.022x`) but costs `1.055 GiB` at depth 20, so it was rejected. A no-memory vec2 kernel became PATCH 0089L.
2. **QKV/RoPE processing:** standalone CuPy total `4.412 ms` (`2.23%` of baseline). Torch folds this work into generated kernels. A targeted packed RoPE/projection-boundary study is the next device-work candidate.
3. **Residual + conversion integration:** excess `5.849 ms` across categories, but prior isolated residual/norm fusion delivered only ~0.5%. A new candidate must cross more than one existing boundary to justify implementation.
4. **Execution gaps / launch graph:** gap difference `9.303 ms`; conservative host-starvation component `6.242 ms`. This is the largest aggregate opportunity but is profiler-sensitive. A CUDA Graph replay experiment is now evidence-supported, provided cuBLAS/Triton/gradient buffers can be captured without changing the graph.
5. **Parameter-gradient addition:** `1.578 ms`; prior direct-accumulation experiment was only 0.34% faster overall, so it is not repeated.

## Raw artifacts

```text
benchmarks/results/0089k_depth8_{cupy,torch}_baseline.json
benchmarks/results/0089k_depth20_{cupy,torch}_baseline.json
benchmarks/results/0089k_depth20_{cupy,torch}_profile.nsys-rep
benchmarks/results/0089k_depth20_{cupy,torch}_profile.sqlite
benchmarks/results/0089k_depth20_comparative_analysis.json
benchmarks/results/0089k_depth20_comparative_categories.csv
benchmarks/results/0089k_depth20_comparative_cupy_top20.csv
benchmarks/results/0089k_depth20_comparative_torch_top20.csv
```

Large profiler artifacts remain ignored under `benchmarks/results/`.
