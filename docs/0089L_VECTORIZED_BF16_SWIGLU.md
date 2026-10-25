# PATCH 0089L — Vectorized BF16 SwiGLU forward/recompute

## Profiling basis

PATCH 0089K measured the complete B3/T2304/depth-20 graph and found:

```text
CuPy SwiGLU forward/recompute: 8.598 ms, 40 launches
Torch generated SwiGLU forward: 3.398 ms, 20 launches
CuPy excess:                   5.200 ms
```

Twenty CuPy launches are ordinary forward and twenty recompute the down-projection input during backward. Caching that `[6912,4096]` BF16 tensor in every layer would cost `1.055 GiB`, so the cache-based maximum `~4.30 ms` saving was rejected for the 16 GB production target.

The existing CUDA kernel instead loaded and stored one 16-bit BF16 element per thread. The selected candidate processes two adjacent BF16 values per thread through aligned 32-bit loads/stores while preserving the exact scalar arithmetic independently for each lane.

## Implementation

`mini_llm/ops/experts.py` adds `bf16_swiglu_fwd_vec2` and selects it only when:

- backend is CuPy;
- fused SwiGLU is active;
- dtype is BF16;
- input layouts are compact;
- element count is even;
- all three pointers are 32-bit aligned;
- `MINI_LLM_SWIGLU_VEC2` is enabled.

Odd, strided, unaligned, FP16, disabled, or unsupported cases retain the original paths.

The exact per-lane operation remains:

```text
gf = BF16_to_FP32(g)
s  = 1 / (1 + expf(-gf))
h  = round_BF16((gf * s) * BF16_to_FP32(u))
```

No backward equation, cache ownership, parameter layout, optimizer behavior, or model architecture changes.

After validation, `mini_llm/runtime_defaults.py` enables the flag through `gpu-fast` and derived `consumer-gpu` profiles using `os.environ.setdefault`. Explicit fallback remains:

```bash
export MINI_LLM_SWIGLU_VEC2=0
```

Reference profiles and direct low-level use without a runtime profile retain the scalar default.

## Correctness

### Kernel parity

`tests/test_swiglu.py` checks an aligned even shape and an odd fallback shape against scalar BF16 arithmetic. The vec2 and scalar CUDA outputs are bitwise equal.

### Existing FFN/MoE contracts

```text
15 passed
```

Coverage includes fused SwiGLU, BF16 expert input-gradient accumulation contracts, and CUDA BF16 MoE recomputation parity.

### Complete depth-2 native parity

The existing 0089E 22-tensor contract compared vec2 CuPy against scalar-kernel CuPy:

- output: bitwise equal;
- boundary output and boundary gradient: bitwise equal;
- input gradient: bitwise equal;
- all attention and FFN weight gradients: bitwise equal;
- only RMSNorm gamma reductions showed tiny run-order nondeterminism;
- worst relative L2: `9.34e-8`;
- worst maximum absolute difference: `1.526e-5`;
- all 22 tensors passed.

### Real optimizer steps

Three matched Hybrid-Muon steps resumed independently from the same checkpoint. Loss and gradient norm were exactly equal on every step:

| Step | Loss | Gradient norm |
|---:|---:|---:|
| 14701 | 2.2514927387 | 1.6740740538 |
| 14702 | 2.6870081425 | 2.4055724144 |
| 14703 | 3.1809725761 | 2.9791724682 |

This validates repeated backward accumulation and optimizer integration for the measured short run. The two jobs wrote only to separate temporary directories; the source checkpoint was not modified.

## Performance

### Alternating same-process kernel A/B

Exact production tensor shape `[6912,4096]`, BF16, five warmups, 40 alternating pairs:

| Metric | Scalar | Vec2 | Change |
|---|---:|---:|---:|
| GPU median | 0.220608 ms | 0.189184 ms | **-14.24%** |
| Speedup | 1.000x | **1.166x** | |
| Bitwise parity | reference | pass | |

The kernel keeps the same number of launches and allocations.

### Complete 20-layer F+B

Two independent baseline/candidate process pairs, each with five warmups and twenty samples:

| Run | Scalar GPU median | Vec2 GPU median | Saved | Reduction |
|---:|---:|---:|---:|---:|
| 1 | 197.696 ms | 196.288 ms | 1.409 ms | 0.71% |
| 2 | 198.193 ms | 195.750 ms | 2.443 ms | 1.23% |
| Median of run medians | **197.945 ms** | **196.019 ms** | **1.926 ms** | **0.97%** |

Host-wall median of run medians improved from `198.922` to `197.046 ms` (`0.95%`).

A final run without manually setting the flag confirmed that `gpu-fast` promoted the candidate:

```text
MINI_LLM_SWIGLU_VEC2 applied as 1
GPU median:  196.212 ms
Host median: 197.212 ms
```

### Complete no-optimizer model

Matched real checkpoint/data, B3/T2304, one deterministic history-free batch, strict Triton attention, two warmups, twenty samples:

| Metric | Scalar | Vec2 | Change |
|---|---:|---:|---:|
| GPU median | 241.202 ms | 235.951 ms | -5.251 ms (-2.18%) |
| Host median | 241.283 ms | 236.074 ms | -5.209 ms (-2.16%) |
| Loss | 2.8750963211 | 2.8750963211 | exact |

The p90 intervals overlap (`244.208` versus `244.326 ms`), so this larger full-model delta is supportive rather than a precise long-run estimate.

### Real Hybrid-Muon training

Matched settings:

```text
checkpoint: checkpoints/0083_intermediate_20L_1024_fullmix_100k
dataset: token_shards_ensemble_32k_mixed
B=3, T=4096, grad accumulation=1
strict Triton attention
Hybrid Muon, full CPU optimizer-state offload
supervised-token gradient weighting
```

| Step | Supervised tokens | Scalar step | Vec2 step | Scalar tok/s | Vec2 tok/s |
|---:|---:|---:|---:|---:|---:|
| 14701 | 12,288 | 1530.8 ms | 1521.0 ms | 8,027 | 8,079 |
| 14702 | 5,645 | 883.0 ms | 878.7 ms | 6,393 | 6,425 |
| 14703 | 6,275 | 1139.0 ms | 1134.2 ms | 5,509 | 5,532 |
| Aggregate | 24,208 | 3552.8 ms | 3533.9 ms | **6,813** | **6,850** |

Aggregate supervised-token throughput improved approximately `0.54%`. Excluding the first warm/JIT-heavy step, throughput improved approximately `0.47%`. Three steps are too few for a high-confidence long-run estimate, but every matched step moved in the expected direction with exact loss/gradient-norm parity.

Both arms used the same runtime settings, but this historical run did not explicitly record `MINI_LLM_AR_ACCUM_LENGTH_BUCKETS=1`. Treat the throughput delta as a matched A/B result, not as a rerun under the later complete 0089M flag-reporting contract. `profile_0087a_training.py` now applies and reports that flag together with the other required production benchmark settings.

### Memory

Depth-20 pool reservation across the two process pairs:

| Backend | Minimum | Maximum |
|---|---:|---:|
| Scalar | 7,338,940,416 B | 7,338,940,416 B |
| Vec2 | 7,338,940,416 B | 7,353,096,192 B |

The candidate adds no arrays or cache tensors. The `14.2 MB` maximum-process difference is allocator/run variation, not a vec2 allocation requirement. A final promoted run returned to `7,338,940,416 B`.

## Acceptance decision

Promoted. The gain is smaller than the preferred few-percent complete-model target, but the change is simple, bitwise at its direct output, repeatable in two independent depth-20 process pairs, positive in every matched real optimizer step, and adds no memory or kernel launches.

## Reproduction

Kernel A/B:

```bash
.venv/bin/python benchmarks/benchmark_0089l_swiglu_vec2.py \
  --warmup 5 --repeats 40 \
  --output-json benchmarks/results/0089l_swiglu_vec2_microbenchmark.json
```

Complete stack candidate/fallback:

```bash
MINI_LLM_SWIGLU_VEC2=1 .venv/bin/python \
  benchmarks/benchmark_0089e_block_stack.py \
  --backend cupy \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 20 --warmup 5 --repeats 20 \
  --output-json benchmarks/results/0089l_depth20_cupy_vec2.json

MINI_LLM_SWIGLU_VEC2=0 .venv/bin/python \
  benchmarks/benchmark_0089e_block_stack.py \
  --backend cupy \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 20 --warmup 5 --repeats 20 \
  --output-json benchmarks/results/0089l_depth20_cupy_scalar.json
```

## Raw artifacts

```text
benchmarks/results/0089l_swiglu_vec2_microbenchmark.json
benchmarks/results/0089l_depth2_cupy_vec2.json
benchmarks/results/0089l_depth2_cupy_vec2_tensors.npz
benchmarks/results/0089l_depth20_cupy_{baseline_r2,vec2,vec2_r2,promoted}.json
benchmarks/results/0089l_full_model_vec2_{baseline,candidate}.json
benchmarks/results/0089l_real_training_comparison.json
```

## Remaining limitation and next target

0089L does not close the 20-layer gap. Relative to the reproduced compiled-Torch reference, promoted CuPy is approximately `24.32 ms` slower (`196.21` versus `171.89 ms`).

The next measured device-work target is QKV/RoPE packing/materialization (`4.412 ms` of independently visible CuPy activity). The larger aggregate target is the `~9.3 ms` execution-gap difference, but a CUDA Graph experiment should be gated by a fresh unprofiled complete-stack A/B because the host-starvation portion is profiler-sensitive.

Worst-case 61,440-token historical processing remains a separate uncompleted profiling stage; the three-step training sample reached 21,619 history tokens, not the worst case.
