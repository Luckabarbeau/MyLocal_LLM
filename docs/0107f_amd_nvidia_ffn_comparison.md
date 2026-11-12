# AMD/NVIDIA FFN comparison and HIP SwiGLU fix

## Change and computational contract

`mini_llm/ops/experts.py` now selects `<hip/hip_fp16.h>` for CuPy HIP instead
of `<cuda_fp16.h>`. The CUDA source path is unchanged. Strict compilation had
identified the missing CUDA header as the cause of the HIP fused-kernel failure.
This small header selection restores the existing fused BF16 forward/backward
kernels; no new algorithm, gradient casts, projection shapes, weights, or
checkpoint schemas were introduced.

The preserved graph is:

- G = X W_gate; U = X W_up.
- H = SiLU(G) * U; Y = H W_down.
- dW_down = H^T dY; dH = dY W_down^T.
- dG = (dH * U) * SiLU'(G); dU = dH * SiLU(G).
- dW_gate = X^T dG; dW_up = X^T dU.
- dX = dG W_gate^T + dU W_up^T.

The fused path retains BF16 intermediate storage and FP32 parameter-gradient
accumulation, as on NVIDIA. Backward has six projection GEMMs versus forward's
three, so equal total forward/backward time is not the expected contract.

At 8K context, width 1024, FFN width 4096, every individual gate/up/down forward,
input-gradient and weight-gradient GEMM has **2 * 8192 * 1024 * 4096 =
68,719,476,736 FLOPs**. Their shapes/transposes and bandwidth differ, so equal
FLOPs do not guarantee equal execution time.

## Full-model benchmark

Same synthetic 348,261,376-parameter sparse Model B, BF16, batch 1, 8,192
fully scored tokens, consumer-gpu defaults, no block activation checkpointing,
Triton attention, query chunk 1,536, one warmup and three measured iterations.
Random weights; no checkpoint reads/writes. After-fix cross-GPU measurements
exclude optimizer construction/updates. The earlier AMD baseline constructed
an optimizer but timed its update separately.

| Measurement | R9700 before | R9700 fixed | RTX 5080 |
|---|---:|---:|---:|
| Forward median, ms | 686.19 | 130.51 | 87.12 |
| Loss-forward median, ms | 7.29 | 7.19 | 7.22 |
| Backward median, ms | 2795.83 | 770.05 | 220.81 |
| Tokens/s from median iteration total | 2347.90 | **9023.03** | **25119.71** |
| Loss | 10.610343 | 10.610338 | 10.602576 |

Throughput improves **3.84x** on AMD. Median iteration totals, rather than sums
of stage medians, determine throughput. Cross-GPU random initialization and
numerics need not produce identical loss; these are timing runs, not parity
or training-convergence tests. Desktop GPU activity and profiling overhead can
also influence measurements; this is not a controlled hardware peak comparison.

## Per-layer FFN GPU timings

Deferred event timings inside the intrusive operation-profile run; 20 calls per
operation. Use the unprofiled runs above for throughput.

| GEMM | R9700 fixed, ms | RTX 5080, ms |
|---|---:|---:|
| Gate forward | 0.588 | 0.679 |
| Up forward | 0.587 | 0.659 |
| Down forward | 0.934 | 1.132 |
| Gate input gradient | 0.586 | 0.626 |
| Up input gradient | 0.585 | 1.062 |
| Down input gradient | 0.620 | 0.661 |
| Gate weight gradient | 0.652 | 1.015 |
| Up weight gradient | 0.593 | 0.650 |
| Down weight gradient | 1.610 | 0.652 |

AMD's former down forward was 26.59 ms and gate/up input gradients were
25.95/25.74 ms. Those mixed FP32/BF16 outliers are removed. Down weight-gradient
remains slower on AMD; do not claim every GEMM is identical.

Across 20 layers, profiled FFN forward/backward are **50.96 / 115.61 ms** on
AMD and **54.77 / 107.61 ms** on NVIDIA: comparable timescales and roughly the
expected 2x backward workload. Whole-model backward is **not yet balanced**.

## Remaining attention bottleneck and bounded experiments

AMD attention backward: 566.22 ms; RTX: 91.64 ms. Local attention backward alone:
389.59 ms versus 29.12 ms. These nested scopes must not be summed.

`benchmarks/benchmark_0107f_local_backward.py` isolates delta, long/short dQ,
and combined dK/dV launches, using synthetic contiguous Q/K/V and the mixed
4096/512 windows. It compares launch variants with unchanged gradients, not an
independent derivative oracle. It is not an exact packed/sparse full-model fixture.

AMD default (four warps, launch stages 3), median kernel times:

- Delta: 0.170 ms; long dQ: 1.247 ms; short dQ: 2.597 ms; dK/dV: **17.410 ms**.
- Compiled dQ kernels: 256 registers, 214–215 reported spills.
- Compiled dK/dV: 256 registers, **866 reported spills**.

Changing launch stages from 3 to 1 retained the same spill counts and barely
helped. Eight warps reduced reported dK/dV spills to 623 but slowed it to
19.80 ms (stages 3), or 19.37 ms (stages 1). No production launch settings were
changed. Stage-only gradient comparisons had zero error; eight-warps dQ maximum
absolute difference was 6.61e-8, dK/dV differences zero, passing atol/rtol 1e-5.

The evidence supports investigating an AMD-specific smaller-live-state dK/dV
schedule/tile next, not applying the unsuccessful warp/stage settings. Register
spilling and the slow launch are established; profiling does not prove that
spilling alone accounts for the entire cross-GPU gap.

## Commands and artifacts

AMD warmed run:

```sh
sh scripts/run_r9700_preflight.sh -u benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --architecture matrix-pool-sparse --context 8192 --model-context 65536 \
  --score-last-k 8192 --head-chunk-tokens 8192 --activation-checkpoint none \
  --runtime-profile consumer-gpu --query-chunk 1536 \
  --warmup-iterations 1 --repeat-iterations 3 --output logs/r9700_8k_swiglu_fixed.json
```

For NVIDIA replace the launcher prefix with `MINI_LLM_BACKEND=cupy .venv/bin/python -u`
and use a different output filename. For FFN detail set
`MINI_LLM_MOE_DETAIL_PROFILE=1`, add `--profile-operations`, use repeat 1.

Reports:

- `logs/r9700_8k_swiglu_fixed.json`, `logs/r9700_8k_ffn_fixed.json`.
- `logs/rtx5080_8k_warm_reference.json`, `logs/rtx5080_8k_ffn_reference.json`.
- `logs/r9700_local_backward_staging.json`, `logs/r9700_local_backward_warps8.json`.

Validation:

- ROCm: `sh scripts/run_r9700_preflight.sh -m pytest -q tests/test_swiglu.py tests/test_moe_cache_recompute_0080o7.py` — **14 passed**.
- CUDA: `MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q tests/test_swiglu.py tests/test_moe_cache_recompute_0080o7.py` — **14 passed**.
- CPU: `MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv-rocm/bin/python -m pytest -q tests/test_swiglu.py tests/test_dual_resolution_full_model_0102d.py` — **11 passed**.
- `git --no-pager diff --check` — passed.

The added BF16 backward test covers contiguous and packed-strided G/U,
BF16 output dtype, both reverse product paths, and analytical references rounded
to BF16 with atol/rtol 0.008. Existing tests cover FP16/BF16 forward and the
smooth FP64 SwiGLU directional derivative. Both full-model benchmarks passed
finite/nonzero branch-gradient checks. These tests do not establish bitwise
checkpoint continuation, long-run training stability, or full optimizer parity.
