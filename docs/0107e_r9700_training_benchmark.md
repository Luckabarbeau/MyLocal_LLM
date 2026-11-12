# R9700 Model B training bottleneck benchmark

## Scope

Synthetic full-size registered sparse Model B (348,261,376 parameters), random
weights, 8,192 input tokens, all 8,192 next-token targets scored, batch 1, BF16,
consumer-gpu runtime defaults, no block activation checkpointing, Triton
local/global/compression paths, local query chunk 1,536. No checkpoint reads or
writes. This is not identical to resumed training: document lengths, frozen
parameters, accumulation and checkpoint states differ.

Extended `benchmarks/benchmark_dual_resolution_full_model_0102d.py` with sparse
Model B selection, optional Hybrid Muon optimizer and synchronized operation
profiling. Existing defaults remain unchanged. CPU regression gate:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv-rocm/bin/python -m pytest -q tests/test_dual_resolution_full_model_0102d.py
```

Result: 7 passed, including the new sparse-config and Hybrid Muon/profile gates.

## Reproduce warmed baseline

```sh
sh scripts/run_r9700_preflight.sh -u benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --architecture matrix-pool-sparse --context 8192 --model-context 65536 \
  --score-last-k 8192 --head-chunk-tokens 8192 --activation-checkpoint none \
  --runtime-profile consumer-gpu --query-chunk 1536 \
  --warmup-iterations 1 --repeat-iterations 3 \
  --optimizer-step --optimizer-kind hybrid-muon --lr 1.5e-4 \
  --output logs/r9700_8k_warm_baseline.json
```

The optimizer is constructed before warm-up and updated once after the measured
microbatches; optimizer time is separate, not amortized into reported tokens/s.
The benchmark checks representative branch gradients, not full optimizer
numerical parity. Memory samples are not true allocation peaks.

## Results

Warmed forward/loss/backward totals: **3,489.07 / 3,500.12 / 3,488.56 ms**.

- Median forward: 686.19 ms.
- Median loss forward: 7.29 ms; head backward is in the backward timing.
- Median backward: 2,795.83 ms.
- Full-model throughput excluding optimizer: **2,347.90 tokens/s**.
- Separate cold Hybrid Muon update plus compute-buffer refresh: 2,989.69 ms.
- Sampled device memory around 11.2 GiB; not a true peak.

This reproduces a throughput limitation after warm-up, not just first-run JIT.
The user's reported one-step run also used mixed document lengths, 49.8% target
fill, 1,812 supervised tok/s, 1,917 executed tok/s and 3,636 capacity tok/s.
Capacity tokens count configured slots, not actual supervised or executed work.
A card's peak BF16 FLOPS cannot directly establish training tokens/s.

## Profiling and concrete fallback

Additional artifacts:

- `logs/r9700_8k_profile.json`: deferred global-branch events.
- `logs/r9700_8k_cpu.prof`: Python cProfile, including imports and warm-up.
- `logs/r9700_8k_operations.json`: intrusive synchronized operation timings.
- `logs/r9700_8k_ffn_detail.json`: same with `MINI_LLM_MOE_DETAIL_PROFILE=1`.

Use `--profile-operations --repeat-iterations 1` in place of the baseline's
optimizer options for coarse profiling. Add `MINI_LLM_MOE_DETAIL_PROFILE=1`
before the wrapper for FFN detail. Profiling perturbs performance; use the
unprofiled baseline for throughput. Nested scopes must not be summed.

Coarse profile across 20 layers:

| Region | Time |
|---|---:|
| Dense FFN backward (`block.moe.backward` historical label) | 2,154.75 ms |
| Dense FFN forward | 607.96 ms |
| All attention backward | 563.33 ms |
| Local attention backward (inside preceding scope) | 386.78 ms |

FFN event times per layer:

| Operation | Average |
|---|---:|
| gate forward GEMM | 0.629 ms |
| up forward GEMM | 0.623 ms |
| down forward GEMM | 26.588 ms |
| gate input-gradient GEMM | 25.954 ms |
| up input-gradient GEMM | 25.737 ms |
| down weight-gradient GEMM | 19.124 ms |
| gate / up weight-gradient GEMMs | 13.688 / 14.092 ms |

A direct HIP dtype probe with `MINI_LLM_FUSED_SWIGLU=1` established:

- BF16 G and U inputs.
- `_fused_swiglu_forward(G,U)` returns `None`.
- `_FUSED_SWIGLU_DISABLED` becomes `True`.
- Fallback `silu(G)` and `silu(G)*U` have **FP32 dtype**.

Thus `ExpertFFN.forward`'s down projection receives FP32 H with BF16 weights,
not two BF16 operands. `ExpertFFN.backward` rejects its fused BF16 derivative
when gradient/input dtypes do not match; mixed-precision intermediate paths
are an evidence-supported optimization target. The exact dtype of every full
backward intermediate has not yet been instrumented.

The CPU profile's largest Python-attributed time was 10.045 s in 20 backend
synchronization calls over the entire 15.977 s run. These calls wait for GPU
work; they do not establish CPU computation as the limiting factor. Imports
also contributed 3.085 s. They are not equivalent to a device kernel trace.

## Next targeted change

Implement an AMD-compatible fused BF16 SwiGLU forward/backward path, preserving
BF16 intermediate storage and FP32 parameter-gradient accumulation. Validate
its output/derivatives against the existing mathematical reference, then rerun
this same warmed benchmark. Do not silently cast arbitrary FP32 gradients to
BF16 or change precision policy solely for speed. No production math/kernels
were modified in this benchmarking task; no speedup is claimed yet.
