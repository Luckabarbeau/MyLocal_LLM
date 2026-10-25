# PATCH 0089M — Fair benchmark flags, prepared RoPE, and graph/QKV decisions

> Follow-up: PATCH 0089N implemented the fused forward Q/K RoPE recommendation, also folded compact-V materialization into the kernel, and measured a repeatable complete-stack improvement. See `docs/0089N_FUSED_QKV_ROPE_AND_MIGRATION_GATE.md`.

## Scope

This follow-up closes two questions left by PATCH 0089K:

1. whether the PATCH 0089E block-stack comparison gave CuPy and compiled PyTorch equally prepared RoPE inputs;
2. whether inverse-RoPE QKV packing or CUDA Graph replay can materially reduce the remaining depth-20 gap.

No model architecture, precision boundary, gradient-accumulation rule, checkpoint format, or historical-retrieval behavior was changed.

## Required environment contract

The following settings are now explicitly applied and recorded by the CuPy block-stack benchmark. The production training profiler uses the same values as A/B-safe `setdefault` defaults and includes them in `runtime.effective_flags`.

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

### PATCH 0089E applicability

The block-stack fixture has no optimizer, decoder sampling/loss pipeline, validation/checkpoint phase, or training-loop memory-trim interval. Those settings are still fixed and reported, but are computationally inactive in this benchmark.

The grouped-local cuBLAS settings are fallback-contract settings: PATCH 0089E requires strict Triton current attention, so grouped local attention does not execute. `MINI_LLM_ROPE_PREPARED_POSITIONS=1` is active. Detail profiling, fused expert GEMM, and GPU memory audit are actively disabled as requested.

For real training, the optimizer, autoregressive accumulation, memory-trim, and memory-audit settings apply normally when their corresponding phases execute.

## Prepared-RoPE fairness correction

Previously, PATCH 0089E called `TransformerBlock` directly with raw position IDs. That bypassed `DecoderLanguageModel.forward_body`, which is the production location that consumes `MINI_LLM_ROPE_PREPARED_POSITIONS`. Consequently, CuPy repeatedly validated and gathered positions in every Q/K operation and layer while the Torch reference prepared cosine/sine inputs outside its compiled graph.

The corrected harness now calls `prepare_rope_positions(...)` once outside all timed regions and passes the same prepared factors through every layer.

The result JSON records:

```text
environment_contract
environment_applicability
rope_positions = prepared once outside timing and shared by every layer
```

### Native parity

Depth-2 complete F+B was compared against the established 22-tensor CuPy reference.

| Metric | Result |
|---|---:|
| Accepted | yes |
| Bitwise-identical tensors | 18 / 22 |
| Worst relative L2 | 9.69e-8 |
| Maximum absolute difference | 7.63e-6 |
| Non-bitwise tensors | four RMSNorm gamma gradients |

The differences are limited to established FP32 reduction-order variation. Outputs, input gradients, boundary tensors, attention projection gradients, and FFN weight gradients are bitwise identical.

## Corrected unprofiled baseline

All values are complete numerical forward and backward. CUDA-event and synchronized host-wall medians use 5 warmups and 20 measured repetitions. Compilation and fixture preparation are excluded.

| Depth/backend | GPU median | GPU MAD | Host median | Host MAD |
|---|---:|---:|---:|---:|
| 8 CuPy | 79.724 ms | 0.191 ms | 80.184 ms | 0.194 ms |
| 8 compiled Torch | 71.956 ms | 0.184 ms | 72.048 ms | 0.171 ms |
| 20 CuPy, bracket run | 199.972 ms | 0.401 ms | 200.988 ms | 0.395 ms |
| 20 compiled Torch | 180.203 ms | 0.443 ms | 180.315 ms | 0.393 ms |

The corrected same-day depth-20 difference is:

```text
199.972 - 180.203 = 19.769 ms
10.97% relative to compiled Torch
9.89% of current CuPy latency must be removed to match Torch
```

An earlier corrected CuPy process measured 201.163 ms. This process-level variation is why old CuPy and new Torch runs are not mixed when quoting the gap.

## Corrected Nsight Systems breakdown

The table contains summed GPU event durations inside the one warmed F+B capture. The global execution unions are reported separately; inclusive NVTX ranges are not summed as exclusive GPU time.

| Component | CuPy | Torch | Difference | Kernels CuPy/Torch |
|---|---:|---:|---:|---:|
| GEMM | 137.327 ms | 135.787 ms | +1.540 ms | 300 / 300 |
| Attention | 21.570 ms | 20.161 ms | +1.408 ms | 120 / 60 |
| SwiGLU forward/recompute | 8.410 ms | 4.013 ms | +4.397 ms | 40 / 20 |
| SwiGLU backward | 7.256 ms | 7.409 ms | -0.153 ms | 20 / 20 |
| RMSNorm | 6.446 ms | 6.348 ms | +0.099 ms | 120 / 80 |
| Residual | 5.703 ms | 1.498 ms | +4.206 ms | 80 / 60 |
| QKV/RoPE processing | 4.343 ms | folded | +4.343 ms visible | 280 / 0 |
| BF16/FP32 conversion | 3.416 ms | 0.656 ms | +2.759 ms | 120 / 100 |
| Parameter-gradient accumulation | 1.553 ms | folded | +1.553 ms visible | 60 / 0 |
| Memory/data movement | 1.616 ms | 2.697 ms | -1.082 ms | 100 / 180 |
| Other elementwise | 0.149 ms | folded | +0.149 ms | 20 / 0 |
| Unknown | 0.000 ms | 0.305 ms | -0.305 ms | 0 / 40 |

Execution accounting:

| Metric | CuPy | Torch |
|---|---:|---:|
| GPU execution union | 197.778 ms | 178.875 ms |
| Summed all event duration | 197.789 ms | 178.875 ms |
| CUDA kernels | 1,260 | 860 |
| All GPU events | 1,380 | 920 |
| Internal execution gaps | 3.539 ms | 1.926 ms |
| Candidate host-starved time | 0.241 ms | 0.415 ms |

The summed category difference is 18.915 ms and the execution-union difference is 18.904 ms. The small discrepancy is ordinary stream/event accounting, not unclassified work.

The largest positive category differences explain the gap as follows (positive contributions exceed 100% because faster CuPy categories offset part of them):

| Category | Difference | Share of union gap |
|---|---:|---:|
| SwiGLU forward/recompute | 4.397 ms | 23.3% |
| QKV/RoPE processing | 4.343 ms | 23.0% |
| Residual | 4.206 ms | 22.2% |
| BF16/FP32 conversion | 2.759 ms | 14.6% |
| Parameter-gradient accumulation | 1.553 ms | 8.2% |
| GEMM | 1.540 ms | 8.1% |
| Attention | 1.408 ms | 7.5% |

Prepared positions reduced visible QKV/RoPE launch count from 400 to 280 by removing 120 repeated validation kernels. The category changed from 4.412 ms in the older capture to 4.343 ms in the corrected capture, but these separate profiler processes had different overall GPU speed. The launch removal is established; a standalone latency gain is not claimed.

## Avenue 1: inverse-RoPE QKV pair vectorization

### Bound before implementation

The scalar `inverse_rope_pack_qkv_bf16` kernel accounted for:

```text
20 launches
1.627940 ms summed duration
80.784 us median launch duration
0.814% of the 199.972 ms unprofiled CuPy stack
```

Even eliminating it completely could improve the stack by only 0.814% (`1.0082x`).

### Experimental result

A temporary pair-vectorized CUDA kernel used aligned `float2` loads, one cosine/sine load per RoPE pair, and one `__nv_bfloat162` store. It was bitwise equal to both the scalar kernel and:

```text
concat(rope_backward(dQ), rope_backward(dK), dV).astype(bfloat16)
```

Its profiled result was:

| Metric | Scalar | Vec2 | Change |
|---|---:|---:|---:|
| 20-launch kernel total | 1.628 ms | 1.221 ms | -0.407 ms (-25.0%) |
| Complete-stack theoretical contribution | 0.814% | 0.610% | -0.204 percentage points |

Four fresh-process depth-20 runs were alternated scalar/vectorized. They were externally noisy: scalar MAD reached 16.51 ms, and the apparent median-process improvement was 7.04 ms—more than four times the kernel's absolute possible benefit. That result is physically inconsistent with the changed code scope and is therefore rejected as confounded.

The separate candidate profile also did not reduce the complete QKV/RoPE category (`4.343 ms` scalar versus `4.468 ms` candidate), demonstrating that the 0.407 ms kernel-only gain was below whole-category/process variation.

### Decision

The vectorized production code and experimental flag were removed. The established scalar path remains active. Profiler and JSON artifacts are retained under `benchmarks/results/` for research evidence.

## Avenue 2: CUDA Graph replay

The corrected trace does not support a training CUDA Graph implementation:

- only 0.241 ms is correlation-qualified as candidate CuPy host starvation;
- the fixed-GPU-work graph-only upper bound is `1.0012x`;
- every one of the 1,380 correlated GPU events had the next operation already queued;
- the 3.539 ms internal-gap total therefore cannot be attributed wholesale to Python launch starvation;
- graph replay would not fuse the 400 extra CuPy kernels or remove their memory traffic.

Read-only feasibility probes also established substantial implementation constraints:

- high-level CuPy BF16 matmul cannot be captured (`cuBLAS API during stream capture is currently unsupported`);
- direct project cuBLAS wrappers can capture, but do not cover all training GEMM dtype/transpose/accumulation contracts;
- Triton mixed attention captures only after all device metadata, including active-head indices, is preallocated;
- exact training replay would require fixed caller-owned activation, cache, scratch, parameter, and gradient addresses;
- preserving BF16 scratch followed by FP32 parameter-gradient accumulation is mandatory; silently adopting direct FP32 wgrad accumulation would change rounding semantics.

### Decision

Do not implement the proposed depth-1 static training executor. Its measured ideal host-side benefit is at most 0.241 ms (`0.12%`) while its semantic and maintenance risk is high. Reconsider CUDA Graphs only if a future trace shows several milliseconds of correlation-qualified host starvation or a workload with materially smaller kernels and genuine queue starvation.

## Validation

Commands executed include:

```bash
.venv/bin/python -m pytest -q tests/test_0089e_block_stack_contract.py
.venv/bin/python -m pytest -q \
  tests/test_0087a_profile_ranges.py \
  tests/test_0089a_optimized_training.py \
  tests/test_0089e_block_stack_contract.py
```

Results:

```text
PATCH 0089E contract: 9 passed, 1 skipped
Combined profiler/benchmark contracts: 19 passed, 2 skipped
Restored scalar depth-2 22-tensor parity: accepted
Worst restored-scalar relative L2: 9.31e-8
Worst restored-scalar absolute difference: 1.14e-5
```

The last scalar check used one measured repetition only as a post-revert correctness check, not as latency evidence.

Nsight Compute hardware counters remain unavailable because of `ERR_NVGPUCTRPERM`; no register-pressure, occupancy, DRAM-bandwidth, or Tensor Core utilization claim is made.

## Artifacts

```text
benchmarks/results/0089m_depth2_prepared_rope.json
benchmarks/results/0089m_depth8_cupy_fair_flags.json
benchmarks/results/0089m_depth8_torch_fair_flags.json
benchmarks/results/0089m_depth20_cupy_fair_flags.json
benchmarks/results/0089m_depth20_cupy_fair_flags_run2.json
benchmarks/results/0089m_depth20_torch_fair_flags.json
benchmarks/results/0089m_depth20_{cupy,torch}_profile.nsys-rep
benchmarks/results/0089m_depth20_{cupy,torch}_profile.sqlite
benchmarks/results/0089m_depth20_comparative_analysis.json
benchmarks/results/0089m_depth20_comparative_categories.csv
benchmarks/results/0089m_depth20_comparative_{cupy,torch}_top20.csv
benchmarks/results/0089m_depth20_inverse_rope_{scalar_run1,scalar_run2,vec2_run1,vec2_run2}.json
benchmarks/results/0089m_depth20_inverse_rope_vec2_profile.nsys-rep
benchmarks/results/0089m_depth20_inverse_rope_vec2_profile.sqlite
benchmarks/results/0089m_depth20_inverse_rope_vec2_analysis.json
```

## Next measured opportunity

CUDA Graph replay and isolated inverse-RoPE vectorization are closed by evidence. The next investigation should target **cross-operation kernel fusion**, not submission replay or GEMM replacement.

The largest measured excesses are SwiGLU forward/recompute (`4.397 ms`), residual processing (`4.206 ms`), QKV/RoPE processing (`4.343 ms` visible), and BF16/FP32 conversion (`2.759 ms`). The practical target is a source-level sequence that removes both a materialized tensor and multiple launches. A single sub-kernel optimization below 1% cannot reliably move the complete stack.

Within QKV/RoPE specifically, forward RoPE arithmetic is a more credible next target than inverse packing: 160 BF16 multiply launches consume 1.779 ms, and 80 add/subtract launches consume another 0.729 ms. A fused Q/K forward-RoPE kernel has an ideal category bound of 2.508 ms (1.25% of current CuPy latency) and could remove roughly 200 launches. This is a hypothesis and speedup ceiling, not a measured candidate result; it requires exact prepared-position, output, backward, depth-2, and complete-stack validation before promotion.
