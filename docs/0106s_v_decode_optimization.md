# 0106S–V — Measured decode optimization toward 30 tokens/s

## Status

**Verified real-model warm decode reached 20.193 tokens/s, not 30 tokens/s.**
The 120–130 tokens/s Strata result reported by the user remains the longer-term
reference. These are short single-prompt measurements, not universal throughput.

Base revision: `cbd41e463154f924e3d7441224d5d496baaf1d9b`.
No commits, branches, training changes or numerical-tolerance reductions were made.
The earlier 0106R reports and user work were preserved.

An additional **0106W experimental recurrent CUDA-graph path is unresolved**.
It is opt-in through `MINI_LLM_RECURRENT_GRAPH=1`, disabled by default, and is
not part of the measured 20.193 tokens/s result. Its full tiny-model test failed
with CuPy's explicit rejection of cuBLAS during stream capture. Do not enable it
for production or claim the complete test suite passes.

## Controlled workload

Installed model:
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`

RTX 5080 16 GB; FP32 activations; packed quantized weights; raw greedy completion.
Prompt `def add(a, b):\n    return`, eight prompt tokens, context 64, 16 generated
tokens, two serial requests per process. Decode throughput counts the actual
15 decode forward calls, excluding prefill. No speculation, full-logit replay,
reference process, cache clearing, full-model hashing or concurrent model runs.

The larger cache test used **8192 MiB expert cache**, **12288 MiB private pool**,
**2048 MiB free-VRAM reserve**, **16384 MiB host reserve**, **12288 MiB RSS guard**
and two CPU expert workers. The RAM guard remains observational, not an OS limit.

Command pattern, from the repository root:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  benchmarks/benchmark_0106r_generation.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --max-new-tokens 16 --requests 2 --context 64 --cpu-workers 2 \
  --expert-cache-mib 8192 --device-budget-mib 12288 --device-reserve-mib 2048 \
  --ram-reserve-mib 16384 --max-rss-mib 12288 \
  --output benchmarks/results/NEW_UNIQUE_FILENAME.json
```

The executable still identifies its benchmark schema as 0106R; filenames identify
optimization stages. Output files are never overwritten. First use includes
lazy compilation/uploads; page/JIT caches were not cleared. Warm repeats the
same prompt with fresh generation state and retained model/cache/history.

## Measured progression

| Stage | Warm decode tokens/s | Local raw artifact |
|---|---:|---|
| Earlier R, 1 GiB cache | 1.746 | `0106r_real_generation_1024.json` |
| S baseline, 8 GiB cache | 3.230 | `0106s_baseline_cache8192.json` |
| S single-token dispatch | 3.694 | `0106s_singleton_cache8192.json` |
| S constant-cost hit tracing | 8.622 | `0106s_hit_trace_cache8192.json` |
| T grouped packed experts | 11.434 | `0106t_grouped_cache8192.json` |
| U recurrent/norm/shared fusion | 17.183 | `0106u_fused_cache8192.json` |
| V fused normalized router | **20.193** | `0106v_router_cache8192.json` |

Artifacts live under `benchmarks/results/` (ignored local files).
All 8 GiB-cache warm requests had **7200/7200 decode expert-cache hits**, zero
CPU expert assignments, and zero expert-weight uploads. Their token IDs matched
the pre-optimization baseline exactly:

`[264, 488, 293, 271, 750, 32256, 2877, 11, 293, 982, 262, 470, 264, 481, 293, 271]`

The 8 GiB cache budget was not fully occupied by this short repeated prompt:
3326 experts, about 6.35 GB packed bytes, were resident. Therefore 100% hits here
must not be generalized to longer/new coding workloads.

The V report reserved 8,989,087,744 private-pool bytes and reported peak process
RSS 11,421,093,888 bytes, both below their limits. No memory guard was relaxed.

## Implemented changes

### S: managed single-token dispatch and tracing

`mini_llm/ops/experts_inference.py` specializes managed `N == 1` routing with unique
expert IDs: one selected-ID readback, host ordering by expert ID, the same input
row for all experts, basic routing-weight slices, and no indexed gather/scatter.
Duplicate IDs, multi-token prefill and native unmanaged inference retain general
routing. Submit-all/resolve-all, error draining and in-flight protection remain.

`mini_llm/ops/expert_residency.py` avoids copying/scanning the complete resident
key set for each traced ordinary hit. A hit cannot evict when idle eviction is
disabled. Misses and idle eviction preserve the full observation. The large gain
here exposes prior profiling overhead, not a numerical-kernel improvement.

### T: grouped packed GPU FFN

`mini_llm/ops/grouped_quantized_experts.py` reuses existing GGUF lane decoders and
fused SwiGLU, with three grouped launches: gate/up activation, weighted down
projection, sequential expert-order accumulation. Only a small pointer table is
uploaded; no dense expert matrices are allocated. FP32 product storage and
ascending-ID addition order are preserved.

The executor probes eligibility without changing counters. It batches only
compatible resident packed experts inside the protected dispatch scope.
CPU misses, mixed layouts and unfused opt-outs retain the original hybrid path.
Tracing records one grouped-compute interval, not duplicated per-expert times.

### U: general low-level fusions

- `rmsnorm_inference.py`: contiguous FP32 inference RMSNorm in one CUDA kernel;
  existing FP16 and CPU/training behavior and opt-out remain.
- `gated_delta_decode.py` / `gated_delta_inference.py`: generic FP32 single-step
  convolution/history, sum-L2, delta-rule memory update/readout and gated RMSNorm.
  Arbitrary head maps and rectangular key/value dimensions are tested. CPU,
  FP64, multi-token and unsupported cases keep the portable implementation.
  Disable with `MINI_LLM_GATED_DELTA_DECODE=0` before mixer construction.
- `external/gguf_projection.py`: shared lazy weight resolution and packed SwiGLU
  projection helper; Qwen MoE maps its shared FFN to the existing fused kernel.
  Dense/mixed fallbacks and upload/operator injection remain supported.

### V: normalized inference router

`router_decode.py` fuses full softmax, exact top-k selection and selected-mass
renormalization for FP32 single-row opt-in routing. It keeps the normalization
floor, stable ID tie order, and reference nonfinite behavior. Native/default
router behavior and training are unchanged. Unsupported shapes/dtypes fall back.
Disable with `MINI_LLM_ROUTER_DECODE=0` before router construction.

The first CUDA test run failed because NVRTC could not find `math.h`. One localized
correction removed the header dependency and used CUDA bit intrinsics for NaN/
infinity constants. The exact same tests then passed, with tolerances unchanged.

## Correctness evidence

- S singleton tests: NumPy **14 passed, 12 skipped**; CuPy **24 passed, 2 skipped**.
- S hit observation + existing tracing/singleton CPU checks: **35 passed, 13 skipped**.
- T + S + trace/async integration CUDA checks: **104 passed, 4 skipped**.
- U + T + S + trace/async CUDA checks: **206 passed, 10 skipped**.
- V new router + existing router: CPU **20 passed, 23 skipped**; CUDA **43 passed**.
- Grouped default-tile results were bitwise equal to the matched sequential GPU
  FFN in tested formats, including mixed Q4_K/Q5_K/Q6_K. Independent FP64 oracle
  checks retained the established packed-FFN tolerance.
- FP32 RMSNorm maximum errors: 2.38e-7 versus FP32, 7.27e-7 versus FP64.
- Nine-step recurrent 128-by-128 checks: output maximum error 1.43e-6 versus
  portable FP32 and 1.69e-6 versus FP64; final memory 1.49e-8 / 4.00e-8;
  raw history exact. Existing rtol=3e-4, atol=5e-6 retained.
- Router IDs matched exactly, including ties/near-ties/nonfinite fixtures;
  probability tolerance rtol=atol=2e-7 retained.

These focused passes do not establish a fresh full-repository pass, longer-model
state accuracy or strict real-model llama.cpp parity. Full test success must not
be claimed while the experimental W test remains failing.

## Remaining bottleneck and experimental W failure

V warm host attribution, per token:

- Mixers: 17.013 ms
- Router: 2.025 ms
- Experts: 19.469 ms
- Entire MoE: 28.781 ms (includes router/experts; do not sum these)
- Normalizations: about 1.370 ms

Measured forward latency was 49.521 ms/token. These are inclusive host times,
not isolated GPU kernel durations. The 30 tokens/s target needs <33.333 ms/token.

W adds `ops/inference_graph.py`, an opt-in wrapper in `qwen3next_mixer.py`, and
`tests/test_0106w_recurrent_graph.py`. Private capture buffers/pool keep warmup
from mutating caller state and retain captured temporary storage. The isolated
array-graph and failed-allocation safety tests passed, but the full tiny hybrid
model test failed before replay:

```text
NotImplementedError: calling cuBLAS API during stream capture is currently unsupported
```

Failing command:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -x -q -p no:cacheprovider \
  tests/test_0106w_recurrent_graph.py
```

Observed **1 failed, 2 passed, 1 skipped**. Failure path:
`Qwen3NextRecurrentMixerInference._forward_uncaptured` -> GGUF dense projection
-> `hidden @ weight.T` -> CuPy cuBLAS stream-capture rejection. Portable recurrence
also contains matmuls, so removing only the first blocked projection would not
establish a capture-compatible full graph. No speculative replacement or second
implementation was attempted; W stays disabled and unresolved.

**Next useful experiment:** establish a capture-compatible FP32 dense projection
primitive and a capture-safe recurrent primitive, then rerun that exact tiny-model
graph test before any real-model W benchmark. Preserve the S–V verified state.
Longer and different prompts, CPU-miss throughput, graph accounting and compatible
Strata/llama comparisons remain required before declaring a production speed win.
