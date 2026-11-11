# 0106Y — Default recurrent graphs and indexed expert routes

## Status and default policy

Eligible Qwen3-Next recurrent decode now uses CUDA graphs **by default** on CuPy.
No environment flag is needed. `MINI_LLM_RECURRENT_GRAPH=0` remains an explicit
opt-out for diagnosis/comparison. CPU and unsupported calls keep their portable
paths. This promotes the verified 0106X improvement rather than leaving it hidden.

The general graph wrapper exposes a metadata-only stream-eligibility check.
When an adapter is used on a different caller stream, it executes eagerly instead
of reusing live graph scratch or raising a new default-path error. The wrapper's
explicit cross-stream rejection contract remains unchanged. The full tiny-model
lifecycle test now tests the **unset/default** environment, explicit opt-out,
state reset/fork/new requests, and alternate-stream eager fallback.

Base: `cbd41e463154f924e3d7441224d5d496baaf1d9b`, incremental working-tree changes.
No commit, branch, training changes, or tolerance changes.

## Current upstream Strata review

Official source: https://github.com/Niko1221/Strata

Examined revision: **`6c642df7530a5a4ba2f6ae52deaa5be436271195`**
(commit timestamp 2026-11-11 06:43:08 UTC). Source only; no downloaded release
binary, source compilation, or upstream executable was used.

Concrete transferable observations at that revision:

- [`src/kernels/cuda/verify_kernels.cu`, `resident_plan_kernel`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/src/kernels/cuda/verify_kernels.cu#L1426-L1534)
  translates device expert IDs through residency slots into weight addresses and
  preallocated group metadata. Routed ranks remain explicit.
- [`src/kernels/cuda/s2_expert_grouped.cu`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/src/kernels/cuda/s2_expert_grouped.cu)
  contains GPU hit compaction preserving route ranks, gate/up paired work and
  down-row pairing. Reusing activation loads is distinct from changing each
  output's arithmetic reduction.
- [`src/core/verify.cpp`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/src/core/verify.cpp)
  specializes fully resident stages to omit host expert-service rendezvous and
  CPU/PCIe merge work. Fully resident stages are a stronger guarantee than one
  observed all-hit route; the latter cannot safely assume future hits.
- [`include/strata/kernels/s2_expert_grouped.hpp`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/include/strata/kernels/s2_expert_grouped.hpp)
  and [`src/core/graph.cpp`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/src/core/graph.cpp)
  illustrate caller-owned stable storage/device metadata and capture/replay.
  A graph API can still synchronize at its boundary; graph usage alone is not
  proof of end-to-end asynchronous execution.
- [`src/kernels/cuda/iq_kernels.cu`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/src/kernels/cuda/iq_kernels.cu)
  has native grouped and gate/up/SwiGLU fusion paths with format/geometry limits
  and quantized activations. These are not exact replacements for our FP32
  activation path; merely copying them would change the numerical contract.
- [`docs/DETAILS.md`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/docs/DETAILS.md)
  and [`docs/EXPERTDEC.md`](https://github.com/Niko1221/Strata/blob/6c642df7530a5a4ba2f6ae52deaa5be436271195/docs/EXPERTDEC.md)
  provide scheduling/kernel context. Reported upstream performance is not a
  local reproduction.

The next larger transferable change is a persistent GPU expert-address directory,
with runtime route IDs as buffer contents and explicit invalidation/event ordering.
This patch implements only the smaller route-rank indirection; it does **not**
claim to implement GPU-side cache management or remove route readback.

## Implemented route-slot indirection

Previously `try_forward_grouped()` built ascending-expert routing weights with
`xp.concatenate()` over per-slot views on every layer/token. The existing down
kernel can read the same values directly from the original route vector.

The public primitive now accepts an optional host tuple/list:

```python
grouped_quantized_experts_forward(
    x, experts_in_accumulation_order, routing_weights,
    routing_slots=(2, 0, 1),
)
```

The existing pointer table gets a fourth uint64 column containing the route slot.
The down kernel uses it when loading the route weight. There is no extra gather,
concatenation, kernel launch, or separate metadata transfer. For ten experts this
adds 80 bytes to the already-uploaded pointer table. Gate/up/down computation,
FP32 product storage, and sequential expert-order accumulation are unchanged.

Omitting `routing_slots` preserves the old public contract. Duplicate slots and
strided/reversed/broadcast route vectors are supported. Invalid host slots are
rejected before allocation/launch. The executor uses this path directly, without
an opt-in flag. Protected dispatch scopes and cache miss/CPU fallbacks remain.

## Measurements

### Isolated grouped dispatch

`benchmark_0106y_grouped_dispatch.py` uses finite synthetic packed fixtures,
D=2048, F=512, E=10, formats Q4_K/Q4_K/Q6_K, 50 iterations on RTX 5080.
This benchmark requires the repository's pytest development fixtures.

| Per-call measurement | Materialized routes | Indexed routes |
|---|---:|---:|
| Host submission | 145.05 us | 107.55 us |
| Wall through completion | 148.13 us | 108.36 us |
| CUDA-event full-loop interval | 144.79 us | 104.90 us |
| Warmed 3-kernel graph replay interval | 73.53 us | 81.82 us |

Host submission falls about 25.9%, but the graph replay interval is higher for
indexed routes in this measurement. This is **not** an unconditional GPU-kernel
speedup. Event intervals include scheduling/host gaps. The warmed graph excludes
metadata upload, validation, allocation and route permutation; it is not used as
an executor optimization in this patch. Outputs are bitwise equal (max error 0).
No confidence interval or general performance claim is established.

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  benchmarks/benchmark_0106y_grouped_dispatch.py \
  --output benchmarks/results/NEW_GROUPED_DISPATCH_FILENAME.json
```

### Real-model decode

Same installed Qwen3-Coder-Next-UD-Q4_K_XL GGUF, RTX 5080, eight-token prompt
`def add(a, b):\n    return`, sixteen generated tokens, context64, two serial
requests, packed weights and FP32 activations. Speculation disabled.

| Configuration | Warm decode tokens/s | Mean decode ms |
|---|---:|---:|
| Before: graphs explicitly enabled, materialized routes | 21.5253 | 46.4570 |
| After: graphs enabled by default, indexed routes | 21.4896 | 46.5340 |

**End-to-end throughput is effectively flat; no production speedup from indexed
routes is claimed.** The previous X graph result was 21.9745 tokens/s. Neither
30 tokens/s nor the user's larger-model production results have been reached.
The benchmark still includes summary routing traces and host-stage instrumentation.

All four requests match the established baseline token IDs. Warm decode has
7200/7200 hit assignments, zero expert uploads/evictions and no CPU expert work.
The after run reserves 9,088,803,328 bytes across main/graph private pools, with
36 recurrent graphs; peak RSS is 11,493,965,824 bytes. Existing 16 GiB available
host reserve, 12 GiB observational RSS guard, 2 GiB VRAM reserve and 8 GiB expert
cache budget are unchanged. Graph pools remain outside the main pool hard cap.

Reproduce the default without a graph environment flag:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  benchmarks/benchmark_0106r_generation.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --max-new-tokens 16 --requests 2 --context 64 --cpu-workers 2 \
  --expert-cache-mib 8192 --device-budget-mib 12288 --device-reserve-mib 2048 \
  --ram-reserve-mib 16384 --max-rss-mib 12288 \
  --output benchmarks/results/NEW_DEFAULT_FILENAME.json
```

Raw local artifacts: `0106y_before_indexed_routes.json`,
`0106y_default_indexed_routes.json`, `0106y_grouped_dispatch.json` and
`0106y_default_synthetic_parity.json` under ignored `benchmarks/results/`.
The real-model run preceded the stream-eligibility safeguard; the final safeguard
is covered by the final CUDA suite and does not alter its single-stream path.

## Correctness and validation

- Final CUDA integration: **385 passed, 11 skipped in 8.40 s**.
- Final broad CPU regression: **1346 passed, 177 skipped in 9.35 s**.
- Indexed/materialized CUDA outputs are bitwise equal for all six existing format
  combinations, reversed/broadcast routes, repeated handles and duplicate slots.
- Synthetic full-model default-path comparison passes on a nondefault stream:
  maximum full-forward error 1.788e-6 versus CPU and 1.644e-6 versus independent
  FP64. Largest incremental CPU/CUDA error is 2.086e-6. Existing
  `rtol=4e-4`, `atol=8e-6` remain unchanged; greedy IDs match.
- The explicit graph wrapper still rejects cross-stream scratch reuse; the model
  adapter's default path falls back eagerly, verified through the full model.

Exact commands and machine-readable measurements are in `0106y_validation.json`.
These are not strict real-model llama.cpp parity or a full-repository CUDA pass.

## Performance target and next step

The user reports approximately 80 decode tokens/s for Qwen3.8-Flash-Next Q3 and
30–40 at Q4 on this hardware, with more active parameters than the installed
80B Qwen-Coder model. Those reports set the performance objective; this patch
does not independently verify that model configuration or infer proportional
speed from total/active parameter counts.

The current short warm all-hit case exposes significant remaining overhead.
A useful next A/B is persistent device address/rank metadata and stable grouped
scratch, then graph replay of the expert region without changing the kernels.
Preserve cache invalidation and in-flight ownership; do not retain evicted expert
allocations just to keep graph addresses valid. Measure an uninstrumented serving
baseline separately from the summary-traced benchmark before attributing its
overhead to GPU computation. Do not adopt Strata's activation quantization or
multi-token scheduling as an implicit FP32/single-token replacement.
