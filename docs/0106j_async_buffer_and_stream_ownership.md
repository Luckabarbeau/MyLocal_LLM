# 0106J — Correct caller-stream ownership for hybrid expert execution

Date: 2026-11-09. Incremental continuation of 0106I. No training, routing,
weight-decoding equations, cache replacement policy, or inference API changes.

**Verified outcome:** the original larger 0106I Q4_0 mixed-GPU benchmark now
passes unchanged with bit-for-bit agreement across instrumentation modes.
The new independent NumPy-oracle benchmark passes all 48 configurations.
Four broader, older Q3_K/Q4_K numerical checks remain failing and are recorded
below; this is not a claim that all GPU tests or full Qwen inference work.

> **0106K follow-up:** the four numerical failures recorded below are now
> resolved using matched-input FP64 references with explicit activation storage
> rounding. Original tolerances and production kernel math are unchanged.
> The broader GPU selection now passes all 49 tests. See
> `0106k_matched_precision_numerical_audit.md` and `0106k_validation.json`.
> The historical 0106J measurements and failure descriptions are preserved.

## Evidence and localization

The previous 0106I report deliberately stopped at an unlocalized mixed-GPU
failure. Its recorded next experiment was performed without production edits:

- Same 128 inputs and routes, D=F=256, 16 Q4_0 experts, top-4.
- Preload expert IDs 0..8 into a 1-MiB cache, then disable new admission.
- Compare workers=0 and workers=2, each with disabled/summary/detailed tracing.
- Independent NumPy dequantized expert oracle; FP32 activations and outputs.

With **zero evictions**, all synchronous cases agreed within
`8.731149137020111e-10`. All two-worker cases failed, with maximum absolute
errors between approximately .00465 and .00477. Dynamic cache replacement
therefore was not necessary for the corruption.

A subsequent observation checked every worker's host input against the known
input bank: maximum error **0**. Keeping all worker GPU results alive did not
fix the failure (.00505 maximum error). This ruled out corrupted input and
showed that merely extending result lifetime was insufficient.

### 1. Worker destination allocation ordered after the wrong fence

Previously, `_copy_compute_copy` allocated its result on the worker thread's
default CUDA stream, then wrote it through its nonblocking transfer stream.
The input-ready event had already been recorded before this allocation. A
CuPy memory-pool block could have intervening, queued caller-stream uses not
covered by that earlier event. Synchronizing the transfer after writing only
establishes write completion; it does not order the write after prior readers.

The result is now allocated in `submit()` on the **caller stream**, before
recording the existing producer-ready event, and passed into the worker.

```text
Caller: previous uses of reusable memory -> allocate destination -> ready
Worker:                                      wait ready -> D2H -> CPU -> H2D
Caller:                                                      Future -> consume
```

The existing worker copy-completion waits preserve pinned-buffer lifetime and
result visibility. No new synchronization, copy, or GPU allocation is added;
one existing allocation moves to the dispatcher thread. Public
`submit`/`resolve`/`forward` semantics are unchanged. The private worker helper
now receives its destination explicitly. Expert output shape equals input
shape, as required by the existing MoE residual contract.

CPU fake-stream tests verify allocation-before-event ordering, caller thread
and stream identity, no worker GPU allocation, FP16/FP32 behavior, profiling
handoff, Future behavior, and the same two existing worker waits.

### 2. Packed Triton launches ignored non-default CuPy streams

After the allocation fix, default-stream regression cases passed, while a
non-default caller-stream case failed with maximum absolute error .00116963.
A direct installed-runtime probe established:

```text
CuPy 14.2.0 current nonblocking stream: nonzero pointer
Triton 3.8.0 driver current stream:     0
```

The packed kernels used Triton's ordinary JIT runner, which obtains the
stream from its own driver. CuPy surrounding operations, memory-pool reuse,
and expert event fences therefore did not order those launches correctly.

The two launch sites in `ops/quantized_linear.py` now share a small helper:

1. Use `kernel.warmup(...)` to compile/look up the specialization without
   executing it.
2. Launch the compiled kernel with its explicit `stream=` argument set to
   `xp.cuda.get_current_stream().ptr`.

This uses the installed/pinned Triton 3.8 interface. It does not mutate Triton's
process-global driver, import PyTorch, add a global stream switch, change
kernel math, or synchronize. The compiled grid is padded to three dimensions.
Updated 0106H ABI stubs cover both zero and nonzero caller-stream pointers.

## Files in this increment

- `mini_llm/ops/expert_residency.py`: caller-owned CPU result destination.
- `mini_llm/ops/quantized_linear.py`: explicit CuPy stream at both packed launches.
- `tests/test_0106h_triton_cupy_pointer.py`: compile/launch ABI and stream tests.
- `tests/test_0106j_async_buffer_ownership.py`: CPU ownership tests and real-GPU
  repeated mixed-execution regression.
- `benchmarks/benchmark_0106j_async_parity.py`: shared deterministic fixture,
  independent oracle, fixed/dynamic placement and both caller streams. It saves
  diagnostic JSON, including failures, before raising a parity failure.
- This document and `docs/0106j_validation.json`.

0106I implementation files and historical measurements are preserved. Its
original failed GPU artifact was absent; a successful artifact now exists at
`benchmarks/results/0106i_gpu.json`. Local raw benchmark results are ignored by
Git; the numbered validation summary below is kept with the source changes.

## Tests and benchmarks

### Passing NumPy regression selection

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106*.py tests/test_0085b_external_llama.py \
  tests/test_0085c_external_safetensors.py tests/test_0085f_external_gguf.py \
  tests/test_0085g_gguf_tokenizer.py tests/test_moe_experts.py \
  tests/test_moe_router.py tests/test_topk.py
```

**222 passed, 34 skipped in .76 seconds.**

### Passing GPU ownership/parity regression

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106j_async_buffer_ownership.py
```

**9 passed in 3.55 seconds.** The real-GPU test includes 24 repeated runs:
fixed/dynamic residency × default/non-default stream × two repeats × three
profiling modes, with two CPU workers. A separate CPU fake-stream selection
also passed all eight cases. No training/inference process was using the GPU
at the initial availability check; no existing workload was stopped.

### Previously failing benchmark, unchanged

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106i_routing_trace.py --tokens 128 \
  --d-model 256 --d-ff 256 --n-experts 16 --top-k 4 \
  --device-budget-mib 1 --output benchmarks/results/0106i_gpu.json
```

All original exact-equality assertions passed:

| Mode | Mean ms/layer-token | Max absolute error vs disabled |
|---|---:|---:|
| Disabled | 1.050153 | 0 |
| Summary | .987992 | 0 |
| Detailed | 1.120175 | 0 |

These are one synthetic layer on the RTX 5080, not full-model generation.
Fixed comparison order, desktop GPU use, compilation/driver warmup and short
runs prevent a statistical speedup claim. The earlier invalid result is not a
performance baseline. Correctness, rather than throughput improvement, is the
purpose of this increment.

### Independent oracle benchmark

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106j_async_parity.py --cpu-backend native \
  --output benchmarks/results/0106j_async_parity_gpu.json
```

**48 completed cases; zero failures:** the GPU test's matrix above, additionally
including workers=0. Maximum absolute error versus the independent batched
NumPy dequant/SwiGLU reference: **1.57160684466362e-9**. Maximum difference from
the corresponding disabled-profile run: **0**. Oracle tolerance is
`rtol=1e-4, atol=1e-7`, dtype FP32; no finite differences are involved.

GPU cache retention is capped at **1,048,576 bytes**, with nine experts retaining
995,328 bytes. Fixed placement performs no evictions; dynamic placement tests
actual promotions and replacements. No model-scale allocation was attempted.

For scale only, the two disabled-mode dynamic/default-stream samples were:

- Synchronous: .766997 and .766965 ms/layer-token.
- Two workers: .745166 and .795385 ms/layer-token.

This tiny fixture does **not** demonstrate an overlap speedup. New benchmark
measurements include terminal output collection; warmup is queued before the
timed region and can have a remaining GPU tail. No extra per-token validation
synchronization is inserted. Existing routing-count host copies and worker
copy waits are still part of the runtime and its measured cost.

## Separate unresolved numerical checks

A broader GPU kernel selection ran:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106b_quantized_linear.py tests/test_0106c_fused_packed_expert.py \
  tests/test_0106g_q5k_packed.py tests/test_0106h_triton_cupy_pointer.py \
  -k 'gpu or triton'
```

**45 passed, 4 failed, 31 deselected.** The failures were:

| Existing test | Observed/reference example or maximum | Requested tolerance |
|---|---|---|
| Resident Q3_K expert | At [0,45], -14.101455688476562 vs -14.098133087158203; absolute .00332260, relative .00023568 | FP32, rtol=2e-5, atol=.002 |
| Resident Q4_K expert | At [1,44], 7506.140625 vs 7506.56005859375; absolute .41943359, relative .00005588 | FP32, rtol=2e-5, atol=.002 |
| Fused FP16 Q4_K/Q4_K | Maximum absolute error among violations .07406708; 9/1536 violating elements | FP16 actual versus FP32-input reference, rtol=.02, atol=.006 |
| Fused FP16 Q3_K/Q4_K | Maximum absolute error among violations .01021945; 3/1536 violating elements | FP16 actual versus FP32-input reference, rtol=.02, atol=.006 |

A separate process restored **only pre-0106J JIT-launch behavior in memory**
(no source files changed) and reran those four exact cases. **All four failed
again with the same reported values.** Thus the explicit-stream helper did
not cause these discrepancies. The earlier 0106I CPU run had skipped these GPU
cases and never established them as passing on this hardware.

The FP16 tests compare rounded GPU inputs against unrounded FP32 CPU inputs.
The full-expert fixtures also produce large values and cancellation-sensitive
entries. Those observations make test formulation/conditioning plausible, but
do not prove a tolerance or implementation defect. Neither kernel equations
nor tolerances were changed. The verified Q4_0 hybrid state above remains
valid; the broader GPU suite is explicitly not all green.

**Single next numerical experiment:** construct an FP64 oracle for the existing
failing fixtures, using exactly the device-rounded inputs and explicit output
rounding, and compare intermediate projection/activation errors. This would
distinguish input-rounding/test effects from an actual packed-kernel defect.
Stop before changing production math if that evidence remains inconclusive.

## Next architecture work

After characterizing those numerical checks, the next model-compatibility
increment remains actual-file Q6_K/GGUF-BF16 support and reference-correct Qwen
recurrent state/mixer integration. No Qwen decoder or IQ quantization support
was added here. The exact requested Qwen3.8 target inventory/pruning metadata
is still required; the installed 512-expert Qwen3-Coder-Next file is not that
256-expert target. No new Strata revision was examined in this increment;
0106I's recorded source review remains the reference.
