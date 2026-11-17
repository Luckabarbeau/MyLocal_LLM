# Patch0204: sparse source and execution contracts

This is an additive formalization of the existing sparse executor, not a model
runtime. It does not implement a Qwen4Exp graph, tokenizer, attention, generation
loop, or full-model inference. Patch0201's actual mixed-GGUF header validation
and Patch0202/0203's CPU format support remain separate prerequisites.

## Canonical source versus execution

`ExpertBundle` is a frozen complete gate/up/down handle with `.weights` (tuple),
`.nbytes`, `.key`, `.owner`, and explicit `.source_tier` (`mmap` or `host`).
Canonical buffers must be read-only. Writable inputs are rejected rather than
changing caller-owned flags. Read-only views are accepted; their source owner
must not mutate writable aliases. Neither tier implies pinned memory.

`Qwen4ExpGGUFIndex.open(path, model_id=...)` uses `MultiShardTensorStore`.
Construction checks architecture `qwen4exp`, positive integral dimensions,
`top_k <= expert_count`, and every layer's exact router and gate/up/down bank
names and shapes. It does not require a hardcoded layer/expert count: the actual
48-layer, 512-expert contract is valid. All expert bank formats must match the
existing packed decoder layout. Header validation does not decode payloads.

`index.source(ExpertKey(...))` selects only that expert's three packed matrix
views and returns a bundle retaining the store. `index.provider(key)` returns
the existing tuple API. Exported arrays retain mmap base chains even if the
store is subsequently closed; new lookups after close fail. No bank is expanded
or copied, and source lookup never depends on cache residency.

Example (explicit FP32 and native CPU opt-in; legacy executor defaults unchanged):

```python
index = Qwen4ExpGGUFIndex.open(path, model_id="checkpoint")
executor = HeterogeneousExpertExecutor(
    index.provider, model_id="checkpoint",
    d_model=index.d_model, d_ff=index.d_ff,
    execution_mode="mixed", compute_dtype="float32",
    cpu_packed_backend="native", device_budget_bytes=cache_bytes,
)
```

## Execution capability and policy

`SUPPORTED_GPU_PACKED_FORMATS` and `supports_packed_gpu_type` derive from the
existing `_FORMAT_ID`: Q8_0, Q4_0, Q3_K, Q4_K, Q5_K, Q6_K. CPU support in
`BLOCK_FORMATS` is not GPU support. IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS and
Q2_0 have CPU kernels but **no GPU packed kernels**. Device wrapper construction
and `packed_to_device` reject unsupported formats before transfer/allocation.

- `mixed` (default): existing promotion/cache policy; unsupported packed GPU
  types execute on CPU without attempting materialization.
- `cpu`: no GPU expert promotion or grouped GPU expert dispatch. With a CuPy
  process backend, activation transfers still use the existing staging path.
- `gpu`: requires the CuPy process backend, promotes on first eligible request,
  and raises if residency cannot be obtained (including zero/insufficient cache
  budget or recoverable allocation failure). Unsupported formats raise explicitly.
  It never silently executes a CPU expert. Kernel/runtime/shape failures propagate.

Execution policy does not select a process backend. Set `MINI_LLM_BACKEND=cupy`
before importing the package for actual GPU execution; CuPy and Triton are needed.
FP32 is explicit, not a change to the historical FP16 default.

With zero host dequantization budget, reference execution uses one decoded row
at a time: FP32 uses `matvec_reference(tile_rows=1)` for each activation row;
FP64 uses row-wise decoded FP32 weights with FP64 accumulation, preserving the
executor's existing precision semantics. No complete three-matrix expansion is
performed. Native CPU execution is the intended path for real experts. A
positive host dequantization budget retains the existing dense host-cache policy.

## Residency, lifetime, and memory

The existing cache publishes a handle only after its materializer returns, and
retains protection/in-flight fences before allowing eviction or storage reuse.
Non-memory materialization failures propagate. The interface is synchronous and
slot-capable; this patch does **not** implement fixed slots, asynchronous weight
publication, or a new transfer scheduler. Existing asynchronous activation
workers and `routing_profiler` remain in use, with no duplicate profiler.

`index.memory_plan(static=..., state=..., activation=..., cache=...,
staging=..., headroom=..., device_total=...)` requires explicit nonnegative
integer byte reservations, checks their sum against the device total, and
returns the reservations, sum, remaining bytes, and canonical mmap logical byte
count. It allocates no weights and does not change the existing cache budget.
Mapped address space is not a RAM allocation estimate: file-backed pages can
consume page cache/RSS as accessed. Host cache, OS page cache, activation staging,
and kernel scratch must be considered separately by the caller.

## Validation scope

`tests/test_0204_sparse_iq_execution.py` covers read-only ownership, real sparse
GGUF headers with 48 layers and 512 experts, exact selected expert access,
reservation validation, generated-on-demand top-10 native CPU execution against
a bounded oracle, zero-host-cache FP32/FP64 execution, capability preflight,
strict GPU failure, complete publication failures and fake in-flight fences.
The 512-expert test creates only selected expert weights, never 512 dense experts.

Actual CuPy tests exercise all six existing GPU formats and IQ CPU fallback when
run under that backend; they skip under NumPy. Fake capability tests are not GPU
execution evidence. Related residency, asynchronous ownership and grouped tests
must also be run. Test execution results are reported separately; neither these
fixtures nor header validation establish full-model correctness or throughput.

### Recorded corrective validation (2026-11-16)

The actual old-format GPU expert test now uses the finite packed fixture from
`test_0106e_cpu_packed_and_gpu_swaps.packed`, which initializes Q3_K/Q6_K trailing
scales and Q4_K/Q5_K scale/min fields. Both oracle and actual output must be
finite before comparison. This supersedes the earlier warning-producing random
fixture run; matching NaNs are not accepted as numerical evidence.

Only the format decorator of the existing
`test_triton_kernel_matches_cpu_oracle` was narrowed to
`SUPPORTED_GPU_PACKED_FORMATS` (with its corresponding import). CPU format
parametrization remains unchanged. The six formats times three `block_k` values
produce 18 GPU GEMV cases, each checking FP16 and FP32 inputs.

Both commands used `MINI_LLM_NATIVE_BUILD_DIR` pointing to the repository-local
`.pytest_cache/native_0204` directory:

```sh
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q tests/test_0204_sparse_iq_execution.py tests/test_0106b_quantized_linear.py -k 'actual_gpu or test_triton_kernel_matches_cpu_oracle'
# 24 passed, 43 deselected in 0.81s; no warnings reported.

MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q tests/test_0204_sparse_iq_execution.py
# 12 passed, 6 skipped in 0.32s.
```

The GPU run comprises six actual 0204 GPU expert/IQ CPU-fallback cases and 18
existing GPU GEMV cases. No production code was changed in this correction.
These are bounded operator/executor results, not full-model validation.

The following six legacy NumPy/CuPy array-contract failures from the previous
broader CuPy run were intentionally left unchanged and were not rerun here:

- `tests/test_0106b_quantized_linear.py::test_packed_gguf_experts_cpu_fallback_and_cache_integration`
- `tests/test_0106c_fused_packed_expert.py::test_full_packed_expert_and_opt_out_cpu_reference`
- `tests/test_0106d_async_experts.py::test_host_cache_dequantizes_only_on_first_cpu_expert_execution`
- `tests/test_0106e_cpu_packed_and_gpu_swaps.py::test_native_complete_expert_mixed_formats_matches_fp32_oracle[fmts0]`
- `tests/test_0106e_cpu_packed_and_gpu_swaps.py::test_native_complete_expert_mixed_formats_matches_fp32_oracle[fmts1]`
- `tests/test_0106e_cpu_packed_and_gpu_swaps.py::test_native_complete_expert_mixed_formats_matches_fp32_oracle[fmts2]`

Five reported forbidden implicit conversion of a CuPy result to NumPy; the fused
reference test reported a NumPy array passed to an operation requiring CuPy.
The broader suite is therefore not claimed green.
