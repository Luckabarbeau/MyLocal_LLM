# 0106E — Packed CPU execution and bounded expert swapping

## Purpose

0106E extends the **existing** `ExpertFFNInference` / `ExpertsInference` /
`HeterogeneousExpertExecutor` path. It does not introduce a separate runtime
and does not modify training. Qwen3-Coder-Next architecture and GGUF model
loading are **not** yet complete.

## A. Quantized CPU compute

A minimal C++17 CPU kernel directly consumes GGUF Q8_0, Q4_0, Q3_K and Q4_K
packed bytes. It supports native packed GEMV (`quantized_linear`) and a full
SwiGLU expert forward (gate, up, down) with one transient FP32 intermediate
activation vector per token. No full dense weight matrix is allocated.
The C++ functions release the Python GIL when called via `ctypes.CDLL`, allowing
0106D CPU workers to overlap with CuPy GPU operations. This is a generic
correctness-first packed decoder; **it does not incorporate ggml's highly
vectorized AVX/AMX CPU kernels**. Benchmark before assuming a speedup.

The helper builds `mini_llm/ops/native_packed_cpu.cpp` on first use via local
`g++` or `clang++`, writing the `.so` into
`~/.cache/mylocal_llm/native/` (override with `MINI_LLM_NATIVE_BUILD_DIR`).
No admin rights are required. Rebuilds occur when source content changes.
Use the reference path if no compiler exists. `native` fails clearly on build
errors; `auto` warns once and falls back to FP32 reference decoding.

To opt in on the managed executor:

```python
executor = HeterogeneousExpertExecutor(
    host_provider,
    model_id="external-model",
    d_model=2048, d_ff=512,
    device_budget_bytes=6 * 1024**3,
    cpu_workers=3,
    cpu_packed_backend="native",
    idle_eviction_requests=8192,
)
```

The initial default remains `cpu_packed_backend="reference"` so existing tests
and users retaining the D-era decoded host cache have **identical semantics**.
Use `host_dequant_budget_bytes=0` when choosing the native packed kernel: no
FP32 host expert cache is required. Dense CPU experts remain unchanged.
Standalone CPU operations can be invoked with
`quantized_linear(x, packed_w, cpu_packed_backend="native")`.

## B. GPU expert swapping (VRAM cache)

With an authoritative packed copy already in system RAM (or mmap), evicting a
GPU expert usually means **dropping its VRAM byte buffers** rather than
transferring them back to RAM. This is much cheaper than classic bidirectional
swap. Subsequent reuse can promote it again through the existing packed upload.

The existing decayed request/byte-density and `promote_after` scoring remain.
0106E adds:

- Auto-eviction **only when needed** for promotion, subject to replacement
  margin and total VRAM-byte budget; `swap_evictions` records these replacements.
- Optional proactive idle cleanup: `idle_eviction_requests > 0`. Explicit
  `executor.cache.evict_idle(min_idle_requests, target_bytes=0)` is also
  available. Set `0` to disable automatic idle cleanup.
- **Protected router working set** while a layer dispatch is in progress.
  Selected residents cannot be evicted to promote a competing expert that
  will execute later in the same router invocation.
- **CUDA event fences** for weights read by queued GPU kernels. A single
  event per managed MoE forward protects its selected resident experts until
  the kernels in the calling stream have finished. Eviction checks ignore
  in-flight weights, avoiding GPU use-after-free. Direct `executor.forward`
  calls use a per-call event instead.
- Counters: `hit_rate`, `busy_victim_skips`, `swap_evictions`, `idle_evictions`,
  `inflight_dispatches` and `resident_bytes`.

`device_budget_bytes` is a **logical weight-retention budget**, not a complete
VRAM allocator. Leave reserves for KV cache, activations, CUDA graph pools, and
staging buffers. Cached packed weights are reference-counted by Python/CuPy
and released back to its memory pool; actual driver-reported VRAM may not
immediately fall when an expert is evicted. There is no disk spilling or
asynchronous prefetch pipeline in this patch. Do not use cache mutation during
CUDA graph capture; keep dynamic decisions outside captured regions.

### Tests / benchmarks

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q \
  tests/test_0106e_cpu_packed_and_gpu_swaps.py \
  tests/test_0106a_expert_residency.py \
  tests/test_0106d_async_experts.py

MINI_LLM_BACKEND=numpy .venv/bin/python \
  benchmarks/benchmark_0106e_cpu_packed.py \
  --format Q4_K --d-model 2048 --d-ff 512 --tokens 1 --iterations 30

MINI_LLM_BACKEND=cupy .venv/bin/python \
  benchmarks/benchmark_0106d_heterogeneous_moe.py \
  --format Q4_K --d-model 2048 --d-ff 512 --n-experts 64 \
  --top-k 10 --device-budget-mb 64 --cpu-packed-backend native \
  --host-dequant-budget-mb 0 --idle-eviction-requests 1000 \
  --workers 3 --iterations 20
```

The CuPy/Triton paths and CUDA event correctness have to be exercised on the
user's RTX 5080; this CPU-only environment cannot validate them. Report
single-layer wall time, cache hit rate, swap evictions, CPU workers, and GPU
utilization. Never equate this with full-model decode tokens/sec.
