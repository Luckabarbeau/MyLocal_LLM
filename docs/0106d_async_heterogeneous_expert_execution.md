# Patch 0106D — Overlap CPU and GPU execution of selected MoE experts

This is an **opt-in extension of the shared backend** from 0106A–C, not an
independent inference runtime. Training, custom native models, and the
fully resident inference path are unchanged. A single managed `ExpertsInference`
can now dispatch selected experts to CPU workers and the existing resident
GPU kernel path before waiting for their results.

## Design

- `HeterogeneousExpertExecutor(cpu_workers=N)` enables an executor-owned worker
  pool. `ExpertsInference.forward()` starts all selected expert jobs, enqueues
  GPU-resident kernels on the caller's stream, and only then resolves CPU jobs.
  The final weighted sparse scatter preserves original expert-id order.
- `submit()/resolve()` can also be used by custom external-model adapters;
  plain `forward()` retains its synchronous contract. `close()` drains workers.
- On **CuPy**, each worker reuses a nonblocking CUDA stream. A producer event
  recorded after the routing gather makes the CPU worker's D2H transfer wait
  for its input. Both D2H and H2D use **pinned staging memory** and explicitly
  synchronize the worker stream at the points where the CPU reads input and
  where the output staging buffer can be freed. Resident GPU work can proceed
  on the caller stream in the meantime. Future resolution propagates worker
  errors, and the scatter waits for the finished GPU output as necessary.
- On **NumPy**, the same dispatch contract uses background CPU workers for
  correctness/concurrency testing; potential CPU throughput depends on GIL,
  BLAS threading, and thread scheduling.
- A new optional `CPUDequantizedExpertCache(host_dequant_budget_bytes=...)`
  caches **only already selected** packed GGUF experts as FP32 CPU matrices.
  Its bounded LRU avoids decoding the same missing expert's Q3/Q4 bytes on
  every token while keeping the original quantized host provider unchanged.
  A zero byte budget disables the decoded host cache. Use an explicit budget
  that fits alongside GGUF mmap pages and other application RAM requirements.
- This patch uses existing `AdaptiveExpertCache` replacement and promotion
  policy. It does not force every expert miss into VRAM. It doesn't duplicate
  the quantized Triton kernels and does not alter the training MoE dispatcher.

## Example

```python
executor = HeterogeneousExpertExecutor(
    host_provider, model_id="qwen-next-checkpoint", d_model=2048, d_ff=512,
    device_budget_bytes=6 * 1024**3,   # configure after reserving VRAM for state
    host_dequant_budget_bytes=512 * 1024**2,  # optional CPU hot expert cache
    compute_dtype="float16", promote_after=3, cpu_workers=3,
)
managed = ExpertsInference(2048, 512, n_experts=512,
                           expert_executor=executor, layer_id=layer_number)
# managed.forward(x, router_weights, router_indices)  # existing API
# ... shutdown after request/service lifetime:
executor.close()
```

One executor should normally be shared across MoE layers so the worker pool,
VRAM budget, and host dequantization budget cover the *whole model*. The example
VRAM budget is illustrative, not an optimal allocation for a specific GPU.

## Validation

```bash
python -m pytest -q \
  tests/test_0106d_async_experts.py \
  tests/test_0106a_expert_residency.py \
  tests/test_0106b_quantized_linear.py \
  tests/test_0106c_fused_packed_expert.py \
  tests/test_moe_experts.py

# Benchmark one synthetic MoE layer, synchronous vs overlapping mode:
MINI_LLM_BACKEND=cupy .venv/bin/python \
  benchmarks/benchmark_0106d_heterogeneous_moe.py \
  --format Q4_K --d-model 2048 --d-ff 512 --n-experts 64 --top-k 10 \
  --device-budget-mb 8 --host-dequant-budget-mb 512 --workers 3 \
  --iterations 20
```

The deliberately small 8 MiB GPU budget keeps some of the ten routed
experts on CPU (after the warmup), rather than eventually promoting all of
them to VRAM. The benchmark verifies numerical parity and reports single-layer
latency and cache statistics. **It does not claim full-model tokens/s**. Tune CPU worker
count (1–4 initially), CPU BLAS threading, VRAM expert budget, and host cache
budget on the target hardware; more parallelism can cause DRAM/PCIe contention.

## Important limits / next optimizations

1. **CUDA path not tested without CuPy/Triton and a real GPU.** Validate the
   CuPy worker-stream/event and fused Q3/Q4 kernels on the RTX 5080.
2. The CPU packed path remains a *reference* implementation: the LRU reduces
   repeated dequantization but does not replace the need for native quantized
   CPU GEMV (or an optimized ggml/llama.cpp-backed operator). CPU FP32 GEMV
   consumes far more host bandwidth than Q4_K direct execution.
3. There are still **one D2H and one H2D per nonresident expert**. Later
   optimize by coalescing transfers for all selected cold experts per layer,
   and investigate persistent staging rings and expert-prefetch batches.
4. The worker stream deliberately synchronizes for CPU read/output lifetime.
   This is overlapping *experts*, not an entirely nonblocking multi-stage
   pipeline; transferring a missing expert to VRAM remains synchronous.
5. The device cache does not have fixed-address slots. **Do not CUDA-graph
   capture the dynamic-residency execution path**: future selective graph
   capture requires stable device memory addresses and explicit concurrency
   coordination.
6. No model-level Qwen3-Coder-Next assembly/Gated DeltaNet, prefill batching,
   GGUF expert tensor adapter, or speculative decoding is included here.
