# 0106A — Foundation for external-model quantization and adaptive MoE residency

## Scope and architecture decision

MyLocal_LLM keeps **one shared NumPy/CuPy/Triton computational backend**.
External pretrained models (starting with Qwen3-Coder-Next 80B-A3B) are new
model/weight adapters using the same backend, not a parallel inference engine.
The existing dense and fully resident inference/training pathways remain the
default and do not import or invoke managed experts during execution.

Reference implementation: [Strata](https://github.com/Niko1221/Strata), notably
its hybrid CPU/VRAM expert execution and expert-cache sizing guidance. This
patch is independently implemented and does not incorporate Strata source.
**Do not assume that filling all free VRAM is optimal**: on some systems it
triggers GPU memory paging and causes severe throughput regressions. Keep a
separate GPU reserve for decode activations, graph pools, attention/DeltaNet
state, and long-context KV memory.

The first target's official `Qwen3-Coder-Next/config.json` declares
`model_type=qwen3_next`, 48 layers, hidden size 2048, 512 routed experts,
10 selected experts per token, expert hidden size 512, and full attention
once every four layers. The other three use hybrid Gated DeltaNet. Its
architecture **cannot** be executed by the existing Llama-only GGUF loader.

## This patch implements

- `mini_llm/ops/quantized_weight.py`: immutable-metadata packed GGUF views for
  Q8_0, Q4_0, Q3_K, Q4_K. Zero-copy access to bytes and selecting one expert
  from a packed 3D expert tensor. CPU FP32 *reference-only* dequantization.
  This is **not** yet direct quantized matmul on GPU or CPU.
- `GGUFWeightMap.get_packed(name)`: non-dequantizing accessor that preserves the
  existing dense `__getitem__` import. The legacy Llama import can now
  deliberately dequantize supported new formats but must **not** be used for
  80B MoE inference (it materializes every full weight).
- `mini_llm/ops/expert_residency.py`: optional device budget, per-model/layer/
  expert keys, exponentially decayed usage scores, conservative replacement,
  statistics, memory-failure fallback, and reference CPU computation.
- Optional expert-executor injection in existing `ExpertsInference` and
  `MoEInference`, using `RoutingPlan.active_expert_ids()` to skip empty experts.
  Training and resident inference retain their existing weight paths.

### Minimal adapter usage

```python
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference

# host_provider(ExpertKey) -> (gate[D,F], up[D,F], down[F,D])
# These are dense reference matrices, typically backed by host RAM/mmap.
executor = HeterogeneousExpertExecutor(
    host_provider, model_id="external-checkpoint-id", d_model=D, d_ff=F,
    compute_dtype="float16", device_budget_bytes=2 * 1024**3,
    promote_after=3,
)
experts = ExpertsInference(D, F, n_experts=512,
                           expert_executor=executor, layer_id=0)
y = experts.forward(x, router_weights, expert_indices)
print(executor.cache.statistics())
```

With NumPy the executor always computes on CPU, even with a nonzero device
budget. With CuPy the CPU fallback transfers selected activations/output and
executes off-GPU; promoted experts use existing `ExpertFFNInference` GPU
kernels. This is intentionally **synchronous** and is not yet performance
competitive with a highly optimized native quantized MoE engine.

## Crucial boundaries / known limitations

1. `host_provider` currently returns *dense* host arrays. Packed GGUF views
   are ready, but a packed-weight-aware CPU/GPU expert execution callback and
   fused kernels are still required. Do not unpack the whole model eagerly.
2. No CPU/GPU asynchronous overlap, pinned ring staging, prefetch, or batched
   multi-expert scheduling is implemented. GPU uploads run synchronously.
3. A device byte budget accounts for retained expert weights **only**;
   temporary workspace can exceed it. Allocate a safe margin and benchmark.
4. A dynamic device-cache pointer must **not** be captured by existing CUDA
   graphs. Initially use uncaptured execution or capture graph-stable regions
   only. Later introduce fixed-address expert slots before graph integration.
5. The existing native `RoutingPlan` sorts and transfers an expert histogram
   to the CPU. For 512 experts x 48 layers in single-token decode, benchmark
   that synchronization and implement a device-resident decode routing path.
6. No Qwen3Next tensor-mapping, DeltaNet kernels, recurrent state, tokenizer,
   shared-expert gate, tool-call formatting, or external-model end-to-end
   inference is included in 0106A.
7. GPU performance and CPU/GPU numerical differences are not measured here;
   they need the user's CUDA machine and a reference checkpoint.

## Next milestones (incremental patches)

**0106B — Packed execution:** implement actual **fused Q4_K/Q3_K dequant-dot**
Triton kernels for decode GEMV and prefill GEMM; validate against GGUF CPU
reference, measure bandwidth and output error versus llama.cpp. Add remaining
formats that matter for selected checkpoints (e.g. Q5_K, Q6_K, IQ variants).
Preserve packed weights throughout. Reuse backend matmul dispatch.

**0106C — Real expert provider:** parse external GGUF tensor directories
without requiring all weights to be dense; map Qwen3Next routed/shared
experts, and have `host_provider` return quantized expert byte views. Support
RAM/mmap backing and per-layer/quant-format shape validation, with explicit
RAM/VRAM budgets. CPU quantized GEMV is critical: dequantizing every missing
expert on each token is not acceptable.

**0106D — Model operations and correctness:** implement Qwen3Next Gated
DeltaNet plus recurrence, gated attention, router/shared experts and model
assembly over the existing backend. Prove prefill + multi-token and incremental
decode parity against known model-generated reference activations/logits.

**0106E — Heterogeneous performance:** overlap CPU expert GEMV, cached GPU
experts, and transfers via separate execution queues; add pinned-memory
staging, contention-aware promotion, fixed-address GPU cache slots and
selective graph capture, speculative/MTP decoding where supported.

**0106F — Benchmarks and serving:** consistent single-user decode and prefill
benchmarks, expert residency hit/miss/eviction curves, PCIe traffic, RAM and
VRAM peaks, long-context behavior, cold/warm prompts, and vs llama.cpp/Strata
under identical quantization, context and generation configuration. Integrate
with existing model registry and OpenAI-compatible serving.

## Validation

```bash
python -m pytest -q \
  tests/test_0106a_packed_gguf.py \
  tests/test_0106a_expert_residency.py \
  tests/test_0085f_external_gguf.py \
  tests/test_moe_block.py tests/test_moe_experts.py
```

When the optional `gguf` dependency is installed, the Q3_K and Q4_K
reference parity checks also compare directly against `gguf.quants.dequantize`.
Those checks are skipped in environments without the dependency.
