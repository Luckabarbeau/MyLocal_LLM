# 0106I — Runtime audit, routing traces, and cache-policy baselines

Date: 2026-11-09. Incremental changes to the existing NumPy/CuPy/Triton
managed-expert path; no new inference engine, training edits, or commit.

**Status:** instrumentation and focused correctness checks pass. Full Qwen
inference remains unavailable. A larger mixed packed-GPU benchmark failed
parity and remains an explicit correctness gate. Do not interpret this patch
as verified model-scale hybrid inference or a throughput optimization.

> **0106J follow-up:** the historical mixed packed-GPU failure below has now
> been resolved by caller-owned worker destinations and explicit CuPy-stream
> packed-kernel launches. The original exact-parity benchmark passes unchanged.
> See `0106j_async_buffer_and_stream_ownership.md` and `0106j_validation.json`
> for current validation, including separate legacy numerical failures. The
> original 0106I observations and machine-readable snapshot are preserved.

## 1. Repository audit

0106A–G documentation and A–H tests already exist. H is the Triton/CuPy
pointer adapter; I is the next unused identifier within this series.

| Component | Established implementation and limits |
|---|---|
| Packed weights | `ops/quantized_weight.py`: zero-copy packed GGUF views and reference decoding for Q8_0, Q4_0, Q3_K, Q4_K, Q5_K. No Q6_K or IQ-family support. |
| Expert kernels | `ops/quantized_linear.py`, `native_packed_cpu.py`, `native_packed_cpu.cpp`: packed CPU C++ GEMV/SwiGLU, Triton packed linear and fused gate/up. CPU reference and native paths are separate choices; native is not an AVX/AMX-tuned library. |
| GPU cache | `ops/expert_residency.py`: byte-bounded, decayed request-density replacement, admission after reuse, CPU fallback, idle eviction, selected-set protection and event-fenced allocations. No static pinned region, profile loading, calibrated cost model, GPU slot table, or asynchronous expert-promotion queue. The byte budget excludes state, workspaces, allocator pools and reserve. |
| CPU/GPU overlap | Existing persistent Python worker pool; native ctypes kernels release the GIL. CuPy workers use nonblocking streams, pinned activations and producer events. Each CPU expert currently stages its own input/output and waits for copies. Promotions occur in the caller path, not a bounded transfer pipeline. |
| Routing | `ops/routing_plan.py` groups active experts once and copies host counts once per dispatch. There is still a host synchronization per plan. `ExpertsInference` launches all jobs before collecting, drains failures, and accumulates in deterministic expert order. |
| Host/SSD | Provider contract accepts host/mmap views, and the Qwen GGUF index retains its reader. Optional bounded *dequantized* host LRU is not a bounded packed-RAM/SSD tier. No asynchronous cold-expert retrieval contract or n-gram row reader yet. |
| Dense external models | Llama-family Safetensors/GGUF loading, dense inference, prefill, incremental KV decode and optional specialized CUDA graph path exist. Dense GGUF loading expands quantized weights; it is not the packed routed-expert path. |
| Tokenization/serving | Canonical external loading and serving currently select the Llama adapter. Embedded GGUF reconstruction supports GPT-2/SmolLM pre-tokenization, not Qwen2; a tokenizer JSON is an alternative, not a Qwen architecture adapter. Chat-template metadata is not a general Jinja renderer. |
| Qwen | `external/qwen3next_gguf.py` is a metadata/expert index, explicitly **not** a decoder. Gated DeltaNet/convolution/recurrent state, gated attention/KV geometry, shared-expert and norm/residual wiring, non-expert format coverage, tokenizer/logit parity and loader integration remain missing. |
| Existing tests | 0106 A–H cover parser/views, quantized operations, CPU/native parity, async dispatch, cache and event safety, Qwen synthetic indexing, Q5_K and pointer ABI. 0085 tests cover Llama full-forward and cached decoding, loaders and tokenizer. Operator/layer benchmarks do not establish full-model decode throughput. |

### Installed model versus requested target

A metadata/tensor-directory-only inspection of the installed file
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`
found:

- Declared architecture `qwen3next`, model name `Qwen3-Coder-Next`.
- 48 layers, hidden width 2048, 512 routed experts, 10 selected, expert FFN 512.
- Full attention interval 4; 16 attention heads, 2 KV heads, key/value width
  256, rotary dimensions 64. SSM convolution kernel 4, state size 128.
- `gpt2` tokenizer with `qwen2` pre-tokenizer, 151936 vocabulary entries.
- 843 tensors: F32 361, Q8_0 297, Q4_K 130, Q5_K 49, BF16 5, Q6_K 1.
- All 144 routed tensors: Q4_K 94, Q5_K 49, **Q6_K 1**.
- Unsupported routed tensor: `blk.46.ffn_down_exps.weight`, logical shape
  `[512, 2048, 512]`, Q6_K. A successful layer-0 smoke test cannot establish
  whole-file compatibility.
- BF16 tensors in layer 46 include `attn_gate`, `attn_qkv`, and the three
  shared-expert projections. GGUF BF16 needs deliberate support too.

This is **not** metadata for the requested 256-expert Qwen3.8 IQ1_M target.
Its exact model repository/revision or local GGUF metadata has not been
established. Do not infer pruning maps, n-gram semantics or IQ quantization
coverage from that name. No full-model weights were loaded for this audit.

## 2. Strata source reference

Official source: <https://github.com/Niko1221/Strata>

Examined revision: **`504b143342a9fe1aa8320a7b31e11f5c0ac265c3`**.
Source-only clone and static review; no release binaries, builds or execution.

Reviewed `docs/DETAILS.md`, `docs/MODELS.md`, `docs/MULTI_GPU.md` and the
following implementation areas at that revision:

| Source | Transferable implementation detail |
|---|---|
| `include/strata/core/expert_cache.hpp`, `src/core/expert_source.cpp` | Profile-seeded resident experts, GPU hits and CPU misses with optional PCIe execution share. Missing experts need not all be uploaded. Host modes include arenas, mappings, GPU-complement-only RAM and budgeted hot RAM. |
| `src/program/generate.cpp` | Decayed frequency, same-layer replacement with gain thresholds and capped adaptation rounds. Publish new residency only after copies complete. Not simply global LRU. |
| `src/kernels/cpu/pool.cpp` | Persistent workers schedule row tasks across experts; multiple token rows reuse weights. Gate/up and down are phased with activation quantization. More scalable than allocating a Python job for every tiny matrix operation. |
| `src/core/verify.cpp`, `src/core/graph.cpp`, `src/prefill/prefill.cpp` | Event-ordered side streams, graph execution and mapped doorbells. All-resident paths can avoid host expert dispatch. Prefill has separate copying/staging machinery. These mechanisms are not automatically transferable as one Python thread wrapper. |
| `src/core/pinned.cu`, `include/strata/core/pinned_ring.hpp` | Bounded pinned bounce rings with event-guarded reuse and capped/partial registration. Source buffers must outlive DMA; async API spelling alone does not establish overlap. |
| `src/kernels/cuda/s2_expert_grouped.cu`, `src/kernels/cpu/native_expert.cpp`, `src/prefill/moe_iq4_w4a8.cu` | Specialized packed kernels, integer dot products and scaling; decode GEMV and multi-token prefill GEMM are separate tuning problems. |
| `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp` | PLE weight lookup hashes token history to 16 rows/token for the supported architecture. Async issue/collect, deduplicated/sorted page reads, bounded I/O and a row cache. Actual row-cache replacement is eight-way round-robin, despite a clock-cache header description. Do not assume these semantics for an unidentified GGUF. |
| `docs/MULTI_GPU.md`, `src/core/verify.cpp` | Qwen contiguous-layer pipeline partitioning rather than tensor parallelism; separate DeepSeek engines/policies must not be conflated with this path. |
| `bench/bench_vs_llama.py`, `strata_cli/bench_standard.py`, `bench/results/` | Alternating comparisons/warmup and memory reporting; server prompt time is not necessarily client streaming TTFT. Historical results are not measurements of the examined revision. |

Priorities for integration: profile and measure routing, coalesce activation
staging, reuse bounded pinned buffers, adapt residency outside critical compute,
then reduce host dispatch with stable device metadata. Preserve event ownership
and exact expert selection. Strata also has optional route-tail skipping and
route-prior biases; these are **not correctness-preserving placement changes**
and are not enabled or implemented here. Speculative decoding is excluded.

## 3. Implementation and invariants

`RoutingTrace` is optional on `HeterogeneousExpertExecutor`. Existing callers
need no changes. Only managed `ExpertsInference` invokes trace layer scopes;
native resident inference and training do not instantiate this instrumentation.

The mathematical invariant remains:

```text
y[token] = sum over selected slots(weight[token, slot]
          * down[expert](SiLU(gate[expert](x[token])) * up[expert](x[token])))
```

Neither the trace nor offline replay changes routing IDs, weights, expert
mapping, placement policy, accumulation order, or production kernels.

### Modes and measurements

- **Summary:** reuses the existing plan's host counts; no additional routing
  ID/weight copies and no CUDA timing events. Counts token assignments by
  expert separately from batched expert execution requests. Records host
  stage intervals; it is low-overhead relative to detailed tracing, not free.
- **Detailed:** copies flattened token/top-k IDs and routing weights, which
  **can synchronize**. Keeps bounded route and expert-record deques, linked
  through dispatch IDs. Per-layer flattened token offsets support prefill and
  decode; batch lanes are flattened, not independent conversation IDs.
- Decisions include CPU/GPU hit/promotion, cache misses, evicted IDs, successful
  promotion bytes, and activation staging bytes. CPU-only requests have null
  GPU hit/miss fields, rather than pretending to miss an enabled cache.
- Host selection, CPU compute, copy enqueue, mandatory worker waits and resolve
  waits are measured separately. Detailed CUDA events measure compute, upload
  and activation-copy **stream intervals**, not isolated DMA bandwidth.
- `snapshot()` only polls completed CUDA events. It does not call synchronize.
  Pending timers remain pending, never reported as zero durations. Timer queues
  and record counts are bounded; dropped-record/timer counters are explicit.
  A single large prefill record can still be large: limits are record counts,
  not a byte-budgeted logging system. Session-long counts/frequencies survive
  record eviction. Snapshots copy retained mutable records.
- Worker timings overlap: their sum is not layer latency. Resolve wait and
  worker waits are not a complete accounting of all CUDA synchronization.
  Router-plan sync, scatter cost, utilization and peak memory are not newly
  instrumented here. Use the existing performance profiler/external tools for
  those measurements. CPU-only has no fabricated GPU times.

### Correctness fix discovered on the actual GPU

Installed CuPy **14.2.0** exposes `Event.done`, not `Event.query()`. The prior
cache caught that AttributeError conservatively and could keep completed
allocations protected indefinitely. `_busy_keys` now polls `.done` for CuPy,
retaining `.query()` compatibility for existing fake/adapter events. Unknown
or failing fences still remain protected. The new profiler uses `.done` too.
No device synchronization was added. A real-GPU regression test checks that
completed dispatches leave no busy keys/in-flight groups.

### Usage

```python
from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor

trace = RoutingTrace(
    model_id="checkpoint-revision-or-content-id",
    quantization="exact-mixed-format-manifest-id",
    expert_mapping="pruning-map-content-id",
    workload="python-coding",
    mode="detailed",
    max_records=4096,
)
executor = HeterogeneousExpertExecutor(
    host_provider, model_id=trace.identity["model_id"],
    d_model=2048, d_ff=512,
    device_budget_bytes=0,  # choose a safe explicit budget for GPU use
    cpu_workers=2, cpu_packed_backend="native",
    routing_profiler=trace,
)
# Attach executor to the existing ExpertsInference layer(s), execute workload.
executor.close()  # drain workers; GPU caller-stream completion is separate
# For a complete GPU report, synchronize outside the timed region first.
trace.write_json("routing.json")
```

Identity includes model, quantization, mapping and workload. The recorder does
not hash a 58-GB file; callers supply stable identities. It is not a profile
loader and cannot seed residency yet. Do not attach one recorder concurrently
to separate model dispatch threads; parallel CPU workers within one executor
are supported. For independent conversations, use distinct traces.

## 4. Offline cache analysis

```sh
MINI_LLM_BACKEND=numpy .venv/bin/python -m mini_llm.diagnostics.routing_analysis \
  --trace routing.json --output routing-analysis.json \
  --capacities 0 16 32 64 128 256 --windows 128 512 2048 8192
```

Replay compares existing `AdaptiveExpertCache`, LRU, LFU, recency/frequency,
static hot experts and offline optimal Belady **with CPU-bypass allowed**.
The latter two are labeled in-sample/offline oracles, not production policies.
All curves start cold at the retained boundary. Expert keys include layer.
Dispatch order and sorted distinct experts within each dispatch match the
runtime grouped-request convention.

Per-layer nonoverlapping token windows count selection frequencies and the
number of static slots covering 50/75/90/95/99% of those selections. Partial
windows are identified. These are **static coverage estimates, not LRU hit-rate
capacity requirements**. Replay slots assume equal expert bytes and omit busy
set protection, transfer costs, state and allocator reserve. Batched prefill
coalesces requests; a flattened-token trace is not a sequential decode trace.
Summary-only and malformed/mixed-identity traces are rejected.

## 5. Validation and measured results

Machine-readable summary: `docs/0106i_validation.json`.

### Passing verification

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106*.py tests/test_0085b_external_llama.py \
  tests/test_0085c_external_safetensors.py tests/test_0085f_external_gguf.py \
  tests/test_0085g_gguf_tokenizer.py tests/test_moe_experts.py \
  tests/test_moe_router.py tests/test_topk.py
# 205 passed, 33 skipped

MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106i_routing_trace.py \
  tests/test_0106d_async_experts.py::test_cuda_copy_worker_and_mixed_cpu_gpu_experts
# 15 passed
```

The first broad CuPy invocation also ran legacy CPU-oriented tests: four
failed because NumPy comparison attempted implicit conversion of CuPy arrays.
Those tests were not rewritten; their NumPy execution passes. The same initial
run exposed the event API mismatch, fixed once locally and retested above.
The entire legacy suite is **not** claimed to pass under CuPy.

Instrumentation tests show bit-for-bit equality to disabled profiling on
synchronous and worker-pool paths. The small mixed GPU/CPU test uses FP32 with
`rtol=1e-4, atol=1e-7`. Offline analysis has 57 focused passing cases, including
hand-derived cache behavior, layer identity and malformed inputs.

### Successful CPU microbenchmark

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106i_routing_trace.py \
  --output benchmarks/results/0106i_cpu.json
```

Q4_0, D=F=64, 8 experts, top-2, 128 single-token layer calls, 8 warmup calls,
2 CPU workers, native packed CPU kernels, GPU budget zero, fixed random seeds.

| Instrumentation | Mean ms/layer-token | Single-layer tokens/s | Max absolute error vs disabled |
|---|---:|---:|---:|
| Disabled | 0.083087 | 12035.58 | 0 |
| Summary | 0.106770 | 9365.89 | 0 |
| Detailed | 0.106183 | 9417.68 | 0 |

These are **not full-model generation rates**. Fixed execution order and one
short run are unsuitable for statistical speedup claims. Summary adds about
24 microseconds on this tiny CPU fixture; detailed was not significantly
faster. Use disabled profiling for final throughput measurements.

Cold offline replay of the measured synthetic phase-changing trace:

| Slots | Existing adaptive hit rate | LRU | Offline Belady with bypass |
|---:|---:|---:|---:|
| 0 | 0% | 0% | 0% |
| 2 | 23.83% | 80.86% | 87.89% |
| 4 | 47.27% | 91.41% | 94.14% |
| 8 | 90.62% | 96.88% | 96.88% |

This supports measuring adaptation on real traces, **not** replacing the
production policy based on a synthetic locality pattern. Static selection
coverage in its 128-token window was 4/6/8/8/8 slots for 50/75/90/95/99%.

### Failed larger packed-GPU benchmark — unresolved

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106i_routing_trace.py --tokens 128 \
  --d-model 256 --d-ff 256 --n-experts 16 --top-k 4 \
  --device-budget-mib 1 --output benchmarks/results/0106i_gpu.json
```

RTX 5080, FP32 activations, Q4_0 packed experts, native CPU workers, bounded
1-MiB GPU cache. No training/inference compute process was running at the GPU
availability check. No existing workload was stopped.

- Disabled measurement: 0.7628 ms/layer-token (1310.94 single-layer tokens/s).
  **Unverified performance candidate only:** parity subsequently failed.
- Disabled versus summary: 1024/32768 output elements differed (3.12%).
- Maximum absolute error **0.0038457**, maximum reported relative error **562.8196**.
- Example at `[68, 0, 0]`: summary `0.001226971042342484`,
  disabled `0.0030300903599709272`.
- Dtype FP32; exact equality was requested; no finite differences apply.
- The benchmark aborted before detailed mode and before writing its GPU JSON.
  There is no successful three-mode GPU result artifact or claimed speedup.

One targeted experiment removed the cache, eviction and CPU worker staging:
all 16 individual Q4_0 experts, D=F=256, input seed 10609, weight seed
`106090 + 3*expert + projection`, input scale .02, FP32. Maximum errors against
`PackedGGUFWeight.dequantize_reference()` plus NumPy SwiGLU were:

- Native CPU: **1.4551915228366852e-9**.
- Packed GPU: **1.0477378964424133e-9**.

Thus isolated kernels are established to agree on this fixture. The failing
combined managed/asynchronous path is not localized to a specific operation.
Cache/stream ownership and worker staging remain plausible; the available
evidence does not distinguish them. No kernel rewrite or relaxed tolerance
was justified. Debugging stopped rather than adding a speculative correction.

**Single next experiment:** replay the same 128 inputs/routes with a fixed
resident set (no eviction/admission after warmup), comparing synchronous versus
two-worker execution against the CPU oracle. This separates worker staging
from dynamic residency without changing expert math.

## 6. Remaining work and ordering

1. Resolve the combined packed-GPU parity gate above before performance tuning.
2. Add actual-file-required Q6_K and GGUF BF16 coverage; preserve packed weights.
3. Implement the smallest reference-correct Qwen recurrent mixer/state adapter
   and hybrid layer wiring, then incremental/full-sequence parity. Do not rename
   a Llama adapter to Qwen or bypass geometry validation.
4. Obtain the exact Qwen3.8 target's GGUF inventory, pruning map and model docs
   before IQ/n-gram work. Synthetic/small-model work need not wait for download.
5. Only then calibrate explicit CPU-only/mandatory-promotion/hybrid/adaptive
   cost modes, profile-seeded placement, static/warm cache splits, bounded
   pinned staging and asynchronous promotion. Those are not implemented here.
6. Add async RAM/SSD expert retrieval and a separately validated sparse n-gram
   row interface after observed access patterns justify it. No full SSD cache,
   speculative decoding or Strata performance comparison was attempted.

Keep real Python, C++, numerical, reasoning and conversation profiles separate.
The current synthetic benchmark does not establish their routing locality,
large-window behavior, 2–8-GiB cache budgets, peak memory or full-model latency.
