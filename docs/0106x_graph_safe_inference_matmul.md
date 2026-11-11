# 0106X — Reusable FP32 inference matmul and recurrent graph validation

## Verified status

The 0106W recurrent CUDA-graph blocker is resolved for the tested paths. The full
tiny hybrid-model graph test now passes, including reset, forked generation
state, and new requests. The historical failure in `0106s_v_decode_optimization.md`
is intentionally retained; this report records its resolution.

Real Qwen3-Coder-Next warm decode reached **21.974 tokens/s**, not 30 tokens/s.
The graph path remains opt-in (`MINI_LLM_RECURRENT_GRAPH=1`), serial, and bound to
one replay stream. Training and native inference policies were not changed.
Base revision: `cbd41e463154f924e3d7441224d5d496baaf1d9b`; changes are incremental
working-tree edits, without a commit or branch.

## Existing inference optimization audit

The implementation follows existing backend patterns rather than replacing them:

- `external/cuda_graph_decode.py` already documents that CuPy high-level matmul
  rejects stream capture. Its specialized decoder owns intermediates and uses
  direct cuBLAS calls outside CuPy's high-level wrapper.
- Native graph decoders and `ops/cublas_fp16_gemm.py` provide specialized
  low-precision paths. They are not an FP32 replacement for this model's current
  activations/state. Existing dense/QKV fused kernels have dtype/geometry
  restrictions; changing model precision to force eligibility was avoided.
- The new kernel reuses the existing `quantized_linear` Triton imports and
  zero-copy `_TritonCuPyPointer` adapter, with launches on the current CuPy stream.
- Existing packed expert/SwiGLU kernels, routing, residency, and native graph
  implementations remain in place. This change does not unpack expert matrices.

## General library primitive

`mini_llm/ops/inference_matmul.py` exposes:

```python
from mini_llm.ops.inference_matmul import matmul_inference, supports_inference_gemv

result = matmul_inference(a, b)             # NumPy-style a @ b
matmul_inference(a, b, out=output_buffer)   # Optional caller-owned output
```

Single-row CuPy FP32 products use a Triton GEMV with FP32 multiplication and
reduction, without TF32/FP16 approximation. Batch broadcasting, transposed,
reversed and sliced operand strides are supported. FP64, CPU, general multi-row
matrices and other unsupported cases retain backend matmul. Aliased or
noncontiguous outputs also retain backend behavior. The function does not read
GPU values on the host or synchronize the device.

The exact shape/stride specialization must be warmed before capture, as with the
existing Triton operators. Results can differ from cuBLAS in FP32 reduction order;
bitwise equivalence to cuBLAS is not claimed. The primitive has no model-specific
names, state, routing, or quantization assumptions.

### Integration

- `external/gguf_projection.py` uses the primitive for dense CUDA projections;
  NumPy/operator-injected inputs retain array-native dispatch.
- `ops/gated_delta_inference.py` uses it for the two portable recurrent products:
  `prediction = k @ decayed_memory` and `readout = scaled_q @ updated_memory`.
  The decay/correction/rank-one update order is unchanged.
- The W wrapper can therefore capture the portable recurrent graph without
  CuPy cuBLAS calls. It deliberately does not capture the fused recurrence's
  global pending-event bookkeeping.
- `benchmarks/benchmark_0106r_generation.py` now reports each recurrent graph's
  private pool, their sum, and the combined main/graph private-pool reservation.

Private input/state buffers isolate capture warmup from caller state. Replay
copies caller state in/out and returns fresh output ownership. Cross-stream reuse
is rejected rather than permitting concurrent scratch overwrite. This is not a
whole-model graph or a concurrent-serving implementation.

## Failure history and localized corrections

1. The first direct compiled Triton launch used a two-dimensional grid and raised
   `IndexError: tuple index out of range`. Direct compiled launches require three
   grid dimensions. The localized correction was `(N, batch_count, 1)`.
   The X/W CUDA retest passed: **28 passed, 1 skipped**.
2. A broader CUDA suite found nine injected-NumPy projection tests failing with
   `TypeError: Argument 'a' has incorrect type (expected cupy..., got numpy.ndarray)`.
   Those tests intentionally substitute NumPy operators; the new helper retained
   the actual CuPy backend. The localized projection guard now requires a CUDA
   array before choosing `matmul_inference`; the original `hidden @ weight.T`
   fallback is preserved. The exact broader suite then passed:
   **297 passed, 11 skipped**.

No kernel equations or test tolerances were loosened for either correction.

## Validation

Final broad CPU regression: **1337 passed, 159 skipped in 9.06 s**.
Final focused/broad CUDA integration: **297 passed, 11 skipped in 7.66 s**.
Exact commands are recorded in `0106x_validation.json`.

Coverage includes matmul shape/broadcast/stride behavior, independent FP64 oracle,
input immutability, caller output/alias fallback, general matrix fallback,
nondefault stream graph replay, full tiny-model graph state lifecycle, packed
expert execution, routing, tracing, async ownership, and benchmark integration.
This is not a fresh full-repository CUDA suite or an optional llama.cpp reference run.

A bounded synthetic full-model comparison also passed on a nondefault stream:

| Comparison | Maximum absolute logit error |
|---|---:|
| CUDA prefill vs CPU | 7.153e-7 |
| CUDA incremental decode vs CPU, largest of three steps | 2.086e-6 |
| CUDA full forward vs CPU | 1.788e-6 |
| CUDA full forward vs independent FP64 | 1.644e-6 |

Existing tolerance: `rtol=4e-4`, `atol=8e-6`. Matched greedy IDs were
`[48, 28, 55, 9]`. These are tiny synthetic fixture results, not strict real-GGUF
logit parity against llama.cpp. W's separate full-model test establishes graph
replay parity; the FP64 comparison uses the synthetic P benchmark.

Reproduce from repository root (choose a new output filename):

```sh
MINI_LLM_RECURRENT_GRAPH=1 MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B benchmarks/benchmark_0106p_qwen_model.py \
  --max-new-tokens 4 --context 32 --expert-cache-mib 16 \
  --device-budget-mib 512 --device-reserve-mib 2048 \
  --output benchmarks/results/NEW_SYNTHETIC_PARITY_FILENAME.json
```

## Real-model performance

Installed model:
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`.

RTX 5080; FP32 activations; packed quantized weights. Prompt
`def add(a, b):\n    return`, eight input tokens, sixteen generated tokens, context
64, two serial requests, greedy raw completion, no speculative decoding.
Decode throughput counts fifteen actual decode forward calls, excluding prefill.

| Measurement | X ordinary | X recurrent graphs |
|---|---:|---:|
| First-use decode tokens/s | 6.787 | 7.342 |
| Warm decode tokens/s | 19.385 | **21.974** |
| Warm end-to-end completion tokens/s | 13.502 | 14.501 |
| Warm first sampled token ms | 399.627 | 408.747 |
| Main private-pool reserved bytes | 8,925,934,080 | 8,923,836,928 |
| Graph private-pool reserved bytes | Not created | 164,966,400 |
| Combined private-pool reserved bytes | 8,925,934,080 | 9,088,803,328 |
| Peak process RSS bytes | 11,471,704,064 | 11,448,328,192 |

Graph mode improves this warm ordinary-X decode measurement by about 13.4%.
It improves the preceding V measurement (20.193 tokens/s) by about 8.8%.
Ordinary X alone is slower than V in these single runs: do not describe this as
an unconditional GEMV throughput improvement. No run-to-run confidence interval
has been established.

All four X requests retain the established token sequence:
`[264, 488, 293, 271, 750, 32256, 2877, 11, 293, 982, 262, 470, 264, 481, 293, 271]`.
Both warm decode phases have 7200/7200 expert hits, no CPU expert assignments,
no uploads, no evictions and zero expert-weight transfer bytes. This short
repeated prompt is favorable and does not establish new-workload cache behavior.

There are 36 recurrent graphs. Their pools are additional to the main private
pool and are not a measurement of global/process VRAM. The main pool ceiling
remains 12 GiB; each graph has its own 32 MiB ceiling. The combined reservation
is about 8.46 GiB, but no aggregate hard graph-pool budget is enforced by the
benchmark. Account for this before expanding model/shape coverage.

Preserved guards: 8 GiB expert-cache budget, 2 GiB device reserve, 16 GiB host
available reserve, 12 GiB observational RSS guard, two CPU workers. Guards are
observational, not hard OS memory limits. No cache clearing, model hashing,
reference-model overlap or existing workload interruption was performed.

Reproduce graph mode, choosing a new filename:

```sh
MINI_LLM_RECURRENT_GRAPH=1 MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B benchmarks/benchmark_0106r_generation.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --max-new-tokens 16 --requests 2 --context 64 --cpu-workers 2 \
  --expert-cache-mib 8192 --device-budget-mib 12288 --device-reserve-mib 2048 \
  --ram-reserve-mib 16384 --max-rss-mib 12288 \
  --output benchmarks/results/NEW_GRAPH_FILENAME.json
```

Set `MINI_LLM_RECURRENT_GRAPH=0` for the ordinary comparison. The benchmark keeps
its 0106R schema label; filenames and this report identify the implementation.
Raw local reports are under ignored `benchmarks/results/`; the tracked validation
summary retains the essential results.

## Next measured bottleneck / remaining work

Warm inclusive host attribution, ms per token:

| Stage | Ordinary | Graph |
|---|---:|---:|
| Mixer | 17.411 | 9.662 |
| Router | 2.087 | 2.059 |
| Experts | 20.866 | 22.588 |
| Entire MoE | 30.382 | 32.178 |

These nested host intervals are not additive and are not isolated GPU kernel
measurements. The graph path averages 45.507 ms/decode call; 30 tokens/s needs
less than 33.333 ms. The next useful experiment is a warmed grouped-expert
microbenchmark separating host dispatch overhead from CUDA-event compute time,
using the existing packed kernels at actual Qwen shapes before changing them.
Expert dispatch/execution is now the largest attributed stage; mixer/attention
work still matters. Do not infer pure GPU compute costs from the table.

Longer/different prompts, cache misses and hybrid CPU/GPU scheduling performance,
chat templates, long-state accuracy, fair matched Strata/llama.cpp comparisons,
and aggregate graph-memory admission remain unverified here. Qwen3.8 architecture,
pruning/ngrams and SSD tiers remain deferred. No strict real-model reference
parity or 30-token/s result is claimed.
