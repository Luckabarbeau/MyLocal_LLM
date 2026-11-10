# 0106L — Packed Q6_K and lazy GGUF BF16 access

Date: 2026-11-09. Continues the existing NumPy/CuPy/Triton external-model
backend. No training, optimizer, router policy, native model architecture, or
serving behavior was changed. Prior 0106I–K work is preserved.

**Result:** the installed Qwen3-Coder-Next GGUF's previously unsupported routed
Q6_K tensor can now execute as a packed expert on CPU and GPU. Actual-file
BF16 non-expert loading also passes bit-exact expansion. **This is weight-format
compatibility, not a complete Qwen decoder or text-generation implementation.**

## 1. Q6_K format and execution

The independent layout reference is the installed `gguf` **0.19.0** package's
`gguf.quants.Q6_K` decoder and `(256, 210)` GGML format size. No source was
vendored and no upstream binary or package was downloaded for this increment.

A block contains:

| Offset | Bytes | Content |
|---:|---:|---|
| 0 | 128 | Low four-bit payload |
| 128 | 64 | High two-bit planes |
| 192 | 16 | Signed int8 sub-scales, one per 16 values |
| 208 | 2 | Little-endian FP16 global scale |

For lane i in [0,255], half h=floor(i/128):

```text
lo = (ql[h*64 + i%64] >> (4*floor((i%128)/64))) & 15
hi = (qh[h*32 + i%32] >> (2*floor((i%128)/32))) & 3
q  = (lo | (hi << 4)) - 32
w  = (float32(d) * float32(signed_scale[floor(i/16)])) * float32(q)
```

Implementation extends the existing paths:

- `ops/quantized_weight.py`: block validation, byte accounting, mmap slicing,
  and the small FP32 reference decoder.
- `ops/native_packed_cpu.py` / `.cpp`: native format ID 5, direct packed GEMV,
  and complete mixed-format SwiGLU experts. The existing source fingerprint
  automatically rebuilds the local CPU library; no ABI change is required.
- `ops/quantized_linear.py`: shared Triton register-local lane decoder, reused
  by GEMV, fused gate/up, and down projection. The 0106J explicit CuPy-stream
  launch path remains in use.
- Existing Qwen packed extraction and residency management require no separate
  Q6_K branch: their format table and packed operator contracts are reused.

CPU/GPU execution does not expand the expert matrices. The reference decoder
is still only a correctness oracle/deliberate fallback. No SIMD optimization,
prefill-specific GEMM, IQ-format support, or approximate routing was added.

## 2. GGUF BF16 loading

New `external/gguf_tensor.py::decode_gguf_dense(tensor)` decodes one explicitly
requested F32, F16, or BF16 tensor in canonical row-major shape.

BF16 bytes are **bit patterns**, not numeric FP16 values:

```text
word = little_endian_uint16(payload)
fp32_bits = uint32(word) << 16
values = bit_view_float32(fp32_bits)
```

The helper validates exact payload size and shape, accepts the reader's uint8
payload plus endian-aware uint16 fixtures, and preserves zeros, subnormals,
infinities and NaN payload bits. F32/F16 retain their prior view and strict
shape-checking behavior. It performs no model-wide conversion.

- The existing Llama `GGUFWeightMap` accepts BF16 through this helper and retains
  architecture checks and HF Q/K row unpermutation. Llama loading is still its
  existing dense import path; BF16 source is expanded losslessly to FP32 before
  any configured runtime conversion.
- `Qwen3NextGGUFIndex.non_expert_weight(name)` explicitly decodes one requested
  tensor without caching. Supported packed non-expert tensors use the existing
  reference decoder; F32/F16/BF16 use the shared dense helper. Index construction
  remains lazy. Routed expert banks are rejected so this method cannot expand
  an entire expert bank; n-gram table requests also fail closed.
- This is not native BF16 expert arithmetic, automatic GPU backbone placement,
  or sparse n-gram lookup. A caller must deliberately request each tensor.

## 3. Tests

New `tests/test_0106l_q6k_packed.py` covers:

- Hand-packed lanes, negative/zero/extreme signed sub-scales and six-bit values.
- Exact reference parity with the installed independent GGUF decoder.
- Direct native identity-vector GEMV probing every lane.
- Read-only mmap expert slicing without copying and malformed-layout rejection.
- Matched-input FP64 GEMV parity for FP16/FP32, 2D/3D shapes.
- Native complete experts, mixed Q4_K/Q5_K/Q6_K, and managed CPU fallback.
- Fused/separate Qwen expert extraction without whole-bank dequantization.
- GPU tile sizes 256/512/1024, masked tail lanes, non-default streams,
  mixed fused projections, and cache hits without reference expansion.

New `tests/test_0106l_gguf_bf16.py` covers all **65,536 BF16 bit patterns**,
byte order/shape/error checks, lazy access, preserved expert contracts, tiny
Llama logits with BF16-loaded weights, and a real GGUFWriter/Reader fixture.

Seven existing synthetic fixture/benchmark factories now initialize Q6_K's
scale at **208:210**, not the unrelated low payload bytes 0:2. CLI choices and
most format parametrizations already follow the shared format table.

### One reference correction, no tolerance relaxation

Expanding the generic GEMV test's format list exposed another instance of the
FP16 input mismatch investigated in 0106K. At Q6_K index [0,2,2]:

- GPU FP16: **178.0**.
- Legacy unrounded-input FP32 reference: **176.91973876953125**.
- Absolute discrepancy: **1.08026123046875**, relative approximately .006106.
- Requested tolerance: `.2 + .003*abs(reference)` = **.7307592163085939**.
- Matched-input FP64 reference rounded to FP16: **178.0**, discrepancy **0**.

The test now uses actual storage-rounded inputs, FP64 accumulation, and the
appropriate output cast for each dtype. Its original tolerances remain
unchanged. Neither test failure nor correction changed a kernel equation.

### Passing commands

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106l_q6k_packed.py tests/test_0106l_gguf_bf16.py
# 57 passed, 13 skipped

MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106b_quantized_linear.py tests/test_0106c_fused_packed_expert.py \
  tests/test_0106g_q5k_packed.py tests/test_0106h_triton_cupy_pointer.py \
  -k 'gpu or triton'
# 52 passed, 33 deselected

MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106l_q6k_packed.py tests/test_0106i_routing_trace.py \
  tests/test_0106j_async_buffer_ownership.py \
  tests/test_0106d_async_experts.py::test_cuda_copy_worker_and_mixed_cpu_gpu_experts
# 48 passed

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106*.py tests/test_0085b_external_llama.py \
  tests/test_0085c_external_safetensors.py tests/test_0085f_external_gguf.py \
  tests/test_0085g_gguf_tokenizer.py tests/test_moe_experts.py \
  tests/test_moe_router.py tests/test_topk.py
# 298 passed, 50 skipped
```

The full new Q6_K selection also passed all 24 cases under CuPy. These are
selected unit/regression suites, not the entire repository's test suite.

## 4. Reproducible measurements

`benchmarks/benchmark_0106l_formats.py` supports synthetic Q6_K or mixed weights
and one explicitly selected expert from a local GGUF. It uses the 0106K
matched-storage FP64 arithmetic oracle. Native compilation, GPU uploads and
initial kernel compilation occur before measured iterations. Native CPU and
GPU weights remain packed. Cached dense BLAS and per-call reference
expansion are measured separately and consume additional RAM.

Synthetic command:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106l_formats.py --iterations 50 \
  --output benchmarks/results/0106l_synthetic_q6_gpu.json
```

Actual-file command:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106l_formats.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --layer 46 --expert 0 --iterations 20 \
  --output benchmarks/results/0106l_real_qwen_layer46_gpu.json
```

| Measurement | Synthetic Q6_K/Q6_K/Q6_K | Actual layer 46 expert 0 |
|---|---:|---:|
| Gate/up/down formats | Q6_K / Q6_K / Q6_K | Q5_K / Q5_K / Q6_K |
| D / F | 512 / 256 | 2048 / 512 |
| Packed expert bytes | 322,560 | 2,301,952 |
| Dense FP32 reference bytes | 1,572,864 | 12,582,912 |
| Native packed CPU mean | .473136 ms | 3.314376 ms |
| Cached dense FP32 BLAS mean | .020124 ms | .268094 ms |
| Dequantize-every-call CPU mean | 1.749899 ms | 11.802579 ms |
| GPU host wall mean | .048209 ms | .056076 ms |
| GPU stream interval mean | .048025 ms | .055629 ms |
| Native CPU max abs error vs oracle | 1.12057e-5 | 4.65661e-10 |
| GPU max abs error vs oracle | 1.90735e-6 | 5.82077e-11 |

FP32 input/output; one token; tolerance `rtol=2e-4, atol=2e-5`. Stream intervals
include enqueue gaps, not only active kernel time. Results are short warm
single-expert microbenchmarks on the RTX 5080, **not full-model decode rates**.
Uploads, expert misses, routing, attention, recurrent state, CPU activation
staging and layer accumulation are excluded. The scalar native CPU kernel is
not a tuned SIMD baseline. No model-wide speedup follows from this table.

A CPU-only synthetic mixed Q4_K/Q5_K/Q6_K run also passed:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106l_formats.py --mixed --iterations 30 \
  --output benchmarks/results/0106l_synthetic_mixed_cpu.json
```

It reported .360640 ms/native expert and max absolute error 8.94070e-7;
GPU measurements are explicitly marked skipped.

## 5. Actual installed model evidence

The existing GGUF remains memory-mapped; no full model was materialized or
hashed. Only one routed expert was expanded for its reference, and one small
BF16 tensor was decoded on explicit request. The benchmark caps expert packed
bytes at 32 MiB and BF16 sample source bytes at 32 MiB. No multi-GiB GPU cache
or automatic free-VRAM allocation was attempted. No training/inference process
was present in the initial GPU compute-process query; no process was stopped.

- Inventory remains `qwen3next`, 48 layers, D=2048, F=512, 512 routed experts.
- Formats remain BF16 5, F32 361, Q4_K 130, Q5_K 49, Q6_K 1, Q8_0 297.
- **`unsupported_routed_formats` is now empty** for this installed file.
- Layer 46 expert 0's Q5_K/Q5_K/Q6_K execution passes both CPU and GPU parity.
- Actual BF16 sample: `blk.46.ffn_down_shexp.weight`, [2048,512], 2,097,152
  source bytes -> 4,194,304 FP32 bytes, finite and bit-exact expansion.

Local raw JSON artifacts are under ignored `benchmarks/results/`. The concise
source-controlled summary is `docs/0106l_validation.json`.

## 6. Remaining blockers and next increment

Full-model decoding is still explicitly unavailable. We still need gated
DeltaNet/convolution/recurrent state, gated attention/KV geometry, shared and
routed layer wiring, placement of non-expert computation, Qwen tokenization,
and prefill/incremental/logit parity through existing generation interfaces.
Individual tensor format support is not that architecture integration.

The next correctness-first increment should implement and test the Qwen
recurrent mixer/state graph, starting with synthetic full-sequence versus
incremental execution and an independent reference. Do not resume cache
performance tuning ahead of that graph.

The requested 256-expert Qwen3.8 IQ1_M model is still not this installed
512-expert Qwen3-Coder-Next file. Its exact metadata, pruning/remapping and
required IQ formats remain unverified. No IQ or n-gram support is implied by
Q6_K/BF16 coverage. Strata's previously recorded revision remains reference
material; no new Strata source or release was fetched for 0106L.
