# 0106B — Native packed GGUF decoding on the shared backend

Depends on: **0106A** (`PATCH_0106A_SHARED_QUANTIZED_MOE_FOUNDATION.patch`).

## Implemented

- `mini_llm.ops.quantized_linear.quantized_linear(x, weight)` implements
  `x[..., K] @ weight[N, K].T` for Q8_0, Q4_0, Q4_K and Q3_K, retaining the
  GGML quantization block layout in memory. On **CuPy + Triton** it reads the
  packed bytes directly and reconstructs only local lane values in registers.
  **It never materializes a dense quantized matrix on the GPU.**
- `packed_to_device(PackedGGUFWeight)` copies packed bytes for one 2D matrix
  or one selected expert; it does not read/copy the full MoE expert pool.
- `HeterogeneousExpertExecutor` now accepts either its previous dense matrices
  in native `(D,F)/(D,F)/(F,D)` orientation or **all-packed** GGUF matrices in
  `(F,D)/(F,D)/(D,F)` orientation, and uses native packed projection kernels
  for GPU-resident experts. Its first-stage host miss is a small, synchronous
  CPU reference computation. The residency cache budgets retained **packed**
  GPU bytes, not expanded FP16/FP32 bytes.
- Existing training and dense/resident inference paths are unchanged.

## Contracts and limitations

- The GPU packed GEMV kernel is deliberately optimized for an individual
  token or small token batch: one program per `(token, output row)`. Large
  prefill requires a separate tiled, fused quantized GEMM.
- Input activations must be contiguous `[M,K]` or `[B,T,K]`, float16,
  bfloat16 or float32 on the GPU. The output retains input dtype and is
  accumulated in FP32 internally. The reference NumPy path uses FP32 math.
- `K` must be divisible by 32 (Q8_0/Q4_0) or 256 (Q3_K/Q4_K).
- Device transfers are **synchronous** on residency promotions. The CPU
  fallback still dequantizes one expert's weights into temporary FP32 arrays.
  Neither is the final performance path.
- No end-to-end Qwen3-Coder-Next model support yet: this still needs its
  hybrid attention/Gated DeltaNet adapter, GGUF naming, and a tokenizer.
- This change does **not** imply that every GGUF quantization variant (e.g.
  IQ formats, Q2_K) is supported, or that quantized GEMV will beat llama.cpp.
  Real GPU throughput and kernel parity must be measured on the target card.
- The backend currently requires Triton for the packed GPU operator; native
  NumPy/CuPy dense model execution is unaffected without it.

## Validation

CPU and format-addressing regressions:

```bash
.venv/bin/python -m pytest -q \
  tests/test_0106b_quantized_linear.py \
  tests/test_0106a_packed_gguf.py \
  tests/test_0106a_expert_residency.py
```

On the RTX 5080, GPU tests must be run with the actual CuPy backend:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q \
  tests/test_0106b_quantized_linear.py -k triton_kernel
MINI_LLM_BACKEND=cupy .venv/bin/python \
  benchmarks/benchmark_0106b_quantized_linear.py \
  --format Q4_K --k 2048 --n 512 --iterations 100
MINI_LLM_BACKEND=cupy .venv/bin/python \
  benchmarks/benchmark_0106b_quantized_linear.py \
  --format Q3_K --k 2048 --n 512 --iterations 100
```

Benchmark measurements report **operator latency** (microseconds/call), not
end-to-end generated tokens per second. CPU tests alone **cannot certify**
that Triton compiles and executes successfully on Blackwell.

## Proposed next steps

1. Verify CUDA parity for Q3_K/Q4_K at representative widths and produce
   side-by-side kernel benchmarks versus local llama.cpp on the same GPU.
2. Fuse gate/up dequantization and SiLU into one GPU kernel; group routed
   experts to reduce the Python dispatch and CUDA-launch overhead.
3. Replace the temporary CPU dequantization fallback with fused, packed
   quantized CPU GEMV. Overlap nonresident CPU experts and GPU experts.
4. Add attention/Gated DeltaNet model-specific adapters while keeping the
   shared quantized operators and residency manager architecture-neutral.
5. Provide an inference-only per-layer benchmark, and later end-to-end
   Qwen3-Coder-Next model validation, before claiming any speedup.

Reference quant block layout: GGML `ggml-common.h`, Q3_K=110 bytes/256
weights and Q4_K=144 bytes/256 weights.
