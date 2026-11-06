# 0106C — Fused GGUF routed-expert decode (inference only)

This patch follows **0106A** (packed GGUF storage and conservative expert cache)
and **0106B** (Triton packed GEMV). It extends the same
`mini_llm.ops.quantized_linear` and `HeterogeneousExpertExecutor` used by the
existing runtime, without creating an additional backend or modifying training.

## Operation

For a GGUF expert with row-major packed matrices `gate[F,D]`, `up[F,D]`,
`down[D,F]`, compute:

```text
h = SiLU(x @ gate.T) * (x @ up.T)
y = h @ down.T
```

`quantized_swiglu` uses a **single Triton kernel** for the gate GEMV, up GEMV,
and activation. It reads the input activations once per row, accumulates both
projections in FP32, writes only one `[tokens,F]` activation buffer, and leaves
all GGUF weights packed in VRAM. The down projection uses the existing packed
GEMV operator. It supports mixed GGUF types for gate and up if both dimensions
are block-aligned. Formats implemented here remain Q8_0, Q4_0, Q4_K, Q3_K.

The shared GGUF block decoder is now a Triton helper reused by both paths so
future bitpacking bug fixes affect fused and unfused execution consistently.

No new model-specific execution hierarchy was introduced. Once the existing
`HeterogeneousExpertExecutor` promotes a packed expert to GPU, it automatically
uses the fused implementation through `PackedExpertFFNInference`.

## Testing

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q \
  tests/test_0106b_quantized_linear.py \
  tests/test_0106c_fused_packed_expert.py \
  tests/test_0106a_expert_residency.py
```

This environment lacked CUDA/CuPy/Triton; passing CPU tests cannot certify
CUDA kernel compilation or numerical parity. CUDA tests compare Q3_K/Q4_K and
mixed-format gate/up against the CPU dequantization oracle. A baseline can be
selected explicitly with `PackedExpertFFNInference(..., fuse_gate_up=False)`.

GPU operator benchmark:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python \
  benchmarks/benchmark_0106c_packed_expert.py \
  --format Q4_K --d-model 2048 --d-ff 512 --tokens 1 --iterations 100
```

## Deliberate limits

- No fused down projection or fused multi-expert routing: the down kernel
  follows the gate/up kernel because it consumes the completed feature vector.
- No CPU quantized fast path: CPU fallback still dequantizes **one selected
  expert per miss** as a correctness reference. Native quantized CPU compute,
  bounded host staging, and asynchronous GPU/CPU overlap remain work to do.
- No device-memory prefetch, cache-slot address stability or CUDA graph capture
  for cache misses yet.
- This is optimized for GEMV / small decode batches, not prompt-prefill GEMM.
- End-to-end Qwen3-Coder-Next support additionally requires its architecture,
  correct GGUF tensor mapping, and DeltaNet/KV-state implementations.
- Fused GPU numerical differences may occur for FP16/BF16 because gate/up
  accumulate and apply SwiGLU in FP32 before a single final cast, whereas
  the old path rounded each intermediate projection to the activation dtype.

Do not infer tokens/sec improvements before the full decode pipeline is
benchmarked on the target RTX 5080.
