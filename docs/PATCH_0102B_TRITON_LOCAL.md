# Patch 0102B — Optional Triton local attention

This patch adds an **experimental, opt-in** Triton local-attention forward and backward implementation for `dual-resolution-1024-20l`. It is adapted from `mini_llm/ops/triton_mixed_attention.py` (the pre-existing 0088A kernels) without changing the legacy kernel or other model variants.

## Scope

- Replaces **only the 512-token causal local attention** part of each dual-resolution layer.
- Processes all **16 query heads** and **4 shared KV heads**, with head dimension **64** and BF16 Q/K/V.
- Uses online softmax; retains a compact FP32 output and row log-sum-exp for backward rather than Q×K probability matrices.
- Supports **document IDs per token** directly in all three attention-gradient kernels (forward, dQ, and shared dK/dV); does not allow cross-document attention.
- Keeps the full block compressor, learned attention pooling, global summary attention, and learned gate on the original CuPy path.
- Does not change the model's parameters, checkpoints, inference path, or causal block-summary rules.
- The existing CuPy local kernel remains the default. Triton is **strictly opt-in**; unsupported head geometry, input dtype, window, or missing Triton raises an error rather than silently reporting a CuPy benchmark as Triton.

## Usage

Run correctness tests first on the RTX 5080:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q tests/test_triton_dual_resolution_local_0102b.py
```

Compare with the existing benchmark. **Each benchmark times one attention operator, not the 20-layer model.** Both commands use the same architecture, random seed, and benchmark geometry. Outputs are separate JSON reports.

```bash
MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_0102.py \
  --contexts 4096,16384,65536 --global-query-chunk 512 \
  --compression-block-chunk 64 --local-kernel cupy \
  --skip-inference --warmup 2 --repeats 5 \
  --output dual_local_cupy_0102b.json

MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_0102.py \
  --contexts 4096,16384,65536 --global-query-chunk 512 \
  --compression-block-chunk 64 --local-kernel triton \
  --skip-inference --warmup 2 --repeats 5 \
  --output dual_local_triton_0102b.json
```

Production training **remains on CuPy unless explicitly selected** using:

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL=triton python train_model.py --model dual-resolution-1024-20l ...
```

The environment flag is read during attention-operator construction. Do not enable experimental Triton on an important long training run before the full CUDA parity tests, end-to-end model validation, and memory/throughput measurements pass.

## Current limitations

1. Only BF16, 16Q/4KV, D64, window512 is supported; this is intentional until GPU verification. Other geometry raises `ValueError`.
2. Global-summary attention and block compression remain unfused CuPy operations; results cannot be attributed to a complete Triton dual-resolution backend.
3. Training/inference parity is not revalidated on the GPU here; native inference is untouched.
4. This environment has neither CUDA/CuPy nor Triton, so Triton kernel compilation, numerical parity, production stability, memory use, and speed **must be confirmed on the RTX 5080**.
5. The older 0088A mixed-attention code is used as a reference but remains unchanged. The experimental file keeps independent kernels to avoid changing full-head behavior in older model variants.

## Causality

The local branch uses a 512-token causal window and equal document IDs. This patch does **not** alter the global mechanism, which can only attend to summaries of fully completed preceding 128-token blocks. The new local kernels never read outside the sequence or across EOS-defined document boundaries.

## What to report

Please send the GPU test output (including any Triton compilation errors), benchmark JSON summaries, and peak GPU memory. If a test fails, do not continue into a long training run.
