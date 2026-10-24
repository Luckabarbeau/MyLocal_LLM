# PATCH 0089C — Training residual/RMSNorm boundary fusion

## Scope

This patch fuses the non-gated BF16 attention residual boundary inside a training Transformer block:

```text
residual1 FP32 + attention_output BF16
→ residual2 FP32
→ RMSNorm2 FP32 statistics
→ normalized BF16 branch
```

The fused CUDA kernel emits the exact cache consumed by the existing RMSNorm backward implementation:

```python
{
    "x": residual2_fp32,
    "inv_rms": inv_rms_fp32,
    "original_dtype": float32,
    "fused_bf16_residual": True,
}
```

No backward equation, parameter layout, checkpoint format, optimizer behavior, or model architecture changes.

## Activation

The low-level opt-in is:

```bash
MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM=1
```

After complete-block validation, the flag is enabled for CuPy training by the `gpu-fast` and derived `consumer-gpu` runtime profiles. Explicit environment overrides still win. Inference and residual-gated blocks retain the existing path.

## Correctness

Focused validation:

```text
MINI_LLM_BACKEND=numpy pytest ...: 26 passed, 1 skipped
MINI_LLM_BACKEND=cupy pytest tests/test_0089c_residual_rmsnorm.py: 2 passed
```

The CuPy test checks:

- FP32 residual output;
- BF16 normalized output;
- FP32 inverse RMS;
- FP32 residual gradient;
- FP32 gamma-gradient accumulation.

The B3/T2304 complete production block passed all 11 predeclared output/gradient parity checks against the compiled PyTorch fixture:

```text
relative L2 <= 0.02
cosine >= 0.999
```

Worst relative L2 in the profile-driven smoke was `0.00810221` (`grad_wv`).

## Performance

Controlled same-process alternating A/B measurements used the PATCH 0089B B3/T2304 fixture on an RTX 5080. The existing FFN accumulation experiment was explicitly disabled.

### Run 1

| Scope | Existing | Fused | Reduction |
|---|---:|---:|---:|
| Isolated boundary median | 0.130304 ms | 0.088560 ms | 32.04% |
| Complete block F+B median | 10.876048 ms | 10.829888 ms | 0.42% |
| Complete block F+B mean | 10.943554 ms | 10.836118 ms | 0.98% |

### Run 2

| Scope | Existing | Fused | Reduction |
|---|---:|---:|---:|
| Isolated boundary median | 0.128832 ms | 0.085328 ms | 33.77% |
| Complete block F+B median | 10.866960 ms | 10.772208 ms | 0.87% |
| Complete block F+B mean | 10.843500 ms | 10.760284 ms | 0.77% |

The end-to-end gain is intentionally described as modest: the fused boundary removes two elementwise materialization launches, while GEMMs and attention still dominate complete-block latency.
