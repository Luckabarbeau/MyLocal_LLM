# Patch 0102K — Triton pooling pointer launch correction

Patch 0102J introduced an opt-in Triton learned-pooling kernel, but its
Python wrapper passed raw CuPy ndarrays to the Triton launcher. On the
supported Triton runtime this raises `TypeError: failed to specialize
argument of type: ndarray` before either kernel executes.

The forward/backward pooling launchers now use the same typed `data_ptr()`
adapter as existing working Triton dual-resolution attention kernels. This
is a launch-protocol fix only: the pooling mathematics, output dtypes,
optional flag, and default projection optimization are unchanged.

## Verification

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q tests/test_dual_resolution_pooling_pointer_0102k.py
MINI_LLM_BACKEND=cupy python -m pytest -v tests/test_dual_resolution_fused_pooling_0102j.py
MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_pooling_0102j.py \
    --blocks 1,16,64,256 --warmup 3 --repeats 20
```

GPU execution is required to verify the fused kernel's numerical parity
and speed. Fused pooling remains opt-in:
`MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=1`.
