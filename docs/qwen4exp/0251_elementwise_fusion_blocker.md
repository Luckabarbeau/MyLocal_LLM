# 0251 — Elementwise fusion candidate: blocked validation

Status: **Implemented but unvalidated; not integrated into production dispatch**.

Candidate files: `mini_llm/ops/qwen4exp_elementwise_gpu.py`, `tests/test_0251_qwen4exp_elementwise_gpu.py`. Candidate functions fuse shared-expert SwiGLU, shared-output combination and Gated DeltaNet coefficient expressions. FP32 and explicit non-FMA boundaries retained. The candidate reuses installed CuPy's logaddexp routine via a private API; this dependency needs explicit compatibility validation. Existing model/MoE/mixer call sites do not use it.

Latest delegated execution on CuPy14.2.0 / RTX5080:

`MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q -x tests/test_0251_qwen4exp_elementwise_gpu.py`

**39 passed,1 failed**. Failure at test line179: `AttributeError: 'ElementwiseKernel' object has no attribute 'options'`. Kernel constructor options are supplied, but this CuPy release does not expose them as the test assumes. This is a test API defect, not observed numerical evidence against the kernels. All numerical comparisons reached passed bitwise checks, including signed zero, saturation, random/strided arrays, nondefault streams and1M-element arrays. Later tests did not run, so no complete validation claim.

The delegated task already used its allowed localized correction and retest. Under the repository's stop rule, no further corrective edit or integration was attempted. Next focused experiment: verify compiler-option forwarding by intercepting ElementwiseKernel construction instead of accessing an undocumented `.options` attribute, then rerun the same suite. No performance result and no production model arithmetic change for0251.
