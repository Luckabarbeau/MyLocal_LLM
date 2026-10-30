# PATCH 0103A — Model B numerical and execution regression baseline

## Scope

This patch adds regression *gates*, **not** new training/inference kernels.
The frozen production architecture is `model_b_348m_sparse16`, namely 20 layers,
1024 hidden, 4096 FFN, 348,261,376 parameters, four summaries per completed
128-token block, a 512-token local window, a 4096-token long-local head, and a
stride-16 historical sparse head on layers 2–17 (zero-based indices).
The sparse head is **part of the baseline**, not an optional enhancement.

0103-0's public `Model` API, checkpoint representation, optimizer state and
existing runtime behavior are unchanged here. A future backend may implement
PyTorch, but the current CPU regression suite executes the NumPy reference.

## Local CPU regression

From the repository root (safe while a GPU training job is running):

```bash
python scripts/run_0103a_regressions.py --mode cpu --report /tmp/model_b_cpu_regression.json
```

The runner launches **five independent pytest processes**: public model API,
matrix-pooling forward/backward, stride-16 historical and incremental parity,
checkpoint compatibility, and runtime-profile behavior. A selected set of
older CUDA tests embedded in those files will be **skipped on NumPy**. Results
are marked failed if any subprocess exits nonzero. The runner removes inherited
`MINI_LLM_*` experimental flags and pins `MINI_LLM_BACKEND=numpy` to make this
an independently reproducible reference gate. It does not execute the 348M
forward on CPU; it asserts the production parameter count analytically and
runs small models with the same numerical operators.

The runtime-profile tests themselves also restore `MINI_LLM_*` variables after
each test. `apply_runtime_defaults()` writes to `os.environ` directly and these
changes were previously able to contaminate later test files in a combined
pytest invocation. Both fixture-level restoration and process isolation are
intentional safeguards.

To preview the exact selected tests:

```bash
python scripts/run_0103a_regressions.py --mode cpu --list
```

## GPU correctness regression (run only when the GPU is free)

```bash
python scripts/run_0103a_regressions.py --mode gpu --report /tmp/model_b_gpu_regression.json
```

This requires CuPy, Triton, an accessible CUDA device, and sufficient available
VRAM. The preflight fails explicitly rather than treating unavailable GPU tests
as successful skips. GPU suites check stride-16 Triton parity, matrix-pooling
Triton parity, and native Model B CUDA Graph capture/reuse against the existing
fallback. **Do not run alongside the ongoing 348M training job.**

`--mode all` runs the CPU gates followed by the GPU gates. No GPU results are
claimed unless executed successfully on hardware.

## What a PASS means—and what it does not mean

- CPU: tested reference math, finite-difference gradients, causality/document
  isolation, current-block exclusion, small-model inference parity, native
  checkpoint compatibility, and the immutable production configuration.
- GPU: if run and passed, tested selected GPU operator equivalence and graph
  activation/correctness under the tested conditions.
- **Not measured:** production 348M throughput, prefill tokens/sec, steady-state
  decode tokens/sec, peak memory, long-context performance, or the identity of
  every actual accelerated kernel chosen in serving. Those remain separate
  0103D/0103G work. An enabled flag is **not** proof of a captured CUDA Graph.

Machine-readable reports have schema `0103A-regression-v1`. They record tiers,
selected suites, exit status and execution time, not synthetic performance
numbers. Existing `ExecutionReport.require_fast()` continues rejecting
unverified execution or unexpected fallbacks.

## Acceptance conditions

1. Production specification remains exactly 348,261,376 parameters including
   the stride-16 sparse head, without allocating full weights on CPU.
2. Reference numerical checks pass after runtime-profile tests, not just alone.
3. Training-vs-incremental decoding matches across completed-block boundaries.
4. Native checkpoint round-trips and public model wrapping remain compatible.
5. CPU testing runs in an isolated environment independent of GPU flags.
6. GPU gates cannot silently pass when the GPU or required toolchain is absent.
7. The underlying mathematical implementation, training and inference
   pipelines, and checkpoint schema remain untouched by this patch.

## CI

`.github/workflows/0103a-cpu-regressions.yml` runs the isolated CPU gates on
pull requests and pushes, after installing the declared development extras.
The GitHub workflow is provided but has not been executed on GitHub in this
patch environment. GPU tests remain opt-in and local.
