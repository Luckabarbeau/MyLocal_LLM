# PATCH 0089D — Compiled-PyTorch LM-head and cross-entropy comparison

## Scope

0089D adds an isolated, backend-neutral benchmark for the production autoregressive LM-head and masked cross-entropy forward/backward graph. It adds only benchmark, test, and documentation files; no production model, loss, training, or runtime-default file is modified.

The exact benchmark geometry is:

```text
B=3
T=2304
D=1024
V=32768
N=B*T=6912
configured LM-head chunk capacity=8192
one effective tile=[0,6912)
```

The checkpoint directory contains configuration and training artifacts but no usable `model_params` payload for a standalone export. The fixture is therefore deterministic synthetic BF16 data at the exact production shape rather than checkpoint-derived tensors.

## Files

- `benchmarks/reference_0089d_torch_lm_head.py`
- `benchmarks/export_0089d_lm_head_fixture.py`
- `benchmarks/benchmark_0089d_lm_head.py`
- `tests/test_0089d_lm_head_contract.py`
- `docs/0089D_COMPILED_TORCH_LM_HEAD.md`

## Locked production graph

The fixture manifest records this fail-closed graph contract:

```text
head_full BF16 [3,2304,1024]
→ target slice [0:2304] C-contiguous copy
→ X BF16 [6912,1024]

forward:
    logits BF16 = X @ W, W BF16 [1024,32768]
    stable log-sum-exp and row loss in FP32
    multiply prefix-valid FP32 loss mask
    divide by global masked normalizer
    return FP32 scalar loss
    retain selected X/targets/mask/metadata only; do not retain logits/probabilities

backward:
    recompute logits BF16 = X @ W
    softmax in FP32
    explicitly round probabilities FP32 → BF16 → FP32
    subtract target and apply mask/normalizer/grad scale in FP32
    round dlogits to BF16
    dW tile BF16 = X.T @ dlogits
    promote and accumulate dW tile into FP32 dW accumulator
    dx_selected BF16 = dlogits @ W.T
    allocate zero BF16 full-head gradient and scatter target slice
```

The output projection is bias-free. The CuPy harness uses the real `TiedOutputProjection` layout: an underlying `[V,D]` embedding `Parameter` with BF16 data and FP32 grad, exposed to the head as `[D,V]` transpose views.

## Fixture

The exporter writes one uncompressed NPZ containing:

| Array | Shape | Storage |
|---|---:|---|
| `head_bits` | `[3,2304,1024]` | portable `uint16` BF16 bit patterns |
| `weight_bits` | `[1024,32768]` | portable `uint16` BF16 bit patterns |
| `targets` | `[3,2304]` | `int32` |
| `loss_mask` | `[3,2304]` | FP32 binary prefix mask |
| `manifest_json` | scalar | JSON |

The deterministic valid lengths are:

```text
[2304, 1792, 1024]
```

so the global masked normalizer is exactly `5120`, not `B*T=6912`.

Every array hash covers:

1. NumPy dtype string;
2. exact shape encoded as `int64` values;
3. contiguous content bytes.

This prevents a reshape or reinterpretation from passing as the same fixture.

Generate the fixture outside any measured process:

```bash
.venv/bin/python benchmarks/export_0089d_lm_head_fixture.py \
  --seed 20261024 \
  --output benchmarks/results/0089d_b3_t2304_fixture.npz
```

The fixture is approximately 78 MiB before filesystem overhead: 64 MiB of BF16 weights, 13.5 MiB of BF16 hidden rows, and small target/mask arrays.

## CuPy baseline

The CuPy runner creates only the tied output parameter and invokes the real production methods:

- `DecoderLanguageModel.chunked_lm_head_loss_forward`
- `DecoderLanguageModel.chunked_lm_head_backward`

It does not instantiate the complete 338M-parameter decoder.

The runner sets:

```text
MINI_LLM_BACKEND=cupy
MINI_LLM_FUSED_BF16_CE=1
MINI_LLM_FUSED_BF16_CE_STRICT=1
```

Before timing, it requires both `_fused_bf16_ce_enabled(...)` and `_get_fused_bf16_ce_module()` to succeed. The production methods perform the target-slice copy, loss-only forward, backward logits recomputation, BF16 probability-rounding boundary, tied FP32 gradient accumulation, and BF16 target-gradient scatter.

Fixture host-to-device transfer, tied-parameter construction, and fused RawModule setup are reported separately from warmed timings.

## Compiled Torch candidate

The Torch implementation is deliberately manual. It does **not** call ordinary `torch.nn.functional.cross_entropy` and does not use autograd for the LM-head backward.

`build_torch_functions` returns separate forward and backward functions. For `torch-compile`, each is compiled as:

```python
torch.compile(function, fullgraph=True, dynamic=False, mode=...)
```

Keeping them as separate compiled graphs is a recomputation barrier: the combined timing calls compiled forward and then compiled backward, so Inductor cannot common-subexpression eliminate the required second `X @ W` across one larger compiled graph.

Explicit `.to(torch.bfloat16)` boundaries preserve:

- BF16 logits;
- FP32 softmax;
- BF16 probability rounding before target subtraction;
- BF16 `dlogits`;
- BF16 `dW` tile before FP32 accumulation;
- BF16 `dx`.

TF32 is disabled and Torch float32 matmul precision is set to `highest`.

## Timing contract

Each backend reports independent distributions for:

1. forward loss;
2. backward recomputation;
3. complete forward plus backward.

Each distribution retains:

- raw CUDA-event samples;
- raw synchronized host-wall samples;
- median;
- p10 and p90;
- standard deviation;
- median absolute deviation.

Compilation, fixture loading, device transfer, fused-kernel setup, and gradient clearing are excluded or reported separately. Forward/cache construction for backward-only timing occurs outside the backward event. Forward, backward, and combined medians are independent measurements and must not be added.

The CuPy high-level method obtains the masked normalizer through the existing production mask reduction and `.item()` path. The Torch graph uses the fixture-validated constant `5120`; this preserves the exact objective and backward scale while avoiding a host synchronization inside a compiled graph. Reports should therefore treat the comparison as the matched numerical LM-head/CE compute graph, not as a claim that Torch reproduces Python/device synchronization overhead.

## Exported correctness tensors

Every run exports these arrays as FP32 host data:

1. `loss` — scalar;
2. `dx` — `[3,2304,1024]`, converted from the production BF16 boundary;
3. `dweight` — `[1024,32768]`, the FP32 head-oriented tied-gradient view.

The exported `dweight` is the isolated LM-head contribution. It is captured before any decoder-body embedding backward could add token-lookup or historical-input contributions to the shared embedding gradient.

## Declared parity gates

The gates are defined in the runner before timing:

```text
dx and dweight:
    relative L2 <= 0.02
    cosine      >= 0.999

loss:
    absolute error <= 5e-3
    relative error <= 2e-3
```

Reports include maximum absolute error, RMSE, relative L2, cosine similarity, bias, and worst value pair for tensor outputs. The runner exits with status 2 if a supplied reference fails any gate.

Bitwise equality is not required: the production raw CUDA kernel and Torch/Inductor may use different `expf` approximations and FP32 reduction orders. The explicit BF16 rounding points, tensor shapes, masking, recomputation, and accumulation dtypes are nevertheless fixed.

## Measured results

Environment:

| Component | CuPy | Compiled Torch |
|---|---|---|
| GPU | RTX 5080, SM120 | RTX 5080, SM120 |
| Framework | CuPy 14.2.0 | Torch 2.14.1+cu130 |
| CUDA | runtime 13.2 | 13.0 |
| Compiler | production CUDA/CuPy kernels | Triton 3.8.0 / `torch.compile` |

Twenty-sample warmed CUDA-event medians:

| Backend | Forward | Backward | Complete F+B |
|---|---:|---:|---:|
| Production CuPy | 6.3463 ms | 20.9034 ms | 28.6730 ms |
| Compiled Torch | 5.9027 ms | 20.7150 ms | 27.6849 ms |

Compiled Torch relative to production CuPy:

```text
complete speedup:            1.0357x
complete latency reduction:  3.45%
forward latency reduction:   6.99%
backward latency reduction:  0.90%
```

The LM-head discrepancy is therefore substantially smaller than the complete-block discrepancy measured by 0089B. The large projection GEMMs and production fused BF16 CE are already competitive; compiled Torch has only a modest complete-graph advantage in this scope.

Compilation/setup was excluded from steady-state timing:

```text
first compiled forward:  1738.47 ms
first compiled backward: 1558.41 ms
```

### Numerical parity

All declared parity gates passed:

| Result | Relative L2 | Cosine | Max absolute error |
|---|---:|---:|---:|
| `dx` | 0.00288397 | 0.99996334 | 1.19209e-7 |
| `dweight` | 0.00014405 | 1.00000000 | 9.53674e-7 |

Scalar loss:

```text
CuPy:          10.4047746658
compiled Torch: 10.4047756195
absolute error: 9.53674e-7
relative error: 9.16574e-8
```

### Interpretation for the overall discrepancy

This measurement rejects LM-head/loss as the principal explanation for the remaining history-free PyTorch advantage. At B3/T2304:

- the complete block remains roughly 14–18% behind compiled Torch, depending on the current CuPy run distribution;
- the isolated LM-head/loss is only 3.45% behind;
- the end-to-end no-history model discrepancy should therefore be dominated by repeated block-level materialization and fusion boundaries, not vocabulary projection or CE.

These component medians must not be added to estimate a full-model latency because the full-model profile uses different allocator pressure, stream history, and range boundaries. A direct multi-block compiled comparison is still required for an authoritative overall number.

## Reproduction

Create a CuPy reference:

```bash
.venv/bin/python benchmarks/benchmark_0089d_lm_head.py \
  --backend cupy \
  --fixture benchmarks/results/0089d_b3_t2304_fixture.npz \
  --warmup 5 --repeats 20 \
  --output-tensors benchmarks/results/0089d_b3_t2304_cupy_tensors.npz \
  --output-json benchmarks/results/0089d_b3_t2304_cupy.json
```

Run compiled Torch in the isolated environment:

```bash
.venv-0089b-torch/bin/python benchmarks/benchmark_0089d_lm_head.py \
  --backend torch-compile \
  --fixture benchmarks/results/0089d_b3_t2304_fixture.npz \
  --reference-tensors benchmarks/results/0089d_b3_t2304_cupy_tensors.npz \
  --warmup 5 --repeats 20 --compile-mode default \
  --output-tensors benchmarks/results/0089d_b3_t2304_torch_compile_tensors.npz \
  --output-json benchmarks/results/0089d_b3_t2304_torch_compile.json
```

An eager Torch diagnostic uses the same runner with `--backend torch-eager`.

## Tests

CPU-fast contracts:

```bash
.venv/bin/python -m pytest -q tests/test_0089d_lm_head_contract.py
```

Syntax validation:

```bash
.venv/bin/python -m py_compile \
  benchmarks/reference_0089d_torch_lm_head.py \
  benchmarks/export_0089d_lm_head_fixture.py \
  benchmarks/benchmark_0089d_lm_head.py \
  tests/test_0089d_lm_head_contract.py
```

The isolated compiled-Torch smoke is opt-in so the default contract suite remains CPU-fast:

```bash
RUN_0089D_TORCH_SMOKE=1 \
  .venv/bin/python -m pytest -q \
  tests/test_0089d_lm_head_contract.py::test_optional_isolated_torch_manual_compiled_smoke
```

The expensive exact B3/T2304 benchmark is intentionally not part of the test suite.
