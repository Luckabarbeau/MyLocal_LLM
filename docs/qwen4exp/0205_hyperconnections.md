# Patch0205 — separate FP32 Qwen4Exp hyper-connections

## Scope

Only `mini_llm/external/qwen4exp_hyperconnection.py`, its focused test file,
and this document belong to this patch. No shared modules, Qwen3 code, parent
status documents, full decoder, GGUF loader, placement policy, or integration
are changed. Dense validated FP32 matrices are intentional for this first
reference; there are no projection callbacks or implicit quantization paths.

## Source and graph contract

Parent: [reference_contract.md](reference_contract.md), Hyper-connections.
Actual local source inspected before implementation:

- llama.cpp `171e8846b4af9766c354064cb776cb34a50f053f`,
  `src/models/qwen4exp.cpp:267–350` under
  `/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp`.
- Radiance `a555ef392a48fbdac3df4a0eeb1197ec3f6e6100`,
  `arch/common/rad_block_hc.h` and `libref/ref_hc.cpp` under
  `/home/lucka/Documents/my_local_LLM/qwen4exp-references/radiance`.

Both source revisions were confirmed locally; both working trees were clean.
For each token independently, with C=4 and epsilon supplied by the caller:

```
n_c = (h_c / sqrt(mean_D(h_c**2) + epsilon)) * gamma_c
n = flatten_streams(n_c)
a = down @ n / 4
low = a * sigmoid(a)
g = sigmoid(up @ low)
block_input = sum_c(n_c * g_c) / 4
gains = 2 * sigmoid(inject @ n / 4)
new_h_c = ORIGINAL_h_c + gains_c * block_output
```

The low-rank division precedes SiLU. Read gates are per feature; injection
gains are one scalar per stream. The write never uses normalized residuals.
Gamma is the multiplicative **converted** scale: no additional `+1`.
The stream axis is per-token state across blocks, not token history.

## API

```python
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection

hc = Qwen4ExpHyperConnection(gamma, down, up, inject, epsilon=epsilon)
streams = hc.initialize(token_embeddings)
block_input, gains = hc.read(streams)
block_output = block(block_input)  # caller owns this computation
streams = hc.write(streams, block_output, gains)

final_hc = Qwen4ExpHyperConnection(final_gamma, final_down, final_up,
                                 epsilon=epsilon)
hidden = final_hc.collapse(streams)
```

- Uses existing `mini_llm.backend.xp` (NumPy/CuPy). No process backend change.
- All weights/activations must be arrays on that backend with explicit
  `float32` dtype. FP16/Float64 inputs are rejected, not silently narrowed.
- Constructor accepts gamma `[4,D]` or exactly `[4*D]`, converting the latter
  to canonical `[4,D]`. No shared/broadcast gamma accepted.
- Output-first weights: down `[R,4D]`, up `[4D,R]`, optional inject `[4,4D]`.
  D and R must be positive. Construction copies weights.
- Activations: streams `[...,4,D]`; embeddings/input/output `[...,D]`;
  gains `[...,4]`. Leading dimensions must match exactly on write.
- `read` returns `(block_input, gains)`; without injection, gains is `None`.
  `collapse` computes only the read mix, even if injection weights are present.
  The caller must pair each read's gains with its own original streams and
  output; shape checks cannot prove gain provenance.
- All methods are out-of-place and keep no activation/history/cache state.
  There is no backward/training API. Weight arrays are publicly inspectable;
  callers must not modify them during inference.
- Epsilon must be positive, finite, and representable in FP32. Stable sigmoid
  uses `exp(-abs(x))` so saturated finite gate arguments do not overflow exp.
  This is not an arbitrary-magnitude safe RMSNorm/GEMM: FP32 products can still
  overflow for extreme input/weight magnitudes. No NaN sanitization is done.

## Validation and evidence boundary

Tests cover exact hand fixtures (zero read gates give half the normalized
stream mean; zero injection gives gain 1), nonzero scaling fixtures, independent
Float64 scalar-loop read/gain equations, two paired nonlinear synthetic blocks
and a final read-only collapse, multiple token/leading/feature/rank dimensions,
non-mutation, flat gamma conversion and constructor ownership, shape/dtype
rejection, epsilon validation, and saturated sigmoid/zero norm.

The Float64 oracle uses the same FP32 weight values but independent scalar
loops and double arithmetic. Synthetic test comparison uses `rtol=2e-6` and
`atol=2e-6`; tests print maximum absolute differences. The optional CuPy test
launches this same suite in a fresh interpreter with `MINI_LLM_BACKEND=cupy`
when CuPy and a device are available; it does not substitute NumPy for CuPy.

### Recorded verified state

Command (run from the repository root):

```sh
MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest tests/test_0205_qwen4exp_hyperconnection.py -q -s
```

Latest execution: **24 passed** in 1.51 s, including the actual CuPy child
suite (**23 passed, 1 deselected** in 0.51 s; the deselected test is the
subprocess launcher itself). No GPU skip occurred.

| Backend | Largest absolute difference across printed oracle comparisons | Final collapse maximum across tested geometries |
|---|---:|---:|
| NumPy FP32 | 3.38620144e-7 | 8.51649637e-8 |
| CuPy FP32 | 4.04360494e-7 | 1.00955372e-7 |

Both largest discrepancies occur after the second paired block write, compared
against the independently propagated Float64 synthetic trajectory. Exact zero
fixtures pass. The first run exposed a host/device mismatch in the **test
comparison helper**, not in the HC equations. One localized correction moved
both comparison operands to host; the identical command then passed. Production
implementation was not changed after its initial implementation. This records
a verified state for the new module and focused tests, not the full repository.

**Independent-source execution parity is pending.** Radiance libref equations
were inspected, not compiled/executed here: the original functions depend on
RadArgs/tensor helpers, dtype/plane decoders and optional quantization support.
No rewritten C++ shim is presented as original-source parity. Radiance also
has dtype rounding boundaries (BF16/FP8 in its served graph) unlike this FP32
module. The Float64 tests are mathematical tests on synthetic weights only.
There is no claim of llama.cpp execution parity, real-checkpoint layer parity,
full decoder/logit validation, GPU performance, or readiness for parent graph
integration. Parent owns integration/status updates after reviewing evidence.
