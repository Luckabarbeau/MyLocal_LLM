# 0106K — Matched-input FP64 audit of packed expert numerical parity

Date: 2026-11-09. Continuation of the four numerical checks left unresolved by
0106J. **No production inference kernel, routing, synchronization, weight
format, or training code changed in this increment. No tolerances increased.**

## Conclusion

The evidence supports numerical roundoff/reference mismatch for these four
fixtures, not an incorrect GPU kernel equation:

1. The two FP16 tests compared device-rounded inputs to an oracle using the
   original, unrounded FP32 inputs. Matching input and output storage rounding
   removes their tolerance violations.
2. The two FP32 complete-expert tests treated NumPy FP32 BLAS accumulation as
   ground truth despite cancellation-heavy outputs. The GPU is closer to the
   matched FP64 reference than that CPU FP32 reference at the failing entries.
3. All four cases pass the **original numerical tolerances** against the
   corrected reference. The wider selected GPU and CPU suites pass too.

This conclusion is specific to the observed fixtures. It does not establish
bit-exact FP64 equivalence, independent correctness of every GGUF decoder, or
full-model Qwen inference.

## Forward graph and reference contract

For storage dtype d (FP16 or FP32 in the audited GPU cases):

```text
x_d  = cast_d(x)
g64  = float64(x_d) @ float64(W_gate).T
u64  = float64(x_d) @ float64(W_up).T
h64  = SiLU(g64) * u64
h_d  = cast_d(h64)
y64  = float64(h_d) @ float64(W_down).T  # complete expert only
y_d  = cast_d(y64)
```

For fused gate/up tests, the output is h_d. There is **no FP16 storage rounding
of g64 or u64**: the production fused kernel accumulates and applies its
nonlinearity in FP32 before storing the hidden activation. FP64 arithmetic is
used by the oracle to reduce reference error, not as a claim about production
accumulation dtype. The hidden-storage boundary before down is preserved.

`mini_llm/diagnostics/quantized_numerics.py` implements this small-fixture
reference independently of production matrix and SiLU operators. It promotes
inputs and decoded weights **before** multiplication and uses a stable FP64
sigmoid. Requesting FP64 output from `quantized_linear` would not suffice:
its CPU path deliberately computes its matrix product in FP32 before casting.

Packed decoding still uses `PackedGGUFWeight.dequantize_reference()`. This is
an independent arithmetic oracle, **not an independent bit-layout decoder**.
Existing scalar addressing/format tests remain responsible for that coverage.
The diagnostic expands small matrices and must not be used to load an entire
external model or as an inference fallback.

## Reproducible experiment

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106k_matched_precision.py \
  --output benchmarks/results/0106k_matched_precision_gpu.json

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106k_matched_precision.py \
  --output benchmarks/results/0106k_matched_precision_cpu.json
```

The audit reuses the **actual** fixture factories from 0106B/C via `runpy`;
it requires the dev dependency `pytest`, but executes no test functions.
Seeds, shapes, scales, original FP32 references and tolerance values are
preserved. The legacy formulas are explicitly retained in the audit so their
failures remain reproducible after correcting the test references.

The GPU audit was run on the RTX 5080 with no training/inference compute
process present at the initial availability check. No existing process was
stopped. The CPU run reports all GPU measurements as skipped, not as passes.
The script writes JSON diagnostics before raising on a matched-reference
failure. Legacy discrepancies are observations, not exit causes.

## Measurements

FP32 full experts: D=F=256, [2,256] inputs, Q3_K or Q4_K throughout,
weight seeds 91/92/93, input seed 111, standard deviation .01.

FP16 fused hidden: D=512, F=256, [2,3,512] inputs, gate/up seeds 171/172,
input seed 173, standard deviation .07.

| Case | Legacy violating elements | Legacy maximum absolute error, all entries | Matched maximum absolute error | Maximum tolerance ratio, matched |
|---|---:|---:|---:|---:|
| Full FP32 Q3_K | 1/512 | .0048828125 | .001159667969 | .433373 |
| Full FP32 Q4_K | 2/512 | 16.0 | 2.0 | .250362 |
| Fused FP16 Q4_K/Q4_K | 9/1536 | .3042602539 | 9.536743164e-7 | .000158321 |
| Fused FP16 Q3_K/Q4_K | 3/1536 | .04100036621 | 6.103515625e-5 | .00782203 |

The tolerance ratio is `abs(actual-reference) / (atol + rtol*abs(reference))`;
a ratio <=1 passes. Maxima in different columns need not occur at the same
coordinate. In particular, the Q4_K full-expert fixture has outputs of order
10^7: an absolute maximum of 2 is not a two-unit error at the small,
cancellation-sensitive entries. Original full-expert tolerances remain
`rtol=2e-5, atol=.002`; original fused FP16 tolerances remain
`rtol=.02, atol=.006`.

### FP32 cancellation examples

| Entry | GPU FP32 | Original CPU FP32 reference | Matched FP64 arithmetic with FP32 stores |
|---|---:|---:|---:|
| Q3_K [0,45] | -14.1014556885 | -14.0981330872 | -14.1004667282 |
| Q4_K [1,44] | 7506.140625 | 7506.560058594 | 7506.178710938 |
| Q4_K [1,154] | -17024.0703125 | -17023.697265625 | -17024.05078125 |

At those coordinates, the down-projection cancellation ratios
`sum(abs(h_j * W_ij)) / abs(sum(h_j * W_ij))` are approximately **998.5,
327.3, and 165.3**, respectively. These ratios describe cancellation in the
actual down dot product, not a complete nonlinear condition number for the
whole FFN. They explain why small accumulation differences can be amplified
relative to the final output. The original NumPy FP32 reference itself fails
the old tolerances against the matched FP64 oracle for both full-expert
fixtures; the GPU passes them.

Separately probed FP32 gate/up products differ from their FP64 references by
at most approximately 6.54e-6 across these four cases. Those probes use the
actual device-rounded inputs and are **not direct instrumentation of the
fused kernel's internal registers**. Local down comparisons fix the actual GPU
hidden activation before applying the FP64 down matrix, avoiding double
counting upstream error. Detailed stage metrics are in the raw audit JSON.

### FP16 input/output rounding examples

At Q4_K/Q4_K [1,2,146]:

- Original reference: `.03207305818796158`.
- GPU output: `.10614013671875`.
- Matched reference: `.10614013671875`.

The apparently large relative error was a comparison of different rounded
inputs, not evidence of a bad expert selection or packed kernel.

Across the FP16 fixtures, input rounding alone changes the unrounded output
by as much as .2433464 (Q4_K/Q4_K) and .03643064 (Q3_K/Q4_K).
Output rounding changes the mathematical hidden values by as much as .2476978
and .02977269, respectively. These isolated maxima are not additive error
budgets. The remaining matched errors are 9.54e-7 and 6.10e-5.

## Minimal correction

- `tests/test_0106b_quantized_linear.py`: the resident full-expert reference now
  uses FP64 arithmetic and explicit FP32 hidden/output stores. The test still
  rejects runtime dequantization, validates cache residency and bytes, and
  checks provider reuse. Reference construction happens before its monkeypatch.
- `tests/test_0106c_fused_packed_expert.py`: the GPU fused reference now rounds
  the input/output to the tested dtype while keeping the reference arithmetic
  FP64. Shapes and GPU output dtype assertions remain.
- Neither tolerance literal changed. No test was deleted or skipped to obtain
  a pass. CPU-reference behavior of the production operators is unchanged.

New oracle tests independently construct dense Q8_0 matrices from integer
payloads. They cover 2D/3D inputs, FP16/FP32/FP64 storage, actual-input dtype
matching, unrounded fused projections, hidden storage before down, stable
large-negative gates, and invalid inputs/projection contracts. One initial
API-validation failure was corrected locally by checking invalid input type
before deriving its default storage dtype; arithmetic was not modified.

## Verified state

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106b_quantized_linear.py tests/test_0106c_fused_packed_expert.py \
  tests/test_0106g_q5k_packed.py tests/test_0106h_triton_cupy_pointer.py \
  -k 'gpu or triton'
# 49 passed, 31 deselected in .70s

MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106i_routing_trace.py tests/test_0106j_async_buffer_ownership.py \
  tests/test_0106d_async_experts.py::test_cuda_copy_worker_and_mixed_cpu_gpu_experts
# 24 passed in 3.62s

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106*.py tests/test_0085b_external_llama.py \
  tests/test_0085c_external_safetensors.py tests/test_0085f_external_gguf.py \
  tests/test_0085g_gguf_tokenizer.py tests/test_moe_experts.py \
  tests/test_moe_router.py tests/test_topk.py
# 237 passed, 34 skipped in .79s
```

The oracle's focused CPU selection also passed all **15 tests**. GPU audit:
**four measured cases, zero matched-reference failures**. CPU audit: GPU
measurements explicitly skipped. Machine-readable summary is
`docs/0106k_validation.json`; local detailed audit artifacts are in the
ignored `benchmarks/results/` directory. No throughput optimization or
performance benchmark was attempted in this numerical investigation.

## Remaining scope

The four previously unresolved numerical checks are now resolved without a
production kernel change. This does not remove the Qwen decoder, Q6_K/BF16,
or unidentified target-model metadata blockers recorded in 0106I/J. Future
numerical checks should match storage boundaries and use a higher-precision
oracle before changing a mathematically consistent implementation.
