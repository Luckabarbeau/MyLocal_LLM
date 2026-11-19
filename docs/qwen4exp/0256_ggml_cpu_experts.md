# 0256 — Optional GGML packed CPU experts and parent integration

## Objective and status

Provide an explicitly selected external GGML CPU integer-dot executor for packed experts, avoiding dense weight materialization and retaining FP32 SwiGLU and outputs. Parent executor/model integration is present and opt-in. The default CPU expert backend remains native, and **FP32 GPU arithmetic is unchanged**.

Established results supplied for this report:

- Original focused CPU validation: **30 passed, 1 optional real-expert NPZ test skipped**.
- Parent integration CPU validation across GGML, benchmark, async, and model scopes: **64 passed, 4 skipped**.
- Tiny synthetic CPU expert: **37.3 µs GGML versus 465.6 µs native**, approximately **12.49×**.
- Completed long-run protocol: 128-token warmup, then the standard `short-r1` response, **427 emitted tokens ending at EOS**; **426 timed decode forwards in 53.00188 s**.

Implementation and relevant tests were inspected for this documentation update. No tests, benchmarks, or model workloads were run here; these numbers are the supplied established record, not newly reproduced measurements.

## Files and integration contract

- `mini_llm/ops/ggml_packed_cpu.py`: explicit library loading, ABI/traits checks, provenance, and FP32 linear/expert API.
- `mini_llm/ops/ggml_packed_cpu.cpp`: small original C++ bridge calling external quantizer/dot function pointers.
- `mini_llm/ops/expert_residency.py`: opt-in packed CPU dispatch and `ggml_cpu_experts` counter.
- `mini_llm/external/qwen4exp_model.py`: model backend selection and executor compatibility checks.
- `tests/test_0256_ggml_packed_cpu.py`: direct-ctypes oracle, formats, mixed experts, quantization reuse, concurrency, provenance, validation, optional real NPZ, timing, and executor integration.
- Related parent contracts are exercised in the async/model/benchmark validation scope, including `tests/test_0106d_async_experts.py`, `tests/test_0106j_async_buffer_ownership.py`, and `tests/test_0209_qwen4exp_model.py`.

Use `HeterogeneousExpertExecutor(..., cpu_packed_backend='ggml', compute_dtype='float32')` or `Qwen4ExpGGUFModel`'s `expert_cpu_backend='ggml'`. The model default is `expert_cpu_backend='native'`; the executor rejects GGML selection with non-FP32 compute. The model-created executor retains `host_dequant_budget_bytes=0`.

The GGML module's introductory integration note describes the earlier standalone stage and is stale: inspected parent constructor/dispatch and model forwarding establish the current integration. No source file was changed to correct that note in this documentation-only task.

## Explicit external dependency and provenance

The established library selection is:

```text
MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
```

The standalone API also accepts an explicit `library_path`. There is no automatic library search/download or implicit switch from native. A missing selection is an error; the test fixture skips external-library tests when the explicit dependency is unavailable.

| Provenance field | Established value or meaning |
| --- | --- |
| Reference source/ABI target | `171e8846b4af9766c354064cb776cb34a50f053f` |
| Selected library SHA-256 | `924ac13f3e949cb1b62eddff20221600a6efcb1a2b99eb4abd390e75c23df839` |
| Runtime revision verification | `library_revision_verified=False` |

The executor records the resolved library path, actual binary hash, ABI reference revision, bridge path, and activation quantizers. **An ABI target record is not independent proof that a binary was built from that revision.** Traits and symbol checks detect expected type/function mismatches, not arbitrary incompatible fork ABIs. Use the pinned compatible build and retain build provenance; hashing is identification, not source attestation.

The bridge is compiled locally with a C++ compiler (`CXX`, or available g++/clang++), using the existing `MINI_LLM_NATIVE_BUILD_DIR` convention. It does not link or download GGML source. The selected external shared library and its runtime dependencies must already be available.

## Arithmetic and API contract

The six supported weight formats select activation quantization **per weight**, not through one generic Q8 representation:

| Packed weight | GGML activation type |
| --- | --- |
| `IQ2_S` | `Q8_K` |
| `IQ3_S` | `Q8_K` |
| `IQ3_XXS` | `Q8_K` |
| `IQ4_XS` | `Q8_K` |
| `IQ4_NL` | `Q8_0` |
| `Q2_0` | `Q8_0` |

`linear(x, weight)` accepts FP32 `[M,K]` or `[B,T,K]` and returns FP32 with the final dimension replaced by weight output rows. Noncontiguous activations are copied to contiguous row storage. Weights must be supported packed `[N,K]` matrices with valid byte counts; Q8_K input widths must be divisible by 256 and Q8_0 widths by 32. Empty activation batches are supported.

`expert(x, (gate, up, down))` accepts FP32 `[M,D]`, with gate/up `[F,D]` and down `[D,F]`, and computes:

```text
q_input = GGML_quantize(x, activation_type_for_gate)
g = GGML_dot(gate, q_input)
u = GGML_dot(up, shared_q_input_or_separately_quantized_x)
h = FP32(SiLU(g) * u)
q_hidden = GGML_quantize(h, activation_type_for_down)
y = GGML_dot(down, q_hidden)
```

Gate/up share the quantized input when their activation types match, even if their weight formats differ. If activation types differ, each required representation is produced. Down always quantizes the FP32 hidden using its own required type. **Routing weights are not applied inside this API**; the parent routing/reduction layer remains responsible for them.

This Q8 integer-dot graph is not the native dequantized-weight FP32 graph, nor the GPU Q8_1 experiment. Different arithmetic can change model outputs even when all arrays crossing the API are FP32.

Packed weights are read directly; no dense weight cache is created. Scratch and hidden buffers are local to each call. The synchronous bridge has no OpenMP/internal worker pool or shared activation buffers; ctypes releases the GIL, and parent `cpu_workers` supplies external parallelism. Keep the executor/library and immutable packed storage alive while worker calls are in flight. This CPU backend does not change GPU stream/fence ownership rules.

## Established validation and synthetic timing

The original **30-pass/1-optional-real-NPZ-skip** result is a focused validation scope. The later **64-pass/4-skip parent integration** result covers GGML, benchmark, async, and model integration; do not sum the counts as distinct cases or reinterpret optional skips as successful real-expert validation.

The focused oracle independently declares ctypes ABI functions and calls the **same selected GGML library** directly per row, without using production bridge/dispatch internals. Comparisons use FP32 `atol=2e-7`, `rtol=3e-5`, not a bit-exact whole-model promise. This independently checks bridge dispatch/composition, but does not independently prove GGML's quantization mathematics or binary source revision.

Coverage includes all six formats at multiple widths, seven expert format triples (including different gate/up activation types), `D=256, F=320`, immutable memmapped weights, input preservation, empty batches, validation/native ABI guards, external thread-pool calls, and quantizer-call instrumentation proving shared gate/up input reuse. Production weight dequantization is forbidden in relevant tests. Separate comparisons explicitly show Q8 results differ from the native dequantized-FP32 baseline.

Optional real-expert validation uses `MINI_LLM_GGML_CPU_EXPERT_NPZ` with `allow_pickle=False`; its skip leaves that focused real-NPZ path unverified. Optional warmed timing is selected by `MINI_LLM_GGML_CPU_TIMING=1`.

For the supplied **synthetic single CPU expert, D=256/F=320**, the warmed result was **37.3 µs versus 465.6 µs**, approximately **12.49×**. This compares different arithmetic graphs and is not a real-model speedup or accuracy equivalence claim.

## Completed long-run record and counting conventions

The supplied matched-protocol 0256 run completed **128 warmup tokens**, then the standard **`short-r1` response of 427 emitted tokens, ending at EOS**. The timed decode interval contains **426 model forwards**, not 427:

| Metric | Calculation | Result |
| --- | --- | ---: |
| Decode forward rate | `426 / 53.00188 s` | **8.03745 forwards/s** |
| Strata-N emitted-token convention | `427 / 53.00188 s` | **8.05632 emitted tokens/s** |

The same interval is used for both conventions; emitted-token rate is not evidence of an additional timed forward. Neither figure includes a claim that initialization/warmup cost is part of that decode interval.

The prior native baseline timed out after **180 s during warmup**, so it has no comparable completed response/decode measurement. Observed warmup rates were **1.7739/s old native** and **8.1142/s GGML**. Policy, workers, and arithmetic also changed: this is **not an isolated CPU-backend A/B**, and neither the completed run nor the warmup comparison permits attributing the entire change to GGML alone.

The prolonged **50-token/s goal remains unmet**. Completion of this response establishes a completed run under the selected configuration, not quality parity, general long-context robustness, or achievement of the throughput target. No additional transcript, configuration manifest, or artifact contents are invented here; the inspected `diagnostics_0256_long_v1/` directory was empty during this update, so the supplied long-run record was not independently reconstructed from local artifacts.

## Remaining dependencies and licensing

Next validation dependencies are a pinned reproducible external build record, the optional real-expert oracle case, and an isolated matched backend comparison holding policy/workers/other switches fixed. Matched model-level numerical/quality evaluation is required because activation quantization changes arithmetic. Further throughput work remains necessary for 50 tokens/s; no default backend or GPU precision change follows from this report.

The Python/C++ bridge and focused test carry `LGPL-3.0-or-later` SPDX notices. GGML is a separately selected **MIT-licensed** dependency, attributed to Georgi Gerganov and the ggml authors; preserve its applicable notices when distributing the library. The original bridge calls external ABI functions and does not copy GGML quantization mathematics. External dependency licensing and binary/build provenance remain distinct from repository licensing and from numerical validation.
