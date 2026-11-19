# 0253 — Standalone IQ3_S × Q8_1 GPU GEMV

## Objective and status

Implement and validate an **IQ3_S-only** packed GPU GEMV using the validated 0252 Q8_1 producer, without changing production projection, expert, or model dispatch.

Established results: **38 IQ3_S GPU tests passed**, and the **41 Q8_1 GPU tests passed** alongside them. Per-32-element integer-dot partials and rows match the pinned source oracle bit-for-bit under the explicitly declared row reduction. This is not a claim of whole-engine Strata scheduling or model equivalence.

Results and timings are established evidence supplied for this documentation update. Relevant implementation, fixtures, and tests were inspected; no tests, benchmarks, or model workloads were run for this update.

## Files and provenance

- `mini_llm/ops/iq3s_q8_gpu.py`: standalone GEMV, prepacked-activation entry point, and diagnostic partial kernel.
- `mini_llm/ops/q8_1_gpu.py`: unchanged activation producer used by the experiment.
- `mini_llm/ops/gguf_codebooks.py`: canonical IQ codebook storage.
- `tests/test_0253_iq3s_q8_gpu.py`: source-oracle, FP64-reference, ownership, and API tests.
- `tests/fixtures/strata_iq3s_q8_reference.cuh`: extracted IQ3_S dot/helpers and table/layout provenance.

The source oracle is pinned to Strata `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`, `src/kernels/cuda/iq_kernels.cu`, with extracted dot lines 226–251 and helpers 46–52. The fixture records source hashes and a test-only scheduling wrapper. The Q8_1 producer validation is documented in [0252_q8_1_validation.md](0252_q8_1_validation.md).

## Computational graph and API

Weights are packed IQ3_S `W[N,K]`, **110 bytes per 256 elements**, with positive dimensions and `K` divisible by 256. `linear(x, weight, shape=None, out=None, arithmetic='native')` maps FP32 `[K]` or `[M,K]` to FP32 `[N]` or `[M,N]`. A weight object supplies `data`, `shape`, and `quant_type='IQ3_S'`; raw CuPy uint8 bytes require explicit `(N,K)`.

The forward graph is:

1. Quantize each activation row once with 0252, selecting native or IEEE producer arithmetic, and reuse that packed row across weight rows.
2. For each 32-element group, decode IQ3_S grid/sign fields, accumulate signed integer codebook/code products with `__dp4a`, and multiply by the odd integer scale `1 + 2*nibble`.
3. Convert the half weight and activation scales to FP32 and compute `mul_rn(mul_rn(weight_scale, activation_scale), float(integer_dot))`. Q8_1 `s` is unused.
4. Lane `l` accumulates groups `l, l+32, ...` in increasing order. Shuffle-down FP32 additions at offsets `16, 8, 4, 2, 1` produce the row result.

The module uses `--fmad=false` and explicit rounded operations. There are no dense dequantized weights or per-weight FP32 products. `linear_q8_1(packed, weight, shape=None, out=None)` skips activation packing and consumes already packed Q8_1 rows.

Inputs and outputs must be contiguous, correctly typed/aligned, and on the current CUDA device. Supplied output must have exact shape and be disjoint from activation and weight storage. Empty activation batches are supported. Dimensions/activation size are bounded by INT32 and activation rows by the launch's 65,535-row limit.

Execution is asynchronous on the current CuPy stream; callers own lifetimes and cross-stream dependencies. Only compiled kernels and codebook constants are cached per device. `linear` allocates Q8 scratch even with supplied output; `linear_q8_1` only needs an output allocation when none is supplied. This is a single/few-row GEMV experiment, not a prefill GEMM or a claim of allocation-free graph capture.

## Established validation and interpretation

**38 IQ3_S GPU tests plus 41 Q8_1 GPU tests passed.** The IQ3_S suite checks:

- Zero-bit-tolerance FP32 partials against the extracted source dot and rows against the source dot under the declared test reduction tree.
- Native/IEEE activation packing, multiple shapes, encoding coverage, half-scale edges, individual activation lanes, and unchanged quantizer inputs.
- A separate decoded-weight/decoded-Q8 FP64 reference, with magnitude-scaled forward-rounding bounds rather than an exact-FP64 promise.
- Current-stream dependencies, immutable inputs/weights, supplied-output guard regions, and no production host transfers, weight dequantization, or device-wide synchronization.
- Validation failures, empty rows, device weight objects, and per-device module caching.

The suite also has an optional bounded real-GGUF slice path via `PATCH0253_GGUF`; it copies only small IQ3_S slices rather than constructing a model. The reported pass count is not a full-model validation record.

FP32-versus-decoded-FP64 absolute/relative errors are diagnostic measurements of a different accumulation graph. They do not promise model accuracy, equality to the old unquantized-activation path, or whole-engine row parity. No unrecorded error maxima are inferred here.

## Bounded timing, not a production speedup

The supplied tiny timing was approximately **34.63 µs** for the new path versus **30.73 µs** for the old path. This does not demonstrate a speedup. The supplied-output setup was unfair: the Q8 path used caller-supplied output, so the measurements must not be treated as an equal-allocation comparison. They establish only bounded tiny-workload timing, not model throughput or a production performance decision.

## Limitations and next dependencies

Only IQ3_S is implemented; IQ2_S and IQ3_XXS do not inherit validation from it. Production dispatch is **not integrated**, and existing FP32 GPU arithmetic remains unchanged. Other formats, shared gate/up activation reuse, expert-level hidden re-quantization, matched allocation/timing conditions, and matched model/quality evaluation are separate prerequisites before considering integration. The prolonged 50-token/s goal remains unproven by this work.

## Licensing

Retain the fixture's Strata and ggml MIT notices, source hashes, and codebook attribution. The test scheduling wrapper is not an extracted whole-engine kernel. Upstream notices remain applicable alongside the repository's LGPL license.
