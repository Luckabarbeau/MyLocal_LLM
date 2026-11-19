# 0252 — Q8_1 GPU primitive: oracle validation completed

## Objective and status

Validate the standalone Strata-aligned Q8_1 activation codec before using it in packed integer-dot experiments. The established completed result is **41 GPU tests passed**, with byte-exact comparison against the pinned CUDA source oracle in both native-fast-math and IEEE-control modes.

This report supplements, rather than rewrites, [0252_q8_1_contract.md](0252_q8_1_contract.md). That historical report records the earlier NVRTC fixture-compilation blocker and its then-current 19-pass/22-fail state. Those failures were compilation failures, not measured byte discrepancies. The completed 41-pass result is the later validation state; the blocked historical document is intentionally preserved.

Results here are established results supplied for this documentation update. Source and tests were inspected; no tests, benchmarks, or model workloads were run for this update.

## Files and reference

- `mini_llm/ops/q8_1_gpu.py`: standalone CuPy CUDA quantizer/dequantizer.
- `tests/test_0252_q8_1_gpu.py`: independent-source byte comparisons, contract and lifecycle checks.
- `tests/fixtures/strata_q8_1_reference.cuh`: source-extracted CUDA oracle, pinned URLs, source hashes, and license notices.

The Strata pin is `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`. The audited producer is `src/kernels/cuda/native_mmvq.cu`, with finite-range rules from `include/strata/kernels/q8_1_finite.hpp`. Its recorded llama/ggml attribution includes revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`; this is not the separate CPU ABI target used by 0256.

## Arithmetic and API contract

Each 32-element FP32 block is packed into **36 bytes**: half `d`, half `s`, and 32 signed int8 codes. For finite inputs:

1. Compute `A = max(abs(x))` and `d32 = finite_half_range(A / 127)`.
2. Produce codes using the **unrounded FP32** `d32`, CUDA ties-away rounding, and clamping to `[-127, 127]`; all-zero blocks produce zero codes.
3. Store `d` with half round-to-nearest-even.
4. Sum the original FP32 inputs using warp-XOR offsets `16, 8, 4, 2, 1`, clamp to the finite half range, and store half-RN-even `s`.

`arithmetic='native'` uses `--use_fast_math`, including its division lowering and flush-to-zero behavior. `arithmetic='ieee'` compiles without that flag. Each mode is compared with its corresponding source-oracle compilation; this does **not** mean that native and IEEE outputs must equal each other.

`quantize_q8_1(x, out=None, arithmetic='native')` accepts contiguous, four-byte-aligned CuPy FP32 `[K]` or `[M,K]`, with positive `K` divisible by 32 and total elements within INT32. Zero activation rows are allowed. The result is uint8 with last dimension `K/32*36`.

`dequantize_q8_1(packed, out=None)` returns FP32 `float(d_half) * signed_code`; it does not use `s` and does not invert quantization losslessly. Supplied outputs must have exact shape and dtype, current-device placement, alignment, and storage disjoint from the input.

Launches are asynchronous on the current CuPy stream. Callers own storage lifetimes and cross-stream ordering. Only kernels are cached per device and arithmetic mode; there is no retained activation or weight cache. With supplied outputs, the codec needs no scratch allocation. Nonfinite input codes are unspecified, and HIP is not supported by this pinned CUDA implementation.

## Established validation

**41 GPU tests passed.** The focused suite covers:

- Zero-tolerance comparison of every `d`, `s`, and code byte against the extracted CUDA oracle, separately for native and IEEE modes.
- Random shapes, rounding boundaries and adjacent representable values, ties, subnormals, saturation, sum overflow, signed zeros, and cancellation.
- Unrounded-FP32-scale code generation and ties-away behavior; limited CPU equation checks away from CUDA division ambiguity.
- Exact dequantization of stored metadata/codes, input immutability, supplied-output guard regions, and empty-row behavior.
- Current-stream execution and warmed CUDA graph capture with supplied buffers, with host readback/device-wide synchronization forbidden by the test.
- Invalid dtype, rank, stride, dimensions, mode, alignment, overlap, and packed size; per-device cache identity and conditional cross-device checks.

These are direct byte comparisons, not finite-difference or tolerance-based approximations. A single-GPU run does not establish execution on a second device merely because the conditional test exists.

## Scope, limitations, and next dependencies

The codec is a standalone primitive; 0253 consumes it in another standalone GPU experiment. This validation does not establish production model/executor dispatch, model quality, whole-engine arithmetic equivalence, or decode throughput. The existing FP32 model path is not replaced by this report.

Packed-dot consumers must validate their own integer arithmetic and reduction graph. In particular, Q8_1 is not interchangeable with Q8_K or Q8_0, and a dequantized floating dot is not an integer-dot oracle. Expert SwiGLU hidden activations require a new quantization step before down projection. Integration and matched model validation remain separate dependencies; the prolonged 50-token/s objective is not established here.

## Licensing

The source-extracted fixture retains Strata and ggml MIT notices and provenance. Preserve those notices when reusing extracted source; this validation does not change the repository's LGPL licensing or relicense upstream material.
