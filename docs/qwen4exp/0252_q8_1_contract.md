# 0252 — Strata-aligned Q8_1 primitive: implemented, oracle validation blocked

## Pinned contract

Read-only source audit against Strata `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`: `src/kernels/cuda/native_mmvq.cu`, `src/kernels/cuda/iq_kernels.cu`, `include/strata/kernels/q8_1_finite.hpp`, and `cmake/strata_gpu_engine.cmake`. Strata's recorded ggml source is `3cf03257f219afbe7334045ff7c6a06ac68c627d`, distinct from the older llama numerical-reference pin. Source-extracted fixture records pinned URLs, hashes and applicable Strata/ggml MIT notices.

The native IQ activation is Q8_1: contiguous36-byte blocks holding half d, half s and32 signed int8 codes. For FP32 block x, A=max(abs(x)), d32=clamp_to_half_finite_range(A/127). Codes use unrounded FP32 d32, CUDA roundf ties-away and clamp[-127,127]; zero block codes are zero. Stored d is half RN-even. Stored s is half RN-even of the clamped original-input FP32 warp-XOR sum, offsets16/8/4/2/1. Native producer uses `--use_fast_math`; an IEEE division control is distinct. Finite-range clamping and division lowering are part of the contract.

For IQ3_S/IQ3_XXS/IQ2_S, the requested packed dot chains use d and signed codes, not s. Effective represented activation is float(d_half)*qs, but a dequantized-weight floating dot does **not** reproduce the integer-packed arithmetic automatically. In particular IQ3_XXS and IQ2_S contain nested truncating integer divisions; do not replace those with approximate floating factors. IQ3_S scales the signed int32 grid/code dot by its odd integer scale. Each expert's SwiGLU hidden is quantized again for down projection. Native GPU uses Q8_1; CPU ggml arithmetic and other canonical formats may instead require Q8_K/Q8_0. Do not force one generic Q8 representation across all sites.

## Candidate files

- `mini_llm/ops/q8_1_gpu.py`: asynchronous CuPy quantize/dequantize, contiguous1D/2D FP32 inputs with K divisible32, bounded supplied-output support, per-device kernel cache. Explicit native versus IEEE arithmetic selection.
- `tests/test_0252_q8_1_gpu.py`.
- `tests/fixtures/strata_q8_1_reference.cuh`: pinned minimal independent CUDA oracle with licenses.

**No model, projection, MoE or executor call site uses this candidate.** Existing FP32 path retained as mathematical oracle. The user authorizes Strata-aligned activation quantization, but this patch does not claim numerical or quality validation.

## Actual validation and stop

Delegated focused suite latest result: **19 passed,22 failed**. Ordinary equation, scale/code rounding, ties-away, dequantization, bounded output, metadata validation and CUDA graph tests reached pass. All22 independent-oracle cases fail at fixture compilation, not at a measured numeric discrepancy. Initial compiler failure involved host stdint headers; one localized correction used CUDA CCCL headers. Retest fails because NVRTC does not accept the fixture's offsetof expression in a static_assert constant expression:

`expression must have a constant value: static_assert(... && offsetof(Q81Block, qs)==4)`

No quantized-byte mismatch has been measured because the oracle kernel has not compiled. Native parity remains unverified. The allowed corrective attempt was used; no further edit or production integration was made in this cycle.

Single next focused action: make the fixture layout assertion NVRTC-compatible without altering extracted arithmetic, then rerun the same41-case suite. Require byte equality of d/s/qs on ties, subnormals, cancellation and saturation before IQ integer-dot integration or throughput claims. The production50t/s target remains unmet.
