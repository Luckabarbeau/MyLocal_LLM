# Patch0203: direct packed CPU IQ experts

## Scope and execution contract

Extends `native_packed_cpu.cpp` and its existing ctypes bridge with direct
packed GEMV and complete SwiGLU experts for all six Patch0202 formats:

| Format | Elements/block | Bytes/block | Native ID |
|---|---:|---:|---:|
| IQ2_S | 256 | 82 | 6 |
| IQ3_S | 256 | 110 | 7 |
| IQ3_XXS | 256 | 98 | 8 |
| IQ4_NL | 32 | 18 | 9 |
| IQ4_XS | 256 | 136 | 10 |
| Q2_0 | 64 | 18 | 11 |

Existing IDs 0–5 and Python/C signatures are unchanged. Format-specific block
sizes drive row strides and alignment checks; unknown IDs are rejected before
block-table access. In particular, a down projection with input width 640 is
valid for IQ4_NL and Q2_0, not for the 256-element formats.

The forward graph is:

```
g = W_gate @ x
u = W_up @ x
h = (g * sigmoid(g)) * u
y = W_down @ h
```

Outputs are **unweighted**: router weights must be applied by the caller once.
The existing stable sigmoid is retained. Packed bytes are read-only, including
mmap-backed sources. Each decoded FP32 weight is local to its dot-product lane;
no FP32 weight block, row or matrix is allocated. The existing FP32 hidden vector
is local to each expert call. Activations and accumulators are FP32 internally;
the Python bridge preserves the input/output FP16, FP32 or FP64 dtype contract
(FP64 inputs do not imply FP64 computation).

There is no new thread runtime or global mutable scratch. Existing ctypes
`CDLL` calls release the GIL, and callers may reuse their CPU worker pool.
The native cache fingerprint now includes **both C++ source and codebook header
bytes**, plus the existing compiler/platform inputs. Header-only changes cannot
reuse a stale cached library.

No changes are made to quantized-weight decoding, quantized-linear routing,
residency policy, GPU capability, or parent capability dispatch. Native kernel
support alone is not a claim of GPU support.

## Reference provenance and notices

Before implementation, local reference HEADs were verified and the actual C
`dequantize_row_*` routines inspected:

- llama.cpp `171e8846b4af9766c354064cb776cb34a50f053f`,
  `ggml/src/ggml-quants.c`, for the five IQ formats.
- Strata `c77572806996349b49995e4f98d41b5294d36a21`,
  `third_party/strata-ggml/ggml/src/ggml-quants.c`, for Q2_0 (raw type 42).

Native decoding includes the existing `gguf_codebooks.h` byte-lane tables.
Its embedded GGML MIT attribution/license is preserved; the complete notice is
also in `third_party_ggml_LICENSE.txt`. No reference checkout is needed at runtime.

## Validation

Run from the project root with the repository virtual environment:

```sh
MINI_LLM_NATIVE_BUILD_DIR=.pytest_cache/native_0203 .venv/bin/python -m pytest -q -s tests/test_0203_native_iq_experts.py
```

Recorded result: **35 passed in 1.81 s**, no skips. The production native library
and independent GGML C oracle libraries compiled successfully. The independent
fixture reuses Patch0202's extraction of actual pinned upstream functions,
structures and upstream tables, rather than a second handwritten decoder.
External roots and optional-reference skip behavior remain as described in
`0202_quantization.md`.

Coverage includes:

- All six formats against the bounded Patch0202 FP32 oracle, with noncontiguous
  3-D activations, FP16/FP32/FP64 output contracts and immutable read-only mmap.
- Identity probes for every lane of independently GGML-decoded synthetic
  matrices, plus independent random GEMV reductions.
- Complete three-token experts for each format, mixed synthetic experts at
  representative dimensions `D=2560, F=640` with both IQ4_NL and Q2_0 down
  projections, and ten mixed experts called through four parallel workers.
- Exact repeatability of parallel versus subsequent serial native execution,
  unchanged packed sources, caller-side router weighting, invalid native C
  format/alignment rejection, and actual header-only cache invalidation builds.
- All three reference execution methods monkeypatched to raise during native
  calls. Large-shape oracle execution uses eight-row decoding tiles, not full
  FP32 expert matrices. These are synthetic bytes, not real checkpoint weights.

### Exact measured maximum absolute differences

All independent decoded-lane comparisons are FP32 and **exactly zero** for all
six formats (`atol=0, rtol=0`). For reductions and FP32 experts the requested
comparison tolerances are `atol=2e-5, rtol=5e-5`. FP16 GEMV uses
`atol=1e-4, rtol=5e-5`. Differences are measured after casting outputs to FP64.

| Format | FP32/FP64-output GEMV vs 0202 | FP32 GEMV vs GGML | FP32 complete expert vs 0202 |
|---|---:|---:|---:|
| IQ2_S | 2.6077032089233398e-8 | 1.862645149230957e-8 | 3.4924596548080444e-10 |
| IQ3_S | 5.9604644775390625e-8 | 2.9802322387695312e-8 | 3.4924596548080444e-9 |
| IQ3_XXS | 1.0430812835693359e-7 | 7.4505805969238281e-8 | 5.1222741603851318e-9 |
| IQ4_NL | 3.3527612686157227e-8 | 3.7252902984619141e-8 | 6.4028427004814148e-10 |
| IQ4_XS | 1.0728836059570312e-6 | 3.5762786865234375e-7 | 3.814697265625e-6 |
| Q2_0 | 5.8207660913467407e-10 | 6.984919309616089e-10 | 7.105427357601002e-15 |

FP16-output GEMV differences were zero for every format. FP64-output GEMV
has the same errors as FP32 output because computation remains FP32.

Representative `D=2560, F=640`, IQ2_S gate / IQ3_S up:

- IQ4_NL down: FP32 max absolute difference `8.847564458847046e-9`.
- Q2_0 down: FP32 max absolute difference `1.2369127944111824e-10`.

Ten concurrent mixed experts: maximum FP32 absolute difference
`2.9802322387695312e-8`; caller-weighted sum difference
`3.0267983675003052e-9`. All passed the tolerances above. These differences are
reduction-order effects; decoded-lane checks themselves are exact.

### Existing tests and unresolved integration failure

Only obsolete packed-registry unsupported-format sentinel literals were changed
to `IQ1_S` in `test_0106a_packed_gguf.py`, `test_0106f_qwen3next_gguf.py`,
`test_0106l_gguf_bf16.py`, `test_0106m_qwen3next_mixer.py`, and
`test_0106n_qwen3next_attention.py`. Dense-helper and old Llama-loader IQ3_S
rejection tests were deliberately preserved.

```sh
MINI_LLM_NATIVE_BUILD_DIR=.pytest_cache/native_0203 .venv/bin/python -m pytest -q tests/test_0106e_cpu_packed_and_gpu_swaps.py tests/test_0106a_packed_gguf.py tests/test_0106f_qwen3next_gguf.py tests/test_0106l_gguf_bf16.py tests/test_0106m_qwen3next_mixer.py tests/test_0106n_qwen3next_attention.py
```

Recorded result: **212 passed, 1 failed in 0.49 s**. All existing native tests,
including dynamically parameterized six-format additions, passed. The sole
failure is:

```
test_llama_unsupported_arch_and_formats_stay_failclosed[IQ3_S]
Failed: DID NOT RAISE <class 'ValueError'>
```

Evidence: `GGUFWeightMap.__init__` in `mini_llm/external/gguf_loader.py` accepts
`{"F32", "F16", "BF16", *BLOCK_FORMATS}`. Patch0202's registry expansion therefore
accepts IQ3_S at construction, contradicting this preserved legacy-loader test.
Neither this loader nor that intentional rejection assertion was modified.
The next scoped integration action is to establish the legacy loader's explicit
capability contract separately from the packed registry. No speculative fix or
test weakening was made here. The requested combined regression state is **not
green**, despite passing native validation.
