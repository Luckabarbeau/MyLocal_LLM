# Patch0202: CPU reference decoding

## Scope and provenance

Adds direct `PackedGGUFWeight` CPU FP32 decoding; no full checkpoint or actual
expert was expanded during implementation or validation. Existing Q-format
entries and decoding paths are preserved. `BF16`, `F16`, and `F32` remain the
existing `gguf_tensor` floating-point path, not packed formats.

Pinned references (local repository HEADs verified before implementation):

- llama.cpp `171e8846b4af9766c354064cb776cb34a50f053f`:
  `ggml/src/ggml-common.h` tables/layouts and `ggml/src/ggml-quants.c`
  `dequantize_row_iq2_s`, `dequantize_row_iq3_s`,
  `dequantize_row_iq3_xxs`, `dequantize_row_iq4_nl`,
  `dequantize_row_iq4_xs`.
- Strata `c77572806996349b49995e4f98d41b5294d36a21`:
  `third_party/strata-ggml/ggml/src/ggml-common.h` and
  `ggml-quants.c`, `dequantize_row_q2_0`.

Tables were parsed directly from the pinned `GGML_TABLE_BEGIN` definitions,
then stored as explicit little-endian byte lanes. Only the five necessary
tables are vendored. MIT attribution and license are embedded in both table
files; the complete upstream license is also in
`third_party_ggml_LICENSE.txt`.

## Block contract

| Format | Elements | Bytes | Notes |
|---|---:|---:|---|
| IQ2_S | 256 | 82 | 10-bit grid indices; independent 8-bit signs; nibble scales |
| IQ3_S | 256 | 110 | 9-bit grid indices; independent signs; odd positive scales |
| IQ3_XXS | 256 | 98 | 8-bit grid indices; 7-bit parity sign indices; scale nibble |
| IQ4_NL | 32 | 18 | Nonlinear signed nibble lookup |
| IQ4_XS | 256 | 136 | Nonlinear lookup; signed six-bit subscales |
| Q2_0 | 64 | 18 | Strata raw type 42, not Q2_K |

`IQ3_S` is **not** `Q3_K`, despite their equal block sizes. Q2_0 uses four
adjacent two-bit lanes per byte, `q = (byte >> (2 * lane)) & 3`, and
`y = d * (q - 1)`. All formats start with a little-endian FP16 scale.
Innermost dimensions must be whole blocks and packed byte counts must match
exactly. FP32 multiplication order follows GGML's reference routines.

## Table ABI for Patch0203

`mini_llm/ops/gguf_codebooks.h` is packaged alongside the native source. It
exports file-local `static const` symbols (include guard
`MINI_LLM_GGUF_CODEBOOKS_H`):

```c
uint8_t ksigns_iq2xs[128];
uint8_t iq2s_grid[1024][8];
uint8_t iq3xxs_grid[256][4];
uint8_t iq3s_grid[512][4];
int8_t  kvalues_iq4nl[16];
```

The declarations above omit `static const` only for readability. Grid entries
are byte arrays, **not** host-endian uint64/uint32 objects. Python exports the
same lowercase names as immutable NumPy arrays with the same shapes and
`uint8`/`int8` dtypes. No runtime dependency on `gguf` or a reference checkout
is required by the production decoders.

## Bounded reference execution

- `dequantize_reference()` explicitly materializes the entire selected tensor;
  use only for small fixtures or deliberately bounded tensors.
- `dequantize_rows_reference(indices, tile_rows=32)` requires a 2-D matrix,
  preserves row order and duplicates, and materializes only requested rows.
- `matvec_reference(vector, tile_rows=32)` computes FP32 `W @ vector` with an
  exact `(columns,)` vector shape. Its FP32 decoding scratch is bounded by
  `tile_rows * columns`, plus small decoder intermediates and the result.
- For `[experts, out, in]`, first call zero-copy `select_outer(expert)`.
  Neither method needs a full expert FP32 expansion. The bound is by complete
  rows, not by columns. An explicitly oversized tile can expand a full matrix.

## Validation design

`tests/test_0202_iq_formats.py` compiles the actual pinned GGML C functions,
extracted verbatim from their source, against the actual upstream structure
and table headers. The sole adapter converts FP16 using compiler `_Float16`;
it does not implement quantization. These C oracles do not include production
Python decoders or vendored tables. C is compiled without fast-math and with
FP contraction disabled. IQs are additionally compared with installed
`gguf==0.19.0`. Optional external references skip explicitly when unavailable;
root overrides are `PATCH0202_LLAMA` and `PATCH0202_STRATA`, and SHAs are checked.

Fixtures cover every grid index (including high-bit extensions), independent
sign bytes/parity sign indices, every scale encoding, all nonlinear/two-bit
lanes, and random payloads with finite FP16 scales including signed zero,
subnormals and extreme finite values. Compiled vendored-header bytes and Python
tables are checked against compiled upstream tables. Synthetic memmap tests
check selected rows, expert views, shape errors and maximum decoding tile size.
Tests print maximum absolute errors for reference decoding and tiled GEMV.

### Recorded validation (2026-11-16)

Environment: repository `.venv`, NumPy FP32 outputs, installed `gguf==0.19.0`,
system `cc`. Both pinned source checkouts were present. All independent C
oracles compiled successfully; no oracle test was skipped.

- `.venv/bin/python -m pytest -q -s tests/test_0202_iq_formats.py`:
  **32 passed**. Every decoder comparison (random/finite-half, exhaustive grids,
  sign/scale/lane fixtures) had **max absolute error 0** against compiled GGML.
  All five IQ comparisons against `gguf` also had **max absolute error 0**.
- Tiled GEMV versus independently C-decoded FP32 matrix multiplication:

  | Format | Maximum absolute error |
  |---|---:|
  | IQ2_S | 1.52588e-5 |
  | IQ3_S | 6.10352e-5 |
  | IQ3_XXS | 3.05176e-5 |
  | IQ4_NL | 3.81470e-6 |
  | IQ4_XS | 2.44141e-4 |
  | Q2_0 | 1.19209e-7 |

  Reduction-order differences passed `rtol=3e-6, atol=2e-5`. Decoder
  comparisons use exact array equality, not that GEMV tolerance.
- `.venv/bin/python -m pytest -q tests/test_0201_gguf_store.py`:
  **37 passed** (standalone count in this checkout).
- `.venv/bin/python -m pytest -q tests/test_0201_gguf_store.py
  tests/test_0106a_packed_gguf.py tests/test_0106g_q5k_packed.py
  tests/test_0106l_q6k_packed.py`: **64 passed, 17 skipped, 1 failed**.
  The sole failure is the old IQ3_S-unsupported assertion: it expects
  `Unsupported`, but the now-supported format correctly reports
  `GGUF IQ3_S: innermost dimension must be a multiple of 256` for its
  deliberately invalid `(1, 32)` shape. No test was changed to conceal this.
- `.venv/bin/python -m pytest -q
  tests/test_0106e_cpu_packed_and_gpu_swaps.py`:
  **22 passed, 12 failed**. Exactly six new formats times two activation dtypes
  fail with `KeyError` at `native_packed_cpu.py:117`,
  `_FORMAT[weight.quant_type]`. Native registration/kernels remain Patch0203;
  the existing native formats continue to pass.
- `git --no-pager diff --check`: passed.

No corrective implementation edits were needed after testing. The new CPU
reference functionality is verified; the repository-wide/native test state is
**not green**, for the explicit integration reasons above.

## Deliberately unsupported integration

This patch does **not** add native CPU kernels, GPU kernels, quantized-linear
routing or residency support. Adding entries to `BLOCK_FORMATS` is not proof of
kernel capability. **`packed_to_device` needs an explicit capability guard
before these formats are promoted.** No GPU support is claimed.

Existing tests dynamically parameterized over `list(BLOCK_FORMATS)` now include
these six formats; native tests will fail until Patch0203. The older
`test_invalid_packed_shapes_and_lengths` also asserts that IQ3_S is unsupported;
that assertion is stale after this patch. Those tests are intentionally not
modified, filtered inside their source, or marked xfail here.
