# 0106Q — Independent Qwen3-Next reference checks

Date: 2026-11-10. Production baseline remains
`f133d3f2bafd47d7edc21b3fdb452a90a347a2db`.

**Verified:** strict matched-weight/F32 small-model parity against independently
executed llama.cpp, including full and incremental logits. **Characterized, not
strictly verified:** the installed quantized Coder-Next GGUF. Both engines chose
the same next token at all four tested positions, but their raw logits differ
substantially more than FP32 roundoff.

No production equation, kernel, tolerance, loader or generation API changed.
The existing MyLocal_LLM runtime remains the inference backend. llama.cpp is an
optional test oracle, never a production fallback. No model or release binary
was downloaded and no Strata binary was executed.

## 1. Why two reference checks are necessary

MyLocal_LLM's current path evaluates packed weights with FP32 activations. The
pinned llama.cpp CPU implementation uses quantized activation operands for
quantized matmuls: Q8_K for Q4_K/Q5_K/Q6_K weights and Q8_0 for Q8_0 weights.
This applies to ordinary and expert-indexed matmuls. FP32 output/KV storage does
not make these two execution graphs precision-equivalent.

Source: llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c:272–275,311–330`, with activation
conversion at `:1273–1274,1346–1358,1567–1568,1632–1641`, revision
`171e8846b4af9766c354064cb776cb34a50f053f`.

Therefore:

1. **Strict structural test:** generate a tiny packed original and an all-F32
   equivalent reference. Every reference weight is exactly the original's
   dequantized value. This avoids activation quantization in the independent
   runtime without expanding the installed 80B model.
2. **Actual checkpoint:** execute the same quantized GGUF and explicit IDs in
   both engines. Report errors, rankings, margins and within-engine variability;
   do not invent a strict cross-engine threshold.

The strict fixture threshold was declared before executing it:
`rtol=3e-4, atol=2e-5`. That strict reference threshold was not changed after execution.
Actual-mode `strict_reference_parity_passed` is explicitly **null**.

## 2. Added files and contracts

### `benchmarks/reference_0106q_llama_logits.cpp`

A small C++17 exporter using the existing source-built llama.cpp C API:

- Explicit token IDs, absolute positions and one sequence; no tokenizer,
  implicit BOS/EOS, sampler or speculative decoder in the reference path.
- A fresh model/context process for full-sequence and incremental tests.
- Zero GPU layers; empty offload-device list; K/Q/V and operation offload
  disabled; flash attention disabled; FP32 K/V; MTP disabled; two CPU threads.
- mmap/lazy loading with extra repack buffers disabled; no whole-model
  floating-point expansion.
- Every requested input position has its logits flag set. Logits are copied
  before the next decode call and correspond to the next token after that row.
- Bounds: at most 64 tokens, context at most 128, threads at most eight.
- Versioned little-endian binary output carries token IDs, dimensions and FP32
  values. Malformed/nonfinite output is rejected; existing output is not replaced.
- Linux `dl_iterate_phdr` reports the actually loaded reference-library paths.

The installed library is CUDA-linked. CPU-only execution flags **do not guarantee
zero CUDA discovery/initialization**; the build may enumerate GPU devices. These
runs were started after the GPU was reported free. A strictly no-CUDA reference
requires a separate CPU-only build of the same source.

### `benchmarks/benchmark_0106q_reference.py`

Builds the exporter against an explicitly selected source/build tree, runs both
reference modes, and exports MyLocal full/incremental logits in an isolated
NumPy subprocess. Backend selection is never changed inside a numerical process.
All runtime stdout/stderr and binary/NPZ data remain in a run-specific directory.
Reports/artifacts are not silently overwritten; use a new output name to rerun.

Checks before comparing values:

- Source revision and clean tracked source/header/GGML files.
- Build information and `CMAKE_HOME_DIRECTORY` from the **selected** library
  build, not an unrelated default build directory.
- Hashes of source, exporter, libraries and outputs. Actual loaded libraries
  must resolve to the recorded selected files and match their hashes.
  `LD_PRELOAD` is removed from reference subprocesses.
- Full streaming SHA-256/size identities of each distinct GGUF input, with
  bounded 8 MiB read buffers. One actual file shared by both engines is hashed
  once. Size/mtime/inode/device are checked for changes during the run.
- Identical token IDs, positions, shape, vocabulary and finite FP32 values.

These are provenance checks for the selected local build, not a claim of an
independently reproducible bit-for-bit rebuild of all vendor libraries.

Metrics include maximum/RMS/relative-L2/centered errors, actual/reference
values at the largest absolute discrepancy, per-position cosine similarity,
top-five sets, top-token IDs and margins. Strict pass/fail is separate from
greedy agreement and characterization.

### `benchmarks/fixture_0106q_qwen_reference.py`

A deterministic, bounded pair with 73 tensors:

| Quantity | Fixture |
|---|---|
| Model width / layers | 32 / 4 |
| Hybrid pattern | three recurrent, one full-attention |
| Attention Q / KV heads | 4 / 2 |
| Recurrent K / V heads | 2 / 4 |
| Key/value dimensions | 32 / 32 |
| Rotary prefix / convolution width | 8 / 4 |
| Experts / selected / expert width | 4 / 2 / 32 |
| Shared width / vocabulary | 64 / 260 |
| Original / reference file bytes | 460,320 / 653,536 |

Both recurrent head dimensions are equal, as required by this llama.cpp
architecture; the earlier Python-only unequal-dimension fixture is not reused.
Weights are well scaled, effective GGUF norm gamma is direct, and `ssm_a` is
already negative transformed decay. Routed banks and embedding/head use Q8_0
in the original; the reference file contains only exact-equivalent F32 values.
Tests verify every tensor against `gguf` and an additional literal Q8 decoder.

Inputs are explicit `[256,32,97,10,256]`, corresponding to `ab a\nab` in the
fixture's complete byte-level vocabulary. No large checkpoint conversion is
performed, and the production adapter is not modified to accept dense expert banks.

## 3. Reference and model identity

Reference checkout:
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp`

Revision: `171e8846b4af9766c354064cb776cb34a50f053f`.
Selected build: `build-cuda`; generated build number 11192, GNU 13.3.0.
Actual loaded libllama/GGML/CPU/CUDA library paths and hashes are retained in
both final run reports. No vendor source or library was modified.

Installed checkpoint:
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`

- Size: **49,608,478,720 bytes**.
- Full SHA-256:
  `4bb93f0a0221ef4ff963ca9094df629c8dfdfabc3b4fdd85c1a2e4c0624fce36`.
- Matched tokens: **`[750,912,2877,11]`**, no implicit special tokens.
- Explicit context 16, reference CPU threads two; all vocabulary logits exported.

## 4. Results

### Strict equivalent-F32 fixture: passed

| Comparison | Maximum absolute error | RMS error | Maximum tolerance ratio |
|---|---:|---:|---:|
| MyLocal full vs independent full | 2.682209e-7 | 6.782727e-8 | 0.008268 |
| MyLocal incremental vs independent incremental | 1.937151e-7 | 5.370185e-8 | 0.006053 |
| MyLocal full vs MyLocal incremental | 2.384186e-7 | 5.883915e-8 | 0.006469 |
| Independent full vs independent incremental | 2.160668e-7 | 5.134189e-8 | 0.007300 |

All strict comparisons passed. Greedy IDs agree at all five positions:
`[67,174,143,54,47]`. This independently verifies the tested hybrid graph,
normalization/residual composition, routing, shared/routed FFN weighting and
state chronology. It does not prove every installed-model geometry/format.

### Installed quantized model: characterized, strict parity unresolved

| Comparison | Maximum absolute error | RMS error | Relative L2 error |
|---|---:|---:|---:|
| MyLocal full vs independent full | 0.872999907 | 0.090815733 | 0.037850775 |
| MyLocal incremental vs independent incremental | 0.540585518 | 0.082342436 | 0.034188436 |
| MyLocal full vs MyLocal incremental | 9.536743e-6 | 1.597028e-6 | 6.622148e-7 |
| Independent full vs independent incremental | 0.741590738 | 0.092713315 | 0.038494408 |

All comparisons choose the same top token at each position:
**`[220,2877,11,293]`**. Cross-engine full-sequence top-five sets overlap 5/5
at every position; their internal ordering can differ. Per-row full-sequence
cosine similarities range from approximately 0.99913 to 0.99981.

Largest full-sequence discrepancy at `[position=0,vocab=474]`:

- MyLocal FP32 value: **3.2872507572174072**.
- Independent quantized-CPU value: **2.4142508506774902**.
- Absolute / relative error: **0.872999906539917 / 0.3616028162**.
- Strict tolerance: **not assigned**, because arithmetic precision differs.
- Finite-difference step: **not applicable**.

This is not described as ordinary tiny FP32 roundoff. The independent runtime's
own batching discrepancy is also substantial. Greedy agreement is useful
behavioral evidence, but is not a substitute for strict numerical agreement.

### Small quantized-activation control

The same independent exporter was also run on the tiny **packed** original,
not just its all-F32 equivalent:

| Comparison | Maximum absolute error | RMS error |
|---|---:|---:|
| MyLocal FP32-activation path vs packed CPU reference | 0.003705982 | 0.001045121 |
| Independent F32 vs independent packed CPU reference | 0.003705807 | 0.001045111 |

Greedy IDs still agree. The nearly identical shifts show that reference
arithmetic differences exist even when weights and the graph are known to
match. This supports the precision-mismatch explanation, but does **not**
isolate every cause of the installed model's larger discrepancy.

No production defect was established and no production edit was justified.
Remaining explanations are quantized activations, reference batch-dependent
arithmetic/routing sensitivity, or an installed-configuration-specific mismatch
not exposed by the small fixture. The next useful experiment is a matched-token
per-layer hidden/router-logit/expert-ID trace to locate the first divergence,
not speculative changes to norms, routing or kernels.

## 5. Tests and reproduction

- Focused fixture/comparator/provenance tests, with optional linked integration:
  **26 passed in 2.00 s**.
- Broad CPU external/native/MoE/generation regression including linked integration:
  **1,030 passed, 78 skipped in 10.35 s**.
- C++ exporter compiled and both final independent runs completed successfully.
- One new fixture causality test initially demanded bitwise FP32 equality after
  changing suffix routing. The observed maximum difference was 1.27e-7; changed
  expert GEMM row counts need not preserve bitwise reduction results. That local
  test now uses rtol=2e-6, atol=3e-7. Existing tests and the independent strict
  reference threshold were not weakened; no production correction was made.
- No MyLocal CUDA inference or production performance benchmark was added in Q;
  the previous 0106P CPU/CUDA verified state is unchanged.

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_0106q_reference.py \
  --llama-root /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp \
  --output benchmarks/results/0106q_strict_fixture_new.json

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_0106q_reference.py \
  --llama-root /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --token-ids 750 912 2877 11 --context 16 --threads 2 \
  --output benchmarks/results/0106q_real_reference_new.json
```

Tests: `tests/test_0106q_reference_fixture.py`,
`test_0106q_reference_comparison.py`, `test_0106q_reference_provenance.py`.
Set `MYLOCAL_LLAMA_REFERENCE_ROOT` to opt into compiled integration; otherwise
that test skips rather than downloading a reference runtime. Selected library
directories must retain matching CMake/build-info provenance.

Final artifacts: `benchmarks/results/0106q_strict_fixture_final.json`,
`0106q_real_reference_final.json`, and
`0106q_quantized_fixture_control_final/control.json`; accompanying directories retain
inputs, logs, binaries, logits and provenance. The durable summary/checksums are
in `0106q_validation.json`. These measurements are correctness diagnostics,
not model throughput or a Strata performance comparison.
