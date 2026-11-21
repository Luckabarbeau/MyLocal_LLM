# 0335 — Identical-input CPU expert comparison

## Objective and outcome

Test whether a Strata CPU arithmetic implementation is materially faster for the **same expert bytes, FP32 inputs, selected expert IDs, thread count and scheduler**. No production model mathematics, precision defaults, kernels or cache policy changed.

**Confirmed opportunity:** the pinned Strata AVX2/AVX-VNNI Q2_0 row kernel is 1.32–1.55× faster in the repeated projection experiment **even after importing the exact local GGML Q8_0 activation values and FP16-rounded scales**. This does not require adopting a different activation quantizer. It is a candidate for opt-in integration, not an achieved full-model speedup.

**Rejected broad claim:** swapping to pinned Strata GGML singleton traits is not a general CPU-backend upgrade. All 24 whole-expert comparisons are bit-exact. The local IQ3_S specialization is often faster, while IQ2_S and IQ3_XXS cases are broadly similar. Pinned generic GGML Q2_0 is slower than local SIMD, but it is **not Strata's optimized production Q2 path**.

Target 50 forwards/s remains unmet; prior hot aggregate 47.17321/s is unchanged and was not rerun in this patch.

## Files and diagnostic interfaces

- `scripts/build_0335_strata_q2.py`: isolated, content-addressed adapter build with archive and dependency verification, compiler/flags/source/library hashes, and timeout-bounded native smoke subprocesses.
- `scripts/native_0335_strata_q2.cpp`: narrow C ABI around **unchanged** pinned `q2_avx2.cpp`; bounded, 64-byte-aligned ActQ ownership; CPUID/OS-supported AVX2/VNNI dispatch. Quantization, raw GGUF row ranges, exported values/scales, and Q8_0 import are separate endpoints. Errors invalidate the activation state. Bit-plane mode is excluded from Q8_0 import.
- `scripts/native_0335_row_team.cpp`: neutral persistent native team; participants−1 workers plus host; approximately three tasks/participant. Identical scheduler for GGML row dots and Strata row ranges; no Python callbacks. Timed loops run natively; inputs stay alive through synchronous completion.
- `scripts/probe_0335_cpu_expert_ab.py`: actual checkpoint expert slices, retained independent-reference activation captures, alternating-order trials, independent bounded FP64 Q2 block baseline, versioned output. Full forward/generation is never constructed.
- `tests/test_0335_cpu_expert_ab.py`: metric shape/nonfinite/denominator behavior and alternating two/three-arm trial bookkeeping.

Only diagnostic sources, artifacts and documentation were added. Existing uncommitted production work was preserved.

## Reference provenance

The existing external Strata checkout is `c77572806996349b49995e4f98d41b5294d36a21`, **not** the required pin. It was not changed or used as the numerical source.

Fetched a separate source archive, not model weights:

- URL: `https://codeload.github.com/Niko1221/Strata/tar.gz/712fec6b4297c9cfd32ab43daaa2979a9aa01edf`
- Revision: `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`
- Archive SHA256: `26984c09b4d42476a8a3dee72d227d4583a07d0640572a1cb7e25b4f4d27b4f6`
- Separate location: `reference_0335/`; retained MIT license at `reference_0335/source/LICENSE`, copyright Niko1221 and Strata contributors. GGML's license/attribution remains in the pinned third-party subtree.

Built pinned vendored GGML CPU/base only, Release, `GGML_NATIVE=ON`, `GGML_OPENMP=OFF`, shared libraries; no engine/CUDA backend. This is a workstation-native source comparison, **not a reconstruction of the installed portable Strata engine build**. CMake cache and compile commands remain under `reference_0335/ggml-build/`. Actual pinned GGML library hashes are recorded in both benchmark result files. Its 0.24 dependent library SONAMEs differ from the local 0.25 libraries.

Q2 adapter compiles pinned `src/kernels/cpu/q2_avx2.cpp` unchanged with `-mavx2 -mfma -mf16c -DSTRATA_AVXVNNI=1`. Baseline-ISA adapter performs feature checks before dispatch. No AVX512 kernel is run. Content-addressed manifests under `reference_0335/build/` record the complete Q2 dependency/flags/compiler provenance. The archive digest verifies downloaded bytes, not an independent signed Git attestation.

Source inspection confirms singleton IQ paths normally use GGML below the multi-token threshold. Optional `STRATA_IQ3S_MT1`, all-exact-one and grouped kernels are not implicitly benchmarked here. Do not label the traits leg complete installed-engine arithmetic parity.

## Experimental design

Inputs and IDs come from `0224_isolated_moe_v1.json` and `diagnostics_0224_v1/p0_l{layer}/`, not newly generated tokens. Tested:

| Layer | Gate/up | Down |
|---:|---|---|
| 0 | IQ3_XXS | IQ4_NL |
| 1 | IQ2_S | Q2_0 |
| 2 | IQ2_S | IQ4_NL |
| 17 | IQ3_S | IQ4_NL |

Each has 3, 6 and 10 selected experts, 16 and 20 participants, identical unpinned allowed CPUs 0–23, and bounded warm packed weights. The whole-expert arms share the **same local native row pool**, registration, 50µs spin, task factor3, shared input/hidden quantization, SwiGLU and output buffer contracts. Dot/quantizer function pointers are captured during idle registration, then restored before execution. Outputs are unweighted; routing combination is outside this experiment.

- v1: 7 alternating trials ×30 calls after warmup.
- v2 repeat: 11 alternating trials ×60 calls; 24 whole-expert configurations and 6 three-arm Q2 projection configurations.
- Timing is warm steady operator cost, not cold storage or a complete engine throughput result. Raw samples and ranges are retained; no confidence interval claim.

The Q2 projection experiment uses the **same retained FP32 hidden vector from the first selected expert** for all selected down matrices, deliberately teacher-forced to isolate projection arithmetic. It does not approximate a full MoE forward: real per-expert hidden vectors differ. All three arms share the neutral persistent native team. Quantization/import, baseline decoding and Python setup are outside native row timing. The Q2 comparison includes row scheduling/completion overhead equally, not pure instruction timing.

## Whole-expert result

All 24 tested configurations are **bit-exact** between local and pinned Strata GGML traits, including the local IQ3_S gather and local Q2 SIMD compared with corresponding GGML dots. This is local-versus-pinned-GGML arithmetic, not native ActQ whole-expert equivalence.

Selected v2 median times with 20 participants:

| Layer | Expert count | Local ms | Pinned GGML traits ms |
|---:|---:|---:|---:|
| 0 | 3 | 0.085975 | 0.087323 |
| 0 | 10 | 0.253225 | 0.252922 |
| 1 | 3 | 0.085921 | 0.145185 |
| 1 | 10 | 0.238612 | 0.484766 |
| 17 | 6 | 0.167019 | 0.189493 |
| 17 | 10 | 0.272445 | 0.324149 |

Exact times/samples for all cases: `diagnostics_0335_cpu_ab_v2/result.json`. Replacing IQ dots wholesale is not supported by these results. Neither thread-count comparison nor differing format/layer cases should be extrapolated into an end-to-end rate.

## Q2_0: separate activation quantization from row kernel

Legacy Strata ActQ uses per-32-value **FP32 scales**, half-away rounding, and precomputed correction terms. Local GGML Q8_0 stores scales as FP16. One Q2_0 weight block is 64 values/18 bytes and spans two separate activation scales. The import arm copies the exact local quantized values and decoded FP16 scales into ActQ; it never requantizes the input.

Repeated v2 projection medians:

| Participants | Experts | Local Q8_0 row ms | Strata legacy ActQ ms | Strata same-Q8_0 row ms | Same-Q8 speedup |
|---:|---:|---:|---:|---:|---:|
| 16 | 3 | 0.036011 | 0.025533 | 0.025799 | 1.396× |
| 16 | 6 | 0.060785 | 0.040460 | 0.040334 | 1.507× |
| 16 | 10 | 0.095364 | 0.061937 | 0.061319 | 1.555× |
| 20 | 3 | 0.031624 | 0.023110 | 0.023881 | 1.324× |
| 20 | 6 | 0.050600 | 0.033468 | 0.033626 | 1.505× |
| 20 | 10 | 0.078478 | 0.053036 | 0.051366 | 1.528× |

Numerical results across expert counts:

- Legacy ActQ versus local output relative L2: **1.775e-4–1.833e-4** (~0.018%). This is predominantly a different activation representation, not a graph defect.
- Same-Q8 Strata versus local relative L2: **1.955e-7–2.003e-7**, maximum absolute difference **2.98e-8–3.35e-8**. Same dequantized mathematical input, different FP32 SIMD/FMA summation.
- Local versus its independent FP64 dequantized-Q8 baseline: **7.02e-8–7.13e-8**.
- Same-Q8 Strata versus that baseline: **1.82e-7–1.88e-7**. Local is slightly closer to this baseline; speed does not imply higher arithmetic accuracy.
- The JSON also reports maximum absolute differences, activation differences, legacy-own-baseline errors, effective packed GB/s and each path versus the dequantized-weight×original-FP32-input baseline.

Thus the row gain **survives removing the activation quantization difference**. This is evidence for optimizing row decoding/reduction, not changing model mathematics or tolerances. The first v1 legacy-Q2 test independently measured 1.30–1.62×, consistent with this repeat. A different row kernel can expose import/correction overhead when integrated; those costs are excluded here and must be measured next.

## Validation and memory

Guarded commands:

```sh
.venv/bin/python scripts/run_0290_isolated_benchmark.py \
  --memory-high-mib 4096 --memory-max-mib 6144 --reserve-gib 12 --timeout 150 \
  --output docs/qwen4exp/diagnostics_0335_cpu_ab_guard_v2 -- \
  .venv/bin/python scripts/probe_0335_cpu_expert_ab.py \
  --output docs/qwen4exp/diagnostics_0335_cpu_ab_v2 --trials 11 --iterations 60
```

Use fresh output directories for repetitions; existing results are preserved. Pinned GGML build prerequisite:

```sh
cmake -S docs/qwen4exp/reference_0335/source/third_party/strata-ggml/ggml \
  -B docs/qwen4exp/reference_0335/ggml-build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=ON
cmake --build docs/qwen4exp/reference_0335/ggml-build --target ggml-cpu -j 2
```

Build was executed inside a 4/6GiB isolated service, not directly in the desktop cgroup. No checkpoint download/copy/write. Only selected mmap expert slices and bounded one-matrix FP64 baselines are touched. v2 process peak RSS **1165.875 MiB**; sampled cgroup peak **1,048,317,952 bytes (~0.976GiB)**; swap0. RSS and cgroup charges have different accounting scopes. No model/CUDA execution or VRAM allocation is intended by this CPU-only probe.

Selected regression command inside isolation with `MINI_LLM_BACKEND=numpy` and explicit local `MINI_LLM_GGML_CPU_LIBRARY`:

```sh
.venv/bin/python -m pytest -q tests/test_0335_cpu_expert_ab.py \
  tests/test_0256_ggml_packed_cpu.py tests/test_0259_ggml_row_pool.py \
  tests/test_0261_q2_cpu_simd.py --junitxml docs/qwen4exp/0335_cpu_tests_v2.xml
```

**77 passed, 4 skipped.** The first invocation omitted the explicit library environment: 8 passed/73 skipped; preserved as v1 and not counted as native-backend validation.

Adapter smoke rerun passes automatic VNNI, forced plain AVX2, and opt-in bit-plane modes: 70 synthetic row checks/mode, plus35 imported-Q8 rows each in the first two. Numeric, width, padded-stride/range, malformed-buffer/state-invalidation, exact half/tie/zero cases pass. Imported-Q8 is deliberately rejected in bit-plane mode. Full results: `0335_adapter_tests_v1.json`. General synthetic absolute errors use contribution-scaled roundoff allowances, not bitwise requirements. Row-team synthetic dispatch/error/ownership checks also passed during construction; its role is diagnostic scheduling, not a production pool replacement.

## Limitations and next dependency

- No full-model gain or natural-conversation rate was measured. A faster Q2 down projection does not alone prove that the50/s target is reachable.
- These warm buffers are at most tens of MiB; full expert DRAM/NUMA/page-pressure behavior and sustained mixed-device contention are not characterized.
- This first experiment separates whole-expert cost and isolated Q2 down cost; detailed gate/up versus quantizer phase timing, installed-engine optional IQ dispatch and multi-token grouped kernels remain open.
- Production remains the existing full-top10, bounded-memory path. No tail omission, increased spin, extra pinned arena or checkpoint re-encoding.

**Next patch:** integrate a switchable Q2 row-range kernel preserving existing GGML Q8_0 codes/scales and using preallocated, owned correction scratch. Avoid per-row conversion/allocation or a second scheduling team. Validate complete experts and mixed MoE first, then rerun the retained longer forced trajectory under unchanged full-model memory guards, measuring exposed CPUwait rather than summed operator gain. Enable only after numerical and measured end-to-end gates. Legacy FP32-scale ActQ is a separately controlled arithmetic experiment, not required for the demonstrated speedup.
