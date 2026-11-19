# 0261 — Exact-contract AVX2 Q2_0 CPU dot

## Objective and reference findings

Eliminate scalar2bit extraction/multiplication in the fallback down projection. Directly reread pinned Strata `712fec6b4297c9cfd32ab43daaa2979a9aa01edf` files `include/strata/kernels/cpu/{native_expert,expert}.hpp`, `src/kernels/cpu/{native_expert,q2_avx2}.cpp`. Strata's customActQ usesFP32scales and half-away rounding; it is not interchangeable with pinnedGGMLQ8_0. Do not silently substitute that representation.

Inspected pinnedGGML `171e8846b4af9766c354064cb776cb34a50f053f` CPU `quants.c`: Q2_0_generic uses64weights/block,18bytes, codes{0,1,2,3}offset-1, two32valueQ8_0blocks with separate halfscales, integer dots, then ordered scalarFP32 accumulation. Disassembly of the actually selected library's exportedQ2dot confirms scalarimul and separatevmulss/vaddss, notFMA. RTXworkstation CPU supportsAVX2/F16C but notAVX512.

## Files and interfaces

- New original `mini_llm/ops/q2_cpu_simd.h`: SIMD code unpacking, exact unsignedcode×signedactivation integerdot with offset correction, scalar float ordering retained.
- `ggml_packed_cpu.cpp`: includes kernel; existing pooled and synchronous schedulers unchanged.
- `ggml_packed_cpu.py`: `GGMLPackedCPUExecutor(..., q2_simd=True)` or explicit `MINI_LLM_GGML_Q2_SIMD=1`; opt-in only. Provenance records selectedQ2dot. Native build hash includesheader; contraction disabled; CPU feature gate rejects unsupported systems before selecting wideISAfunction.
- `pyproject.toml`: package newheader andGGMLbridgeCpp with existingnative resources.
- `tests/test_0261_q2_cpu_simd.py`.

The native dot uses the same ABI asGGML. OnlyQ2_0 dot pointer changes; originalGGMLactivation quantizer is retained. No fullFP32weight expansion, new activation format orGGMLsource/build modification. The exported dot is internal to validated dispatch, not an arbitrary-layout publicAPI. Otherformats and unsupported/defaultpaths retainGGML.

Integer pair products cannot saturate16bits, including activation-128; offset is subtracted inintegers before scaling. Half conversion isexactF16C; twochunk scales are not collapsed into one64value scale. Scalarblock accumulation order remains the reference order. This is an originalSIMD kernel motivated by Strata's algorithm, not a port of its pool/plugin/ActQ runtime.

## Validation

Focused0261+0259+0256run:71passed3skipped. Includes21linearshape/seedcases,600directdotcases with arbitrarysignedQ8codes/finitehalfscales, zero inputs, negative/subnormal scales, oddblockcounts, mixedIQ3_S/Q2_0experts and pooledtop10. Outputs are **bit-exact** against the retained pinnedGGMLpath in these tests. No tolerance was loosened. BroaderCPU1875passed962skipped and selectedCuPy147passed3skipped withSIMDenabled.

Warmed syntheticD2560/F640,100samples/operator: Q2downmatrix0.36149ms ->0.08735ms (**4.138x**); completeIQ3_S/IQ3_S/Q2_0expert0.58926ms ->0.31598ms (**1.865x**). Artifact `0261_operator_timing_v1.json`. Warm operator timing is not a sustained model claim.

## Real model measurement

`diagnostics_0261_frozen_simd_v1/` repeats0260frozen settings: samepromptIDs/sampler, boundedmemory/time,15workers,6GiBcache. Both supervisors completed0. Warmup128 and response324tokenIDs match0260frozen exactly. Longresponse323forwards/30.12780s = **10.72100forwards/s**, versus10.31149 withoutSIMD in a separate process. Observed improvement~4%; onepair, not confidence bounds. No full-model logit/state capture was performed by this sampler benchmark; tokenIDequality is not called equivalence.

SampledRssAnon824,172,544bytes/RssFile55,092,994,048bytes; GPUallocatorused12,148,603,904/reserved12,334,209,024bytes, not globalVRAMpeak. No new modelweightallocation. Allpriorartifacts preserved.

## Conclusion / dependency

ScalarQ2_0 is a genuine avoidable cost but not the dominant remaining walltime. Fasteroperator does not imply proportionate modelgain. Strata's customactivation arithmetic remains unported. Next0262 tests the actual shared/routed overlap gap; nextlarge structural work is a validated native execution plan/capturedGPUsegments with stable runtime metadata, not further scalar tuning.50t/s remains unmet.
