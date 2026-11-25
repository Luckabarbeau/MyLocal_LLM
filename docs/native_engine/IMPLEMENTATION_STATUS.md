# Native engine implementation status

Updated 2026-11-24. The user-supplied WP0–WP11 specification is authoritative. Existing Python Qwen and Model B/C infrastructure are unchanged by native-engine work. No production backend selection changed. The experimental engine now executes the full native model, but strict whole-model numerical acceptance remains OPEN. All current GPU validation uses physical GPU 1 only, exposed as device 0 through `CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631`; GPU 0 and running training jobs must not be used or stopped.

## Work packages

| Package | Status | Evidence / remaining gate |
|---|---|---|
| WP0 Reference baseline | in progress | `native_reference_manifest.json`, `REUSE_AND_BASELINE.md`, `WP0_QSA_reference_run.json`; bounded CPU QSA reference tests 11 passed/11 deselected. Full checkpoint payload hashes completed in `WP0_CHECKPOINT_HASHES.json` (83,617,662,656 bytes, stable file identities). Matched model fixtures, effective baseline configs and new throughput/resource runs remain outstanding. Authoritative checkpoint revision and shipped Strata build provenance are external dependencies. |
| WP1 Runtime foundation | completed | `mini_llm/native_cpp/{CMakeLists.txt,runtime.h,runtime.cpp,README.md}`, `mini_llm/native/__init__.py`, `tests/test_native_engine_runtime.py`; C++20 CUDA-runtime library builds; all 8 foundation tests pass within latest combined run. Synthetic token copies only. Allocation accounting covers requested session buffers, not CUDA context/allocator overhead; no claim of leak-detector validation. |
| WP2 Model loading | completed | All 1,224 descriptors/13 checkpoint formats match Python; immutable mmap leases, exact payload hashes, bounded dense uploads, packed expert sources, architecture/schema validation and whole-model binding are implemented. `native/decoder.py` binds every layer with budget preflight; the actual 48-layer model has executed and reset/replayed. This closes the loader/descriptor/resource gate only, NOT strict whole-model numerical parity (WP7). Standalone C++ metadata interpretation is not duplicated. |
| WP3 Native FP32 QSA | completed | Native complete-layer hidden->projections/norm/RoPE/pool->GPU token selection->GQA/gate->output->transactional state commit. Both policies; synthetic boundary/state tests through16K; all12checkpoint layers×2policies×9steps pass. 14 actual layer3 long-context benchmark cases validated vs unchanged CuPy. Details `WP3_COMPLETE_QSA_VALIDATION.md` and JSON; component acceptance, not full-model throughput. |
| WP4 Recurrent/HC/PLE | completed | Native HC, recurrent/GDN and sparse leased-table PLE ownership/math/reset/checkpoint validation; direct HC read->QSA/GDN->HCwrite chaining and PLE-first order pass. CUDA memcheck clean for these focused suites. See `WP4_HC_INTEGRATION.md`, `WP4_RECURRENT.md`, `WP4_PLE.md`, `WP4_NATIVE_CHAINING.md`. FFN/MoE and full decoder remainWP5/WP7. |
| WP5 Native MoE | completed | Native mixed-placement owner, full routing/shared branch, ordered CPU/GPU combination and HC FFN chaining pass synthetic and actual-checkpoint CPU/GPU/mixed checks. Explicit `cpu_ggml_q8_gpu_reference_fp32` policy, leased mmap sources and static placement; no Python expert dispatch. See `WP5_MOE.md`, chaining validation/benchmark JSON. Current GGML unload/reload abort fixed at native loader with explicit process-resident GGML mapping; worker arithmetic unchanged (`WP5_GGML_LIBRARY_LIFETIME.md`). |
| WP6 Paged/quantized KV | in progress | GPU page tables, bounded host backing, stable GPU slots/generations, native compact misses, chunked selected gather and explicit FP32/INT8 policies integrated into complete QSA. Populated single-layer 64K/128K/262K gates now pass both formats and selection policies; snapshot fixtures use actual layer-3 weights, NOT a 262K-token model trajectory. Quantized model quality, full-model long conversations and fault-injection acceptance remain open. See `WP6_FULL_CAPACITY.md` and evidence JSON. |
| WP7 Complete decoder | blocked | Implemented native 48-layer execution, embedding, PLE, HC, GDN/QSA, MoE and vocabulary projection, retained ownership, failed-session quarantine and reset/replay. Strict synthetic and actual full-model comparisons FAIL. Identical-input diagnostics show correct MoE/GDN execution but upstream FP32 roundoff crosses CPU Q8 thresholds, amplifying into recurrent histories. No supported production-equation defect or accepted quality envelope has been established; no tolerances changed. See `WP7_DECODER.md`, `WP7_DIVERGENCE.md`, `WP7_FULL_MODEL_DIAGNOSTICS.json`. |
| WP8 Generation/serving | in progress | Experimental native greedy/stochastic sampler, resident-logit path, C++ generation worker, bounded token queue/backpressure, cancellation and drain/join are implemented and independently tested. Native replay is not Python-model parity. Production HTTP integration, prefix caching and acceptance remain gated on WP7; no serving behavior changed. See `WP8_SAMPLER.md`, `WP8_SAMPLER_OPTIMIZATION.md`, `WP8_GENERATION.md`. |
| WP9 Production validation | pending | No new native model throughput claim. |
| WP10 Native prefill | pending | Existing prefill components retained, not completion of new engine gate. |
| WP11 Speculation | pending | No speculative arithmetic introduced. |

## Verified state

Build directory: `/tmp/mini-llm-wp1-build` (ephemeral, rebuild using README).

```
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/mini-llm-wp1-build
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_gguf.py tests/test_native_engine_runtime.py
```

Prior verified combined execution: **28 passed in 0.54s**, no skips (preserved). CUDA toolkit 13.4.92 / GCC 13.3. Three misleading-indentation compiler warnings are documented, not test failures.

Actual checkpoint header parity: **passed in 4.95s**; 1,224 descriptors, 83,606,638,080 tensor bytes, 83,617,662,656 file bytes; unchanged stat identities. This is header/descriptor parity, not tensor payload hash or decoded numerical parity.

Bounded existing CPU QSA selection/reference tests: **11 passed, 11 deselected**, no skips; exact invocation and limits are in `WP0_QSA_reference_run.json`. Historical benchmark numbers in the reuse inventory were not rerun and are not new performance results.

## WP2 bounded required-tensor/layer milestone

`GGUFStore.validate_qwen4exp()` refreshes metadata/descriptors from the live native owner under its lock, adapts storage format identity using the existing GGUF registry, and calls unchanged `Qwen4ExpGGUFIndex` and `qwen4exp_weights.inspect_headers`. Those remain the sole architecture/schema authority: recurrent/QSA schedule, PLE placement/configuration, HC grouped-norm aliases, routing restrictions, required names/shapes, dense controls and packed experts are not independently redefined in C++. The resolved exact schema is checked against the native owner's descriptors by `ml_gguf_validate_schema`, including missing/extra names, duplicate requirements, ranks, shapes and type identity. The generic C ABI is not independently sufficient to certify Qwen architecture; callers must use the authoritative binding. No validated-model handle, payload lease, payload pointer, GPU allocation or upload is added.

Exact latest verification:

```
cmake --build /tmp/mini-llm-wp1-build
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_qwen_contract.py tests/test_native_engine_gguf.py tests/test_native_engine_runtime.py
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m docs.native_engine.probe_qwen_contract /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf docs/native_engine/WP2_QWEN_CONTRACT.json
```

Build completed. Latest combined tests: **47 passed in 0.70s**, no skips: prior 28 plus 19 contract cases. Synthetic cases reuse `TinyStore` from the existing Python model tests and cover recurrent/QSA schemas, flat/grouped HC gamma aliases, closed owners, detached-snapshot tampering, architecture/missing/extra/layer-count/compression/routing/HC/PLE/control/embedding rejection, and direct native schema failures. Initial execution was 46 passed / 1 failed because a changed quantization fixture retained the old payload size and overlapped ranges; one localized fixture-size correction was made, then the exact combined retest passed. No production correction was needed.

Actual checkpoint probe: **passed in 0.459s**, 1,224 required tensors, width 2560, vocabulary 248320, context 262144, HC rank 320, 48 layers (36 recurrent, QSA `3,7,...,47`), PLE layer 1. Shard stat identities unchanged. Evidence includes schema/source/library hashes in `WP2_QWEN_CONTRACT.json`. This is descriptor/architecture compatibility only, not payload hashing, decoded numerical parity, or independent end-to-end model validation. Prior descriptor parity and QSA reference evidence below/above remain historical verified states, not newly rerun claims.

## Next actionable task

WP6 populated-capacity gate is now verified; preserve its current results and extend only remaining quality/full-model long-context gates. WP7 strict acceptance is the critical blocker: inspect the recorded full-model identical-input/Q8-threshold evidence before any production numerical edit; do not rederive it or relax tolerances solely to make tests pass. A reference-arithmetic compatibility change needs a specific supported incorrect operation, or an explicitly accepted measured quality envelope for the documented mixed arithmetic. HTTP production integration and accepted performance benchmarks stay gated. Native sampler/worker can be exercised independently without claiming model acceptance. User explicitly removed the old one-correction/retest stop limit; continue evidence-supported localized debugging, not speculative rewrites.

## Continued execution results

- Payload leases: existing 47 plus 24 new cases, **71 passed**. Native leases keep maps alive independently of store handles; bounded reads expose no raw pointers.
- Exact-storage CUDA uploads: **110 passed**, including all 20 supported storage descriptor types and device 1; no implicit weight decode or activation conversion. Pageable mmap source transfers may block host return; no asynchronous-host-return promise.
- GPU QSA scoring and selection: **170 passed**. Both `ggml_cpu` and `strata` token membership conventions tested; boundary positions through 16384 and explicit GPU diagnostics. Serial reference heap is correct but slow; no model throughput improvement claimed (`WP3_SELECTION.md`, raw benchmark JSON).
- Selected QSA attention: **247 passed, 1 optional skipped**. Optional benchmark also ran separately and passed. A measured redundant full-history finite scan was moved from every head into one shared parallel pass without removing checks: n=16384 ggml_cpu attention-only median 15.338458 -> 0.626294 ms, selection+attention 26.591709 -> 6.942007 ms. These are separate-run component measurements, not full-model speedups (`WP3_ATTENTION.md`, before/after JSON).
- Native QSA output projection: **257 passed, 2 optional skipped**; actual-checkpoint projection validation separately passed all 12 QSA weights (Q4_K layer 3, Q6_K remaining layers). Largest observed absolute output error 1.5497207641601562e-6. See `WP3_PROJECTION.md` for exact commands, tolerances, budgets and timings.
- Projected-input norm/RoPE/pool frontend: latest combined **337 passed, 2 optional skipped**; focused **80 passed**. Converted gamma used directly; pooling preserves left-to-right raw FP32 additions before normalization/RoPE. Gate/n/pool flags are exact; normalized/rotated outputs use explicit componentwise FP32 error bounds. Maximum observed rotated discrepancy 0.000641822815 at n=16384; this is not bitwise reference parity or full-layer acceptance (`WP3_FRONTEND.md`). No tolerance relaxation followed numerical failures.

Latest combined command:

```
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_*.py
```

- Hidden-to-frontend packed projections: prior verified state **353 passed, 3 optional skipped**. Added Q5_K/IQ4_XS based on actual tensor inventory, no activation quantization. All 12 actual layer frontend boundary comparisons passed; peak tracked payload ~29.9 MiB. See `WP3_HIDDEN_FRONTEND.md`.
- Complete state-owning QSA layer: build passed; most recent focused run **16 passed, 1 failed in 8.00s**. All 16 trajectory cases passed both executions, including n=4/8 pooling, 2051/2052/2053 and reaching full 4K/16K capacity, both policies, exact controlled selected IDs and output/state comparisons. One localized diagnostic test correction succeeded; retest then hit invalid `r.engine(budget_bytes=1)` fixture before cross-engine assertion. No second correction was made. Prior 353-pass state predates these integration edits and is NOT verification of current broader regressions. Details and exact failing command: `WP3_COMPLETE_QSA.md`. Current full-layer optional checkpoint parity, later admission tests and benchmarks remain unexecuted. `git diff --check` passed after integration.

## Verification resumed after user correction-limit override

The user explicitly removed the old fixed correction-count stop rule. The test fixture budget changed from1to12bytes; no production admission rule changed. Focused QSA layer suite **19passed,1optional skipped**; actual layer3checkpoint9steps **1passed**, maxabs2.2351741790771484e-7; broad native **372passed,4optional skipped**.

Further WP3 gates passed: actual12layers×bothpolicies×9steps **24passed**; full native **396passed,3optional skipped**. Maximum actualcheckpoint output error3.5762787e-7. Reproducible benchmark14cases(layer3,n128/2051/2052/2053/4096/8192/16384×bothpolicies) validates outputs, states, exactIDs with fresh matched history against unchangedCuPy. Maxerror1.3783574e-7; measured combined native+CuPyreserved payload424.61MiB. See `WP3_COMPLETE_QSA_VALIDATION.md`, `WP3_COMPLETE_QSA_VALIDATION.json`, `WP3_COMPLETE_QSA_BENCHMARK.json`, `benchmarks/benchmark_native_qsa_layer.py`. No complete-model speedup claim.

WP4 HC integration **52focusedpassed**, full **448passed,3skipped** withQSAcheckpoint; HCmemcheck0errors. Recurrentfullstep **24focusedpassed**, full **472passed,3skipped**, recurrentmemcheck0errors; actual layer0steps16+reset+4 maxoutputerror3.54e-8. PLEruntime full **480passed,3skipped**, PLEmemcheck6passed0errors; actual32stepsreset/replay maxoutputerror1.19e-7,maxhistoryerror3.34e-6,peak63.89MiB. NativeHCbranchchaining full **495passed,3skipped**, focused15memcheckpassed0errors. No full decoder claim.

WP5 CPU caller: existingbridge declarations extracted tosharedheader withoutworkerarithmeticchange; cache digest includesheader. Explicitdynamiclibrariesandtraitsvalidated. Caller+0256/0259/0271regressions **60passed,3skipped**; full native latest **505passed,3skipped** withQSAcheckpoint/PLECUDAflags. NativeCPUarithmetic is explicitlyggml_q8,notrefFP32. See `WP5_CPU_POOL.md`. GPUstandaloneprimitives **451comparisons**, **315memcheckcomparisons0errors**, all7routedtriples/8sharedcombinations; routedterm/orderedcombine bitwiseequalexistingPythonwarporacle. See `WP5_GPU_PRIMITIVES.md`; runtimeintegrationnotyetaccepted.

All optional benchmark/checkpoint executions and raw artifact limitations are described in their milestone documents. Earlier failure paragraphs are historical evidence, superseded by the passing gates above. WP3/WP4 component acceptance does not certify full decoder or production performance.

## Latest GPU1 continuation — 2026-11-24

### WP6 populated-capacity verification

`tests/test_native_engine_qsa_full_capacity.py`: **6 passed in 22.61s**. Four cases cover FP32/INT8 × ggml_cpu/strata, each progressively populated to 65,536, 131,072 and 262,144 positions. All execute the final legal step and reject overflow without changing state. History is a controlled prepared snapshot using actual layer-3 projection weights, not 262K sequential checkpoint tokens. Full-capacity tracked layer device bytes: 85,186,704 FP32 / 78,960,784 INT8; final host backing+metadata: 1,073,776,728 / 276,858,968 bytes. One physical cache slot safely handles 3–6 selected historical pages per measured step. INT8 decoded-oracle errors (~1e-7) are separate from its measured ~1e-3 FP32 storage-policy difference. Raw evidence: `logs/wp6_full_capacity_gpu1/`; summary `WP6_FULL_CAPACITY.md` and `WP6_FULL_CAPACITY_EVIDENCE.json`. Reduced-capacity memcheck: **6 passed, zero errors**. Capture hashes identify that build, not subsequent rebuilds.

### WP8 independent sampler and generation

Sampler: **64 focused tests passed**, including original47 unchanged; optimization pre/post IDs, probabilities, CDF and metadata compare bitwise. Top-k=50 measured sampler wall median improved 197.598 -> 2.303ms on GPU1 under concurrent training; not a model speedup. Full-vocabulary stochastic CDF order remains unchanged; no approximation. Sampler memcheck: **8 passed, zero errors**. `WP8_SAMPLER_OPTIMIZATION.md` contains raw before/after and bitwise evidence.

Native generation worker: **12 focused tests passed**; bounded native queue, backpressure, atomic cancellation independent of the registry/GPU lock, exclusive session ownership, EOS/capacity/max termination and drain/join lifecycle. Events distinguish committed position from pending unconsumed sampled token. Native manual-loop replay is bitwise, NOT Python-model parity. Focused memcheck **4 passed, zero errors**. No production server enablement (`WP8_GENERATION.md`).

### Correctness fixes found by combined regressions

1. MoE checkpoint memory test counted unused CuPy blocks retained by earlier tests. Added synchronize/GC/free-unused-pool at the bounded test boundary; no runtime budget or tolerance change. Same WP4/WP5 combined command: initially9 memory-assertion failures/153 passes, then **162 passed in39.41s**.
2. Real native library lifetime defect: last-owner GGML `dlclose` followed by reload aborted in its process-global terminate-handler initializer. Native loader now requests explicit `RTLD_NODELETE` for GGML only, with balanced per-owner `dlclose`; worker, weights, jobs, streams and engine cleanup unchanged. Original subprocess: exit134 -> exit0. Four new fresh-child regressions cover native reopen, CDLL reload, Python pools and MoE owners. GGML code/library dependencies remain mapped until process exit; this deliberate exception is documented in `WP5_GGML_LIBRARY_LIFETIME.md`.

### Latest split regression results

All commands use the physical-GPU1 UUID mask, one visible CUDA device, BLAS/OMP threads1 and disabled pytest plugin autoload. Build and `git diff --check` passed.

| Group | Backend oracle | Result |
|---|---|---|
| Runtime/GGUF/schema/payload/upload/CPU experts, after loader fix | CuPy | **123 passed, 1 skipped** (two-device restoration cannot run with GPU0 hidden) |
| HC/GDN/PLE/chaining/MoE, checkpoint + PLE opt-ins | CuPy | **162 passed** |
| QSA selection/attention/frontend/projection/hidden frontend, all opt-ins | CuPy | **246 passed** |
| QSA layer/paged layer/GPU pages/host-INT8 pages, checkpoint opt-in | NumPy; native computations still CUDA GPU1 | **93 passed in64.10s** |
| QSA all12 checkpoint layers × both policies, additional separate run | CuPy | **24 passed,19 deselected** |
| Sampler/optimization/generation/divergence/decoder safety, after loader fix | CuPy | **87 passed,2 explicitly deselected** |
| Fully populated QSA gate, separate run | NumPy; native CUDA GPU1 | **6 passed** |

The two explicitly deselected strict tests are `test_synthetic_all48_reference_replay_lifetime` and `test_actual_all48_two_steps_reset`; their recorded failures are NOT fixed/hidden/xfail or counted as passes. There is no green complete-suite or WP7 acceptance claim. The CuPy long-history group timed out at120s during Python-oracle preparation; the subsequent NumPy-oracle93-case command and separate24-case CuPy checkpoint command passed. Earlier broad/sanitizer timeout attempts remain incomplete, not passes. JUnit outputs for parent split runs are `/tmp/native-gpu1-*-final.xml` (ephemeral).

### Remaining critical blocker

Strict full-model gate remains unchanged and failing. Retained two-token actual full-model diagnostics (`WP7_FULL_MODEL_DIAGNOSTICS.json`) show final-logit relative L2 errors1.16%/3.78%, maxabs0.15479/0.41911; top1 agrees and top10 overlap10/10 for these two samples only. Same-input CPU experts are bit-identical, dense HC starts with4.768e-7 roundoff, Q8_K crosses a28->29 threshold, and later recurrent-state differences follow prior input history. No specific incorrect production equation was established. These measurements do not establish model quality. Do not enable HTTP production or declare WP7/WP9 accepted; further arithmetic-compatibility work requires a supported defect or explicit quality acceptance policy, not speculative precision changes. GPU1 currently also hosts training; full-model 10GiB+headroom tests must preflight available memory and must not borrow GPU0.

## Preservation

Pre-existing changes in `mini_llm/heterogeneous_production.py`, its `.bak`, and `strata_launch_bench_capture.sh` were left untouched. No agent-created commits. The repository HEAD changed externally during execution; existing source work was preserved. No protected runtime data, checkpoint files, or training contracts were modified.
