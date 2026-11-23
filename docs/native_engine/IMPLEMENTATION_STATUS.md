# Native engine implementation status

Updated 2026-11-23. The user-supplied WP0–WP11 specification is authoritative. Existing Python Qwen and Model B/C infrastructure are unchanged. No production backend selection changed. Synthetic transport is not model inference.

## Work packages

| Package | Status | Evidence / remaining gate |
|---|---|---|
| WP0 Reference baseline | in progress | `native_reference_manifest.json`, `REUSE_AND_BASELINE.md`, `WP0_QSA_reference_run.json`; bounded CPU QSA reference tests 11 passed/11 deselected. Full payload hashes, matched model fixtures, effective baseline configs and new throughput/resource runs remain outstanding. Authoritative checkpoint revision and shipped Strata build provenance are external dependencies. |
| WP1 Runtime foundation | completed | `mini_llm/native_cpp/{CMakeLists.txt,runtime.h,runtime.cpp,README.md}`, `mini_llm/native/__init__.py`, `tests/test_native_engine_runtime.py`; C++20 CUDA-runtime library builds; all 8 foundation tests pass within latest combined run. Synthetic token copies only. Allocation accounting covers requested session buffers, not CUDA context/allocator overhead; no claim of leak-detector validation. |
| WP2 Model loading | in progress | `mini_llm/native_cpp/gguf_loader.{h,cpp}`, `tests/test_native_engine_gguf.py`, `WP2_GGUF_DESCRIPTORS.md`, `WP2_IQ_INVENTORY.md`, `WP2_GGUF_INVENTORY.json`; read-only mmap/header descriptors validated for all 1,224 tensors and 13 formats of actual two-shard checkpoint. Required-tensor/layer binding now passes using the existing Python architecture authority over native descriptors plus native exact-schema validation (`WP2_QWEN_CONTRACT.json`). Standalone C++ metadata interpretation is deliberately not duplicated. Payload-consumer leases and budgeted dense GPU weights remain. |
| WP3 Native FP32 QSA | pending | Existing decode/reference kernels inventoried; native complete-layer acceptance not met. |
| WP4 Recurrent/HC/PLE | pending | Existing kernels inventoried; native ownership/wrappers not implemented. |
| WP5 Native MoE | pending | Existing workers/residency/handoff inventoried; native scheduler integration not implemented. |
| WP6 Paged/quantized KV | pending | Depends on validated native QSA. |
| WP7 Complete decoder | pending | Depends on computational integration. |
| WP8 Generation/serving | pending | No serving behavior changed. |
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

Define payload-consumer leases retaining immutable mmap ownership and invalidation/lifetime tests before exposing any weight views; then implement bounded, explicitly converted dense-weight uploads under engine allocation/stream ownership as a separate milestone. No dense GPU uploads were attempted in this milestone. Keep WP0 matched baseline work open and do not enable a production native backend before its numerical and performance gates pass.

## Preservation

Pre-existing changes in `mini_llm/heterogeneous_production.py`, its `.bak`, and `strata_launch_bench_capture.sh` were left untouched. No commits have been created. No protected runtime data, checkpoint files, or training contracts were modified.
