# Patches 0274–0279 — Critical-path CPU/GPU handoff optimization

## Result and benchmark scope

**Best completed longer workload: 26.95159 decode forwards/s, not 50/s.**

All runs replay the captured Strata 75/77-token prompt IDs, 128-token warmup, 512-token response cap, temperature 1 / top-p .95 / top-k 20 / seed 1 / thinking xhigh. Checkpoint processes run sequentially with a 180-second child/parent watchdog, 8 GiB anonymous-memory limit and 4 GiB MemAvailable floor. Expert mappings are touched up to 50 GiB without a private/pinned full expert copy. No reference engine or checkpoint was modified.

| Patch/run | Workers + helper | Response tokens | Decode forwards/s | Evidence |
|---|---:|---:|---:|---|
|0273 control|15 + 1|413|18.88463|`diagnostics_0273_shared_graph_v1`|
|0274 deferred shared|15 + 1|413|19.31426|`diagnostics_0274_deferred_v1`|
|0275 native host handoff|15 + 1|413|24.29482|`diagnostics_0275_native_handoff_v1`|
|0276 captured handoff|15 + 1|413|25.86397|`diagnostics_0276_native_graph_v1`|
|0277 specialized resident kernels|15 + 1|413|24.62057|`diagnostics_0277_specialized_v1`|
|0277 larger CPU team|23 + 1|413|**26.95159**|`diagnostics_0277_workers23_v1`|
|0278 QSA frontend graph|23 + 1|413|26.12946|`diagnostics_0278_qsa_frontend_v1`|
|0279 input-only graph operands|23 + 1|413|25.41335|`diagnostics_0279_input_only_v1`|

**Every row has the identical 413-token response IDs.** These are sequential single runs, not randomized trials or confidence intervals. The best observed rate should not be interpreted as a guaranteed sustained rate. Specialized kernels, QSA frontend capture and input-only operands show no throughput benefit in these runs and remain independently switchable.

Strata still differs: context capacity 262144 versus 1024, INT8 versus FP32 KV, suffix speculation and eligible route-tail omission versus complete top-10/no drafting, pinned complement/PCIe staging versus reclaimable mmap/CPU misses, and sized slots versus 2419 maximum-stride slots. Full configuration parity is false. Do not equate Strata emitted-token/s with identical-input full forward/s.

## 0274 — Deferred shared work

**Files:** `ops/{device_expert_plan,expert_residency,experts_inference}.py`, `external/qwen4exp_model.py`, workload/parity/critical-path runners, `tests/test_0274_deferred_shared.py`.

`ExpertsInference.forward(..., on_cpu_submitted=None)` adds a once-only scheduling seam. Ordinary managed paths invoke it after queuing CPU work and before waiting; all-resident/unmanaged paths still invoke once. Unsupported accelerated paths decline without invoking it. Submission or callback errors drain started CPU work before source owners can retire. Existing calls without the keyword remain supported.

`Qwen4ExpMoE(..., defer_shared=True)` requires `overlap_shared=True`. Shared GPU work is queued at that seam rather than ahead of routing/input readiness. A producer event protects hidden; a completion event protects shared consumers on success/failure. Later graph dispatch queues the complete CPU handoff rather than synchronously submitting it; shared work depends on hidden's producer, **not** the graph's CPU-result join.

Focused new/integration checks passed; final broad results below supersede intermediate counts. `0274_deferred_parity_v1.json`: bit-exact logits and all active-state hashes, two isolated requests, prefixes 1/2/4/8/16. Same arithmetic, not new llama.cpp equivalence. The speed change 18.88 → 19.31/s is small in one pair. No new large storage allocation.

## 0275 — Stream-ordered native host function

**Files:** new `ops/native_expert_handoff.py`; existing original `ggml_packed_cpu.cpp` bridge, device plan and executor cleanup/statistics; `tests/test_0275_native_expert_handoff.py`.

Opt-in `MINI_LLM_NATIVE_MOE_HANDOFF=1`, together with the existing device plan. Frozen cache/no idle eviction only. Registration retains immutable canonical source views for every possible CPU miss, bounded by the existing 32768-descriptor registry; it does not load or copy their payloads.

Dependency chain:

1. GPU sorts routes and selects packed resident pointers.
2. Main-stream D2H copies publish pinned input and metadata.
3. A dedicated CPU-work CUDA stream waits on the producer event.
4. `cudaLaunchHostFunc` invokes a native `noexcept` function: validate IDs, select registered misses, submit and finish the existing native row pool.
5. That stream transfers complete unweighted CPU rows and publishes completion.
6. Resident GPU kernels run concurrently; main stream joins CPU completion and performs the unchanged ascending weighted reduction.

The host function enters neither Python nor CUDA. No Python callback holds the GIL while CPU experts execute. Persistent output ownership remains fresh per call. Native errors poison CPU rows rather than consuming stale outputs, and explicit statistics/close reports failed dispatches.

All layer residents receive fixed leases for the plan lifetime. Cache changes retire plans before releasing them. **Executor close now retires every plan before destroying the shared CPU pool**, including when one plan reports an error. Aggregate counters are collected at synchronized statistics/cleanup seams, not through per-token Python route reads. Per-expert decayed heat is explicitly paused in this frozen-placement experiment; it is not an adaptive mover.

`0275_native_handoff_parity_v1.json`: bit-exact logits/state hashes against 0274 at fixed prefixes over two requests. Synthetic mixed/all-CPU/all-GPU tests check retained outputs, counters, placement invalidation, callback failure, invalid IDs and close with pending work. Long same-token throughput increased 19.31 → 24.29/s (~25.8%).

Memory observations: sampled anonymous RSS 852914176 bytes (~0.79 GiB), file-backed RSS 55074443264 bytes (~51.29 GiB). CuPy default pool used/reserved 12157392896 / 12345454080 bytes; these are snapshots, not device-wide peak VRAM. PLE remains mapped and expert bytes remain canonical.

## 0276 — Capture the native handoff

**Files:** native handoff module, parity/workload integration; `tests/test_0276_native_handoff_graph.py`.

Opt-in `MINI_LLM_NATIVE_MOE_GRAPH=1`. Capture the route plan, explicit pinned copies, native host node, resident arithmetic, joins and combination as one reusable graph. CuPy ndarray `.get()` rejects capture even for pinned buffers; direct CUDA pointer-copy APIs use the same owned slabs and legal stream dependencies. Dynamic activations/routes/IDs are replay inputs, not constants. Private warmup native calls are excluded from production counters. Shared work remains an independent stream waiting on hidden's producer.

One MiB cap per layer; observed graph reservations total **1523712 bytes** across 48 layers in the later timed run. Model preflight reserves the full cap. `0276_native_graph_parity_v1.json`: bit-exact fixed-prefix logits and states against 0275. Dynamic-route tests vary inputs and selections over 24 steps. Longer result 25.86/s, same IDs. No asynchronous weight-placement claim.

## 0277 — Specialized resident kernels and native CPU timing

**Files:** device plan selects the existing specialized warp module; native bridge/handoff/executor expose aggregate elapsed CPU time and row-batch counts. Tests reuse retained independent warp and GGML arithmetic oracles.

`MINI_LLM_DEVICE_PLAN_SPECIALIZED=1` constant-folds the known dimensions/format descriptors. Packed decoding, activation precision, reduction order and routing weighting stay unchanged. Relevant specialized/native GPU run: 86 passed / 1 skipped. Despite operator potential, the long result **24.62/s** does not improve the 25.86/s control; keep opt-in.

Native timestamps measure input quantization, pool submission/participation and task completion together. They are operation dependency spans, not exclusive wall-time attribution. The response case including its 77 prefill positions measured about **14.96 ms/forward** of native CPU work in the specialized run. Enlarging the team to **23 workers + participating callback thread** reduced this to **13.21 ms/forward** and measured **26.95/s**, with ~17.06 active-core equivalents. No affinity pinning or cgroup throttling was introduced. Peak per-worker memory was not measured separately; all scratch remains preallocated/bounded.

## 0278 — QSA frontend graph (no demonstrated gain)

**Files:** `external/qwen4exp_qsa_frontend_graph.py`, QSA adapter/model planning and cleanup, parity/workload reporting, `tests/test_0278_qsa_frontend_graph.py`.

Opt-in `MINI_LLM_QSA_FRONTEND_GRAPH=1` with fast QSA. Captures query/gate, key, value, index-key projections, normalization and partial half-layout RoPE in the mathematically all-causal-cells short-context regime. Runtime cosine/sine slices are replay inputs, so positions do not freeze. KV/index history, pooling, output projection, finite checks before state commit, diagnostics and wider sparse selection stay in the established implementation. Dense capture-ready projections use existing Triton FP32 GEMV; quantized projections retain native packing. No skipped finite validation or approximate attention.

Two MiB/layer cap; observed ~2.49 MB total reservations. GPU preflight rejects host-packed projection capture; replanning credits only owned graph pools; close fences bound streams before weight owners retire. Focused QSA group 51 passed. `0278_qsa_frontend_parity_v1.json` compares same-GEMV uncaptured/captured paths: bit-exact logits/state hashes over two fixed 16-token requests. Synthetic tests verify changing positions, reset, sparse fallback and unchanged persistent state on nonfinite input. Longer result 26.13/s versus 26.95/s; no speedup established.

## 0279 — Input-only graph operands (no demonstrated gain)

**Files:** `ops/inference_graph.py`, handoff/QSA graph adapters, `tests/test_0279_graph_input_only.py`.

`CapturedArrayInference(..., copy_back_mutable=True)` keeps prior semantics by default. Setting false means additional operands are input-only: copy into private replay buffers, do not write them back after execution. Outputs still retain the selected fresh/borrowed ownership contract. `MINI_LLM_GRAPH_READONLY_INPUTS=1` applies this only to read-only route metadata and RoPE factors, never recurrent/convolution state.

Tests count the eliminated copies and verify unchanged operands, varying inputs and retained outputs. `0279_input_only_parity_v1.json`: bit-exact logits and state hashes against 0276 over two requests. Longer result 25.41/s, same response; the simpler copy contract is correct but a throughput benefit is not established. No new storage or numerical tolerance change.

## Latest critical-path evidence and limitation

- `0274_critical_path_v1`: 16-token window ~835 ms, summed GPU kernels ~187 ms, event synchronization ~149 ms, nested CPU completion waits ~122 ms. These overlap.
- `0275_critical_path_v1`: ~589 ms window; native handoff removes explicit Python completion waits, but ~180 ms in runtime memcpy calls remains. Runtime API wall time is not DMA duration.
- `0278_critical_path_v1`: **15 workers**, captured native handoff, shared/pre graphs, **QSA frontend off**, ~588 ms/16 tokens; ~182 ms runtime memcpy API time. This trace used default graph tracing, which omits captured kernel-node detail: do not present its visible kernel sum as total GPU time. Next trace needs node-level CUDA graph tracing and copy-call/source correlation.
- A bounded standalone mapped-memory `cuStreamWaitValue32` capability probe returned **801 / CUDA_ERROR_NOT_SUPPORTED**. Not integrated and not a model failure. A native doorbell replacement cannot simply assume that driver primitive is available here.
- Re-read pinned Strata `src/kernels/cpu/native_expert.cpp`: singleton IQ arithmetic ordinarily uses GGML; optional AVX2 IQ row kernels and exact variants have explicit dispatch gates, and Q8_K input quantization has a byte-exact SIMD implementation. Our supplied GGML build reports AVX2/FMA/AVX-VNNI enabled, AVX512 disabled; no evidence of an accidentally scalar-only IQ build. Multi-token/suffix batching and PCIe work splitting are real remaining reference differences.

## Verified final-tree state / next dependency

Final broad commands (explicit pinned GGML library, OPENBLAS/OMP threads=1):

- NumPy: `python -m pytest -q tests/test_02*.py tests/test_0106*.py` — **1939 passed, 1016 skipped**.
- Actual CuPy: `python -m pytest -q tests/test_0228*.py tests/test_023*.py tests/test_024*.py tests/test_025*.py tests/test_026*.py tests/test_027*.py tests/test_0106n_qwen3next_attention.py tests/test_0106m_qwen3next_mixer.py tests/test_0106o_qwen3next_decoder.py tests/test_0106p_qwen3next_model.py` — **1202 passed, 18 skipped**.
- Optional input-only/captured native/QSA/segment group — **17 passed**.

Final XML and verified source hashes are retained in `0279_{cpu,gpu}_regression_v1.xml` and `0268_0279_performance_v2.json`. Two stale synthetic fixtures were fixed: constructor-owned worker capacity and backend-native fake chat logits. No production mathematical correction was made.

**Next dependency:** measure node-level captured GPU/CPU dependencies and attribute the remaining runtime memcpy blocking before another scheduling redesign. Evaluate a bounded native doorbell service only with a supported, ownership-tested completion mechanism; do not delete joins. Independently measure CPU SIMD quantization/row arithmetic and Strata's emitted-token/speculation contribution. The 50/s target remains unresolved; this report does not claim completed performance optimization.
