# 0281–0288 — Mixed-format Q8, bounded ownership and measured execution costs

## Result and scope

**50 decode forwards/s is not reached.** Best observation in the retained safe callback configuration: **31.42545/s**,359 response IDs including the first token from prefill. Final callback reproduction29.29410/s and final native-service29.55954/s show run-to-run variability. An earlier experimental service iteration measured31.71163/s but its timeout-copy path was subsequently corrected; it is not the recommended implementation. This is not an equivalent Strata configuration or a50/s result.

Evidence: `0281_0288_performance_v3.json`, versioned `diagnostics_0281_*` through `diagnostics_0288_*`, `0282_operator_q8_v1.json`, `0286_cpu_pcie_cost_v{1,2}.json`, `0285_critical_path_summary_v1.json`. Raw node-level Nsight reports/SQLite exports are retained. No prior captures overwritten, checkpoint downloaded/modified, pinned-reference equation changed, or training code edited. The user committed the earlier 0281–0284 source as `52c42e8`; subsequent changes preserve that state rather than reverting it.

The longer workload retains the captured Strata template/prompts/sampling, 128-token warmup, 512 response cap, all top-10 experts, no speculation, FP32 KV/context1024, one request at a time, 50 GiB canonical page warming and memory/time guards. Strata's captured context262144, INT8 KV, suffix drafting, eligible tail omission, pinned complement and placement differ. Report **forwards/s**, not a fabricated equivalent emitted-token rate.

## Measurements

| Patch/configuration | Response tokens | Decode forwards/s | Interpretation |
|---|---:|---:|---|
|0281 expert Q8,23 workers,500us spin|448|23.45571|Different trajectory from prior26.95/s control|
|0282 expert+dense Q8,23 workers,500us|415|23.17058|No gain established from arithmetic alone|
|0282 same Q8,19 workers,50us|415|28.61418|Same warmup/response IDs; scheduling matters|
|0283 fused hidden-quantization phase|415|27.96543|No gain; remains off|
|0284 direct BF16 HC storage,6GiB cache|415|30.62288|Same415 IDs as0282; exact HC arithmetic|
|0284 same storage,7GiB cache|359|31.42545|Changed placement/trajectory; no identical-input speedup ratio|
|0284 QSA/frontend/read-only/deferred flags together|359|31.30024|Same359 IDs; no gain|
|0285 cumulative warmup-frequency placement|325|26.65913|Higher hit rate did not establish faster decoding|
|0285 15 workers,50us spin,7GiB|359|30.54409|Same359 IDs;19-worker configuration preferred|
|0287 initial mapped native CPU service|359|31.71163|Historical iteration; timeout-copy path subsequently corrected|
|0287 gated mapped-status copy|359|19.60218|Safe but repeated system-acquire loads were expensive|
|0287 final gated device-status copy|359|29.55954|Same359 IDs; no material gain over callback configuration|
|0288 size-classed fixed slots, sampled|351|29.96722|3673 slots; different continuation|
|0288 controlled359-token replay, uniform slots|359|30.19602|Full head/sampler retained; forced trajectory|
|0288 controlled359-token replay, sized slots|359|31.14912|Same inputs; ~3.2% rate increase in one sequential pair|
|0288 final natural7GiB reproduction|359|29.29410|Same359 IDs; exposes run-to-run variability, not a new speedup|

These are single sequential measurements, not confidence intervals. The final natural reproduction is29.29410/s on the same359 IDs; small single-pair differences are not robust speed improvements. Do not compute isolated speedup ratios across different token IDs. Cache placement also changes CPU-Q8_K/Q8_0 versus GPU-Q8_1 execution sites and can change rounding/routing.

## 0281 — Mixed-format resident expert GPU Q8

**Status: numerically validated within the selected path, not performance optimized.**

Files: `mini_llm/ops/mixed_q8_gpu.py`, `device_expert_plan.py`, `native_expert_handoff.py`; `tests/test_0281_mixed_q8_gpu.py`, extracted MIT-licensed `tests/fixtures/strata_mixed_q8_reference.cuh`.

- Covers IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS and Q2_0; all seven actual gate/up/down triples are tested.
- `MINI_LLM_EXPERT_GPU_Q8=1` selects `GroupedQ8Plan` for frozen device-plan experts. It owns only O(D + selected*F) activation scratch, not weight copies. CPU misses retain GGML's format-specific Q8_K/Q8_0.
- DP4A integer dots, FP32 products/reductions and existing FP32 SwiGLU remain separate from once-only routing combination.
- IQ2_S and IQ3_XXS preserve Strata's integer divisions/truncation. IQ4_NL preserves two partial dots per32-value block. These cannot be replaced by an assumed real-valued odd-scale multiplication while claiming exact Strata parity.
- General/warmup fused expert paths are not silently relabeled Q8. This is not whole-engine Strata bit parity.

Tests compare extracted pinned Strata parts/rows bit-for-bit, an independent FP64 decoded-weight/decoded-Q8 baseline with explicit integer-truncation bounds, mixed CPU/GPU combination, synchronous/native/captured dispatch and output lifetime. Original FP32 kernels remain available. Expert-only longer run did not improve throughput.

## 0282 — K-format Q8 and dense integration

**Status: numerically validated within path.**

Files: `mixed_q8_gpu.py`, `quantized_linear.py`; `tests/test_0282_k_q8_gpu.py`, `tests/fixtures/strata_k_q8_reference.cuh`; `scripts/probe_0282_mixed_q8.py`, workload/profiling runners.

Adds Q4_K, Q5_K, Q6_K and Q8_0. `MINI_LLM_LINEAR_GPU_Q8=1` opts contiguous FP32 packed GPU projections into Q8_1, preserving2D/3D output shape. Host/reference calls, unsupported types and other activation dtypes retain their original paths. Generic fused `quantized_swiglu` is not converted by this flag.

Synthetic parts and row sums are bit-exact against extracted Strata across widths256/768/2560, including signed Q6 scales, Q5 high bits, K minima, capture and layout checks. Real bounded32-row checkpoint slices cover Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/Q8_0 and are bit-exact to extracted row dots. FP64-Q8 baseline and Q8-vs-FP32 differences are recorded, not hidden behind a model-wide tolerance.

Same-input operator event timings: dense Q4_K2.11x, Q5_K2.43x, Q6_K1.20x, Q8_01.56x; small shared IQ4_NL0.89x/IQ4_XS0.99x. Real resident top10 operators2.12–3.53x. These are **not model speedups**.

Initial long run stopped safely at the second request: the planner needed1850991104 free bytes but saw1819607040. Unused CuPy scratch, not a weight-budget exemption, caused the conservative check. The runner now releases unused pool blocks only at its synchronized, untimed request seam. Live model/state/cache ownership is unchanged. Retest completed. The runner also trims unused constructor blocks at the synchronized setup seam before the first request.

A later8GiB-cache capacity probe was **safely rejected**, both before/after setup trimming: required10568360448 free bytes versus9913368576 available in the final attempt. No benchmark result is inferred for8GiB, no memory check/headroom was relaxed, and the verified7GiB configuration remains the recommendation. Captures: `diagnostics_0288_cache8_v{1,2}/child.log`.

19workers+participating waiter and50us spin reduced native CPU spans from15.91 to11.49ms/forward including prefill, with identical415-token IDs. More spinning/threads was not automatically better after GPU work shrank.

## 0283 — Fuse the tiny hidden-quantization worker phase

**Status: numerically validated, no measured performance gain.**

Files: `ggml_packed_cpu.cpp/.py`; `tests/test_0283_fused_hidden_phase.py`.

`MINI_LLM_GGML_POOL_FUSED_HIDDEN=1` / `GGMLPackedCPURowPool(..., fused_hidden=True)` makes the last completed gate task quantize the bounded hidden rows before publishing down work. The acquire/release task-completion counter retires every gate write first. Quantizers, dot order and SwiGLU are unchanged; no row/expert is omitted. Default remains false.

Exact tests vary1/4/19/23 workers,1/3/10 experts, host participation, registered/ordinary submission,80 epochs and output lifetime. The natural415-token run did not beat the unfused control; do not recommend enabling it for speed.

## 0284 — Exact BF16 HC storage, FP32 arithmetic

**Status: numerically validated including scoped real-model/state replay; demonstrated longer-run benefit at6GiB.**

Files: `hyperconnection_read_gpu.py`, `qwen4exp_hyperconnection.py`, `qwen4exp_weights.py`, `qwen4exp_model.py`; `tests/test_0284_bf16_hc_storage.py`.

`MINI_LLM_HC_BF16_WEIGHTS=1` retains checkpoint HC down/up/inject matrices as uint16 BF16 bytes and reconstructs FP32 values at loads. Activations, gamma, products, reduction order and outputs stay FP32. No activation BF16 rounding or FP16 compute policy is introduced.

`Qwen4ExpWeights.configure_hc_bf16()` validates checkpoint types before payload access/placement; `hc_matrix()` copies only these small packed matrices. Preflight subtracts the exact retained storage saving and reports `hc_bf16_saved_bytes=1279262720`. CPU/default dense storage remains unchanged. `packed_bf16=True` requires native, single-token CuPy HC; unsupported batching fails explicitly instead of allocating full FP32 fallback copies.

Synthetic reads/writes/collapse/capture are bit-exact versus the same BF16 values expanded to FP32. Actual checkpoint graph/direct-HC control: **all10 captured logits arrays and all active-state hashes match exactly at prefixes1/2/4/8/16 across two requests**. CPU experts are held identical in this probe; this is not new independent llama.cpp equivalence.

Sampled6GiB-cache run: RssAnon851–865MB-class, file-backed working set about50GB; see per-run exact values in aggregate. At7GiB the uniform arena retains2823 experts, GPU default-pool snapshot about11.97GB rather than the earlier12.35GB-class allocation. These snapshots exclude private graph pools/driver allocations and are not device-wide high-water marks.

## 0285 — Profile placement and refreshed critical path

**Status: placement and analysis implemented/tested; profile policy not performance optimized.**

Files: `expert_profile.py`, `expert_residency.py`; `tests/test_0285_expert_profile.py`; `scripts/analyze_0285_critical_path.py` and profiling runner.

`frequency_placement()` produces stable bounded cumulative-frequency placement. `install_observed_profile()` is a serialized request-free setup seam: rejects live consumers, frozen/native plans and sized layouts, retains canonical sources, replaces only complete bundles and keeps CPU fallback. `AdaptiveExpertCache.retain_only()` refuses protected/in-flight removals. No future response routes are used. `--warmup-frequency-placement` is optional and not recommended by its measured result.

Fresh node trace,6GiB cache/19workers/Q8/BF16HC,16 response positions: GPU operation union217.214ms, kernel sum214.856ms, GPU span602.011ms. Leading kernels: Q8 linear73.716ms; resident gate/up25.107ms; HC down14.386/up14.106ms; QSA13.540ms; resident down11.580ms. Largest idle categories are gaps before102400-byte H2D CPU-result transfers (158.368ms),10240-byte H2D copies (55.641ms), and planned combination (35.013ms). They include CPU dependencies, graph scheduling and host submission; they are **not an exclusive additive causal decomposition**. Profiling overhead applies.

## 0286 — Measure CPU versus PCIe expert execution

**Status: bounded host tier/operator validation implemented; not integrated into model decode.**

Files: `mapped_packed_experts.py`, `tests/test_0286_mapped_packed_experts.py`, `scripts/probe_0286_pcie_experts.py`.

`MappedPackedExpertTier` owns one explicit pinned packed arena, complete expert aliases and immutable canonical source handles. Exported projection aliases retain the pinned owner. Host budget/floor and allocator rounding are checked. This is host storage with GPU execution, **not VRAM residency**. No full expert bank is pinned/copied.

Read-only mmap `cudaHostRegister(mapped|readonly)` returned `cudaErrorNotSupported`; the checkpoint was never made writable. Bounded owned pinned memory is CUDA-readable via `cudaHostGetDevicePointer`. This functionality is separate from the earlier unsupported stream-wait API.

All tested remote/staged outputs are bit-exact to VRAM Q8 output. Actual D2560/F640, all seven triples,19workers+waiter, shared inputs,1/2/3/5/10 experts:

- One CPU expert median wall ~46–79us; direct mapped GPU reads ~122–470us. **Do not enable direct PCIe dots as an optimization.**
- Staging one complete packed expert into a reusable GPU buffer + Q8 compute ~56–76us; five experts ~219–326us versus CPU~119–205us. Moving all misses to GPU is not justified.
- Splitting1–2 misses concurrently with remaining CPU work may help but requires a bounded asynchronous staging scheduler and model critical-path validation. The isolated probe does not demonstrate such overlap or a model gain.

Each operator tier is32MiB, with allocator/temporary ownership potentially retaining additional freed blocks. No whole-process pinned-memory peak is claimed.

## 0287 — Native mapped-doorbell CPU service

**Status: numerically/lifecycle validated; no material throughput gain established.**

Files: `native_expert_service.h/.py`, `ggml_packed_cpu.cpp/.py`, `native_expert_handoff.py`, `expert_residency.py`, `qwen4exp_model.py`; `tests/test_0287_native_expert_service.py`.

`MINI_LLM_NATIVE_MOE_SERVICE=1` substitutes one native polling thread per CPU pool for per-layer CUDA host-function nodes. Stream-ordered copies publish pinned input/IDs, a system-release mapped doorbell publishes a new epoch, and the existing row pool computes unchanged complete outputs. GPU resident/shared work can execute while CPU work runs. A system-acquire wait retires CPU rows before a completion-gated mapped-output copy and combination. There are no CUDA/Python calls inside the service thread and no per-token weight allocation.

128-byte cache-line-separated flags per handoff; at most64 registrations. Epoch exhaustion is explicit. GPU waits have a2-second bound; failure poisons results and sets a sticky service status. One completion-ordered model-step check rejects failure **before successful state-length advance/output return**. Failed executors must be unloaded, not silently resumed. Removal holds the native service lock until an active batch retires; close destroys all clients/service before the row pool/canonical descriptors. The native build hash now includes the service header.

Tests cover dynamic epochs, two layers sharing a service, graph replay, complete CPU/GPU outputs, error/timeout/overflow, poisoned output, failed-state reporting and close. Full real-model fixed-prefix service/control logits and all active-state hashes are bit-exact over two16-token requests, including the final `diagnostics_0287_service_parity_v3` implementation. Actual359-token sampled continuation is unchanged. Final service29.55954/s does not establish a material gain; keep opt-in. GPU polling time must be classified as waiting, not useful arithmetic, in future traces.

**Timeout correction:** initial service code poisoned output but still queued an unconditional output DMA after a GPU wait timed out. That could read a slab still being written by the CPU. The final implementation uses a conditional GPU copy that reads no CPU payload after failure; tests deliberately pass a null payload pointer on failed epochs. Wait completion publishes status into VRAM once, including worker-failure publication ordered before its done flag. Downstream copy/poison kernels read that device-local status.

The intermediate safe implementation did a system-acquire mapped-status load per payload warp/thread. `0287_copy_status_cost_v1.json` measured~0.70ms/100KiB copy versus~0.0063ms with device-local status, explaining its19.60/s regression. The final path removes those repeated system-acquire PCIe status loads without weakening the wait or error gate; `0287_copy_status_cost_v2.json` retains the corrected isolated measurement. This is a demonstrated new execution-safety defect corrected locally, **not a model mathematical defect**.

## 0288 — Size-classed fixed expert slots and controlled trajectories

**Status: numerical/ownership validation passed; modest controlled performance improvement only.**

Files: `expert_slot_layout.py`, `packed_expert_slots.py`, `expert_residency.py`, `qwen4exp_model.py`; `tests/test_0288_sized_expert_slots.py`; workload runner/tests and `scripts/probe_0288_report.py`.

`MINI_LLM_SIZED_EXPERT_SLOTS=1` plans256-byte-aligned fixed slots using actual header bundle sizes, with approximately equal resident fractions per size class. A single bounded GPU allocation and staging slab remain. `reservation_size()`/`has_free_slot()` expose physical compatibility; `AdaptiveExpertCache.request(..., needs_slot=..., compatible_victim=...)` can replace a compatible slot even with spare global bytes, but never evicts an unrelated class merely to pretend capacity exists. Projection leases, quarantine, complete publication and CPU fallback are preserved.

7GiB now provides3673 instead of2823 slots, with7515878400 physical bytes; no per-token GPU malloc/free. Uniform class allocation is not heat/cost optimal. Static cumulative-profile installation currently rejects sized layouts rather than silently composing incompatible policies.

`--force-tokens-from <retained result.json>` enables a controlled long trajectory for both warmup and response. Full logits/head/sampler still execute; only sampled token choice is overridden. Sources and forced status are reported explicitly. It must not be advertised as natural generation or as an identical arithmetic/residency path.

On the identical359-token trajectory: uniform30.19602/s versus sized31.14912/s; native CPU spans including prefill11.054 versus9.466ms/forward. Natural sampled sized run29.96722/s uses351 response IDs and does not beat the best sampled observation. Do not enable sized slots on the assumption that capacity alone guarantees speed.

## Final verification and next dependency

- NumPy inference regression: **1969 passed,1124 skipped**, `0288_cpu_regression_v2.xml`.
- Actual CuPy inference regression, including explicit Qwen3-Next mixer/attention/decoder/model and GPU recurrence/attention: **1367 passed,14 skipped**, `0288_gpu_regression_v2.xml`.
- Focused service/native/captured/mixed-Q8 group after the timeout/device-status correction: **52passed**. Earlier broader focused group122passed is retained as `0288_focused_regression_v1.xml`; final broader suites above cover the corrected code.
- Full training suite was not run; no training implementation changed.
- Reference/production executions remain sequential and memory bounded. No tolerance relaxed to hide a failure. Several test-fixture API/shape expectations were corrected locally; failed artifacts remain retained. Current suites pass.

Best retained safe callback observation31.43/s is below50/s; final reproductions are about29.3–29.6/s. The next substantial candidates are: a measured **staged** CPU/PCIe split (not direct mapped dots), cost/heat-aware sized-slot allocation, and a larger decode-capture scope with runtime QSA position/index/commit metadata. Any whole-token capture must preserve evolving positions/state and failure checks. Recheck against the pinned Strata batching/suffix protocol: its emitted-token rate cannot be equated to this engine's complete-top10, non-speculative forward rate. Do not delete finite checks, skip experts, change precision silently, or extrapolate kernel ratios into a50/s claim.
