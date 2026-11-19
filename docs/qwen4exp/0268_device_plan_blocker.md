# 0268 — Persistent device expert plan: historical blocker, now resolved

## Resolution — verified after the original report

The retained test oracle now explicitly uses the same warp arithmetic as the device plan. Mixed/all-GPU output comparisons are bit-exact. A separate block-versus-warp test records maximum absolute FP32 difference **3.725290298461914e-9**, relative L2 **2.3721347696171626e-7**. No production mathematical correction or broad tolerance relaxation was necessary.

The follow-up focused GPU group passed **97 tests / 1 skipped**. Real longer model run `diagnostics_0268_plan_v1` completed at **17.57999 decode forwards/s**, 512 response tokens. Subsequent verified pipeline work reaches a best longer **26.95159/s**, not 50/s; see `0268_0279_performance_v2.json` and current implementation status. The old one-correction stopping rule was explicitly superseded by the user.

**Everything below is the preserved historical failure report, not the current validation state.**

## Objective / architecture

Remove repeated host pointer-table uploads, per-expert GPU launches and transient transfer slabs on the measured mixed-decode critical path. This adapts Strata's device hit planning and persistent-buffer separation without porting its runtime. CPU fallback remains complete; no route omission, activation precision change or full expert copy.

Files: new `mini_llm/ops/device_expert_plan.py`; integration in `ops/{expert_residency,experts_inference}.py`; zero-pointer lanes in reused `ops/iq_grouped_warp_gpu.py`; explicit model flag/preflight in `external/qwen4exp_model.py`; `tests/test_0268_device_expert_plan.py`.

Opt-in `device_expert_plan=True`, defaultFalse. Requires CuPy,GGMLrowpool, frozen admissions and no idle evictions. Supported bounded geometry: at most64layers/512experts/10selected, widths<=4096. Model planning reserves additional persistent device scratch and pinned staging. Unsupported configurations retain the established executor.

Per-layer persistent residency pointers refer to complete leased packed experts. A GPU kernel sorts routes by expertID and selects resident pointers; only selected experts execute. One metadata/input producer fence precedes native CPU misses. Reused warp gate/up/down kernels skip nonresident pointer0 lanes; one ordered combine merges weightedGPU and unweightedCPU outputs, applying router weights once. Result copies preserve output ownership. Reused pinned slabs have completion fences. Cache promotion/eviction counters invalidate plans; resuming adaptive admission drains/releases them. CPU jobs drain on injected GPU failure. No whole-token graph or asynchronous placement claim.

## Latest verification — NOT complete

First focused test run failed in a new fixture: cached provider branch referenced a local `e` assigned only on cache miss. One localized fixture correction moved assignment before the branch.

The retest completed with **94passed,1skipped,2failed** across0268/0257/0237. Failures:

- `test_exact_mixed_order_and_retained_output[3]`
- `test_exact_mixed_order_and_retained_output[10]`

The all-CPU comparison, route-validation/resume test and injected GPU-failure drain test passed. GPU-containing cases fail a bitwise assertion. No real-model0268benchmark has been executed; the workload CLI has not enabled it. Do not mark this path validated, enable it by default, relax the assertion blindly, or advertise an inference gain.

## Recorded discrepancy / current hypothesis

All-GPU case, FP32:

- Representative actual0.0014968726318329573, reference0.0014968723990023136.
- Representative absolute error2.3283064365386963e-10, relative~1.56e-7.
- Maximum absolute difference3.7252903e-9.
- Maximum elementwise relative difference5.436712e-5 (near-small-magnitude values can amplify this metric).
- Required tolerance: exact equality (zero tolerance).
- No finite differences or epsilon; relativeL2 was not captured.
- Three-hit case also failed exactness; its full metrics were not retained in the shortened terminal excerpt.

Source evidence after the retest: the new planner explicitly uses the existing **warp** kernels, while its `grouped_terms` test oracle inherits `_grouped_kernel()`'s default **block** mode when `MINI_LLM_IQ_GPU_GROUPED_KERNEL` is unset. Existing warp module documentation explicitly states that its within-row FP32 reduction order differs from the block oracle. Thus an arithmetic-reference mismatch is a concrete explanation; these small differences do not demonstrate an incorrect model equation. A planner defect is not ruled out by this observation alone.

## Stop condition and single next experiment

Stopped after the permitted correction/retest; no further production/test correction was made for this failure. The last completed benchmark is0267at15.21239forwards/s, before this default-off candidate's source edits. Existing operator regressions passed in the same retest, but a complete final-tree green or new full-model state claim is not established.

The single cheapest next experiment is to run only0268tests with the retained oracle explicitly set to `MINI_LLM_IQ_GPU_GROUPED_KERNEL=warp`, keeping identical inputs, quantization, worker and source settings. Record maxabs/relativeL2 for both oracle modes. This experiment has **not** been run in the current correction cycle. If same-warp parity holds, verify owned buffers/state and rerun the longer checkpoint benchmark before considering default enablement. If it fails, isolate gate/up, SwiGLU, down and ordered combine before modifying production arithmetic.

50t/s remains unmet. No checkpoint or pinned reference source/build was modified.
