# 0265 — Capture the complete recurrent GPU segment

## Objective and contract

Capture the five projections, coefficient calculation, convolution, DeltaNet recurrence, gated normalization and output projection for one recurrent token. This boundary contains no routing, PLE lookup, QSA position or selection width. The recurrent equations depend on explicit array contents, not the Python token counter.

Files: `external/qwen4exp_mixer.py`, `external/qwen4exp_model.py`, `external/qwen4exp_weights.py`, `ops/gated_delta_{decode,inference}.py`, `ops/inference_graph.py`, `scripts/probe_0254_strata_workload.py`, `scripts/probe_0265_gdn_graph_parity.py`, `tests/test_0265_gdn_segment_graph.py` (all under `mini_llm/` where applicable).

Opt-in model keyword `gdn_graph_decode=True` / workload flag `--gdn-graph-decode`. Default is unchanged. Singleton batch/time only; other widths use the existing forward. Each recurrent adapter owns one bounded `CapturedArrayInference` and a16MiB private pool. Every recurrent projection must be GPU-resident before graph construction is approved. Model preflight reserves36×16MiB=603,979,776bytes; realized graph pools in the sampled run reserve124,600,320bytes. Session memory planning credits actual graph allocations; close synchronizes before retiring graph/weight owners.

The existing wrapper copies caller input/state into private graph buffers and copies state/output back. Warmup/capture never advances caller state. Replay advances caller convolution and recurrence counters exactly once. Fresh outputs prevent later replay overwrites; replay is bound to one stream. This is not allocation-free whole-token capture: copies remain, as do the PLE/QSA/routing/CPU seams.

## Capture-safe kernels and precision

Initial tests found CuPy explicitly rejects cuBLAS API calls during capture. The localized correction reuses `ops/inference_matmul.py`'s existing Triton FP32 GEMV for graph-enabled dense projections. The captured and uncaptured controls use that same arithmetic. It differs from cuBLAS's FP32 summation order; no FP16/TF32 or new quantizer was introduced.

Fused DeltaNet kernels remain the existing implementation. An explicit `capture_owned` opt-in avoids global pending-event retention when the bounded graph adapter owns all arrays. Warmup is fenced on failure before private buffers retire. Default ordinary calls retain existing event-based lifetime protection. Qwen3-Next's API/defaults are preserved.

## Validation

After the one cuBLAS correction: GPU graph/recurrent/FP32 GEMV group **41passed,2skipped**. Tests cover32steps, alternating independent sessions, reset, native/fused coefficient paths, stream binding, graph state metadata rejection, no global pending refs from capture, and failure after private recurrence without caller-state mutation. Outputs/state arrays are bit-exact against the same-GEMV uncaptured path.

CPU model/generation/Qwen3-Next/mixer/workload group: **57passed,4skipped** (before later0268 additions; not a final whole-worktree claim).

Real checkpoint: sequential processes, fixed IDs1..16, two isolated requests, all selected experts computed with the same native CPUQ8 path and no GPU admissions. Logits and hashes of **all active recurrent, convolution, QSA and PLE state** match exactly at prefixes1/2/4/8/16 across capture/control. Reset logits match. Captured graph calls1,152. Artifact `0265_graph_parity_v1.json`, source captures `diagnostics_0265_parity_control_v2/` and `diagnostics_0265_parity_graph_v1/`.

The first parity probe retained request1's state while allocating request2; the aggregate state guard correctly rejected it. A localized diagnostic fix releases it before the next allocation. Partial `diagnostics_0265_parity_control_v1/` preserved. Both corrected processes completed. No model equation was changed to fix a diagnostic.

This establishes replay parity within the selected arithmetic, not new full-model llama.cpp equivalence or cuBLAS bitwise parity.

## Longer sampled workload

`diagnostics_0265_gdn_graph_v1/`: normal128token warmup, captured77token response prompt and512output cap, matching sampling settings,15CPU workers, frozen post-warmup placement, shared overlap, exactQ2 SIMD. Fully supervised, return0.273response tokens;272forwards/21.57591s = **12.60665forwards/s**. Previous0264 was11.42412/s but generated a different sequence after the dense GEMV arithmetic changed; do not call this an identical-input speedup ratio.

SampledRssAnon838,004,736bytes/RssFile55,145,779,200bytes, separate category maxima. Main CuPy pool12,332,611,584reserved bytes **plus** private graph pools124,600,320bytes; neither is globalVRAMhighwater. Child memory guards passed; no checkpoint copy or dequantized expert bank.

Next dependency: measure the current long-response critical path with correlated CPU/GPU events before choosing another optimization.50t/s remains unmet.
