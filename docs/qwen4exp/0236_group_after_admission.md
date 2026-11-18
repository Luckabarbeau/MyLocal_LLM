# Patch 0236 — group complete newly admitted experts

## Objective and files

Previously a single cache miss disabled grouped execution for the entire layer, even after every selected bundle had been synchronously promoted.

- `mini_llm/ops/expert_residency.py`: constructor `group_after_admission=False`.
- `mini_llm/external/qwen4exp_model.py`: forwards the explicit option to its owned executor.
- `tests/test_0236_group_after_admission.py`.
- `0229_decode_performance_v8.json`: preserves/reuses earlier successful variants.

## Execution contract

Inside the existing protected singleton dispatch, prepare each selected expert exactly once. If every complete bundle is GPU-resident and grouped shapes/formats are supported, use the existing three-launch grouped operator. An incomplete/unsupported/missing bundle still follows the existing mixed/CPU path. Profiling and non-singleton execution retain their old paths.

Router weights multiply once, followed by ascending expert-ID FP32 additions. Newly admitted handles are protected by the same scope/fence lifecycle as old residents. Counters distinguish `admission_grouped_dispatches`; no double request accounting or policy change is introduced. No new persistent buffer or full weight copy is allocated.

## Validation and performance

GPU focused run including grouped/batch regressions: **288 passed,1 skipped in2.03s**. NumPy singleton/batch regressions: **27 passed,29 skipped in0.34s**. Tests cover cold top10 across all7 actual triples, bit-exact sequential GPU output, one request per expert, current-stream dependency, a budget-forced CPU miss and failure protection cleanup.

Full-model forced-token rate: **9.174106→10.197347tokens/s**; same-prefix logits and state trajectories are unchanged by this scheduling-only addition. Latest optimization error versus original FP32 baseline remains ≤2.065315e-6 relative L2. RSS10,310,344,704 bytes; default pool9,829,571,072 reserved bytes, not global VRAM high-water.

Next dependency: improve the grouped IQ reduction/kernel schedule and transfer lifecycle. This patch does not implement partial-resident grouped combination or asynchronous admission.
