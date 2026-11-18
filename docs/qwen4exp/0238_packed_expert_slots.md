# Patch 0238 — fixed-slot packed expert cache foundation

## Objective/status

Implement the fixed-storage foundation of logical Patch0214 without claiming asynchronous movement. Replace arbitrary per-admission GPU weight allocation with bounded reusable slots; retain canonical mmap sources and safe CPU fallback.

**Status: numerically validated synchronous slot subpath; full asynchronous mover in progress.**

## Files and interfaces

- New `mini_llm/ops/packed_expert_slots.py`: `PackedExpertSlotArena`.
- `mini_llm/ops/expert_residency.py`: `expert_slot_stride_bytes=0`, `resident_device_bytes` and slot statistics.
- `mini_llm/external/qwen4exp_model.py`: `expert_cache_slots=False`, header-derived largest complete bundle, staging reservation and physical arena accounting.
- `tests/test_0238_packed_expert_slots.py`.
- `0229_decode_performance_v10.json`, `0233_fast_lifecycle_v2.json`.

## Ownership, budgets and publication

One lazily allocated GPU arena; fixed stride rounds the largest gate/up/down bundle to256 bytes. Charge padded stride per cached expert, not just payload bytes. Empty slots still occupy VRAM, so physical replanning credits the whole owned arena independently of cached-handle counts.

One reusable pinned staging slab packs all three projections and one dedicated copy stream transfers the complete bundle. Publication occurs only after copy-stream completion. This is intentionally synchronous: there are no unready resident handles or an unbounded prefetch queue.

Projection arrays use `UnownedMemory` leases. A retained handle **or projection view** keeps its slot unavailable after cache eviction; exhausted leases cause safe CPU fallback rather than overwriting old views. Existing cache fences protect outstanding GPU consumers, and executor close waits for arena consumers before retirement. Unknown DMA completion quarantines a slot instead of recycling it. Callers of unmanaged raw handles retain the usual obligation to preserve weights through stream completion.

CPU inputs, canonical weights and PLE remain mapped/bounded. No whole expert bank is expanded, copied to pinned RAM or allocated on GPU. Arena GPU allocation happens once, not per token/miss.

## Failure and tests

Initial focused GPU run exposed an API error: CuPy `PinnedMemoryPointer` has no `len()`. One localized correction measures the buffer protocol's actual rounded byte size with `memoryview(...).nbytes`; the rerun passes.

GPU focused slot/admission/ownership tests: **32 passed in2.19s**. CPU planner/model tests:27 passed/14 skipped. Tests cover all7 format triples, exact packed bytes and GPU output, projection-view retention, stable address reuse, padded accounting, forced CPU fallback, copy-complete-before-publication, quarantine, staging limits and physical-budget credits.

## Actual original-model results

For6GiB expert budget: stride2,662,400 bytes, capacity2,419 slots, actual arena6,440,345,600 bytes and pinned staging4,194,304 bytes.

Forced16-token run retains1,952 bundles, charges5,197,004,800 padded resident bytes, and transfers3,994,496,000 payload bytes. Rate **11.976041tokens/s**, versus12.033481 dynamic-cache warp path; this single-run0.5% difference is not a meaningful speedup/regression claim. All checked logits and active-state hashes are **bit-exact** between these two paths. RSS10,302,685,184 bytes; default pool12,267,438,592 reserved bytes, not global VRAM high-water.

Extended optimized lifecycle passes:16-token incremental/prefill logits and all active-state hashes identical; two fresh greedy requests both return `[220,100561,198]`. It retains2,332/2,419 slots, charges6,208,716,800 bytes and reports no quarantines.No throughput is inferred from the lifecycle workload: actual child elapsed11.1123s, sampled RSS11,180,748,800 bytes and pool12,270,584,320 reserved bytes.

Next dependency: asynchronous staging/shadow-slot admission, completion publication and bounded movement, then realistic longer cache-pressure benchmarks. Existing slots are not an asynchronous mover; fixed capacity/padding can lower useful cache capacity versus dynamic allocation, so keep this policy independently selectable.

## Tested configuration on the16GiB RTX5080

Set backend/kernel options in a **fresh process before importing MyLocal_LLM**. These explicit budgets were validated for the short test context, not arbitrary context lengths or smaller/free-memory-constrained GPUs. The normal constructor defaults remain conservative and preserve opt-outs.

```python
import os
os.environ.update(
    MINI_LLM_BACKEND='cupy',
    MINI_LLM_IQ_GPU_CODEBOOK_STORAGE='global',
    MINI_LLM_IQ_GPU_GROUPED_KERNEL='warp',
)
from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel

GiB = 1024**3
path = '/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S'
with Qwen4ExpGGUFModel.open(
    path, max_length=16, cpu_workers=4,
    static_device_budget_bytes=6*GiB,
    device_budget_bytes=14*GiB,
    cache_bytes=6*GiB,
    expert_promote_after=1,
    group_after_admission=True,
    expert_cache_slots=True,
    qsa_fast_decode=True,
    hc_fast_decode=True,
    router_fast_decode=True,
    gdn_fast_decode=True,
    hc_graph_decode=False,
) as model:
    print(model.greedy([1, 2], max_new_tokens=3))
```

Expected smoke IDs: `[220,100561,198]`. This example uses diagnostic token IDs, not a chat/tokenizer integration or a throughput benchmark. GPU admission remains budget-checked, expert sources remain mmap-backed, and cache pressure can trigger the retained native CPU path.
