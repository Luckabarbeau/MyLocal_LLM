# 0255 — Opt-in heap expert-residency victim index

## Objective and status

Avoid sorting all resident experts on each full-pressure admission request, while retaining existing cache ownership, accounting, admission, and in-flight safety rules. The heap is **opt-in**; the existing sort policy remains the default.

Established validation: **23 CPU tests passed, 2 GPU tests skipped** in the CPU validation scope; a combined parent GPU validation scope recorded **37 passed**, including fence and packed-slot coverage. Representative differential testing covers **18,000 requests**. These are supplied completed results, not commands rerun for this documentation update. Source and relevant tests were inspected; no tests, benchmarks, or model workloads were run here.

## Files and API

- `mini_llm/ops/expert_victim_index.py`: bounded, handle-free lazy heap index.
- `mini_llm/ops/expert_residency.py`: cache updates, victim selection, and executor option forwarding.
- `mini_llm/external/qwen4exp_model.py`: model-level option forwarding.
- `tests/test_0255_expert_victim_index.py`: differential, safety, storage-bound, no-scan, and policy microbenchmark coverage.
- Related parent coverage includes `tests/test_0106a_expert_residency.py`, `tests/test_0106d_async_experts.py`, `tests/test_0106j_async_buffer_ownership.py`, and `tests/test_0238_packed_expert_slots.py`.

Selection is `AdaptiveExpertCache(..., victim_policy='heap')`, `HeterogeneousExpertExecutor(..., expert_victim_policy='heap')`, or the model's `expert_victim_policy='heap'`. All retain `'sort'` as the default; invalid policy values are rejected.

## Ranking relation and preserved invariants

At request step `t`, a resident's effective score is

```text
effective_score(t) = score * 2 ** (-(t - last_step) / H)
density(t) = effective_score(t) / bytes
log2(density(t)) = log2(score / bytes) + last_step/H - t/H
```

The final `-t/H` term is common to all residents and cancels for ranking. The implementation stores `log2(score) - log2(bytes) + last_step/H`, avoiding explicit exponentiation for each resident rank.

**This is not guaranteed policy identity in floating point.** Log rounding, very large timestamps, and underflow in the original effective-score expression can change near-tie ordering. Ties in heap priority use last request and resident insertion order. A test explicitly demonstrates that adjacent representable scores can collapse to the same log priority. Representative differential equality does not override this limitation.

The heap changes ranking only. Admission still evaluates the selected victim's exact current effective score and retains the existing comparison:

```text
requested_score / requested_bytes > replacement_margin * victim_density
```

The following contracts remain in place:

- Hits retain their original resident byte size, even when the request supplies a different size.
- Protected experts and experts with unfinished or unqueryable fences cannot be evicted. Nested protection remains reference-counted.
- Multiple victims are selected before eviction; insufficient eligible bytes cause CPU fallback without partial eviction.
- Victim selection retains keys/scalars, not weight handles. Victim handles are released before materialization so slots can be reused and budgets are not temporarily exceeded by retained references.
- Allocation OOM is counted and permits CPU fallback; unrelated materialization exceptions propagate.
- Idle eviction removes index entries; clear synchronizes tracked in-flight work and clears resident/index/usage state.

## Storage and complexity

Updates carry lazy generations; stale entries are discarded during selection. At normal operation boundaries, heap storage is bounded by **`2 * resident_count + 64`** through compaction of live index entries. The bound applies to heap metadata, not all cache memory or model weights.

The index stores only keys and scalar ranking metadata, never device/weight handles. Updates/removals are amortized logarithmic with occasional linear compaction; normal pressure requests do not sort or scan all resident ranks. Iteration restores temporarily removed live entries on early exit or exception. The cache serializes index mutation and iteration: the index is not independently thread-safe. Explicit idle eviction still scans/sorts residents and is not made scan-free by this change.

## Established validation

The supplied **23-pass/2-GPU-skip CPU** and **37-pass combined parent GPU** results cover the policy and surrounding ownership paths; they are different validation scopes and should not be summed into a new distinct-test count.

The focused differential test runs 6,000 requests for each half-life `17`, `193`, and `8192`: **18,000 total**, comparing request results, resident order, statistics, and byte/index coherence between sort and heap. Additional tests cover busy-victim restoration, multi-victim admission, failed admission, exact thresholds, OOM, propagated errors, idle eviction, clear/reinsertion, generations, deterministic ties, and weak-reference proof that handles are not retained.

A no-scan test forbids resident iteration and checks effective-score call counts: a hit evaluates the requested key, while the tested one-victim pressure request evaluates only the requested key and selected victim. Parent GPU results include fences and packed-slot interactions; CPU skips alone would not establish those GPU behaviors.

## Policy-only benchmark

The supplied CPU microbenchmark fills **2,419 resident scalar handles** and times **3,000 full-pressure misses**, with opposite sort/heap ordering across two rounds. Each round checks 3,000 evictions and coherent cache/index accounting.

| Policy | Supplied elapsed time |
| --- | ---: |
| Existing sort | approximately 2.154 s |
| Heap | approximately 0.0121–0.0125 s |

The measured policy-loop ratio is approximately **173–178×**. This is **not model speedup**: there are no model computations, real expert weights, device transfers, or GPU execution in this scalar-handle microbenchmark.

## Limits, next dependencies, and licensing

Keep the heap opt-in until matched real-workload measurements establish the effect on admission choices, cache hit rates, stalls, and end-to-end latency. Near-floating-tie policy divergence is documented behavior, not evidence of bit-exact sort equivalence. Heap metadata bounds do not establish a VRAM bound for temporary allocations outside the cache's existing budget contract. This work alone does not establish the prolonged 50-token/s goal.

The index uses Python standard-library facilities and adds no external binary dependency or upstream source extraction. Repository LGPL terms remain applicable; this policy change introduces no separate third-party licensing requirement.
