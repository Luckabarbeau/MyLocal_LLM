# 0267 — Reduce exposed native worker wake-up latency

## Objective / source basis

0266 measured321.591ms of CPU completion waits in16decode tokens. Current pool parks workers at each phase; pinned Strata's worker loop uses bounded spinning before safe sleep. Adapt that mechanism without replacing the row scheduler or removing barriers.

Files: `mini_llm/ops/ggml_packed_cpu.{cpp,py}`, `scripts/probe_0254_strata_workload.py`, `tests/test_0267_native_pool_spin.py`.

## Interface and ownership

`GGMLPackedCPURowPool(..., spin_us=None)` uses explicit value or `MINI_LLM_GGML_POOL_SPIN_US`, default0 (sleeping). Accepted0..2000microseconds. Native setter configures an idle pool; busy updates are rejected. Benchmark records the selected value.500microseconds is opt-in, not a new general default.

Epoch/phase/busy publication becomes atomic; existing mutex/condition-variable phase barriers remain. Workers poll with pause/yield for a bounded interval, then reacquire the same mutex and use the original predicate-based wait. Stop/drain/join and all-worker barriers remain intact; no late worker is permitted to use a new batch's descriptors. Cache-line separation reduces control-variable contention. No new worker threads, expert allocations, arithmetic or router weighting.

The budget is not a realtime scheduling guarantee: a preempted thread can resume after the requested wall time. Spinning increases CPU utilization/power and may hurt oversubscribed machines. No physical-core affinity or host task draining was ported in this patch.

## Validation

0267/0259/0261 group withQ2 SIMD: **57passed,2skipped**, including optional timing. Tests cover spin0/50/500/2000,1/4/15workers,60back-to-back epochs perconfiguration, varying1..10experts, occasional3ms idle gaps, asynchronous wait, invalid settings, exact row arithmetic and shutdown. No sanitizer/race-proof claim beyond these tests and the preserved synchronization protocol.

SyntheticD2560/F640,5experts,15workers,100warmed samples with300us inter-batch gaps: median0spin360.011us;50spin346.843us;500spin175.846us. About2.05x latency reduction at500us in this test, not model throughput.

## Longer model experiment

`diagnostics_0267_spin500_v1/` completed fully supervised using the same0265options exceptspin500. Warmup128 and response273tokenIDs **exactly match0265**. Response272forwards/17.88016s = **15.21239forwards/s**, compared with12.60665/s sleeping. Observed1.2067x rate in one sequential pair; not randomized confidence bounds. Prefill4.89423s; emitted273/total22.77508s =11.98678/s. Keep these timing definitions distinct.

SampledRssAnon838,385,664bytes/RssFile54,895,443,968bytes, separate category maxima. No additional modelweight storage. Graph/VRAM reservations unchanged; source pages remain reclaimable and memory guards passed.

This is the **last completed model performance experiment before0268's experimental source edits**.0268 remains default-off and has known failing tests; do not claim the complete final worktree is verified or50t/s reached. Next dependency: validate persistent device planning to attack remaining host submission overhead.
