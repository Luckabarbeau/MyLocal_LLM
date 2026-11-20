# Restart handoff — through Patch 0323

Work paused at the user's explicit request for a PC restart. **Do not launch further benchmarks before that restart.** All changes are uncommitted; preserve the worktree, checkpoints, pinned references, and prior artifacts. Do not reset/clean the repository.

## Where we stand

**50 complete decode forwards/s has NOT been reached.**

| Completed controlled experiment | Response 1 | Response 2 |
|---|---:|---:|
| Fresh Strata, full top10, no drafting, warm-dependent subset | 36.48/s | 49.26/s |
| Local graph/publication combination, 0318 | 40.69/s | 43.36/s |
| Local exact tiled-QSA + retirement correction, 0321 | **41.99/s** | **43.70/s** |
| Local whole-token graph, 0322 | 39.32/s | 42.22/s |

These use the retained prompts and **forced** continuations: 128 emitted warmup tokens, then two 359-output-token responses, each with 358 post-prefill decode forwards. They are not natural conversation rates, not an aggregate sustained 49.26/s Strata result, and not a full new `standard-v1` run. Both engines execute all routed top10 experts without model/suffix drafting. Strata has INT8 KV, a 7.233 GiB/3786-slot profile cache and different placement; local has FP32 KV and a 7 GiB/3673-slot cache. Cold reboot timings must be recorded separately.

## Final verified source state

- `0323_final_cpu_v1.xml`: **732 passed, 116 skipped, 1 deselected**.
- `0323_final_gpu_v1.xml`: **586 passed, 24 skipped, 1 deselected**.
- Selected Qwen3-Next regressions are included; some GPU modules are collection-skipped. No full training-suite claim.
- `diagnostics_0321_parity_v1`: ten saved actual-checkpoint logits and every saved persistent-state hash exactly match `diagnostics_0318_combined_parity_v1` over two requests at prefixes 1/2/4/8/16.
- `diagnostics_0322_parity_v3`: whole-token graph covers all48 layers, 12QSA layers and PLE; ten logits/all state hashes exactly match **its prepared arithmetic control**, `diagnostics_0322_baseline_parity_v1`.
- Full-model parity fixtures freeze empty GPU expert placement. Mixed/resident native execution has separate operator tests; this is not new independent llama.cpp full-model equivalence.
- Latest complete evidence/source hashes: `0318_0323_execution_results_v1.json`. Its analyzer is `scripts/analyze_0318_0323_results.py`; `--check` is read-only and loads no model/CUDA.

## What was implemented and what stays off

1. **0310:** bounded direct installed-Strata stdin harness. It checks exact executable SHA256, protocol, forced IDs/counts, cleanup and timing denominators; profiling uses Strata's internal stage reports.
2. **0311–0312:** diagnostic attribution and optional bounded PLE packed-row reader, at most8 I/O workers, 16 submitted reads/batch, 65536 cached90-byte rows. One batched IQ4_NL decoder preserves exact values. Cold PLE fault/latency reduction is established; no robust hot model gain.
3. **0313/0315:** complete recurrent-block and adjacent 2–3-block capture, borrowing frozen native-service plans. Bound request states, private warmup accounting and outside-capture retirement events are tested. No robust standalone model gain.
4. **0314:** cooperative Triton FP32 QSA is fast as an operator but changes reduction order; saved model logits differ3.6–6.9%. **Not quality accepted; leave off.**
5. **0316:** higher CPU spin and copied Strata affinity were tested and regressed. Do not repeat the same knobs blindly. Native service creation now precedes submitter pinning in the diagnostic setup, fixing inherited single-core confinement; the pinned setup still loses.
6. **0317:** complete QSA-block capture uses runtime positions/RoPE factors, request-bound caches and finite-validated conditional commits. It falls back outside proven all-cell short contexts or bounded cache payloads. No robust isolated model gain.
7. **0318:** publication/graph combinations tested without changing expert arithmetic; latest combined warm observation43.36/s.
8. **0319:** exact shared-memory QSA staging was slower. **Not integrated.**
9. **0320:** exact-order warp dots plus32-dimension value tiles are about2x faster as standalone QSA. Normal and dynamic-position graph dispatch are opt-in via `MINI_LLM_QSA_WARP_DOTS=1` / `--qsa-warp-dots`. No reduction-order change; original serial path remains available.
10. **0321:** retires orphaned bound request graphs before replacement-state admission, only with no live weakly registered states, under lock and fences. This fixes a demonstrated second-request VRAM-admission failure without changing budgets/headroom. Latest complete run41.99/43.70/s.
11. **0322:** whole-token capture reuses validated block math, keeps embedding/PLE lookup outside capture, binds recurrent/QSA state, returns owned logits and fresh immutable PLE history, and caps its private pool at256MiB (long-run reserved179653120bytes). **Leave off:** no throughput gain, and preparing PLE projections for capture changes old cuBLAS-path logits2.3–6.7%. Exact capture parity does not quality-accept that arithmetic change. The actual head is packed **Q6_K**; Q8 activation compute is not Q8 checkpoint encoding.
12. **0323:** decode-only submitting-thread cProfile wrapper; profile artifacts retained. No safety readback was removed.

All performance modes remain opt-in. No checkpoint, pinned reference mathematics, unrelated training code, or entire expert bank representation was changed.

## What the critical-path evidence actually says

The fresh pinned Strata pool source defaults to **20ms** bounded worker spinning; the local measured configuration is **50us**. More CPU usage is not automatically useful work: a 2000us local trial raised decode active-core equivalents from~9.5 to~18.2 while reducing warm throughput41.0→34.9/s. Native local CPU expert work is roughly8.7–9.1ms/modelstep versus Strata11.1ms, but expert placements/counts differ.

`0317_critical_path_v1.json` versus `0313_critical_path_v1.json`: host driver launches13264→3520 per16tokens, GPU span443.33→420.17ms. Larger captures did not yield the extrapolated long-run gain. Uncovered intervals and summed polling kernels are not useful GPU arithmetic or an additive CPU critical-path budget.

`diagnostics_0323_profile_v1/python_critical_path_pid1177617_instance001_repeat001.{json,txt,pstats}`:

- 5.385M calls,11.498instrumentedseconds; **not a throughput measurement**.
- 716CuPy.get calls consume9.470s;358asnumpy calls include9.464s. These await queued GPU/native work, not9s of physical PCIe data transfer.
- TokenGraph.forward0.920s, is_eligible0.544s, _validate_live0.474s, _configuration0.398s, _validate_states0.264s are **nested/overlapping**, not additive.
- Other/native workers are not profiled. CPU utilization cannot be inferred from this profile.

**Do not delete validation/fences based on wait-inclusive API time.** It would largely move the wait to sampling rather than eliminate the dependency.

## Single next investigation

After restoring a warm, comparable baseline, align native CPU task-phase timestamps with a current node-level GPU trace and the fresh Strata stage table. Rank **useful packed GPU operations and actual native CPU work by exposed critical-path time**, normalizing for expert site/count/format. Existing0298 timing support and latest0317/0323 artifacts are the starting points. Avoid another broad knob sweep, another cache copy, or another capture rewrite without evidence. Preserve the faster exact0321 configuration as the control; whole-token capture is an experimental negative result, not the recommended default.

## Paths and pinned references

- Repository: `/home/lucka/Documents/my_local_LLM/MyLocal_LLM`
- Python: `.venv/bin/python`
- Checkpoint, unchanged: `/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/`
- Installed engine: `/home/lucka/.local/share/strata/components/engine/0.1.59-linux-x86_64-cuda-cuda13/strata`
- Strata source: `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`
- Engine SHA256: `0ba914b28df1c8e09c0b5209c2ca6040c75fb650a7408ff3d9ef3ad159a13eb9`
- CPU GGML library: `/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so`
- Independent llama.cpp revision: `171e8846b4af9766c354064cb776cb34a50f053f`; preserve no-prefetch reference and original builds.
- `strata_launch_bench_capture.sh` is preserved and was consulted. Do not run unchanged merely to refresh a profile: it leaves a server running and the historical setup can pin~43GiB or request missing drafter assets. Use the bounded0310 harness for the controlled subset.

## Restart safety and commands

First, read this handoff, `implementation_status.md`, and `0318_0323_execution_report.md`. Check Git status without reverting anything, available memory, idle GPU and active model/benchmark services. No commit or branch was created. Only small read-only preflight work should occur outside isolation.

All model/GPU/large CPU experiments must use **`scripts/run_0290_isolated_benchmark.py`**. Keep12GiBhostreserve, service swap0, sequential model processes and exclusive artifact directories. No bulk50GiBwarming, complete pinned expert complement, concurrent Strata/local model, weight download or checkpoint writes. Original model runs used32/36GiBhigh/hard; later desktop pressure required **stricter30/34 then28/32GiB**. Guards refused several runs before loading, and one partial run failed VRAM admission. Do not bypass either guard or silently increase limits.

After reboot, file pages will be cold and newly charged to the benchmark cgroup. A previously successful warm run may safely abort under MemoryHigh. Do not relabel a cold/aborted request as a throughput result or increase budgets automatically. Recheck pressure and use the smallest informative experiment.

Read-only evidence check (requires unchanged source/evidence hashes, not CUDA):

```sh
.venv/bin/python scripts/analyze_0318_0323_results.py --check
```

Reproduce the exact0321 configuration, with **new** directory names. This example retains the stricter28/32GiB limits; its cold-cache result is not directly comparable to an old warm run:

```sh
MINI_LLM_PLE_BATCHED_ROWS=1 .venv/bin/python scripts/run_0290_isolated_benchmark.py \
  --memory-high-mib 28672 --memory-max-mib 32768 --reserve-gib 12 --timeout 200 \
  --output docs/qwen4exp/restart_0323_guard_r1 -- \
  .venv/bin/python scripts/probe_0302_decode_configuration.py \
    --output docs/qwen4exp/restart_0323_run_r1 \
    --shared-grid --recurrent-chain-graph --qsa-block-graph \
    --kernel-publish --qsa-warp-dots --expert-cache-gib 7 --response-repeats 2
```

Keep `--token-graph`, `--token-graph-baseline`, `--qsa-cooperative`, `--strata-affinity` and increased `--spin-us` **off** for this control. For a short exact-state fixture instead of throughput, use a new directory and `--parity`, omitting `--response-repeats 2`; its frozen CPU-expert scope must remain explicit.

## Reports to retain

- `0318_0323_execution_report.md` and `0318_0323_execution_results_v1.json` — latest source/artifact/guard/parity/profile ledger.
- `0310_0317_strata_comparison.md` and `0310_0317_results_v2.json` — prior source snapshot, Strata comparison and negative experiments. Historical hashes are intentionally not overwritten.
- `0323_final_cpu_v1.xml`, `0323_final_gpu_v1.xml` — final selected regression results.
- `numerical_root_cause.md`, `numerical_validation.md`, `memory_architecture.md`, `performance_history.md`, `reference_contract.md` — maintained scope and invariants.

**Resume instruction:** continue from the exact0321 control and measured critical path. Goal remains50/s with bounded memory and validated arithmetic; neither correct-looking text nor operator-only speedup completes that goal.
