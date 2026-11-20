# 0318–0323 — retained execution results

## Decision and evidence boundary

**Implemented and scope-validated, not model-performance-optimized; 50 complete decode forwards/s remains unmet.**
Best new controlled warm response: **43.70105 forwards/s** (0321); this is not a sustained aggregate.
0319 remains unintegrated. Whole-token graph and its prepared PLE arithmetic remain **off**.
No production code, model/GPU execution, old report refresh, or parent status-document edit accompanies this report.
Parent owns updates to `implementation_status.md` and `performance_history.md`; their older headings remain historical here.
The companion `0318_0323_execution_results_v1.json` contains artifact/source SHA256s, per-patch file roles,
raw timing denominators, saved parity comparisons, operator results, profile records, JUnit skips and every retained guard.
Current hashes describe this working tree, not reconstructed historical source identity.
Existing `0310_0317_results_v1.json` and `v2.json`, including their source-hash verifications, remain untouched.

## Measurement contract and retained rates

Each long response emits **359 forced IDs / 358 post-prefill model forwards**, after a 128-output-ID warmup.
Reported rate is `358 / decode_wall_seconds_after_first_token`, including the full output head and sampling.
Keep response 1 and response 2 separate; do not substitute emitted-N/decode-second or N/total-second.
Forced replay is not natural-conversation quality/throughput, and not the unchanged full `standard-v1` workload.
Local full top10/context1024 uses FP32 KV, 19 CPU workers and 7 GiB expert cache; no expert tail is discarded.
Strata's INT8 KV, effective cache and CPU/GPU placement differ; see `0310_0317_strata_comparison.md`.
Cold/warm I/O and cgroup page-cache charging differ between runs: these observations do **not** prove small gains.

| Retained run (`diagnostics_` prefix) | Response 1 / response 2 forwards/s | Interpretation |
|---|---:|---|
| `0318_publish_long_v1` | 40.72434 / 41.49273 | Existing publication + recurrent/QSA block graphs |
| `0318_combined_long_v1` | 40.69313 / 43.36186 | Existing publication + recurrent chain + QSA block |
| `0320_long_v1` | 42.19665 / **not completed** | Request 2 rejected by physical VRAM guard |
| `0321_long_v1` | 41.99128 / 43.70105 | Resource fix permits complete two-response execution |
| `0322_long_v1` | 39.32466 / 42.22064 | Whole-token graph gives **no gain** |

0323's profiled response is diagnostic-only: profiler overhead is included, not a competing benchmark rate.
No terminal 50/s claim, unchanged-launcher claim, or full-configuration-equivalence claim is made.

## 0318 — reuse kernel publication with newer graphs; no new production code

**Objective/interface:** test existing `--kernel-publish` with `--shared-grid`, `--qsa-block-graph`,
and either `--recurrent-block-graph` (publish) or `--recurrent-chain-graph` (combined).
Reused files: `scripts/probe_0302_decode_configuration.py`, `mini_llm/ops/native_expert_handoff.py`,
`mini_llm/external/qwen4exp_recurrent_block_graph.py`, `qwen4exp_recurrent_chain_graph.py`,
and `qwen4exp_qsa_block_graph.py` in that same external directory; **0318 adds no production module**.
Tests/numerics: both `0318_{publish,combined}_parity_v1` match `0317_parity_v1` at all ten saved logits
(two requests × prefixes 1/2/4/8/16) and all saved recurrent/QSA/PLE state records.
Memory: existing publication ordering, fences, bounded state and guards are retained; no new residency budget.
Performance is the two separate pairs above, not an additive claim for each switch.

| 16-token trace artifact | GPU span ms | Useful non-wait union ms | Wait-kernel sum ms | GraphLaunch / token |
|---|---:|---:|---:|---:|
| `0313_critical_path_v1.json` | 443.334255 | 215.746344 | 137.167184 | 992 / 16 = 62 |
| `0317_critical_path_v1.json` | 420.166631 | 203.745185 | 114.231601 | 800 / 16 = 50 |

Host driver `cuLaunchKernel` calls fall **13,264 → 3,520**. Gaps preceding publication in 0317 total
**44.610139 ms**, but include dependencies/scheduling and overlap CPU/API spans: **not an exclusive critical budget**.
Next dependency: attribute remaining useful execution, not assume every uncovered gap is removable host overhead.

## 0319 — exact shared staging; deliberately unintegrated

Files: `mini_llm/ops/qsa_staged_gpu.py`, `scripts/probe_0319_qsa_staged_operator.py`,
`tests/test_0319_qsa_staged.py`. Interface: standalone `qsa_staged_gpu` / fail-closed `staged_source`.
Objective: coalesced key loads through a padded **256×33 FP32 shared tile (33 KiB)**, no context-sized scratch.
Each cell still accumulates d=0..255 left-to-right; original 128-cell softmax, weighted V and sigmoid remain unchanged,
with `--fmad=false`, no cache writes, and exact FP32 bitwise checks across supported views.
`0319_gpu_tests_v1.xml`: **78 passed / 1 skipped** (second GPU required).
`0319_qsa_operator_v1.json` serial/staged ratios at n=141/435/1024:
**0.620000 / 0.559672 / 0.545307**, i.e. roughly **1.6–1.8× slower**, despite exact arithmetic.
Status: operator-correct, performance-rejected, **not integrated**. Next dependency: expose parallel output work.

## 0320 — exact warp dots, then localized value tile32

Files: `mini_llm/ops/qsa_warp_dots_gpu.py`, `qsa_decode_gpu.py`, `qsa_graph_gpu.py` in the same ops directory;
`scripts/probe_0320_qsa_warp_dots_operator.py`, `scripts/probe_0302_decode_configuration.py`,
`tests/test_0320_qsa_warp_dots.py`, `tests/test_0320_qsa_graph_dispatch.py`.
Objective/interface: `MINI_LLM_QSA_WARP_DOTS=1` / `--qsa-warp-dots`, normal and dynamic-position graph dispatch.
Warp-per-dot coalesces products but shuffles them back into the **same d=0..255 FP32 addition order**.
Untiled `0320_qsa_operator_v1.json` kernel ratios **1.0262 / 0.9017 / 0.8203** are neutral/slower.
The localized **value-dimension tile32** partitions independent outputs, not reductions: each dimension retains
**every FP32 operation order**, including max/softmax/weighted-V/gate; no FP16 or approximation is introduced.
`0320_qsa_tile32_operator_v2.json`, n=141/435/1024: kernel **2.38309 / 2.05060 / 1.84916×**,
API **2.00200 / 1.97022 / 1.88397×**. Operator gains are not automatically model gains.
Tests: **92** initial warp tests, **168** tiled tests, **231** dispatch/integration tests; suites overlap.
Logits scratch is bounded by **24×8192 FP32 = 786,432 bytes (768 KiB)** plus output; D=256 and ≤24 heads.
Dynamic graphs use runtime positions with fixed-capacity bounded scratch, not capture-time host positions;
only committed rows plus current k/v are read, and caller inputs remain unchanged.
Actual full-model `0320_parity_v2` matches `0318_combined_parity_v1`: **10 logits / all saved states exact**.
`0320_parity_guard_v1` was refused by RAM preflight **before launch and before any artifact directory**;
there are no v1 results to analyze. Successful v2 used **30/34 GiB** high/max caps.
`0320_long_v1` retains warmup and first response **42.19665357/s**, but has **no completed result.json**.
On request 2, `create_generation_state → memory_plan` raises `MemoryError`: need **1,861,791,744** bytes,
free **1,820,459,008** (short **41,332,736**). This is admission/resource failure, not arithmetic failure.
Next dependency: release bound orphaned-request graphs without crediting or retiring any live request.

## 0321 — genuine execution-resource fix, not guard relaxation

Files: `mini_llm/external/qwen4exp_model.py`, `tests/test_0321_orphaned_request_graphs.py`.
`_retire_orphaned_state_graphs` runs **before allocation/admission**, only when the weak live-state set is empty.
It acquires the call lock, rechecks live states, closes/fences graphs retaining request arrays, then releases
unused default-pool blocks. Concurrent forward/live requests remain protected; no cache or VRAM guard is reduced.
Tests: **8 focused**; `0321_cpu_tests_v1.xml` **67 passed / 20 skipped** in the selected CPU suite.
Numerics: `0321_parity_v1` vs `0318_combined_parity_v1`: **10 exact logits / all saved states exact**.
`0321_long_v1` completes both responses (**41.99128 / 43.70105/s**), with 7 GiB cache unchanged.
Status: scope-validated resource fix, not a robust throughput optimization. Next dependency: whole-token scheduling test.

## 0322 — whole-token graph, prepared-math parity only; remains off

Files: `mini_llm/external/qwen4exp_token_graph.py`, `qwen4exp_model.py`, `qwen4exp_ple.py` in that directory;
`scripts/probe_0302_decode_configuration.py`, `tests/test_0322_token_graph.py`.
Reused interfaces: existing recurrent/QSA block adapters, `mini_llm/ops/qsa_graph_gpu.py`,
`mini_llm/ops/inference_graph.py`, capturable projection in `mini_llm/external/qwen4exp_weights.py`,
and existing `scripts/probe_0265_gdn_graph_parity.py`; no new independent full-reference implementation.
`MINI_LLM_TOKEN_GRAPH=1` / `--token-graph` captures all **48 blocks / 12 QSA / one PLE at layer 1**.
Expert plans must already be complete, frozen CPU/GPU plans with matching stream/device/fingerprint.
Runtime QSA positions/GPU factors stay runtime inputs. Bounded all-cells context and state geometry are checked.
CPU PLE row lookup/hash stays **outside capture**; existing `apply_embedding` is read-only pure GPU math.
Original immutable `PLEState` is not mutated; new owned logits/history survive replay and graph retirement.
Request-bound arrays force recapture; old subgraph pools retire **before** whole-graph allocation;
model close fences the token graph **before** releasing borrowed expert/native/weight handles.
Private pool cap **256 MiB**; long-run reserved **179,653,120 bytes** (not a measured device-wide live peak).
`0322_gpu_tests_v1.xml`: **56 passed**; final-source tests also cover supported packed heads and alias/lifecycle rules.

First `0322_parity_v1` failed: CuPy cuBLAS PLE-key projection refuses capture; guard exit 1, no result.json.
Localized fix: `MINI_LLM_TOKEN_GRAPH_PREP=1` selects existing capturable **FP32 Triton matmul** PLE projections.
`--token-graph-baseline` prepares the **same math without token replay**; it is not the old cuBLAS baseline.
`0322_baseline_parity_v1` vs `0322_parity_v3`: **10 exact logits / all saved states exact**;
v3 records **30 token-graph calls, 2 captures/requests, 48/12/PLE1**.
Against old PLE (`0318_combined_parity_v1`), prepared prefix 1/2/4/8/16 max abs differences are
**0.397332 / 0.581697 / 0.499264 / 0.931747 / 0.885948**; relative L2
**0.023090 / 0.031244 / 0.031301 / 0.067017 / 0.067354**, repeated identically for both requests.
**NOT quality-accepted relative to the old path**: FP32 projection/reduction-order sensitivity, not FP16 or graph failure.
The independent full reference is unchanged. Within-prepared-path equality does not close the old-path quality gate.
Output-head storage is actually **Q6_K**, not Q8_0 (`tensor_inventory.json`); Q8 activation compute is not weight encoding.
Current eligibility/source tests accept supported resident packed formats; the earlier Q8_0-only assumption was corrected.
`0322_parity_guard_v2` RAM preflight refused before directory creation: **no artifacts/results**; actual successful candidate is v3.
Long run **39.32466 / 42.22064/s** gives no gain. Keep token graph **and PREP off**; next: useful-work ranking, not fence deletion.

## 0323 — exact-boundary submitting-thread cProfile

Files: `scripts/probe_0323_python_critical_path.py`, `tests/test_0323_python_critical_path.py` (35 tests in final CPU XML).
The wrapper retains 0290 isolation/0302 flags and profiles **after first original sampler return → after last return**,
not prefill, text decoding, journal finalization or cleanup. It adds no GPU synchronization/events.
Actual files under `diagnostics_0323_profile_v1`: `python_critical_path_pid1177617_instance001_repeat001.{json,txt,pstats}`.
Analyzer reads JSON, hashes text/pstats, never unpickles pstats. Recorded **5,385,211 calls / 11.498236 profiled host seconds**.
716 CuPy `.get` calls: **9.470211 s** cumulative; 358 `asnumpy`: **9.464072 s** cumulative.
Token graph `forward` **0.919657 s**, eligibility **0.544281 s**, `_validate_live` **0.473639 s**,
`_configuration` **0.398176 s**, `_validate_states` **0.264035 s** cumulative: **overlapping, do not add**.
Tiny GPU status reads wait for already queued GPU/native work: **not nine seconds of physical PCIe transfer**.
Dropping a check generally moves the wait; it does not establish saved critical time or permit deleting fences.
Native worker threads are **not profiled**; CPU utilization cannot be inferred from this submitting-thread profile.
Sorted source/state alias checks remain **O(n log n)** and protect ownership; frozen/live-plan validation is retained.
Best underlying CPU measurement remains about **8.7 ms/model step**, versus Strata **11.1 ms/step** under different placement
(historical 0310 report). Earlier 50→2000 µs spin doubled active cores but regressed; not a useful new knob.

## Final source verification and memory safety

User execution record confirms final production source is the **same as the last model parity after all production changes**.
This is an execution-record attestation; current hashes alone cannot prove historical identity. No production edits made here.
`0323_final_cpu_v1.xml`: **732 passed / 116 skipped / 1 deselected**.
`0323_final_gpu_v1.xml`: **586 passed / 24 skipped / 1 deselected**; Qwen3-Next GPU modules have explicit collection skips.
Deselections come from the execution record (JUnit does not encode them); exact selected commands are parsed from guards.
CPU and GPU suites overlap and have backend/original-model/two-device skips: not universal coverage. **Full training not run.**

All **24** retained 0318–0323 `guard.jsonl` files are parsed, including failed executions and final regressions.
Authority is sampled service/cgroup `memory_peak` and events, not tiny systemd parent RSS. **12 GiB reserve remains unchanged**;
`memory.events` high/max/oom/oom_kill are zero. Nonzero PSI exists (up to **1.08 some/full avg10**); do not claim no pressure.
Model high/max caps tighten **32/36 → 30/34 → 28/32 GiB** as editor RSS grows; **swap.max=0**, sequential processes.
Latest model commands below use 28/32, not relaxed caps. VRAM admission and **7 GiB expert cache** are unchanged.
Profile charged peak **26,674,728,960 bytes** differs strongly from warm-run charges: cached/shared file pages matter.
Final CuPy used/reserved and graph-pool bytes are allocator snapshots, **not NVML totals or measured capture-time live peaks**.
RAM preflight refusals above are not fabricated guard/results artifacts; the two retained nonzero exits remain historical failures.

## Reproduction and next dependency

From repository root, artifact-only commands (no model/backend imports):
```sh
python3 -B scripts/analyze_0318_0323_results.py
python3 -B scripts/analyze_0318_0323_results.py --check
```
Creation uses exclusive `open('x')`, refuses overwrite; `--check` writes nothing and compares the whole JSON snapshot.
Manifest covers inputs/report and Git-listed source hashes, including every modified/staged/deleted/untracked source path;
old weights/assets, binary models and diagnostic payload trees are excluded from the **source** path inventory.
Saved parity payloads themselves are hashed/read as bounded FP32 NPY with stdlib, never NumPy/pickle/model execution.

**Optional future model reproduction only; NOT executed by this analysis.** Use fresh guard/run directories, one command at a time;
existing local checkpoint/tokenizer/native-library dependencies must already be available. Do not download or bulk-warm weights.
Latest completed old-PLE exact pipeline (0321 flags; whole-token/PREP off), tightened safe caps:
```sh
.venv/bin/python scripts/run_0290_isolated_benchmark.py --memory-high-mib 28672 --memory-max-mib 32768 --reserve-gib 12 --timeout 210 --output docs/qwen4exp/diagnostics_0318_0323_repro_guard_v1 -- .venv/bin/python scripts/probe_0302_decode_configuration.py --output docs/qwen4exp/diagnostics_0318_0323_repro_v1 --workers 19 --task-factor 3 --spin-us 50 --shared-grid --recurrent-chain-graph --qsa-block-graph --kernel-publish --qsa-warp-dots --expert-cache-gib 7 --response-repeats 2
```
For exact saved-prefix verification instead, use **different new** directories and replace `--response-repeats 2` with `--parity`.
The runner writes exclusive artifact directories, verifies caps/swap/reserve before launch, and may safely refuse RAM preflight.
These flags reproduce an **opt-in measured configuration**, not recommended defaults. General experiments should keep unproven
switches off: omit chain/block/kernel-publication/warp-dot/shared-grid options unless testing that exact configuration;
never add `--token-graph` or `--token-graph-baseline` as a default improvement. Keep checks/fences and complete routing.
Next single useful experiment: **rank remaining useful packed-GPU and native-CPU execution against matched hot Strata with identical
expert placement and CPU counts**, retaining current safety/quality gates. No speculative new scheduler knob or validation removal.
