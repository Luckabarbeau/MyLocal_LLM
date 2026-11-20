# 0290 — Desktop OOM incident and isolated benchmark execution

## Objective and current disposition

Protect the workstation before resuming expert-residency optimization. **GPU initialization, pinning, large anonymous expert copies and full-model runs remain paused.** This patch adds diagnostic execution containment; it does not change model equations, checkpoint bytes, reference code or inference precision. The 50 tokens/s objective remains unmet.

## Confirmed incident evidence

At **2026-11-19 19:06:43**, systemd-oomd reported:

```text
Killed /user.slice/user-1000.slice/user@1000.service/session.slice/org.gnome.Shell@x11.service
... memory pressure for /user.slice/user-1000.slice/user@1000.service
being 87.49% > 50.00% for > 20s with reclaim activity
```

The victim cgroup held **50.6G** according to the journal. Evidence is preserved in `0290_oom_kill_evidence_v1.log` and `0290_oom_candidates_v1.log`. The earlier investigation recorded GNOME's unit result as `oom-kill`, 16 processes killed, and an unchanged multi-day kernel uptime: evidence supports a desktop-session OOM kill, not a machine reboot. Earlier NVIDIA allocation failures are not established as the cause of this incident.

The current editor terminal's `/proc/self/cgroup` also identifies the GNOME Shell service. Earlier `start_new_session=True` subprocesses inherited that cgroup; separate process groups are **not memory isolation**. Existing sampled RssAnon/MemAvailable guards did not protect against file-cache residency, parent-cgroup pressure or reclaim-triggered desktop victim selection.

### Attribution remains unresolved

The OOM journal identifies the victim cgroup, not its largest individual process. The 44GiB resident-complement benchmark was **not launched by the agent** before the report. The preceding focused tests passed; that neither demonstrates causation nor rules out contributions from earlier inference work or other desktop processes. No pre-incident per-process cgroup accounting was preserved that resolves attribution. Do not invent an offending PID from these logs.

The useful next attribution experiment is prospective: retain cgroup-level memory, PSI and event telemetry for every isolated run, along with its workload command and timestamps. Do not recreate the OOM.

## Implementation and contracts

### `scripts/run_0290_isolated_benchmark.py`

Standard-library-only launcher; importing/running the guard initializes no CUDA or model backend.

1. Starts a uniquely named transient **systemd user service** under `app.slice`, not a desktop child scope. The small forwarding launcher remains in the desktop cgroup; the gate, model process and its ordinary descendants run in the new service.
2. Requests `MemoryHigh`, `MemoryMax`, `MemorySwapMax=0`, `OOMPolicy=kill`, `KillMode=control-group`, bounded runtime and five-second systemd stop timeout.
3. Before spawning the workload, the gate verifies its actual unified cgroup path and kernel `memory.high`, `memory.max`, `memory.swap.max`, and `memory.oom.group`. Missing delegation/controller files, unlimited bounds, wrong placement or unavailable systemd **refuse execution**. There is no uncontained fallback.
4. Default hard limit **1GiB**, high threshold **768MiB**, host reserve **12GiB**, timeout **120s**. The complete hard limit plus reserve must fit current available memory before launch. CLI cannot reduce the reserve below12GiB or extend timeout above1800s.
5. Polls every0.5s: cgroup current/peak/events, host availability, finite ancestor high/max headroom, PSI from own/ancestor cgroups and host. Stops at its high threshold, reserve violation, OOM event, `full avg10 >=10%`, `some avg10 >=30%`, or timeout. The polling guard is supplemental; kernel limits remain authoritative during polling gaps.
6. Retires the child process group on abort. The outer launcher also stops the entire service on completion/interruption; systemd bounds and kills remaining service descendants.
7. Creates an exclusive artifact directory and line-flushed `guard.jsonl` with verified pre-exec limits, workload argv, samples and normal/aborted termination. Prior artifacts are not overwritten.
8. Explicit inference/GGML environment and BLAS/OpenMP thread choices are forwarded to the service. This is not a general secret/environment export mechanism.

`require_isolation()` validates the service/limits for descendant model probes. `scripts/probe_0254_strata_workload.py` now invokes it before either parent or child execution, prior to headers, GPU initialization or model construction. **Historical direct CLI commands now intentionally refuse execution**; wrap them in the isolated launcher. Imported helper functions and synthetic tests remain usable.

### Limits of protection

- This is not a claim that the workstation can no longer OOM. Sibling applications and shared user-manager ancestors still exist; PSI/reserve monitoring is sampled, not atomic admission against all processes.
- Kernel cgroup accounting includes charged file cache, not only RssAnon. Previously charged/shared pages and GPU-driver allocations complicate total physical-memory attribution.
- There is **no VRAM cap** in this launcher. Existing model VRAM/headroom validation is still required; driver failures cannot be contained by host cgroup limits alone.
- No OOM protections were disabled. No GPU reset, cache dropping, unrelated process killing, checkpoint rewriting or large allocation stress test was performed.
- The copied anonymous source plus canonical file pages may temporarily demand substantially more memory than the final complement alone. A nominal host-expert budget does not constitute a safe process-memory budget.

## Validation and artifacts

All new validation used CPU-only/no-model workloads. No large allocation or intentional OOM test.

| Experiment | Result | Artifacts |
|---|---|---|
| Isolated `/usr/bin/true` | Exit0; verified separate service,1GiB max/768MiB high/swap0/group OOM | `diagnostics_0290_isolation_smoke_v1/guard.jsonl` |
| Initial guard unit tests |17passed|`0290_guard_tests_v1.xml`, `diagnostics_0290_guard_tests_v1/`|
| Final guard + resident-source/model-budget/native-pool/workload synthetic regressions |**89passed,4skipped,1deselected**|`0290_cpu_regression_v1.xml`, `diagnostics_0290_cpu_regression_v1/`|
| One-second timeout of `/usr/bin/sleep 30` |Expected refusal/abort exit125; ~1.03s service lifetime|`diagnostics_0290_timeout_v1/guard.jsonl`|
| Post-test service listing |Zero `mylocal-bench-*.service` units loaded|Observed with `systemctl --user list-units`|
| Tracked diff whitespace check |`git diff --check` passed|No production numerical edits|

The final regression's kernel cgroup `memory.peak` was **643,858,432 bytes (~614MiB)**; minimum sampled host available **56,884,445,184 bytes**. Maximum sampled host/ancestor/own PSI some/full avg10 was0. These are cgroup telemetry scopes, not process RSS or VRAM measurements. The transient systemd summary printed a different small memory-peak figure; the retained kernel `memory.peak` readings are the reported measurement here.

The final tests covered membership/layout validation, refused unlimited/wrong/swap-enabled limits, reserve/pressure decisions, malformed PSI, independent artifact preservation, inherited thread settings, failed-child propagation, root/ancestor memory accounting, and workload CLI refusal before dispatch. CPU source regressions covered tiny exact packed copies/readonly ownership, budget failure, native CPU source use and same-arithmetic tiny-model parity. CUDA-only tests were skipped; the actual-checkpoint tokenizer preflight was explicitly deselected.

Reproduction of the bounded CPU regression (use **new** artifact/XML names):

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so \
.venv/bin/python scripts/run_0290_isolated_benchmark.py \
  --output docs/qwen4exp/diagnostics_0290_cpu_regression_v2 --timeout 90 -- \
  .venv/bin/python -m pytest -q \
  tests/test_0290_isolated_benchmark.py tests/test_0289_resident_expert_source.py \
  tests/test_0209_qwen4exp_model.py tests/test_0229_resident_budget.py \
  tests/test_0259_ggml_row_pool.py tests/test_0254_strata_workload.py \
  -k 'not actual_installed_tokenizer' \
  --junitxml=docs/qwen4exp/0290_cpu_regression_v2.xml
```

## Patch0289 disposition and next dependency

0289's optional compact anonymous CPU source is **implemented and small-fixture validated**, not large-memory validated or performance optimized. Canonical GGUF ownership stays immutable; complete copied bundles only shadow CPU reads. Native registrations are setup-only and retired before replacement. Optional paced CUDA registration has earlier small-fixture validation, but was **not rerun after the incident**. No44GiB copy/pinning result or speedup exists.

Previously measured mmap warming was substantial but incomplete/reclaimable:50,292,326,400 total expert bytes,47,207,305,216 planned/touched,44,229,771,264 estimated resident, zero pinned source bytes. The retained0288 response accumulated16,555major faults and1,428,045,824disk-read bytes. This motivates testing stable residency; it does not establish that pinning a large complement is safe or that it improves throughput.

**Next dependency:** review the incident/limits, then validate a small isolated CPU-only anonymous source (no pinning/model payload) with measured physical/cgroup memory and cleanup. Increase budgets only conservatively with the12GiB host reserve and all kernel limits intact. Do not jump to a44GiB source. Only after safe residency is measured should packed GPU-staging/CPU split implementation and full-model critical-path comparisons resume. No new speed result or model-numerical-equivalence claim is made by0290.
