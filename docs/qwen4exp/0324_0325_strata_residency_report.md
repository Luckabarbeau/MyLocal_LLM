# 0324–0325 — Exact-launch evidence and bounded placement diagnostics

## Outcome

**50 complete decode forwards/s remains unmet. No production mathematics, packed arithmetic, residency policy or default changed.** The retained local0321 controlled replay remains41.99128/43.70105 forwards/s; it is not natural generation. This work corroborates a historical75.2 emitted tokens/s Strata session, separates it from a fresh memory-blocked startup, and implements diagnostics for the next placement experiment. Neither new model attempt supplies a completed response benchmark.

No checkpoint was downloaded, copied or modified. No complete expert arena was allocated locally. Existing reference mathematics/builds were not changed.

## Patch0324 — Establish what the exact Strata launch actually does

### Objective and interfaces

- `scripts/probe_0324_strata_exact_launch.py`: preserves the user's exact launch argv, checks isolation/port/process ownership, captures native logs and process telemetry, and retires owned processes. It uses retained prompts for a bounded warmup and three-turn conversation, **not the full standard-v1 suite**.
- Added `outcome_before_cleanup.json`: execution outcome and phase are persisted **before** snapshots or slow launcher cleanup. `finalized=false, cleanup_status=pending` explicitly does not assert resource retirement. Final `result.json`/`cleanup.json` remain authoritative when available. The parent retains its bounded SIGTERM/SIGKILL/systemd containment.
- `scripts/analyze_0324_strata_capture.py`: offline, standard-library-only analysis; <=2MiB input, explicit current timestamp, exclusive output, source SHA256 and inclusive line spans. Requires a unique matching engine-start marker. Rejects truncated or byte-incomplete head/tail captures; excludes unscoped prefix; never attributes historical generation rates to the current session.
- `tests/test_0324_strata_exact_launch.py` and `tests/test_0324_strata_capture_analysis.py`: exact command/environment, ownership/isolation, SSE/timeouts, interruption ordering and session attribution contracts.

### Provenance

- Installed engine0.1.59, CUDA13, RTX5080/compute12.0.
- Engine SHA256: `0ba914b28df1c8e09c0b5209c2ca6040c75fb650a7408ff3d9ef3ad159a13eb9` (retained pinned provenance; not a new mathematical audit of the binary).
- Strata source pin: `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`.
- Captured engine-log JSON SHA256: `a37f7806300e8e793f977b98bf9b430c5630cb136a881153d6fd81d659b0cd2a`.
- Session-scoped analysis: `0324_strata_sessions_v1.json`:23sessions,22historical and1current;114parsed generation summaries, none in the current session.
- Sources remain under `diagnostics_0324_exact_v1/`; guard under `diagnostics_0324_exact_guard_v1/`.

### Fresh session:2026-11-20 21:22:52, engine-log lines1866–1904

Exact command:

```sh
strata launch --model iq3_s --context 262144 --parallel 1 \
  --vision off --tune skip --access local --no-api-key --port 8095 --yes
```

No download was required: the launcher found the installed engine and checkpoint.

The38GiBhigh/40GiBhard cgroup **changed the engine's automatic residency decision**. Strata reported that the full arena required50.84GiB including4GiBoverhead/headroom and rejected it under the40GiBlimit. It selected mmap/file-backed experts instead. It filled2871GPUslots/5.50GiB from a24576-pair profile, then began allocating a32.82GiBpage-locked host complement for17246experts. **Completion of that allocation is not recorded.** Native flags included `--kv int8 --pool-workers 15 --spec 4`;15workers plus participating host form16threads.

The outer guard stopped startup at~17s: `MemoryHigh reached; stopping before sustained reclaim`. Recorded cgroup peak40,803,143,680bytes (38.000889GiB), swap0, OOM/kill0. Device-wide sampled GPU memory peaked14942MiB; this includes desktop use and is a startup observation, not steady-state expert distribution/utilization. Sampled engine RSS peaked34.969864GiB; RSS, charged cgroup memory and device-wide GPU usage are different scopes.

`result.json`/`cleanup.json` are absent in this retained run: the parent's2s termination bound outlasted neither slow launcher cleanup nor final artifact writing. Systemd containment removed the processes. The new pre-cleanup outcome file prevents losing the execution outcome in subsequent interrupted captures; this behavior is unit-tested, not revalidated by repeating the expensive launch.

### Historical session:2026-11-20 20:40:37, lines1779–1850

This is real retained evidence for the user's observed speed, **not a fresh benchmark performed by this patch**:

| Evidence | Historical value |
|---|---:|
| Expert arena |46.84GiB, `cudaHostRegister PORTABLE ok` |
| Arena load |4.56GiB/s |
| GPU expert cache |2792slots/5.35GiB, profile-filled |
| CPU team |15workers + participating host |
| Request prompt |92tokens, none reused;581ms prefill |
| Engine generation summary |12261tokens/163053ms, reported75.2tokens/s |
| Server summary |12260tokens/164s, disconnect with cancellation |
| GPU cache |3899900hits/4942014lookups, reported78.9% |
| Additional experts read by GPU |278807, reported5.3% of5220821routed |
| Adaptive tier |Blocking->async trial,13.74->13.27ms/token |
| Async activity |2999rounds,235515experts swapped,14.4ms/round start-to-flip |
| Suffix drafts |134/362accepted,311windows |

Important equivalence limits:

1. Default `STRATA_ROUTE_TAIL_SKIP=7` was **enabled**. The log explicitly permits omitting missed experts ranked7or lower across verify windows. This changes computation, not merely floating-point precision. Cumulative skipped-expert counters rise3333->776114; entries3718->777951. These are session counters; do not divide them by request routing totals as if they had the same scope.
2. The async adaptive tier was active, while the separately named adaptive-overlap feature was off. Its14.4ms/round includes overlapped time, not14.4msserialized delay per token.
3. INT8KV, profile placement, full pinned arena, up-to12-token verification windows, suffix/lookup drafting and natural response length differ from local full-top10 forced replay. Accepted drafting is recorded; a separate DFlash2 model being necessary/active is **not established**. Do not download it to explain this result.
4. The engine/server token counts differ by one and the server reports cancellation. Preserve both accounting scopes; do not call it a normally finished response.
5. Native log text calls the521472000-byte head “Q5_K”; the audited checkpoint `output.weight` is Q6_K with exactly521472000bytes. That label alone does **not** establish that weights were requantized or that head arithmetic matches. Format parity requires descriptor/kernel evidence, not interpretation of that string.

Unchanged CLI is consequently not equivalent execution under a different memory budget. Nor can the historical75.2/s be used to claim that exact full-top10 Strata has been newly measured at that rate.

### Reproduce the offline analysis only

```sh
.venv/bin/python scripts/analyze_0324_strata_capture.py \
  --current-start '2026-11-20 21:22:52' \
  --engine-capture docs/qwen4exp/diagnostics_0324_exact_v1/native_log_2960189662183_0.json \
  --output NEW_SESSION_REPORT.json
```

No model or subprocess is launched. Output is exclusive; original captures remain unchanged.

## Patch0325 — Measure placement opportunity before building a mover

### Objective and interfaces

- `scripts/probe_0302_decode_configuration.py --route-profile`: explicitly enables the existing bounded native route counters and records `route_profile_diagnostic=true` in `reproduce.json` on ordinary completion. Default remains off. Staging=0, promotion-limit=0, sized slots, host-source budget and execution arithmetic are unchanged. Environment exports cannot silently override the wrapper's explicit setting.
- `scripts/analyze_0325_request_placement.py`: offline, <=64MiBsource, exclusive output and source SHA256. Requires complete48x512top10profiles, nonnegative integer monotonic counters, unchanged residents and slot ledgers. Enforces `sum(heat_delta)=10*calls_delta` per layer and unique-route bounds. Uses only requestA's completed observations to propose at most8complete swaps/32MiBincoming bytes, matching256-byte aligned physical size classes; evaluates requestB's unchanged trajectory. Rejects missing profiles/resets. Zero cap preserves baseline counts.
- Reports saved misses, displaced hits and net change by class. Formats are attributed only if explicit gate/up/down metadata exists; none are guessed from size. Each boundary starts from recorded placement, not a rolling hypothetical model. No actual movement, quality change, latency or throughput estimate.
- `tests/test_0325_request_placement.py`: causality, deterministic ties, size/byte/bundle bounds, incomplete/reset counters, source preservation and exclusive output.

This targets a real gap: native frozen plans already gather aggregate statistics but do not update Python admission heat. Request graphs hold resident leases and pointer snapshots; changing placement underneath them would be unsafe. Existing profile installation rejects sized slots/device plans. A measured causal opportunity is required before adding a retirement/rebuild/movement lifecycle.

### Bounded natural-generation attempt

```sh
MINI_LLM_PLE_BATCHED_ROWS=1 .venv/bin/python scripts/run_0290_isolated_benchmark.py \
  --memory-high-mib 28672 --memory-max-mib 32768 --reserve-gib 12 --timeout 210 \
  --output docs/qwen4exp/diagnostics_0325_route_profile_guard_v1 -- \
  .venv/bin/python scripts/probe_0302_decode_configuration.py \
  --output docs/qwen4exp/diagnostics_0325_route_profile_v1 \
  --shared-grid --recurrent-chain-graph --qsa-block-graph --kernel-publish \
  --qsa-warp-dots --expert-cache-gib 7 --response-repeats 2 \
  --natural --route-profile --telemetry
```

**Incomplete: exit125, guard reached MemoryHigh after57.073s. No completed response, no placement-benefit estimate, no new clean throughput.** Warmup128outputs completed; last retained response progress is32outputs. The parent killed the process group within its existing bound. `result.json`, `reproduce.json` and wrapper-buffered GPU telemetry were not finalized; do not fabricate them. Original partial artifacts remain preserved.

- Charged cgroup peak30,066,290,688bytes (~28.0014GiB); final sampled high events96, OOM/max/kill0; service swap0.
- Last guard host availability54,701,928,448bytes (~50.95GiB). Host availability counts reclaimable file cache; it is not cgroup headroom.
- Last retained response snapshot: anonymous RSS876,285,952bytes (~0.816GiB), file-backed RSS31,437,664,256bytes (~29.28GiB), VmHWM33,081,507,840bytes. RSS and charged memory differ because mapped/shared page accounting differs; these cannot be added to infer total physical use.
- Partial response native counters **validated successfully**: all48layers,512experts,108counted calls/layer,51840total routes; correct top10totals and physical sized-slot occupancy. This includes prompt/native work and is not32decode-only calls.
- Profile source SHA256: `68f3c13efb499351566f36976d5e7f75e8387401b88083a944c129f036e6b5b2` (`progress.jsonl`). It is insufficient for a completed requestA->B analysis; the analyzer intentionally requires two.
- After retirement: no `mylocal-bench-*` units remained; device memory959MiB, host available53967MiB. No retained model process or server was intentionally left running.

The cap was **not increased**, no competing model was run, and neither attempt is an OOM or demonstrated model-equation defect. This cold file-page charging case was anticipated in `RESTART_HANDOFF_0323.md`.

## Tests, numerical scope and performance

```sh
.venv/bin/python -m pytest -q \
  tests/test_0324_strata_exact_launch.py \
  tests/test_0324_strata_capture_analysis.py \
  tests/test_0325_request_placement.py
```

**74passed,61subtests passed.** No CUDA/model is imported by these tests. Actual-checkpoint partial profile invariants also passed the offline `decode_snapshot()` check. No new logit/state/reference equivalence claim and no performance improvement is established. Previous732CPU/586CuPy selected regression results remain historical verified production evidence, not newly executed tests in this patch. Production inference sources were not changed.

Status:0324diagnostics implemented and fixture-validated; exact launch memory-blocked, fresh throughput unvalidated.0325diagnostics implemented and fixture-validated, actual partial-counter contract validated; request-to-request placement benefit unvalidated.

## Next dependency

1. Before another model run, inspect/record cgroup `memory.stat` (file/anon/unevictable, inactive file and workingset faults) alongside canonical expert/PLE access. Attribute the cold page charge and determine a bounded working-set plan **within existing limits**; do not bypass the guard or warm the checkpoint outside containment.
2. Obtain two complete profiles with no actual movement, then evaluate the8bundle/32MiBcausal policy. Earlier equal-slot forced-trajectory analysis found<0.7%miss savings; do not assume a mover pays for itself on current natural routes.
3. Implement request-boundary refresh only if exposed CPU/storage savings plausibly amortize transfer and safe graph/lease rebuild. Between-request heat is not a substitute for a measured decode critical path.
4. A fresh faithful full46.84GiBpinned-arena Strata benchmark cannot fit the presently approved limit/reserve plan. Historical evidence is usable without repeating it. Any different resource plan requires explicit review, not an automatic limit increase.

Do not silently copy tail omission, require a drafter download, repeat unsuccessful spin/affinity/4GiBsource tests, or label instrumented/forced/partial execution as sustained natural generation.
