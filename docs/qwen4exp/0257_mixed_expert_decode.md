# 0257 — Grouped GPU terms during CPU/GPU hybrid decode

## Implementation

Previously a CPU miss disabled grouped execution of otherwise resident GPU experts. New opt-in `group_mixed_decode=True` groups supported IQ GPU experts into weighted FP32 terms while CPU jobs run, then materializes GPU/CPU terms into ascending expert order and performs serial FP32 addition from zero. GPU terms are not weighted twice; CPU terms receive their route weight once. Unsupported cases retain existing dispatch. Default unchanged.

Files: `ops/iq_packed_gpu.py` adds independently validated `grouped_terms` and internal return-terms seam; `ops/mixed_expert_combine_gpu.py` provides two-launch materialization/ordered reduction; `ops/expert_residency.py` integrates after CPU job submission with existing exception draining and residency scope; `external/qwen4exp_model.py` exposes the option; tests in `test_0257_mixed_expert_terms.py`. Host metadata bounded to64 selected experts. Scratch remains O(selected experts * hidden/intermediate width), no dense weight expansion. Expert leases and dispatch completion protection remain intact.

## Validation

Building-block GPU suite294 passed1 deselected (opt-in real benchmark). Integrated GPU mixed/CPU miss/admission/slot group **71 passed**; CPU benchmark/model30 passed1 skipped. Exact addition-order and no-double-weight tests pass; generic versus grouped within-row reduction ordering follows the existing bounded FP32 operator tolerance. No whole-model equivalence claim solely from generated text.

## Longer sampled measurements

`diagnostics_0257_long_v1`: inference child completed128 warmup tokens and352 short-r1 tokens endingEOS;351 decode forwards in44.80595s =**7.83378/s**, versus previous separate-run8.03745/s. Different sampled lengths/content and timing variability prevent a controlled speed ratio; **no demonstrated gain**, keep opt-in. Parent telemetry failed during child exit (returncode0/result file retained); not a successful supervisor run.

`diagnostics_0257_profile_v1`: bounded child CPU profile on the128-token warmup.127 decode steps logged approximately6.7s in future results/waits and3.4s in synchronous expert materialization. These are nested/interleaved host profiles, not exclusive GPU critical-path measurements; worker activity/reentrant statistics must not be summed as serial components. Profile RSS limit60GiB admits mapped file pages, while the diagnostic's own8GiB anonymous and8GiB MemAvailable guards remain active. Sampled peakRSS54,820,855,808bytes was mostly file-backed.

`diagnostics_0257_fullwarm_v1`: touched all50,292,326,400 expert-source bytes; estimated50,093,154,304bytes resident, no full private copy,4GiB system-available reserve. Child completed128 warmup,127 forwards/16.45828s =**7.71648/s**—essentially the preceding mixed warmup7.71750/s. Therefore incomplete warming alone does not explain the remaining steady-state latency in these runs. Parent `/proc/io` exit race reoccurred after one corrective attempt; further edits stopped. Anonymous/file peaks828,813,312 /54,414,520,320bytes. System-available RAM after warming58,032,041,984bytes includes reclaimable file cache and is not physical free/pinned memory.

## Next dependency

Resolve supervisor exit-race test, then instrument CPU job submit/start/finish and input-ready fences plus promotion copy windows on the actual long workload. Persistent native CPU task scheduling, batched input quantization and bounded asynchronous promotion are candidates, not proven fixes yet. GPU Q8_1 GEMV is operator-validated only for IQ3_S (0253), not yet routed-expert integration or all formats. The50t/s sustained goal is unmet; preserve existing FP32 oracle and separately report Strata's suffix drafting/tail omissions.
