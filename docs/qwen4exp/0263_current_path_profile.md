# 0263 — Profile current no-churn SIMD/overlap path

## Objective and method

Identify the next structural cost after0260 removes response-time promotions. Instrumented only the real sampled responsecase after the normalwarmup, using current0261SIMD, frozenplacement and0262sharedoverlap. No production equations orreferencebuild changed. Reused0254runner functions from a diagnosticPython wrapper; cProfile wraps `replay_case` and writes `decode.prof`. Child retains180sdeadline,8GiBAnoncap and4GiBMemAvailablefloor; terminal195stimeout also bounded it. This direct diagnostic invocation did not use the normal parent supervisor; do not describe it as a fullyparent-supervised run.

Artifacts: `diagnostics_0263_host_profile_v1/{decode.prof,result.json,...}` and `0263_host_profile_summary_v1.json`.400tokenforwards =77prefill+323decode. Profile includesprefill,49.904instrumentedseconds and77,283,639Pythoncalls. It is **not throughput**, does not include worker CPU samples inside nativekernels and is not an exclusiveGPUcriticalpathtimeline.

## Main observations

| Function / path | Internal seconds | Cumulative seconds | Interpretation |
|---|---:|---:|---|
|GGMLrowpool `_finish`|6.212|6.220|Mainthread native completionwait, not totalCPU arithmetic|
|`try_forward_decode_batch`|3.334|27.322|Parent includes childfunctions/waits; do notsum|
|MoE `__call__`|1.315|35.627|Includesrouted/shared/combination and dependencies|
|CuPyempty|1.243|1.243|506,376allocations/wrappers in thisprofile|
|Canonicalexpert `source`|0.600|5.904|103,879calls reconstruct immutablebundles|
|GGUFrows|0.917|3.235|318,437calls includingexpertviewconstruction|
|Packedweight `__post_init__`|0.648|1.205|325,237repeateddescriptorvalidations|

CPU completionwait remains visible, but there is substantial Pythonconstruction/validation/dispatch overhead. Faster CPUdots alone cannot eliminate it. Old0257threadedprofile had interleaved accounting; this new nativepool profile avoids that Pythonworker ambiguity. Do not compare totalprofiledwall with uninstrumentedwall as a clean regression.

## Evidence-supported next action

0264 caches canonical expert **handles**, not payloads, to remove demonstrated repeatedsource/view construction while preserving mmapownership and unloadcleanup. This is a prerequisite for reusable nativeexecutionplans, not a promise of50t/s. Beyondthat, the largecandidate is validated pre/postGPUsegments and native dispatch with stable activation/output buffers, routingIDs/counts/positions as runtime data and properCPUcompletiondependencies. Any wholedecodegraph must preservePLEI/O, QSApositions/selection and recurrence; an oldshortwarmNsightkernel sum is not proof such a graph will work or reach50/s.
