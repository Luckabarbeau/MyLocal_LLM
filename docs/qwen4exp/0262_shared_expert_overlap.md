# 0262 — Shared GPU expert during routed CPU execution

## Objective / graph invariant

MyLocal previously waited for routedexperts before launching independent sharedexpert projections. Strata's session/layer design overlaps sharedGPU and residenthit work with CPUmiss execution (pinned0258audit). Dependency is `hidden -> {routed, shared} -> combination`; neither branch depends on the other's output. No followinglayer may execute before the combination/residual isready.

## Files and interface

`mini_llm/external/qwen4exp_model.py`: `Qwen4ExpMoE(..., overlap_shared=False)` and modelbuildkeyword with same name. `scripts/probe_0254_strata_workload.py`: `--overlap-shared`, propagated/recorded. New `tests/test_0262_shared_expert_overlap.py`.

Opt-inCuPy mode creates one nonblockingstream perMoElayer. Caller event publisheshidden; sharedstream waits, enqueues the same gate/up/SwiGLU/down and sharedscore expressions; caller continues routing/routedexecution, then waits for sharedcompletion before unchanged combination. No host wait on the shared event during successful scheduling. Exceptions after enqueuing shared work also fence caller reuse. Capture at this dispatchseam is explicitly rejected rather than launching an unexpected uncapturedstream inside a graph. Oldserial path remains default, including NumPy. Not wholemodelgraphsupport.

## Validation

ActualCuPy59passed1skipped in0262/0251/0260group. Tests cover native/fusedsharedexpressions, default/nondefaultcallerstreams, changinghiddenproducer values, retainedoutputs, routefailure fences and evidence sharedwork is enqueued before the routedCPUwait. Syntheticoutputs arebit-exact. BroaderselectedCuPy147passed3skipped; CPU1875passed962skipped. ExistingQwen3-Next tests pass inCPUgroup.

Longsampled `diagnostics_0262_overlap_v1/`: both128warmup and324responseIDs exactly match0261. Response323forwards/29.91392s = **10.79765/s**, prior10.72100/s. This does **not** demonstrate a material gain in a single pair; leaveopt-in. No claim that CPU/GPU resourcecontention is solved. The event schedule is tested; the full reference graph is not newly established by matchingtokens.

RssAnon823,582,720bytes/RssFile55,142,375,424bytes sampledcategorymaxima. GPUallocatorused12,148,603,904/reserved12,335,560,704bytes; not globalVRAMhighwater. Existingboundedmodelcache/state allocations retained; stream/event resources are bounded by48layers and the serializedcallcontract. Per-call scratch/events are not yet preallocated graph/doorbell infrastructure.

## Remaining / nextdependency

No throughputdefault change. Sharedbranch is not the principal remaining delay at this workload.0263 profiles the actual frozen/SIMD/overlap path rather than interpreting an oldkernel timeline as currentcriticalpath. Persistent execution descriptors, device-resident runtime metadata, native routing/dispatch and validated capturedsegments remain larger dependencies.50t/s not reached.
