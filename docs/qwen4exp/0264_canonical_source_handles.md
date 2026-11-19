# 0264 — Reuse immutable canonical expert handles

## Objective / implementation

0263 measured103,879canonicalbundle constructions/318,437GGUFrow requests in one responsecase. Sourcebytes are immutable, and keys identify the same expert for the index lifetime. Rebuildingviews/shapevalidation does not improve accuracy.

- `mini_llm/external/qwen4exp_gguf.py`: per-index `_sources` of complete `ExpertBundle` handles, lazily created, bounded by validated `n_layers*n_experts` namespace (24,576targetexperts). Default `cache_source_handles=True`; false retains uncacheddiagnostic behavior. `source(key)` validatesidentity/range/closedstore before returning a cachedhandle. `clear_source_cache()` retiresmetadata; outstanding returnedbundles/batches retain their owners.
- `mini_llm/external/qwen4exp_weights.py`: clears canonicalhandle metadata duringmodelweightclose, includingborrowedindexmode, before ownedstoreclosure. No trainingcode orQwen3-Nextmodule modified.
- `tests/test_0264_canonical_source_handles.py`.

No expertpayloadcopy, pinning, denseexpansion or bankconcatenation. Onlyreadonly NumPyviews/PackedGGUFWeight objects and completebundles are retained. Payloadpages remain reclaimablemmap. Modelcalls are serialized as before; no new concurrent-indexmutationcontract. This is sourceownership reuse, notGPUresidency or an unboundedworkqueue.

## Tests / numerical scope

Initialsource/model/generation/Qwen3-Nextgroup46passed14skipped. Identityreuse, exactlythreeviewconstructioncalls per firstexpert,1000repeatedsource/providerrequests withoutreconstruction, zero-copyreadonlyviews, explicitretirement/rebuild, closedstore rejection, exportedviewlifetime andoptout/range/identityvalidation tested.

FinalCPUgroup020x/021x/022x/023x/025x/026x/0106x withQ2SIMDenabled: **1875passed962skipped in32.58s**. SelectedactualCuPy026x/0259/0257/0251/0106j: **147passed3skipped in2.66s**. Syntheticshape tests do not establishfullmodel referenceequivalence. No productionequation or quantization changed.

## Real workload

`diagnostics_0264_sources_v1/`, samecapturedStrataprompts/settings and0262frozen/SIMD/overlap options. Fullysupervised,returncode0.128warmup and324responseIDs match0262exactly. Response323forwards/28.27350s = **11.42412forwards/s**, previous10.79765/s in a separateprocess; observed~5.8%, not a randomizedconfidenceclaim. Includingprefill:324/35.11621s =9.22651emittedtokens/s. No fullmodellogit/statebuffers captured by thissamplerbenchmark.

SampledRssAnon829,919,232bytes/RssFile55,192,346,624bytes (separatecategorymaxima); GPUallocatorused12,148,603,904/reserved12,335,560,704bytes, not globalVRAMpeak. Nominal6GiBexpertarena andmappedsourcewarm remainunchanged. Sourcecache ismetadata-only; RSSfilemax is not metadata-ownedRAM. Modelmemoryguardspassed.

Aggregate and retainedartifacts: `0260_0264_pipeline_comparison_v1.json`. No previousresults overwritten.

## Remaining / nextdependency

The50tokens/s goal remainsunmet. Response-time synchronousmovement iszero in this candidate, CPUQ2dot isSIMD andsharedGPUwork isoverlapped, but Python layer/routing/launch coordination and otherGPUpaths remain. The large nextcandidate is an executionplan with stable source/devicehandles, persistentbuffers and validated capturedGPUsegments/nativehandoffs. All-formatGPUQ8, profile/sizedcacheplacement, safeasyncmover, efficientprefill,INT8KV andStratasuffixspeculation remain separate outstandingitems. This patch is not completion ofthatarchitecture.
