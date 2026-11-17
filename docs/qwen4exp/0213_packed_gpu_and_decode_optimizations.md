# Patch 0213 — packed GPU experts and decode optimizations

## Scope and objectives

This note documents the existing Patch0213 implementation and diagnostics0228–0233 from source and retained artifacts. It adds no implementation, changes no defaults, and does not rerun tests or model inference. The objectives are to execute the checkpoint's packed expert formats directly on GPU, improve backbone placement and singleton-decode staging, reduce small-operation launch overhead, and retain full top-10 routing, FP32 activation/state semantics, bounded logical budgets, and the existing reference paths.

The measured endpoint is **8.566126 tokens/s on an RTX 5080**, for the short forced-token experiment described below. It is not 70–80 tokens/s, natural-language generation throughput, independent full-model parity, or an end-to-end long-context result.

The checkpoint remains the original local two-shard `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, with 48 layers, width 2560, 512 experts/layer and top-10 routing. No full-model floating-weight copy, dense expert bank, full token-embedding copy, or full PLE-table copy is introduced. Expert weights remain canonical packed data; CPU misses use the native packed backend. See [reference_contract.md](reference_contract.md) for the graph and unresolved checkpoint-provenance limitations.

**Numbering:** Patch0213 is the logical implementation/documentation patch. Diagnostic/test IDs **0228–0233** identify separately reviewable validation and optimization subpaths, not six alternative names for the patch: 0228 packed GPU arithmetic/integration; 0229 staging, cache accounting and lifetime; 0230 QSA; 0231 grouping/profiling; 0232 HC; 0233 sigmoid-compatible recurrent decode and optimized lifecycle. Each backend/arithmetic subpath needs its own applicable tests or oracle; a passing CPU suite does not independently validate CuPy kernels, and optimized-path self-consistency does not establish independent full-model reference equality.

## Implementation map and interfaces

Paths in this section are relative to the repository root.

| Files | Responsibility / interface |
|---|---|
| `mini_llm/ops/iq_packed_gpu.py` (new) | CuPy RawKernel `supports_type`, `linear(x, weight, block_k=512)`, `swiglu(x, gate, up, block_k=512)`, and `grouped(x, experts, routes, slots=None, block_k=512)`; direct packed decode and FP32 reductions. |
| `mini_llm/ops/quantized_linear.py` | Public packed-GPU capability and linear/SwiGLU dispatch integrate the new formats without conflating CPU format registration with GPU support. |
| `mini_llm/ops/grouped_quantized_experts.py` | Metadata-only eligibility and grouped dispatch to the IQ implementation; legacy paths remain separate. |
| `mini_llm/ops/expert_residency.py`, `mini_llm/ops/experts_inference.py` | Selected-expert residency, singleton CPU-miss batching, dispatch/fence lifetime cleanup, and sorted exactly-once weighted combination. |
| `mini_llm/external/qwen4exp_weights.py` | Header-first placement and memory planning; newly supported packed backbone projections can remain on GPU within the static budget. |
| `mini_llm/external/qwen4exp_model.py` | Factory controls, promotion threshold, explicit fast-path switches, and model-owned residency credit on subsequent memory checks. |
| `mini_llm/ops/qsa_decode_gpu.py` (new), `mini_llm/external/qwen4exp_attention.py` | Opt-in FP32 all-causal-cell QSA decode with online stable softmax and sigmoid output gate; reference sparse path retained. |
| `mini_llm/ops/hyperconnection_decode_gpu.py` (new), `mini_llm/external/qwen4exp_hyperconnection.py` | Opt-in single-token HC normalization/gating/mixing/write fusion; dense matmuls remain CuPy operations. |
| `mini_llm/ops/gated_delta_decode.py` | Extends the existing decode path to sigmoid Z gating as well as SiLU, preserving the configured key/value head map and recurrent-state computation. |

### Model factory controls

`Qwen4ExpGGUFModel.open(path, *, max_length, ..., **budgets)` forwards the following controls to `_build`; `from_index` uses the same build seam:

| Keyword | Default | Meaning |
|---|---:|---|
| `expert_promote_after` | `3` | Positive integer admission threshold for the owned heterogeneous executor. `1` requests first-use promotion, not preloading every expert. |
| `qsa_fast_decode` | `False` | Enable the eligible all-causal-cell QSA path. |
| `hc_fast_decode` | `False` | Enable eligible single-token HC fusion. |
| `router_fast_decode` | `False` | Enable the existing fused decode router; retain full softmax/top-10 normalization, not route omission. |
| `gdn_fast_decode` | `False` | Enable the recurrent mixer's decode fast path, including Qwen4Exp sigmoid Z gating. |

The four fast-path controls must be Python booleans. A borrowed executor retains its own configuration and must satisfy the factory's geometry, FP32/native-backend, model-identity and zero-host-dequant-cache checks; the factory's promotion setting is used when constructing an owned executor.

`HeterogeneousExpertExecutor(..., batch_cpu_decode=True)` separately defaults to batched CPU misses; use constructor `batch_cpu_decode=False` to opt out. This is an executor control, **not** an additional model `open` budget keyword. The ablation sets `model.expert_executor.batch_cpu_decode` explicitly so baseline/backbone can reproduce the older staging behavior.

Eligible singleton FP32 dispatch shares one immutable activation among CPU jobs: one current-stream D2H activation copy, concurrent native packed CPU work queued before unchanged resident GPU forwards, one result-slab H2D copy, then ascending-expert-ID FP32 multiply followed by sequential add. Synchronization protects producer readiness and pinned-slab lifetime; every started future is drained even on failure. Scratch is dispatch-local, proportional to selected experts times width, not the expert bank. Ineligible shapes/dtypes, absent workers and routing-profiler use retain the existing path.

### Packed arithmetic and grouped execution

The six newly supported encodings are `IQ2_S`, `IQ3_S`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS`, and `Q2_0` (the last is not named IQ). The mathematical interfaces are:

```
linear(x, W) = x @ W.T
swiglu(x, G, U) = SiLU(x @ G.T) * (x @ U.T)
expert(x) = linear(swiglu(x, G, U), D)
```

Packed bytes are read lane-wise; weights are not materialized as floating matrices. Products, reductions and SwiGLU use FP32. The standalone linear/SwiGLU interfaces accept contiguous 2D/3D CuPy FP32 or FP16 activations and cast only the final output to input dtype. **FP32 is the tested model path; FP16 is supported, not a preferred performance/precision policy. No Q8 activation approximation is used.** These are decode-oriented kernels, not a large-prefill GEMM implementation. Launches use CuPy's current stream; compiled modules are cached by device/dtype, and canonical lookup tables are read-only CUDA constants with no per-call table transfer.

Grouped IQ execution requires one FP32 row, compatible projection geometry/format across handles, and all three projection encodings supported by this module. Legacy/IQ hybrid triples are deliberately not accepted by this grouped path. Caller-ordered handles and route slots preserve ascending expert-ID accumulation; routing weights apply exactly once after down projection. Three compute launches perform fused gate/up, weighted down, and ordered sequential reduction. The only uploaded metadata is an `E x 3` or `E x 4` uint64 pointer/slot table. Scratch is `E*F + E*D` FP32 values plus the `[1,D]` result; the caller retains weight ownership through stream completion. This does not require Triton for IQ grouping.

### QSA, HC, router and recurrent constraints

- QSA fusion is valid for sequential text when all causal cells fit the selection width: `n <= index_topk + block_size - 1`, here **2051 cells**. It avoids index-score selection and selected-KV copies, not required history updates: raw/pooled index state still advances. Diagnostics use the reference path. Beyond 2051, sparse selection and attention remain on the reference path; this is not evidence of fused long-context speed.
- HC fusion is eligible for contiguous FP32 single-token four-stream inputs on the current device. It preserves per-stream normalization, the three distinct `/4` factors, stable sigmoid gates, paired read/write injection gains and the final read-only collapse. Dense projections are not replaced with quantized activations.
- Router fusion remains full softmax and normalized top-10 routing. No routing rank is dropped.
- GDN extends the existing kernel's output normalization/gate to sigmoid Z, while retaining SiLU support for existing callers. Qwen4Exp's mapped key head is `h % 16` for 48 value heads of width 128; it must not be replaced with a contiguous `h // 3` grouping. Gate choice does not change the recurrent-state equation.

## Memory budgets: defaults, tested configuration and correction

All GiB values here mean `1024**3` bytes. The default planner uses `static_device_budget_bytes=4*GiB`, `device_budget_bytes=8*GiB`, `cache_bytes=0`, `host_budget_bytes=8*GiB`, 512 MiB headroom, and 64 MiB each for staging and activation reservations (raised to computed minima when needed). The **4 GiB static default is conservative**, not a promise of the measured performance: supported packed backbone projections are admitted only while they fit after dense FP32 controls. Performance-oriented placement is explicit user configuration, not an automatic increase in budget.

The RTX 5080 experiment used four CPU workers and, except baseline, **6 GiB static / 14 GiB total device budget**. Backbone and batch had zero expert cache; cache and later variants used **6 GiB expert-cache capacity**. Eager and later variants used `expert_promote_after=1`; cache retained `3`. All final fast switches were enabled only for their respective cumulative variants. These values are a tested configuration, not safe universal defaults for other cards, contexts, concurrent allocations or sessions.

The retained short-run plan reports:

| Quantity | Bytes |
|---|---:|
| Dense FP32 weights | 3,066,007,040 |
| Placed device static weights | 5,704,430,080 |
| Constructor temporary reservation | 340,992,000 |
| Recurrent state | 117,669,888 |
| QSA state at length 16 | 915,456 |
| PLE state | 368,640 |
| Total state | 118,953,984 |
| Logical host reservation | 944,971,776 |
| Logical device reservation including cache capacity | 13,277,915,648 |
| Canonical mmap tensor bytes | 83,606,638,080 |

These are logical planning quantities, not interchangeable with physical free VRAM, CuPy pool occupancy or process RSS. The planner includes static weights, full cache capacity, state, activation/staging, constructor temporaries and headroom in its total device reservation. Mmap size is an addressable packed source size, not a dense RAM allocation; accessed file pages can nevertheless contribute to RSS outside the logical host reservation.

### Preserved v1 failure and v2 fix

`0229_decode_performance_v1.json` is intentionally **blocked**, with its original `diagnostics_0229_v1/cache.log`. Cache state creation failed with:

```
MemoryError: Insufficient free device memory: need 13277915648, free 8585216000
```

The postconstruction check compared the entire reservation to remaining physical free memory after allocating the backbone, counting already-owned storage a second time. The correction credits only this model's live static allocations, actual admitted cache bytes and live-session reserved state when checking the **remaining** physical requirement. It does not credit unfilled cache capacity, another model's allocations, or global CuPy pool usage. Logical host/device budget comparisons still use the complete reservation and remain strict; initial construction has no residency credit. Unused pool-cached blocks may still make physical checks conservative. Replanning cannot expand the approved weight placement.

v2 reused the three successful v1 variants and measured the corrected cache variant. This was an accounting correction, not permission to exceed a configured logical budget. Completed dispatch fences are also reaped below budget so retained handles do not accumulate indefinitely; pending/unknown fences and protected experts remain non-evictable.

For the final `fused_gdn` run, v5 records process peak RSS **10,318,860,288 bytes**, CuPy pool used **9,819,179,008 bytes**, and pool reserved **9,829,571,072 bytes**. The supervisor sampled peak RSS **10,318,323,712 bytes**; sampling and process high-water marks differ. The expert cache held **1,952 experts / 3,994,496,000 packed bytes**, below its 6 GiB capacity, with 7,680 requests, 1,952 misses/promotions and 5,728 hits (0.745833 hit rate). All 7,680 selected experts executed on GPU in this short eager run. This does not mean all experts fit, or that a later cache-full workload avoids CPU fallback. Host-dequant cache bytes remained zero.

## Retained measurements and exact method

Primary sources are [0228_real_gpu_experts_v1.json](0228_real_gpu_experts_v1.json), [0231_grouped_iq_gpu_v1.json](0231_grouped_iq_gpu_v1.json), [0229_decode_performance_v1.json](0229_decode_performance_v1.json) through [v5](0229_decode_performance_v5.json), and [0231_fast_profile_v1.json](0231_fast_profile_v1.json). Reproduction logic is in `scripts/probe_0228_real_gpu_experts.py`, `scripts/probe_0229_decode_performance.py` and `scripts/probe_0231_fast_profile.py`.

### Cumulative decode ablation

The decode probe feeds **forced token IDs 1 through 16**, not its sampled/argmax outputs, to fresh batch-one length-16 state. Activations and state are FP32; every layer retains full top-10 routed work; there is **no tail skipping or speculation**. Each step is timed with `perf_counter` through current-stream synchronization. Positions **8–15, zero-based**, form the eight measured steps; throughput is `8 / sum(step_seconds[8:16])`, not reciprocal median or total-process throughput. Build time is reported separately. Admission/copy costs occurring inside these measured steps remain included; early JIT/first-use costs can fall in the first eight steps.

Every successful variant was executed **once**, in its own process, sequentially rather than concurrently. The supervisor set `MINI_LLM_BACKEND=cupy`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`, sampled RSS every 50 ms, and imposed a 20 GiB sampled-RSS guard and 120-second child deadline. Successful results were reused, **not rerun**, through exclusive-created versioned artifacts:

| Version | Newly retained successful measurements | Reuse / failure |
|---|---|---|
| v1 | baseline, backbone, batch | cache failed during state creation; failure retained |
| v2 | cache | reuses v1 successes after memory-accounting correction |
| v3 | eager, fused_qsa | reuses v2 |
| v4 | grouped_iq, fused_hc, fused_router | reuses v3 |
| v5 | fused_gdn | reuses v4; complete ten-variant cumulative report |

| Cumulative variant | Added configuration / implementation | Measured tokens/s |
|---|---|---:|
| baseline | Reproduced pre-0213 packed capability/placement; default budgets, no CPU batching | 0.272867 |
| backbone | New packed capability plus explicit 6 GiB static placement; no expert cache | 0.626706 |
| batch | Batched CPU-miss staging | 0.628522 |
| cache | 6 GiB expert cache, promotion threshold 3 | 1.274565 |
| eager | Promotion threshold 1 | 4.757141 |
| fused_qsa | Opt-in all-causal-cell QSA | 6.054848 |
| grouped_iq | Three-launch grouped packed experts | 6.098446 |
| fused_hc | Opt-in single-token HC fusion | 7.386917 |
| fused_router | Opt-in fused decode router | 7.740246 |
| fused_gdn | Opt-in sigmoid-compatible recurrent decode | 8.566126 |

This is a historical cumulative ablation, not repeated independent A/B sampling. Baseline is reproduced by restricting packed capability to the legacy format map; backbone also changes explicit placement budget, so that row is not a kernel-only speedup. Grouped execution was added before the later retained variants; the current probe script is not a mechanism for reconstructing every historical source state. Small differences, notably batch versus backbone and grouped versus fused_qsa, have no confidence intervals here and should not be generalized from one run.

### Real packed experts (0228)

On NVIDIA GeForce RTX 5080, CuPy 14.2.0, CUDA runtime integer 13020, the artifact compares one selected packed expert in **each of 48 layers**, covering seven gate/up/down format triples, against native packed CPU execution. Inputs are fixed synthetic FP32 `[1,2560]`, seed 228, standard deviation 0.1; this is real checkpoint weight execution, not natural model activations. All records passed. Maximum single-expert absolute error is **5.587935447692871e-9**; maximum relative-L2 is **1.6228204350673205e-6** (approximately **1.62e-6**; a literal `<=1.62e-6` bound would round down the recorded maximum).

A separate representative layer-47 test uses its real uncorrected BF16 router on the fixed synthetic input and computes all ten selected experts. Ordered FP32 combination has absolute error **1.4151737559586763e-9** and relative-L2 **1.3365165557256045e-6**. Warm median CPU compute-only time is 25.277588 ms; GPU event time 0.734048 ms and synchronized public-dispatch wall time 0.741476 ms. These operator timings exclude expert upload and JIT and are **not full-model tokens/s**.

The probe records 127,472,640 logical mmap-read bytes, maximum single read 2,621,440 bytes, and additional peak RSS 891,842,560 bytes against a 2 GiB additional-RSS limit. Checkpoint size/mtime were unchanged; this is not a payload-hash integrity proof.

### Grouped experts and profile (0231)

The grouped artifact tests ten evenly spaced ascending expert IDs with synthetic normalized routes, FP32 seed 231/std 0.1, at representative layers **0, 1, 2, 12, 17, 21, 47**, one per format triple. All seven grouped outputs are exactly equal to sequential GPU outputs in the recorded comparisons (zero absolute and relative-L2 error). This compares two GPU paths, not independently validated full-model logits.

Warm medians over 20 repeats exclude upload/JIT but include grouped pointer-table upload: grouped event times range **0.376688–0.542880 ms**, event speedups **1.2134–1.3150x** over sequential execution. Three grouped compute launches replace twenty sequential projection launches. Canonical constant tables remain unchanged: 11,408 codebook bytes; the recorded gate kernel uses 48 registers, 2,048 shared bytes and zero local bytes. No unmeasured lookup-table storage optimization is claimed. Additional peak RSS is 872,243,200 bytes; checkpoint size/mtime were unchanged.

The separate `0231_fast_profile_v1.json` is instrumented and is **not** the final GDN throughput run. Its script enables eager promotion and fast QSA, with grouping available, but does not enable the later HC/router/GDN switches. Over eight profiled tokens it records device spans of 0.505973 s routed experts, 0.259608 s GDN, 0.129470 s FFN HC read, 0.100924 s attention HC read, 0.113195 s QSA and 0.081678 s router. Nested totals (for example `moe_total`) must not be added to their children, and host/device spans are not interchangeable with isolated kernel times or uninstrumented throughput.

The latest retained [0231_fast_profile_v2.json](0231_fast_profile_v2.json) profiles the current **all-fusions** configuration, separately from v1. Dividing aggregate host durations by the eight profiled tokens gives:

| Stage | Instrumented host seconds/token |
|---|---:|
| Routed experts | 0.0594004 |
| GDN | 0.0120806 |
| QSA | 0.0215229 |
| Attention HC read + FFN HC read | 0.0076960 |
| Router | 0.0017160 |

The HC row sums the two read stages only, not writes or final collapse. These are nested, **non-exclusive instrumented spans**, not an additive kernel-time decomposition: host durations include implicit waits, CUDA event spans can include CPU idle gaps, and tokens synchronize at boundaries rather than after each operation. Profile jitter and instrumentation mean v2 does **not** replace the retained uninstrumented **8.566126 tok/s** measurement or establish a new throughput rate.

### Same-prefix full-model logits: v5, not an independent oracle

The ablation saves full logit arrays and state digests at prefixes **1, 2, 4, 8, 16**. `same_prefix_vs_baseline` compares logits after conversion to FP64, using max absolute difference and `norm(actual-baseline)/max(norm(baseline),1e-30)`. All nine non-baseline variants preserve argmax and top-10 **logit-set** overlap 10/10 at all five snapshots. This output-vocabulary overlap is distinct from executing top-10 routed experts.

For the final `fused_gdn` row, v5 gives:

| Prefix | Max absolute logit error | Relative-L2 | Actual = baseline argmax |
|---:|---:|---:|---:|
| 1 | 2.193450927734375e-5 | 1.4416099795342332e-6 | 271 |
| 2 | 2.4318695068359375e-5 | 1.2523463190951007e-6 | 220 |
| 4 | 1.9550323486328125e-5 | 1.5213010919138865e-6 | 5 |
| 8 | 2.7894973754882812e-5 | 2.065314616616117e-6 | 198 |
| 16 | 2.288818359375e-5 | 1.7883890271292477e-6 | 17 |

Across every non-baseline row and saved prefix in v5, the maximum absolute difference is **2.7894973754882812e-5**, and the maximum relative-L2 is **2.065314616616117e-6**. These compare optimized paths with this project's baseline on identical forced prefixes. They do not resolve the separate independent-reference full-model numerical disagreement, prove bitwise state identity, or establish agreement on an unconstrained generated continuation. The artifact status explicitly says `completed_measurement_not_full_parity_claim`.

### Optimized lifecycle self-consistency (0233)

[0233_fast_lifecycle_v1.json](0233_fast_lifecycle_v1.json), with retained detail in `diagnostics_0233_lifecycle_v1/result.json`, reports `status=passed` for the actual optimized CuPy FP32, full-top-10 path. For input IDs **1–16**, prefill and incremental logits are **bit-exact**, and all active-state hashes match (`21d17fc3aede01a84b7af68ee3f28be1e156b30d2b160e0a403b71f080684794`). A fresh repeated greedy request reproduces IDs **`[220, 100561, 198]`**. This is a three-token generation smoke and lifecycle self-consistency check, not independent full-model reference equality or a long-continuation claim.

The lifecycle supervisor records return code 0, no stop reason, sampled peak RSS **11,173,183,488 bytes**, and CuPy pool reserved **10,616,590,848 bytes** (used **10,486,193,152 bytes**). These belong to the lifecycle workload, not the earlier throughput run; they must not silently replace its memory measurements. Host-dequant cache remains zero; 2,332 packed experts occupy 4,778,188,800 bytes of the 6 GiB cache.

## Reviewable test coverage, not a new test-run claim

The following existing source files describe coverage; their presence is not evidence that this documentation task executed them:

- `tests/test_0228_iq_packed_gpu.py`, `tests/test_0228_iq_gpu_integration.py`: all packed lanes, linear/SwiGLU/expert integration, dtypes, tiles, current/nondefault streams, no host dense decode/transfer, capability validation, misses/promotions and full selected dispatch.
- `tests/test_0229_batched_cpu_misses.py`: batching on/off, mixed resident/CPU work, exact ordered combination, overlap, exception/future drainage and observational ineligible fallback.
- `tests/test_0229_cache_fence_cleanup.py`: completed-fence cleanup below budget and preservation of pending/unknown-fence protections.
- `tests/test_0229_resident_budget.py`: strict initial physical check, owned-residency remainder, live sessions, strict logical budgets, invalid credit and fixed placement.
- `tests/test_0230_qsa_gpu_decode.py`: sequential parity/state and width fallback; stable FP64 oracle including 2051 cells, strides/current key/nondefault stream and failure-before-commit behavior.
- `tests/test_0231_grouped_iq_gpu.py`: format triples, CPU and sequential oracles, slots/strides/duplicates, ordered rounding, exactly three launches, bounded scratch, capability fallback and no Triton dependency.
- `tests/test_0232_hc_gpu_decode.py`: FP64 reference, paired tokens/final collapse, saturation, stream handling, validation and unchanged fallback.
- `tests/test_0233_sigmoid_delta_decode.py`: real-geometry 16-token continuous recurrence, sigmoid and SiLU, state/head-map checks, saturation, stream/lifetime behavior, environment escape hatch and optional independent 0226 captured-operand replay. Optional replay support does not establish that replay ran.

Existing related tests `tests/test_0204_sparse_iq_execution.py` and `tests/test_0106u_gated_delta_decode.py` also have integration changes. Repository-wide status/history/root-document updates are outside this single-file documentation change.

### Reported final test results

The following final results were supplied for this documentation update; they were **not rerun here** and are distinguished from the independently inspected diagnostic JSON. No per-agent/per-patch totals are inferred from them.

| Reported selection | Result | Elapsed |
|---|---|---:|
| CPU/NumPy, single-thread BLAS/OpenMP; test-file globs `tests/test_020*.py tests/test_021*.py tests/test_022*.py tests/test_023*.py tests/test_0106*.py` | **1,650 passed, 654 skipped** | 19.35 s |
| Selected CuPy suite, excluding the two old CPU-only cases below | **894 passed, 20 skipped, 2 deselected** | 8.20 s |

The reported CPU environment was `MINI_LLM_BACKEND=numpy`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. The globs above record the reported pytest file selection; the CuPy totals describe a **selected suite**, not an unqualified all-tests GPU pass. The two deselected old CPU-only cases are:

- `test_packed_gguf_experts_cpu_fallback_and_cache_integration`
- `test_full_packed_expert_and_opt_out_cpu_reference`

Both pass in the CPU run. An exploratory GPU suite included the latter and failed on a **NumPy/CuPy mismatch**, an unchanged known issue in that CPU-reference case. Preserve that failure caveat: deselection does not mean the exploratory failure was fixed, nor does it establish that every backend/math subpath passed under CuPy. The final selected result and its exclusions are the supported GPU claim.

## Reference pins, licenses and Strata comparison caveats

Preserve the existing source pins and attribution from [reference_contract.md](reference_contract.md):

| Reference | Pinned commit |
|---|---|
| Strata | `c77572806996349b49995e4f98d41b5294d36a21` |
| Radiance | `a555ef392a48fbdac3df4a0eeb1197ec3f6e6100` |
| llama.cpp | `171e8846b4af9766c354064cb776cb34a50f053f` |

The inspected Strata checkout is **`/home/lucka/Documents/my_local_LLM/qwen4exp-references/Strata`**; its HEAD matches the pin. Strata's own GGML pin `3cf03257f219afbe7334045ff7c6a06ac68c627d` is separate from this project's llama.cpp reference. IQ lane equations use the existing canonical decoder/codebooks; GGML/llama.cpp MIT attribution and license from `gguf_codebooks.py` are embedded into generated CUDA source with the constant tables. Keep [third_party_ggml_LICENSE.txt](third_party_ggml_LICENSE.txt). Strata `iq_kernels.cu` was consulted for layout context, notably raw Q2_0 type 42; the new module states no Strata kernel code was copied. Nothing here changes source licenses or upgrades checkpoint filename/header provenance into a verified immutable payload revision.

Strata's pinned `docs/DETAILS.md`, “Speed (measured)” section, describes **RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows, engine 0.1.26**, one code-agent prompt per length, **256 generated tokens and MTP speculation enabled**, with setup-selected prefill/KV policies. Its table gives IQ3_S output **52.4 tok/s at 1K and 53.3 at 4K**, while the roughly 70–80 range also appears for other quantizations/context points. Its later history describes other hardware, runtime versions and launch/kernel settings; those are not measurements of this project's forced-token FP32 path.

Source caveat: `src/core/expert_source.cpp:4917–4930,5009,5307` and `src/program/generate.cpp:13817` onward implement/configure route-tail skipping. Where enabled, a missed expert routed sufficiently low by every token in the verification window can contribute zero rather than be copied/computed; no compensating renormalization restores the omitted contribution. Rank 7 is a conditional serve default, not a universal setting: cache residency, batching, peer/helper GPUs and expert formats affect eligibility. Set **`STRATA_ROUTE_TAIL_SKIP=0`** for an explicitly full-top-10 comparison, and record the effective startup settings rather than assuming a default. This optimization is not adopted here.

A **70–80 tok/s comparison requires like-for-like evidence**, not merely the same checkpoint name: actual weight revision/quantization, hardware/CPU/PCIe, prompt/output lengths and content, cache warmness/residency, placement, launch runtime (including CUDA graphs/compiled versus Python dispatch), arithmetic/KV modes, speculation/draft acceptance and route-skip settings must be recorded and aligned or explicitly separated. These artifacts do not establish that this implementation reaches that range, nor quantify how much any unimplemented feature would close the gap.

## Remaining limits

Next optimization priorities are **bounded transfer/admission**, **grouping the resident subset in partially resident dispatches**, and **graph/native dispatch to reduce Python launch overhead**. These are follow-up work, not implemented capabilities or measured speedups. Evaluate them with their own backend/arithmetic tests and like-for-like timing; no 70–80 tok/s claim follows from the current profile.

- A bounded-slot asynchronous expert mover/prefetch/publication pipeline is **not implemented** by these changes. First-use promotion and batched CPU staging are not that pipeline.
- CPU fallback remains when admission is unavailable or cache capacity/protection prevents residency. The short eager experiment did not exhaust its 6 GiB cache.
- Beyond 2051 QSA causal cells, the reference sparse-selection path remains; no end-to-end long-context performance claim follows from the standalone width tests or metadata context capacity.
- FP32 activation/state semantics, all ten routed experts and canonical packed weights are retained. There is no speculative decoding, route omission, full-model dequantized copy or claim that FP16 is preferred.
- Retained same-prefix logits establish small differences against the project's own baseline only. Independent full-model numerical validation remains a separate unresolved requirement.
- Timing results are single successful runs with historical artifact reuse, not repeated confidence-bounded benchmarks. No tests or model runs were performed to write this document.
