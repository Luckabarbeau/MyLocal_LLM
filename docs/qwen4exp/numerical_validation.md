# Qwen4Exp numerical validation record

## Current follow-ups — 0281–0288

Mixed-format GPU Q8_1 is now opt-in for IQ2_S,IQ3_S,IQ3_XXS,IQ4_NL,IQ4_XS,Q2_0,Q4_K,Q5_K,Q6_K,Q8_0. Tests compare extracted pinned Strata parts/rows exactly, independent FP64 decoded-Q8 products with explicit integer-truncation bounds, real bounded checkpoint slices, all seven expert triples, captured/native/mixed dispatch and output lifetime. CPU GGML Q8_K/Q8_0 is a separate contract. Flags do not convert generic warmup fused experts or prove whole-engine Strata parity.

Exact BF16 HC storage keeps FP32 arithmetic unchanged. Real model BF16/graph versus expanded/direct control, and native-service versus retained graph control, have bit-exact logits and all active-state hashes at1/2/4/8/16 over two requests. The final native service additionally tests timeout, epoch exhaustion, no payload reads on failure (null-source probe), poisoned output, failure before state advancement and retirement before pool destruction. An unconditional timeout-path DMA was corrected; completion status is now published once into VRAM before gated copying. Final v3 real-model logits/state hashes remain exact. Size-classed slots test exact bytes/output, compatible eviction under spare global bytes and held-view fallback. The mapped-host operator tier is numerically exact to VRAM Q8 but slower and not integrated.

Final selected regressions:1969NumPy passed/1124skipped;1367actualCuPy passed/14skipped, including Qwen3-Next GPU components. This is **not** a new independent-reference full-model or arbitrary-long-context equivalence claim. See `0281_0288_execution_optimization.md` and `0281_0288_performance_v3.json`. Failed earlier fixtures/probes are retained; most recent relevant suites pass, with no tolerance loosened to hide discrepancies.

## Historical Q8 execution follow-ups0252–0257

GPU Q8_1 producer matches pinned Strata712fec6b4297c9cfd32ab43daaa2979a9aa01edf byte-for-byte in41cases; IQ3_S×Q8_1 GEMV passes38 source-oracle/independent-baseline cases. These remain isolated GPU operators, not all-format model integration. Optional CPU `expert_cpu_backend='ggml'` uses the explicitly selected numerical-reference library's per-format Q8_K/Q8_0 integer dots; synthetic direct-GGML expert results agree exactly, while FP32-baseline differences are expected and separately measured. Library source ancestry differs from installed Strata; do not claim whole-engine exact parity. Full long requests complete with that backend, but finiteness/coherent text is not independent mathematical equivalence.

Opt-in mixed GPU/CPU grouping retains weighted-term FP32 storage and ascending expert addition, with within-row reduction differences bounded by existing GPU tolerances. New arithmetic/residency modes can change sampled trajectories; comparisons after divergent sampled tokens are not same-prefix logit comparisons. Earlier FP32 numerical oracle remains available. See0252/0253/0256/0257reports for precise scopes and remaining gates.

## Validation levels

Operator parity does not establish layer/model/generation parity. In particular, a passing shape/import test must not mark a graph numerically validated. Real-checkpoint logits have been checked for finiteness, prefill/incremental self-consistency and fresh-request isolation, and three greedy decisions have executed on CPU and GPU. They have also been compared against independent full-model reference logits; several-percent differences remain incompletely attributed. Subsequent identical-input MoE and16-token DeltaNet/QSA state replays expose no mathematical defect, and full-model active-state/prefill consistency passes through16 tokens. The scoped short-text gate permits initial profiling, not a claim of full numerical equivalence.

| Level | Evidence so far | Outstanding |
|---|---|---|
| Packed operator | Six new decoders match compiled pinned GGML reference lanes exactly; direct packed GPU GEMV/SwiGLU and grouped experts now tested across all actual format triples | Large-prefill GPU GEMM/kernel tuning |
| Expert | Native mixed-format oracle, six identical-input MoE reference cases,48 real GPU expert cases and grouped real top10 across7 format triples | Full-model arithmetic attribution; partial-resident grouping and async mover |
| Sparse dispatch | Generated-on-demand 512-expert/top-10 CPU oracle; supported-format GPU execution with IQ CPU fallback | Full 512-expert/top-10 GPU/mixed topology on actual model; all-IQ GPU unsupported |
| HC/PLE/GDN/QSA layer | FP64 synthetic graph tests; compiled source hash, recurrence, text RoPE and tie checks | Full independent linked-layer/checkpoint parity |
| Model | Original checkpoint executes on CPU/CuPy; finite logits and diagnostic hooks | Independent comparison now runs; logit maxabs~1, relativeL2~5–6%; unresolved numerical discrepancy |
| Generation | Three greedy decisions on both backends; exact within-backend fresh-state/prefill parity | Independent greedy parity, realistic prompt/tokenizer/chat serving |

## Quantization — Patch0202

References are extracted actual C reference functions compiled against original pinned headers/tables, not a second call into production decoders. SHA checks guard the external source roots. Optional references skip explicitly when absent; all were present in this development run.

All six new block decoders had **maximum absolute difference0**, including grid high bits, explicit/parity signs, scale encodings, lane orders and finite FP16 scales. IQ3_S is not treated as scalar uniform three-bit packing. Exact block comparisons use array equality.

Tiled GEMV uses FP32 accumulation; observed maximum absolute differences against independently decoded matrix multiplication: IQ2_S1.52588e-5, IQ3_S6.10352e-5, IQ3_XXS3.05176e-5, IQ4_NL3.81470e-6, IQ4_XS2.44141e-4, Q2_01.19209e-7. Tests use combined `atol=2e-5,rtol=3e-6`, not absolute tolerance alone. See `0202_quantization.md` for exact commands and historical integration failures.

## Native experts — Patch0203

Native identity-probe lanes matched independently C-decoded weights exactly. Largest observed synthetic FP32 expert error3.814697265625e-6; tolerance `atol=2e-5,rtol=5e-5`. Representative target D2560/F640 tests cover mixed IQ2_S/IQ3_S with IQ4_NL or Q2_0 down. Tests forbid production reference decoder calls during native execution and exercise immutable mmap buffers plus concurrent workers. Native FP64 input/output still performs FP32 arithmetic, as in the existing API; it is not a FP64 oracle.

## Sparse placement — Patch0204

CPU selected expert execution is unweighted; final combine applies routing weights once. New tests cover capability rejection before allocation, immutable complete source bundles, sparse selected bank slices, memory reservation rejection, native top-10 sparse output, bounded zero-host-cache reference computation, incomplete-materializer failure and in-flight retention.

Corrected actual-GPU tests initialize finite Q3_K/Q6_K scales at trailing offsets and Q4_K/Q5_K minimum scales correctly, and explicitly reject nonfinite results. Six supported-format mixed-path cases plus18 existing GEMV cases pass without warnings. New IQ GPU formats fail explicitly rather than being mistaken for Q3_K. No IQ GPU numerical claim is made.

## Known regression boundaries

Earlier integration runs exposed stale tests equating all CPU formats with GPU capability. The GPU GEMV parametrization now uses the actual GPU capability registry; CPU tests still cover every packed format. A legacy Llama import acceptance boundary was restored separately after registry expansion. Historical failures remain recorded in each patch document, and final consolidated outcomes are recorded in `implementation_status.md`.

Six older tests are not backend-neutral under a CuPy process: they pass NumPy to CuPy-only paths or implicitly convert a CuPy result in NumPy assertions. They were not rewritten as part of this change. Their names are listed in `0204_sparse_execution.md`; they are known broader-GPU-suite failures, not silently counted as passing.

## Real-checkpoint lifecycle — Patches0209–0210

`test_original_incremental_reset_and_three_greedy_tokens` passed under NumPy and CuPy with the original two shards. Input IDs `[1,2]`, greedy IDs `[220,100561,198]` on both. Within each backend, fresh-state incremental logits equal prefill logits exactly; all recurrent/QSA positions match. Six token forwards per test execute2880 native expert calls with zero decoded-expert host cache and no GPU expert residents. These checks do not establish independent numerical parity or cross-backend logit equality.

The original reference preflight exited2 before loading because of eager expert prefetch. That historical blocker was then resolved using a separate CPU-only source copy with exactly one loader-prefetch boolean change; the original pin remained clean. No reference graph/quantization mathematics changed. See `0210_noprefetch_reference.patch` and `0210_noprefetch_reference_validation.json`.

The actual comparison produced maxabs logit differences0.958172 /1.006777, relativeL2=6.1643% /5.0675% after prefixes[1] /[1,2]. Both argmax IDs match (271 then220), top10 overlap9/10, maxabs log-probability differences1.07785 /0.94804. Both stored outputs are FP32; production FP32 native activation arithmetic differs from GGML quantized-activation matmul. No acceptance tolerance was imposed and these results do **not** pass the numerical milestone. Reference peakRSS5.72GiB; production8.71GiB, sequential processes.

The subsequent0218–0223 investigation verifies all386 boundary comparisons by exact names and graph ancestry. The old prefix selector captured scheduler-created reshaped aliases and overwrote HC files. Corrected layer0 attention-read error is0.00102728 relativeL2, not1.832984. Composed BF16-activation HC replay reduces it to8.48e-8, explaining the first genuine divergence without changing production. Forty-five isolated arithmetic cases quantify Q8_K/Q8_0/BF16 rounding. Full diagnostic activation matching still leaves5–7% final-logit differences. Follow-up0224–0225 fixed the state diagnostic and completed35 retained state comparisons without model execution. Six identical-input MoE cases match top10 sets; teacher-forced operations agree at roughly1e-7 precision, and controlled quantization/input perturbations demonstrate amplification to percent-level logits. Follow-up0226 validates16-token representative DeltaNet/QSA updates with236 comparisons each in teacher-forced and continuous-state modes. Follow-up0227 establishes bit-exact full-model local incremental/prefill logits and all active-state hashes at prefixes1/2/4/8/16. No production mathematics changed. Full independent model/long-context equivalence is still not established, but the requested short-text gate passes and initial profiling has begun without a claimed speedup. Latest consolidated regression:1492 passed,186 skipped. See numerical_root_cause.md, activation_quantization_comparison_v2.json, layer_error_evolution.csv/png and state_validation.md. Historical result files are preserved.

## Patch0213 optimization validation

FP32 packed GPU IQ kernels, grouped experts and switchable HC/QSA/router/GDN fusion preserve the established mathematical graph. Real CPU/GPU expert comparison worst relativeL2=1.622820435e-6. Same-prefix optimized full-model versus prior FP32 baseline: relativeL2≤2.065315e-6, maxabs≤2.789498e-5, matching argmax/top10 at prefixes1/2/4/8/16. This is a new optimization comparison, not a resolution of the5–6% independent GGML discrepancy.

Optimized16-token prefill/incremental logits and all active-state hashes are bit-exact, and two fresh requests reproduce greedy IDs `[220,100561,198]` (`0233_fast_lifecycle_v1.json`). CPU regressions1650 passed/654 skipped; selected CuPy regressions894 passed/20 skipped/2 documented CPU-only exclusions. No long-context or whole-repository GPU claim. See `0213_packed_gpu_and_decode_optimizations.md` for controls and test scope.

## Follow-up0234–0238 validation

Global codebook storage and grouping after complete admission preserve output bits. Warp-per-row scheduling changes FP32 summation order; real top10 worst relativeL2=3.176314e-7 versus block kernels across7 actual format triples. Full-model warp/fixed-slot optimization errors versus the original FP32 production baseline are ≤2.014602e-6 relativeL2 and ≤2.717972e-5 maxabs, with matching checked argmax/top10 sets.

Dynamic-warp and fixed-slot logits/all active-state hashes are bit-exact at1/2/4/8/16. The new fixed-slot lifecycle (`0233_fast_lifecycle_v2.json`) passes16-token incremental/prefill equality and repeated fresh greedy IDs `[220,100561,198]`. HC graph capture is separately tested, disabled by default and not a measured speedup. No extra independent full-model GGML comparison is claimed.

Latest CPU regression includes existing inference and selected training components:1691 passed/780 skipped. Selected actual CuPy suite:1031 passed/22 skipped/2 documented CPU-only exclusions. All previous reference discrepancy/long-context limitations remain.

## Required future diagnostic contract

For every graph comparison retain input IDs, source/weight revisions, activation dtype, execution policy, KV/state precision, context and reference flags. Compare intermediate HC reads/writes, PLE indices/history, GDN output/state, QSA selection/masks/KV/indexer state, routed/shared outputs, then logits. Report the **first** divergent operation and tensor, absolute/relative errors, requested tolerances and accumulation/storage boundaries. Correct-looking text is not evidence. Do not broaden tolerances to conceal state/order errors. Disable Strata route-tail skipping for exact full-expert comparisons.
