# Qwen4Exp numerical validation record

## Validation levels

Operator parity does not establish layer/model/generation parity. In particular, a passing shape/import test must not mark a graph numerically validated. Real-checkpoint logits have been checked for finiteness, prefill/incremental self-consistency and fresh-request isolation, and three greedy decisions have executed on CPU and GPU. They have **not** been compared against independent full-model reference logits.

| Level | Evidence so far | Outstanding |
|---|---|---|
| Packed operator | Six new decoders match compiled pinned GGML reference lanes exactly; five IQs also match installed gguf | GPU IQ kernels |
| Expert | Native packed GEMV/SwiGLU/down, mixed D2560/F640 and concurrent workers vs bounded FP32 reference | Actual routed layer activations and reference arithmetic diagnostics |
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

386 boundary comparisons were captured, but the first reported layer0 HC-attention difference has a suspect duplicate-name association: attention/FFN reference read norms coincide. This is not evidence of a specific incorrect production equation. Numerical debugging stopped without changing production code. The single next experiment is token1/layer0 capture with tensor/op/source identities to verify association, then classify the first actual divergence.

## Required future diagnostic contract

For every graph comparison retain input IDs, source/weight revisions, activation dtype, execution policy, KV/state precision, context and reference flags. Compare intermediate HC reads/writes, PLE indices/history, GDN output/state, QSA selection/masks/KV/indexer state, routed/shared outputs, then logits. Report the **first** divergent operation and tensor, absolute/relative errors, requested tolerances and accumulation/storage boundaries. Correct-looking text is not evidence. Do not broaden tolerances to conceal state/order errors. Disable Strata route-tail skipping for exact full-expert comparisons.
