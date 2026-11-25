# WP7 historical numerics: evidence and limits

## Conclusion and audit scope

**Yes: the earlier Python implementation already exhibited small conditional floating-point differences that BF16/Q8 activation arithmetic could amplify into percent-level full-model logit differences.** Historical teacher-forced operations often agreed at roughly `1e-7` relative L2; continuous, unforced intermediate trajectories did not necessarily remain that close. This is evidence for numerical sensitivity, **not automatic justification for global `allclose`, a new tolerance, or acceptance of every later discrepancy as roundoff**.

This document audits retained records only. No model execution, build, test, GPU query, runtime/test edit, or status-file update was performed. The only new file is this document. Historical test outcomes below are reported evidence, not tests rerun for this audit. Any subsequently authorized GPU validation must use **physical GPU 1 only**; a historical logical device number is not a new device-selection instruction.

Primary records inspected:

- `docs/qwen4exp/numerical_validation.md`, real-checkpoint lifecycle and 0218–0227 summary (lines 88–96).
- Full `docs/qwen4exp/numerical_root_cause.md`, especially sections A–H and its later precision/optimization caveats.
- `docs/qwen4exp/activation_quantization_comparison_v2.json`: metadata, individual-case schema/example, and all 45 cases aggregated read-only.
- Full `docs/qwen4exp/state_validation.md`.
- `docs/qwen4exp/0310_0317_strata_comparison.md`, 0314 subsection; `docs/qwen4exp/0314_qsa_operator_v1.json`; corresponding 0314 discussion in `numerical_root_cause.md`.
- `docs/native_engine/WP7_DIVERGENCE.md`, section 8, compared with fields in `docs/native_engine/WP7_FULL_MODEL_DIAGNOSTICS.json`.

Relative L2 values below are norm-relative errors, not percentages of individual elements or componentwise `allclose` limits. Max-absolute errors have the units of the compared tensor. Historical artifact names in the next sections are relative to `docs/qwen4exp/` unless otherwise stated.

## 1. What the earlier Python investigation established

### Capture identity came before numerical attribution: 0218–0221

`numerical_root_cause.md`, sections A–C, invalidates the old HC-read relative L2 `1.832984`: prefix selection captured scheduler-created reshaped aliases and overwrote the attention-read file. Corrected captures use exact names and norm-weight ancestry, not allocator address alone. This was an instrumentation defect, not a production HC equation correction.

For position 0/layer 0, the corrected production/reference attention HC read has maxabs `0.0147502422`, relative L2 `0.00102728060`. The composed replay (`0221_hc_arithmetic_v1.json`, as recorded in the root-cause report) uses the same embedding and full HC weights:

| Comparison | Relative L2 | Maxabs |
| --- | ---: | ---: |
| Production vs unrounded replay | `5.69648e-8` | `4.76837e-7` |
| Unrounded replay vs verified GGML read | `1.02728709e-3` | `0.0147504807` |
| BF16-activation replay vs verified GGML read | `8.48003e-8` | `4.76837e-7` |

Only the two HC projection activation casts were changed in diagnostic replay; normalization, gamma, nonlinearities and combination stayed F32. This explains the **first verified HC divergence for this input**, not every layer/state.

### Activation representation is distinct from FP32 accumulation: 0219

`activation_quantization_comparison_v2.json` labels itself **bounded arithmetic only; not full model parity**. It records linked single-thread CPU GGML graphs, upstream codecs/dequantizers, F32 input storage, 15 checkpoint tiles × 3 inputs = 45 cases, 32 output rows per tile, and `1263360` packed bytes read. Some inputs are retained activations; others are deterministic substitute-width/random/multirange inputs.

The following ranges were extracted from all cases, grouped by `traits.activation_type`:

| CPU activation representation | Cases | `activation_rounding_vs_fp64.relative_l2` | `graph_vs_rounded_oracle.relative_l2` |
| --- | ---: | ---: | ---: |
| Q8_K | 30 | `0.002320989397419537`–`0.014748766508204052` | `5.639714190568481e-8`–`4.3269801881522756e-7` |
| Q8_0 | 9 | `0.0030363630065313365`–`0.00785954940177478` | `2.886591593005075e-8`–`1.0084474534224863e-7` |
| BF16 | 6 | `0.0012123560312913124`–`0.0017271791000969356` | `5.496316940111429e-8`–`1.5063335618792788e-7` |

All measured `optimized_reference_bytes_equal` flags are true. The decoded-weight F64 baseline isolates activation representation; the rounded-input oracle isolates the remaining arithmetic discrepancy. For example, the first case (`blk.17.ffn_gate_exps.weight`, IQ3_S, expert 511, rows 0–31, `real_A`) has F32-vs-F64 relative L2 `3.4701629273024917e-7`, activation-rounding effect `0.013360169626647583`, and graph-vs-rounded-oracle `8.082074216434805e-8`.

The weight-to-CPU-activation mapping is Q8_K for IQ3_S/IQ3_XXS/IQ2_S/IQ4_XS/Q4_K/Q5_K/Q6_K; Q8_0 for IQ4_NL/Q2_0; BF16 for BF16 weights. An FP32 output tensor does not imply that its matmul consumed unquantized FP32 activations.

### Teacher forcing localized operations, not full trajectories: 0224–0225

`numerical_root_cause.md`, sections E–F, reports six identical-reference-input MoE cases at `(position, layer)` `(0,0)`, `(1,0)`, `(0,1)`, `(0,2)`, `(0,17)`, `(0,45)` from `0224_isolated_moe_v1.json`. All six top-10 **sets** agree; order is recorded separately and expert rows are aligned by identity.

- Matched router logits: relative L2 `7.52e-8`–`8.87e-8`.
- Selected normalized weights: `1.44e-7`–`3.08e-7`.
- SwiGLU given identical gate/up: `3.78e-8`–`1.08e-7`.
- Down given identical reference SwiGLU and required Q8 rounding: `3.97e-7`–`4.85e-7`; shared down: `3.34e-7`–`4.92e-7`.
- Combination given identical reference outputs/weights: `3.96092e-8`–`9.71085e-8`.

Even an identical-input **composed** MoE is not uniformly `1e-7`: activation-matched final errors range from `3.53244e-7` to `3.73278e-4`. At layer 2, shared SwiGLU error `1.17953e-6` becomes shared-down error `5.50407e-4`; supplying the exact reference SwiGLU reduces it to `3.76204e-7`.

`0225_rounding_sensitivity_v1.json`, as recorded in section F, provides causal interventions:

- A relative input perturbation `1.47270e-7` changes one decoded Q8-rounded element and takes shared-down relative error from `3.76204e-7` to `5.50407e-4`.
- A single layer-0/position-0 MoE-input perturbation, relative L2 `1.31186561e-6`, maxabs `5.48362732e-6`, changes one BF16-rounded element in the positive case. With forced IDs `[1,2]`, downstream logit changes versus the unperturbed activation-matched model are:

| Perturbation sign | Logit position 0 relative L2 / maxabs | Position 1 relative L2 / maxabs |
| --- | --- | --- |
| Positive | `0.0411539` / `0.6865921` | `0.0230992` / `0.4243503` |
| Negative | `0.0348462` / `0.5346808` | `0.0633422` / `1.1362445` |

Thus an ordinary F32-scale perturbation can produce **2.31–6.33%** logit changes in that quantized diagnostic pipeline. This is a sensitivity witness, not an attribution of every baseline discrepancy.

### Independent full-model differences remained: 0220–0223

`0220_layer_evolution_v2.json`, summarized in root-cause section C, contains 386 identity-verified comparisons over all 48 layers for IDs `[1,2]`. Those are **propagated trajectory comparisons**, not 48 layers of identical-input conditional replay.

| Prefix | Baseline logit relative L2 | Baseline maxabs | Diagnostic activation-matched relative L2 / maxabs |
| --- | ---: | ---: | --- |
| `[1]` | `0.0616433932` | `0.958172083` | `0.0691502` / `1.0631301` |
| `[1,2]` | `0.0506754260` | `1.006776512` | `0.0550970` / `0.9776649` |

These are the historical **roughly 5–7% logit differences**. Matching activation codecs did not establish full GGML arithmetic equivalence (`0222_matched_activations_v1.json`); local dots, nonlinearities and state operators still ran locally. Both baseline argmax IDs matched (271, then 220); that did not establish numerical/quality acceptance.

`state_validation.md` distinguishes the 35 sampled independent comparisons (`0223_state_evolution_v2.json`) from whole-state equivalence. At prefixes 1/2/4, logit relative L2 is `0.0616434 / 0.0506754 / 0.0410662`, and layer-0 recurrence error is `0.00548179 / 0.00540416 / 0.00500166`. Fourth-token layer-3 K/V/raw-indexer errors are `0.0328451 / 0.0341913 / 0.0211221`. Their upstream inputs already differ; these are cache-input captures, not complete physical KV exports. PLE selected IDs and initial zero states agree. Nonmonotone short-range error is not proof of general stability.

### Representative continuous state and whole-model self-consistency: 0226–0227

`state_validation.md` and root-cause sections G–H report:

- `0226_identical_block_states_v2.json` and `0226_continuous_block_states_v1.json`: 236 comparisons each over IDs 1–16, independent reference **prefix layers 0–3 only**. One mode teacher-forces reference pre-state each step; the other evolves local state from zero without per-step resets, using captured operands to separate state equations from projection arithmetic.
- Continuous maxima: DeltaNet matrix relative L2 `8.57284e-7`, normalized/gated output `2.21826e-7`, QSA gated output `3.26062e-7`, K update `1.08275e-7`, completed indexer pools `8.46406e-8`, indexer scores `1.82212e-7`. Identical-input GDN/QSA output-projection maxima are `1.32276e-6 / 1.99285e-6`, not uniformly `1e-7`.
- `0227_short_sequences_profile_v1.json`: **local Python incremental versus chunked prefill**, all 48 layers, bit-exact logits and hashes of all active persistent state at prefixes 1/2/4/8/16. This does not compare independent full-model reference logits/states over 16 tokens.

The independent 16-token QSA case fits every causal cell inside width 2051; it does not independently certify sparse selection beyond that width, arbitrary long context or multimodal positions.

## 2. FP32 storage, activation quantization and reduction order are different claims

Three distinctions are essential:

1. **Storage:** FP32 saved activations, states or logits say how values are represented at those boundaries. Original packed/BF16 weights may remain stored compactly. The later direct-BF16 HC-storage optimization in root-cause section 0281–0288 preserved checkpoint values and original FP32 operations/reduction order, with exact local logits/state parity. It was not a new BF16 activation policy.
2. **Internal activation arithmetic:** CPU GGML Q8_K/Q8_0 or BF16 operand conversion rounds an activation inside a matmul even if its API input and output are F32. Historical Python-versus-GGML comparisons could therefore differ in activation representation, not just summation order.
3. **FP32 operation order:** two unquantized FP32 operators can differ by a few ULP because of reductions, FMA/order or layout, without introducing BF16/FP16/TF32. Those small differences can encounter an existing downstream quantizer and be amplified. Percent-level final errors must not be described as percent-level plain FP32 accumulation error.

### 0314: a direct FP32-order precedent, explicitly unaccepted

`0314_qsa_operator_v1.json` specifies FP32 only: no tensor-core dot, TF32, FP16 or BF16. Tree reductions and split-context log-sum-exp merging differ from serial accumulation. Standalone synthetic FP64-oracle relative L2:

| Context cells | Cooperative vs FP64 | Serial vs FP64 | Cooperative vs serial |
| --- | ---: | ---: | ---: |
| 141 | `1.662529570399675e-7` | `4.981483416630426e-7` | `4.923039637136669e-7` |
| 435 | `1.6408297615292512e-7` | `5.166030692817455e-7` | `5.370676437328804e-7` |
| 1024 | `1.8508544561513247e-7` | `5.621060961653984e-7` | `5.898693414239123e-7` |

`0310_0317_strata_comparison.md`, 0314 subsection, reports versus 0313: prefix 1 exact, but prefixes 2/4/8/16 logit relative L2 `0.045195973 / 0.036204690 / 0.068878359 / 0.052042546`, reproduced after reset; state hashes also differ. Historical operator/integration results were 44 passed/1 skipped and 39 passed/1 skipped. Operator acceptance did **not** establish model quality acceptance: the candidate remained opt-in and **NOT quality accepted**. The root-cause report explicitly treats quantization amplification as plausible, not complete attribution of this candidate's state differences.

## 3. Exact comparison with WP7 section 8

WP7 compares **native versus unchanged Python**, not historical Python versus independent llama.cpp. `WP7_FULL_MODEL_DIAGNOSTICS.json` records arithmetic `cpu_ggml_q8_gpu_reference_fp32`, tokens `[1,7]`, full 48-layer original models, all-CPU mmap routed experts, FP32 GPU branches/native paged QSA, 192 layer-boundary arrays per token and 248320 logits. Native reset replay is exact. This is broader trajectory coverage than a prefix model, but not all-48 conditioned component validation.

### Independently localized native/Python amplification

`WP7_DIVERGENCE.md` section 8 and JSON fields `token0.first_dense_hc_identical_embedding`, `token0.same_input`, `token0.q8_k_changes`, and `gdn8` establish:

- First HC on the identical decoded embedding: hidden maxabs `4.76837158203125e-7` (ID 1), `1.430511474609375e-6` (ID 7); relative L2 `6.938541844042036e-8 / 1.1067949244205178e-7`. This difference occurs **before CPU activation quantization**, not from an embedding-codec mismatch.
- Token 0/layer 6: different hidden lineages differ by maxabs `8.34465027e-7`, relative L2 `2.27641651e-7`. Both implementations on either identical hidden have **exactly equal ten CPU rows**, same selected IDs, combined-output maxabs `1.862645149230957e-8 / 1.1175870895385742e-8`, and zero existing-envelope violations. Combined-output relative L2 is `2.639981305705635e-7 / 1.7702275344614185e-7`.
- Across those different lineages, CPU rows differ by maxabs `0.000377766788`, relative L2 `0.00264213406`; routed output by maxabs `0.000149392523`, relative L2 `0.00222906445`. Routing membership does not flip; the native tenth/eleventh logit margin is `0.0010704994201660156`.
- Actual GGML Q8_K bytes expose block 9/lane 190, coordinate 2494: inputs `0.4080895185470581 / 0.4080897271633148`, stored scales `0.014318935573101044 / 0.014318937435746193`, integer codes **28 / 29**. Input/stored-scale diagnostics are `28.499989853552947 / 28.500000715454508`. Q8_0 bytes are unchanged. This is measured threshold crossing, not a hypothetical router flip.
- Token 1/layers 6 and 7: CPU rows remain equal even across lineages. Layer-7 original internal owner buffers were not retained; standalone replay reconstructs aggregate HC writes, which is a narrower provenance claim.
- Layer-8 GDN conditioned on either complete two-token hidden history: output maxabs at most `2.98023224e-8`, convolution history at most `9.53674316e-7`, matrix at most `5.96046448e-8`; existing checks pass. Holding current input identical while changing prior history produces output maxabs `0.00012425333261489868`, relative L2 `0.0013051676327233443`. The actual lineage output delta is `0.000124245882`; history-only residual maxabs is `2.123415470123291e-7`, residual L2/actual-delta norm `0.0006326991060732575`.

This directly supports upstream FP32 roundoff → Q8_K threshold → downstream/state propagation **for the measured causal path**. The historical precedent is corroborating evidence, not the sole basis for that WP7 diagnosis.

### Strict WP7 acceptance still fails

The unchanged existing componentwise rule is `rtol=2e-4, atol=2e-4`, both reported dtypes F32. It is quoted here, not changed or newly endorsed.

| Metric | Token 0 / ID 1 | Token 1 / ID 7 |
| --- | ---: | ---: |
| First violating boundary | Layer 7 HC FFN read | Layer 8 HC FFN read |
| Boundary maxabs | `0.0015584230422973633` | `0.000667273998260498` |
| Boundary relative L2 | `0.0005356750911888023` | `0.00032932771588950623` |
| Boundary violations / 2560 | `719` | `287` |
| Full-logit maxabs | `0.15479016304016113` | `0.41910791397094727` |
| Full-logit relative L2 | `0.011583157054027043` | `0.037831372035262864` |
| Logit violations / 248320 | `244163` | `247078` |
| Greedy ID native/Python | `271 / 271` | `16 / 16` |
| Top-10 membership overlap | `10/10` | `10/10` |
| KL(Python || native) | `0.0002736619930995807` | `0.0038681934751346988` |
| Probability total variation | `0.008869246762275665` | `0.037915566229285144` |

These WP7 logit errors are **1.16% / 3.78%**, not the historical Python/GGML 5–7%; the comparators, tokens and arithmetic policies differ. Matching top-1/top-10 membership does not imply equal ordering, distributions, perplexity or generated quality.

JSON retains `strict_whole_model_acceptance_passed=false`, per-token strict flags false and quality flags false. `diagnostic_acceptance.scope` is only same-input MoE and same-input-history GDN8, with `changes_production_acceptance=false`. Historical focused/bounded results (11 passed; 134 passed) do not waive the failed trajectory gate or close WP7/WP2.

## 4. Missing conditioned checks needed to distinguish roundoff from a bug

The current evidence does not identify an incorrect production equation in the localized path, and does not exclude bugs elsewhere. Full trajectory capture alone cannot distinguish a new local defect from propagation of already different operands. To make an **all-48, tested-input attribution**, the missing evidence is:

1. **Condition every layer on identical complete operands from both retained lineages.** For each layer 0–47 and each fixed token, retain verified identities/layouts and compare HC attention/FFN reads (hidden and gains), mixer branches, full MoE, paired HC writes, final collapse/norm/head. Reuse the same original streams and corresponding read's gains for each write; do not substitute a different boundary or compare already-divergent inputs as if identical. Keep full, uninjected trajectories separate from conditional jobs.
2. **Condition state as well as current hidden.** For every recurrent layer, compare projections, convolution/history, recurrence matrix, gates/output and commits with identical pre-state; also replay identical complete input histories continuously from zero without resetting each step. For every QSA layer, condition active KV, positions/RoPE, raw/pooled index state and masks/selected cells; check updates and output. Include PLE history/predecessor IDs and row selection. Compare logical active cells with verified layout conversions, not unwritten capacity or masked padding. Representative layers 0–3 historically and GDN8 in WP7 do not cover the other stateful layers.
3. **Locate every material discontinuity under matched arithmetic.** At each remaining growth boundary, retain pre-quantizer F32 operands and actual codec bytes/scales/codes (including expert gate/up/hidden/down boundaries); compare routing logits, selected identities/order/weights/margins, all ten unweighted rows, exactly-once weighting, shared branch and ordered sum. Use both input lineages and keep quantization-representation changes separate from FP32 reduction differences. An independent decoded-weight F64 or linked GGML oracle can isolate accumulation from codec effects at a disputed operation. A persistent mismatch on identical inputs/state/codec bytes needs localized equation/layout/ownership investigation, not a roundoff label.
4. **Close the attribution with controlled replay, not just an error jump.** At each unexplained boundary, test whether the measured upstream difference reproduces the downstream delta while keeping all other inputs/state fixed, and whether identical intermediates remove it. Preserve per-layer/token maxabs, relative L2, componentwise violations and exact discrete-state/ID checks, with source/library/artifact provenance. Extend conditioned coverage to longer fixed sequences and sparse-selection transitions before claiming those regimes; neither two WP7 tokens nor the historical 16-cell case certifies them.

The single next useful experiment is an **all-48 conditioned replay of the retained two-token native/Python captures, including matched pre-state/history**, reporting the first unexplained local mismatch separately from propagated trajectory drift. This is a diagnostic evidence requirement, not a runtime redesign or a new acceptance policy. Even successful conditioned checks would be scoped to the tested inputs and would not automatically pass the existing strict full-model gate or establish model quality.

**Bottom line:** historical logs confirm that the Python implementation had analogous quantized numerical sensitivity. They strengthen, but do not generalize beyond, WP7's measured roundoff/threshold/history explanation. Preserve the failed strict status and existing arithmetic policies; do not infer global `allclose` or complete equivalence from small teacher-forced errors, deterministic reset, or matching sampled tokens.
