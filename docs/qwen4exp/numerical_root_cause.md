# Qwen3.8-Flash-Next numerical investigation — short-text validation gate passed

## Conclusion

**The evidence supports legitimate arithmetic differences and quantization amplification, not a demonstrated production graph defect.** The earliest genuine divergence is explained by BF16 activation rounding. Identical-input MoE tests now verify routing, expert projections, SwiGLU, shared experts and combination at selected early/late layers. A one-time 1.31e-6 relative perturbation produces 2.31–6.33% final-logit changes in the quantized-arithmetic diagnostic path. This demonstrates sensitivity at the observed discrepancy scale, but is not a proof that every operation/state is correct.

Production model/kernel mathematics and the pinned reference's mathematical implementation remain unchanged. Identical-input DeltaNet/QSA replay and continuous-state replay now pass over 16 fixed text tokens; full-model incremental/prefill logits and all active-state hashes match at prefixes 1/2/4/8/16. **The requested short-text validation gate has passed, with no demonstrated mathematical defect.** Initial performance profiling has started. Full independent model equivalence and arbitrary long-context correctness remain unclaimed.

## Provenance and safety

Original source HEAD: `171e8846b4af9766c354064cb776cb34a50f053f`, clean at inspection. Isolated reference: `/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch`, CPU-only shared build. Positive source inventory comparison verifies the only source difference is `init_mappings(true, ...)` → `false`; zero-prefetch control flow and CPU build flags were checked before loading. This run compared 3,603 source files; earlier report counts differ because inventory exclusions differ, not because a different model graph was authorized.

Original checkpoint shards remained mapped read-only. No checkpoint download, weight conversion, expert-bank expansion, or concurrent reference/production model residency occurred. Model probes use one CPU thread, context 32, one sequence, FP32 KV, lazy PLE and no CPU weight repacking. CPU child probes use a sampled 12 GiB RSS limit and 112 GiB address-space limit; these are not a hard cgroup RSS guarantee. The subsequent CuPy short-sequence probes retain the RSS/time guards and explicit runtime VRAM planner but omit the address-space cap because CUDA reserves large virtual ranges. State diagnostics used a shared 120-second deadline. The original source/CUDA build was untouched.

## A — Confirmed reference instrumentation defect

### Identity evidence

At token position 0, layer 0, attention and FFN HC reads are **different tensor objects** with the same name `hc_mixed-0`. Both use the same allocator data address at different times. Their copied values differ. Graph ancestry reaches `blk.0.hc_attn_norm.weight` or `blk.0.hc_ffn_norm.weight`, respectively. Data address is not tensor identity, and evaluation occurrence alone is not a reliable stage identifier.

A's capture reads were contiguous F32 with GGML `ne=[2560,1,1,1]`, `nb=[4,10240,10240,10240]`. These are already-collapsed reads: dimension 1 is token count, **not** four residual streams. Residual writes use `ne=[2560,4,1,1]`, `nb=[4,10240,40960,40960]`, compatible with Python C-order `[4,2560]` flattening.

### Why the old norms were identical

The original `selected()` used prefix matching. It selected both `hc_mixed-0` and scheduler-created `hc_mixed-0 (reshaped)` nodes. The old `stoi` layer parser accepts the numeric prefix before the suffix. Occurrence counters are keyed by the **full name**, so the reshaped FFN alias starts its own occurrence count at zero and is written to `hc_attention_read`, overwriting the real attention vector. This is a diagnostic file-association error, not evidence of an incorrect HC equation.

The first all-layer capture recorded 486 events rather than 386: 144 HC-read-prefix events per token rather than 96, plus additional `l_last ... (reshaped)` aliases. Retained v1 events provide the source-pointer/name evidence. A localized diagnostic correction requires exact numeric-suffix boundary names; stage classification then follows norm-weight ancestry. No reference graph names or model mathematics changed.

### Corrected measured boundaries, position 0/layer 0

| Boundary | Max absolute error | Relative L2 |
|---|---:|---:|
| HC attention read | 0.0147502422 | 0.00102728060 |
| Post-mixer residual | 0.000964082778 | 0.0177257835 |
| HC FFN read | 0.0389494300 | 0.0208672249 |
| Post-MoE residual | 0.00114515424 | 0.00956588161 |

The old 1.832984 relative-L2 HC-read comparison is invalidated **as a capture comparison**, not fixed by a production code edit.

Artifacts: `0218_hc_identity_v2.json`, `diagnostics_0218_v1/`, `0220_layer_evolution_v2.json`. Initial failed reporting/capture artifacts remain preserved as v1 files. Values were copied inside eval callbacks before allocator reuse.

## B — Quantified activation arithmetic

The experiment executes actual linked single-thread `ggml_mul_mat` graphs, upstream weight dequantizers and activation codecs. It compares each graph with (a) F32 decoded-weight multiplication, (b) an F64 accumulation oracle using the same decoded weights, and (c) the decoded-weight product with upstream-rounded activations. No local quantized dot transcription serves as the GGML oracle.

45 cases cover 15 original checkpoint tiles, 32 output rows each, with 1,263,360 total packed weight bytes read. Inputs are an actual retained HC-read vector where widths match, plus deterministic substitute-width, random and multirange inputs. Not all cases use actual model activations; the JSON identifies each source.

| Checkpoint weight types | Actual CPU activation representation |
|---|---|
| IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, Q4_K, Q5_K, Q6_K | Q8_K |
| IQ4_NL, Q2_0 | Q8_0 |
| BF16 | BF16 |

| Activation representation | Relative-L2 effect versus decoded-weight F64 oracle | Graph versus rounded-input oracle |
|---|---:|---:|
| Q8_K | 0.00232–0.01475 | 5.64e-8–4.33e-7 |
| Q8_0 | 0.00304–0.00786 | 2.89e-8–1.01e-7 |
| BF16 | 0.00121–0.00173 | 5.50e-8–1.51e-7 |

Reference and CPU activation-quantizer bytes matched in all measured cases. These observations distinguish activation representation error from ordinary dot-product accumulation differences. They support retaining production FP32 activations: closer agreement with the decoded-weight mathematical operation is not a defect merely because GGML rounds its inputs.

Detailed formats, tensors, row/expert selections, input characteristics, max-absolute/relative errors and F32/F64 baselines: `activation_quantization_comparison_v2.json` (full versioned report). The requested `activation_quantization_comparison.json` remains the original summary; `0219_activation_quantization_v1.json` is preserved.

## Earliest genuine divergence — composed HC proof

Using the **same original token-1 embedding and full layer-0 HC weights**, replay only the two HC projection activation casts. Norm gamma, normalization, nonlinearities and combination stay F32. Both BF16 cast inputs were checked exactly against upstream conversion.

| Comparison | Relative L2 | Max absolute error |
|---|---:|---:|
| Actual production read vs unrounded replay | 5.69648e-8 | 4.76837e-7 |
| Unrounded replay vs verified GGML read | 1.02728709e-3 | 0.0147504807 |
| BF16-activation replay vs verified GGML read | 8.48003e-8 | 4.76837e-7 |

**Thus the first trustworthy divergence is accounted for by BF16 rounding in HC down/up matmuls.** Remaining error in this isolated comparison is at F32 accumulation/nonlinearity scale. This establishes the first HC read's mathematical ordering for this real input; it is not a proof for all layers or states. Artifact: `0221_hc_arithmetic_v1.json`; source: `scripts/probe_0221_hc_arithmetic.py`.

## C — Verified propagation through all 48 layers

`0220_layer_evolution_v2.json` contains 386 identity-verified comparisons across fixed IDs `[1,2]`: attention HC read, post-mixer residual, FFN HC read, post-MoE residual for every layer, plus final HC collapse. Source ancestry classifies HC reads; injection ancestry and block/residual/gain dimensions validate writes. Exact names exclude reshaped aliases. All compared values are finite and shape-checked.

The original logit results reproduce unchanged:

| Prefix | Max absolute logit error | Relative L2 | Actual/reference argmax |
|---|---:|---:|---|
| `[1]` | 0.958172083 | 0.0616433932 | 271 / 271 |
| `[1,2]` | 1.006776512 | 0.0506754260 | 220 / 220 |

Residual relative errors are not monotone. Post-mixer range across both positions: 0.00917–0.08893; post-MoE: 0.00927–0.09657. At layer 47, post-MoE errors are 0.07601 and 0.08434. A large positive stage-to-stage change occurs at layer 45 FFN read (position 0: +0.06032); QSA mixer layers 43 and 47 also show increases. These are observations, not proof that QSA or HC is defective. Each boundary has its own norm/conditioning, so an error-ratio jump is not a causal variance decomposition.

Artifacts: `layer_error_evolution.csv`, `layer_error_evolution.png`, `0220_layer_evolution_v2.json`. CSV includes max error, relative L2, RMS difference/reference RMS, cosine, norm and nonfinite count; JSON additionally retains large-magnitude-element errors and tensor/source metadata. Plot panels (top-left to bottom-right): attention HC read, post-mixer residual, FFN HC read, post-MoE residual. Blue=position0/ID1, red=position1/ID2; gray bands=QSA layers. X spans layers0–47; Y is log10 relative L2 from -12 to1. The PNG includes this legend in its metadata and uses no optional plotting dependency.

Measured C reference/production sampled peak RSS: 6,194,638,848 /9,418,584,064 bytes; sequential run times 1.41 /23.43 seconds.

## Controlled activation-matched full replay — important rejected conclusion

A diagnostic-only child monkey-patches projection activation rounding, HC down/up/injection inputs, router/shared gate inputs and expert gate/up/down inputs using upstream codecs. Production files remain untouched. Local packed dots, nonlinearities and state operators still execute locally; this is **not** the full GGML arithmetic path.

It nearly removes layer-0 differences at position0: HC read7.22e-8, post-mixer1.27e-6, FFN read1.31e-6, post-MoE6.86e-6 relative L2. But later differences grow: position0 layer1 post-MoE5.48e-4, layer2 post-mixer3.04e-3. Final logits remain different, even slightly more than baseline:

| Prefix | Baseline relative L2 | Diagnostic matched-activation relative L2 | Matched max absolute error |
|---|---:|---:|---:|
| `[1]` | 0.0616434 | 0.0691502 | 1.0631301 |
| `[1,2]` | 0.0506754 | 0.0550970 | 0.9776649 |

**Rejected:** matching activation codecs alone establishes whole-model equivalence. It does not. Tiny intermediate differences can cross subsequent quantizer/routing thresholds, but that causal explanation is not yet demonstrated for the later growth. A missed arithmetic boundary or actual model/state defect also remains possible. Artifact: `0222_matched_activations_v1.json`, with all boundary metrics and codec call counts. No production precision change was adopted.

## D — Retained state captures successfully analyzed

The three-line indentation repair in `scripts/probe_0223_state_evolution.py` is complete. Focused tests: **4 passed**. `--reanalyze` produced `0223_state_evolution_v2.json` from the existing captures: neither large model was re-executed. Pointer-verified V-projection/reshape filtering resolves the earlier naming ambiguity. All 35 sampled state comparisons have compatible shapes; all five sets of PLE row IDs match exactly.

Initial recurrence/convolution/PLE histories are exactly zero on both sides. Selected post-prefix relative-L2 differences are:

| Consumed prefix length | Layer0 recurrence | Layer0 convolution | Layer1 convolution | PLE normalized history |
|---|---:|---:|---:|---:|
|1 |0.00548179 |0.00314654 |0.00672520 |0.00154468 |
|2 |0.00540416 |0.00409740 |0.00814450 |0.00318735 |
|4 |0.00500166 |0.00415664 |0.01122788 |0.00384177 |

The layer0 recurrence discrepancy does not grow monotonically or explode over these prefixes. Layer3 current K/V/indexer cache-input vectors also differ; for the fourth input token their relative errors are 0.0328451 /0.0341913 /0.0211221. These are cache-input measurements, not a complete physical KV-buffer export, and their upstream activations already differ.

Logit relative-L2 errors for fixed prefixes `[1]`, `[1,2]`, `[1,2,3,4]` are 0.0616434, 0.0506754, 0.0410662. Local fresh-state prefill versus incremental execution remains bit-exact for logits and the sampled matrix/conv/PLE states. None of this converts a different-arithmetic state comparison into an equality proof. Eight/sixteen-token validation and identical-input recurrence/cache-update isolation remain outstanding. See `state_validation.md`.

## E — Isolated identical-input MoE versus linked llama.cpp/GGML

New diagnostic: `scripts/probe_0224_isolated_moe.py`, linked graph driver `tests/fixtures/qwen4exp_moe_isolated.cpp`; artifact `0224_isolated_moe_v1.json`. Cases use retained **reference FFN inputs** at `(position,layer)` = `(0,0)`, `(1,0)`, `(0,1)`, `(0,2)`, `(0,17)`, `(0,45)`. No full model is loaded. Only ten selected expert bundles per case are staged; global 512-way routing is evaluated in full. Selected banks are reindexed locally without changing their values. Shapes/limits are explicit, and every compared path receives identical F32 inputs.

The linked driver follows the pinned qwen4exp MoE graph: BF16 router matmul → softmax → argsort/top10 → gather and selected-mass normalization with floor2^-14; actual `ggml_mul_mat_id` gate/up → `ggml_swiglu_split` → down; routing weights applied once; independently sigmoid-gated shared expert. It does not substitute a scalar dot transcription for GGML. Detailed vectors, max-absolute errors and relative metrics are retained for all stages.

All six cases have the same top10 **sets** in reference, local F32 routing and activation-matched routing. Slot order is recorded separately; expert outputs are aligned by expert identity. With matched router activation arithmetic, routing-logit errors are 7.52e-8–8.87e-8 relative L2; selected normalized-weight errors 1.44e-7–3.08e-7.

| Position / layer | F32 MoE final vs GGML | Activation-matched MoE final vs GGML | Combination with identical reference outputs/weights |
|---|---:|---:|---:|
|0 /0 |0.00540809 |6.93396e-6 |3.96092e-8 |
|1 /0 |0.00628931 |3.53244e-7 |9.71085e-8 |
|0 /1 |0.00929952 |3.78220e-7 |8.49574e-8 |
|0 /2 |0.01023072 |3.73278e-4 |7.47768e-8 |
|0 /17 |0.01060910 |3.64955e-7 |8.22314e-8 |
|0 /45 |0.01362544 |4.90304e-7 |8.23001e-8 |

These are conditional identical-input comparisons with the reference expert set; no selection difference is concealed because the sets agree in all cases. Production's ascending-expert combination is compared with reference slot-order combination, not assumed bitwise identical.

Teacher-forcing reference intermediates isolates each operation: SwiGLU on identical gate/up values differs by 3.78e-8–1.08e-7; down projections on identical reference SwiGLU with the required Q8 rounding by 3.97e-7–4.85e-7; shared down by 3.34e-7–4.92e-7. Scalar shared-gate/sigmoid and final weighting/combination also agree at F32 arithmetic scale. No missing/doubled router weighting, wrong projection orientation, wrong SwiGLU graph or omitted shared expert is demonstrated in these tested cases.

### Why the layer2 matched shared expert is still different

Routed gate/up/SwiGLU values differ only around 1e-6 relative; the routed down path is about4.18e-7. The larger final error comes from the **shared** down path: shared SwiGLU differs by1.17953e-6, but the subsequent Q8-rounded down result differs by5.50407e-4. Feeding exactly the reference shared SwiGLU reduces that down error to3.76204e-7. This localizes the amplification before the down matmul rather than demonstrating a wrong down equation.

## F — Controlled perturbations establish numerical amplification

Artifact `0225_rounding_sensitivity_v1.json`, script `scripts/probe_0225_rounding_sensitivity.py`. Both experiments retain the mathematical operations and use upstream codecs; no production precision change is adopted.

### Isolated shared-expert threshold

Interpolate from reference shared SwiGLU toward the tiny observed local discrepancy. At layer2, a relative input perturbation of **1.47270e-7** changes one decoded Q8-rounded element and increases down-output relative error from **3.76204e-7 to5.50407e-4**. Larger factors produce the same rounded input/output step. Thus ordinary upstream floating-point differences can cross a quantization threshold and create a much larger discontinuous output change. A different layer0 routed-expert experiment similarly changes rounded input/output at a finite perturbation threshold. Not every tested input is sensitive.

### Full-forward, one-time early perturbation

Use the existing diagnostic activation-matched model, unchanged weights, forced token IDs `[1,2]`. Inject once, only at the layer0 MoE input at position0, then let the same downstream graph and state evolve. The deterministic direction is the observed local-minus-reference FFN-input discrepancy, with both signs. The actual pre-injection input is asserted bit-identical to the retained baseline before applying the perturbation.

Perturbation relative L2 **1.31186561e-6**, maxabs **5.48362732e-6**. The chosen positive input changes one BF16-rounded element. The sweep distinguishes decoded Q8 scale changes from code-threshold evidence rather than treating every changed decoded value as a changed integer code.

| Sign | Logit position | Relative L2 versus unperturbed matched model | Max absolute logit change |
|---|---:|---:|---:|
|+ |0 |0.0411539 |0.6865921 |
|+ |1 |0.0230992 |0.4243503 |
|− |0 |0.0348462 |0.5346808 |
|− |1 |0.0633422 |1.1362445 |

The two trajectories always consume identical external token IDs, even if argmax/rankings change. They execute sequentially in one child with fresh states, not as concurrent resident models. The bounded run took40.32s, sampled peak RSS9,435,844,608 bytes.

**Confirmed:** perturbations at ordinary F32-error scale can cause final-logit changes at the observed several-percent scale in this quantized pipeline. **Not confirmed:** that this particular intervention accounts for every baseline-reference difference, or that all later operations are correct. It is a sensitivity witness, not a variance decomposition or an acceptance tolerance.

## Earliest significant unexplained divergence — revised interpretation

The first HC read is explained. For localization only (not as an acceptance threshold), the earliest activation-matched residual error above1e-4 is position0/layer1 post-MoE (5.48269e-4), or position1/layer0 post-MoE (5.00979e-4). Their local/reference FFN inputs are already different. On identical reference inputs those MoEs agree to3.78e-7 and3.53e-7, respectively. Therefore those end-to-end boundaries do **not** identify a demonstrated MoE graph error.

At position0/layer2, the isolated matched shared-down discrepancy is causally explained by an upstream tiny SwiGLU perturbation and one Q8 rounding threshold, as measured above. Later layer45's large accumulated residual jump also does not indict its MoE operation: its identical-input matched MoE final error is4.90e-7. No earliest **proven incorrect mathematical operation** has been found.

The remaining unexplained component is attribution of the complete propagated state/logit difference across all mixers and positions—not evidence that the measured F32 accumulation error by itself is several percent. Plain F32 accumulation, BF16/Q8 representation changes, nonlinear/quantization thresholds, routing and recurrent feedback must remain distinguished.

## G — Identical-input DeltaNet/QSA and continuous state, 16 tokens

The isolated CPU reference executes the actual checkpoint decoder prefix layers 0–3 for fixed IDs 1–16, stopping after QSA layer3. Later decoder layers cannot feed back into this prefix; no unfinished full-model logits are read. Diagnostic files: `tests/fixtures/qwen4exp_block_state_reference.cpp`, `scripts/probe_0226_identical_block_states.py`, `tests/test_0226_identical_block_states.py`.

Replay separates projection arithmetic from state equations using captured operands. Both modes pass **236 comparisons**: teacher-forced identical reference pre-state at each step (`0226_identical_block_states_v2.json`), and continuous local state from zero without per-step resets (`0226_continuous_block_states_v1.json`). The latter artifact's explicit `state_policy` identifies continuous execution; its older general scope wording must not be interpreted as per-step state resets. The initial diagnostic adapter failure remains preserved in v1; v2 reused captures rather than rerunning the reference.

| Boundary | Maximum relative L2, identical pre-state | Maximum relative L2, continuous state |
|---|---:|---:|
| DeltaNet matrix state |5.68496e-7 |8.57284e-7 |
| DeltaNet normalized/gated output |2.04524e-7 |2.21826e-7 |
| QSA gated attention output |2.78174e-7 |3.26062e-7 |
| QSA K update |1.08275e-7 |1.08275e-7 |
| Completed indexer pools |8.46406e-8 |8.46406e-8 |
| Indexer scores |1.45355e-7 |1.82212e-7 |
| GDN output projection, identical input |1.32276e-6 |1.32276e-6 |
| QSA output projection, identical input |1.99285e-6 |1.99285e-6 |

Convolution shifts, raw indexer keys and V copies match exactly where expected. Selected **causal** cells agree at every position. Reference padded cache includes masked physical cells whereas local storage uses contiguous text cells; these are not identical physical cache layouts. At 16 tokens every causal cell fits the default 2051 selection width. Synthetic tests separately cover sparse selection beyond that width; this experiment does not independently certify long-context or multimodal QSA.

## H — Full-model short-sequence gate and initial profiling

`0227_short_sequences_profile_v1.json` records two sequential CuPy processes consuming the same IDs 1–16, with max_length16, four native CPU expert workers, FP32 compute/state, bounded backbone placement, no GPU expert cache, 120-second child deadlines and sampled 12GiB RSS guards.

At prefixes **1,2,4,8,16**, incremental and chunked-prefill logits are bit-exact and SHA256 hashes of **all active** recurrent, convolution, PLE history, KV, raw-indexer and completed pooled-index state match. Unwritten cache capacity is excluded. This validates whole-model local state consistency, not independent full-model reference logits over 16 tokens.

Incremental/prefill process times are67.99/65.30s; sampled peak RSS10,369,241,088/10,391,388,160 bytes. GPU pool snapshots are4,414,818,816 used /4,420,900,352 reserved bytes, not global VRAM high-water. Each process executes7,680 native CPU experts with zero decoded-expert cache and zero GPU expert residents.

The first instrumented profile, incremental positions8–15, measures mean token latency3.6965s. Routed expert dispatch/compute/transfers account for1.55152s host span/token; complete GDN including projections1.36484s; complete QSA including projections/indexing0.55624s. Spans include waiting and can nest; they are not an exclusive critical-path decomposition. See `0212_initial_profile.md`. No production speedup has been implemented.

## Validation and recommended next steps

Earlier focused command (historical verified state):

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0218_reference_identity.py tests/test_0219_activation_arithmetic.py \
  tests/test_0220_layer_evolution.py tests/test_0222_matched_activations.py \
  tests/test_0223_state_evolution.py tests/test_0224_isolated_moe.py tests/test_0106*.py
```

**1,295 passed,176 skipped in10.86s.** This includes existing Qwen3-Next/inference regressions and new linked-GGML diagnostics; optional GPU/benchmark cases skip as declared. The focused state tests pass4/4, isolated MoE tests2/2, retained state reanalysis completes, and both bounded perturbation trajectories execute. No production source file changed; original reference mathematics/source and weight files remain untouched. No performance optimization was implemented.

Latest consolidated regression command:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -m pytest -q \
  tests/test_020*.py tests/test_021*.py tests/test_022*.py tests/test_0106*.py
```

**1,492 passed,186 skipped in16.75s** on the final regression rerun (earlier successful run:16.54s). `git diff --check` also passes; `git diff -- mini_llm` is empty. These include the new diagnostic capture/layout and active-state hashing tests. The separate real-checkpoint replays and CuPy short-sequence runs above also passed. No production source or reference mathematical implementation changed.

**Recommendation at the validation/performance handoff:** proceed with measured performance work while preserving these numerical oracles. First separate GDN projection fallback cost from recurrence/launch overhead, and implement independently validated packed GPU expert execution for the checkpoint's mixed formats. Keep bounded source ownership and CPU fallback; retain FP32 production arithmetic rather than forcing Q8/BF16 merely to chase reference bits.

**Current assessment:** verified HC/MoE operators and representative DeltaNet/QSA state updates are mathematically consistent with the reference within measured arithmetic error. BF16/Q8 representation differences and quantization-threshold amplification are established. The baseline 5–6% full-model reference logit difference is not attributed element-by-element, and full independent 16-token logits/all-layer states or long-context equivalence remain unproven. These are explicit limits, not a demonstrated graph defect. The requested short-text gate is sufficient to begin profiling and controlled optimization, not to claim complete mathematical equivalence.

## I — Performance execution changes retain the validated graph

Patch0213 adds exact-format packed GPU GEMV/SwiGLU/down, grouped expert execution, shared CPU-miss staging, and explicitly switchable FP32 QSA/HC/router/GDN fusion. These are implementation changes, not corrections to model equations. The pinned independent reference and checkpoint remain untouched. Historical statements above that no production source changed refer to the diagnostic phase before this patch.

Real checkpoint packed expert comparisons cover all48 layers/all7 format triples: worst relativeL2=1.6228204350673205e-6, maxabs5.587935447692871e-9 against identical-input native CPU arithmetic. Grouped top10 outputs match sequential GPU bit-for-bit across all7 triples. Retained0226 checkpoint operands also exercise the new sigmoid-gated DeltaNet path independently; continuous-state tests remain active.

The final optimized full-model path versus the previous FP32 production baseline has relativeL2 errors1.441610e-6 /1.252346e-6 /1.521301e-6 /2.065315e-6 /1.788389e-6 at fixed prefixes1/2/4/8/16. Maximum absolute difference over these checks is2.789498e-5; argmax and top10 sets match. These small **optimization-induced** errors must not be confused with the pre-existing5–6% GGML-reference discrepancy, which is not reclassified as resolved by this result.

`0233_fast_lifecycle_v1.json` verifies optimized16-token incremental/prefill logits bit-for-bit and SHA256 hashes of all active persistent state, followed by two fresh greedy requests both returning `[220,100561,198]`. No state contamination or newly demonstrated graph defect was exposed. The new QSA shortcut is enabled only when selection must retain every causal cell; wider contexts and diagnostics retain the reference path.

Latest regressions: **1650 passed,654 skipped** in the NumPy inference/diagnostic suite; selected actual CuPy suite **894 passed,20 skipped,2 deselected**. The two exclusions are previously documented CPU-only NumPy-input assumptions, not a claim that every GPU test passes; see implementation_status.md. Latest memory/ownership fixes address demonstrated resident-budget double-counting and completed-fence accumulation, not mathematical discrepancies.

Measured short forced-token rate rises0.272867→8.566126tokens/s (~31.39×).70–80tokens/s remains unmet. Further bounded optimization should target expert transfer/admission and dispatch/launch overhead, with no route-tail skipping, Q8 activation approximation or speculative arithmetic substitution. Long-context and full independent model-equivalence limitations above remain in force.

## J — Follow-ups0234–0238 retain the numerical gate

Read-only global codebooks change no output bits. Grouping newly admitted complete bundles likewise preserves sequential GPU output exactly. Warp-per-row scheduling changes only FP32 within-row summation: real top10 across all7 format triples has worst relativeL2=3.176314e-7 and maxabs3.492460e-10 versus the retained block kernels (`0237_iq_warp_scheduling_v1.json`). No activation quantization or expert dropping is added.

Static HC graph capture initially encountered the installed CuPy cuBLAS capture restriction. A localized correction reuses the repository's graph-capturable FP32 GEMV;53 focused GPU tests pass. It does not demonstrate an end-to-end speedup, remains disabled, and captures no positions, QSA selection or expert placement.

The full-model warp and synchronous fixed-slot paths match each other **bit-for-bit** for logits and all active-state hashes at prefixes1/2/4/8/16. Versus the original FP32 production baseline, their largest relativeL2 is2.014602e-6 and maxabs2.717972e-5; all checked argmax/top10 sets match. The optimized fixed-slot lifecycle (`0233_fast_lifecycle_v2.json`) additionally passes16-token incremental/prefill/logit/all-state equality and repeats greedy IDs `[220,100561,198]` on fresh requests.

Fixed slots change storage ownership only. Tests verify exact checkpoint bytes, retained projection-view leases, stable addresses, synchronous completion before publication, CPU fallback on exhausted leases and quarantine on unknown completion. Initial pinned-pointer size API misuse was locally corrected and the focused GPU rerun passes32 tests; no model mathematical correction was indicated.

Latest CPU inference/diagnostic/selected-training regression: **1691 passed,780 skipped in17.89s**. Selected actual CuPy regression: **1031 passed,22 skipped,2 previously documented CPU-only cases deselected in8.21s**. This is not a claim that every repository GPU test passes.

Best short forced-token throughput is12.033481tokens/s (11.976041 with fixed slots), ~44× the original baseline.70–80tokens/s remains unmet. Next work is bounded asynchronous staging/admission and realistic longer cache-pressure benchmarking; it must preserve these oracles. None of these optimization comparisons establishes independent full-model GGML equivalence or removes the previously documented5–6% reference-logit attribution uncertainty.
