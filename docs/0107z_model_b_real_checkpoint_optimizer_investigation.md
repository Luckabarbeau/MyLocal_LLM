# Model B: real-checkpoint optimizer / gradient-norm investigation

## A. Conclusion

**The step-16,200 anomaly is reproduced exactly, and the causal optimizer group is Muon.** Increasing the Muon-managed displacement makes the actual trained model overshoot the favorable region along its realized update direction on the fixed next accumulated training batch. Changing AdamW LR, or freezing the tied embedding/unembedding, does not remove the effect.

This is now a measured result on trained Model B, **not an inference from synthetic initialization or generic learning-rate sensitivity**. Local-attention and FFN displacements are the principal identified contributors; compressed historical attention alone produces a much smaller response in the component ablation.

**No production implementation defect was demonstrated, and no optimizer/trainer patch was made.** The controlled update, placement, normalization, and backward checks support the correctness of the tested path. They do not constitute a proof that every possible execution/configuration is correct.

## B. Production identity and experiment isolation

Repository: `/home/lucka/Documents/my_local_LLM/MyLocal_LLM`.

Interpreter: repository `.venv/bin/python`, CuPy backend, NVIDIA RTX 5080. The GPU was made available by the user; no active production trainer was interrupted.

Checkpoint: `checkpoints/model_b_batch1_8k_norecompute_long_16200`.

- Trainer step: **16200**; saved optimizer counter: **16250**.
- Supervised-token total: **25,304,476,274**.
- Full parameter inventory: **348,261,376**, 302 arrays.
- Muon: 220 arrays / 314,572,800 elements.
- AdamW: 82 arrays, including the single tied input/output `embedding.W`.
- BF16 compute parameters, FP32 gradients and optimizer masters, full CPU optimizer offload.
- Context 8192, batch 1, **1024 accumulation microbatches**, supervised-token weighting, clip 1.0, no activation checkpointing.
- Saved optimizer state, Adam birth steps, train RNG, source shard cursors, original mixture, masks, scheduler horizon, and runtime environment restored by the **native production CLI**.
- The diagnostic's current resume contract equalled the saved contract: **no field differences**.

The counter offset was preserved, as production does (`train_model.py:716–729`). Exact master/compute parity and the reproduced logged first gradient show no tensor inconsistency in this experiment. This does **not** establish the historical origin of the offset. At these optimizer ages Adam bias-correction powers are negligible; the offset does not explain the measured factor-of-four response.

Verified SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| Checkpoint `config.json` | `de1daf350233a20ffe36b96f6b1d80a69781580782a5057b4a15fa194b679947` |
| Checkpoint and dataset tokenizer | `20781a3078c60fd5a7634fba801b0a76943e5f993f0b1eaa61b0a4238922f2ba` |
| Checkpoint and dataset mixture manifest | `7cc51eefc76516358dd8cacb717b3bbb0e9b8d8ac0cfaaceaf4cc7ccecb34085` |

**Environment detail:** the checkpoint saves `MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=0`, whereas the short `scripts/0105_model_b_profile.env` sets 1. The GPU diagnostics restored the checkpoint environment before backend imports, rather than substituting that profile.

The original CSV was located as `logs/model_b_batch1_8k_norecompute_test.csv`. Restart branches were separated by LR and row order, not collapsed by step. All 58 paired steps 16201–16258 have matching executed/supervised-token counts; 57 post-update losses/norms differ. Historical counts alone are not proof of identical samples; see batch identity below.

The earlier CPU pilot was not used as production evidence: it used FP32 parameters, one shard, and reduced accumulation. Production GPU scripts instead intercept `ExtendedTrainer.train` **after the entire native resume sequence**, omit checkpoint/log destinations, and never call trainer save. Lazy document-index sidecar writes are disabled in memory. No production sources, checkpoint files, logs, or benchmark archives were overwritten.

## C. Exact numerical reproduction and optimizer-group ablation

Actual LRs for the first resumed update:

- Low: **7.104382866412835e-5**.
- High: **2.841753146565134e-4**.
- Ratio: **4**.

The first accumulation was captured once at the production optimizer boundary. It was already token-normalized, with clip scale exactly **1.0**. Each update branch restored the original BF16 compute weights, FP32 masters, moments, momentum, counters, and birth steps, then used that same gradient. Every post-update probe replayed the **entire same next 1024-microbatch accumulation**, not a synthetic or single-example probe.

First accumulation, before any update:

| Quantity | Reproduced | Original CSV |
|---|---:|---:|
| Loss | 2.2153923511505127 | 2.2153923511505127 |
| Preclip global norm | 0.30366477370262146 | 0.30366477370262146 |
| Supervised tokens | 3,400,191 | 3,400,191 |

Fixed next batch: **3,244,627 supervised tokens**.

| Branch | Muon LR | AdamW LR | Probe CE | Probe preclip norm |
|---|---:|---:|---:|---:|
| No update | — | — | 2.252253532 | 0.313699991 |
| Both low | low | low | **2.251154184** | **0.172673687** |
| Both high | high | high | **2.259387493** | **0.725413024** |
| Muon high / Adam low | high | low | 2.259134769 | 0.718762755 |
| Muon low / Adam high | low | high | 2.251184225 | 0.172742769 |
| Both high, embedding low | high | high, except embedding low | 2.259192467 | 0.720857203 |
| Both high, embedding frozen | high | high, except embedding frozen | 2.259135008 | 0.719833255 |

The native both-high and both-low outcomes match the original step-16202 CSV values exactly. The no-update control also establishes that the next data batch alone does not cause the jump to 0.725.

### Batch equivalence

Each replay hashes actual input tokens, targets, source identity, and metadata **including target-loss masks**, shapes and dtypes for all 1024 microbatches. All next-batch branches share aggregate identity:

`b1deba2b1be5bb59…` (full digests and per-microbatch identities in `report.json`).

The initial/repeated first accumulation shares `8537ae6e1b5d908b…`.

This verifies exact data/mask equivalence **inside the controlled experiment**, not merely equal token counts. The checkpoint's original local-AR `sample_trace_sha256` is null, so original historical sample-ID/mask hashes cannot be independently compared. Exact reproduction of historical loss and norm strongly corroborates the replay but does not replace those missing original hashes.

### Pre-update LR independence

A second full first-batch pass at low LR, after checkpoint restoration, produced exactly the same loss and logged global norm as high LR. **220/220 Muon gradient tensors hash identically.** Of AdamW tensors, 40/82 hash identically; the remaining 42 are embedding/norm parameters, with a maximum difference in per-tensor norm of approximately **3.92e-10**. Bitwise equality of every tensor is therefore not claimed. Their execution-level discrepancy was not isolated; it is consistent in scale with reduction-order roundoff and is not evidence of LR-dependent gradient scaling.

## D. Real parameter displacement and component measurements

All measurements below use trained checkpoint weights and restored state—not initialization-scale estimates. Component RMS values are weighted by element count. Full before/after RMS, FP32 and BF16 displacement, gradient RMS, relative update and direction comparisons are recorded for **every one of the 302 arrays** in each branch's `*_updates.csv`. Per-parameter probe gradients and per-layer norms are in `report.json`.

### Actual master-weight displacement

| Quantity | High | Low |
|---|---:|---:|
| Global `||ΔW_master||` | 0.960734512 | 0.240183602 |
| Ratio high/low | approximately 4 | — |

High-branch group displacement norms: Muon **0.890639942**, AdamW **0.360237834**. Muon contributes **85.94%** of total squared master displacement.

Per-Muon-array normalized update directions agree: minimum high/low delta cosine **0.999999973**; maximum relative error against quarter-high delta **2.32e-4**. Across all parameters the maximum relative scaling error is **0.00602**, on a norm parameter with a tiny displacement against an order-one FP32 weight. These finite-precision displacement differences are reported, not treated as exact proportionality or as a demonstrated implementation error.

### Trained RMS and actual update magnitude

| Component | Master RMS before | Master RMS after high | High master Δ RMS | Low master Δ RMS | High relative master update |
|---|---:|---:|---:|---:|---:|
| Tied embedding/unembedding | 0.04114406 | 0.04114491 | 6.21280e-5 | 1.55320e-5 | 0.001510 |
| Local attention QKV | 0.02774945 | 0.02774931 | 4.92945e-5 | 1.23236e-5 | 0.001776 |
| Local attention output | 0.02432437 | 0.02432451 | 4.94181e-5 | 1.23545e-5 | 0.002032 |
| Compressed global attention | 0.02754002 | 0.02754108 | 5.25258e-5 | 1.31315e-5 | 0.001907 |
| FFN gate/up | 0.02839911 | 0.02839906 | 5.00087e-5 | 1.25022e-5 | 0.001761 |
| FFN down | 0.02562671 | 0.02562694 | 5.08702e-5 | 1.27175e-5 | 0.001985 |
| Matrix-pool score | 0.05443355 | 0.05443664 | 4.37203e-5 | 1.09301e-5 | 0.000803 |
| Matrix-pool position bias | 0.05742610 | 0.05742988 | 6.13810e-5 | 1.53452e-5 | 0.001069 |
| Block norms | 0.99361098 | 0.99361137 | 3.72493e-5 | 9.31247e-6 | 0.0000375 |
| Final norm | 1.9583869 | 1.9584220 | 5.22354e-5 | 1.30586e-5 | 0.0000267 |

The earlier synthetic suggestion that compressed-global weights had a uniquely large relative update due to initialization scale is **not supported by these trained weights**. Compressed-global and local/FFN relative update scales here are comparable.

### Which gradients increase?

| Component | Before update, first batch | Next batch, no update | Next batch, low update | Next batch, high update |
|---|---:|---:|---:|---:|
| Tied embedding/unembedding | 0.0473381 | 0.0464607 | 0.0311063 | 0.1082022 |
| Local attention QKV | 0.1675377 | 0.1728191 | 0.0931156 | 0.4180865 |
| Local attention output | 0.1843146 | 0.1904378 | 0.1022497 | 0.4184657 |
| Compressed global attention | 0.0760022 | 0.0802321 | 0.0348608 | 0.2018033 |
| FFN gate/up | 0.1266831 | 0.1313278 | 0.0817743 | 0.2935125 |
| FFN down | 0.0774994 | 0.0795455 | 0.0424042 | 0.1923870 |
| Matrix-pool score | 0.0041916 | 0.0046835 | 0.0028065 | 0.0099512 |
| Block norms | 0.0079811 | 0.0081202 | 0.0038094 | 0.0213625 |
| Final norm | 0.0042776 | 0.0040746 | 0.0005224 | 0.0128159 |

These norms combine in quadrature, not by summation. Post-high Muon/AdamW group norms are **0.7167962 / 0.1114781**; post-low they are **0.1697820 / 0.0314685**. `blocks.0.attention.Wo` is the largest individual high-branch gradient contributor, norm **0.3296950**; this does not by itself prove that its update alone causes the jump.

### Finer displacement ablation

Use a quarter of the **same realized high-update delta** everywhere, except the selected component uses the full delta:

| Full-high component, rest quarter-high | Probe CE | Probe norm |
|---|---:|---:|
| None, all quarter-high | 2.251169682 | 0.172807083 |
| Local-attention projections | 2.252314091 | 0.328875721 |
| Compressed-global projections | 2.251162052 | 0.175018623 |
| FFN matrices | 2.253726006 | 0.399431884 |

Local-attention and FFN displacements individually produce substantial increases. The compressed-global-only contribution is much smaller in this test. This is a **fixed-displacement ablation**, including the original decay contribution, not another exact native component-LR replay; it does not measure a unique local×FFN interaction or isolate one matrix. The stored `analytic_dloss_dalpha` field for these three rows projects onto the **entire high delta**, not the selected component, and must not be interpreted as a component-specific derivative.

## E. Mechanism: demonstrated overshoot along a fixed real update

Let `Δ_high = W_master_after_high − W_master_checkpoint`, captured from the actual native high-LR optimizer step. Evaluate **the same next accumulation** at

`W_compute(α) = BF16(W_master_checkpoint + α Δ_high)`.

The direction, initial weights, batch data, masks, and token weighting are fixed. Optimizer momentum/direction is not recomputed between α probes.

| α | Probe CE | Global preclip norm | Backward projection `g · Δ_high` |
|---:|---:|---:|---:|
| 0 | 2.252253532 | 0.313699991 | -0.00821928 |
| 0.125 | 2.251447916 | 0.222348347 | -0.00431483 |
| 0.25 | **2.251169682** | **0.172807083** | -0.00046086 |
| 0.5 | 2.251980543 | 0.278186589 | +0.00714718 |
| 0.75 | 2.254763842 | 0.491096735 | +0.01474509 |
| 1 | 2.259387970 | 0.725403905 | +0.02231717 |

**The sampled minimum is at α=0.25; the high update α=1 overshoots it.** Even loss-only measurements, independently of backward norms, establish this response. A quadratic fit to sampled losses is approximately

`L(α) ≈ 2.25224258 − 0.00814167 α + 0.01529646 α²`,

with maximum residual **2.33e-5** and an interpolated smooth minimum near **α=0.266**. This is an interpolation, not a precisely measured optimum of a BF16-quantized function or a universal LR recommendation.

The backward directional projections agree with the broad loss trend and change sign near the favorable region. They are **smooth/straight-through directional proxies**, not literal derivatives through BF16 rounding. The experiment does not partition the contributions of smooth curvature and quantization to every fine-scale fluctuation.

Quarter-high reconstruction is not bit-identical to a native low-LR step because of FP32 subtract/add rounding and BF16 thresholds. That explains why its norm is 0.172807 rather than native low's 0.172674; similarly reconstructed α=1 differs slightly from the original high result. Neither state is silently called identical to the native branch.

Muon orthogonalizes its momentum direction and therefore does not use raw gradient magnitude as a direct per-element update scale. The source's shape-adjusted update direction is multiplied by LR once. Here the larger valid Muon displacement moves beyond the favorable region and produces a much larger subsequent gradient; lowering LR moves close to that region and reduces the gradient. This accounts for the immediate response without inventing a direct LR multiplier on raw gradients.

## F. Optimizer, offload, synchronization, and backward correctness

### Source-level evidence

| Path | Relevant implementation |
|---|---|
| `mini_llm/train_extended.py:2877–2921` | LR assignment is separate from data/loss; supervised-token microbatch weighting |
| `mini_llm/train_extended.py:3181` | Single scaled entry to chunked-head backward |
| `mini_llm/train_extended.py:3272–3285` | Undo loss scale, then divide accumulated gradients by total weight |
| `mini_llm/train_extended.py:3322–3326` | Global clip reports preclip norm |
| `mini_llm/train_extended.py:3379–3407` | Existing component monitor, optimizer step, compute-buffer refresh, zero gradients |
| `mini_llm/train_extended.py:3526–3535` | Returned/logged gradient norm is pre-update/preclip |
| `mini_llm/optim/hybrid_muon.py:175–225` | Muon momentum, Nesterov, NS direction, rectangular scaling, decay, one LR on direction, offload synchronization |
| `mini_llm/optim/hybrid_muon.py:289–307,348–383` | Hybrid groups, state restoration and counters |
| `mini_llm/optim/adamw.py:367–374,401–527` | Bias correction and device/streamed-offload paths |
| `train_model.py:1803–1880,2377–2399,2488–2490` | Native parameter/state/birth-step/RNG restoration |
| `mini_llm/heterogeneous_production.py:38–55` | Existing sampler snapshot/restore, reused without starting a peer |
| `mini_llm/diagnostics/component_updates.py:109–227` | Existing read-only update monitor, reused for all native ablation branches |
| `mini_llm/blocks/transformer_block.py:204–249,503–534` | Residual/local/pooling/global dependencies and reverse graph |

The intended Muon equations are

`m' = 0.95 m + 0.05 g`,

`n = 0.95 m' + 0.05 g`,

`D = 0.2 sqrt(max(rows, cols)) NS_5(n)`,

`W' = (1 − η wd) W − η D` for decayed parameters.

LR enters the direction displacement once and separately enters the intended decoupled decay. It does not enter momentum or Newton–Schulz. The independent FP32 reference tests cover square and rectangular matrices; real-checkpoint delta scaling supports this contract on the trained production path. AdamW's scaled bias-correction/epsilon form is algebraically the intended update.

### Exact full-optimizer placement parity on trained state

Construct an all-device optimizer with **all 302 real parameters**, the same saved master weights/moments/Muon state/Adam birth steps/counters, and the same full accumulated gradient. Compare one high-LR update against production full CPU offload.

| Comparison | Result |
|---|---:|
| Maximum FP32 master absolute difference | **0** |
| Maximum Adam/Muon first-moment difference | **0** |
| Maximum Adam second-moment difference | **0** |
| BF16 compute bit mismatches | **0** |
| Compute vs `BF16(master)` mismatches | **0** |

Every factorial branch also passed exact compute/master parity after update. Packed compute buffers were refreshed before the next forward pass.

Placement parity uses the same native equations in different placement paths; it is not an independent mathematical reference. Combined with independent equation tests and trained-state scaling, it rules out the suspected placement/state-copy mechanism for this reproduced update.

BF16 compute displacement need not scale fourfold even when masters do. For example, local QKV compute-delta RMS is **7.28809e-5 high / 3.45247e-5 low**, versus master **4.92945e-5 / 1.23236e-5**. This is consistent with exact BF16 rounding of each master, not evidence of a stale copy. A complete production FP32-vs-BF16 counterfactual was not run, so quantization's separate causal share is not established.

### Trained-block directional derivative checks

Real block-entry activations were exported from a production-sampled unpadded 256-token prefix. Trained FP32 master parameters from layers **0, 10, 19** were loaded into a scoped smooth FP64 reference, retaining the existing block equations and both residual dependencies. Existing hard-coded FP32 dtype sites were promoted **in memory only**; production sources were not changed.

Targets: block input, history-only input direction, local `Wo/Wv`, global `Wo/Wv`, pool score/position, FFN gate/up/down. Gradient-aligned and independent random unit-L2 directions were tested over ε=`1e-3,1e-4,1e-5,1e-6,1e-7`.

- **66/66 target/direction checks passed** the predefined criterion of two adjacent passing epsilons.
- **328/330 individual epsilon checks passed**.
- All 66 passed at each of ε=`1e-4,1e-5,1e-6`.
- Tolerance: absolute error ≤ `1e-8 + 3e-5 max(|FD|, |analytic|)`.
- At ε=`1e-6`, maximum absolute discrepancy: **3.59e-9**.

Two isolated out-of-tolerance epsilon points were investigated numerically, without editing production code:

1. Layer 19 pool score, gradient-aligned: analytic **1.9131452609**, FD at ε=`1e-3` **1.9113581883**, absolute error **1.787e-3**, symmetric relative error **4.673e-4**, requested absolute threshold **5.740e-5**. Error fell approximately quadratically to **1.788e-5**, **1.786e-7**, **3.588e-9** as ε decreased, then rose to **2.313e-8** at `1e-7`. This supports finite-difference truncation followed by roundoff, not an incorrect derivative.
2. Layer 19 pool-position random direction at ε=`1e-7`: analytic **2.2677486691e-4**, FD **2.2675195055e-4**, absolute error **2.292e-8**, symmetric relative error **5.053e-5**, threshold **1.680e-8**. All four larger epsilons passed; error at `1e-3` was **3.581e-12**. This supports cancellation/roundoff.

These checks validate tested smooth reference equations at real trained states, **not an exhaustive end-to-end 8K BF16 derivative proof**. GPU reference-versus-Triton pooling/attention backward tests complement them.

### Tests executed

GPU command:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -m pytest -q \
  tests/test_model_b_training_correctness_audit.py::test_cupy_bf16_full_offload_matches_device_optimizer \
  tests/test_matrix_pool_dual_resolution_0102o.py::test_production_width_triton_mixed_window_and_global_parity \
  tests/test_matrix_pool_fused_backward_0102u.py::test_fused_backward_cuda_small_parity \
  tests/test_matrix_pool_fused_backward_0102u.py::test_fused_backward_cuda_production_width
```

**6 passed** (parameterized cases).

CPU audit command:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -m pytest -q \
  tests/test_hybrid_muon.py tests/test_model_b_training_correctness_audit.py \
  tests/test_gradient_accumulation.py tests/test_gradient_norm_audit.py \
  tests/test_model_b_resume_audit.py
```

**26 passed, 2 GPU-only cases skipped.** One intentional FP32-overflow warning from the norm-overflow audit test.

The first GPU harness run failed before optimizer replay because `.set()` cannot target a noncontiguous gradient buffer. One localized diagnostic-only correction used stride-aware assignment. The complete rerun, including every branch, passed. This was a harness-copy error, not a production defect.

## G. Fix, training implications, confidence and limits

### Proposed correction

**No production correctness patch is warranted by the evidence.** Do not change accumulation normalization, Muon LR application, momentum restoration, or offload synchronization to hide this response: those tested paths behave consistently with their contracts.

For this checkpoint/batch, a smaller **Muon** displacement avoids the demonstrated overshoot. The low-Muon/high-Adam branch shows that lowering AdamW or the embedding LR is not necessary to remove this particular jump. This identifies the relevant control; it is not a request to change the running trainer, nor evidence for one globally optimal LR. Production configuration was left unchanged.

### Existing checkpoints

No evidence here establishes corrupted optimizer state or mathematically invalid prior updates. Existing weights can remain valid continuation candidates; this experiment creates no correctness-based requirement to roll back. High-Muon updates are nevertheless demonstrably worse on this fixed probe, and the user's longer matched-loss/held-out results separately favor the low-LR continuation. A rollback or checkpoint-selection decision remains an optimization-quality/held-out-evaluation decision, not a repaired-state migration.

No claim is made that lower LR universally improves all capabilities, and benchmark ZIPs were not re-evaluated in this GPU phase. The step-9600 experiment was not independently replayed here.

### Confidence

**High confidence:** real checkpoint/environment restored; original anomaly exactly reproduced; Muon LR is causal; tied embedding/AdamW are not the main cause; data/masks match within replays; clip is inactive; actual master updates scale as intended; full offload/device parity and master/BF16 synchronization pass; actual fixed-direction losses demonstrate overshoot.

**Supported but bounded:** tested smooth backward graph is correct at selected trained blocks; GPU kernels agree with references on the tested configurations; local-attention and FFN displacements are the largest identified component contributors.

**Unresolved/not claimed:** exact historical sample hashes; origin of the historical +50 counter offset; universal optimizer correctness; one uniquely responsible matrix; separate quantization-vs-smooth-curvature contributions; long-run optimal Muon LR; exhaustive BF16 end-to-end derivative correctness.

## Reproduction and artifacts

From repository root, with the GPU free:

```sh
TRITON_DEFAULT_BACKEND=nvidia LD_LIBRARY_PATH= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -u scripts/probe_0107z_gpu_checkpoint_ablation.py --repeat-initial

TRITON_DEFAULT_BACKEND=nvidia LD_LIBRARY_PATH= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -u scripts/probe_0107z_gpu_update_validation.py
```

Each chooses a fresh timestamped output directory and never saves a production checkpoint.

Verified run artifacts:

- `logs/lr_gpu_16200_20261115_170243/report.json`: factorial ablation, full parameter/layer/component gradients, masks/data hashes, runtime/resume contract.
- Same directory, `*_updates.csv`: complete 302-array master/compute before/after measurements per branch.
- Same directory, `*_monitor.csv`: reused existing component update monitor.
- `logs/lr_gpu_16200_20261115_173614/validation_report.json`: full trained optimizer placement parity, fixed-direction sweep, component displacement ablations.
- Same directory, `block_inputs.npz`: actual trained block-entry residual prefixes.
- Same directory, `trained_block_gradcheck.json` and `.jsonl`: complete derivative values/error/tolerance/epsilon records.

Trained-block check command:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python scripts/probe_0107z_trained_block_gradcheck.py \
  --inputs logs/lr_gpu_16200_20261115_173614/block_inputs.npz \
  --contiguous-prefix --weights masters \
  --output logs/lr_gpu_16200_20261115_173614/trained_block_gradcheck.json
```

Use a fresh output filename when rerunning to preserve the verified report. GPU memory was released when the diagnostic processes exited.
