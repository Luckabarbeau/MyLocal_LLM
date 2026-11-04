# Development log

This is the single durable human/agent change log for MyLocal_LLM. It consolidates the numbered and unnumbered patch reports that were historically spread across the repository root, `docs/`, `docs/research/`, and `benchmarks/`.

The standalone reports were removed after consolidation. Use the migration ledger and `git show <commit>:<path>` to recover their full historical text.

## Current status

- **Protected architecture contract:** Model B is registry ID `model_b_348m_sparse16` / CLI preset `dual-resolution-matrix-pool-sparse-1024-20l`, with **348,261,376 parameters**, 20 layers, width 1024, FFN 4096, four matrix-pooled summaries per completed 128-token block, 512-token local attention, one 4096-token long-local head, and the stride-16 historical head across the established 16-layer phase schedule. This identity is encoded in current code/tests and must not be changed incidentally by refactoring. Its bounded CPU contracts passed during this cleanup; the full-size external run and GPU throughput were not validated here.
- **Checkpoint/run status:** an external **`step165`** artifact or run is **user-reported/unverified**. It is not present as verified repository evidence in this checkout; no checkpoint manifest, hashes, logs, loss, benchmark, or resume test was inspected here. Do not present it as validated until those artifacts are recorded.
- **Specialist status:** production-oriented specialist training code exists through 0107–0108, but the historical 0105 report explicitly left full-size GPU parity, 12-layer memory/performance, long-range quality, and composed generation open. Commit `eb61df0` closes several code and CPU-contract gaps; it does not by itself prove a successful external training run.
- **Historical reports:** consolidated and removed from the working tree; their filenames and Git provenance remain in the migration ledger. New durable changes go in this file, not in a new patch-report Markdown file.

## Mandatory logging policy

1. **Humans and coding agents must log every durable behavior, architecture, compatibility, data-policy, checkpoint, runtime-default, or operational-safety change here.** Pure formatting changes may be omitted.
2. Add entries **newest first**.
3. Every entry must include: **date, author, commit, status, subsystem, behavior, compatibility, validation, and remaining work**.
4. Use explicit status words: `proposed`, `experimental`, `accepted`, `rejected`, `superseded`, or `user-reported/unverified`. A feature can be accepted but remain opt-in; say so.
5. Separate repository-observed evidence from user reports. Record commands and results only when actually run; otherwise write `not run` or cite the historical report as a reported result.
6. Link durable evidence to code, tests, benchmark/result schemas, and commits. Do not make a deleted patch file the only evidence.
7. **Do not create a new patch-report Markdown file.** Update this log, current guides, code comments where necessary, and the relevant tests instead.
8. Never edit, move, overwrite, or benchmark against a live checkpoint/runtime directory without explicit approval and a verified path.

## Mandatory future-entry template

Copy this block directly below this section and fill every field:

```markdown
## YYYY-MM-DD — Short durable-change title

- **Author:** Human name or agent/model plus operator
- **Commit:** `<hash>` or `uncommitted (working tree)`
- **Status:** proposed | experimental | accepted | rejected | superseded | user-reported/unverified
- **Subsystem:** architecture / training / inference / serving / data / checkpoint / benchmark / documentation
- **Behavior:** What changed, including defaults and opt-in flags.
- **Compatibility:** Parameter/checkpoint/API/data/runtime impact; migration or rollback requirements.
- **Validation:** Exact commands and observed results, or `not run` with the reason. Label user-reported evidence.
- **Remaining work:** Open correctness, performance, quality, migration, or cleanup gates.
- **Evidence:** `path/to/code.py`, `path/to/test.py`, benchmark result paths/schemas, and relevant commits.
```

---

## 2026-11-04 — Second readability and duplication cleanup

- **Author:** GPT-5.6-Sol coding agent, at user request
- **Commit:** `uncommitted (working tree)`
- **Status:** accepted documentation/test-maintenance cleanup; high-risk runtime refactors deferred
- **Subsystem:** documentation, tests, benchmark maintenance, repository hygiene
- **Behavior:** Established explicit canonical ownership for the landing page, exact baseline, operational guide, developer APIs, checkpoints, data, tests, benchmarks, roadmap, and development history. Reduced `README.md` to a compact landing page and moved the full native/SmolLM2 GGUF/Safetensors/remote serving workflow into `docs/USER_GUIDE.md`; standardized local examples on `serving_models.local.json` and port 8000; defined foundation and specialist flag arrays once; expanded `docs/FUTURE_DEVELOPMENT.md` into a risk-ranked cleanup plan with affected symbols, benefits, gates, and prerequisites. Added all 0107–0108 specialist contracts to the permanent CPU runner, made the optional Safetensors test skip cleanly when its extra is absent, corrected stale benchmark/data/history instructions, and ignored local downloaded-model/registry state. Removed the unreferenced `tests/fixtures/0080l_cpu_reference.json` and two obsolete shell wrappers, `benchmarks/run_checkpoint_evaluations.sh` and `benchmarks/run_fusion_evaluations.sh`, which hard-coded absent historical checkpoints and duplicated maintained runners.
- **Compatibility:** No model mathematics, parameter names/order, training hot path, optimizer, sampler, checkpoint schema/loading, resume semantics, inference kernel, or serving implementation changed. The removed fixture had no consumers; the removed wrappers targeted retired experiments and had no repository references. Optional Safetensors coverage still runs when the dependency is installed and is an explicit skip otherwise. Existing checkpoints, including a future immutable copy of the protected Model B run, retain the same code compatibility boundary.
- **Validation:** CPU-only collection succeeded with **2,203 tests collected** and no optional-Safetensors import error. The expanded permanent CPU gate passed **282 tests with 4 expected skips** across ten isolated suites, including **30** 0107–0108 specialist tests. Focused cleanup/regression and historical benchmark contracts passed **49 tests with 5 expected skips**. All local Markdown links resolved, both serving example JSON files parsed, removed-file references were absent, and `git diff --check` passed. No GPU test, model download, checkpoint load, or training step was run.
- **Remaining work:** Follow the prioritized work in `docs/FUTURE_DEVELOPMENT.md`: extract benchmark/test helpers, add explicit item-level GPU opt-in, decide the orphan SFT workflow and old launcher policy, then consider medium-risk constant/preset/tokenizer/inference consolidation one change at a time. Trainer, checkpoint/resume, config/inference-model, generation-adapter, and process-global runtime refactors remain gated on copied-checkpoint and idle-GPU parity.
- **Evidence:** `README.md`, `docs/README.md`, `docs/CURRENT_BASELINE.md`, `docs/USER_GUIDE.md`, `docs/MODEL_CENTERED_DEVELOPER_GUIDE.md`, `docs/CHECKPOINT_COMPATIBILITY.md`, `docs/FUTURE_DEVELOPMENT.md`, `tests/README.md`, `scripts/run_0103a_regressions.py`, `tests/test_regression_runner_0103a.py`, `tests/test_0085c_external_safetensors.py`, `/tmp/model_b_cpu_cleanup.json`.

## 2026-11-04 — Restore pretrained-model loading and local serving guide

- **Author:** GPT-5.6-Sol coding agent, at user request
- **Commit:** `uncommitted (working tree)`
- **Status:** accepted documentation and example configuration
- **Subsystem:** model loading, serving, external models, documentation
- **Behavior:** Restored the useful end-to-end workflow from the former README using current interfaces and model names. The root README now documents compatible native, GGUF, Safetensors, and remote sources; an official SmolLM2 Safetensors download; GGUF inspection and direct generation; mixed registry validation; lazy local serving; completion/chat/streaming requests; IDE base-URL use; remote-provider proxying; and security/compatibility limits. Updated `serving_models.example.json` from the retired native-338M example to Model B plus SmolLM2 GGUF and Safetensors entries.
- **Compatibility:** Documentation and example-registry change only. No loader, tokenizer, model, checkpoint, inference kernel, server implementation, or training hot path changed. Registry paths remain placeholders and models are loaded lazily.
- **Validation:** The example registry parsed through `ModelCatalog` and resolved `native`, `gguf`, and `safetensors` formats. Serving/native-loading/GGUF/tokenizer contracts passed **48 tests**. `serve.py --help`, `benchmarks.try_external_gguf --help`, all local Markdown links, JSON parsing, and `git diff --check` passed. The Safetensors-specific test could not collect because this environment lacks the optional `safetensors` package; no weights were downloaded and no GPU/model execution was performed.
- **Remaining work:** Validate the documented SmolLM2 Safetensors and GGUF paths with immutable model files on an idle GPU, record exact model/tokenizer hashes, and verify generated prompt framing and output parity before enabling strict fast paths or tool envelopes.
- **Evidence:** `README.md`, `serving_models.example.json`, `serve.py`, `mini_llm/model_sources.py`, `mini_llm/model_loading.py`, `mini_llm/serving_api.py`, `benchmarks/try_external_gguf.py`, serving and external-loader tests under `tests/`.

## 2026-11-04 — Consolidate durable change history

- **Author:** GPT-5.6-Sol coding agent, at user request
- **Commit:** `uncommitted (working tree)`
- **Status:** accepted documentation policy and completed report consolidation
- **Subsystem:** documentation and project governance
- **Behavior:** Established this file as the sole future change log, consolidated historical status and negative results, added a deletion-safe migration ledger, and removed superseded standalone reports. Registered `tests/test_unified_benchmark_0103g.py` directly in the permanent CPU regression runner, then removed its one-shot registration helper. Removed unlinked generic repository-plan TeX/PDF/SVG assets that predated and did not describe current Model B. Cleared regenerable caches and editable-install metadata while preserving runtime data and the virtual environment.
- **Compatibility:** No model, optimizer, checkpoint, data, inference, or training hot-path code changed. The CPU regression gate gained one existing test suite; runtime/checkpoint/data directories were untouched.
- **Validation:** After direct 0103G registration, the bounded CPU runner passed **252 tests with 4 expected skips** across nine isolated suites. Focused checkpoint/profile tests passed 18 tests. The production audit matched all 36 environment values and 27 CLI options. All local Markdown links resolved, `git diff --check` passed, and the registry reported 348,261,376 parameters. No GPU test was run.
- **Remaining work:** Keep the ledger current and reject future standalone patch-report Markdown files; recover full historical reports from Git only when needed. GPU throughput and the unavailable external step-165 checkpoint remain unverified.
- **Evidence:** `docs/DEVELOPMENT_LOG.md`, `docs/README.md`, `docs/FUTURE_DEVELOPMENT.md`, `docs/research/README.md`, `scripts/run_0103a_regressions.py`, `tests/test_regression_runner_0103a.py`.

## 2026-11-03 — 0107–0108 production specialist hardening

- **Author:** Askael
- **Commit:** `eb61df0` (`specialist training is now up`)
- **Status:** accepted implementation; external run success remains user-reported/unverified
- **Subsystem:** specialist training, Model B compatibility, logging, checkpoint resume
- **Behavior:** Added fail-closed production provenance checks; required the sparse Model B foundation; derived specialists from exact foundation block geometry; supported positive arbitrary microbatch sizes with rowwise masks; added accumulation length regrouping, stage profiling, source-weighted CE, optional same-minibatch foundation CE, production-style logging, exact sparse-phase coverage, and prepared-RoPE reuse through checkpoint replay.
- **Compatibility:** The production launcher now requires CuPy and exact Model B geometry and rejects mismatched specialist IDs/layer counts on resume. Legacy tiny/reference specialist geometry remains for tests and old reference checkpoints. A 16-layer specialist maps exactly once across all 16 stride phases. Foundation parameters remain frozen. Repacking is target-equivalent but can change floating-point/order behavior; resumed sampler batch size must match.
- **Validation:** Commit includes CPU contract tests, but no tests were rerun for this documentation migration. The external `step165` claim has no inspected repository artifact and is **user-reported/unverified**.
- **Remaining work:** Verify the external run with immutable checkpoint/config/tokenizer hashes, logs and resume; run idle-GPU full Model B/specialist parity; measure 12-layer context-8192 VRAM and throughput; validate long-range quality and composed incremental generation.
- **Evidence:** `mini_llm/specialists/production_contract_0107.py`, `mini_llm/specialists/train_logging_0107c.py`, `mini_llm/specialists/residual_specialist.py`, `mini_llm/specialists/training_run.py`, `scripts/train_0105_specialist.py`, `tests/test_specialist_production_contract_0107a.py`, `tests/test_specialist_profile_0107b.py`, `tests/test_specialist_train_logging_0107c.py`, `tests/test_specialist_multibatch_0107d.py`, `tests/test_specialist_model_b_shape_0108.py`, `tests/test_specialist_prepared_rope_0108.py`.

## 2026-11-02 — 0106 external SmolLM2 retrieval control

- **Author:** Askael
- **Commit:** `68844dd`
- **Status:** accepted benchmark adapter; GPU/model result unverified
- **Subsystem:** external-model evaluation and long-context retrieval
- **Behavior:** Added a SmolLM2 GGUF runner for the same 0097 synthetic retrieval bank, with shared case/scoring definitions, explicit unsupported cases, own-tokenizer positions, optional counterfactuals, and chunked external scoring coverage.
- **Compatibility:** Does not modify Model B or extend the external model's native context. Prompts beyond native context are skipped, never silently truncated. Full-causal external prefill can have materially different memory complexity from Model B, so throughput is not directly comparable.
- **Validation:** Historical report says CPU contract tests were used; GPU GGUF inference was not run during development. No tests were rerun here.
- **Remaining work:** Run on a free GPU, record exact GGUF/tokenizer hashes, native context, unsupported cases, confusion matrix and answer bias; do not infer 16K–32K capability from an 8K-native model.
- **Evidence:** `benchmarks/evaluate_external_long_retrieval.py`, `tests/test_0106a_external_long_retrieval_control.py`, `tests/test_0106b_chunked_external_retrieval.py`, commit `68844dd`.

## 2026-11-01 — 0105 specialist training gates

- **Author:** Askael
- **Commit:** `8586f79` (with model-centered groundwork in `bf8f6e2` and `4baae24`)
- **Status:** accepted CPU/reference plumbing; superseded in part by 0107–0108; not production-validated by this report
- **Subsystem:** training API, frozen residual specialists, production profile
- **Behavior:** Added a read-only Model B profile audit, CPU trajectory parity between the model-centered `TrainingRun` and legacy path, a document-first specialist sampler, specialist-only Hybrid-Muon training/checkpoint/resume, and a guarded dry-run-first launcher.
- **Compatibility:** Explicitly preserves Model B's 348,261,376-parameter schema, sparse head, legacy foundation trainer, optimizer and active checkpoint. Specialist checkpoints are independent and bind to a foundation hash. Initial rows are single-document because packed multi-document specialist isolation was not yet implemented in 0105.
- **Validation:** Historical report records CPU testing only and states no GPU hardware was available in its authoring environment. No tests were rerun here.
- **Remaining work:** 0107–0108 addressed exact production geometry, multibatch masks, logging and RoPE reuse. GPU full-model parity, full 12-layer performance/VRAM, long-range quality and composed serving remain open.
- **Evidence:** `mini_llm/production_profile_0105.py`, `mini_llm/specialists/`, `scripts/train_0105_specialist.py`, `tests/test_model_b_gpu_contract_0105b.py`, `tests/test_training_path_parity_0105c.py`, `tests/test_specialist_sampler_0105d.py`, `tests/test_specialist_training_run_0105e.py`, `tests/test_specialist_cli_0105f.py`.

## 2026-10-30 to 2026-11-01 — 0103 model-centered API, execution, serving and evaluation

- **Author:** Lucka and Askael
- **Commits:** `7bc41a2` (WIP), `ae248a9` (explicit revert), `a15f664` (accepted cleanup/model facade), `fd20236` (0103K fine buckets), plus merge/continuation commits `4baae24` and `bf8f6e2`
- **Status:** accepted additive compatibility foundation; WIP remodel superseded/reverted; GPU gates remain open
- **Subsystem:** public model API, loading, checkpoint safety, generation sessions, serving, evaluation, sequence bucketing
- **Behavior:** Introduced strict `Model` loading/inspection, regression gates for protected Model B, canonical native/external loading, explicit execution policies and observed CUDA-Graph reporting, request-scoped unified generation, mixed local/remote OpenAI-compatible serving, incremental unified evaluation, and opt-in fine sequence-length buckets.
- **Compatibility:** Existing CLIs and legacy imports remain compatibility paths. Missing or shape-incompatible native weights fail closed. Remote providers do not expose native logits/KV/training. Runtime backend flags remain process-global. `strict-fast` requires observed graph execution rather than a requested flag. Fine buckets are opt-in and do not alter sampled targets/checkpoints, but increase possible compiled shapes. Model B remains exactly 348,261,376 parameters with the stride-16 head.
- **Validation:** Historical reports record CPU regression suites. 0103A2 explicitly classified low-VRAM GPU runs as blocked, not passed, and states GPU parity remained unverified in its authoring environment. No tests were rerun here.
- **Important negative results:** a production-width Triton gate hit CUDA OOM while another 348M job may have occupied VRAM; this was not a numerical mismatch. CUDA Graph support was not established for every model. Per-instance backend isolation and append-only conversational KV continuation were not implemented. Unified evaluation did not replace specialist scorers.
- **Remaining work:** Run GPU gates on an idle device; verify Model B graph capture, long prefill/decode, full-scale loading and checkpoint transaction paths; complete per-model backend isolation and remaining benchmark migration only with golden-result parity.
- **Evidence:** `mini_llm/model_api.py`, `mini_llm/model_loading.py`, `mini_llm/model_sources.py`, `mini_llm/generation.py`, `scripts/run_0103a_regressions.py`, `benchmarks/run_unified.py`, `tests/test_model_api_0103_0.py`, `tests/test_baseline_contract_0103a.py`, `tests/test_regression_runner_0103a.py`, `tests/test_canonical_model_loading_0103c.py`, `tests/test_inference_policy_0103d.py`, `tests/test_generation_session_0103e.py`, `tests/test_remote_serving_0103f.py`, `tests/test_dual_resolution_fine_shape_buckets_0103k.py`.

## 2026-10-28 to 2026-10-29 — 0102 dual-resolution attention and protected Model B

- **Author:** Askael
- **Commits:** `7b64c77`, `9dbd60c`, `5dec566`, `a6cb830`, `52b8ae3` (with benchmark correction `edfcb72`)
- **Status:** mixed: accepted architecture/correctness work, accepted defaults, and retained experimental/default-off alternatives
- **Subsystem:** architecture, document-first training, Triton kernels, memory/throughput, inference
- **Behavior:**
  - **0102 base–G — accepted correctness foundation:** added dual-resolution local attention plus completed-block compression/global summaries, bounded inference, full-model smoke tools, document-first full-sequence supervision, efficient metadata reuse/short paths, and the unequal-length summary-RoPE sentinel fix.
  - **0102B–C — experimental/default-off at introduction:** optional Triton local/global/compressor kernels; authoring reports did not claim GPU compilation or production readiness.
  - **0102F negative constraint:** 65K backward could OOM on 16 GB; short runs were not evidence of 65K correctness. Long documents must not be silently skipped/truncated.
  - **0102H–J — accepted after a negative first measurement:** the first 0102H single cold run showed a reported 3.62× slowdown. 0102I identified cold/JIT confounding and added warmed A/B. 0102J reported warmed gains and promoted packed/BF16 projections as the CuPy default; NumPy retained exact reference behavior. Fused pooling remained opt-in.
  - **0102K–M — fixes/experiments:** fixed Triton pointer launch; compressor RMSNorm and local/global gate fusion remained default-off pending GPU parity and end-to-end evidence.
  - **0102N — experimental opt-in:** BF16 output-projection `dX`; preserved FP32 parameter-gradient accumulation and exposed global backward as the next bottleneck.
  - **0102O — accepted Model B:** introduced matrix-pooled Model B as a separate architecture with 348,261,376 parameters; the prior 406,199,296-parameter dual-resolution model remained available.
  - **0102P–T — mixed:** benchmark ablation fix accepted; Q contiguous view and R metadata reuse default-on with guarded fallbacks; S fused matrix-pool forward remained off. Its later measured +0.34% was within variation and was not promoted on that evidence.
  - **0102U–W — experimental/default-off:** fused matrix-pool backward, chunked packed-projection backward, and empty-global-tile skipping retained fallbacks and required GPU parity/memory/throughput measurements.
  - **0102X — opt-in:** source-preserving accumulation regrouping; target-equivalent but not bitwise resume-equivalent because document order within an optimizer step changes.
  - **0102Y — accepted default optimization:** prepared local RoPE reuse; no unmeasured production speedup was claimed.
  - **0102Z — accepted protected architecture extension:** added the deterministic stride-16 original-token head with no new learned tensors, established phases and same-document causality. This became the protected Model B contract in 0103.
- **Compatibility:** Most performance work preserved parameter names/checkpoint layouts and retained reference fallbacks. Model B is a separate architecture, not an in-place reinterpretation of the 406M model. Weight initialization from a prior Model B checkpoint is weights-only; optimizer state must not be silently reused across architecture changes. Data-policy changes apply only to dual-resolution training. Environment flags and effective defaults must be logged with runs.
- **Validation:** Individual historical reports record CPU and selected RTX results; several explicitly say GPU tests were not run in their authoring environment. No tests were rerun here.
- **Remaining work:** Preserve Model B exactly; verify any active/external checkpoint manifest; run idle-GPU parity and full optimizer-step measurements; keep rejected/default-off paths off unless a new logged gate promotes them; do not infer 65K fit from reduced-tail smoke benchmarks.
- **Evidence:** `mini_llm/ops/dual_resolution_attention.py`, `mini_llm/ops/matrix_pool_dual_resolution_attention.py`, `mini_llm/ops/dual_resolution_attention_inference.py`, `mini_llm/ops/triton_dual_resolution_local.py`, `mini_llm/ops/triton_dual_resolution_dense.py`, `mini_llm/config.py`, `mini_llm/train_extended.py`, `tests/test_dual_resolution_0102.py`, `tests/test_dual_resolution_document_sampling_0102e.py`, `tests/test_dual_resolution_multibatch_rope_0102g.py`, `tests/test_dual_resolution_packed_projections_0102h.py`, `tests/test_matrix_pool_dual_resolution_0102o.py`, `tests/test_matrix_pool_contiguous_0102q.py`, `tests/test_matrix_pool_metadata_0102r.py`, `tests/test_matrix_pool_fused_0102s.py`, `tests/test_matrix_pool_fused_backward_0102u.py`, `tests/test_packed_projection_chunk_0102v.py`, `tests/test_global_skip_empty_tiles_0102w.py`, `tests/test_dual_resolution_accum_length_buckets_0102x.py`, `tests/test_matrix_pool_prepared_local_rope_0102y.py`, `tests/test_sparse_stride_history_0102z.py`.

## 2026-10-27 — 0100 independent full-history heads

- **Author:** Lucka
- **Commit:** `f98bbe5`
- **Status:** accepted opt-in research implementation; superseded as the current baseline by Model B; target-GPU production gate incomplete
- **Subsystem:** historical attention, checkpoint migration, training/inference
- **Behavior:** Added independent per-layer full-history heads with zero-output initialization, rectangular streaming Triton attention, valid-row compaction, exact positions, handwritten backward, cache integration, and transactional checkpoint/optimizer migration.
- **Compatibility:** Adds 2,097,152 parameters for the documented eight-layer layout and therefore requires explicit conversion to a new checkpoint directory. Source checkpoints are preserved; conversion is atomic and validates optimizer groups/birth steps. Arbitrary dense CUDA masks and provisional/shared-context routes remain unsupported. It is a historical alternative, not Model B.
- **Validation:** Historical report records CPU coverage and functional tests on a 4 GiB Quadro P1000. It explicitly states RTX 5080 4K×65K throughput, 16 GiB fit, microbatch 3, real migration and two-step resume were not executed there. No tests were rerun here.
- **Important negative results:** P1000 timings cannot be extrapolated to RTX 5080; numerical differences versus a square baseline were not evidence of a 4.7–15× intrinsic speedup. CUDA-Graph capture and strict refresh parity remained open.
- **Remaining work:** Treat as historical unless revived deliberately; if revived, use copied checkpoints and rerun target-GPU migration/resume/parity/memory gates.
- **Evidence:** `mini_llm/ops/full_history_head.py`, `mini_llm/ops/triton_full_history_attention.py`, `examples/migrate_full_history_heads.py`, `tests/test_full_history_upgrade_0100.py`, `tests/test_full_history_triton_0100.py`, commit `f98bbe5`.

## 2026-10-26 — 0092 live benchmark and supervised-token instrumentation

- **Author:** Lucka
- **Commit:** `58e8799`
- **Status:** accepted instrumentation; benchmark quality results not implied
- **Subsystem:** training-time evaluation and token accounting
- **Behavior:** Added optional live benchmark stops, exact supervised-token tracking, per-benchmark result schemas and resumable reporting.
- **Compatibility:** Instrumentation is operationally optional and can add evaluation cost. `steps × batch × context` is not a valid substitute for supervised-token counts. Branching from a benchmark result was not implemented in 0092.
- **Validation:** Historical report records CPU validation before GPU handoff; no tests were rerun here.
- **Remaining work:** Keep benchmark definitions/version hashes stable and distinguish model quality from instrumentation completion.
- **Evidence:** `benchmarks/live_suite.py`, `benchmarks/configs/`, benchmark tests under `tests/`, commit `58e8799`.

## 2026-10-26 — 0090 residual specialist prototype

- **Author:** Lucka
- **Commit:** `41c8fa2`
- **Status:** accepted CPU prototype; superseded by 0105 and 0107–0108 for current specialist training
- **Subsystem:** frozen-foundation residual specialists and research design
- **Behavior:** Established a frozen foundation plus zero-initialized hidden residual specialist, strict no-foundation-gradient behavior, specialist-only serialization concepts, and a two-layer CPU smoke implementation. The intended scientific specialist was 12 layers; two layers were never a capacity claim.
- **Compatibility:** Historical prototype targeted the earlier 338M historical-hybrid foundation and did not implement private specialist history, 12-layer GPU execution, BF16 production validation, or a trained ensemble combiner. The 0090B LaTeX note is a design/evidence record, not proof of a specialist ensemble.
- **Validation:** Historical 0090A report records `43 passed, 1 skipped` on NumPy and explicitly says no BF16/64K production test was performed. Not rerun here.
- **Remaining work:** Current work should use the 0105/0107/0108 Model B path and revalidate all scientific conclusions against protected Model B.
- **Evidence:** `mini_llm/specialists/`, current specialist tests, commit `41c8fa2`; historical design source `docs/research/0090B_modular_residual_specialist_combination.tex`.

## 2026-10-24 to 2026-10-26 — 0089 profiling, accepted fusions, rejected kernels and historical-read optimization

- **Author:** Askael
- **Commits:** `96ed2c3`, `4488335`, `33fceba`
- **Status:** mixed; see decisions below
- **Subsystem:** GPU performance, numerical parity, historical-read kernels, framework migration analysis
- **Behavior and decisions:**
  - **0089A–E:** accepted benchmark/profiling infrastructure. 0089C residual/RMSNorm fusion was promoted in GPU-fast profiles after reported parity and modest complete-block gains. Compiled-Torch block/LM-head/stack comparisons were references, not approval to replace the trainer. A packed gate/up GEMM ablation was rejected.
  - **0089J–K:** accepted fair comparison and correlation-correct profiling infrastructure; historical work, not host launch starvation, dominated the heavy-history case.
  - **0089L:** vectorized BF16 SwiGLU was **promoted** despite a small reported complete-training gain because direct output was bitwise, paired runs were consistently positive, and no durable allocation was added.
  - **0089M:** inverse-RoPE vectorization result was **rejected as confounded**; measured apparent gain exceeded the physical scope bound. Static depth-1 CUDA Graph work was **rejected** because the measured ideal host-side bound was about 0.12% with high semantic risk.
  - **0089N:** fused packed-QKV RoPE forward was **accepted/promoted** for CuPy fast profiles. Full PyTorch migration was **rejected/deferred**: historical custom ops, 61K fit, Hybrid-Muon offload and exact resume were not demonstrated.
  - **0089O:** accepted correlation-correct historical audit and staged optimization plan.
  - **0089P:** first streaming deterministic block forward was **rejected and removed**. It was fast and reduced reservation, but failed complete-model output/gradient parity (`body rel-L2 5.611e-3`, global-gradient rel-L2 `1.3895e-2`, worst cosine `0.998038`).
  - **0089R:** corrected two-pass forward remained **rejected for default training**; full-model parity still failed (`body rel-L2 5.5648e-3`, gradient rel-L2 `1.3625e-2`, worst cosine `0.9923513`) despite small isolated error and scalar-loss difference.
  - **0089S:** accepted generic kernel/prewarm plumbing only; it did not promote the forward.
  - **0089T:** retained the deterministic historical forward as an **explicit opt-in**, not numerical parity and not a silent default. Direct production calls were near `1e-6`, but full-model differences remained. Reported ~25k→slightly above 26k supervised tok/s was explicitly **user-observed**, not a controlled artifact.
  - **0089U:** historical failed shared-memory prototype, **superseded by 0089V**.
  - **0089V:** retained streaming deterministic historical backward as an **explicit opt-in**. Historical report measured a 10.46% no-optimizer F+B reduction for the fixed workload; complete BF16 gradient rel-L2 `0.003442` still failed the old strict threshold, so it was not made a silent default.
  - **Unnumbered fusion optimization:** accepted several probability-fusion GPU changes; rejected duplicate `dpre` recomputation (large slowdown despite ~1 GiB saving), `float4` elementwise specialization (noise), and one large SGEMM for scalar gradients. One broad-suite failure was unrelated argument validation, not silently counted as a pass.
- **Compatibility:** Promoted CuPy fusions preserved parameter/checkpoint formats and FP32 parameter-gradient accumulation. Historical forward/backward optimizations change floating-point reduction order and must be recorded in run metadata; do not switch them silently mid-run. PyTorch model weights could be mapped, but optimizer/checkpoint semantics were not compatible with stock `torch.optim`.
- **Validation:** Results above are historical report records, not tests run during this migration.
- **Remaining work:** Preserve negative results; do not resurrect removed kernels without full-model gates. Any framework migration must first pass compact historical correctness, memory, optimizer and native-resume gates. Validate opt-in historical kernels over controlled training/validation trajectories.
- **Evidence:** `mini_llm/ops/triton_historical_block.py`, `mini_llm/ops/historical_read.py`, `benchmarks/profile_0087a_training.py`, `benchmarks/analyze_0089o_historical_nsys.py`, `tests/test_0089a_optimized_training.py`, `tests/test_0089b_block_contract.py`, `tests/test_0089c_residual_rmsnorm.py`, `tests/test_0089d_lm_head_contract.py`, `tests/test_0089e_block_stack_contract.py`, `tests/test_0089j_optimization_comparison.py`, `tests/test_0089k_block_stack_nsys.py`, `tests/test_0089n_fused_qkv_rope.py`, `tests/test_0089o_historical_nsys.py`, commits `96ed2c3`, `4488335`, `33fceba`.

## 2026-10-20 to 2026-10-22 — 0084–0088 fusion, external models, attention and serving

- **Author:** Askael
- **Commits:** `801167d`, `68ba347`, `69c5826`, `5516c61`, `1a54a06`, `647c280`
- **Status:** mixed accepted infrastructure and rejected/default-off kernel candidates
- **Subsystem:** probability fusion, GGUF, attention performance, inference and serving
- **Behavior and decisions:**
  - **0084B:** accepted online frozen-teacher probability fusion with bounded teacher windows and partition/provenance controls. It did not prove fused quality or make cross-tokenizer comparisons valid.
  - **0085H:** accepted WikiText-2 adapter for supported GGUF checkpoints; the report left full real-model WikiText execution pending.
  - **0086A:** accepted attention baseline/oracle audit; no production files changed.
  - **0086B:** fused local forward candidate **rejected for production** because complete-step throughput regressed; grouped cuBLAS remained default and custom backward was not pursued.
  - **0086C:** cuDNN SDPA integration was a **no-go** in the audited environment; version strings alone did not establish required support.
  - **0087A:** profiling concluded ordinary Python launch starvation was not a material bottleneck for measured F+B; a mechanical C++ rewrite was not justified.
  - **0087B:** compiled FlexAttention established a useful fused-backward reference and Amdahl bound, but did not justify replacing the whole model/framework.
  - **0088A attention:** accepted opt-in Triton mixed-attention F+B after reported parity/performance; grouped remained the code default at that point and no six-fold full-model claim was made.
  - **0088A server / 0088B / 0088C:** accepted experimental native OpenAI-compatible serving, historical-read inference support and per-model sampling. Optimized incremental historical K/V updates, multi-user scheduling and routed-prefix support were not implemented.
- **Compatibility:** Fusion preserves frozen teachers and requires matching data/tokenizer interpretation. External loaders support specific Llama-family GGUF/Safetensors contracts, not arbitrary formats. Server APIs preserve common OpenAI fields but tool-call format was experimental. Attention alternatives remained selectable/fallback-safe rather than silently replacing semantics.
- **Validation:** Historical reports include CPU and RTX measurements; no tests were rerun here.
- **Remaining work:** Keep rejected 0086 candidates rejected; benchmark supported external models with exact hashes; verify serving concurrency/state ownership; use full-model parity and optimizer-step evidence before changing attention defaults.
- **Evidence:** `mini_llm/collective/online_training.py`, `mini_llm/collective/probability_fusion.py`, `mini_llm/external/`, `mini_llm/ops/triton_mixed_attention.py`, `mini_llm/serving_api.py`, `serve.py`, `tests/test_0084b_core_online_fusion.py`, `tests/test_0085h_gguf_wikitext.py`, `tests/test_0086a_attention_oracle.py`, `tests/test_0086b_fused_local_forward.py`, `tests/test_0086c_cudnn_sdpa.py`, `tests/test_0087a_profile_ranges.py`, `tests/test_0087b_attention_contract.py`, `tests/test_0088a_custom_attention.py`, `tests/test_0088a_openai_server.py`, `tests/test_0088c_serving_sampling.py`.

## 2026-10-13 to 2026-10-19 — 0080 persistent soft-plan/block-proposal research

- **Author:** Barbeau, Lucka and Askael
- **Commits:** `e97fafe`, `2e07d49`, `122d707`, `ac9d529`, `056077e`, `1e5ffe3`, `d359a29`, `a27487b`
- **Status:** historical research line; mixed implemented references, experimental paths, inconclusive audits and open production gates
- **Subsystem:** block proposals, soft-vocabulary interfaces, prefix decoding, MoE routing and checkpoints
- **Behavior and decisions:**
  - Implemented K-block proposal/prefix-decoder research with strict causal alignment, handwritten gradients, checkpoint/optimizer state, production data integration and read-only benchmark tools.
  - Production live parity found a **negative horizon-1 validation result** and weaker later gains; this is preserved as evidence against assuming proposal benefit.
  - The decoder checkpoint discontinuity investigation was **inconclusive**; reconstruction could neither prove nor rule out a save/load discontinuity. A before-save/after-load same-process test remained the definitive next check.
  - Checkpoint round-trip work accepted authoritative packed-QKV state and warned not to restore initialization-era packed matrices.
  - 0080AE–AJ3 built bounded CPU/full-vocabulary references, streamed/all-source reads and exact source-mixture factorization. The production CLI and cached generation intentionally rejected unsupported variants until later gates; several reports explicitly made no RTX/full-training speed claim.
  - 0080AJ2 batching and 0080AJ3 CUDA fusion remained experimental; correctness did not imply faster performance.
  - 0080AM learned probability embedding introduced versioning/initialization implications and remained a research option.
  - 0080AO/AP changed deep/top-1 MoE routing. Unified routing was not bitwise or optimization-equivalent to prior top-1 behavior, and strict old config readers could reject new fields. Do not overwrite old checkpoints merely to adopt it.
  - 0080AQ diagnostics were observational and did not alter training objective.
  - Full 64K routed main-model cached generation and broad production acceptance remained explicitly open in the implementation summary.
- **Compatibility:** Architecture/checkpoint variants require explicit version gates. CPU tiny-vocabulary references are not safe substitutes for 65,280-vocabulary BF16/CuPy training. Read-only production benchmarks require consistent frozen checkpoint copies and exact dataset/sample fingerprints. This research line is not protected Model B.
- **Validation:** Historical reports contain their own recorded CPU/CuPy results and known missing-test-module caveats. No tests were rerun here.
- **Remaining work:** Treat as historical unless a new experiment is explicitly approved. Preserve negative/inconclusive evidence and compare against current Model B using matched data/token/compute definitions.
- **Evidence:** `mini_llm/predictive_refinement/`, `mini_llm/ops/hierarchical_memory.py`, block-proposal tests matching `tests/test_block_proposal_0080*.py`, `tests/test_0080ao_deep_top1_moe.py`, `tests/test_0080ap_unified_expert_routing.py`, `tests/test_0080aq_expert_usage_logging.py`, commits listed above.

## 2026-10-02 to 2026-10-05 — 0058 hierarchical routed memory

- **Author:** Askael
- **Commits:** implementation history around `5da3846`, `75f66b4`, `7512e6d`, `7b2fc48`, `730c9d2`, `e6332d9`, `6d5f1a8`, `9d97a1b`; current report path recorded by `a27487b`
- **Status:** accepted historical architecture; superseded as current baseline
- **Subsystem:** long-context routing and target-only training
- **Behavior:** Added hierarchical same-document memory routing: shallow block selection over long history, exact selected-token reopening with chronological/original positions, fixed deep active length, target-only LM projection/loss, and router diagnostics. Mean pooling was selected for the production memory lifecycle; learned pooling retained more memory.
- **Compatibility:** The 16K/32K/64K presets were shape-compatible with each other, but `--context-length` was deliberately rejected when hierarchical memory was enabled to avoid confusing source horizon with deep context. Compatible with optimizer offload, chunked head and block checkpointing. It is not Model B's matrix-pooled/stride-16 architecture.
- **Validation:** Historical validator covered reference and optional CuPy/BF16 paths. It was not rerun here.
- **Remaining work:** Preserve as research history; use current Model B tests for present development rather than assuming routed-prefix behavior.
- **Evidence:** `mini_llm/ops/hierarchical_memory.py`, `mini_llm/model/decoder_lm.py`, `mini_llm/data/token_shards.py`, `validate_0058_hierarchical_memory.py`, Git commits above.

## 2026-09-29 — Legacy extended-training foundation

- **Author:** Askael
- **Commit:** `6119842` (later reorganized by `a27487b`/`4baae24`)
- **Status:** accepted historical foundation; operational details partly superseded by current guides
- **Subsystem:** tokenizer, data shards, trainer and checkpoints
- **Behavior:** Recorded BPE merge-rank correctness, separate tokenizer training, deterministic multiprocessing shard generation, streaming corpus handling, accumulation, validation and checkpoint basics.
- **Compatibility:** The old model sizes, commands and VRAM estimates are historical and must not override current Model B or current CLI documentation.
- **Validation:** Legacy report states deterministic worker-count checks; not rerun here.
- **Remaining work:** Keep current operational guidance in `USER_GUIDE.md` and `DATA_PIPELINE.md`; use this entry only for provenance.
- **Evidence:** `mini_llm/tokenizer/`, `mini_llm/data/`, `mini_llm/train_extended.py`, `tests/test_tokenizer.py`, commit `6119842`.

---

## Migration ledger for old Markdown reports

This ledger is intentionally filename-oriented. Entries above are authoritative summaries; old report contents remain recoverable with `git show <listed-commit>:<path>` after deletion.

| Old report file or grouped range | Consolidated entry | Git provenance / durable evidence |
|---|---|---|
| `docs/IMPLEMENTATION_SUMMARY.md` (originally root `IMPLEMENTATION_SUMMARY.md`) | Legacy extended-training foundation | Recover as `git show 6119842:IMPLEMENTATION_SUMMARY.md` or `git show a27487b:docs/IMPLEMENTATION_SUMMARY.md`; current code under `mini_llm/tokenizer/`, `mini_llm/data/`, `mini_llm/train_extended.py` |
| `docs/research/HIERARCHICAL_MEMORY_0058.md` | 0058 hierarchical routed memory | current path at `a27487b`; implementation history listed in entry |
| `docs/research/README_persistent_soft_plan_0080.md`, `docs/research/IMPLEMENTATION_0080.md` | 0080 | `1e5ffe3`, `a27487b`; `mini_llm/predictive_refinement/` and `tests/test_block_proposal_0080*.py` |
| `docs/research/PRODUCTION_LIVE_PARITY_AUDIT_0080AA.md`, `docs/research/DECODER_CHECKPOINT_DISCONTINUITY_0080AC.md`, `docs/research/PRODUCTION_CHECKPOINT_ROUNDTRIP_0080AD.md` | 0080 negative/inconclusive/checkpoint evidence | `a27487b`; matching `tests/test_block_proposal_0080aa_live_audit.py`, `...0080ac_checkpoint_discontinuity.py`, `...0080ad_roundtrip.py` |
| `docs/research/0080AE_full_soft_interface_audit.md`, `docs/research/0080AF_cpu_full_vocab_reference.md`, `docs/research/0080AG_teacher_forced_soft_vocab_integration.md`, `docs/research/0080AH_AI_combined_own_all_streamed_read.md`, `docs/research/0080AJ_exact_source_mixture_optimization.md`, `docs/research/0080AJ2_batched_full_vocabulary_gemm.md`, `docs/research/0080AJ3_fused_vocab_cuda.md` | 0080 full-vocabulary research | `a27487b`; matching `tests/test_block_proposal_0080a*.py` |
| `docs/research/0080AM_learned_probability_embedding.md`, `0080AO_32_layer_two_expert_top1.md`, `0080AP_unified_moe_expert_routing.md`, `0080AQ_expert_utilization_diagnostics.md` | 0080 later architecture/routing work | `a27487b`; matching `tests/test_block_proposal_0080am*.py`, `tests/test_0080ao*.py`, `tests/test_0080ap*.py`, `tests/test_0080aq*.py` |
| `docs/research/PRODUCTION_BLOCK_PROPOSAL_BENCHMARK.md` | 0080 read-only benchmark | `a27487b`; `benchmarks/evaluate_block_proposal_production.py` and its tests |
| `docs/0084B_CORE_ONLINE_FUSION.md` | 0084–0088 | `801167d`; `tests/test_0084b_core_online_fusion.py` |
| `docs/0085H_EXTERNAL_GGUF_WIKITEXT.md` | 0084–0088 | `68ba347`; `tests/test_0085h_gguf_wikitext.py` |
| `docs/0086A_ATTENTION_BASELINE_AUDIT.md`, `0086B_FUSED_LOCAL_FORWARD.md`, `0086C_CUDNN_SDPA_FWD_BWD_BENCHMARK.md` | 0084–0088 | `68ba347`, `69c5826`; matching `tests/test_0086*.py` |
| `docs/0087A_TRAINING_TIMELINE.md`, `0087B_COMPILED_TORCH_ATTENTION.md` | 0084–0088 | `5516c61`; matching `tests/test_0087*.py` |
| `docs/0088A_CUSTOM_TRITON_MIXED_ATTENTION.md`, `0088A_NATIVE_OPENAI_SERVER.md`, `0088B_HISTORICAL_READ_NATIVE_INFERENCE.md`, `0088C_PER_MODEL_SAMPLING.md` | 0084–0088 | `1a54a06`, `647c280`; matching `tests/test_0088*.py` |
| `docs/0089A_OPTIMIZED_PRODUCTION_REPROFILE.md`, `docs/0089B_COMPILED_TORCH_BLOCK.md`, `docs/0089C_TRAINING_RESIDUAL_RMSNORM.md`, `docs/0089D_COMPILED_TORCH_LM_HEAD.md`, `docs/0089E_COMPILED_TORCH_BLOCK_STACK.md` | 0089 | `96ed2c3`; matching `tests/test_0089a*`–`test_0089e*` |
| `docs/0089J_OPTIMIZATION_VALIDATION.md`, `0089K_COMPARATIVE_GPU_BOTTLENECK_AUDIT.md`, `0089L_VECTORIZED_BF16_SWIGLU.md`, `0089M_FAIR_FLAGS_ROPE_AND_GRAPH_AUDIT.md`, `0089N_FUSED_QKV_ROPE_AND_MIGRATION_GATE.md` | 0089 | `4488335`; profiler/benchmark code and matching tests |
| `docs/0089O_HISTORICAL_BOTTLENECK_AUDIT.md`, `0089O_HISTORICAL_OPTIMIZATION_PLAN.md`, `0089P_STREAMING_HISTORICAL_BLOCK_FORWARD_NEGATIVE.md`, `0089R_TWO_PASS_HISTORICAL_BLOCK_FORWARD_NEGATIVE.md`, `0089S_HISTORICAL_BLOCK_TRITON_PREWARM.md`, `0089T_HISTORICAL_BLOCK_ACCEPTANCE_AUDIT.md` | 0089 historical read, including rejected forward results | `4488335`; `mini_llm/ops/triton_historical_block.py`, profiler/analyzer tests |
| `docs/0089U_STREAMING_HISTORICAL_BLOCK_BACKWARD_PROTOTYPE.md`, `0089V_STREAMING_HISTORICAL_BLOCK_BACKWARD.md` | 0089 superseded prototype and retained opt-in backward | `33fceba` |
| `docs/FUSION_OPT_01_COMPREHENSIVE_GPU_OPTIMIZATION.md` | 0089 unnumbered fusion optimization | `4488335`; fusion code/tests in that commit |
| `docs/0090A_RESIDUAL_SPECIALIST_PHASE0_PHASE2.md` | 0090 | `41c8fa2`; superseded by current `mini_llm/specialists/` work |
| `benchmarks/README_0092_LIVE_BENCHMARKS.md` | 0092 | `58e8799`; current benchmark code/configs/tests |
| `FULL_HISTORY_HEADS_0100.md` | 0100 | `f98bbe5`; full-history code/tests listed in entry |
| `docs/DUAL_RESOLUTION_0102.md`, `docs/PATCH_0102B_TRITON_LOCAL.md`, `docs/PATCH_0102C_TRITON_GLOBAL_AND_COMPRESSION.md`, `docs/PATCH_0102D_FULL_MODEL_TRAINING_AND_MEMORY.md`, `docs/PATCH_0102E_DOCUMENT_FIRST_FULL_SEQUENCE.md`, `docs/PATCH_0102F_FAST_DOCUMENT_EXECUTION.md`, `docs/PATCH_0102G_VARIABLE_BATCH_SUMMARY_ROPE.md`, `docs/PATCH_0102H_PACKED_BF16_PROJECTIONS.md`, `docs/PATCH_0102I_STEADY_STATE_ABLATION.md`, `docs/PATCH_0102J_DEFAULT_PACKED_AND_FUSED_POOLING.md`, `docs/PATCH_0102K_POOLING_TRITON_POINTER_FIX.md`, `docs/PATCH_0102L_COMPRESSOR_RMSNORM_FUSION.md`, `docs/PATCH_0102M_FUSED_LOCAL_GLOBAL_GATE.md`, `docs/PATCH_0102N_ATTENTION_BACKWARD_PROFILE.md` | 0102 base–N | `9dbd60c`; dual-resolution code and matching `tests/*0102*.py` |
| `docs/PATCH_0102O_MATRIX_POOL_MODEL_B.md` | 0102 Model B | `9dbd60c`; `mini_llm/ops/matrix_pool_dual_resolution_attention.py`, `tests/test_matrix_pool_dual_resolution_0102o.py` |
| `docs/PATCH_0102P_MODEL_B_SCATTER_AND_ABLATION.md`, `PATCH_0102Q_TO_0102T_MODEL_B_OPTIMIZATIONS.md`, `PATCH_0102U_FUSED_MATRIX_POOL_BACKWARD.md`, `PATCH_0102V_PACKED_PROJECTION_BACKWARD_CHUNKS.md`, `PATCH_0102W_GLOBAL_SKIP_EMPTY_SUMMARY_TILES.md` | 0102 P–W | `5dec566`; matching matrix-pool/packed/global tests |
| `docs/PATCH_0102X_DUAL_LENGTH_AWARE_ACCUMULATION.md`, `PATCH_0102Y_MODEL_B_PREPARED_LOCAL_ROPE.md` | 0102 X–Y | `a6cb830`; matching tests |
| `docs/MODEL_B_STRIDE16_SPARSE_0102Z.md` | 0102Z protected Model B | `52b8ae3`; `tests/test_sparse_stride_history_0102z.py` |
| `docs/0103_MODEL_CENTRIC_FOUNDATION.md`, `0103A_BASELINE_REGRESSION.md`, `0103A2_GPU_MEMORY_AWARE_PARITY.md`, `0103C_UNIFIED_MODEL_LOADING.md`, `0103D_AUTOMATIC_INFERENCE_EXECUTION.md`, `0103E_UNIFIED_GENERATION_SESSIONS.md`, `0103F_UNIFIED_SERVING_AND_REMOTE_MODELS.md`, `0103G_UNIFIED_EVALUATION.md` | 0103 | `a15f664`; model/loading/generation/serving/evaluation code and tests |
| `docs/PATCH_0103K_ADAPTIVE_FINE_BUCKETS.md` | 0103K | `fd20236`; `tests/test_dual_resolution_fine_shape_buckets_0103k.py` |
| `docs/0105_MODEL_CENTERED_SPECIALIST_GATES.md` | 0105 | `8586f79`; 0105 specialist/profile tests |
| `docs/0106A_SMOLLM2_RETRIEVAL_CONTROL.md` | 0106 | `68844dd`; external retrieval runner/tests |
| No standalone 0107/0108 Markdown report existed | 0107–0108 | `eb61df0`; code/tests listed in entry. Future work must be logged here rather than creating one. |

## Research artifact cross-reference

The retained LaTeX/PDF research artifacts are indexed in `docs/research/README.md`. They are historical design/evidence documents and do not replace this change log or the protected current baseline.