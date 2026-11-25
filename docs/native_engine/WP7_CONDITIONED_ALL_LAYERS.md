# WP7 — executed all-48 conditioned validation

## Result and scope

**PASS: all 48 actual-checkpoint layers, both retained input lineages, both captured tokens `[1, 7]`, and the standalone final collapse/vocabulary projection.** No production code, numerical equation, arithmetic policy, operator tolerance, original capture, legacy strict test, parent status, or reference manifest was changed.

For these tested inputs, native and unchanged Python operations are consistent within the existing operator envelope. The evidence supports **FP32 reduction roundoff amplified by quantization/routing thresholds and persistent-state propagation**, rather than a demonstrated decoder equation defect. The downstream trajectory discrepancy is **not itself tiny roundoff**: quantization and route membership can change discretely. This result does not assert bitwise equivalence, arbitrary-context correctness, independent-reference equivalence, or model quality from two tokens.

The unchanged strict whole-trajectory comparison still fails. Its componentwise `allclose` rule cannot serve as an operation-equivalence criterion once its inputs/history diverge across discontinuous quantizers/routing. We have not silently replaced that test, derived a new quality threshold, or changed its failed status.

## Artifacts and provenance

- Machine-readable summary: `docs/native_engine/WP7_CONDITIONED_ALL_LAYERS.json`.
- Complete arrays, per-comparison maxabs/relative-L2/maxrel/exactness/violations, quantizer bytes and per-layer weight identities: `logs/wp7_conditioned_all_layers_v2/layer00/` through `layer47/` (`evidence.npz`, `report.json`).
- Six completed batch manifests: `batch00-08.json` through `batch40-48.json` in that directory. Each records the executed arguments, original capture hashes, helper source hashes and native/GGML library hashes.
- Final collapse/head: `logs/wp7_conditioned_all_layers_v2/head/`.
- Final test execution record: `logs/wp7_conditioned_all_layers_v2/validation.json`.

Original inputs are the retained `logs/wp7_divergence/actual_full/{native,python}.npz` and `token1/{native,python}.npz`, not a newly captured prefix or surrogate model. Tests verify all four SHA256 values against the original `WP7_FULL_MODEL_DIAGNOSTICS.json` metadata. Original captures are never overwritten. The earlier representative evidence and `docs/qwen4exp/numerical_root_cause.md` remain unchanged.

Initial diagnostic attempt `logs/wp7_conditioned_all_layers_v1/` is preserved: layers 0–2 passed, then paged QSA correctly rejected full raw readback with `paged QSA raw diagnostics disabled`. The localized correction enabled the **existing** `raw_diagnostics=True` option. No native API or arithmetic change was needed. The complete sweep was subsequently executed into a new directory. `summary_draft.json` preserves an initial aggregation draft whose HC-read summary inadvertently also counted MoE hidden intermediates; the final summary filters the actual HC reads. Neither correction changed any numerical result.

## Conditioned contract

For each layer, weights are selected by actual `blk.<layer>` names using existing `checkpoint_upload`, `CheckpointMoE`, recurrent reference factory and QSA components. Only one layer is resident at a time. Router layout matches the captured full-model factory: dense GPU upload followed by its transpose view, not a copied transposed matrix.

The placement is exactly the captured one: **all ten routed experts on CPU GGML**, shared and hidden/mixer/HC math on GPU, explicit policy `cpu_ggml_q8_gpu_reference_fp32`. There is no hidden-math CPU fallback, omitted expert, GPU-resident expert substitution, or duplicate model implementation.

For every native-history and Python-history lineage:

1. **HC reads:** compare attention and FFN hidden/gains on identical incoming streams and identical actual norm/projection weights. Layer 0 uses the retained decoded embedding replicated into four streams. Layer 1 additionally replays the existing PLE with tokens `[1,7]`, compares updated streams/history, and uses the corresponding lineage's result as the common HC operand.
2. **Stateful mixer:** reset once per lineage, then run both steps continuously with the same captured attention hidden sequence in native and Python. GDN output, convolution history and recurrence matrix are checked. Paged QSA uses captured FP32 KV policy and logical active raw-index/K/V state plus exact selected IDs. Empty pooled history is checked explicitly at both steps.
3. **MoE:** submit the exact captured FFN hidden to both owners. Compare all router logits/probabilities/weights, exact sorted ten IDs, exact ten CPU rows individually, weighted terms, routed/shared branches, shared intermediates, total output, once-only weighting and ordered combine. There are **1,920 individually checked CPU expert rows**, all bit-exact.
4. **HC writes:** use identical streams, native branch output and copied native read gains for both implementations. Thus the write comparison has identical *complete operands*, rather than approximately equal independent gains. All **384** paired attention/FFN writes are bit-exact. Aggregate reconstruction against original captured writes is recorded separately, not mislabeled as the same-input operator gate.
5. **Cross-lineage amplification:** separately record different-input metrics, actual GGML Q8_K/Q8_0 bytes, changed scale blocks and signed integer codes, sample changed-code coordinates/operands, changed expert IDs and output deltas. These are explicitly not same-input failures.

Final read-only collapse is compared on each lineage's token-specific layer-47 streams. Both existing vocabulary projections then consume the **identical native collapsed F32 vector**, covering all **248,320** outputs. The standalone harness calls the exported internal CUDA launcher already used by the decoder; it adds no production math, no decoded full vocabulary and no new public API. Its internal C++ symbol name is intentionally test/build-specific.

The fixed operator rule remains `rtol=2e-4, atol=2e-4`, as in the existing actual-checkpoint conditioned/chaining tests. Exact discrete/expert-row/decomposition contracts are stronger where applicable. Near-zero componentwise relative errors are retained, not used to justify tolerance relaxation. No whole-graph tolerance was enlarged.

## Measured results

All conditioned checks have zero envelope violations; exact contracts pass.

| Comparison | Count | Maximum absolute error | Maximum relative L2 |
| --- | ---: | ---: | ---: |
| HC attention/FFN hidden | 384 | `6.042420864e-6` | `3.873271111e-7` |
| HC gains | 384 | `4.768371582e-7` | `7.479786352e-7` |
| HC writes, identical complete operands | 384 | `0` | `0` |
| Mixer output | 192 | `1.534819603e-6` | `3.475975089e-6` |
| GDN convolution history | 144 | `7.629394531e-6` | `1.050439234e-7` |
| GDN recurrence matrix | 144 | `1.907348633e-6` | `2.848883944e-7` |
| QSA active raw index | 48 | `2.384185791e-6` | `2.055444655e-7` |
| QSA active keys | 48 | `1.907348633e-6` | `1.512761414e-7` |
| QSA active values | 48 | `3.814697266e-6` | `1.534925238e-7` |
| MoE total output | 192 | `4.768371582e-7` | `3.940890593e-7` |
| CPU expert-row bundles | 192 × 10 rows | `0` | `0` |
| Final collapse | 4 | `7.629394531e-6` | `6.341102132e-8` |
| Vocabulary projection, identical collapsed input | 4 × 248320 rows | `5.722045898e-6` | `1.565377436e-7` |

All **192/192 native aggregate HC writes** reconstruct the original native full captures **bit-for-bit**. All **384** aggregate reconstruction comparisons, including Python lineage, have zero envelope violations. This links the standalone operator sweep to the original full captures, rather than merely passing on synthetic inputs.

Across different lineages, Q8_K encoded bytes change in **96/96** layer/token cases; actual integer codes change in **82/96**. Route membership changes in **eight** cases: token 0/layer 15; token 1/layers 12, 21, 31, 34, 36, 39, 44. These later route changes do not contradict the earlier layer-6 result, where routing did not flip. Full per-case metrics and code-change examples are in the JSON.

### Legacy failure remains visible

| Strict full-trajectory diagnostic | Token 0 | Token 1 |
| --- | ---: | ---: |
| Full-logit maxabs | `0.1547901630` | `0.4191079140` |
| Full-logit relative L2 | `0.0115831571` | `0.0378313720` |
| Full-logit component violations | `244163` | `247078` |
| Legacy strict acceptance | **FAIL** | **FAIL** |

The all-48 conditioned evidence explains why this is not a valid identical-operation comparison; it does not turn that legacy gate green. Parent-owned status is untouched. No quality/performance acceptance is inferred.

## Resource safeguards and executed commands

Every GPU process used only `CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631`. No other process was stopped and GPU0 was not used. There were no simultaneous full-model allocations or simultaneous GPU batches.

Each layer/head preflight requires **1 GiB target + 2 GiB reserve** free on the sole visible logical device. Native per-layer engine budget is 512 MiB. Maximum accounted layer footprint (native peak + CuPy reserved) was **598,551,912 bytes / 570.82 MiB**; head footprint **620,156,192 bytes / 591.43 MiB**. These exclude CUDA context/driver allocations; preflight separately measures actual free VRAM. Memory availability varied during the run and was not assumed constant. The vocabulary's original Q6_K upload is 521,472,000 packed bytes, not a decoded FP32 bank.

Common environment:

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_BACKEND=cupy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH=.:tests
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

**Actually executed**, each layer command bounded to 180 seconds, head to 120 seconds by the terminal runner (not pending commands). The equivalent shell-bounded form is:

```sh
timeout 180 .venv/bin/python tests/native_decoder_conditioned.py logs/wp7_conditioned_all_layers_v2 --start 0 --stop 8
timeout 180 .venv/bin/python tests/native_decoder_conditioned.py logs/wp7_conditioned_all_layers_v2 --start 8 --stop 16
timeout 180 .venv/bin/python tests/native_decoder_conditioned.py logs/wp7_conditioned_all_layers_v2 --start 16 --stop 24
timeout 180 .venv/bin/python tests/native_decoder_conditioned.py logs/wp7_conditioned_all_layers_v2 --start 24 --stop 32
timeout 180 .venv/bin/python tests/native_decoder_conditioned.py logs/wp7_conditioned_all_layers_v2 --start 32 --stop 40
timeout 180 .venv/bin/python tests/native_decoder_conditioned.py logs/wp7_conditioned_all_layers_v2 --start 40 --stop 48
timeout 120 .venv/bin/python tests/native_decoder_conditioned_head.py logs/wp7_conditioned_all_layers_v2
```

Do **not** rerun into these existing directories: the harness rejects existing layer/head directories and exclusively creates batch manifests. Choose a new output directory for a fresh experiment. Reporting is CPU-only and also exclusively creates its destination JSON:

```sh
timeout 120 .venv/bin/python tests/native_decoder_conditioned_report.py logs/wp7_conditioned_all_layers_v2 docs/native_engine/WP7_CONDITIONED_ALL_LAYERS.json
```

Executed validation:

- `python -m pytest -q tests/test_native_engine_conditioned.py -x --tb=short`: initial **51 passed in 2.51 s**; after adding published-summary provenance/coverage checks, final **52 passed in 2.47 s**, no skips. The JSON's initial validation record is intentionally retained; `validation.json` records the final result.
- Existing `tests/test_native_engine_divergence.py` plus `tests/test_native_engine_decoder.py::test_last_layer_failure_global_position` and `::test_factory_budget_preflight`, after memory preflight: **11 passed in 2.70 s**, no skips.
- `python -m py_compile` for the three new harnesses and new test file: passed.
- `git diff --check`: passed. New files are untracked until the parent elects to stage them; original user work remains untouched.

The new tests independently verify saved arrays against unchanged envelopes, exact rows/IDs, coverage, input/capture identities, six completed batches, head shape, artifact/source/library hashes and separate legacy/quality status. They fail, rather than skip, if required all-48 evidence is absent. Negative tests reject both a continuous mismatch and an allclose-compatible deviation in an exact-row contract.

## Limits

At two tokens, pooled QSA history has zero active rows: this checks empty logical state, **not** a pooled update, sparse-selection transition or long-context page behavior. Other expert placements and longer histories need their own validation. The native/Python conditioned comparison reuses existing components; it is not a new independent mathematical oracle for every lower-level primitive. Historical Python/llama.cpp evidence corroborates numerical sensitivity but is not substituted for this execution.

**Conclusion:** no unexplained same-input/history mismatch was found anywhere in this executed 48-layer/two-token sweep or final head. There is no evidence-supported production equation fix to make. Continue under the scoped conditioned-operation evidence, retaining strict full-trajectory failure and keeping longer-context/quality acceptance separate.
