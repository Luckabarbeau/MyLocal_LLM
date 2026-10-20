# Production block-proposal checkpoint benchmark

This is a **read-only benchmark** of the main `train_model.py --objective block-proposal`
checkpoint. It uses the main routed Transformer, proposal head, and prefix decoder;
**no model presets, preprocessing tools, or research example scripts are added**.
The `--ablations` setting is optional because it is much more expensive than ordinary
validation CE, especially for 4K target windows and a 64K memory horizon.

## Setup

From the repository root of the trained production model:

```bash
export MINI_LLM_BACKEND=cupy
export MINI_LLM_LM_HEAD_CHUNK_TOKENS=256
mkdir -p benchmarks/results
python -m pytest -q tests/test_block_proposal_production_benchmark.py
```

Check your checkpoint path; it must contain `training_state.json`, `config.json`,
`tokenizer.json`, and `model_params/`. It must have `prediction_objective=block-proposal`.
For a live checkpoint, make a consistent frozen copy between saves.

## 1. Per-source held-out validation (fast mode)

```bash
python -m benchmarks.evaluate_block_proposal_production \
  --checkpoint ./checkpoints/your_500m_block_checkpoint \
  --mixed-shard-root ./data/your_existing_mixture \
  --mode heldout --batches-per-source 1 --batch-size 1 --group-size 1 \
  --seed 78078 \
  --output ./benchmarks/results/500m_heldout_smoke.json
```

Includes proposal/final CE and gains at each K-token horizon, weighted by actual
valid targets. Reports fixed validation-sample fingerprint per dataset. Increase
`--batches-per-source` for reliable metrics; held-out sources are never merged.

## 2. Cross-position ablations (expensive; same held-out split)

```bash
python -m benchmarks.evaluate_block_proposal_production \
  --checkpoint ./checkpoints/your_500m_block_checkpoint \
  --mixed-shard-root ./data/your_existing_mixture \
  --mode heldout --ablations --batches-per-source 1 --batch-size 1 --group-size 1 \
  --seed 78078 \
  --output ./benchmarks/results/500m_ablations_smoke.json
```

Five inference-time plan interventions: `all`, `own_only`, `past_and_current`,
`all_except_current`, `neutral`. Compare `all` versus `own_only` for other-position
information; `all` versus `past_and_current` for future proposals. Only the plan
embeddings change; genuine previous-token prefixes and original proposal logits
remain unchanged. This is **not** equivalent to retraining under each condition.

## 3. WikiText-2 block-refined perplexity

```bash
python -m benchmarks.evaluate_block_proposal_production \
  --checkpoint ./checkpoints/your_500m_block_checkpoint \
  --mode wikitext --wiki-test ../benchmarks/wikitext-2-raw/wiki.test.raw \
  --document-mode article --max-wiki-tokens 1024 \
  --output ./benchmarks/results/500m_wikitext_smoke.json
```

Use `--max-wiki-tokens 0` for full WikiText scoring. Unlike the existing
trunk-only `evaluate_wikitext.py`, the metric here evaluates the complete
block-proposal + prefix decoder objective. It is teacher-forced fixed-block
perplexity in **nats/target** (`exp(CE)`), not free-running sequence likelihood
and not directly comparable across different tokenizers or block-alignment rules.

## 4. Actual two-stage generation (greedy correctness reference)

```bash
python -m benchmarks.evaluate_block_proposal_production \
  --checkpoint ./checkpoints/your_500m_block_checkpoint \
  --mode generation --prompt 'The purpose of numerical simulation is' \
  --new-tokens 32 --commit-stride 4 \
  --output ./benchmarks/results/500m_generation_smoke.json
```

Generation uses the production `InferenceModel` + `SequentialCommitBlockGenerator`,
not the separately trained AR head. Main cache updates are causal and performed
one committed token at a time; no K-fold decoding-speed claim is made.

## Important constraints

- **Read-only**: no optimizer is loaded, no training update occurs, checkpoint files
  are untouched, and outputs refuse to overwrite existing reports.
- Checkpoint weights/config/tokenizer and (when supplied) mixture provenance are
  checked strictly to prevent silently scoring the wrong architecture.
- The 500M 64K model can be expensive even for a small number of validation
  batches. Start with one batch and expand cautiously based on VRAM and time.
- Comparisons across checkpoints require identical dataset manifest, seed,
  configuration and `validation_sample_sha256`.

## 0080X validation-parity audit

The original standalone 0080W held-out benchmark forced `use_retrieval=True`
for all document fragments, including partial-target windows where the
production validator disables retrieval. The default benchmark now matches
production eligibility (`--retrieval-policy production`). For a paired
comparison on identical targets, pass `--retrieval-policy force` to reproduce
the old behavior and save a DIFFERENT output filename.

`--sampling-policy document_first` (default) is the representative production
validation/training policy. `--sampling-policy legacy` matches the fixed
per-source validation sampler, which preferentially selects documents with
sufficient target/history lengths. These are different validation populations.

`--router-stochastic` evaluates seeded training-like Gumbel selection,
using the checkpoint step to compute the annealed router temperature.
Its document sampler RNG is unchanged, so the benchmark's held-out inputs and
labels are identical to the deterministic run for the same source and seed.
It does not reproduce the exact training Gumbel noise realization.

A difference between training and validation cross-entropy is **not**, by
itself, evidence of a target-label or optimizer bug. First compare fixed
held-out windows under deterministic and stochastic routing, then compare
legacy and document-first samplers. Benchmark and production validation both
call `BlockProposalObjective.evaluate_batch`; if matching metadata on the
same examples still produces different results, investigate checkpoint state,
compute-buffer refresh, and kernel-specific numeric paths.
