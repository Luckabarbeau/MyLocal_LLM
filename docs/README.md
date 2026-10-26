# Documentation map

Use this page to distinguish current operational documentation from research
history. Historical patch snapshots are preserved for provenance and should not
be treated as required setup steps.

## Current guides

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project overview, architecture, and installation |
| [`USER_GUIDE.md`](USER_GUIDE.md) | Training, resume, inference, and runtime controls |
| [`EXTENDED_TRAINING.md`](EXTENDED_TRAINING.md) | Extended training workflow |
| [`TOKENIZER_README.md`](TOKENIZER_README.md) | Tokenizer training and artifacts |
| [`DEBUG_COMMANDS.md`](DEBUG_COMMANDS.md) | Focused diagnostics commands |
| [`FUTURE_DEVELOPMENT.md`](FUTURE_DEVELOPMENT.md) | Planned research directions |
| [`../tests/README.md`](../tests/README.md) | Test categories and safe test commands |
| [`../benchmarks/README.md`](../benchmarks/README.md) | Evaluation and benchmark workflows |

## Research documentation

- [`research/`](research/) contains active predictive and probability-combination
  design reports and experiment audits.
- [`research/0090B_modular_residual_specialist_combination.tex`](research/0090B_modular_residual_specialist_combination.tex)
  is the living LaTeX research note for frozen-foundation residual specialists,
  learned combination, per-specialist historical memory, and the evidence ledger.

## Operational entry points

Stable entry points remain at repository root because tests and existing runbooks
address them there:

- `train_model.py` — main training CLI.
- `train_sft.py` — supervised fine-tuning CLI.
- `inference.py` — inference CLI.
- `generate_token_shards.py` and `generate_packed_cosmopedia_shards.py` — data preparation.
- `benchmarks/run_checkpoint_evaluations.sh` and `benchmarks/run_fusion_evaluations.sh` — checkpoint evaluation.
- `run_training.sh` — the single configurable training launcher.

Run the launcher from the repository root. It requires an explicit
`CHECKPOINT_DIR`, refuses unsafe fresh/resume combinations, and accepts additional
`train_model.py` options after `--`.

## Runtime data

The following are generated state, not source organization targets:

- `checkpoints/`
- `logs/`
- `token_shards*/`
- `runtime/` — bounded streaming caches and optional generated snapshots
- `benchmarks/results/`

Never delete or relocate these directories as part of routine cleanup. Before
running GPU tests or benchmarks, check `nvidia-smi` and confirm that no important
training or inference process owns the device.
