# Documentation

This is the authoritative documentation index. Current behavior belongs in the focused guides below; historical implementation decisions and negative results belong in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).

## Current guides

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project overview, setup, current architecture, and safe quickstart |
| [`CURRENT_BASELINE.md`](CURRENT_BASELINE.md) | Exact Model B architecture, protected run status, and pinned production profile |
| [`CHECKPOINT_COMPATIBILITY.md`](CHECKPOINT_COMPATIBILITY.md) | Strict resume, weights-only initialization, checkpoint transactions, and foundation-bound specialists |
| [`USER_GUIDE.md`](USER_GUIDE.md) | Current training, resume, inference, serving, and specialist operations |
| [`MODEL_CENTERED_DEVELOPER_GUIDE.md`](MODEL_CENTERED_DEVELOPER_GUIDE.md) | Model, generation, training, loading, execution, and specialist API boundaries |
| [`DATA_PIPELINE.md`](DATA_PIPELINE.md) | Tokenizers, packed shards, manifests, Common Pile preparation, and provenance |
| [`FUTURE_DEVELOPMENT.md`](FUTURE_DEVELOPMENT.md) | Short forward-looking roadmap and open acceptance gates |
| [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) | Single durable human/agent change log and historical filename migration ledger |
| [`research/README.md`](research/README.md) | Retained LaTeX/PDF research artifacts and their historical status |
| [`../benchmarks/README.md`](../benchmarks/README.md) | Current evaluation and benchmark entry points |
| [`../tests/README.md`](../tests/README.md) | Test categories, Model B regression coverage, and safe CPU/GPU workflows |

## Documentation policy

Humans and coding agents must record durable architecture, behavior, compatibility, runtime-default, data-policy, or operational-safety changes in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md), newest first, using its mandatory template.

Do **not** create another numbered patch-report Markdown file. Update the applicable current guide and the development log instead. Full historical reports removed during consolidation remain recoverable through the migration ledger and Git history.

## Protected run boundary

The external Model B run is user-reported at step 165 as of 2026-11-04. Its checkpoint and host are not available in this workspace, so that status is not independently verified here. Do not invent paths, claim checkpoint validation, or alter architecture/training contracts based on assumptions about the unavailable artifact.

## Runtime and generated-data safety

These paths contain generated state or potentially irreplaceable work, not documentation-cleanup targets:

- `checkpoints/`
- `logs/`
- `data/` and external datasets
- `token_shards*/`
- `runtime/`
- `benchmarks/results/`
- local pretrained/external model files

Never delete, move, rename, overwrite, or “tidy” them during routine code or documentation work. Do not write smoke tests into an active checkpoint directory. Use a fresh destination for migration/resume tests and verify the path first.

Before GPU tests or benchmarks, inspect `nvidia-smi`. Do not compete with a training, inference, or serving process. A blocked or out-of-memory gate is not a pass.
