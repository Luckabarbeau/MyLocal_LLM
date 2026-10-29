# Common Pile v0.1 long-form preparation

This workflow is additive. Raw files stay compressed under `../pretraining_data/common_pile_v01_filtered`; generated sources are added under `token_shards_ensemble_32k_mixed` without changing or deleting existing shards. It is pinned to the existing 32,768-token tokenizer SHA-256 `20781a3078c60fd5a7634fba801b0a76943e5f993f0b1eaa61b0a4238922f2ba`.

## Priority 1

From `MyLocal_LLM`:

```bash
chmod +x scripts/download_common_pile_longform.sh
./scripts/download_common_pile_longform.sh priority1 ../pretraining_data

python prepare_common_pile_longform.py prepare-stage \
  --stage priority1 \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed \
  --workers 12 --batch-documents 16 --pending-batches 24

python prepare_common_pile_longform.py integrate \
  --stage priority1 \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed

python prepare_common_pile_longform.py report \
  --stage priority1 --output-root token_shards_ensemble_32k_mixed
```

The `priority1` mixture assigns 10% total sampling probability to the four long-form sources and proportionally scales the live weights of all existing sources to 90%. No retokenization is needed to adjust weights: edit `configs/common_pile_longform_v01.json` and rerun `integrate`.

## Priority 2

Run only after Priority 1 tokenization, integration, and report validation succeed:

```bash
./scripts/download_common_pile_longform.sh priority2 ../pretraining_data

python prepare_common_pile_longform.py prepare-stage \
  --stage priority2 \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed \
  --workers 12 --batch-documents 16 --pending-batches 24

python prepare_common_pile_longform.py integrate \
  --stage full \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed

python prepare_common_pile_longform.py report \
  --stage full --output-root token_shards_ensemble_32k_mixed
```

The full mixture assigns 12% total sampling probability to all six sources. Each source weight is independently configurable.

## Operational behavior

- Downloads use pinned revisions, `curl --continue-at -`, retries, and direct local files, avoiding extraction and a duplicate Hugging Face cache.
- DOAB sections are ordered by `section_idx`; Pressbooks chapters retain source/provenance order within `book_url`; USGPO `-ptN` records are ordered numerically. Unrelated records are never joined.
- Exact SHA-256 deduplication uses NFKC/newline-normalized complete documents and persists across all six sources in `common_pile_longform_dedup.sqlite3`.
- Documents are assigned wholly to train or validation with the existing SHA-256 text/seed convention. Every document receives one EOS token; documents are never truncated.
- `provenance.jsonl.gz`, `duplicates.jsonl.gz`, `source_manifest.json`, and `length_stats.json` are retained per source.
- If tokenization is interrupted, the source remains as `<source>.partial` and is never registered. Remove only that partial directory before restarting that source. Completed sources are skipped safely.
- `integrate` requires every source in the requested stage to be complete, creates a timestamped manifest backup, updates the root manifest atomically, and validates it through the trainer's production loader.

Use `--source long_doab` with `prepare-stage` to run or restart one selected source. Use `--max-documents` only with a separate smoke-test output root; capped output is deliberately marked non-production and cannot be integrated.
