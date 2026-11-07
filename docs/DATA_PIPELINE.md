# Data pipeline

This guide documents the current tokenizer, packed-shard, mixture-manifest, and
Common Pile long-form workflow used by the Model B run family.

The central rule is that **tokenizer bytes, vocabulary, EOS identity, split
policy, shard files, and manifest are one data artifact**. A checkpoint does not
make an arbitrary tokenizer or similarly named shard directory compatible.

## 1. Identity rules

### Tokenizer identity

For any run family, record and validate all of the following:

- the exact `tokenizer.json` bytes and SHA-256;
- the vocabulary size produced by loading that file;
- the EOS token string exposed by the tokenizer and its derived token ID;
- tokenizer-training source/config identity, sample policy, split seed, and
  validation ratio; and
- the model/checkpoint configuration that consumes the vocabulary.

Never infer EOS as the last vocabulary entry and never copy a numeric EOS from
another model. Derive it from the tokenizer, then compare it with the manifest:

```python
import hashlib
import json
from pathlib import Path

from train_model import load_tokenizer_file, tokenizer_eos_id

root = Path("token_shards_ensemble_32k_mixed")
tokenizer_path = root / "tokenizer.json"
manifest = json.loads((root / "mixture_manifest.json").read_text())
tokenizer = load_tokenizer_file(tokenizer_path)

actual_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
actual_eos = tokenizer_eos_id(tokenizer)

assert actual_sha == manifest["tokenizer_sha256"]
assert len(tokenizer) == manifest["vocab_size"]
assert actual_eos == manifest["eos_token_id"]
print(actual_sha, len(tokenizer), actual_eos)
```

### Mixture identity

The production loader accepts `weighted_packed_mixture_v1`. It requires:

- `mixture_manifest.json` at the root;
- `tokenizer.json` at the same root;
- a matching tokenizer SHA when the manifest supplies one;
- `complete` not set to false;
- a non-empty source map;
- at least one `train_shard_*.bin` and `val_shard_*.bin` per source; and
- actual train/validation shard counts equal to the manifest counts.

Source sampling probabilities come from manifest weights and are normalized at
load time. They are not inferred from corpus bytes, token counts, or number of
shards. Reweighting therefore changes sampling without duplicating or
retokenizing data.

The loader does not currently store/verify a digest for every `.bin` shard.
Treat shard files as immutable: an in-place binary mutation can violate identity
even if the manifest count still matches.

### Strict training resume

A checkpoint records tokenizer SHA, local mixture-manifest SHA, effective source
weights, source counters/cursors, RNG state, and other data provenance. For an
exact resume, preserve the exact tokenizer, manifest bytes, source set, source
paths, and shard files.

The trainer rejects tokenizer changes. It can treat a changed local manifest or
stream registry as a recorded new data phase while preserving model/optimizer
state, but that is a deliberate continuation on changed data, **not** a strict
resume. Do not reweight, integrate new sources, rewrite the manifest, or replace
shards before a strict resume.

## 2. Pinned long-form Model B run-family identity

The current long-form mixture extension is pinned by
`configs/common_pile_longform_v01.json` to:

| Field | Pinned value |
|---|---|
| Tokenizer SHA-256 | `20781a3078c60fd5a7634fba801b0a76943e5f993f0b1eaa61b0a4238922f2ba` |
| Vocabulary size | 32,768 |
| EOS token ID | 1 |
| Token dtype | `uint16` |
| Validation ratio | 0.01 |
| Split seed | 42 |

These values belong to **this tokenizer/mixture/run family only**. They are not
universal MyLocal_LLM values and must not be applied to another tokenizer,
checkpoint, model preset, or newly trained mixture merely because it also has a
32K vocabulary. In generic code, derive EOS from the tokenizer and validate the
manifest as shown above.

The Common Pile tooling refuses to prepare or integrate a source unless the
existing mixture tokenizer's SHA, loaded vocabulary size, and derived EOS all
match the pinned configuration.

## 3. Create a new mixed-shard run family

Use `prepare_ensemble_mixture.py` for the current multi-source workflow. It is
the sole public entry point for shared-tokenizer training, per-source packed
shards, resumable source-level orchestration, and the root mixture manifest.
The former context-truncating single-corpus generator and standalone tokenizer
trainer were removed to avoid maintaining incompatible parallel pipelines.

Choose a **new output root** for a new tokenizer family. Do not train a new
tokenizer inside an existing mixture or checkpoint family.

### 3.1 Define the source mixture

`configs/pretraining_coding_mix.json` is the current example schema. Each source
has a relative raw-data path plus independent tokenizer-training and pretraining
weights. Preserve the config file and its SHA with run records.

### 3.2 Train the shared tokenizer once

```bash
RAYON_NUM_THREADS=16 .venv/bin/python prepare_ensemble_mixture.py tokenizer \
  --data-root ../pretraining_data \
  --mix-config configs/pretraining_coding_mix.json \
  --output-root /absolute/path/to/new_mixture_root \
  --vocab-size 32768 \
  --sample-documents 200000 \
  --threads 16 \
  --seed 42 \
  --val-ratio 0.01
```

The tokenizer command:

- samples training-only documents according to `tokenizer_weight`;
- excludes documents assigned to validation by the stable split function;
- refuses to replace an existing `tokenizer.json`;
- requires the trained vocabulary to equal the requested size; and
- writes `tokenizer_training.json` with tokenizer SHA, source weights,
  mix-config SHA, sample count, seed, and validation ratio.

Train once, then treat `tokenizer.json` as immutable. A new tokenizer means a new
run family and requires retokenizing every source.

### 3.3 Pack each source

```bash
.venv/bin/python prepare_ensemble_mixture.py shards \
  --data-root ../pretraining_data \
  --mix-config configs/pretraining_coding_mix.json \
  --output-root /absolute/path/to/new_mixture_root \
  --workers 8 \
  --pending-batches 16 \
  --batch-documents 256 \
  --shard-size-mb 256 \
  --seed 42 \
  --val-ratio 0.01
```

Packing uses the shared tokenizer and the repository's single
`pack_text_corpus` engine. It preserves full documents, appends one tokenizer-
derived EOS per document, assigns a whole document to train or validation using
the stable text-hash/seed convention, writes contiguous `uint16` streams, and
keeps each source physically separate. The trainer later samples source names by
manifest weight.

For staged processing, repeat `--source NAME`. Completed sources are skipped
when running the full command. `--max-documents-per-source` is for a separate
small-data smoke root; omit it for production.

After each source completes, the orchestrator atomically rewrites
`mixture_manifest.json`, records source token/shard counts, and updates
`missing_sources`. A partial mixture remains `complete: false` and is rejected by
the production loader.

### 3.4 Validate the complete root

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from mini_llm.data.mixed_shards import load_weighted_shard_sources

root = Path("/absolute/path/to/new_mixture_root")
sources, manifest = load_weighted_shard_sources(root)
print("format:", manifest["format_version"])
print("tokenizer SHA:", manifest.get("tokenizer_sha256"))
for source in sources:
    print(source.name, source.weight, len(source.train_paths), len(source.val_paths))
PY
```

Only after this validation should the root be supplied to
`train_model.py --mixed-shard-root`.

### 3.5 Reweight without retokenizing

```bash
.venv/bin/python prepare_ensemble_mixture.py reweight \
  --data-root ../pretraining_data \
  --mix-config configs/pretraining_coding_mix.json \
  --output-root /absolute/path/to/new_mixture_root
```

Reweighting requires the source set to remain unchanged and only rewrites
sampling weights. It creates a new manifest identity. Do not do this before an
exact checkpoint resume; either keep the old manifest or explicitly treat the
next training invocation as a changed-data phase.

## 4. Add the pinned Common Pile long-form sources

This workflow extends an **existing** compatible mixture root. For the current
run family, commands below use `token_shards_ensemble_32k_mixed`. Raw files stay
compressed under `../pretraining_data/common_pile_v01_filtered`; the downloader
uses pinned dataset revisions and does not create a duplicate Hugging Face cache.

Before starting, verify that the root tokenizer has the pinned identity in
Section 2. Do not run these commands against a newly trained tokenizer family
unless its explicitly configured identity matches.

### Priority 1

From the repository root:

```bash
mkdir -p ../pretraining_data
./scripts/download_common_pile_longform.sh priority1 ../pretraining_data

.venv/bin/python prepare_common_pile_longform.py prepare-stage \
  --stage priority1 \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed \
  --workers 12 \
  --batch-documents 16 \
  --pending-batches 24

.venv/bin/python prepare_common_pile_longform.py integrate \
  --stage priority1 \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed

.venv/bin/python prepare_common_pile_longform.py report \
  --stage priority1 \
  --output-root token_shards_ensemble_32k_mixed
```

Priority 1 adds:

- Project Gutenberg filtered;
- DOAB filtered;
- Pressbooks filtered; and
- UK Hansard filtered.

The configured long-form fraction is 10%. Existing source weights are
proportionally scaled to 90%, and the four long-form weights sum to 10%.

### Priority 2 and full integration

Run only after Priority 1 preparation, integration, loader validation, and report
inspection succeed:

```bash
./scripts/download_common_pile_longform.sh priority2 ../pretraining_data

.venv/bin/python prepare_common_pile_longform.py prepare-stage \
  --stage priority2 \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed \
  --workers 12 \
  --batch-documents 16 \
  --pending-batches 24

.venv/bin/python prepare_common_pile_longform.py integrate \
  --stage full \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed

.venv/bin/python prepare_common_pile_longform.py report \
  --stage full \
  --output-root token_shards_ensemble_32k_mixed
```

Priority 2 adds pre-1929 books and USGPO. The full configured long-form fraction
is 12%; all six long-form source weights sum to that fraction and legacy source
weights are proportionally scaled to 88%.

To prepare or restart one source, repeat `--source` as supported by the CLI:

```bash
.venv/bin/python prepare_common_pile_longform.py prepare-stage \
  --stage priority1 \
  --source long_doab \
  --data-root ../pretraining_data \
  --output-root token_shards_ensemble_32k_mixed
```

Use `--max-documents` only with a separate smoke-test output root. Capped output
is marked `production_complete: false` and cannot be integrated.

## 5. Common Pile reconstruction and provenance

The pipeline reconstructs only records with a stable parent/order contract:

- DOAB sections are grouped by book identity and ordered by `section_idx`;
- Pressbooks chapters retain source/provenance order within `book_url`;
- USGPO `-ptN` records are grouped by parent and ordered numerically; and
- row-mode sources remain one input row per document.

A reconstructed parent must be contiguous and monotonically ordered. Unrelated
records are never joined. For every completed source, retain:

- `source_manifest.json` — dataset ID, pinned revision, raw files, license,
  reconstruction rule, tokenizer identity, split policy, counts, and completion;
- `provenance.jsonl.gz` — document/source identity, split, token length,
  normalized-text SHA, and source metadata;
- `duplicates.jsonl.gz` — rejected duplicate identity and its first owner;
- `length_stats.json` — token-length distribution and long-document counts; and
- the root `common_pile_longform_dedup.sqlite3` deduplication database.

Do not discard these files after packing. Shards without provenance are not an
equivalent data artifact.

## 6. Deduplication rules

Common Pile exact deduplication hashes the complete reconstructed document after:

1. Unicode NFKC normalization;
2. CRLF/CR newline normalization to LF; and
3. leading/trailing whitespace removal.

The SHA-256 database is shared across all configured long-form sources, so the
first accepted document owns the hash and later exact duplicates are logged and
skipped. This is exact normalized-text deduplication, not fuzzy, semantic, or
near-duplicate detection. Do not describe it as removing paraphrases or similar
passages.

The train/validation split is computed on the whole document using the existing
text/seed convention. A document cannot be split across train and validation.
Documents are not truncated, and one derived EOS is appended after tokenization.

## 7. Interruption and publication rules

### Downloads

The download script:

- uses dataset revisions pinned in the Common Pile config;
- checks free disk with an additional reserve;
- writes `*.part` files;
- resumes with `curl --continue-at -` and retries; and
- renames a complete download to its final gzip filename.

The preparer refuses raw sources with missing/empty files or remaining `*.part`
files.

### Source preparation

A source is built in `<source>.partial`. A completed source is atomically renamed
to `<source>` only after:

- all tokenization futures finish;
- train and validation writers close and both produced shards;
- provenance, duplicate, and length reports are complete;
- `source_manifest.json` is written atomically;
- deduplication changes commit; and
- the source is marked production-complete when uncapped.

If preparation is interrupted, the partial directory is never registered.
Inspect the failure, then remove **only** that source's `.partial` directory and
restart it. Completed sources with a manifest are skipped safely. Do not delete a
completed source merely to restart another source.

The deduplication transaction rolls back on a failed source, so an uncommitted
partial source does not permanently claim hashes.

### Manifest integration

Common Pile `integrate` requires every requested source to be complete and to
match the pinned tokenizer. It:

1. rescales existing source weights;
2. inserts complete long-form entries and provenance references;
3. creates a timestamped backup of the previous root manifest;
4. atomically replaces `mixture_manifest.json`; and
5. validates the result through `load_weighted_shard_sources()`.

Integration changes the root manifest SHA and therefore changes training data
identity. Never integrate while another process expects strict resume against
the old manifest. Keep the timestamped backup with the associated checkpoint
family.

## 8. Data-to-checkpoint handoff

Before launching Model B training:

1. validate the mixture through `load_weighted_shard_sources()`;
2. record the tokenizer SHA, manifest SHA, source weights, shard counts, and raw
   source revisions;
3. derive EOS from `tokenizer.json` and compare it with the manifest;
4. verify the model configuration vocabulary equals the manifest/tokenizer;
5. ensure the checkpoint and log destinations are new; and
6. keep the mixture immutable for strict resume.

`train_model.py` copies `config.json`, `tokenizer.json`, and available mixture
metadata into the checkpoint directory. Training checkpoints additionally store
tokenizer/manifest provenance, effective weights, source counters, RNG state,
and mixture-phase history. Those copies support auditing; they do not authorize
changing or reconstructing the live shard root during an exact run.
