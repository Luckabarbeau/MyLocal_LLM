# Research artifacts

This directory retains dated research sources and rendered PDFs. These are useful for provenance and design context, but they are **not automatically the current architecture, production status or operating procedure**. Current status and change history live in [`../CURRENT_BASELINE.md`](../CURRENT_BASELINE.md) and [`../DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md).

## Retained LaTeX/PDF artifacts

| Source | Rendered artifact | Status |
|---|---|---|
| [`persistent_soft_plan_0080.tex`](persistent_soft_plan_0080.tex) | [`persistent_soft_plan_0080.pdf`](persistent_soft_plan_0080.pdf) | **Historical 0080 research report.** Dated 2026-10-14; preliminary small-model soft-plan/block-proposal evidence, not peer reviewed and not a current Model B result. The report itself says the ~500M integration was not evaluated there for quality or production throughput. |
| [`0090B_modular_residual_specialist_combination.tex`](0090B_modular_residual_specialist_combination.tex) | [`0090B-specialist-combiner-preview.pdf`](0090B-specialist-combiner-preview.pdf) | **Historical/living design note.** Dated 2026-10-26; separates earlier probability-fusion evidence from the proposed specialist combiner. Its 338M foundation description predates protected Model B. A trained multi-specialist ensemble was not demonstrated by this artifact. Current specialist code evolved through 0105 and 0107–0108. |

The PDF files are retained snapshots. Some historical citations inside the 0090B source name Markdown reports that were later consolidated and removed; resolve them through the migration ledger in `DEVELOPMENT_LOG.md` and Git history. Treat each `.tex` file as editable source, but update its own evidence/version boundary if it is revised. A paper/design edit does not replace a required change-log entry.

## Building the sources

Build outside the repository so auxiliary files do not pollute `docs/research/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=/tmp/0080-build \
  docs/research/persistent_soft_plan_0080.tex

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=/tmp/0090b-build \
  docs/research/0090B_modular_residual_specialist_combination.tex
```

No LaTeX build was run during the documentation consolidation.

## Historical reports

The former 0058/0080 Markdown audit and implementation reports were consolidated into [`../DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md) and removed from the working tree. Do not add another patch-report file here. Recover full historical text through the migration ledger and Git when needed, for example:

```bash
git show a27487b:docs/research/IMPLEMENTATION_0080.md
```
