# Persistent Soft-Plan Decoding (Architecture 0080)

Repository-ready technical report **v0.1**, dated **2026-10-14**.

- `persistent_soft_plan_0080.tex` — editable, self-contained LaTeX source, figures and tables.
- `persistent_soft_plan_0080.pdf` — compiled 15-page reading copy.

## Build

From this directory, with TeX Live and `latexmk` installed:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error persistent_soft_plan_0080.tex
```

Or on Overleaf, upload the `.tex` file and compile with pdfLaTeX. There is no external `.bib` file or image asset: citations, TikZ diagrams, and PGFPlots figures are embedded in the source. The report contains the code-path map, Windows PowerShell evaluation commands, and experimental limitations.

## Scope and data provenance

The reported numerical results are based on the project's 7M-class CPU research runs and console logs through expanded-corpus step 2,500; the locally trained checkpoints were not independently available to the report-preparation environment. The report does not claim matched autoregressive improvements, production GPU speedup, or high-quality free-running generation. The `~500M` model is explicitly treated as ongoing separate research.

The GitHub source link and authorship are included in the report. The note is a working technical report, not a peer-reviewed publication. Increment the version/date when incorporating new independently verified experiments.
