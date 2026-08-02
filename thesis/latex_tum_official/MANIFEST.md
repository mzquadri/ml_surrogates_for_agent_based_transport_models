# Submission Manifest

## Thesis: Uncertainty Quantification for Machine Learning Models in Transportation Policy Analysis
## Author: Mohd Zamin Quadri
## M.Sc. Mathematics in Science and Engineering, TUM

This directory contains the **final** thesis document (source + compiled artifacts), synced from the author's final submission folder.

### Contents

- `main.tex`, `settings.tex`, `pages/`, `chapters/` — LaTeX source (final version, including `chapters/appendix_a_master_table.tex`).
- `figures/new/` — final thesis figures (PDF + PNG).
- `bibliography.bib` — bibliography (biber).
- `main.pdf` — compiled thesis (May 15, 2026).
- `Zamin_Quadri_Master_Thesis.docx` — Word export of the final thesis.

### Compilation

- Compiler: pdflatex + biber (NOT bibtex)
- Passes: pdflatex -> biber -> pdflatex -> pdflatex
- Build: `latexmk -pdf main.tex` (see `.latexmkrc`)

### Verification

The numerical claims in this document were independently audited. See:

- `docs/verified/AUDIT_SUMMARY.md` — final audit of all 10 UQ methods (0 bugs, 0 re-runs required).
- `docs/verified/VERIFIED_RESULTS_MASTER.csv` — master result table.
- `results/` — canonical result JSONs, prediction archives (.npz), and per-trial metrics.
- `models/` — the 16 trained model checkpoints referenced by the thesis.
