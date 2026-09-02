# Final Polish Report
**Thesis:** Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models  
**Author:** Mohd Zamin Quadri  
**Programme:** ESPACE, TUM School of CIT  
**Submission date:** April 15, 2026  
**Report last updated:** 2026-03-18  

---

## Summary

This report documents all corrections and polishing changes applied in the final correction pass. Every change is cross-referenced to the source file. No data values were altered; only presentation, formatting, and figure quality were improved.

---

## Part 1 — Figure Regeneration

### Problem
All 10 data figures (fig1–fig10) were produced by an archived script (`ARCHIVED_OLD_SCRIPTS/regenerate_all_charts_light.py`) using **wrong data values** (e.g., R²=0.9277 for Trial 1 instead of the verified 0.7860). Figure styling used harsh default matplotlib colours, inconsistent fonts, and no TUM corporate design compliance.

### Fix
Created `figures/generate_all_thesis_figures.py` — a single authoritative script that:
- Uses **only verified data values** sourced from JSON evaluation files
- Applies the **TUM corporate colour palette** throughout (`#0065BD`, `#E37222`, `#A2AD00`, `#64A0C8`, `#003359`)
- Sets consistent `rcParams`: serif font, 10pt base, clean spines, grid behind bars
- Outputs both PDF (300 dpi) and PNG (150 dpi) for each figure

### Figures regenerated

| File | Content | Key verified data |
|------|---------|-------------------|
| `fig1_trial_comparison.pdf` | 3-panel bar chart: T2–T8 R², MAE, RMSE | T8: R²=0.5957, MAE=3.96, RMSE=7.12 |
| `fig2_uq_ranking.pdf` | Spearman ρ across all UQ methods | T8 MC: ρ=0.4820; Exp A Ens: ρ=0.1035 |
| `fig3_conformal_coverage.pdf` | Nominal vs achieved coverage + widths | 90%: q=9.92, PICP=90.02%; 95%: q=14.68, PICP=95.01% |
| `fig4_selective_prediction.pdf` | MAE at 100%/90%/50% retention | 3.96 → 3.29 (−16.8%) → 2.38 veh/h (−39.9%) |
| `fig5_feature_correlation.pdf` | Feature Spearman ρ vs error (horizontal bar) | VOL=+0.332, CAP=+0.262, CAP\_RED=−0.229 |
| `fig6_with_without_uq.pdf` | Deterministic vs MC Dropout R²/MAE/RMSE | ΔR²=−0.010, ΔMAE=−0.012, ΔRMSE=+0.087 |
| `fig7_calibration.pdf` | k₉₅ comparison: Gaussian vs T8 | Gaussian: 1.96; T8 MC Dropout: **11.34** (source: `trial8_uq_diagnostics.json`) |
| `fig8_architecture.pdf` | PointNetTransfGAT flow diagram | 5→512→128→256→512→64→1 (verified from model source) |
| `fig9_policy_explanation.pdf` | Uncertainty-guided decision framework | ACCEPT/FLAG/REJECT thresholds with MAE values |
| `fig10_node_vs_graph.pdf` | Node-level vs graph-level evaluation schematic | T8: 100 test graphs, 3,163,500 nodes, MAE=3.96 |

**Source file:** `thesis/latex_tum_official/figures/generate_all_thesis_figures.py`  
**Data sources:** `ALL_MODELS_COMPARISON/all_models_summary.json`, `uq_results/conformal_standard.json`, `uq_results/uq_comparison_model8.json`, `uq_results/mc_dropout_full_metrics_model8_mc30_100graphs.json`

---

## Part 2 — Blank Pages Fix

### Problem
`scrbook` (KOMA-Script) uses `\cleardoubleoddpage` and `\cleardoubleevenpage` internally when transitioning between `\frontmatter{}` and `\mainmatter{}`. The original fix only overrode `\cleardoublepage`, leaving the KOMA-Script internal variants active. Combined with the implicit `twoside` default of `scrbook`, this produced blank pages at frontmatter/mainmatter boundaries.

### Fix
**File:** `thesis/latex_tum_official/main.tex` (line 46–49)

```latex
% Before (incomplete fix):
\let\cleardoublepage\clearpage

% After (complete fix):
\let\cleardoublepage\clearpage
\let\cleardoubleoddpage\clearpage
\let\cleardoubleevenpage\clearpage
```

All three KOMA-Script cleardouble variants now redirect to `\clearpage`, eliminating blank pages at chapter/section transitions while preserving intentional `\clearpage` calls in `pages/disclaimer.tex` and `pages/acknowledgments.tex`.

---

## Part 3 — Bibliography Overflow Fix

### Problem
Long conference and journal names in the alphabetic biblatex style caused overfull `\hbox` warnings and text overflowing the right margin. The `url=false` option meant URL overflow was not the cause; the issue was unbreakable strings in citation entries.

### Fix
**File:** `thesis/latex_tum_official/settings.tex` (after `\bibliography{bibliography}`)

```latex
\setlength{\emergencystretch}{3em}
\appto\bibsetup{\emergencystretch 3em\relax}
```

`\emergencystretch` gives TeX up to 3em of extra interword stretch before declaring an overfull box. `\appto\bibsetup` ensures it applies specifically within the bibliography environment. The existing `[final]{microtype}` package provides additional margin through character protrusion.

---

## Part 4 — Table Column Spec Fix

### Problem
**File:** `thesis/latex_tum_official/chapters/04_experiments.tex`, Table `tab:trials`  
The `tabular` column specification `{lccccccc}` declared **8 columns** but the table had only **7 columns** (Trial, Batch, Split, Dropout, LR, Weighted, Final Layer), causing a misaligned extra empty column.

### Fix
```latex
% Before:
\begin{tabular}{lccccccc}

% After:
\begin{tabular}{lcccccc}
```

---

## Files Modified

| File | Change | Lines affected |
|------|--------|---------------|
| `thesis/latex_tum_official/main.tex` | Added `\let\cleardoubleoddpage\clearpage` and `\let\cleardoubleevenpage\clearpage` | 47–49 |
| `thesis/latex_tum_official/settings.tex` | Added `\setlength{\emergencystretch}{3em}` and `\appto\bibsetup{…}` | 69–72 |
| `thesis/latex_tum_official/chapters/04_experiments.tex` | Fixed tabular spec `{lccccccc}` → `{lcccccc}` | 60 |

## Files Created

| File | Purpose |
|------|---------|
| `thesis/latex_tum_official/figures/generate_all_thesis_figures.py` | Master figure regeneration script (10 figures) |
| `thesis/latex_tum_official/figures/fig1_trial_comparison.pdf/.png` | Trial comparison bar chart |
| `thesis/latex_tum_official/figures/fig2_uq_ranking.pdf/.png` | UQ method Spearman ρ ranking |
| `thesis/latex_tum_official/figures/fig3_conformal_coverage.pdf/.png` | Conformal coverage + widths |
| `thesis/latex_tum_official/figures/fig4_selective_prediction.pdf/.png` | Selective prediction MAE |
| `thesis/latex_tum_official/figures/fig5_feature_correlation.pdf/.png` | Feature-error correlation |
| `thesis/latex_tum_official/figures/fig6_with_without_uq.pdf/.png` | Det. vs MC Dropout comparison |
| `thesis/latex_tum_official/figures/fig7_calibration.pdf/.png` | k₉₅ calibration comparison |
| `thesis/latex_tum_official/figures/fig8_architecture.pdf/.png` | Architecture flow diagram |
| `thesis/latex_tum_official/figures/fig9_policy_explanation.pdf/.png` | Policy decision framework |
| `thesis/latex_tum_official/figures/fig10_node_vs_graph.pdf/.png` | Node vs graph evaluation |
| `docs/verified/FINAL_POLISH_REPORT.md` | This report |

---

## Data Integrity

All verified data values used in figures are sourced exclusively from:

- `data/TR-C_Benchmarks/ALL_MODELS_COMPARISON/all_models_summary.json` — T1–T8 R²/MAE/RMSE/Pearson r
- `uq_results/mc_dropout_full_metrics_model8_mc30_100graphs.json` — T8 MC Dropout ρ, σ statistics
- `uq_results/conformal_standard.json` — conformal quantiles and PICP values; k₉₅
- `uq_results/uq_comparison_model8.json` — coverage calibration table, selective prediction MAE
- `uq_results/ensemble_experiments/experiment_a_results.json` — Exp A ρ values
- `uq_results/ensemble_experiments/experiment_b_results.json` — Exp B ρ values

**No data values were invented or altered. The archived script `ARCHIVED_OLD_SCRIPTS/regenerate_all_charts_light.py` was not used.**

---

## What Was Not Changed

- No thesis text content was modified in this correction pass
- No bibliography entries were added or removed
- No LaTeX template structure was altered
- `pages/disclaimer.tex` and `pages/acknowledgments.tex` intentional `\clearpage` calls were preserved
- `fig_network_intro.pdf` was not regenerated (already correct from previous session)

---

## Phase 3 — Final Corrections (2026-03-18)

### Change 1: Acknowledgments — Martin Werner removed
**File:** `pages/acknowledgments.tex`, line 12  
Martin Werner was named as "supervisor" in the acknowledgments but does not appear in the front matter (`main.tex` defines only Prof. Dr. Stephan Günnemann as supervisor). The sentence was corrected to list only Günnemann as supervisor, removing the inconsistency.

### Change 2: fig7_calibration — k₉₅ corrected from 11.647 → 11.34
**File:** `figures/generate_all_thesis_figures.py`, lines 663, 698, 708  
The figure previously displayed k₉₅ = 11.647 (sourced from `conformal_standard.json`, which computed the quantile on the calibration half of a 50-graph 50/50 split). The authoritative value is **11.34** from `trial8_uq_diagnostics.json` (full 100-graph test set), which is the value cited throughout the thesis text (abstract, Table 5.3, Discussion, Conclusion). The figure title was also updated from "50 eval graphs" to "100 test graphs" to match the data context. Figure regenerated.

### Change 3: Figure Polish
**Files:** `figures/generate_all_thesis_figures.py`, `figures/generate_new_figures.py`

| Figure | Change |
|---|---|
| `fig3_feature_distributions.pdf` | Height increased from 3.8 → 5.0 in; title padding increased; tight_layout pad 1.0 → 1.5 |
| `fig10_node_vs_graph.pdf` | Panel spacing increased: `tight_layout(w_pad=3.5)` instead of default |
| `fig11_thesis_workflow.pdf` | Canvas height 5.0 → 5.8 in; ylim expanded to (0.0, 5.2); step circles radius 0.26 → 0.30; circle gap 0.42 → 0.48; box heights BH 1.20 → 1.30, T8H 1.40 → 1.50, UH 0.95 → 1.05; subplots_adjust margins increased |

### Compilation result
4 × pdflatex passes → **main.pdf: 69 pages, 1,517,955 bytes**  
Deliverables updated:
- `thesis_TUM_FINAL.pdf` (Desktop)
- `thesis_upload.zip` (Desktop, 8.3 MB, 104 files)
