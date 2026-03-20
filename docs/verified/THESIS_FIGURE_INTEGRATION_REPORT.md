# THESIS_FIGURE_INTEGRATION_REPORT.md
**Thesis:** Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models  
**Author:** Mohd Zamin Quadri (Nazim) — TUM Master Thesis 2026  
**Generated:** 2026-03-14

---

## Status Summary

| Task | Status |
|---|---|
| Verified figures generated | ✅ Done |
| Old wrong figures flagged | ✅ Done (WARNING txt file) |
| 04_experiments.tex rewritten | ✅ Done |
| 05_results.tex rewritten | ✅ Done |
| 06_discussion.tex rewritten | ✅ Done |
| 07_conclusion.tex rewritten | ✅ Done |
| 01_introduction.tex fixed | ✅ Done |
| 03_methodology.tex fixed | ✅ Done |
| 02_background.tex checked | ✅ Clean — no wrong values |

---

## Verified Figures Generated

All figures are in `thesis/latex_tum_official/figures/` as PDF files.  
Script: `docs/verified/figures/generate_verified_figures.py`

| File | Content | Referenced in | Data context |
|---|---|---|---|
| `fig1_trial_comparison.pdf` | T2–T8 R², MAE, RMSE bar charts | `05_results.tex` (fig:trial_comparison) | T1–T6: 50 test graphs; T7–T8: 100 test graphs |
| `fig2_uq_ranking.pdf` | MC Dropout vs Ensemble Spearman ρ | `05_results.tex` (fig:uq_ranking) | T5/T6: 50 graphs; T7/T8/ensemble: 100 graphs |
| `fig3_conformal_coverage.pdf` | Conformal coverage and interval width | `05_results.tex` (fig:conformal_coverage) | 50 eval graphs (1,581,750 nodes) |
| `fig4_selective_prediction.pdf` | 39.9% MAE reduction at 50% retention | `05_results.tex` (fig:selective_prediction) | 50 eval graphs (1,581,750 nodes) |
| `fig5_feature_correlation.pdf` | Feature vs \|error\| Spearman correlations | `05_results.tex` (fig:feature_correlation) | 100 test graphs (3,163,500 nodes) |
| `fig7_calibration.pdf` | k₉₅ = 11.65 vs ideal 1.96 | `05_results.tex` (fig:k95_calibration) | 50 eval graphs (1,581,750 nodes) |

Verification checklist (from script output):
```
T8 R²:   0.5957  ✅ (expected: 0.5957)
T8 MAE:  3.96    ✅ (expected: 3.96)
T8 rho:  0.4820  ✅ (expected: 0.4820)
Conf 95% q: 14.6766 ✅ (expected: 14.6766)
k95:     11.65   ✅ (expected: 11.65)
```

---

## Wrong Values Corrected in LaTeX

### 05_results.tex (complete rewrite)

| Wrong value (old) | Correct value | Source |
|---|---|---|
| T8 R²=0.9374 | **0.5957** | test_evaluation_complete.json |
| T8 MAE=91.55 veh/h | **3.96 veh/h** | test_evaluation_complete.json |
| T8 RMSE=172.80 | **7.12 veh/h** | test_evaluation_complete.json |
| Trial results table (R² 0.92-0.94 for all) | Correct per-trial values (T1=0.7860, T2=0.5117...) | all_models_summary.json |
| MC Dropout ρ=0.160 as headline result | T8 standalone ρ=**0.4820**; ρ=0.1600 is Exp A result | mc_dropout_full_metrics_model8.json |
| Temperature Scaling section (ECE 0.352→0.034, T=2.92) | **REMOVED** — not verified from any JSON | no source |
| Feature correlations (wrong values e.g. +0.339, +0.306) | Verified: +0.3316, +0.2615, -0.2286, +0.2110, -0.0695 | feature_analysis_report.txt |
| Tab:threshold_analysis (fabricated MAE values) | **REMOVED** — replaced with verified selective pred data | uq_comparison_model8.json |
| ECE=0.379 in calibration table | **REMOVED** (ECE not verified from JSON) | no source |
| k95=1.96 (old) shown as "T8" | Verified: k95=**11.65** for T8 | uq_comparison_model8.json |
| Split "70/15/15" | T1-T6: **80/15/5**; T7-T8: **80/10/10** | test_loader_params.json |

### 04_experiments.tex (complete rewrite)

| Wrong value (old) | Correct value | Source |
|---|---|---|
| Trial configs (batch=32, split=70/15/15 for all) | Verified per-trial config (batch 8-32, split 80/15/5 or 80/10/10) | all_models_summary.json |
| T8 dropout=0.15 | **T8 dropout=0.2** | test_evaluation_complete.json |
| All trials show same architecture | T1 uses **Linear final layer**, T2-T8 use **GATConv** | MODEL_SUMMARY.md, point_net_transf_gat.py |
| R²=0.9374 in figure caption | Corrected/removed | — |

### 07_conclusion.tex (complete rewrite)

| Wrong value (old) | Correct value |
|---|---|
| R²=0.9374, MAE=91.55 veh/h | **R²=0.5957, MAE=3.96 veh/h** |
| "55% higher correlation (ρ=0.160)" as headline | T8 standalone **ρ=0.4820**; 55% figure is within Exp A only |
| Temperature Scaling note | **REMOVED** |

### 06_discussion.tex (complete rewrite)
- Removed claim "55% improvement ρ=0.160 vs 0.103" as standalone result
- Added ensemble evaluation mismatch caveat
- Corrected all performance figures

### 01_introduction.tex (targeted fix)
- Line 55: Corrected contribution claim from ρ=0.160 to ρ=0.4820 (T8 standalone)
- Added clarification: ρ=0.160 is Exp A context, not primary T8 result

### 03_methodology.tex (targeted fixes)
- Fixed figure caption: "dropout p=0.15–0.20" → "dropout p=0.2 (T8) or p=0.3 (T5–T7)"
- Fixed figure caption: "ρ=0.160" → "ρ=0.4820 (100 test graphs, S=30)"
- Fixed implementation details: correct dropout, batch, dimension values per trial
- Added T1 vs T2–T8 architecture note

---

## Old Figures Warning

`thesis/latex_tum_official/figures/WRONG_FIGURES_WARNING.txt` lists all ~70 existing PNG figures as containing fabricated values. The new LaTeX chapters reference only the verified PDF figures (`fig1_` through `fig7_`).

---

## Remaining Figures (Not Yet Generated)

These were identified as needed but are not yet scripted:

| Figure | Description | Priority |
|---|---|---|
| Architecture diagram (detailed) | Replacing simplified TikZ in 03_methodology.tex with accurate layer sizes | Medium |
| Pipeline/data flow diagram | LineGraph transform visualization | Low |
| MC Dropout workflow | S=30 passes visualization | Low |
| With/without UQ comparison | Based on WITH_WITHOUT_UQ_SUMMARY_MODEL8.md | Low |

The simplified TikZ diagram in `03_methodology.tex` remains as a placeholder. It is described correctly in the caption (now corrected) but the TikZ boxes are not layer-accurate. A replacement requires a new figure script.

---

## Supervisor / Advisor — Resolved

`main.tex` has been corrected:
- `\getSupervisor{}` = **Prof. Dr. Stephan Günnemann**
- `\getAdvisor{}` = **Dominik Fuchsgruber, Elena Natterer, M.Sc.**

---

## Global Consistency Audit — Findings and Fixes (2026-03-14)

Four critical issues were identified and fixed during a full consistency audit.

### Finding 1 — ρ Context Mismatch (Critical) ✅ FIXED

**Problem:** The abstract compared ρ=0.4820 (T8 standalone, full 100-graph test set) directly to ρ=0.1035 (Experiment A ensemble, mismatched subset). These are from different evaluation contexts and cannot be directly compared.

**Fix applied to:** `pages/abstract.tex`, `pages/zusammenfassung.tex`

**Correct framing:**
- T8 standalone (100 graphs, S=30): ρ = **0.4820**
- Experiment A within-context comparison: MC ρ=0.1600 vs ensemble ρ=0.1035 (55% relative improvement *within that subset*)

### Finding 2 — False Monotonicity Claim (Critical) ✅ FIXED

**Problem:** `chapters/07_conclusion.tex` stated "ρ increasing monotonically as dropout rate decreases from 0.3 to 0.2 across Trials 5–8." This is false on two counts:
1. T5→T6 (both dropout=0.3) shows ρ *decreasing* from 0.4263 to 0.4186.
2. T8 differs from T5–T7 in three variables simultaneously (dropout, split ratio, batch size), so single-variable attribution is confounded.

**Fix applied to:** `chapters/07_conclusion.tex`

**Correct statement:** T8 achieves the highest ρ=0.4820, but improvement over T5–T7 cannot be attributed solely to reduced dropout, as three hyperparameters changed simultaneously.

### Finding 3 — GATConv Heads Wrong (Critical) ✅ FIXED

**Problem:** `chapters/03_methodology.tex` stated "GATConv heads: 4". The source code (`point_net_transf_gat.py` lines 95, 160) confirms GATConv uses **1 head (default)**. The 4-head attention is only in TransformerConv layers.

**Fix applied to:** `chapters/03_methodology.tex`

**Correct architecture:**
- TransformerConv(128→256): 4 heads
- TransformerConv(256→512): 4 heads
- GATConv(512→64): 1 head (default)
- GATConv(64→1): 1 head (default) — T2–T8 final layer

### Finding 4 — Mixed S/T Notation (Consistency) ✅ FIXED

**Problem:** `T=30` (uppercase T, reserved for trial numbers and transformer math) appeared in experimental contexts across multiple files where `S=30` should be used.

**Fix applied to:**
- `chapters/01_introduction.tex` line 59: `$T=30$` → `$S=30$`
- `chapters/03_methodology.tex` lines 134–144: All `T`/`t` loop variables in the MC Dropout algorithm → `S`/`s`
- `chapters/02_background.tex` line 122 (after formal definition): Added note — "In our experiments we use $S = 30$" to bridge the formal `T` notation (Gal & Ghahramani) and the experimental `S` notation used in Ch03–Ch07.

**Notation rule enforced:**
- `T` in Ch02: reserved for the formal Gal & Ghahramani definition (standard in literature)
- `S` everywhere else: number of MC Dropout stochastic forward passes in our experiments
