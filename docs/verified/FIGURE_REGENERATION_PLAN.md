# FIGURE_REGENERATION_PLAN.md
# Plan for Generating Thesis-Ready Figures
# Last verified: 2026-03-14

---

## Overview

All figures must use verified numbers from JSON files only.
The script `docs/verified/figures/generate_verified_figures.py` is the authoritative
figure generator. Do NOT use `scripts/evaluation/generate_thesis_charts.py` (wrong values).

---

## Mandatory Data Context Note (ALL figures)

Every figure must include this note in the subtitle, footer, or caption:

> "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset), unless stated otherwise."

**Verification:** `feature_analysis_report.txt` (T8 folder) line 15 confirms "Total Simulations: Elena=10000, Model 8=1000"; line 71 confirms "1000 simulations vs Elena's 10000". `point_net_transf_gat.py` line 14 states the paper used 10,000 simulations.

Each figure must also mention its **specific evaluation subset** alongside the global 10% context. See per-figure notes below.

---

---

## Figure 1: Trial Comparison Bar Chart (R², MAE, RMSE)

**Purpose:** Show the progression of performance across Trials 2–8 (T1 excluded).
**Type:** Grouped bar chart or side-by-side bars
**Data source:** VERIFIED_RESULTS_MASTER.csv or hardcoded verified JSON values

**Verified values to plot:**

| Trial | R²     | MAE   | RMSE  |
|-------|--------|-------|-------|
| T2    | 0.5117 | 4.33  | 8.15  |
| T3    | 0.2246 | 5.99  | 10.27 |
| T4    | 0.2426 | 6.08  | 10.15 |
| T5    | 0.5553 | 4.24  | 7.78  |
| T6    | 0.5223 | 4.32  | 8.06  |
| T7    | 0.5471 | 4.06  | 7.53  |
| T8    | 0.5957 | 3.96  | 7.12  |

**Annotation:** Highlight T8 bar. Add note: "T1 excluded (different architecture)."

**Data context note (mandatory in figure footer):**
> "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).  |  This figure: T1–T6: 50 test graphs (80/15/5 split); T7–T8: 100 test graphs (80/10/10 split)."

**Thesis caption:** "Predictive performance across training trials. Trial 8 achieves the best R²=0.5957 and MAE=3.96 veh/h. Trials 3–4 show the detrimental effect of weighted loss. All results use 1,000 of 10,000 available scenarios."

---

## Figure 2: UQ Ranking Quality — Spearman ρ Comparison

**Purpose:** Compare MC Dropout uncertainty quality across trials and against ensemble.
**Type:** Horizontal bar chart

**Verified values:**

| Method | ρ      | n_nodes    |
|--------|--------|------------|
| T8 MC Dropout | 0.4820 | 3,163,500 |
| T7 MC Dropout | 0.4437 | 3,163,500 |
| T5 MC Dropout | 0.4263 | 1,581,750 |
| T6 MC Dropout | 0.4186 | 1,581,750 |
| Exp B Multi-model | 0.1167 | — |
| Exp A MC Dropout avg | 0.1600 | 3,163,500 |
| Exp A Ensemble var | 0.1035 | 3,163,500 |
| Exp A Combined | 0.1601 | 3,163,500 |

**Annotation:** Draw vertical line at ρ=0.48 for T8. Color MC Dropout bars differently from ensemble bars.

**Data context note (mandatory in figure footer):**
> "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).  |  This figure: T5/T6 MC Dropout: 50 test graphs (1,581,750 nodes); T7/T8 MC Dropout and ensemble: 100 test graphs (3,163,500 nodes)."

**Thesis caption:** "Spearman rank correlation between predicted uncertainty and absolute prediction error. Single-model MC Dropout (T8: ρ=0.4820) substantially outperforms ensemble-based uncertainty estimates (ρ≈0.10–0.16). Results use 1,000 of 10,000 available scenarios."

---

## Figure 3: Conformal Prediction Coverage

**Purpose:** Show that conformal intervals achieve nominal coverage.
**Type:** Bar chart or table visualization

**Verified values:**

| Level | q (veh/h) | Achieved coverage | Width |
|-------|-----------|-------------------|-------|
| 90%   | 9.9196    | 90.02%            | ±9.92 |
| 95%   | 14.6766   | 95.01%            | ±14.68|

**Optional add:** Show ±1.96σ MC Dropout coverage (~55%) to contrast with conformal.

**Data context note (mandatory in figure footer):**
> "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).  |  This figure: T8: 100 test graphs split 50 calibration + 50 evaluation (1,581,750 nodes each)."

**Thesis caption:** "Conformal prediction achieves nominal coverage: 90.02% at the 90% level and 95.01% at the 95% level, with symmetric intervals of width ±9.92 and ±14.68 veh/h respectively. Results use 1,000 of 10,000 available scenarios."

---

## Figure 4: Selective Prediction Curve

**Purpose:** Show that filtering uncertain predictions improves MAE.
**Type:** Line plot: % data retained (x) vs. MAE (y)

**Verified anchor point:**
- Retaining 50% (rejecting top 50% uncertain) → MAE reduction of 39.9%
- Source: ADVANCED_UQ_SUMMARY_MODEL8.md (auto-generated summary from uq_comparison_model8.json analysis)
- Baseline MAE (100% retained): 3.96 veh/h
- MAE at 50% retained: ~3.96 × (1 - 0.399) ≈ 2.38 veh/h [INFERRED from 39.9% figure]

**Note:** Full curve data may be available in `uq_comparison_model8.json`. Check this file for more data points before plotting. If only the 50% anchor is available, plot as a single highlighted point on a conceptual curve.

**Data context note (mandatory in figure footer):**
> "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).  |  This figure: T8: 50 evaluation graphs (1,581,750 nodes); source: uq_comparison_model8.json."

**Thesis caption:** "Selective prediction: retaining only the 50% most certain predictions reduces MAE by 39.9% (from 3.96 to approximately 2.38 veh/h), demonstrating that uncertainty estimates are actionable. Results use 1,000 of 10,000 available scenarios."

---

## Figure 5: Feature Importance / Correlation

**Purpose:** Show which input features correlate most with prediction error.
**Type:** Horizontal bar chart

**Verified values (from feature_analysis_report.txt):**

| Feature | Correlation with error |
|---------|----------------------|
| VOL_BASE_CASE | +0.3316 |
| CAPACITY_BASE_CASE | +0.2615 |
| CAPACITY_REDUCTION | -0.2286 |
| FREESPEED | +0.2110 |
| LENGTH | -0.0695 |

**Annotation:** Color positive correlations blue, negative orange. Add xlabel "Spearman correlation with |error|".

**Data context note (mandatory in figure footer):**
> "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).  |  This figure: T8: 100 test graphs (3,163,500 nodes); correlations from feature_analysis_report.txt."

**Thesis caption:** "Correlation between input features and absolute prediction error. Baseline car volume shows the strongest positive correlation (ρ=0.332), indicating the model struggles more with high-traffic roads. Results use 1,000 of 10,000 available scenarios."

---

## Figure 6: Architecture Diagram (for thesis Chapter 3)

**Type:** Custom diagram (can be drawn in LaTeX/TikZ or matplotlib)
**Content:**
```
[5 features] → [PointNetConv START] → [PointNetConv END]
             → [TransConv 128→256] → [TransConv 256→512]
             → [GATConv 512→64] → [GATConv 64→1]
             → [Δvol_car per road]
```
**Note:** Annotate dropout positions: after PointNet and Transformer layers, NOT after GATConv.

---

## Figure 7: k95 Calibration Comparison

**Purpose:** Illustrate that MC Dropout sigma is not a calibrated standard deviation.
**Type:** Simple comparison (can be a table or bar chart)

| Method | Factor needed for 95% coverage |
|--------|-------------------------------|
| Perfect Gaussian calibration | ±1.96σ |
| MC Dropout T8 sigma | ±11.65σ |
| Conformal prediction | ±14.68 veh/h (absolute, not σ-multiple) |

**Thesis caption:** "The raw MC Dropout standard deviation σ is not a calibrated uncertainty measure: achieving 95% coverage requires ±11.65σ, compared to ±1.96σ for a perfectly calibrated Gaussian. Conformal prediction provides exact coverage by design."

**Data context note (mandatory in figure footer):**
> "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).  |  This figure: T8: 50 evaluation graphs (1,581,750 nodes); k95 from uq_comparison_model8.json."

---

## Script Implementation Notes

The script `docs/verified/figures/generate_verified_figures.py` should:

1. **Hardcode all verified values** — do not rely on loading JSON at runtime for core metrics
2. **Optionally load .npz files** for curves (selective prediction, reliability diagrams)
3. **Save to `docs/verified/figures/`** with filenames:
   - `fig1_trial_comparison.pdf`
   - `fig2_uq_ranking.pdf`
   - `fig3_conformal_coverage.pdf`
   - `fig4_selective_prediction.pdf`
   - `fig5_feature_correlation.pdf`
   - `fig6_architecture.pdf` (optional — may be better in LaTeX)
   - `fig7_calibration.pdf`
4. Use matplotlib with thesis-quality settings (font size 12, tight layout, 300 dpi)
5. Use a consistent color palette (suggested: colorblind-friendly — blue #1f77b4, orange #ff7f0e)

---

## Files to Check Before Running

```
[ ] data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/
    uq_results/uq_comparison_model8.json  → check for full selective prediction curve
[ ] data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/
    feature_analysis_plots/feature_analysis_report.txt  → verify correlation values
[ ] Check if any .npz files exist with per-node uncertainty/error data for Figure 4
```

---

## Estimated Time to Generate All Figures

| Figure | Time |
|--------|------|
| Fig 1 (trial comparison) | 20 min |
| Fig 2 (UQ ranking) | 15 min |
| Fig 3 (conformal) | 15 min |
| Fig 4 (selective pred) | 20 min |
| Fig 5 (features) | 15 min |
| Fig 6 (architecture) | 45 min (if doing in matplotlib) |
| Fig 7 (calibration) | 10 min |
| Script setup + test | 30 min |
| **Total** | **~2.5 hours** |
