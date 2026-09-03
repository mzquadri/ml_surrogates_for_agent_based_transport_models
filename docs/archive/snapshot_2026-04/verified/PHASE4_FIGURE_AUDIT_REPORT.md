# Phase 4 Figure Audit Report

**Project:** Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models
**Scope:** All figures referenced in the compiled thesis (main.pdf, 94 pages, 7 chapters)
**Date:** 2026-03-26
**Sub-tasks completed:** 4a (inventory), 4b (path/integrity), 4c (script-to-source mapping), 4d (caption verification), 4e (classification)

---

## 1. Figure Verification Table

All paths are relative to `ml_surrogates_for_agent_based_transport_models/`. Generator scripts in `figures/` means `thesis/latex_tum_official/figures/`. File stems resolve to `thesis/latex_tum_official/figures/<stem>.pdf`.

### Chapter 1: Introduction

| # | Thesis Location | Label | File Stem | Generator Script | Source Artifact(s) | Status | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `01_introduction.tex:15` | `fig:network_schematic` | `fig_network_intro` | `figures/generate_network_intro_figure.py` | None (conceptual diagram) | PASS | Conceptual; no numeric data |
| 2 | `01_introduction.tex:91` | `fig:thesis_workflow` | `fig11_thesis_workflow` | `figures/generate_new_figures.py` | Hardcoded: R²=0.5957, ρ=0.4820 | PASS | Values match verified results |

### Chapter 3: Methodology

| # | Thesis Location | Label | File Stem | Generator Script | Source Artifact(s) | Status | Notes |
|---|---|---|---|---|---|---|---|
| 3 | `03_methodology.tex:34` | `fig:feature_distributions` | `fig3_feature_distributions` | `figures/generate_all_thesis_figures.py` | `data/train_data/dist_not_connected_10k_1pct/datalist_batch_{1-4}.pt` | PASS | Reads raw training batches directly |
| 4 | `03_methodology.tex:49` | `fig:node_vs_graph` | `fig10_node_vs_graph` | `figures/generate_all_thesis_figures.py` | `data/TR-C_Benchmarks/.../uq_results/checkpoints_mc30/graph_*.npz` (T8) | PASS | Reads T8 per-graph NPZ files |
| 5 | `03_methodology.tex:93` | `fig:architecture` | `fig8_architecture` | `figures/generate_all_thesis_figures.py` | None (conceptual diagram) | PASS | Architecture schematic; no numeric data |
| 6 | `03_methodology.tex:100` | `fig:data_flow` | `pointnet_data_flow` | `figures/generate_pointnet_dataflow_figure.py` | Hardcoded metrics in diagram | PASS | ρ=0.4820 in caption matches verified |
| 7 | `03_methodology.tex:136` | `fig:trial_progression` | `fig12_trial_progression` | `figures/generate_new_figures.py` | Hardcoded T1-T8 R², MAE, hyperparams | PASS | T8 R²=0.5957, MAE=3.96 match verified |
| 8 | `03_methodology.tex:164` | `fig:mc_dropout_inference` | `fig13_mc_dropout_inference` | `figures/generate_new_figures.py` | None (conceptual diagram) | PASS | Conceptual; caption ρ=0.4820 matches |

### Chapter 5: Results

| # | Thesis Location | Label | File Stem | Generator Script | Source Artifact(s) | Status | Notes |
|---|---|---|---|---|---|---|---|
| 9 | `05_results.tex:43` | `fig:trial_comparison` | `fig1_trial_comparison` | `figures/generate_all_thesis_figures.py` | Hardcoded T2-T8 metrics | PASS | T8 R²=0.5957, MAE=3.96 match |
| 10 | `05_results.tex:92` | `fig:with_without_uq` | `fig6_with_without_uq` | `figures/generate_all_thesis_figures.py` | Hardcoded det vs MC metrics | PASS | ΔR²=−0.010, ΔMAE=−0.01 consistent |
| 11 | `05_results.tex:154` | `fig:uq_ranking` | `fig2_uq_ranking` | `figures/generate_all_thesis_figures.py` | Hardcoded ρ values (post-fix) | PASS | Uses corrected ensemble values: 0.4908, 0.4370, 0.4909, 0.4333. See ensemble note below. |
| 12 | `05_results.tex:166` | `fig:conformal_workflow` | `fig14_conformal_workflow` | `figures/generate_new_figures.py` | Hardcoded coverage values | PASS | 90.02%, 95.01% match verified |
| 13 | `05_results.tex:187` | `fig:conformal_coverage` | `fig3_conformal_coverage` | `figures/generate_all_thesis_figures.py` | Hardcoded: q90=9.92, q95=14.68, cov90=90.02%, cov95=95.01% | PASS | All values match `conformal_standard.json` |
| 14 | `05_results.tex:203` | `fig:conformal_conditional` | `t8_conformal_conditional` | `figures/run_fig56.py` (authoritative) | `docs/verified/phase3_results/conformal_conditional_coverage_t8.json` | PASS | Reads verified JSON; dual-generator exists in `generate_phase3_figures.py` but `run_fig56.py` is authoritative |
| 15 | `05_results.tex:230` | `fig:k95_calibration` | `fig7_calibration` | `figures/run_fig57.py` (authoritative) | Hardcoded: k95=1.96, 11.34 | PASS | k95=11.34 matches `trial8_uq_diagnostics.json`; dual-generator in `generate_all_thesis_figures.py` |
| 16 | `05_results.tex:286` | `fig:selective_prediction_t8` | `t8_selective_prediction_curve` | `run_fig58.py` | `docs/verified/phase3_results/selective_prediction_s30.json` | PASS | Reads verified JSON; caption 50% MAE=2.32, −41.2% match |
| 17 | `05_results.tex:321` | `fig:calibration_curve_t8` | `t8_calibration_curve` | `run_fig59.py` | `data/.../8th_trial/trial8_uq_ablation_results.csv` | PASS | Reads raw T8 ablation CSV |
| 18 | `05_results.tex:328` | `fig:interval_width_t8` | `t8_interval_width_comparison` | `run_fig510.py` | `data/.../8th_trial/trial8_uq_ablation_results.csv` | PASS | Reads raw T8 ablation CSV |
| 19 | `05_results.tex:340` | `fig:reliability_diagram` | `t8_reliability_diagram` | `run_fig511.py` (authoritative) | `docs/verified/phase3_results/reliability_diagram_t8.json` | PASS | Reads verified JSON; ECE=0.265 matches; dual-generator exists |
| 20 | `05_results.tex:354` | `fig:temperature_scaling` | `t8_temperature_scaling` | `run_fig512.py` (authoritative) | `docs/verified/phase3_results/temperature_scaling_t8.json` | PASS | Reads verified JSON; T=2.70, ECE before=0.269, after=0.048 match; dual-generator exists |
| 21 | `05_results.tex:417` | `fig:pit_histogram` | `t8_pit_histogram` | `run_fig513.py` | `docs/verified/phase3_results/pit_t8.json` | PASS | Reads verified JSON; mean=0.433, std=0.399 match |
| 22 | `05_results.tex:476` | `fig:error_detection_t8` | `t8_error_detection_auroc` | `run_fig514.py` | `data/.../8th_trial/trial8_uq_ablation_results.csv` | PASS | Caption rounds AUROC to 0.76/0.74; exact values 0.7585/0.7401 match verified |
| 23 | `05_results.tex:511` | `fig:selective_prediction_t7` | `t7_selective_prediction_curve` | `run_fig515.py` | T7 MC NPZ + deterministic NPZ + verified JSON cross-check | PASS | Reads raw T7 data with assertion guards |
| 24 | `05_results.tex:547` | `fig:calibration_curve_t7` | `t7_calibration_curve` | `run_fig516.py` | T7 MC NPZ + deterministic NPZ | PASS | Reads raw T7 data |
| 25 | `05_results.tex:554` | `fig:interval_width_t7` | `t7_interval_width_comparison` | `run_fig517.py` | T7 MC NPZ | PASS | Reads raw T7 data |
| 26 | `05_results.tex:613` | `fig:t7_vs_t8_comparison` | `t7_vs_t8_uq_comparison` | `run_fig518.py` (authoritative) | Hardcoded verified values for T7+T8 with 18 cross-check assertions | PASS | Dual-generator exists; `run_fig518.py` uses frozen constants |
| 27 | `05_results.tex:627` | `fig:per_graph_variation` | `t8_per_graph_variation` | `figures/generate_phase3_figures.py` | T8 per-graph NPZ files + `test_dl.pt` | PASS | Caption mean ρ=0.464, std=0.023 match verified |
| 28 | `05_results.tex:640` | `fig:feature_correlation` | `fig5_feature_correlation` | `figures/generate_all_thesis_figures.py` | Hardcoded per-feature ρ values | PASS | Feature correlations; descriptive |
| 29 | `05_results.tex:654` | `fig:stratified_uq` | `t8_stratified_uq` | `figures/generate_phase3_figures.py` | T8 per-graph NPZ + `test_dl.pt` | PASS | Reads raw T8 data; stratified sub-analysis |

### Chapter 6: Discussion

| # | Thesis Location | Label | File Stem | Generator Script | Source Artifact(s) | Status | Notes |
|---|---|---|---|---|---|---|---|
| 30 | `06_discussion.tex:17` | `fig:s_convergence` | `t8_s_convergence` | `run_fig61.py` | `docs/verified/phase3_results/s_convergence_results.json` | PASS | Reads verified JSON |
| 31 | `06_discussion.tex:90` | `fig:policy_explanation` | `fig9_policy_explanation` | `figures/generate_all_thesis_figures.py` | None (conceptual diagram) | PASS | Caption MAE=2.32, −41.2% match selective prediction verified |
| 32 | `06_discussion.tex:137` | `fig:pit_after_tempscaling` | `t8_pit_after_tempscaling` | `scripts/compute_pit_after_tempscaling.py` | T8 MC NPZ, T_opt=2.7025 from `temperature_scaling_t8.json` | PASS | Caption KS before=0.245, after=0.104, T=2.70 all match verified |

**Result: 32/32 figures PASS. Zero mismatches with verified results.**

---

## 2. Issues / Mismatch Table

| # | Figure or Script | Issue Type | Severity | Exact Path(s) | Recommended Action | Thesis-Correctness or Cleanup-Only |
|---|---|---|---|---|---|---|
| I1 | `fig4_selective_prediction.pdf/.png` | Unreferenced figure | LOW | `thesis/latex_tum_official/figures/fig4_selective_prediction.{pdf,png}` | Preserve for now. Move to `_REVIEW_CANDIDATES/` in Phase 9 cleanup. Superseded by `t8_selective_prediction_curve` (which reads verified JSON via `run_fig58.py`). | **Cleanup-only.** Not referenced in LaTeX; does not affect compiled thesis. |
| I2 | `VERIFIED_RESULTS_MASTER.csv` — ensemble rows | Stale CSV data | MEDIUM | `docs/verified/VERIFIED_RESULTS_MASTER.csv` lines 24-27 | Update the 4 ensemble ρ values from old buggy results (0.1600/0.1035/0.1601/0.1167) to corrected values (0.4908/0.4370/0.4909/0.4333) and update source file references from `experiment_a_results.json` to `experiment_a_fixed_results.json`, `experiment_b_results.json` to `experiment_b_fixed_results.json`. | **Cleanup-only.** The thesis figures (`fig2_uq_ranking`) and thesis text already use the correct post-fix values. The stale CSV is a documentation artifact, not a source for any figure or LaTeX content. |
| I3 | `VERIFIED_RESULTS_MASTER.csv` — temperature scaling row | Stale CSV note | LOW | `docs/verified/VERIFIED_RESULTS_MASTER.csv` (bottom row) | Update the `ADDITIONAL_FACTS,temperature_scaling_claim` row. It says "NOT VERIFIED - no source file found" but `temperature_scaling_t8.json` now exists in `docs/verified/phase3_results/` with full results. | **Cleanup-only.** The thesis figure (`t8_temperature_scaling` via `run_fig512.py`) reads the correct verified JSON. |
| I4 | Dual generators — `t8_reliability_diagram` | Dual-generator conflict | LOW | `figures/generate_phase3_figures.py` (line 91) vs `run_fig511.py` | Document that `run_fig511.py` is authoritative (reads verified JSON with assertion guards). The function in `generate_phase3_figures.py` recomputes from raw CSV and is superseded. Do not delete either; add a comment in Phase 9. | **Cleanup-only.** Both produce visually similar output; the on-disk figure was generated by the authoritative script. |
| I5 | Dual generators — `t8_temperature_scaling` | Dual-generator conflict | LOW | `figures/generate_phase3_figures.py` (line 1211) vs `run_fig512.py` | Same as I4. `run_fig512.py` is authoritative. | **Cleanup-only.** |
| I6 | Dual generators — `t8_conformal_conditional` | Dual-generator conflict | LOW | `figures/generate_phase3_figures.py` (line 719) vs `figures/run_fig56.py` | Same pattern. `run_fig56.py` is authoritative (reads verified JSON). | **Cleanup-only.** |
| I7 | Dual generators — `t7_vs_t8_uq_comparison` | Dual-generator conflict | LOW | `figures/generate_phase3_figures.py` (line 1573) vs `run_fig518.py` | Same pattern. `run_fig518.py` is authoritative (frozen constants with 18 assertions). | **Cleanup-only.** |
| I8 | Dual generators — `fig7_calibration` | Dual-generator, different styling | LOW | `figures/generate_all_thesis_figures.py` vs `figures/run_fig57.py` | Same numeric values (k95=1.96, 11.34) but different colors and annotation styles. `run_fig57.py` has more polished styling (P_BLUE vs P_GREEN, computed ratio "5.8x"). Treat `run_fig57.py` as authoritative. | **Cleanup-only.** Numeric content identical; visual styling difference only. |
| I9 | Dual generators — `fig3_conformal_coverage` | Thin wrapper (no conflict) | INFO | `figures/generate_all_thesis_figures.py` + `figures/run_fig55.py` | `run_fig55.py` is a thin import wrapper calling the same function. No conflict — identical output. | **No action needed.** |
| I10 | No `\graphicspath` in LaTeX | Missing best-practice directive | INFO | `thesis/latex_tum_official/main.tex` | All `\includegraphics` paths use explicit `figures/` prefix and resolve correctly. Adding `\graphicspath{{figures/}}` is optional best-practice. | **Cleanup-only.** Current setup is fully functional. |
| I11 | Caption rounding in `fig:error_detection_t8` | Minor rounding | INFO | `chapters/05_results.tex:476` | Caption says "AUROC values of 0.76 and 0.74" while exact values are 0.7585 and 0.7401. Acceptable — the table in the text provides exact values. | **No action needed.** Consistent with convention of approximate values in captions + exact values in tables. |

**Thesis-correctness issues found: 0**
**Cleanup-only issues: 8 (I1-I8, I10)**
**Informational notes: 3 (I9, I10, I11)**

---

## 3. Classification Summary

### 3a. Verified Final Figures: 32 of 32 referenced — ALL PASS

Every figure referenced by `\includegraphics` in the thesis:
- Exists as a valid PDF/PNG pair in `thesis/latex_tum_official/figures/`
- Has an identified generator script
- Has caption numeric claims that match Phase 3 verified results (where applicable)
- Has no path resolution errors (all use `figures/<stem>.pdf` relative to `main.tex`)

### 3b. Unreferenced Figures: 1

| File Stem | Files | Status | Notes |
|---|---|---|---|
| `fig4_selective_prediction` | `.pdf` + `.png` | **SUPERSEDED** | Generated by `generate_all_thesis_figures.py`. Not referenced by any `\includegraphics`. Superseded by `t8_selective_prediction_curve` (generated by `run_fig58.py` from verified JSON `selective_prediction_s30.json`). **Preserve for now; move to `_REVIEW_CANDIDATES/` in Phase 9.** |

### 3c. Active/Authoritative Figure Generator Scripts: 19

| Script | Location | Figures (count) | Data Source Type |
|---|---|---|---|
| `generate_all_thesis_figures.py` | `figures/` | fig1, fig2, fig3_conformal, fig3_feature, fig4(unused), fig5, fig6, fig7*, fig8, fig9, fig10 (11) | Hardcoded + raw training data |
| `generate_new_figures.py` | `figures/` | fig11, fig12, fig13, fig14 (4) | Hardcoded/conceptual |
| `generate_phase3_figures.py` | `figures/` | t8_stratified_uq, t8_per_graph_variation (2 unique) | Raw T8 NPZ data |
| `generate_network_intro_figure.py` | `figures/` | fig_network_intro (1) | Conceptual |
| `generate_pointnet_dataflow_figure.py` | `figures/` | pointnet_data_flow (1) | Conceptual |
| `run_fig56.py` | `figures/` | t8_conformal_conditional (1) | Verified JSON |
| `run_fig57.py` | `figures/` | fig7_calibration (1) | Hardcoded |
| `run_fig58.py` | root | t8_selective_prediction_curve (1) | Verified JSON |
| `run_fig59.py` | root | t8_calibration_curve (1) | Raw T8 CSV |
| `run_fig510.py` | root | t8_interval_width_comparison (1) | Raw T8 CSV |
| `run_fig511.py` | root | t8_reliability_diagram (1) | Verified JSON |
| `run_fig512.py` | root | t8_temperature_scaling (1) | Verified JSON |
| `run_fig513.py` | root | t8_pit_histogram (1) | Verified JSON |
| `run_fig514.py` | root | t8_error_detection_auroc (1) | Raw T8 CSV |
| `run_fig515.py` | root | t7_selective_prediction_curve (1) | Raw T7 NPZ |
| `run_fig516.py` | root | t7_calibration_curve (1) | Raw T7 NPZ |
| `run_fig517.py` | root | t7_interval_width_comparison (1) | Raw T7 NPZ |
| `run_fig518.py` | root | t7_vs_t8_uq_comparison (1) | Hardcoded verified values |
| `run_fig61.py` | root | t8_s_convergence (1) | Verified JSON |
| `compute_pit_after_tempscaling.py` | `scripts/` | t8_pit_after_tempscaling (1) | Raw T8 NPZ + verified T_opt |

**Also active:** `thesis_style.py` (shared styling module, dependency of most scripts above).

### 3d. Convenience Wrappers: 3

| Script | Location | What It Does | Notes |
|---|---|---|---|
| `run_fig55.py` | `figures/` | Imports and calls `fig3_conformal_coverage()` from `generate_all_thesis_figures.py` | Identical output to parent function; no conflict |
| `run_fig14.py` | `figures/` | Thin wrapper for fig14 generation | `generate_new_figures.py` is the primary source |
| `run_fig31_fig32_redesign.py` | `scripts/` | Imports `fig3_feature_distributions` + `fig10_node_vs_graph` from `generate_all_thesis_figures.py` | Convenience entry point for regenerating 2 data-driven figures |

### 3e. Obsolete/Superseded Figure-Related Scripts: 0 standalone

No standalone scripts are fully obsolete. However, 4 functions within `generate_phase3_figures.py` are superseded by their corresponding `run_fig*.py` authoritative counterparts (see dual-generator conflicts below). The script itself remains partially active (it still uniquely generates `t8_stratified_uq` and `t8_per_graph_variation`).

### 3f. Dual-Generator Conflicts: 5 pairs, all resolved

| Figure Stem | Script A (non-authoritative) | Script B (authoritative) | Why B is authoritative | Conflict severity |
|---|---|---|---|---|
| `t8_reliability_diagram` | `generate_phase3_figures.py` `analysis_31_reliability_diagram()` | `run_fig511.py` | Reads locked verified JSON + assertion guards; deterministic output | LOW |
| `t8_temperature_scaling` | `generate_phase3_figures.py` `analysis_35_temperature_scaling()` | `run_fig512.py` | Reads locked verified JSON; avoids recomputation drift | LOW |
| `t8_conformal_conditional` | `generate_phase3_figures.py` `analysis_33_conformal_conditional()` | `run_fig56.py` | Reads locked verified JSON | LOW |
| `t7_vs_t8_uq_comparison` | `generate_phase3_figures.py` `analysis_36_t7_auroc()` | `run_fig518.py` | 18 hardcoded assertions on frozen verified values | LOW |
| `fig7_calibration` | `generate_all_thesis_figures.py` `fig7_calibration()` | `run_fig57.py` | Same values, more polished styling (ratio annotation, P_BLUE palette) | LOW |

In all 5 cases: the numeric content is equivalent, and the on-disk figures were generated by the authoritative scripts. No ambiguity in which output is currently compiled into the thesis.

---

## 4. Phase 4 Conclusion

### Are the thesis figures safe and current?

**Yes.** All 32 referenced figures:
- Exist as valid PDF/PNG pairs with no corruption (4a/4b)
- Have identified generator scripts with traceable source data (4c)
- Have caption numeric claims matching Phase 3 verified results — 0 mismatches out of 18 figures with numeric captions (4d)
- Are generated from either verified JSON artifacts in `docs/verified/phase3_results/`, raw trial data with assertion guards, or hardcoded values that match verified results (4c/4d)

### Does any figure depend on outdated or buggy artifacts?

**No.** The only stale data found is in `VERIFIED_RESULTS_MASTER.csv` (issues I2, I3), which is a documentation artifact — no figure generator reads from this CSV. Specifically:

- **Ensemble discrepancy:** The CSV rows 24-27 contain pre-fix ρ values (~0.16) from the buggy evaluation (before `weight_remapping=true`, `strict_loading=true`). The actual thesis figure `fig2_uq_ranking` hardcodes the correct post-fix values (0.4908, 0.4370, 0.4909, 0.4333) which exactly match `experiment_a_fixed_results.json` and `experiment_b_fixed_results.json`. The stale CSV does not affect any figure or any thesis text. It is a cleanup-only issue.

- **Temperature scaling CSV note:** Similarly stale — the figure reads from the correct verified JSON.

### What should be deferred to Phase 9 cleanup?

1. Move `fig4_selective_prediction.pdf/.png` to `_REVIEW_CANDIDATES/` (unreferenced, superseded)
2. Update `VERIFIED_RESULTS_MASTER.csv` ensemble rows (lines 24-27) with corrected values and source filenames
3. Update `VERIFIED_RESULTS_MASTER.csv` temperature scaling row to reflect that verification is now complete
4. Add comments to `generate_phase3_figures.py` marking the 4 superseded analysis functions and pointing to their authoritative `run_fig*.py` replacements
5. Optionally add `\graphicspath{{figures/}}` to `main.tex` or `settings.tex`

None of these affect thesis correctness. The compiled thesis PDF is clean.

---

**Phase 4 is complete.**
