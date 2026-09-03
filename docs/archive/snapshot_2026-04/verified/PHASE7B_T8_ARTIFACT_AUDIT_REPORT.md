# Phase 7b: Trial 8 Artifact Deep Audit Report

**Date**: 2026-03-26
**Scope**: All artifacts in `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/` (699 files, 1.21 GB)
**Objective**: Classify every T8 artifact, verify the thesis-facing result chain, identify outdated/unreferenced items, flag issues.

---

## 1. T8 Checkpoint and Result-Chain Table

The **thesis-facing result chain** traces from model checkpoint through raw predictions to every numeric claim in the thesis.

| Step | Artifact | Size | Status | Verified In |
|------|----------|------|--------|-------------|
| 1. Model | `trained_model/model.pth` | 5.42 MB | FINAL | Phase 6 (loads, 1,416,835 params) |
| 2. Test loader | `data_created_during_training/test_dl.pt` | 297 MB | FINAL | Phase 6 (100 graphs, 80/10/10 split) |
| 3. Scalers | `data_created_during_training/scaler_*.pkl` (6 files) | ~1 KB each | FINAL | Phase 6 (consistent ranges) |
| 4. Deterministic preds | `test_predictions.npz` | 24 MB | FINAL | Phase 6 (cross-checks pass) |
| 5. Deterministic eval | `test_evaluation_complete.json` | -- | FINAL | Phase 3 (R^2=0.5957, MAE=3.96, RMSE=7.12) |
| 6. Deterministic 100g | `uq_results/deterministic_full_100graphs.npz` | 16 MB | FINAL | Phase 6 (46/46 cross-checks) |
| 7. Deterministic metrics | `uq_results/deterministic_metrics_100graphs.json` | -- | SUPPLEMENTARY | Phase 3 (R^2=0.5957, runtime=3.38 min) |
| 8. MC Dropout raw | `uq_results/mc_dropout_full_100graphs_mc30.npz` | 26.7 MB | FINAL | Phase 6 (S=30, cross-checks pass) |
| 9. MC Dropout metrics | `uq_results/mc_dropout_full_metrics_model8_mc30_100graphs.json` | -- | FINAL | Phase 3 (rho=0.482, R^2=0.5857, MAE=3.948) |
| 10. Per-graph checkpoints | `uq_results/checkpoints_mc30/graph_0000..0099.npz` (100 files) | ~260 KB each | FINAL | Phase 6 (per-graph MC outputs) |
| 11. Conformal | `uq_results/conformal_standard.json` | -- | FINAL | Phase 3 (q90=9.92, cov90=90.02%) |
| 12. Ablation CSV | `trial8_uq_ablation_results.csv` | 200 MB | FINAL | Phase 6 (S=50, used by run_fig59/510/514) |
| 13. Diagnostics | `trial8_uq_diagnostics.json` | -- | FINAL | Phase 3 (k95=11.34) |
| 14. Thesis analysis | `trial8_uq_thesis_analysis.json` | -- | FINAL | Phase 3 (calibration data, 30-70 split) |
| 15. Ablation summary | `trial8_uq_ablation_summary.json` | -- | FINAL | Phase 3 (mc_samples=50, rho=0.486) |
| 16. Exp A fixed data | `uq_results/ensemble_experiments/experiment_a_fixed_data.npz` | 158 MB | FINAL | Phase 6 (5 runs x 3,163,500 nodes) |
| 17. Exp A fixed results | `uq_results/ensemble_experiments/experiment_a_fixed_results.json` | -- | FINAL | Phase 3 (MC rho=0.4908, weight_remapping=true) |
| 18. Exp A per-run | `uq_results/ensemble_experiments/exp_a_run_0..4.npz` (5 files) | ~22 MB each | FINAL | Phase 6 (individual run outputs) |
| 19. Exp B fixed data | `uq_results/ensemble_experiments/experiment_b_fixed_data.npz` | 82 MB | FINAL | Phase 6 (5 models + ensemble) |
| 20. Exp B fixed results | `uq_results/ensemble_experiments/experiment_b_fixed_results.json` | -- | FINAL | Phase 3 (Ens rho=0.4333, R^2=0.5656) |

**Chain integrity**: Every link from model.pth -> predictions -> metrics -> thesis text has been independently verified in Phases 3-6. No breaks in the chain.

---

## 2. T8 Final-Source Verdict

The **single correct thesis source chain** for Trial 8:

```
model.pth (5.42 MB, 1,416,835 params)
  |
  +-> test_dl.pt (100 graphs, 80/10/10 split) + scaler_*.pkl
  |     |
  |     +-> test_predictions.npz (deterministic)
  |     |     +-> test_evaluation_complete.json  --> Ch.5 Table 5.1, Ch.6 Table 6.1
  |     |
  |     +-> deterministic_full_100graphs.npz
  |     |     +-> deterministic_metrics_100graphs.json
  |     |
  |     +-> mc_dropout_full_100graphs_mc30.npz (S=30)
  |     |     +-> mc_dropout_full_metrics_model8_mc30_100graphs.json  --> Ch.5 Sec 5.3
  |     |     +-> checkpoints_mc30/graph_0000..0099.npz
  |     |     +-> conformal_standard.json  --> Ch.5 Sec 5.5
  |     |     +-> trial8_uq_diagnostics.json  --> Ch.5 Sec 5.4
  |     |     +-> trial8_uq_thesis_analysis.json  --> Ch.5 reliability
  |     |
  |     +-> trial8_uq_ablation_results.csv (S=50)
  |           +-> trial8_uq_ablation_summary.json  --> Ch.5 ablation
  |           +-> (consumed by run_fig59, run_fig510, run_fig514)
  |
  +-> ensemble_experiments/
        +-> experiment_a_fixed_data.npz + experiment_a_fixed_results.json  --> Ch.6 Exp A
        +-> exp_a_run_0..4.npz (per-run breakdowns)
        +-> experiment_b_fixed_data.npz + experiment_b_fixed_results.json  --> Ch.6 Exp B
```

All thesis-facing numeric claims trace to `*_fixed_*` files (post-GATConv-bug-fix) or to the primary MC/conformal pipeline. No thesis claim traces to any pre-fix or intermediate artifact.

---

## 3. T8 Issues Table

| ID | Severity | Description | File(s) | Action |
|----|----------|-------------|---------|--------|
| 7b-I1 | INFO | Pre-fix `experiment_a_results.json` retained in directory | `uq_results/ensemble_experiments/experiment_a_results.json` | Phase 9: move to `_OLD_OR_DUPLICATE/` |
| 7b-I2 | INFO | Pre-fix `experiment_b_results.json` retained (GATConv bug, all R^2~0) | `uq_results/ensemble_experiments/experiment_b_results.json` | Phase 9: move to `_OLD_OR_DUPLICATE/` |
| 7b-I3 | INFO | Pre-fix `experiment_a_data.npz` (3.3 MB, 3 graphs only) | `uq_results/ensemble_experiments/experiment_a_data.npz` | Phase 9: move to `_OLD_OR_DUPLICATE/` |
| 7b-I4 | INFO | Pre-fix `experiment_b_data.npz` (121 MB, GATConv bug data) | `uq_results/ensemble_experiments/experiment_b_data.npz` | Phase 9: move to `_OLD_OR_DUPLICATE/` |
| 7b-I5 | INFO | Intermediate `ensemble_fixed_results.json` (3-graph pilot) | `uq_results/ensemble_experiments/ensemble_fixed_results.json` | Phase 9: move to `_OLD_OR_DUPLICATE/` |
| 7b-I6 | INFO | Draft text files not in thesis (7 files) | `THESIS_SECTION_MODEL8_UQ.md/.tex`, `MODEL8_FULL_AUDIT_REPORT.md`, `MODEL_SUMMARY.md`, `trial_8_model8_uq_notes.md`, `ADVANCED_UQ_SUMMARY_MODEL8.md`, `WITH_WITHOUT_UQ_SUMMARY_MODEL8.md` | Phase 9: classify as historical documentation |
| 7b-I7 | LOW | `trial_8_model8_uq_notes.md` says "CPU (8 threads)" for MC Dropout; thesis says "T4 GPU" | `trial_8_model8_uq_notes.md` | Note only -- working notes may reflect early experiments. Thesis hardware claim should be verified in Phase 8. |
| 7b-I8 | INFO | `evaluate_model8.py` (1165 lines) superseded by scripts/evaluation/ pipeline | `evaluate_model8.py` | Phase 9: classify as historical |
| 7b-I9 | INFO | 47 unreferenced PNG plots (comparison_plots/9, feature_analysis_plots/11, uq_plots/20, ensemble_experiments/plots/7) | Various subdirs | Phase 9: classify as exploratory visualizations |
| 7b-I10 | INFO | `archive_partial_runs/` has 2 unreferenced partial-run NPZs | `archive_partial_runs/mc_dropout_10graphs.npz`, `mc_dropout_full_20graphs_mc30.npz` | Phase 9: move to `_OLD_OR_DUPLICATE/` |
| 7b-I11 | INFO | 3 empty directories | `trained_model/checkpoints/`, `uq_results/deterministic_checkpoints/`, `archive_partial_runs/checkpoints_legacy/` | Phase 9: remove empty dirs |
| 7b-I12 | INFO | `regenerate_thesis_plots.py` loads pre-fix data but is unreferenced by any .tex file | `scripts/evaluation/regenerate_thesis_plots.py` | Already noted in Phase 6. No action needed -- script is dead code. |

**Critical finding**: NO outdated T8 artifact is referenced by any active thesis-facing code or LaTeX. All 20 search terms across pre-fix filenames, pre-fix variable names, and outdated script names returned zero active references.

---

## 4. T8 Classification Summary

### FINAL (thesis-facing, verified) -- 20 primary artifacts + 100 per-graph + 5 per-run

| Category | Count | Total Size (approx) |
|----------|-------|---------------------|
| Model checkpoint | 1 | 5.4 MB |
| Test infrastructure (loader + 6 scalers) | 7 | 297 MB |
| Deterministic predictions/metrics | 3 | 40 MB |
| MC Dropout raw + metrics | 2 | 27 MB |
| Per-graph MC checkpoints | 100 | 26 MB |
| Conformal + diagnostics + thesis analysis | 3 | <1 MB |
| Ablation CSV + summary JSON | 2 | 200 MB |
| Ensemble Exp A (fixed data + results + 5 runs) | 7 | 268 MB |
| Ensemble Exp B (fixed data + results) | 2 | 82 MB |
| Training logs/config | 1 | <1 MB |
| **FINAL total** | **128** | **~945 MB** |

### OUTDATED (pre-fix or intermediate, safe to isolate) -- 5 files

| File | Size | Reason |
|------|------|--------|
| `experiment_a_results.json` | <1 KB | Pre-fix, 3 graphs, no weight_remapping flag |
| `experiment_b_results.json` | <1 KB | Pre-fix, GATConv bug (all R^2~0) |
| `experiment_a_data.npz` | 3.3 MB | Pre-fix, 3 graphs |
| `experiment_b_data.npz` | 121 MB | Pre-fix, GATConv bug data |
| `ensemble_fixed_results.json` | 1.7 KB | Intermediate 3-graph pilot, superseded |

### SUPPLEMENTARY (not directly in thesis, but useful reference) -- 3 files

| File | Notes |
|------|-------|
| `deterministic_metrics_100graphs.json` | Redundant with test_evaluation_complete.json |
| `uq_comparison_model8.json` | MC vs conformal comparison at multiple levels |
| `test_loader_params.json` | Infrastructure metadata |

### HISTORICAL/EXPLORATORY (documentation + plots, not in thesis) -- ~63 artifacts

- 7 draft/notes markdown/tex files
- 1 superseded evaluation script (evaluate_model8.py, 1165 lines)
- 47 PNG plots across 4 subdirectories
- 2 partial-run NPZ files in archive_partial_runs/
- 3 empty directories
- 1 superseded plot-generation script reference (regenerate_thesis_plots.py)
- 500 per-graph ensemble checkpoints (in ensemble_experiments/per_graph_checkpoints/)

### Unclassified: 0

---

## 5. GATConv Bug Documentation

The pre-fix vs post-fix distinction is critical for understanding the experiment_a/experiment_b artifacts:

**Bug**: When loading models for Experiment B (5-model ensemble), the GATConv layers had incompatible weight key names across trials. Without `weight_remapping=true` and `strict_loading=true`, models loaded with garbage weights.

**Evidence**: `experiment_b_results.json` (pre-fix) shows all 5 individual models with R^2 near 0:
- Model 2: R^2 = -0.0012
- Model 5: R^2 = 0.0031
- Model 6: R^2 = -0.0059
- Model 7: R^2 = -0.0018
- Model 8: R^2 = -0.0060
- Ensemble Spearman rho = 0.117

**Fix applied**: `experiment_b_fixed_results.json` (post-fix) shows correct values:
- Individual R^2 values: 0.45-0.60 range
- Ensemble Spearman rho = 0.4333
- Flags: `weight_remapping: true`, `strict_loading: true`

**Why Exp A was less affected**: Experiment A only reloads model 8 into itself (same architecture, same keys), so the GATConv key mismatch didn't manifest. Pre-fix Exp A still showed reasonable rho=0.471 vs post-fix rho=0.4908.

---

## 6. Carry-Forward Items

| Item | Severity | Phase |
|------|----------|-------|
| I1: pointnet_data_flow figure LR/dropout mismatch | MEDIUM | Fix in Phase 8 or 10 |
| 7b-I7: CPU vs GPU hardware discrepancy in working notes | LOW | Verify in Phase 8 |
| 5 outdated files to isolate | INFO | Phase 9 |
| 47 unreferenced PNGs to classify | INFO | Phase 9 |
| 3 empty directories to remove | INFO | Phase 9 |

---

**Verdict**: Trial 8 artifact chain is **CLEAN and VERIFIED**. All thesis-facing claims trace to post-fix, validated source files. No outdated artifact pollutes the thesis pipeline. The 5 pre-fix files and ~63 historical/exploratory artifacts are clearly separated and can be safely isolated in Phase 9.
