# Phase 7c: Trial 7 Artifact Deep Audit Report

**Date**: 2026-03-26
**Scope**: All artifacts in `data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/` (148 files, ~465 MB)
**Objective**: Verify T7 as the cross-replication trial, classify all artifacts, cross-check thesis claims.

---

## 1. T7 Directory Inventory Summary

| Category | Count | Size |
|----------|-------|------|
| Model checkpoint (.pth) | 1 | 5.5 MB |
| Test loader (.pt) | 1 | 297 MB |
| Scalers (.pkl) | 6 | ~4.4 KB |
| JSON files | 5 | ~2.1 KB |
| NPZ files (predictions + UQ) | 103 | ~104 MB |
| PNG images | 39 | ~12.6 MB |
| Text docs (.md, .txt) | 5 | ~15 KB |
| Python script | 1 | 42 KB |
| **Total** | **148** | **~465 MB** |

**Empty directories (2)**: `trained_model/checkpoints/`, `uq_results/deterministic_checkpoints/`

---

## 2. T7 Result-Chain Table

| Step | Artifact | Status | Verified In |
|------|----------|--------|-------------|
| 1. Model | `trained_model/model.pth` (5.5 MB) | FINAL | Phase 6 (loads, 1,416,835 params) |
| 2. Test loader | `data_created_during_training/test_dl.pt` (297 MB) | FINAL | Phase 6 (100 graphs, 80/10/10) |
| 3. Scalers | `data_created_during_training/scaler_*.pkl` (6 files) | FINAL | Phase 6 |
| 4. Deterministic preds | `test_predictions.npz` (25 MB) | FINAL | Phase 6 |
| 5. Deterministic eval | `test_evaluation_complete.json` | FINAL | Cross-check below |
| 6. Deterministic 100g | `uq_results/deterministic_full_100graphs.npz` (16 MB) | FINAL | Phase 6 |
| 7. Deterministic metrics | `uq_results/deterministic_metrics_model7_100graphs.json` | SUPPLEMENTARY | Consistent with step 5 |
| 8. MC Dropout raw | `uq_results/mc_dropout_full_100graphs_mc30.npz` (27 MB) | FINAL | Phase 6 |
| 9. MC Dropout metrics | `uq_results/mc_dropout_full_metrics_model7_mc30_100graphs.json` | FINAL | Cross-check below |
| 10. Per-graph checkpoints | `uq_results/checkpoints_mc30/graph_0000..0099.npz` (100 files) | FINAL | Phase 6 |
| 11. Conformal | `uq_results/conformal_standard.json` | FINAL | Cross-check below |

**Chain integrity**: Complete and verified. No breaks.

---

## 3. T7 Thesis Cross-Check Results

### 3a. T7 JSON Contents

**`test_evaluation_complete.json`**:
- trial_name: "7th_trial_higher_lr_80_10_10"
- Hyperparams: batch_size=8, dropout=0.3, lr=0.0006, split="80-10-10", use_weighted_loss=false
- Test: R^2=0.5471, MAE=4.0601, RMSE=7.5343, Pearson=0.7409
- Validation: R^2=0.5497, Pearson=0.7427
- num_test_samples: 3,163,500

**`mc_dropout_full_metrics_model7_mc30_100graphs.json`**:
- R^2=0.5367, MAE=4.0737, RMSE=7.6202
- Spearman rho=0.4437 (full population)
- unc_mean=1.2127, num_samples=30, n_graphs=100, n_nodes=3,163,500
- total_time_minutes=276.19

**`conformal_standard.json`**:
- 50/50 cal/test split, seed=42, n_calibration=1,581,750, n_test=1,581,750
- Absolute: q90=10.295, PICP90=90.02%, q95=15.506, PICP95=95.01%
- Sigma-scaled: k90=10.464, PICP90=90.01%, k95=16.169, PICP95=95.02%

**`t7_error_detection.json`** (in `docs/verified/phase3_results/`):
- spearman_rho=0.44599 (20/80 eval split -> rounds to 0.4460)
- AUROC top-10%=0.7416, top-20%=0.7151
- Selective 50% MAE=2.5134 (-38.3%), 90% MAE=3.3156 (-18.6%)
- k95=16.154, raw_gaussian_coverage_95=48.35%

### 3b. Cross-Check Matrix (Thesis vs Source)

| Thesis Claim (Location) | Source | Source Value | Thesis Value | Status |
|--------------------------|--------|-------------|--------------|--------|
| R^2 = 0.5471 (Tab 5.1, line 30) | test_evaluation_complete.json | 0.5471 | 0.5471 | PASS |
| MAE = 4.06 (Tab 5.1, line 30) | test_evaluation_complete.json | 4.0601 | 4.06 | PASS |
| RMSE = 7.53 (Tab 5.1, line 30) | test_evaluation_complete.json | 7.5343 | 7.53 | PASS |
| Pearson = 0.741 (Tab 5.1, line 30) | test_evaluation_complete.json | 0.7409 | 0.741 | PASS |
| Dropout = 0.3 (Tab 4.2, line 65) | test_evaluation_complete.json | 0.3 | 0.3 | PASS |
| LR = 6e-4 (Tab 4.2, line 65) | test_evaluation_complete.json | 0.0006 | 6x10^-4 | PASS |
| Split = 80/10/10 (Tab 4.2) | test_evaluation_complete.json | "80-10-10" | 80/10/10 | PASS |
| 100 test graphs (Tab 5.1) | deterministic_metrics.json | 100 | 100 | PASS |
| 3,163,500 nodes (Tab 5.2) | mc_dropout_metrics.json | 3,163,500 | 3,163,500 | PASS |
| S = 30 MC samples | mc_dropout_metrics.json | 30 | 30 | PASS |
| Spearman rho = 0.4460 (Tab 5.2) | t7_error_detection.json | 0.44599 | 0.4460 | PASS |
| MC MAE = 4.07 (Tab selective) | mc_dropout_metrics.json | 4.0737 | 4.07 | PASS |
| NLL = 37.22 (Tab 5.5, line 380) | nll_results.json (Phase 3) | 37.22 | 37.22 | PASS |
| AUROC top-10% = 0.742 (Tab 5.X) | t7_error_detection.json | 0.7416 | 0.742 | PASS |
| AUROC top-20% = 0.715 (Tab 5.X) | t7_error_detection.json | 0.7151 | 0.715 | PASS |
| Sel 50% MAE = 2.51 (-38.3%) | t7_error_detection.json | 2.5134 (-38.30%) | 2.51 (-38.3%) | PASS |
| Sel 90% MAE = 3.32 (-18.6%) | t7_error_detection.json | 3.3156 (-18.61%) | 3.32 (-18.6%) | PASS |
| k95 = 16.15 (Tab T7-T8 comp) | t7_error_detection.json | 16.154 | 16.15 | PASS |
| Raw 95% cov = 48.4% | t7_error_detection.json | 48.35% | 48.4% | PASS |

**Result: 19/19 cross-checks PASS. 0 failures.**

### 3c. Spearman rho Discrepancy Note (KNOWN)

- Full population (3,163,500 nodes): rho = 0.4437 (from `mc_dropout_full_metrics_model7_mc30_100graphs.json`)
- Evaluation split (2,530,800 nodes, 80% of test): rho = 0.4460 (from `t7_error_detection.json`)
- Thesis uses 0.4460 consistently -- this is the **evaluation-split value**, methodologically correct (matches the same split used for T8's rho = 0.4820)
- Status: LOW-severity editorial note, carried forward from Phase 3

---

## 4. T7 Thesis Usage Summary

T7 serves as the **cross-replication trial** with extensive thesis presence:

### LaTeX references (~60+ mentions across 7 files):
- **Chapter 5**: Dedicated section "Cross-Trial Robustness: Trial~7 Validation" with subsections on selective prediction, calibration audit, error detection AUROC, and T7-vs-T8 summary
- **Chapter 4**: Training configuration table, experiment descriptions
- **Chapter 6**: Discussion of cross-trial robustness findings
- **Chapter 7**: Key finding bullets, practical recommendations
- **Abstract + Zusammenfassung**: Cross-trial replication mentioned

### Dedicated figures (4):
1. `t7_selective_prediction_curve.pdf` (Fig 5.15) -- from `run_fig515.py`
2. `t7_calibration_curve.pdf` (Fig 5.16) -- from `run_fig516.py`
3. `t7_interval_width_comparison.pdf` (Fig 5.17) -- from `run_fig517.py`
4. `t7_vs_t8_uq_comparison.pdf` (Fig 5.18) -- from `run_fig518.py`

### Script references (30+ files):
- 4 root-level `run_fig*.py` scripts (515-518)
- 6 ensemble scripts (T7 as ensemble member in Exp B)
- 10+ evaluation/verification scripts

### No ensemble experiments in T7 directory:
Unlike T8, T7 has no ensemble_experiments/ subdirectory. This is correct -- ensemble experiments are stored in T8's directory since T8 is the anchor model.

---

## 5. T7 Issues Table

| ID | Severity | Description | Action |
|----|----------|-------------|--------|
| 7c-I1 | INFO | 2 empty directories (`trained_model/checkpoints/`, `uq_results/deterministic_checkpoints/`) | Phase 9: remove |
| 7c-I2 | INFO | 39 PNG plots not referenced in thesis (comparison_plots/8, feature_analysis_plots/10, uq_plots/20, test_evaluation_results.png) | Phase 9: classify as exploratory |
| 7c-I3 | INFO | 2 text reports not referenced in thesis (comparison_report.txt, feature_analysis_report.txt) | Phase 9: classify as historical |
| 7c-I4 | INFO | 3 markdown files not referenced in thesis (MODEL_SUMMARY.md, ADVANCED_UQ_SUMMARY_MODEL7.md, WITH_WITHOUT_UQ_SUMMARY_MODEL7.md) | Phase 9: classify as historical documentation |
| 7c-I5 | INFO | `evaluate_model7.py` (42 KB) superseded by scripts/evaluation/ pipeline | Phase 9: classify as historical |
| 7c-I6 | LOW | Spearman rho 0.4437 (full) vs 0.4460 (eval split) -- thesis uses eval split value consistently | Known Phase 3 editorial note, no action needed |

**No HIGH or MEDIUM severity issues found.**

---

## 6. T7 Classification Summary

### FINAL (thesis-facing, verified) -- 112 artifacts

| Category | Count | Size |
|----------|-------|------|
| Model checkpoint | 1 | 5.5 MB |
| Test infrastructure (loader + 6 scalers + params) | 8 | ~297 MB |
| Deterministic predictions + NPZ | 2 | 41 MB |
| MC Dropout raw NPZ + metrics JSON | 2 | ~27 MB |
| Per-graph MC checkpoints | 100 | ~36 MB |
| Conformal JSON | 1 | <1 KB |
| Test evaluation JSON | 1 | <1 KB |

### SUPPLEMENTARY -- 1 artifact
| File | Notes |
|------|-------|
| `deterministic_metrics_model7_100graphs.json` | Redundant with test_evaluation_complete.json |

### HISTORICAL/EXPLORATORY -- 35 artifacts
- 1 evaluation script (`evaluate_model7.py`)
- 3 markdown documentation files
- 2 text report files
- 39 PNG plots (none in thesis)
- 2 empty directories

### OUTDATED -- 0 artifacts

**T7 is notably cleaner than T8** -- no pre-fix/post-fix artifacts, no intermediate files, no ensemble experiment subdirectory. This is expected since T7 didn't undergo the GATConv bug fix cycle.

---

## 7. Verdict

Trial 7 artifact chain is **CLEAN and VERIFIED**. All 19 thesis-facing cross-checks pass. The T7 directory structure is straightforward with no outdated or conflicting artifacts. T7 correctly serves as the cross-replication trial, and all 4 dedicated thesis figures can be regenerated from the verified NPZ/JSON sources.
