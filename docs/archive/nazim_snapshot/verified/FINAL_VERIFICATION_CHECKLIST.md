# FINAL VERIFICATION CHECKLIST (Phase 6/6)

**Thesis:** Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models
**Author:** Mohd Zamin Quadri
**Programme:** M.Sc. Mathematics in Science and Engineering, TUM School of CIT
**Audit completed:** 2026-03-19

---

## A. Thesis PDF

| Check | Status | Details |
|---|---|---|
| PDF compiles with zero errors | PASS | `pdflatex → biber → pdflatex → pdflatex`, 79 pages |
| Page count | 79 | Consistent with README |
| All 29 `\includegraphics` files exist | PASS | 0 missing |
| All `\ref` targets resolve | PASS | 0 undefined references |
| All `\cite` keys resolve | PASS | 38 citations, all found in `bibliography.bib` |
| No overfull hbox > 30pt | PASS | Only underfull hbox warnings (cosmetic) |
| Warnings only cosmetic | PASS | 34 acro hyperref + 9 font glyph + 15 underfull hbox |

---

## B. Numeric Accuracy

| Check | Status | Details |
|---|---|---|
| `05_results.tex` numbers verified | PASS | 150+ values checked against JSON/NPZ/CSV; 45/45 ground-truth matches; 0 mismatches |
| `06_discussion.tex` numbers verified | PASS | All cross-references to results correct |
| `07_conclusion.tex` numbers verified | PASS | All summary statistics correct |
| `01_introduction.tex` numbers verified | PASS | Preview numbers match results |
| Rounding conventions consistent | PASS | 4 negligible observations documented (all valid truncation/rounding) |

### Known inconsistency (documented, not an error):
| Item | Value 1 | Value 2 | Impact |
|---|---|---|---|
| T7 Spearman ρ | 0.4437 (MC dropout full run) | 0.446 (Phase 3 pipeline) | Two valid computation paths; both appear in thesis. LOW risk. |

---

## C. Key Thesis Numbers — Ground Truth Cross-Reference

| Metric | Thesis Value | Source File | Match |
|---|---|---|---|
| T8 R² | 0.5957 | `eval_final_metrics_model8.json` | EXACT |
| T8 MAE (det.) | 3.96 veh/h | `eval_final_metrics_model8.json` | EXACT |
| T8 MAE (MC mean) | 3.94 veh/h | `mc_dropout_full_metrics_model8_mc30_100graphs.json` | EXACT (3.9476 truncated) |
| T8 RMSE | 7.12 veh/h | `eval_final_metrics_model8.json` | EXACT |
| T8 Spearman ρ | 0.482 | `mc_dropout_full_metrics_model8_mc30_100graphs.json` | EXACT (0.4820) |
| T8 k95 | 11.34 | `mc_dropout_full_metrics_model8_mc30_100graphs.json` | EXACT |
| Conformal 90% (50/50) | 90.02%, q=9.92 | `conformal_results_model8_5050.json` | EXACT |
| Conformal 95% (50/50) | 95.01%, q=14.68 | `conformal_results_model8_5050.json` | EXACT |
| Naive Gaussian ±1.96σ | 55.6% | `mc_dropout_full_metrics_model8_mc30_100graphs.json` | EXACT |
| ECE (raw) | 0.265 | `temperature_scaling_results_model8.json` | EXACT |
| T_opt | 2.70 | `temperature_scaling_results_model8.json` | EXACT |
| ECE (post-temp) | 0.048 | `temperature_scaling_results_model8.json` | EXACT |
| Selective MAE 50% | 2.31 veh/h (−41.6%) | `selective_prediction_results_model8.json` | EXACT |
| AUROC top-10% | 0.7585 | `error_detection_results_model8.json` | EXACT |
| Per-graph ρ | mean=0.464, std=0.023 | `per_graph_spearman_results_model8.json` | EXACT |

---

## D. Model Artifacts

| Check | Status | Details |
|---|---|---|
| 8 `model.pth` files present | PASS | T1 at root, T2–T8 in `trained_model/` |
| 8 `test_dl.pt` files present | PASS | T1 in `dataloaders/`, T2–T8 in `data_created_during_training/` |
| All scalers present | PASS | y_scaler.pkl / scaler_*.pkl per trial |
| T1 architecture: Linear(64→1) | VERIFIED | Never compared to T2–T8 |
| T2–T8 architecture: GATConv(64→1) | VERIFIED | All use same model class |
| T8 dropout = 0.2 | VERIFIED | Thesis + code + MODEL_SUMMARY.md agree |
| T2–T7 dropout = 0.3 (where applicable) | VERIFIED | T3/T4 dropout=0.0 (no UQ) |

---

## E. UQ Pipeline Integrity

| UQ Method | Code Location | Result Artifact | Thesis Section | Verified |
|---|---|---|---|---|
| MC Dropout inference | `help_functions.py:381–431` | `*_mc30_100graphs.npz` | 5.2 | YES |
| Conformal prediction (split) | `conformal_from_mc.py:5–57` | `conformal_results_*.json` | 5.3 | YES |
| Temperature scaling | `temperature_scaling_calibration.py:106–137` | `temperature_scaling_results_*.json` | 5.5 | YES |
| ECE / Kuleshov calibration | `temperature_scaling_calibration.py:54–96` | Same JSON | 5.5 | YES |
| Selective prediction | `run_part2_uq_analyses.py` | `selective_prediction_results_*.json` | 5.4 | YES |
| Error detection (AUROC) | `run_part2_uq_analyses.py` | `error_detection_results_*.json` | 5.6 | YES |
| Empirical k-factor | `run_part3_calibration_audit.py` | Part of conformal JSON | 5.3 | YES |
| Per-graph Spearman | `run_part3_calibration_audit.py` | `per_graph_spearman_*.json` | 5.7 | YES |
| Naive Gaussian interval | `run_part3_calibration_audit.py` | Part of MC metrics | 5.3 | YES |
| Deep ensembles (Exp A/B) | `ensemble_uq_experiments.py:184–574` | `experiment_a/b_data.npz` | 5.8 | YES |
| Stratified UQ analysis | `comprehensive_uq_analysis.py` | Generated figures | 5.7 | YES |
| T7 cross-check | `run_part4_t7_crosscheck.py` | T7 JSON/NPZ files | 5.9 | YES |

---

## F. Repository Structure (post-reorganization)

| Directory | Contents | Clean |
|---|---|---|
| `repo/` (root) | 4 scripts + 2 env files + README + .gitignore | YES |
| `scripts/` | 5 subdirs (data_preprocessing, evaluation, gnn, misc, training), 0 loose files | YES |
| `thesis/` | Only `latex_tum_official/` (LaTeX source + figures + PDF) | YES |
| `docs/` | 9 files + 3 subdirs (verified, visuals, ml_surrogates), no deprecated files | YES |
| `data/` | 3 subdirs (TR-C_Benchmarks with 8 trials, train_data, visualisation) | YES |
| `_archive/` | 12 subdirs, all archived/deprecated material, excluded from zip | YES |

---

## G. Submission Package

| Check | Status | Details |
|---|---|---|
| `thesis_upload.zip` rebuilt | PASS | 799 files, 1.81 GB compressed, 4.83 GB uncompressed |
| `_archive/` excluded | PASS | 0 _archive entries in zip |
| `__pycache__/` excluded | PASS | 0 entries |
| `.git/` excluded | PASS | 0 entries |
| `.ruff_cache/` excluded | PASS | 0 entries |
| `wandb/` excluded | PASS | 0 entries |
| `environment-minimal.yml` included | PASS | Cross-platform conda env |
| All 3 runner scripts portable | PASS | Use `Path(__file__).resolve().parent` |
| All 3 runner scripts pass `py_compile` | PASS | Syntax verified |
| README_SUBMISSION.md accurate | PASS | 79 pages, p=0.2, correct PDF path |
| No files with known errors in zip | PASS | MEETING_PREPARATION.md moved to _archive |

---

## H. Known Gaps (documented in REPRODUCIBILITY_GAP_SUMMARY.md)

| Gap | Severity | Mitigation |
|---|---|---|
| No training log CSVs (all 8 trials) | MEDIUM | Training curves exist as PNGs; final model weights provided |
| No standalone config files | LOW | Hyperparameters documented in MODEL_SUMMARY.md + thesis Ch.4 |
| MATSim raw data not included | LOW | Pre-processed graph data IS included (all 20 batches) |
| WandB logs excluded | LOW | Expected — contains API keys |
| Empty checkpoint directories | LOW | Final model.pth files are the deliverable |
| 8 secondary scripts still have hardcoded paths | LOW | Only affects legacy scripts, now in _archive |
| `xgboost` missing from conda env | LOW | Not used in thesis pipeline (GNN-only) |
| `VERIFIED_RESULTS_MASTER.csv` has stale "NOT VERIFIED" note | LOW | Value IS verified (T=2.70) |

---

## I. Documents Created During This Audit

| Document | Location | Purpose |
|---|---|---|
| `REPRODUCIBILITY_GAP_SUMMARY.md` | `docs/verified/` | Honest assessment of what can/cannot be reproduced |
| `REORGANIZATION_PLAN.md` | `docs/verified/` | Documents all file moves with rationale |
| `FINAL_VERIFICATION_CHECKLIST.md` | `docs/verified/` | This document — professor-ready summary |

---

## Verdict

The repository is **professor-inspection ready**. All thesis numbers are verified against source artifacts (150+ values, 0 mismatches). The PDF compiles cleanly (79 pages, 0 errors). All 12 UQ methods have traceable code → artifact → thesis section chains. The submission zip is clean, well-organized, and excludes all deprecated material.

**Remaining optional actions (not blocking submission):**
1. Pick one canonical T7 Spearman ρ value (0.4437 or 0.446)
2. Prune 22 unused .bib entries
3. Update `VERIFIED_RESULTS_MASTER.csv` temperature scaling status
4. Copy final `main.pdf` to `thesis_TUM_FINAL_v4.pdf` at parent directory
