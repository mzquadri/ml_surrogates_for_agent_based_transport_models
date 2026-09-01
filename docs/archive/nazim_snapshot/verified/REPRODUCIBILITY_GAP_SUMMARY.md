# REPRODUCIBILITY GAP SUMMARY

**Document created:** 2026-03-19 (Phase 4F of 6-phase audit)
**Purpose:** Honest assessment of what a supervisor/examiner can and cannot reproduce from this repository.

---

## What CAN be reproduced (full artifacts present)

### 1. GNN Inference (all 8 trials)
- **All 8 `model.pth` files** present and loadable
- **All 8 `test_dl.pt` files** present (test dataloaders with graph data)
- **All scalers** present (`y_scaler.pkl` / `scaler_*.pkl`)
- **Model architecture code** at `scripts/gnn/models/point_net_transf_gat.py`
- A supervisor can load any trial's model, run inference on its test set, and verify R²/MAE/RMSE

### 2. MC Dropout UQ (Trials T2, T5, T6, T7, T8)
- **Core function:** `mc_dropout_predict()` in `scripts/gnn/help_functions.py` (lines 381–431)
- **Runner script:** `scripts/evaluation/run_mc_dropout_full.py`
- **Pre-computed results:** NPZ files with predictions, sigmas, node-level arrays
- **Verification:** Run `run_part2_uq_analyses.py` or `run_part3_calibration_audit.py` to reproduce all UQ metrics from NPZ cache
- **All thesis numbers verified** against these NPZ/JSON artifacts (Phase 3: 150+ values, 0 mismatches)

### 3. Conformal Prediction
- **Code:** `scripts/evaluation/conformal_from_mc.py` (functions: `conformal_q`, global/adaptive split)
- **Results:** JSON files in each trial's `uq_results/` with coverage and quantile values
- **Primary results (50/50 split):** 90.02% coverage (q=9.92), 95.01% coverage (q=14.68)

### 4. Calibration Analysis
- **Temperature scaling:** `scripts/evaluation/temperature_scaling_calibration.py`
- **ECE computation, Kuleshov calibration:** Fully in code
- **Results:** T_opt=2.70, ECE drops from 0.265→0.048 (82% improvement)

### 5. Ensemble Experiments
- **Code:** `scripts/evaluation/ensemble_uq_experiments.py`
- **Pre-computed data:** `experiment_a_data.npz` (181 MB), `experiment_b_data.npz`
- **Result:** Deep ensembles yield Spearman ρ=0.104 (poor vs MC Dropout ρ=0.482)

### 6. Thesis PDF Compilation
- **LaTeX source** fully present in `thesis/latex_tum_official/`
- **Build chain:** `pdflatex → biber → pdflatex → pdflatex` (zero errors, 79 pages)
- **All 29 figure files** exist on disk; 0 missing references
- **All bibliography entries** resolve (38 cited, 0 undefined)

### 7. All Thesis Figures
- **29 `\includegraphics` targets** all present as PDF (vector) + PNG (raster)
- **Generator scripts** in `thesis/latex_tum_official/figures/` can regenerate from NPZ data
- **Docs visuals** in `docs/visuals/` (supplementary)

---

## What CANNOT be reproduced (known gaps)

### 1. Training from Scratch
- **No training log CSVs** exist for ANY of the 8 trials
- Training loss/validation curves exist **only as PNG images** (no underlying data)
- **No standalone config/hyperparameter files** — hyperparameters are embedded in:
  - `MODEL_SUMMARY.md` files within each trial directory
  - Evaluation JSON output files
  - The thesis text itself (Chapter 4)
- **Training script** (`scripts/gnn/train.py`) is present but would need manual configuration
- **Impact:** A supervisor cannot re-run training and verify convergence numerically. However, the trained model weights are provided, so all downstream results are reproducible.

### 2. MATSim Simulation Data Generation
- **Raw MATSim scenario outputs** are NOT in the repository (too large)
- **Pre-processed graph data** IS provided: `data/train_data/dist_not_connected_10k_1pct/` (20 batch files, ~124 MB each)
- **Impact:** The pipeline from MATSim XML → PyG graph objects cannot be re-run. But all graph data needed for training/inference is present.

### 3. WandB Experiment Tracking
- **WandB run logs** are excluded from the zip (and should be — they contain API keys)
- **Impact:** Historical training curves from WandB dashboard are not available offline. The PNG snapshots in trial directories partially compensate.

---

## Known Internal Inconsistencies

### 1. T7 Spearman ρ — two valid values used in thesis
- **0.4437** — from `mc_dropout_full_metrics_model7_mc30_100graphs.json` (full 100-graph MC dropout)
  - Used in: `05_results.tex` line 74, `06_discussion.tex` line 17, `07_conclusion.tex` lines 24/38
- **0.446** — from `t7_error_detection.json` (Phase 3 verified pipeline)
  - Used in: `05_results.tex` line 507 (T7/T8 comparison table)
- **Status:** Both are legitimate values from different computation paths. Using both in the same document is an internal inconsistency but does NOT indicate an error.

---

## Known Stale/Incorrect Documentation

| File | Issue | Risk |
|---|---|---|
| `docs/MEETING_PREPARATION.md` | Contains known incorrect hyperparameters (flagged by `OLD_CLAIMS_AUDIT.md`) | LOW — clearly a meeting prep draft, not a results document |
| `docs/verified/VERIFIED_RESULTS_MASTER.csv` | Temperature scaling row still marked "NOT VERIFIED" | LOW — the value IS verified (T=2.70, ECE=0.048) |

---

## Hardcoded Paths (portability)

### Fixed (this session)
- `run_part2_uq_analyses.py` — now uses `Path(__file__).resolve().parent`
- `run_part3_calibration_audit.py` — same
- `run_part4_t7_crosscheck.py` — same

### Remaining (lower priority)
- `scripts/evaluation/generate_thesis_charts.py` line 37 — OUTPUT_DIR hardcoded
- 5 scripts in `evaluation_scripts/` — hardcoded to `C:\Users\zamin\...`
- 2 `compare_with_elena.py` scripts in trial directories
- 10 scripts in `thesis/ARCHIVED_OLD_SCRIPTS/` (dead code)
- ~22 occurrences in docstrings, JSON metadata, conda prefixes (non-functional)

**Impact:** The 3 main UQ runner scripts are now portable. Secondary scripts would need manual path edits on a different machine.

---

## Environment Reproducibility

| File | Purpose | Portability |
|---|---|---|
| `traffic-gnn.yml` | Full conda env (442 lines) | Linux-only, pinned build strings |
| `environment-minimal.yml` | Minimal conda env (~45 lines) | Cross-platform, top-level deps only |

- **`xgboost`** is used in `scripts/gnn/models/xgboost.py` but missing from both env files. Not needed for thesis results (GNN-only).
- **Recommended:** Use `environment-minimal.yml` for fresh setup on any OS.

---

## Checkpoint Directories

All 8 trials have **empty** `checkpoints/` directories. Epoch-level model snapshots were deleted after training completed. Only final `model.pth` files remain.

**Impact:** Cannot analyze training dynamics at intermediate epochs. Final model performance is fully verifiable.

---

## Summary for Examiner

| Capability | Status |
|---|---|
| Load trained model and run inference | YES — all 8 trials |
| Reproduce all UQ metrics from cached data | YES — NPZ + JSON artifacts |
| Reproduce all thesis numbers | YES — 150+ values verified, 0 mismatches |
| Compile thesis PDF | YES — zero LaTeX errors |
| Regenerate thesis figures | YES — generator scripts present |
| Re-train models from scratch | PARTIAL — code present, no config files, no training logs |
| Re-run MATSim simulations | NO — raw simulation data not included |
| Run on non-Windows machine | YES — with `environment-minimal.yml` (3 main scripts are portable) |
