# Phase 6: Data and Scripts Audit Report (Complete)

**Date:** 2026-03-26  
**Status:** COMPLETE  

---

## Executive Summary

Phase 6 verified all 835+ data files (~4.83 GB) and 116 code files (109 .py + 7 .ipynb) in the project. Key results:

- **46/46 NPZ-to-JSON cross-checks PASS** — raw data arrays reproduce the verified thesis metrics exactly
- **All mathematical formulas verified correct** (PIT, CRPS, Winkler, NLL, R², Spearman, bootstrap CI, conformal, temperature scaling)
- **All 8 model checkpoints load successfully** (1,416,835 parameters each for T2-T8)
- **2 new issues found** requiring action in Phase 8/10 (see Issues section)
- **7 intermediate/debug ensemble scripts** identified for cleanup in Phase 9

---

## Phase 6b: Data File Verification

Full report: `docs/verified/PHASE6B_DATA_VERIFICATION_REPORT.md`

**Summary:** All data files verified — NPZ shapes/keys correct, CSVs loadable, models loadable, scalers consistent, training batches present. 46/46 cross-checks between raw NPZ arrays and verified JSON summary metrics all PASS.

---

## Phase 6c: Script Verification

### Core Compute Scripts (11 scripts)

| Script | Purpose | Math Correct? | Status |
|--------|---------|---------------|--------|
| `compute_pit.py` | PIT histogram (Φ(z)) | YES | FINAL |
| `compute_crps.py` | CRPS (Gneiting & Raftery 2007) | YES | FINAL |
| `compute_winkler.py` | Winkler interval score | YES | FINAL |
| `compute_pit_after_tempscaling.py` | PIT before/after T scaling | YES | FINAL |
| `compute_nll.py` | Gaussian NLL | YES | FINAL |
| `compute_bootstrap_ci.py` | Block bootstrap CI | YES | FINAL |
| `regenerate_fig58_s30.py` | Selective prediction (S=30) | YES | FINAL |
| `run_s_convergence.py` | S-convergence experiment | YES | FINAL |
| `generate_s_convergence_figure.py` | S-convergence figure | N/A (plotting) | FINAL |
| `verify_all_metrics.py` | Comprehensive metric verification | YES | FINAL |
| `verify_all_numbers_final.py` | Thesis number verification | N/A (comparison) | FINAL |

### Ensemble Pipeline Scripts (10 scripts)

| Script | Graphs | Status | Notes |
|--------|--------|--------|-------|
| `run_exp_a_single_run.py` | 100 | **FINAL** | Per-graph checkpointing, S=30 |
| `aggregate_exp_a.py` | 100 | **FINAL** | Produces experiment_a_fixed_results.json |
| `run_exp_b_fixed.py` | 100 | **FINAL** | Produces experiment_b_fixed_results.json |
| `run_ensemble_fix.py` | 10 | INTERMEDIATE | Early combined attempt |
| `run_ensemble_fix_fast.py` | 10 | INTERMEDIATE | Fast variant |
| `run_ensemble_fix_5g.py` | 5 | INTERMEDIATE | Budget-constrained |
| `run_ensemble_quick_test.py` | 3 | DEBUG | Smoke test only |
| `run_ensemble_minimal.py` | 3 | DEBUG | Quick approximate values |
| `run_ensemble_final.py` | 3 | INTERMEDIATE | Misleading name, not actually final |
| `verify_ensemble_fix.py` | 1 | DEBUG | One-time weight-remap verification |

### Evaluation Infrastructure (18 scripts in scripts/evaluation/)

Key findings:
- `eign.py` has a **bug**: calls `self.dropout` (a float) as a function — would crash if invoked, but this model is not used for thesis results
- Most scripts are FINAL or SUPPLEMENTARY
- Several scripts hardcode Model 8 paths (acceptable — T8 is primary model)

### Model Architecture (scripts/gnn/models/)

- **PointNetTransfGAT** defined in `scripts/gnn/models/pointnet_transf_gat.py`
- Architecture: PointNetConv(7→256→512) → PointNetConv(514→256→128) → TransformerConv(128→256) → TransformerConv(256→512) → GATConv(512→64) → GATConv(64→1)
- Dropout applied via `F.dropout` in forward pass — **correct for MC Dropout** (active during training AND inference when `model.train()`)
- 1,416,835 parameters (36 state_dict keys) — **matches loaded checkpoint structure exactly**

---

## Phase 6d: Figure Script Verification

### Root-Level run_fig*.py Scripts (12 authoritative)

| Script | Data Source | Verified Values Match? | Status |
|--------|-------------|----------------------|--------|
| run_fig58.py | Verified JSON | YES | AUTHORITATIVE |
| run_fig59.py | Raw CSV | YES (assertions pass) | AUTHORITATIVE |
| run_fig510.py | Raw CSV | YES | AUTHORITATIVE |
| run_fig511.py | Verified JSON | YES (ECE=0.265 is different bin def than 0.269) | AUTHORITATIVE |
| run_fig512.py | Verified JSON | YES (T=2.70, ECE pre/post) | AUTHORITATIVE |
| run_fig513.py | Verified JSON | YES (KS=0.245) | AUTHORITATIVE |
| run_fig514.py | Raw CSV | YES | AUTHORITATIVE |
| run_fig515.py | Raw NPZ + JSON cross-check | YES | AUTHORITATIVE |
| run_fig516.py | Raw NPZ | YES | AUTHORITATIVE |
| run_fig517.py | Raw NPZ | YES | AUTHORITATIVE |
| run_fig518.py | Hardcoded | YES (minor ±0.01 rounding) | AUTHORITATIVE |
| run_fig61.py | Verified JSON | YES | AUTHORITATIVE |

### Figures Directory Scripts (10 scripts)

| Script | Figures Generated | Status |
|--------|------------------|--------|
| generate_all_thesis_figures.py | fig1–fig10 (11 figures) | **ACTIVE** — all hardcoded values match |
| generate_new_figures.py | fig11–fig14 | **ACTIVE** — all values match |
| generate_phase3_figures.py | 6 analysis figures | **PARTIALLY ACTIVE** (analysis_33 superseded) |
| generate_network_intro_figure.py | fig_network_intro | **ACTIVE** |
| generate_pointnet_dataflow_figure.py | pointnet_data_flow | **ACTIVE** — **HAS LR/DROPOUT DISCREPANCY** |
| thesis_style.py | (shared module) | **ACTIVE** |
| run_fig14.py | Wrapper → fig14 | WRAPPER |
| run_fig55.py | Wrapper → fig3_conformal | WRAPPER |
| run_fig56.py | t8_conformal_conditional | **AUTHORITATIVE** |
| run_fig57.py | fig7_calibration | **AUTHORITATIVE** |

---

## Phase 6e: Complete Script Classification

### FINAL / ACTIVE Scripts (thesis-critical)

**Core compute (11):** compute_pit.py, compute_crps.py, compute_winkler.py, compute_pit_after_tempscaling.py, compute_nll.py, compute_bootstrap_ci.py, regenerate_fig58_s30.py, run_s_convergence.py, generate_s_convergence_figure.py, verify_all_metrics.py, verify_all_numbers_final.py

**Ensemble production (3):** run_exp_a_single_run.py, aggregate_exp_a.py, run_exp_b_fixed.py

**Root-level figure generators (12):** run_fig58.py through run_fig518.py, run_fig61.py

**Figures directory generators (5):** generate_all_thesis_figures.py, generate_new_figures.py, generate_phase3_figures.py, generate_network_intro_figure.py, generate_pointnet_dataflow_figure.py

**Shared modules (1):** thesis_style.py

**Model definitions (12):** All files in scripts/gnn/models/ (PointNetTransfGAT + supporting architectures)

**Training infrastructure (3):** scripts/training/ (training loop, data loading, run_models.py)

**Data preprocessing (4):** scripts/data_preprocessing/ (MATSim → PyG conversion)

**GNN I/O (3):** scripts/gnn/ (load/save utilities)

**Evaluation infrastructure (18):** scripts/evaluation/ (MC Dropout runner, conformal, calibration, etc.)

### SUPPLEMENTARY Scripts (audit/verification/utility)

- run_part2_uq_analyses.py, run_part3_calibration_audit.py, run_part4_t7_crosscheck.py — produce docs/verified/ reports
- run_phase3_wrapper.py, run_phase5_all_scripts.py — orchestrator wrappers
- verify_fig514.py, verify_fig61.py — one-time verification
- regenerate_per_graph_npz.py, regenerate_all_figures.py — utility scripts
- audit_data_integrity.py, audit_hardcoded_figures.py — one-time audits
- create_cross_check_package.py, build_submission_final.py — packaging utilities
- generate_all_hd_plots.py — HD plot generation
- run_fig31_fig32_redesign.py — figure redesign utility

### WRAPPER Scripts (thin delegates)

- run_fig14.py, run_fig55.py — wrappers for generate_new_figures.py / generate_all_thesis_figures.py functions

### INTERMEDIATE / DEBUG Scripts (candidates for cleanup)

- run_ensemble_fix.py (10 graphs)
- run_ensemble_fix_fast.py (10 graphs)
- run_ensemble_fix_5g.py (5 graphs)
- run_ensemble_quick_test.py (3 graphs, smoke test)
- run_ensemble_minimal.py (3 graphs, S=10)
- run_ensemble_final.py (3 graphs, misleading name)
- verify_ensemble_fix.py (one-time GATConv fix verification)

### OBSOLETE / PRESENTATION Scripts

- generate_presentation.py, generate_presentation_final.py — PowerPoint generation (not thesis deliverable)

### Jupyter Notebooks (7)

- scripts/training/run_models.ipynb — SUPPLEMENTARY (training)
- scripts/training/plot_learning_curves.ipynb — SUPPLEMENTARY
- scripts/evaluation/visualize_benchmarking.ipynb — SUPPLEMENTARY
- scripts/evaluation/test_model.ipynb — SUPPLEMENTARY
- scripts/evaluation/in_depth_analysis.ipynb — SUPPLEMENTARY
- scripts/misc/investigate_ensemble_performance_and_uncertainty.ipynb — SUPPLEMENTARY
- scripts/misc/monte_carlo_dropout_for_trained_model.ipynb — SUPPLEMENTARY

---

## Issues Found

### HIGH Severity (affects thesis correctness)

None.

### MEDIUM Severity (affects figure accuracy)

**I1: pointnet_data_flow footer has wrong LR and dropout values**
- File: `thesis/latex_tum_official/figures/generate_pointnet_dataflow_figure.py` line 193-194
- Figure footer says: `LR = 1×10⁻³` and `Dropout p = 0.15`
- Thesis Table 4.1 (04_experiments.tex line 66) says: T8 LR = `5×10⁻⁴`, Dropout = 0.2
- **Impact:** The compiled figure `pointnet_data_flow.pdf` contains incorrect hyperparameter values
- **Action needed:** Fix the footer text and regenerate the figure

### LOW Severity (cosmetic / non-thesis-affecting)

**I2: run_fig518.py MAE rounding (±0.01 veh/h)**
- T8 MAE hardcoded as 3.95 (from S=30 NPZ); thesis uses 3.96 (deterministic)
- T7 MAE hardcoded as 4.07; thesis uses 4.06
- Impact: Comparison figure bar heights differ by 0.01 veh/h from prose text. Negligible visually.
- Action: Document or harmonize in Phase 8.

**I3: verify_fig514.py has stale AUPRC cross-check values**
- Hardcoded 0.315 and 0.455 vs actual JSON 0.321 and 0.447
- Impact: Verification script will fail, but actual figure is unaffected
- Action: Fix or document in Phase 9.

**I4: verify_all_numbers_final.py silently skips conformal checks**
- Key names `quantile_90`/`q_90` don't match actual JSON keys `absolute_q_90`
- Impact: Conformal section produces no checks — verification gap
- Action: Fix key names in Phase 9.

**I5: 6/11 core compute scripts have hardcoded Windows absolute paths**
- Scripts: compute_pit.py, compute_crps.py, compute_winkler.py, compute_pit_after_tempscaling.py, regenerate_fig58_s30.py, generate_s_convergence_figure.py
- Impact: Break on any non-author machine. Does not affect thesis results.
- Action: Convert to `Path(__file__).resolve().parent.parent` in Phase 9.

**I6: generate_s_convergence_figure.py has no main() function**
- regenerate_all_figures.py tries to call main() which doesn't exist
- Impact: Batch regeneration would crash for this one figure
- Action: Add main() wrapper or fix caller in Phase 9.

**I7: eign.py has dropout bug (calls float as function)**
- Impact: None — this model is not used for thesis results
- Action: Document in Phase 9.

**I8: run_phase5_all_scripts.py doesn't include run_fig*.py scripts**
- Impact: Incomplete batch regeneration — must run run_fig*.py separately
- Action: Document or update in Phase 9.

### INFO (documented observations)

- ECE=0.265 (reliability diagram, 10 bins 10%-95%) vs ECE=0.269 (temperature scaling, 10 bins 10%-100%) — both correct, different bin definitions
- trial8_uq_ablation_results.csv uses S=50 MC samples; thesis NPZ uses S=30 — both valid, thesis reports S=30 values
- T7 ρ full population = 0.4437 vs thesis 100K subsample = 0.4460 — documented LOW-severity note from Phase 3
- T1 architecture differs (34 layers, no gat_final) — pre-revision, correctly documented
- T7/T8 test dataloaders are byte-identical (same 100 test graphs by design)
- T7/T8 scalers are numerically identical (same data distribution)

---

## Verdict

**Phase 6: COMPLETE**

All data files are structurally sound and internally consistent. All mathematical formulas in compute scripts are correct. The raw-data-to-thesis pipeline is fully validated with 46/46 cross-checks passing. One medium-severity issue found (wrong LR/dropout in pointnet_data_flow figure footer) requiring correction. Seven intermediate/debug scripts identified for Phase 9 cleanup.

**Carry forward to Phase 8/10:**
- I1: Fix pointnet_data_flow footer (MEDIUM)
- I2: Harmonize MAE rounding in run_fig518.py (LOW)
- All other issues are Phase 9 cleanup items
