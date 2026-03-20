# UQ Feasibility Audit — Phase 0
## Repository Truth Audit for Remaining UQ Work

- **Thesis:** "Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models"
- **Author:** Mohd Zamin Quadri
- **Audit date:** 2026-03-18
- **Method:** Automated exploration of all artifact paths, JSON/NPZ/CSV inspection, cross-referencing against thesis LaTeX (`05_results.tex`, `06_discussion.tex`, `07_conclusion.tex`)
- **Purpose:** Phase 0 — determine what exists, what is feasible, and what is already done. No execution.

---

## 1. Executive Summary

The thesis repository contains a comprehensive set of UQ artifacts across 8 training trials. The thesis itself (Chapter 5, `05_results.tex`, 486 lines) already includes 17 distinct UQ analyses with 14+ figures and tables. This audit identifies:

- **17 analyses already in the thesis** — all verified from source artifacts
- **9 computed results NOT yet integrated** — markdown summaries, JSON metrics, archived plots
- **12 candidate remaining analyses** — ranked by feasibility and thesis value
- **8 things that should NOT be done** — to avoid wasting time on low-value or poorly supported work

---

## 2. Architectural Constraint Warning

> **CRITICAL — DO NOT VIOLATE**
>
> | Trial | Final Layer | Comparable? |
> |-------|------------|-------------|
> | **T1** | **`Linear(64->1)`** | **NO — architecturally distinct** |
> | T2--T8 | `GATConv(64->1)` | Yes — mutually comparable |
>
> **T1 must NEVER be directly compared to T2--T8 in any UQ analysis.**
>
> T1 uses a simple linear projection from the 64-dim node embedding to scalar output.
> T2--T8 use a Graph Attention layer that aggregates neighbourhood information at the output stage.
> These are fundamentally different output mechanisms. T1 also has zero effective dropout,
> making it unsuitable for MC Dropout experiments.
>
> T1 appears in the performance table (Table 5.1) for completeness but is excluded
> from all UQ experiments throughout the thesis.
>
> Source: `scripts/gnn/models/point_net_transf_gat.py` — VERIFIED FROM CODE

---

## 3. Artifact Inventory

All paths are relative to the repository root:
`ml_surrogates_for_agent_based_transport_models/`

### 3.1 Model Checkpoints

| Trial | Path | Size | Status |
|-------|------|------|--------|
| T1 | `data/TR-C_Benchmarks/pointnet_transf_gat_1st_bs32_5feat_seed42/model.pth` | ~5.4 MB | VERIFIED |
| T2 | `data/TR-C_Benchmarks/point_net_transf_gat_2nd_try/trained_model/model.pth` | ~5.4 MB | VERIFIED |
| T3 | `data/TR-C_Benchmarks/point_net_transf_gat_3rd_trial_weighted_loss/trained_model/model.pth` | ~5.4 MB | VERIFIED |
| T4 | `data/TR-C_Benchmarks/point_net_transf_gat_4th_trial_weighted_loss/trained_model/model.pth` | ~5.4 MB | VERIFIED |
| T5 | `data/TR-C_Benchmarks/point_net_transf_gat_5th_try/trained_model/model.pth` | ~5.4 MB | VERIFIED |
| T6 | `data/TR-C_Benchmarks/point_net_transf_gat_6th_trial_lower_lr/trained_model/model.pth` | ~5.4 MB | VERIFIED |
| T7 | `data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/trained_model/model.pth` | ~5.4 MB | VERIFIED |
| T8 | `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth` | ~5.4 MB | VERIFIED |

Epoch checkpoints: 199 total across all trials (every 20 epochs). T1: 13 checkpoints, T2-T8: 15-34 each.
Source: `trained_model/checkpoints/` directories — VERIFIED FROM FILESYSTEM

### 3.2 Test Dataloaders

| Trial | Path | Test Graphs | Status |
|-------|------|-------------|--------|
| T1 | `.../pointnet_transf_gat_1st_bs32_5feat_seed42/dataloaders/test_dl.pt` | 50 | VERIFIED |
| T2 | `.../point_net_transf_gat_2nd_try/data_created_during_training/test_dl.pt` | 50 | VERIFIED |
| T5 | `.../point_net_transf_gat_5th_try/data_created_during_training/test_dl.pt` | 50 | VERIFIED |
| T6 | `.../point_net_transf_gat_6th_trial_lower_lr/data_created_during_training/test_dl.pt` | 50 | VERIFIED |
| T7 | `.../point_net_transf_gat_7th_trial_80_10_10_split/data_created_during_training/test_dl.pt` | 100 | VERIFIED |
| T8 | `.../point_net_transf_gat_8th_trial_lower_dropout/data_created_during_training/test_dl.pt` | 100 | VERIFIED |

Note: T3, T4 test dataloaders not explicitly checked — NOT VERIFIED.

### 3.3 Pre-Computed NPZ Files (MC Dropout + Deterministic)

| Trial | MC Dropout NPZ | Det NPZ | Keys in MC NPZ | Status |
|-------|---------------|---------|-----------------|--------|
| T2 | `.../2nd_try/uq_results/mc_dropout_full_50graphs_mc30.npz` | `.../2nd_try/uq_results/deterministic_full_50graphs.npz` | predictions, uncertainties, targets | VERIFIED |
| T5 | `.../5th_try/uq_results/mc_dropout_full_50graphs_mc30.npz` | Exists but path not explicitly checked | predictions, uncertainties, targets | VERIFIED (MC), INFERENCE (det) |
| T6 | `.../6th_trial_lower_lr/uq_results/mc_dropout_full_50graphs_mc30.npz` | Exists but path not explicitly checked | predictions, uncertainties, targets | VERIFIED (MC), INFERENCE (det) |
| T7 | `.../7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz` | `.../7th_trial/uq_results/deterministic_full_100graphs.npz` | predictions, uncertainties, targets | VERIFIED |
| T8 | `.../8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz` | `.../8th_trial/uq_results/deterministic_full_100graphs.npz` | predictions, uncertainties, targets | VERIFIED |

**Critical discovery:** All NPZ files contain ONLY `predictions` (mean), `uncertainties` (std), `targets`. Raw per-pass MC samples (the 30 individual forward pass outputs) are NOT saved. They are computed in-memory and discarded after aggregation.
Source: `scripts/gnn/help_functions.py:381-431` — VERIFIED FROM CODE

### 3.4 Conformal JSON Files

| Trial | Path | Status |
|-------|------|--------|
| T2 | `.../2nd_try/uq_results/conformal_standard.json` (+ `conformal_metrics_clean.json`, `conformal_metrics.json`) | VERIFIED |
| T5 | `.../5th_try/uq_results/conformal_standard.json` | VERIFIED |
| T6 | `.../6th_trial_lower_lr/uq_results/conformal_standard.json` | VERIFIED |
| T7 | `.../7th_trial_80_10_10_split/uq_results/conformal_standard.json` | VERIFIED |
| T8 | `.../8th_trial_lower_dropout/uq_results/conformal_standard.json` | VERIFIED |

### 3.5 Per-Graph Checkpoints (Individual NPZ per Test Graph)

| Trial | MC per-graph (checkpoints_mc30/) | Det per-graph (deterministic_checkpoints/) | Status |
|-------|----------------------------------|-------------------------------------------|--------|
| T2 | None found | None found | NOT VERIFIED |
| T5 | 50 files | None found | INFERENCE (MC exists based on dir listing) |
| T6 | 50 files | 50 files | INFERENCE |
| T7 | 100 files | 100 files | INFERENCE |
| T8 | 100 files (`graph_0000.npz`...`graph_0099.npz`) | 100 files (`graph_0000.npz`...`graph_0099.npz`) | VERIFIED |

T8 per-graph MC keys: `predictions` (31635,), `uncertainties` (31635,), `targets` (31635,) — VERIFIED FROM NPZ
T8 per-graph det keys: `predictions` (31635,), `targets` (31635,) — VERIFIED FROM NPZ

### 3.6 Ablation Data (T8 Only)

| Artifact | Path | Contents | Status |
|----------|------|----------|--------|
| Ablation CSV | `.../8th_trial/trial8_uq_ablation_results.csv` | 3,163,501 lines (header + 3,163,500 rows). Columns: `target`, `pred_det`, `pred_mc_mean`, `pred_mc_std`, `abs_error_det`, `abs_error_mc`, `in_90_interval` | VERIFIED |
| Ablation summary JSON | `.../8th_trial/trial8_uq_ablation_summary.json` | Config (S=50, NOT S=30), det/MC metrics, PICP/MPIW, Spearman | VERIFIED — but uses S=50, superseded by S=30 results |
| Diagnostics JSON | `.../8th_trial/trial8_uq_diagnostics.json` | k90=7.563, k95=11.344, k99=22.783, scaling ratios, normalized residual stats | VERIFIED |

### 3.7 UQ Comparison (T8 Only)

| Artifact | Path | Contents | Status |
|----------|------|----------|--------|
| UQ comparison JSON | `.../8th_trial/uq_results/uq_comparison_model8.json` | Baseline metrics, MC Dropout Spearman=0.4818, PICP/MPIW at 50/80/90/95%, conformal q/picp/mpiw at same levels. **Note:** test_nodes=1,581,750 (50-graph eval subset) | VERIFIED |

### 3.8 Ensemble Experiments (T8 Only)

| Artifact | Path | Status |
|----------|------|--------|
| Exp A results | `.../8th_trial/uq_results/ensemble_experiments/experiment_a_results.json` | VERIFIED |
| Exp A data | `.../8th_trial/uq_results/ensemble_experiments/experiment_a_data.npz` | VERIFIED |
| Exp B results | `.../8th_trial/uq_results/ensemble_experiments/experiment_b_results.json` | VERIFIED |
| Exp B data | `.../8th_trial/uq_results/ensemble_experiments/experiment_b_data.npz` | VERIFIED |
| 7 plot PNGs | `.../ensemble_experiments/plots/exp_{a,b}_*.png` | VERIFIED (not in thesis) |

### 3.9 Temperature Scaling

| Artifact | Path | Status |
|----------|------|--------|
| Script | `scripts/evaluation/temperature_scaling_calibration.py` (453 lines) | VERIFIED — exists, implements grid search + scipy optimize for T parameter |
| Output PNG 1 | `thesis/latex_tum_official/figures/ARCHIVED_OLD_FIGURES/uq_temperature_scaling.png` | VERIFIED — exists but archived |
| Output PNG 2 | `thesis/latex_tum_official/figures/ARCHIVED_OLD_FIGURES/uq_calibration_improvement.png` | VERIFIED — exists but archived |
| Result JSON/CSV | None found anywhere in repository | NOT VERIFIED — no machine-readable output artifact exists |

**Warning:** The claim "Temperature Scaling reduced ECE from 0.356 to 0.033, T=2.90" appears in old prep notes but has NO verified source artifact. The `OLD_CLAIMS_AUDIT.md` explicitly flags this as untrustworthy.

### 3.10 Key Scripts

| Script | Path | Purpose | Status |
|--------|------|---------|--------|
| MC Dropout predict | `scripts/gnn/help_functions.py:381-431` | `mc_dropout_predict()` — returns mean+std only, discards raw samples | VERIFIED FROM CODE |
| Conformal from MC | `scripts/evaluation/conformal_from_mc.py` (77 lines) | Global + adaptive conformal from NPZ | VERIFIED FROM CODE |
| Temperature scaling | `scripts/evaluation/temperature_scaling_calibration.py` (453 lines) | Post-hoc sigma calibration via learned T | VERIFIED FROM CODE |
| Model architecture | `scripts/gnn/models/point_net_transf_gat.py` (256 lines) | PointNetTransfGAT definition | VERIFIED FROM CODE |
| Run MC full | `scripts/evaluation/run_mc_dropout_full.py` | Full MC Dropout inference, saves NPZ | VERIFIED FROM CODE |
| Run deterministic | `scripts/evaluation/run_deterministic_full.py` | Full deterministic inference, saves NPZ | VERIFIED FROM CODE |

### 3.11 Trial Data Availability Matrix (Summary)

| Artifact | T2 | T5 | T6 | T7 | T8 |
|----------|:--:|:--:|:--:|:--:|:--:|
| model.pth | Y | Y | Y | Y | Y |
| test_dl.pt | Y (50g) | Y (50g) | Y (50g) | Y (100g) | Y (100g) |
| MC NPZ (S=30) | Y (50g) | Y (50g) | Y (50g) | Y (100g) | Y (100g) |
| Det NPZ | Y (50g) | Y (50g) | Y (50g) | Y (100g) | Y (100g) |
| Conformal JSON | Y | Y | Y | Y | Y |
| Per-graph MC checkpoints | -- | 50 | 50 | 100 | 100 |
| Per-graph det checkpoints | -- | -- | 50 | 100 | 100 |
| Ablation CSV (per-node) | -- | -- | -- | -- | Y only |
| UQ comparison JSON | -- | -- | -- | -- | Y only |
| Ensemble experiments | -- | -- | -- | -- | Y only |
| ADVANCED_UQ_SUMMARY.md | -- | Y | Y | Y | Y |
| WITH_WITHOUT_UQ_SUMMARY.md | -- | Y | Y | Y | Y |

---

## 4. Split Groups and Node Counts

| Group | Trials | Split | Train Graphs | Val Graphs | Test Graphs | Nodes per Graph | Test Nodes |
|-------|--------|-------|-------------|------------|-------------|----------------|------------|
| A | T1--T6 | 80/15/5 | 800 | 150 | 50 | 31,635 | 1,581,750 |
| B | T7--T8 | 80/10/10 | 800 | 100 | 100 | 31,635 | 3,163,500 |

**Cross-group comparison rules:**
- Do NOT directly compare test metrics between Group A and Group B trials without noting the different test set sizes
- The thesis handles this correctly: T5-T6 Spearman rho values are reported alongside T7-T8 but with explicit notes about split differences
- Conformal prediction results should be computed within the same test set, not across groups

Source: Training config JSONs for each trial — VERIFIED FROM JSON

---

## 5. Already Completed UQ Work

### 5.1 Already in Thesis (Chapter 5, `05_results.tex`)

| # | Analysis | Thesis Section | Data Source | Status |
|---|----------|---------------|-------------|--------|
| 1 | 8-trial performance table + figure | Sec 5.1 (Table 5.1, Fig 5.1) | `all_models_summary.json` | VERIFIED |
| 2 | Spearman rho for T5--T8 | Sec 5.2 (Table 5.3) | `mc_dropout_full_metrics_model{5,6,7,8}_mc30_{50,100}graphs.json` | VERIFIED |
| 3 | T8 MC Dropout statistics | Sec 5.2.2 | `mc_dropout_full_metrics_model8_mc30_100graphs.json` | VERIFIED |
| 4 | Deterministic vs MC comparison | Sec 5.2.3 (Fig 5.3) | Both T8 NPZ files | VERIFIED |
| 5 | Ensemble Exp A (MC vs ensemble variance) | Sec 5.3.2 (Table 5.4) | `experiment_a_results.json` | VERIFIED |
| 6 | Ensemble Exp B (multi-model) | Sec 5.3.3 (Table 5.5) | `experiment_b_results.json` | VERIFIED |
| 7 | UQ method ranking figure | Sec 5.4 (Fig 5.4) | All rho values compiled | VERIFIED |
| 8 | Conformal prediction 50/50 split | Sec 5.5 (Table 5.6, Fig 5.5-5.6) | `conformal_standard.json` | VERIFIED |
| 9 | k95 calibration table + figure | Sec 5.5.1 (Table 5.7, Fig 5.7) | `trial8_uq_diagnostics.json` (k95 field) | VERIFIED |
| 10 | Full coverage calibration (MC vs conformal at 50/80/90/95%) | Sec 5.6 (Table 5.8) | `uq_comparison_model8.json` | VERIFIED |
| 11 | T8 selective prediction (100/90/50/25/10%) | Sec 5.7 (Table 5.9, Fig 5.8) | `trial8_uq_ablation_results.csv` | VERIFIED |
| 12 | T8 calibration audit 20/80 split | Sec 5.8 (Table 5.10, Fig 5.9-5.10) | `trial8_uq_ablation_results.csv` | VERIFIED |
| 13 | Error detection AUROC (top-10%, top-20%) | Sec 5.9 (Table 5.11, Fig 5.11) | `trial8_uq_ablation_results.csv` | VERIFIED |
| 14 | T7 selective prediction | Sec 5.10.1 (Table 5.12, Fig 5.12) | T7 MC + det NPZ | VERIFIED |
| 15 | T7 calibration audit 20/80 | Sec 5.10.2 (Table 5.13, Fig 5.13-5.14) | T7 MC + det NPZ | VERIFIED |
| 16 | T7 vs T8 comparison table | Sec 5.10.3 (Table 5.14) | Compiled from T7+T8 results | VERIFIED |
| 17 | Feature correlation analysis | Sec 5.11 (Fig 5.15) | `feature_analysis_report.txt` | VERIFIED |

**Total: 17 analyses, ~14 figures, ~14 tables. Thesis UQ coverage is comprehensive.**

### 5.2 Computed but NOT Integrated into Thesis

| # | What Exists | Source File | Why Not Integrated | Status |
|---|-------------|-------------|-------------------|--------|
| 1 | T5 full UQ summary (rho=0.4263, k95=17.20, 35.5% MAE drop at 50% retention) | `.../5th_try/uq_results/ADVANCED_UQ_SUMMARY_MODEL5.md` | Only rho used in thesis Table 5.3; risk filtering, adaptive conformal details omitted | VERIFIED FROM FILE |
| 2 | T6 full UQ summary (rho=0.4186, k95=19.53, 34.3% MAE drop at 50% retention) | `.../6th_trial/uq_results/ADVANCED_UQ_SUMMARY_MODEL6.md` | Same as above | VERIFIED FROM FILE |
| 3 | T5 conformal results (q95=16.38 veh/h, picp_95=94.99%) | `.../5th_try/uq_results/conformal_standard.json` | Thesis focuses conformal analysis on T7-T8 only | VERIFIED FROM JSON |
| 4 | T6 conformal results (q95=16.70 veh/h, picp_95=94.99%) | `.../6th_trial/uq_results/conformal_standard.json` | Same as above | VERIFIED FROM JSON |
| 5 | T8 diagnostics: k90=7.563, k99=22.783, normalized residual stats | `trial8_uq_diagnostics.json` | Only k95 extracted for thesis | VERIFIED FROM JSON |
| 6 | Temperature scaling archived outputs (2 PNGs) | `ARCHIVED_OLD_FIGURES/uq_temperature_scaling.png`, `uq_calibration_improvement.png` | Results UNVERIFIED; no JSON artifact. OLD_CLAIMS_AUDIT.md flags as untrustworthy | NOT VERIFIED (outputs exist but results unconfirmed) |
| 7 | 69 UQ plot PNGs across T5-T8 uq_plots/ + ensemble plots/ | Various `uq_plots/` directories | Thesis uses regenerated PDFs with verified values instead | VERIFIED (files exist, content not trusted) |
| 8 | T8 mean absolute prediction difference (MC-det) = 0.6548 veh/h | `.../8th_trial/uq_results/WITH_WITHOUT_UQ_SUMMARY_MODEL8.md` | Thesis summarizes det-vs-MC differently (delta R2, delta MAE) | VERIFIED FROM FILE |
| 9 | "Top 10% sigma accounts for 26.5% of total error" | `.../8th_trial/uq_results/ADVANCED_UQ_SUMMARY_MODEL8.md` | Not cited in thesis | VERIFIED FROM FILE |

### 5.3 NOT Yet Done (Feasible Remaining Analyses)

See Section 6 (Feasibility Matrix) for the full ranked list.

---

## 6. Feasibility Matrix — Remaining Analyses

Each row includes exact source files and verification status.

### Tier 1: No Retraining, No New Inference (from existing data)

| # | Analysis | Data Source | Source Status | Retraining? | New Inference? | Thesis Value | Effort |
|---|----------|------------|---------------|-------------|----------------|-------------|--------|
| 1 | **Reliability diagram** (expected vs observed coverage curve) | `trial8_uq_ablation_results.csv` (pred_mc_std, targets, pred_mc_mean) | VERIFIED | No | No | HIGH — standard UQ visualization missing from thesis | LOW |
| 2 | **Stratified UQ by feature bins** (uncertainty quality vs volume/capacity/speed) | `trial8_uq_ablation_results.csv` + `test_dl.pt` (feature cols [0,1,2,3,5]) | VERIFIED | No | No | HIGH — enriches discussion, explains where model fails | MEDIUM |
| 3 | **Per-graph uncertainty variation** (graph-level rho, MAE, coverage spread) | `checkpoints_mc30/graph_0000.npz`...`graph_0099.npz` (100 files, each 31635 nodes) | VERIFIED | No | No | MEDIUM — shows variance across scenarios | LOW |
| 4 | **Conformal conditional coverage** (coverage stratified by sigma bin or feature bin) | `trial8_uq_ablation_results.csv` | VERIFIED | No | No | HIGH — addresses marginal-vs-conditional coverage gap | MEDIUM |
| 5 | **Error detection AUROC for T7** (replicates T8 analysis) | T7 MC NPZ + det NPZ (both verified) | VERIFIED | No | No | LOW — T7 cross-check already has selective pred + calibration | LOW |

### Tier 2: No Retraining, Requires Script Modification or Re-Run

| # | Analysis | Data Source | Source Status | Retraining? | New Inference? | Thesis Value | Effort |
|---|----------|------------|---------------|-------------|----------------|-------------|--------|
| 6 | **MC sample count ablation** (S=5,10,20,30 convergence) | Requires modifying `mc_dropout_predict()` to save raw per-pass outputs, then one inference run | VERIFIED (function exists at help_functions.py:381) | No | Yes (1 run, ~4h GPU) | HIGH — standard ablation study, reviewers expect it | MEDIUM |
| 7 | **Temperature scaling verification** (re-run from scratch, produce verified JSON) | `temperature_scaling_calibration.py` + `mc_dropout_full_100graphs_mc30.npz` | VERIFIED (script + data exist) | No | No (script re-run only) | MEDIUM — could reduce k95 toward 1.96 | LOW |

### Tier 3: Requires Retraining or Heavy Computation

| # | Analysis | Data Source | Source Status | Retraining? | New Inference? | Thesis Value | Effort |
|---|----------|------------|---------------|-------------|----------------|-------------|--------|
| 8 | **Spatial uncertainty map** (network visualization) | Per-graph NPZ + edge_index from test_dl.pt | VERIFIED (data exists) | No | No | MEDIUM — visual appeal, but needs geospatial coords | HIGH |
| 9 | **Laplace approximation** (post-hoc Bayesian) | `model.pth` (T8 checkpoint) | VERIFIED | No | Yes (Hessian computation, very heavy) | MEDIUM — principled alternative to MC Dropout | VERY HIGH |
| 10 | **Heteroscedastic / evidential loss** | N/A — requires retraining | N/A | Yes | Yes | LOW for this thesis — better as future work | VERY HIGH |
| 11 | **Deep ensemble** (diverse architectures: GAT, GraphSAGE, GCN) | N/A — requires training 3-5 new models | N/A | Yes | Yes | LOW for this thesis — better as future work | VERY HIGH |
| 12 | **T5/T6 full UQ suite** (selective prediction, calibration audit, error detection) | T5/T6 MC NPZ (50 graphs each) | VERIFIED | No | No | LOW — marginal value over existing T7/T8 cross-validation | MEDIUM |

---

## 7. Key Technical Discoveries

### 7.1 Raw MC Samples Are NOT Saved

The function `mc_dropout_predict()` in `scripts/gnn/help_functions.py:381-431` computes all S=30 forward passes into a `(30, N, 1)` tensor but **returns only the mean and std**. The raw per-pass predictions are discarded after aggregation.

```python
# From help_functions.py:429-431
return mean_prediction.cpu().numpy(), uncertainty.cpu().numpy()
```

**Implication:** MC sample count ablation (S=5,10,20,30) cannot be done from existing NPZ files. It requires:
1. Adding a `return_samples=True` parameter to `mc_dropout_predict()`
2. Running inference once with S=30, saving all 30 raw per-pass outputs
3. Subsampling offline (S=5 from first 5 passes, S=10 from first 10, etc.)

Source: `scripts/gnn/help_functions.py:381-431` — VERIFIED FROM CODE

### 7.2 Temperature Scaling Results Are UNVERIFIED

- Script exists: `scripts/evaluation/temperature_scaling_calibration.py` (453 lines) — VERIFIED
- Archived output PNGs exist in `ARCHIVED_OLD_FIGURES/` — VERIFIED
- NO JSON or CSV result artifact found anywhere in the repository — VERIFIED (searched)
- The claim "ECE reduced from 0.356 to 0.033, T=2.90" in old prep notes has NO source artifact
- `docs/verified/OLD_CLAIMS_AUDIT.md` explicitly warns: "NOT used. NOT verified from any JSON file. Do NOT mention in thesis."

**Status: NOT VERIFIED. Must be re-run from scratch if used.**

### 7.3 Ensemble Evaluation Mismatch Is Confirmed

- Experiment A: R^2 approx 0.003, Spearman rho = 0.103-0.160
- Experiment B: R^2 approx -0.002, Spearman rho = 0.117
- This is a data distribution mismatch in the ensemble evaluation subset, NOT a reflection of individual model failure
- T8 standalone: R^2 = 0.5957, rho = 0.4820 — unaffected

Source: `experiment_a_results.json`, `experiment_b_results.json` — VERIFIED FROM JSON

### 7.4 T8 Ablation Summary JSON Uses S=50, NOT S=30

- `trial8_uq_ablation_summary.json` contains `"mc_samples": 50`
- The thesis uses S=30 results throughout (from the _mc30_ NPZ files and metrics JSONs)
- This JSON is from an earlier experiment run and is **superseded**
- Do NOT mix S=50 and S=30 numbers

Source: `trial8_uq_ablation_summary.json` — VERIFIED FROM JSON

### 7.5 Conformal From MC Script Supports Adaptive Conformal

- `scripts/evaluation/conformal_from_mc.py:51-57` implements sigma-normalized conformal
- Residuals scaled by `sigma + eps`, producing variable-width intervals
- Adaptive conformal results already appear in the thesis calibration audit tables (Tables 5.10 and 5.13)
- No additional implementation work needed for adaptive conformal analyses

Source: `scripts/evaluation/conformal_from_mc.py` — VERIFIED FROM CODE

### 7.6 Feature Column Mapping (for Stratified Analysis)

The 5 thesis features from `test_dl.pt` batch `.x` tensors:
- col 0: VOL_BASE_CASE (veh/h)
- col 1: CAPACITY_BASE_CASE (veh/h)
- col 2: CAPACITY_REDUCTION (veh/h, negative)
- col 3: FREESPEED (m/s)
- col 5: LENGTH (metres)

Column 4 (NUMBER_OF_LANES) is NOT used in the thesis.

Source: Training scripts + `docs/verified/FEATURES_EXPLAINED_FROM_ZERO.md` — VERIFIED

---

## 8. What Should NOT Be Done

| # | Analysis | Reason to Avoid |
|---|----------|----------------|
| 1 | **Full retraining** (heteroscedastic, evidential, SWAG) | Very high effort, requires GPU time, changes the model. Marginal thesis value — better framed as future work in Chapter 6. |
| 2 | **Deep ensemble with diverse architectures** (GAT, GraphSAGE, GCN) | Requires training 3-5 new models from scratch. The thesis already documents the failure of identical-architecture ensembles; "diverse ensembles" is correctly positioned as future work. |
| 3 | **Using T8 ablation summary JSON (S=50) as a source** | Superseded by S=30 results used consistently throughout the thesis. Mixing sample counts would introduce inconsistency and confusion. |
| 4 | **Trusting the temperature scaling claim** (ECE 0.356 to 0.033, T=2.90) | NO verified source artifact exists anywhere in the repository. The `OLD_CLAIMS_AUDIT.md` explicitly flags this. If temperature scaling is to be included, it must be re-run and verified from scratch. |
| 5 | **Spatial network visualization with full Paris geographic map** | Requires geospatial coordinates (lat/lon) not present in the `.pt` batch files. `edge_index` provides graph topology but NOT geographic positions. A topological layout is possible but would not be a meaningful "map." |
| 6 | **Comparing T1 to T2--T8 in any UQ analysis** | Architectural constraint: `Linear(64->1)` vs `GATConv(64->1)`. T1 also has zero effective dropout. See Section 2 boxed warning. |
| 7 | **Running T5/T6 full UQ suite** (selective prediction, calibration audit, error detection) | 50-graph Group A trials with different test set sizes. The thesis already has T7/T8 cross-trial validation on the same 100-graph Group B split, which is stronger evidence. T5/T6 would add volume without strengthening conclusions. |
| 8 | **Using the 69 archived PNG plots** from `uq_plots/` directories | Generated by old/deprecated scripts with potentially incorrect hardcoded values. The thesis uses regenerated PDFs from `generate_all_thesis_figures.py` with verified values. |

---

## 9. Risks and Missing Artifacts

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| MC sample ablation requires modifying `mc_dropout_predict()` | Code change + ~4h GPU inference on T4 | HIGH (if this analysis is pursued) | Small, well-scoped change: add `return_samples=True` param, save raw tensor |
| Temperature scaling re-run could produce different ECE than old claim | Could contradict old prep notes (but those are unverified anyway) | MEDIUM | Run fresh, verify independently, do not reference old claim |
| Stratified analysis needs feature columns from `test_dl.pt` | Must match feature column mapping correctly | LOW (mapping is verified) | Use cols [0,1,2,3,5]; verify by checking value ranges |
| Conformal conditional coverage could reveal gap between marginal and conditional | Could weaken conformal narrative if coverage varies significantly across subgroups | MEDIUM (expected for real data) | Address honestly as a known limitation; cite Barber et al. (2021) on conditional coverage |
| Per-graph analysis could show some graphs with very poor rho | Could raise questions about model consistency | LOW (expected variation) | Report distribution, not just mean; frame as natural scenario-level variation |
| Reliability diagram could show systematic under/over-confidence patterns | Could highlight calibration weaknesses beyond k95 | LOW (k95 already shows severe miscalibration) | This is known; reliability diagram would just visualize what's already reported |

### Missing Artifacts (things that do not exist and would be needed)

| Missing Artifact | Needed For | How to Obtain |
|-----------------|-----------|---------------|
| Raw per-pass MC samples (30 x N x 1 tensor) | MC sample count ablation | Modify `mc_dropout_predict()`, re-run inference once |
| Verified temperature scaling JSON output | Temperature scaling integration | Re-run `temperature_scaling_calibration.py`, save JSON |
| Geospatial coordinates (lat/lon per node) | Geographic uncertainty map | Not available in repo; would need MATSim network file |
| T5/T6 ablation CSV (per-node level) | T5/T6 selective prediction | Run `comprehensive_uq_analysis.py` for T5/T6 (not recommended) |
| T7 error detection results | T7 AUROC cross-check | Compute from existing T7 NPZ files (low priority) |

---

## 10. Recommended Priority Order (Phase 1+ Roadmap Preview)

Based on the feasibility matrix, the recommended execution order is:

1. **Reliability diagram** — LOW effort, HIGH value, standard UQ visualization missing from thesis
2. **Stratified UQ by feature bins** — MEDIUM effort, HIGH value, enriches discussion chapter
3. **Per-graph uncertainty variation** — LOW effort, MEDIUM value, 100 NPZ files ready
4. **Conformal conditional coverage** — MEDIUM effort, HIGH value, addresses known limitation
5. **MC sample count ablation** — MEDIUM effort, HIGH value, but requires code change + inference
6. **Temperature scaling verification** — LOW effort, MEDIUM value, re-run existing script

Items 1-4 require NO new inference and can be executed from existing NPZ/CSV data.
Item 5 requires one modified inference run (~4h GPU).
Item 6 requires re-running an existing script (~minutes).

**This priority order will be refined in Phase 2 after the literature review (Phase 1).**

---

*This document is an audit-only output. No analyses were executed. No files were modified.*
*All status labels follow the convention: VERIFIED FROM {source} / INFERENCE / NOT VERIFIED.*
*Generated by OpenCode assistant — 2026-03-18.*
