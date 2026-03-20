# REPO_OVERVIEW_VERIFIED.md
# Repository Overview — Verified from Source Files
# Last verified: 2026-03-14 | Source: direct inspection of repo structure

---

## Project Identity

- **Title:** Uncertainty Quantification for Machine Learning Surrogates in Transportation Policy Analysis
- **Author:** Mohd Zamin Quadri (Nazim)
- **Institution:** Technical University of Munich (TUM)
- **Degree:** Master's Thesis
- **Supervisor:** Dominik Fuchsgruber (TUM-DAML)
- **Advisor:** Elena Natterer (TUM Chair of Traffic Engineering)
- **Examiner:** Prof. Dr. Stephan Günnemann (TUM-DAML)

---

## Repository Root

```
ml_surrogates_for_agent_based_transport_models/
├── data/
├── scripts/
├── docs/
├── notebooks/           (if present)
└── requirements.txt / environment files (if present)
```

---

## Directory: data/

### data/TR-C_Benchmarks/
The primary results directory. Contains one subfolder per training trial (T1–T8).

```
data/TR-C_Benchmarks/
├── pointnet_transf_gat_1st_bs32_5feat_seed42/          TRIAL 1 — EXCLUDED (T1 uses Linear(64→1) final layer; T2–T8 use GATConv(64→1))
│   └── test_results.json
├── point_net_transf_gat_2nd_try/                        TRIAL 2
│   └── test_evaluation_complete.json
├── point_net_transf_gat_3rd_trial_weighted_loss/        TRIAL 3
│   └── test_evaluation_complete.json
├── point_net_transf_gat_4th_trial_weighted_loss/        TRIAL 4
│   └── test_results.json
├── point_net_transf_gat_5th_try/                        TRIAL 5
│   ├── test_evaluation_complete.json
│   └── uq_results/
│       └── mc_dropout_full_metrics_model5_mc30_50graphs.json
├── point_net_transf_gat_6th_trial_lower_lr/             TRIAL 6
│   ├── test_evaluation_complete.json
│   └── uq_results/
│       └── mc_dropout_full_metrics_model6_mc30_50graphs.json
├── point_net_transf_gat_7th_trial_80_10_10_split/       TRIAL 7
│   ├── test_evaluation_complete.json
│   └── uq_results/
│       └── mc_dropout_full_metrics_model7_mc30_100graphs.json
└── point_net_transf_gat_8th_trial_lower_dropout/        TRIAL 8 — BEST MODEL
    ├── test_evaluation_complete.json
    ├── feature_analysis_plots/
    │   └── feature_analysis_report.txt
    └── uq_results/
        ├── mc_dropout_full_metrics_model8_mc30_100graphs.json
        ├── conformal_standard.json
        ├── uq_comparison_model8.json
        ├── ADVANCED_UQ_SUMMARY_MODEL8.md
        └── ensemble_experiments/
            ├── experiment_a_results.json
            └── experiment_b_results.json
```

**Trust hierarchy for metrics:**
- `test_evaluation_complete.json` > `test_results.json` (more fields, hyperparams block present)
- All `.json` files in `uq_results/` are authoritative for UQ metrics
- Do NOT use `docs/MEETING_PREPARATION.md` for hyperparameters — all wrong

### data/raw/ (if present)
- MATSim simulation outputs for Paris road network
- 10,000 scenarios available; 1,000 used for training/evaluation
- Not directly read during model inference; processed into PyG graph objects

---

## Directory: scripts/

### scripts/gnn/models/
```
point_net_transf_gat.py     VERIFIED architecture — PointNet + TransformerConv + GAT
```

**Architecture layers (verified from point_net_transf_gat.py):**
1. PointNetConv using START position `pos[:,0,:]`
2. PointNetConv using END position `pos[:,1,:]`
   - `pos` shape: `[N, 3, 2]` — 3 points (START, END, MIDPOINT); only START and END used
3. TransformerConv: 128→256 (4 heads) + ReLU [+ optional dropout]
4. TransformerConv: 256→512 (4 heads) + ReLU [+ optional dropout]
5. GATConv: 512→64
6. GATConv: 64→1  ← final output (`self.gat_final`)
- Dropout applied in PointNet + Transformer layers; NOT in GATConv layers
- **T1 difference**: T1 had `read_out_node_predictions` (Linear(64→1)) instead of `gat_final` (GATConv(64→1))

### scripts/data_preprocessing/
```
process_simulations_for_gnn.py    Builds PyG graph objects; applies LineGraph() transform
help_functions.py                  Defines target variable (line 121-123)
```

**Target definition (help_functions.py:121-123):**
```python
edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
```
= policy-induced change in car volume per road segment (veh/h)

**LineGraph transform:** Roads → nodes; shared junctions → edges. This converts an edge-prediction problem into a node-prediction problem for PyTorch Geometric.

### scripts/training/
```
run_models.py     Training loop; hyperparameter config; verified against JSON outputs
```

**Verified training splits:**
- Trials 1–6: 80% train / 15% val / 5% test → 50 test graphs
- Trials 7–8: 80% train / 10% val / 10% test → 100 test graphs

### scripts/evaluation/
```
generate_thesis_charts.py    ⚠️ WARNING: hardcodes WRONG test_r2 values — DO NOT USE
```
Use `docs/verified/figures/generate_verified_figures.py` instead.

---

## Directory: docs/

```
docs/
├── MEETING_PREPARATION.md          ⚠️ ALL HYPERPARAMETERS WRONG vs JSON — do not trust
└── verified/                        ✅ Ground-truth verified outputs (this session)
    ├── VERIFIED_RESULTS_MASTER.csv
    ├── FILE_MANIFEST.csv
    ├── OLD_CLAIMS_AUDIT.md
    ├── DATA_EXPLAINED_FROM_ZERO.md
    ├── FEATURES_EXPLAINED_FROM_ZERO.md
    ├── INPUT_OUTPUT_PIPELINE_EXPLAINED.md
    ├── DIAGRAMS_FOR_UNDERSTANDING.md
    ├── THESIS_STORY_FROM_ZERO.md
    ├── POLICY_EXPLAINED_SIMPLY.md
    ├── NODE_VS_GRAPH_LEVEL_EXPLAINED.md
    ├── UQ_WORKFLOW_EXPLAINED.md
    ├── MORNING_EVENING_NIGHT_STATUS.md
    ├── REPO_OVERVIEW_VERIFIED.md       ← this file
    ├── CLEANUP_PLAN.md
    ├── MEETING_GUIDED_NOTES_HINGLISH.md
    ├── CHEAT_SHEET.md
    ├── WORK_TIMELINE_FROM_COMMUNICATIONS.md
    ├── SUPERVISOR_CONTEXT_NOTES.md
    ├── FIGURE_REGENERATION_PLAN.md
    ├── GRAPH_EXPLANATION_GUIDE.md
    ├── COLAB_RUNBOOK.md
    ├── REMAINING_WORK_AND_ACTION_PLAN.md
    └── figures/
        └── generate_verified_figures.py
```

---

## Key Numbers at a Glance

| Item | Value | Source |
|------|-------|--------|
| Best model | Trial 8 | test_evaluation_complete.json |
| Best R² | 0.5957 | T8 test_evaluation_complete.json |
| Best MAE | 3.96 veh/h | T8 test_evaluation_complete.json |
| Best RMSE | 7.12 veh/h | T8 test_evaluation_complete.json |
| Best UQ (Spearman ρ) | 0.4820 | T8 mc_dropout_full_metrics_model8_mc30_100graphs.json |
| Conformal 90% coverage | 90.02%, width ±9.92 veh/h | T8 conformal_standard.json |
| Conformal 95% coverage | 95.01%, width ±14.68 veh/h | T8 conformal_standard.json |
| MC Dropout inference time | 228.25 min (100 graphs × 30 samples) | T8 mc_dropout json |
| Paris network nodes | 31,635 per graph (after LineGraph) | test_evaluation_complete.json |
| Dataset size used | 1,000 of 10,000 scenarios | data_preprocessing scripts |
| Input features | 5 (VOL_BASE, CAP_BASE, CAP_REDUC, FREESPEED, LENGTH) | process_simulations_for_gnn.py |

---

## What Does NOT Exist in This Repo

- Temperature Scaling results (ECE 0.356→0.033, T=2.90): **NO JSON file found** — do not claim
- Calibration curve plots: not verified
- Trained model weights (.pt/.pth files): present but not audited here
- Raw MATSim .xml outputs: not directly inspected

---

## Safe Thesis Sentences About the Repo

**SAFE:** "The repository contains 8 training trials of a PointNetTransfGAT architecture, with Trial 8 achieving the best predictive performance (R²=0.5957, MAE=3.96 veh/h) and the highest MC Dropout uncertainty correlation (ρ=0.4820)."

**SAFE:** "All reported metrics are sourced directly from JSON evaluation files generated during training and inference."

**UNSAFE:** "Results were verified against an earlier summary document." (The earlier doc has wrong values.)
