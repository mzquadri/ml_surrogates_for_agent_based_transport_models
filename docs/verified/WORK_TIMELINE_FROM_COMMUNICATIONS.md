# WORK_TIMELINE_FROM_COMMUNICATIONS.md
# Work Timeline — Inferred from Repository Evidence
# IMPORTANT: All dates/phases marked [INFERRED FROM REPO] unless explicitly stated
# Last verified: 2026-03-14

---

## Disclaimer

This timeline is **inferred from folder names, JSON file naming conventions, and the
logical sequence of trials** — NOT from email timestamps or explicit dated records.
Do not present this as fact in the thesis. Use it as a personal orientation guide.

---

## Phase 0: Problem Setup and Data Preparation
*[INFERRED: early in project, before any training trials]*

**Evidence:**
- `scripts/data_preprocessing/process_simulations_for_gnn.py` — LineGraph transform, graph construction
- `scripts/data_preprocessing/help_functions.py` — target variable definition
- 1,000 of 10,000 scenarios selected [INFERRED: compute/storage constraint identified early]
- Paris road network chosen (31,635 nodes per graph after LineGraph)

**What happened:**
- MATSim simulation data obtained for Paris
- Preprocessing pipeline written: raw simulation → PyG graph objects
- Target variable defined: `edge_car_volume_difference = vol_car - vol_base_case`
- LineGraph transform applied: roads become nodes, junctions become edges
- Decision to use 5 features (VOL_BASE, CAPACITY_BASE, CAPACITY_REDUCTION, FREESPEED, LENGTH)

---

## Phase 1: First Architecture Attempt (Trial 1)
*[INFERRED: early training phase]*

**Folder:** `pointnet_transf_gat_1st_bs32_5feat_seed42/`
**Key indicators:** `bs32` in name (batch=32), `seed42`, `5feat`

**What happened:**
- First PointNetTransfGAT trained: batch=32, LR=0.001, no dropout, 80/15/5 split
- Achieved R²=0.7860, MAE=2.97, RMSE=5.40
- **Later excluded** from comparison — [INFERRED: architecture was different from T2-T8;
  no dropout capability or different structure]
- Decision: cannot compare T1 to T2-T8 on equal footing

---

## Phase 2: Architecture Standardization and Ablation Trials (Trials 2–6)
*[INFERRED: iterative ablation period, all using 80/15/5 split]*

**Pattern:** Folder names change from `1st` to sequential trial numbering.
The `2nd_try` name suggests T2 was the first "real" attempt with the finalized architecture.

### Trial 2 — Baseline with finalized architecture
- Folder: `point_net_transf_gat_2nd_try/`
- batch=16, LR=0.0005, dropout=0.3, no weighted loss
- R²=0.5117 — major drop from T1 (expected: T1 excluded, different setup)

### Trial 3 — Weighted loss experiment (no dropout)
- Folder: `point_net_transf_gat_3rd_trial_weighted_loss/`
- Name explicitly says `weighted_loss`; dropout=0.0 (removed to test cleanly)
- R²=0.2246 — weighted loss severely hurts performance

### Trial 4 — Weighted loss with dropout
- Folder: `point_net_transf_gat_4th_trial_weighted_loss/`
- Name also says `weighted_loss`; dropout=0.3 restored
- R²=0.2426 — still poor; confirms weighted loss is the problem

### Trial 5 — Return to no weighted loss, smaller batch
- Folder: `point_net_transf_gat_5th_try/`
- batch=8 (halved from T2), no weighted loss
- R²=0.5553 — recovery and improvement
- **First UQ experiments run** on T5 (uq_results/ appears here)

### Trial 6 — Lower learning rate
- Folder: `point_net_transf_gat_6th_trial_lower_lr/`
- LR reduced: 0.0005 → 0.0003
- R²=0.5223 — slight regression; lower LR not better here
- UQ also run on T6

**[INFERRED conclusion from T2–T6]:** Weighted loss is harmful; batch=8 helps;
the sweet spot for LR is around 0.0005; dropout=0.3 is baseline.

---

## Phase 3: Data Split Change and Dropout Tuning (Trials 7–8)
*[INFERRED: late optimization phase — changed from 80/15/5 to 80/10/10]*

### Trial 7 — Larger test set
- Folder: `point_net_transf_gat_7th_trial_80_10_10_split/`
- Name explicitly encodes the split change: `80_10_10_split`
- batch=8, LR=0.0006, dropout=0.3
- R²=0.5471 — more test graphs (100 vs 50) → more reliable evaluation
- UQ run: ρ=0.4437

### Trial 8 — Final best model
- Folder: `point_net_transf_gat_8th_trial_lower_dropout/`
- Name: `lower_dropout` — dropout reduced from 0.3 → 0.2
- batch=8, LR=0.0005, dropout=0.2, 80/10/10 split
- R²=0.5957 — best model
- **Full UQ suite run:** MC Dropout, conformal prediction, ensemble experiments
- Feature analysis also run (feature_analysis_plots/)

**[INFERRED conclusion from T7–T8]:** 80/10/10 provides more reliable test evaluation;
reducing dropout slightly (0.3→0.2) gives the best performance.

---

## Phase 4: Full UQ Study on Best Model (T8)
*[INFERRED: final experimental phase]*

**Evidence:** T8 uq_results/ is the only trial with:
- Conformal prediction results
- Ensemble experiments (experiment_a, experiment_b)
- uq_comparison_model8.json (k95, selective prediction, sigma-normalized)
- ADVANCED_UQ_SUMMARY_MODEL8.md

**What happened:**
- MC Dropout: 100 test graphs × 30 samples = 3,163,500 nodes evaluated
- Conformal prediction calibrated and evaluated (50/50 split)
- Ensemble Experiment A: 5 independent runs of same model (ρ=0.1600)
- Ensemble Experiment B: Multi-model ensemble T7+T8 (ρ=0.1167)
- Comparison shows MC Dropout dominates ensemble for UQ quality

---

## Phase 5: Documentation and Verification
*[This session — 2026-03-14]*

- All results verified against JSON files
- docs/MEETING_PREPARATION.md found to have wrong hyperparameters
- docs/verified/ created with 23 authoritative files
- generate_thesis_charts.py found to have wrong hardcoded values

---

## Summary Timeline (condensed)

```
[Phase 0]  Data prep → LineGraph → 1,000 scenarios → 5 features
[Phase 1]  T1: first model, R²=0.786, later excluded (different arch)
[Phase 2]  T2–T6: ablation of weighted loss, batch size, LR → R² 0.51–0.56
[Phase 3]  T7–T8: split change + dropout tuning → best R²=0.5957
[Phase 4]  Full UQ on T8: MC Dropout ρ=0.482, conformal 95% coverage
[Phase 5]  THIS SESSION: verification + docs/verified/
```

---

## What This Tells the Thesis Story

This timeline maps neatly onto a thesis narrative:
1. Problem: MATSim is slow → need surrogate
2. Architecture choice: PointNetTransfGAT for spatial road network data
3. Iterative refinement: weighted loss fails, smaller batch helps, lower dropout wins
4. UQ: MC Dropout outperforms ensemble; conformal gives guaranteed intervals
5. Conclusion: T8 is the recommendation; MC Dropout + conformal for deployment
