# CHEAT_SHEET.md
# One-Page Thesis Cheat Sheet — All Verified Numbers
# Last verified: 2026-03-14 | Print this and bring to your meeting/defense

---

## THE PROBLEM
MATSim (agent-based simulator) = hours per run. GNN surrogate = seconds.
Predict: edge_car_volume_difference = vol_car_policy - vol_car_baseline (veh/h per road segment)
Dataset: Paris road network, 1,000 of 10,000 available scenarios

---

## THE MODEL (PointNetTransfGAT)

```
Input: 5 features per node (road segment after LineGraph transform)
  [0] VOL_BASE_CASE     baseline car volume (veh/h)       corr: +0.332
  [1] CAPACITY_BASE     road capacity (veh/h)              corr: +0.262
  [2] CAPACITY_REDUC    policy-induced cap. change         corr: -0.229
  [3] FREESPEED         free-flow speed                    corr: +0.211
  [4] LENGTH            road segment length (m)            corr: -0.070

Architecture:
  PointNetConv (START pos) → PointNetConv (END pos)
  → TransformerConv 128→256 (4 heads) + ReLU [+ dropout]
  → TransformerConv 256→512 (4 heads) + ReLU [+ dropout]
  → GATConv 512→64
  → GATConv 64→1  ← output: predicted volume change

Output: scalar per node = predicted Δcar_volume (veh/h)
Nodes per graph: 31,635  |  Test graphs T8: 100  |  Test nodes: 3,163,500
```

---

## PERFORMANCE — ALL 8 TRIALS

| Trial | Batch | LR     | Dropout | Weighted | Split     |   R²   |  MAE  | RMSE  |
|-------|-------|--------|---------|----------|-----------|--------|-------|-------|
| T1    | 32    | 0.001  | 0.0     | No       | 80/15/5   | 0.7860*| 2.97  | 5.40  |
| T2    | 16    | 0.0005 | 0.3     | No       | 80/15/5   | 0.5117 | 4.33  | 8.15  |
| T3    | 16    | 0.0005 | 0.0     | Yes      | 80/15/5   | 0.2246 | 5.99  | 10.27 |
| T4    | 16    | 0.0005 | 0.3     | Yes      | 80/15/5   | 0.2426 | 6.08  | 10.15 |
| T5    |  8    | 0.0005 | 0.3     | No       | 80/15/5   | 0.5553 | 4.24  | 7.78  |
| T6    |  8    | 0.0003 | 0.3     | No       | 80/15/5   | 0.5223 | 4.32  | 8.06  |
| T7    |  8    | 0.0006 | 0.3     | No       | 80/10/10  | 0.5471 | 4.06  | 7.53  |
|**T8** |**8**  |**5e-4**|**0.2**  |**No**    |**80/10/10**|**0.5957**|**3.96**|**7.12**|

*T1 excluded from comparison — T1 uses Linear(64→1) as final layer; T2–T8 use GATConv(64→1). Different architecture, not directly comparable. T1 was trained with an older code version.

---

## UQ RESULTS — MC DROPOUT (BEST: T8)

| Model | ρ (Spearman) | n_nodes   | MC samples | Source |
|-------|-------------|-----------|------------|--------|
| T8    | **0.4820**  | 3,163,500 | 30         | JSON verified |
| T7    | 0.4437      | 3,163,500 | 30         | JSON verified |
| T5    | 0.4263      | 1,581,750 | 30         | JSON verified |
| T6    | 0.4186      | 1,581,750 | 30         | JSON verified |
| Exp A (5-run avg)  | 0.1600 | 3,163,500 | — | experiment_a |
| Exp A (ensemble)   | 0.1035 | 3,163,500 | — | experiment_a |
| Exp B (multi-model)| 0.1167 | —         | — | experiment_b |

---

## CONFORMAL PREDICTION (T8)

| Level | Quantile q   | Coverage  | Interval Width |
|-------|-------------|-----------|----------------|
| 90%   | 9.9196 veh/h | 90.02%   | ± 9.92 veh/h   |
| 95%   | 14.6766 veh/h| 95.01%   | ± 14.68 veh/h  |

Setup: 100 test graphs split 50/50 → 50 calibration + 50 evaluation

---

## KEY UQ FACTS (T8)

```
k95 = 11.65         (empirical: need ±11.65σ for 95% coverage — σ NOT calibrated)
65.8%               narrower intervals for low-uncertainty predictions (σ-normalized conformal)
39.9%               MAE reduction when rejecting top 50% uncertain predictions
228.25 min          MC Dropout inference time (100 graphs × 30 samples)
```

---

## WHAT IS CALIBRATED vs NOT

```
MC Dropout σ    → NOT calibrated (k95=11.65, not 1.96)
                   Use for: ranking/ordering uncertainty
                   Do NOT use for: probability intervals

Conformal PI    → CALIBRATED by construction
                   Use for: "I guarantee X% of true values are in this interval"
```

---

## SAFE THESIS SENTENCES

```
SAFE:   "Trial 8 achieves R²=0.5957, MAE=3.96 veh/h, RMSE=7.12 veh/h."
SAFE:   "MC Dropout uncertainty is positively correlated with prediction error
         (Spearman ρ=0.4820 on 3.16M test nodes)."
SAFE:   "Conformal prediction achieves 90.02% empirical coverage with intervals
         of width ±9.92 veh/h at the 90% confidence level."
SAFE:   "Rejecting the 50% most uncertain predictions reduces MAE by 39.9%."

UNSAFE: "Temperature Scaling reduced ECE from 0.356 to 0.033." [NOT VERIFIED]
UNSAFE: "R² is high." [It is moderate — say 'moderate' or 'reasonable']
UNSAFE: "The model is well-calibrated." [Only conformal is calibrated]
UNSAFE: "Trial 1 is the best baseline." [T1 has different architecture — excluded]
```

---

## FILE LOCATIONS (quick reference)

```
Best model results:      data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/
  Metrics:               test_evaluation_complete.json
  MC Dropout UQ:         uq_results/mc_dropout_full_metrics_model8_mc30_100graphs.json
  Conformal:             uq_results/conformal_standard.json
  UQ comparison:         uq_results/uq_comparison_model8.json
  Ensemble:              uq_results/ensemble_experiments/experiment_{a,b}_results.json

All verified docs:       docs/verified/
Figure script:           docs/verified/figures/generate_verified_figures.py
```

---

## ARCHITECTURE (for whiteboard explanation)

```
Road network → LineGraph() → each road = node, shared junction = edge

                  [5 features per road]
                         ↓
             PointNet (START coordinate)
             PointNet (END coordinate)
                         ↓
          TransformerConv 128→256 (4 heads)
          TransformerConv 256→512 (4 heads)
                         ↓
              GATConv 512→64
              GATConv 64→1
                         ↓
            Δvol_car per road (veh/h)
```
