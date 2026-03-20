# Selective Prediction Analysis — Trial 8

## Metadata

- **Source file:** `trial8_uq_ablation_results.csv`
- **Trial:** T8 — `point_net_transf_gat_8th_trial_lower_dropout`
- **Architecture:** PointNetTransfGAT, GATConv(64→1) final layer, dropout=0.2
- **MC samples used:** 30 forward passes per node
- **Data scope:** Results based on 1,000 of 10,000 available MATSim scenarios (10% subset), Trial 8 test set.
- **T1 note:** Trial 1 used Linear(64→1) final layer and is NOT comparable to T2–T8. All results here are T8 only.

---

## Metric Definitions

- **Retained (%):** Fraction of test-set node predictions kept after rejecting those with highest MC σ
- **n nodes:** Absolute count of node-predictions retained (from 3,163,500 total)
- **MAE:** Mean Absolute Error between `target` and `pred_mc_mean` on retained subset (veh/h)
- **RMSE:** Root Mean Squared Error between `target` and `pred_mc_mean` on retained subset (veh/h)
- **Sorting key:** `pred_mc_std` (MC Dropout σ), descending — highest-uncertainty predictions rejected first

---

## Result Table

| Retained (%) | n nodes | MAE (veh/h) | RMSE (veh/h) |
|---|---|---|---|
| 100.0 | 3,163,500.0 | 3.9448 | 7.2036 |
| 95.0 | 3,005,325.0 | 3.4725 | 6.1026 |
| 90.0 | 2,847,150.0 | 3.2166 | 5.6603 |
| 85.0 | 2,688,975.0 | 3.0402 | 5.3945 |
| 80.0 | 2,530,800.0 | 2.9023 | 5.1971 |
| 75.0 | 2,372,625.0 | 2.7844 | 5.0314 |
| 70.0 | 2,214,450.0 | 2.6775 | 4.8785 |
| 60.0 | 1,898,100.0 | 2.4873 | 4.6111 |
| 50.0 | 1,581,750.0 | 2.3052 | 4.358 |
| 40.0 | 1,265,400.0 | 2.1162 | 4.1015 |
| 30.0 | 949,050.0 | 1.9049 | 3.8156 |
| 25.0 | 790,875.0 | 1.7743 | 3.6558 |
| 10.0 | 316,350.0 | 1.0251 | 2.739 |

**MC baseline (100% retained, no abstention):** MAE = 3.9448 veh/h, RMSE = 7.2036 veh/h  
**Deterministic baseline (100% retained):** MAE = 3.9573 veh/h, RMSE = 7.1183 veh/h

---

## Key Results

| Retention | MAE (veh/h) | MAE reduction vs MC baseline |
|---|---|---|
| 100% (no abstention) | 3.9448 | — |
| 90% | 3.2166 | −18.5% |
| 50% | 2.3052 | −41.6% |
| 25% | 1.7743 | −55.0% |

---

## Thesis Usage

**Safe to include:** YES

### Sentence you CAN write:
> "Applying uncertainty-based abstention using MC Dropout σ, retaining the 90% of
> predictions with the lowest uncertainty reduces MAE from 3.94 to
> 3.22 veh/h (a 18.5% reduction). Retaining only 50% of predictions
> further reduces MAE to 2.31 veh/h (41.6% reduction), demonstrating
> that σ meaningfully ranks prediction reliability on the Trial 8 test set."

### Sentence to AVOID:
> Do NOT write "uncertainty is well-calibrated" — this result shows ranking utility
> (Spearman-style operational evidence), not interval calibration. The calibration
> analysis is separate (conformal prediction results). Do NOT extrapolate these
> retention curves to T1, which has a different final-layer architecture.

---

## Figure

`docs/verified/figures/t8_selective_prediction_curve.pdf`  
`docs/verified/figures/t8_selective_prediction_curve.png`
