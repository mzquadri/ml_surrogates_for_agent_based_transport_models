# Selective Prediction Analysis -- Trial 7

## Metadata

- **Source (MC):** `data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz`
- **Source (Det):** `data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/deterministic_full_100graphs.npz`
- **Trial:** T7 -- `point_net_transf_gat_7th_trial_80_10_10_split`
- **Architecture:** PointNetTransfGAT, GATConv(64->1) final layer, MC Dropout=0.2
- **MC samples:** 30 forward passes per node
- **Data scope:** Results based on 1,000 of 10,000 available MATSim scenarios (10% subset), Trial 7 test set.
- **T1 note:** Trial 1 uses Linear(64->1) final layer and is architecturally distinct. Not compared here.

---

## Method

- Sort all 3,163,500 test-set node predictions by `uncertainties` (pred_mc_std) descending
- For each retention fraction r, keep the bottom floor(r*N) nodes (lowest sigma)
- Compute MAE and RMSE of `|targets - predictions|` on the retained subset
- **Uncertainty signal used:** MC Dropout sigma (30 passes)

---

## Result Table

| Retained (%) | n nodes | MAE (veh/h) | RMSE (veh/h) |
|---|---|---|---|
| 100.0 | 3,163,500.0 | 4.0737 | 7.6202 |
| 95.0 | 3,005,325.0 | 3.5597 | 6.4348 |
| 90.0 | 2,847,150.0 | 3.3156 | 6.0181 |
| 85.0 | 2,688,975.0 | 3.1526 | 5.753 |
| 80.0 | 2,530,800.0 | 3.0313 | 5.5703 |
| 75.0 | 2,372,625.0 | 2.9309 | 5.4218 |
| 70.0 | 2,214,450.0 | 2.8423 | 5.291 |
| 60.0 | 1,898,100.0 | 2.6777 | 5.0391 |
| 50.0 | 1,581,750.0 | 2.5134 | 4.7997 |
| 40.0 | 1,265,400.0 | 2.3331 | 4.5399 |
| 30.0 | 949,050.0 | 2.1204 | 4.2421 |
| 25.0 | 790,875.0 | 1.9913 | 4.0711 |
| 10.0 | 316,350.0 | 1.2314 | 3.078 |

**MC baseline (100%, no abstention):** MAE = 4.0737 veh/h, RMSE = 7.6202 veh/h
**Deterministic baseline (100%):**     MAE = 4.0601 veh/h, RMSE = 7.5343 veh/h

---

## Key Results

| Retention | MAE (veh/h) | MAE reduction |
|---|---|---|
| 100% (no abstention) | 4.0737 | -- |
| 90% | 3.3156 | -18.6% |
| 50% | 2.5134 | -38.3% |
| 25% | 1.9913 | -51.1% |

---

## Comparison to Trial 8

| Metric | T7 | T8 | Direction |
|---|---|---|---|
| Baseline MAE (100%) | 4.0737 | 3.9448 | T8 lower (better) |
| MAE at 90% retention | 3.3156 | 3.2166 | T8 lower |
| MAE at 50% retention | 2.5134 | 2.3052 | T8 lower |
| MAE at 25% retention | 1.9913 | 1.7743 | T8 lower |
| MAE reduction at 50% | 38.3% | 41.6% | comparable |

T8 reference values from Part 2A (UQ_SELECTIVE_PREDICTION_T8.md).

---

## Can I Safely Put This in My Thesis?

| Claim | Safe? | Notes |
|---|---|---|
| "T7 sigma abstention reduces MAE at all retention levels" | **YES** | Directly computed from NPZ |
| "T7 shows the same monotone improvement pattern as T8" | **YES** | Consistent with T8 Part 2A results |
| "T7 performance is lower than T8 across all retention levels" | **YES** | T8 has lower baseline MAE |
| "sigma is a reliable absolute error predictor for T7" | **NO** | It is a ranking signal; do not claim calibration |

---

## Safe Thesis Sentence

"Applying uncertainty-based abstention to Trial 7, retaining the 90% of
predictions with lowest MC Dropout sigma reduces MAE from 4.07 to
3.32 veh/h (-18.6%), and retaining 50% reduces MAE to 2.51 veh/h
(-38.3%), confirming that the sigma ranking utility observed in Trial 8 is
not trial-specific."

## Sentence to Avoid

Do NOT write "Trial 7 uncertainty is calibrated." Sigma is operationally useful
for ranking but not a calibrated predictive interval. Do NOT compare T7 directly
to T1 (different final-layer architecture).

---

## Figure

`docs/verified/figures/t7_selective_prediction_curve.pdf`
`docs/verified/figures/t7_selective_prediction_curve.png`
