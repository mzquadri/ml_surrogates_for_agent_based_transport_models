# Calibration Audit -- Trial 7

## Metadata

- **Source (MC):** `data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz`
- **Trial:** T7 -- `point_net_transf_gat_7th_trial_80_10_10_split`
- **Architecture:** PointNetTransfGAT, GATConv(64->1) final layer, MC Dropout=0.2
- **MC samples:** 30 forward passes per node
- **Data scope:** Results based on 1,000 of 10,000 available MATSim scenarios (10% subset), Trial 7 test set.
- **Split:** First 20% of rows = 632,700 nodes (calibration); remaining 80% = 2,530,800 nodes (test)
- **Split note:** Same 20/80 methodology as Part 3 (T8) for direct comparability.
  The file `conformal_standard.json` in the T7 folder used a 50/50 split (n_cal=1,581,750)
  and will therefore show different q values -- this is expected and documented.
- **T1 note:** Trial 1 uses Linear(64->1) final layer and is excluded from all T2-T8 analyses.

---

## Method Definitions

- **Raw MC Gaussian:** z = norm.ppf((1+p)/2); interval yhat +/- z*sigma; no calibration used
- **Empirical k:** p-th quantile of |residual|/sigma on TEST set; k >> z = heavy tails
- **Global conformal:** conformal_q(|y_cal - yhat_cal|, alpha); constant-width interval
- **Adaptive conformal:** conformal_q(|y_cal - yhat_cal|/(sigma_cal+eps), alpha); sigma-scaled
- **conformal_q formula:** q_level = ceil((n+1)*(1-alpha))/n; q = quantile(r, q_level, method='higher')
  (faithfully reproduced from scripts/evaluation/conformal_from_mc.py)
- **eps = 1e-6** in all sigma divisions

---

## Full Results Table

| Nominal | z (Gauss) | Raw MC Cov | Raw MC Width | Emp. k | q_global | Global Cov | Global Width | q_adapt | Adapt Cov | Adapt Width |
|---|---|---|---|---|---|---|---|---|---|---|
| 50% | 0.6745 | 17.95% | 1.63 | 2.0408 | 1.8017 | 50.38% | 3.60 | 2.0484 | 50.15% | 4.96 |
| 70% | 1.0364 | 26.82% | 2.51 | 3.7587 | 3.9190 | 70.39% | 7.84 | 3.7865 | 70.19% | 9.17 |
| 80% | 1.2816 | 32.75% | 3.10 | 5.8464 | 6.0672 | 80.39% | 12.13 | 5.8979 | 80.17% | 14.28 |
| 90% | 1.6449 | 41.45% | 3.98 | 10.4534 | 10.3877 | 90.18% | 20.78 | 10.4911 | 90.05% | 25.40 |
| 95% | 1.9600 | 48.38% | 4.75 | 16.1445 | 15.6478 | 95.11% | 31.30 | 16.1962 | 95.03% | 39.22 |

*All widths in veh/h.*

---

## Key Findings

### 1. Raw MC severely undercovers at all levels (same as T8)
T7 raw MC coverage at 90% = 41.45% vs nominal 90%.
T8 raw MC coverage at 90% = 49.26% (from Part 3).
Both trials confirm the same pathology.

### 2. Empirical k confirms heavy-tailed residuals
- k at 95%: 16.1445 vs z = 1.9600 (ratio = 8.24x)
- T8 k95 = 11.3407 (from Part 3 / diagnostics.json)
- Both trials show k >> z, confirming non-Gaussian residuals.

### 3. Global conformal meets nominal coverage at all levels
Empirically verified across all five nominal levels.
T7 q_global@90% = 10.3877 veh/h  vs  T8 q_global@90% = 9.9933 veh/h.
T7 q_global@95% = 15.6478 veh/h  vs  T8 q_global@95% = 14.7709 veh/h.
T7 requires wider intervals (higher RMSE -> larger residuals).

### 4. Adaptive conformal also meets nominal at all levels

---

## Comparison to Trial 8

| Metric | T7 | T8 | Notes |
|---|---|---|---|
| Raw MC cov @ 90% | 41.45% | 49.26% | Both severely undercover |
| k_empirical @ 95% | 16.1445 | 11.3407 | Both >> z=1.96 |
| q_global @ 90% | 10.3877 | 9.9933 | T7 needs wider intervals |
| q_global @ 95% | 15.6478 | 14.7709 | T7 needs wider intervals |
| Global cov @ 90% | 90.18% | 90.17% | Both meet nominal |
| Global cov @ 95% | 95.11% | 95.09% | Both meet nominal |

---

## Can I Safely Put This in My Thesis?

| Claim | Safe? | Notes |
|---|---|---|
| "T7 raw MC undercovers at all nominal levels" | **YES** | Directly computed |
| "T7 conformal achieves nominal coverage at all levels" | **YES** | Verified empirically |
| "T7 k95 >> z, confirming heavy tails" | **YES** | Computed from test set |
| "The calibration failure of raw MC is not T8-specific" | **YES** | Both T7 and T8 show same pattern |
| "T7 sigma is a calibrated uncertainty estimate" | **NO** | Same failure mode as T8 |

---

## Safe Thesis Sentence

"Calibration analysis on Trial 7 confirms that the raw MC Dropout interval
miscalibration observed in Trial 8 is not trial-specific: raw MC coverage at the
90% nominal level is 41.45%, while global conformal prediction
achieves 90.18% coverage. The empirical k factor at 95% is
k = 16.145 (vs Gaussian z = 1.9600), consistent with the
heavy-tailed residual distribution found in Trial 8."

## Sentence to Avoid

Do NOT write "T7 and T8 produce identical calibration results." The q values differ
because T7 has higher RMSE (larger residuals -> wider conformal intervals). The
pattern (undercoverage + conformal guarantee) is consistent, but the magnitudes differ.

---

## Figure

`docs/verified/figures/t7_calibration_curve.pdf`
`docs/verified/figures/t7_calibration_curve.png`
`docs/verified/figures/t7_interval_width_comparison.pdf`
`docs/verified/figures/t7_interval_width_comparison.png`
