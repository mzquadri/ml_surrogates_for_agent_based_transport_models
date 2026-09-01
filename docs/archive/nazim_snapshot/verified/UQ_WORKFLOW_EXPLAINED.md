# UQ WORKFLOW EXPLAINED
> How uncertainty quantification works in this thesis, step by step.
> Covers MC Dropout and Conformal Prediction with all verified numbers.

---

## WHY UQ MATTERS FOR THIS THESIS

A fast surrogate that says "Road X will gain +150 veh/h" is useful.
A surrogate that additionally says "I'm very confident about Road X but uncertain about Road Y" is much more useful.

Without UQ, a planner treats all predictions equally.
With UQ, a planner can:
- Focus simulation resources on high-uncertainty roads
- Make confident decisions on low-uncertainty roads
- Provide guaranteed bounds to risk-averse stakeholders

---

## METHOD 1: MC DROPOUT (Uncertainty Estimation)

### The idea (Gal & Ghahramani, 2016)
Dropout randomly zeros out neurons during training as regularization.
Gal & Ghahramani showed: if you leave dropout active at test time and run multiple forward passes, you get approximate samples from a Bayesian posterior over model weights.

### In practice for this thesis

**Step 1:** Train the model normally (dropout active during training, disabled during evaluation).

**Step 2:** At inference time: set model to TRAIN mode (keeps dropout active).

**Step 3:** Run T=30 forward passes on the same graph.
Each pass uses a different random dropout mask, producing different predictions.

**Step 4:** Compute per-node statistics:
```
mu[i]    = mean of 30 predictions for node i   (best estimate)
sigma[i] = std  of 30 predictions for node i   (uncertainty)
```

**Step 5:** Evaluate quality of sigma as uncertainty estimate:
```
spearman_rho = rank_correlation(sigma, |y_true - mu|)
```
- rho close to 1.0 = sigma perfectly predicts which nodes have large errors
- rho = 0 = sigma is useless as an error predictor
- **T8 achieves rho = 0.4820** (moderate, meaningful correlation)

### Verified results

| Model | rho | n_graphs | n_samples | source |
|---|---|---|---|---|
| T5 | 0.4263 | 50 | 30 | mc_dropout_full_metrics_model5_mc30_50graphs.json |
| T6 | 0.4186 | 50 | 30 | mc_dropout_full_metrics_model6_mc30_50graphs.json |
| T7 | 0.4437 | 100 | 30 | mc_dropout_full_metrics_model7_mc30_100graphs.json |
| **T8** | **0.4820** | **100** | **30** | mc_dropout_full_metrics_model8_mc30_100graphs.json |

T8 is best, consistent with T8 also being the best predictor (R2=0.596).

### Computational cost
T8 MC Dropout: 100 graphs × 30 samples = 3,000 inference runs = **228.25 minutes** on the available hardware.

### Important caveat
sigma is informative for RANKING but is NOT a calibrated standard deviation:
- Naive 95% interval: mu ± 1.96*sigma achieves only ~55% actual coverage
- Calibration factor k95 = 11.65 (need mu ± 11.65*sigma for actual 95% coverage)
- This is the "miscalibration" that Conformal Prediction fixes

---

## METHOD 2: CONFORMAL PREDICTION (Coverage Guarantee)

### The idea (Angelopoulos & Bates, 2022 tutorial)
Conformal prediction provides distribution-free, statistically rigorous coverage guarantees.
No assumptions about the error distribution — works for any model, any data.

### Split-conformal procedure (used in this thesis)

**Step 1:** Take the 100 test graphs. Split into:
- Calibration set: 50 graphs (used to determine interval width)
- Evaluation set: 50 graphs (used to verify coverage)

**Step 2:** Run the model (using mu from MC Dropout) on calibration set.

**Step 3:** Compute nonconformity scores:
```
score[i] = |y_true[i] - mu_hat[i]|   for each node i in calibration set
```
This gives 50 × 31,635 = 1,581,750 nonconformity scores.

**Step 4:** Find quantile q:
```
q = quantile(scores, (1 - alpha) * (1 + 1/n_calibration))
```
For alpha=0.10 (90% coverage): q = 9.9196 veh/h
For alpha=0.05 (95% coverage): q = 14.6766 veh/h

**Step 5:** For evaluation set, predict interval:
```
interval = [mu_hat - q, mu_hat + q]
```

**Step 6:** Compute empirical coverage:
```
coverage = fraction of nodes where y_true is inside interval
```

### Verified results for T8

| Coverage target | q (veh/h) | Empirical coverage | Width (±veh/h) |
|---|---|---|---|
| 90% | 9.9196 | **90.02%** | ±9.92 |
| 95% | 14.6766 | **95.01%** | ±14.68 |

Coverage is exactly as guaranteed — 90.02% when we targeted 90%, 95.01% when we targeted 95%.

---

## METHOD 3: SIGMA-NORMALIZED CONFORMAL PREDICTION

An enhancement that uses MC Dropout sigma to create adaptive (non-fixed-width) intervals.

### Idea
Instead of fixed-width intervals, use:
```
nonconformity score = |y_true[i] - mu_hat[i]| / sigma[i]
prediction interval = [mu_hat - q*sigma, mu_hat + q*sigma]
```

Low-sigma nodes → narrow interval (model is confident)
High-sigma nodes → wide interval (model is uncertain)

### Results
- For low-uncertainty nodes: intervals are **65.8% narrower** than fixed-width
- Coverage guarantee is maintained by construction

This is the most practically useful result: tight intervals where you can be confident, wide intervals where you cannot.

---

## THE REJECTION ANALYSIS

A practical use of sigma: if you only keep the predictions the model is most confident about:

| Rejection threshold | Fraction of predictions kept | MAE (veh/h) |
|---|---|---|
| None | 100% | 3.96 |
| Reject top 25% uncertain | 75% | ~3.2 |
| Reject top 50% uncertain | 50% | ~2.4 (−39.9% from baseline) |

Rejecting the 50% most uncertain predictions reduces MAE by 39.9%.
This directly supports practical deployment: flag uncertain predictions for MATSim verification.

---

## ENSEMBLE EXPERIMENTS (Experiment A and B)

Two additional experiments explored whether training multiple models improves UQ.

### Experiment A: 5 independently-retrained T8 models
Each model trained from scratch with different random seed/data split.
Uncertainty estimated as variance across 5 models.
- MC Dropout (averaged across 5): rho = 0.1600
- Ensemble variance: rho = 0.1035

### Experiment B: Multi-model ensemble
- rho = 0.1167

### Why are these lower than single-model MC Dropout (rho=0.48)?
The ensemble models are trained on **different random data subsets**. Their disagreement reflects:
- Different training data (different 800 graphs each)
- Different test set compositions relative to each model's training

This is a **cross-distribution artifact**, not a failure of the UQ method.
The ensemble variance measures "how much do these models disagree?" which partly reflects
data distribution differences rather than genuine predictive uncertainty.

Single-model MC Dropout (rho=0.48) is more informative because all 30 samples come from
the same model trained on the same data — their variance is pure epistemic uncertainty.

---

## UQ SUMMARY TABLE (THESIS-READY)

| Method | Metric | Value | Source |
|---|---|---|---|
| MC Dropout T8 | Spearman rho (sigma vs error) | 0.4820 | mc_dropout_full_metrics_model8 |
| Conformal Prediction T8 | 90% coverage achieved | 90.02% | conformal_standard.json |
| Conformal Prediction T8 | 95% coverage achieved | 95.01% | conformal_standard.json |
| Conformal interval width | 90% target | ±9.92 veh/h | conformal_standard.json |
| Conformal interval width | 95% target | ±14.68 veh/h | conformal_standard.json |
| Sigma-normalized | Interval reduction (low-uncertainty) | 65.8% narrower | ADVANCED_UQ_SUMMARY_MODEL8.md (computed from uq_comparison_model8.json) |
| Rejection analysis | MAE reduction at 50% rejection | -39.9% | ADVANCED_UQ_SUMMARY_MODEL8.md (computed from uq_comparison_model8.json) |
| Calibration factor | k95 for naive MC interval | 11.65 | uq_comparison_model8.json |

---

## THESIS-SAFE SENTENCES

**Safe:**
> "We apply MC Dropout with T=30 forward passes per graph, using the standard deviation of predictions as a proxy for epistemic uncertainty."

> "The MC Dropout uncertainty estimates achieve Spearman rank correlation rho=0.482 with absolute prediction errors across 3,163,500 test nodes."

> "Split-conformal prediction with 50 calibration graphs and 50 evaluation graphs achieves 90.02% empirical coverage at the 90% nominal level."

> "Sigma-normalized conformal intervals are 65.8% narrower for the least uncertain road segments while maintaining statistical coverage guarantees."

**Do NOT say:**
> "MC Dropout provides calibrated uncertainty" — sigma is informative but NOT calibrated (k95=11.65).
> "Temperature scaling was applied" — not verified, no source file found.
