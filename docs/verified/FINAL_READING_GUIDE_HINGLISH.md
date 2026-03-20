# FINAL READING GUIDE — Thesis Navigation
### "Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models"
### Mohd Zamin Quadri (Nazim) | TUM Master Thesis 2026

---

## TL;DR — Ye thesis kya hai?

Bhai/Behen, ek line mein:
> **Paris ki road network pe ek GNN (Graph Neural Network) train kiya traffic predict karne ke liye, phir MC Dropout se uncertainty estimate ki, aur verify kiya ki woh uncertainty actually kaam karti hai.**

---

## Kya kiya, kya mila?

| Kya | Result |
|-----|--------|
| Best model (T8) ka accuracy | R² = 0.5957, MAE = 3.96 veh/h |
| MC Dropout uncertainty quality | Spearman rho = 0.4820 (uncertainty sahi jagah high hai) |
| Conformal 95% coverage | Guaranteed band = ±14.68 veh/h |
| Selective prediction (50% retain) | MAE 39.9% kam ho jata hai |
| kya sigma calibrated hai? | Nahi — k95 = 11.65, ideal = 1.96 |

---

## Chapters ka map

```
Ch 01 — Introduction       : Problem kya hai, kyun GNN, kyun UQ
Ch 02 — Background         : GNN theory, MC Dropout math, Conformal math
Ch 03 — Methodology        : Architecture detail, UQ methods, evaluation metrics
Ch 04 — Experiments        : Dataset, training setup, experiment design
Ch 05 — Results            : Saare numbers, figures fig1-fig7
Ch 06 — Discussion         : Interpretation, limitations, policy implications
Ch 07 — Conclusion         : Summary, future work
```

---

## 8 Trials ka kya chakkar hai?

- **T1**: Linear(64->1) final layer, 0 dropout — best R²=0.7860, but NO UQ possible (dropout=0 hai)
- **T2–T4**: GATConv final, different configs — moderate performance
- **T5–T8**: GATConv + dropout > 0 — UQ experiments yahan hue
- **T8 best hai** among T2–T8: R²=0.5957, dropout=0.2, batch=8, 80/10/10 split

> Note: T1 directly T2–T8 se compare mat karo. Different final layer hai.

---

## MC Dropout kya hota hai? (S not T)

Normal inference mein dropout band hota hai.
MC Dropout mein **S = 30** baar same input pe forward pass karo with dropout ON.

```
for s in 1..S=30:
    y_hat_s = model(x)  # dropout ON
mean_pred = average(y_hat_1..30)
sigma = std(y_hat_1..30)   # ye uncertainty hai
```

Jitna zyada sigma, utna model uncertain hai us node pe.

**Spearman rho = 0.4820** matlab: jahan sigma high hai, wahan actual error bhi tend to be high.
Ye **ranking signal** hai, calibrated probability nahi.

---

## Conformal Prediction — iska guarantee kya hai?

Split conformal:
- 50 graphs calibration, 50 graphs evaluation
- Nonconformity score = |y - y_hat| per node
- q = (1-alpha) quantile of calibration scores

| Level | q (half-width) | Achieved coverage |
|-------|----------------|-------------------|
| 90%   | ±9.92 veh/h   | 90.02%            |
| 95%   | ±14.68 veh/h  | 95.01%            |

**Guarantee**: on average, at least 95% true values fall inside ±14.68 veh/h.
Ye fixed-width band hai — har node ke liye same width.

---

## Ensemble experiments kyun flop hue?

Exp A aur B mein near-zero R² (~0.003) aaya.
**Reason**: Data distribution mismatch — ensemble evaluation subset training distribution se alag tha.
**T8 model fail nahi hua** — standalone T8 ka R²=0.5957 hai.

Ensemble rho (0.10–0.16) < MC Dropout rho (0.48) — MC Dropout better uncertainty estimator hai.

---

## Sigma calibrated kyun nahi hai?

Ideal Gaussian mein ±1.96*sigma = 95% coverage hona chahiye.
T8 MC Dropout mein ±11.65*sigma chahiye 95% coverage ke liye.

Matlab sigma ~6x too small hai absolute standard deviation ke liye.
**Use case**: ranking aur selective prediction — probability intervals nahi.

---

## Selective prediction kya kaam deta hai?

Sabse low sigma wale nodes rakho (high sigma wale reject):

| Retention | MAE | Improvement |
|-----------|-----|-------------|
| 100% (sab) | 3.96 veh/h | baseline |
| 90% retain | ~3.30 veh/h | -16.8% |
| 50% retain | ~2.38 veh/h | **-39.9%** |

> 50% anchor verified from uq_comparison_model8.json.
> 39.9% figure = 50% retention, 50 eval graphs context.

---

## Kahan kya file hai?

### Source of Truth (JSON — touch mat karo)
```
data/TR-C_Benchmarks/
  ALL_MODELS_COMPARISON/all_models_summary.json        <- T1-T8 metrics
  point_net_transf_gat_8th_trial_lower_dropout/
    uq_results/
      mc_dropout_full_metrics_model8_mc30_100graphs.json  <- rho=0.4820
      conformal_standard.json                              <- q, coverage
      uq_comparison_model8.json                            <- k95, selective
      WITH_WITHOUT_UQ_SUMMARY_MODEL8.md                    <- det vs MC
      ADVANCED_UQ_SUMMARY_MODEL8.md                        <- selective pred
      ensemble_experiments/
        experiment_a_results.json
        experiment_b_results.json
```

### Verified Figures (PDF + PNG)
```
docs/verified/figures/generated/
  fig1_trial_comparison.pdf + .png
  fig2_uq_ranking.pdf + .png
  fig3_conformal_coverage.pdf + .png
  fig4_selective_prediction.pdf + .png
  fig5_feature_correlation.pdf + .png
  fig6_with_without_uq.pdf + .png
  fig7_calibration.pdf + .png
  fig8_architecture.pdf + .png
  fig9_policy_explanation.pdf + .png
  fig10_node_vs_graph.pdf + .png
```

### Thesis LaTeX
```
thesis/latex_tum_official/
  main.tex
  chapters/01_introduction.tex  ... 07_conclusion.tex
  figures/  <- fig1-fig10 PDFs here (CLEAN — wrong figures archived)
```

### Figure Generation Script
```
docs/verified/figures/generate_all_figures.py  <- run karo, sab generate hoga
```

---

## Ek common confusing point — T notation

| Context | Notation | Matlab |
|---------|----------|--------|
| Trial number | T1, T2, ..., T8 | Training trial |
| MC Dropout samples | S=30 | Number of stochastic forward passes |
| Formal Bayesian math (Ch02) | T | Gal & Ghahramani notation for sample count |

**Experiment mein hamesha S=30 likho, T nahi** (T = trial number confusion ho sakti hai).

---

## Agar supervisor pooche toh kya bolna hai

1. **Best model**: T8, R²=0.5957, MAE=3.96 veh/h, 100 test graphs, 3.16M predictions
2. **UQ quality**: MC Dropout S=30, rho=0.4820 — meaningful ranking signal
3. **Conformal**: Coverage-guaranteed bands ±14.68 veh/h @ 95% (50/50 calib/eval split)
4. **Sigma not calibrated**: k95=11.65 vs ideal 1.96 — use for ranking, not intervals
5. **Selective prediction**: 39.9% MAE reduction at 50% retention
6. **Ensemble negative result**: Homogeneous ensembles don't help — data mismatch + correlated errors
7. **Limitation**: Only 1,000 of 10,000 scenarios (10% subset), single city (Paris)

---

## Cleanup status

- `__pycache__` / `.pyc` / `.ipynb_checkpoints` — DELETED (10 items)
- Old chart scripts (11 files) — ARCHIVED to `thesis/ARCHIVED_OLD_SCRIPTS/`
- Wrong/old PNG figures (79 files) — ARCHIVED to `thesis/latex_tum_official/figures/ARCHIVED_OLD_FIGURES/`
- Active figures folder: only 10 verified PDFs + TUM logos + WRONG_FIGURES_WARNING.txt

---

*Generated: 2026 | All numbers verified from JSON source files*
