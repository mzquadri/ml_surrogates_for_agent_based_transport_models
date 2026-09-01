# UQ Summary — Uncertainty Quantification for ML Models in Transportation Policy Analysis
> **Submission-era record.** This file is preserved to document the analysis as submitted.
> Post-submission corrections and protocol boundaries are authoritative in
> [`../docs/CORRIGENDUM.md`](../docs/CORRIGENDUM.md) and
> [`../analysis_outputs/THESIS_INTELLIGENCE_REPORT.md`](../analysis_outputs/THESIS_INTELLIGENCE_REPORT.md).

**Thesis:** Uncertainty Quantification for Machine Learning Models in Transportation Policy Analysis
**Author:** Mohd Zamin Quadri | TUM Master's Thesis
**Dataset:** 1,000 of 10,000 MATSim scenarios (Paris road network, 10% subset)
**Test Set:** 100 graphs, 3,163,500 node-level predictions

---

## BASELINE MODEL — Trial 8 (T8)

**Architecture:** PointNetTransfGAT + GATConv(64→1) output layer | Dropout = 0.2 | 80/10/10 split

| Metric | Value | Source |
|---|---|---|
| R² | **0.5957** | test_evaluation_complete.json |
| MAE | **3.957 veh/h** | test_evaluation_complete.json |
| RMSE | **7.118 veh/h** | test_evaluation_complete.json |
| Pearson r | 0.773 | test_evaluation_complete.json |

**Why T8 is the baseline:**
- Best performing trial among T2–T8 (all with GATConv output + non-zero dropout)
- T1 (R²=0.786) is excluded — zero dropout makes MC Dropout undefined (σ=0 everywhere)

---

## ALL UQ METHODS APPLIED

---

### 1. MC DROPOUT (Primary Method)

**What it is:** At inference time, dropout is kept ON. S=30 stochastic forward passes produce a distribution of predictions per node. The standard deviation across passes = uncertainty estimate.

**Applied to:** T8 (primary), T5, T6, T7 (comparison)

**Key Results:**

| Trial | Dropout | Spearman ρ | Mean σ (veh/h) |
|---|---|---|---|
| T5 | 0.3 | 0.4263 | — |
| T6 | 0.3 | 0.4186 | — |
| T7 | 0.3 | 0.4437 | — |
| **T8** | **0.2** | **0.4820** | **1.369** |

**T8 MC Dropout detailed stats (S=30, 100 test graphs):**
- Spearman ρ = 0.4820 (uncertainty vs absolute error)
- Mean σ = 1.369 veh/h | Std σ = 1.383 | Range: 0.042 – 44.29 veh/h
- MC mean R² = 0.5856 (vs 0.5957 deterministic — small expected effect of stochastic averaging)
- MC mean MAE = 3.948 veh/h
- Inference time: 228 minutes (T4 GPU, S=30, 100 graphs)
- Per-graph ρ: mean=0.464, std=0.023, range [0.41, 0.51]

**Selective Prediction (using uncertainty to filter predictions):**

| Retained Fraction | MAE (veh/h) | MAE Reduction |
|---|---|---|
| 100% | 3.95 | — |
| 90% | 3.23 | -18.3% |
| 50% | 2.32 | **-41.2%** |
| 25% | 1.79 | -54.5% |
| 10% | 1.06 | -73.4% |

**S-Convergence Analysis (source: s_convergence_with_rho.json, 10 graphs):**

| S | Mean ρ | Mean σ (veh/h) | ρ change vs S=30 |
|---|---|---|---|
| 5 | 0.4203 | 1.179 | — |
| 10 | 0.4469 | 1.301 | — |
| 15 | 0.4561 | 1.342 | — |
| 20 | 0.4610 | 1.363 | — |
| 25 | 0.4640 | 1.375 | — |
| **30** | **0.4658** | **1.383** | baseline |
| 35 | 0.4677 | 1.389 | +0.42% |
| 40 | 0.4692 | 1.394 | +0.74% |
| 45 | 0.4700 | 1.397 | +0.90% |
| 50 | 0.4706 | 1.400 | +1.03% |

S=5→S=30: **+10.8% ρ gain** (most gain happens early). S=30→S=50: only **+1.03%** — S=30 is firmly on the plateau.

**Key Finding:** MC Dropout σ is NOT a calibrated standard deviation. k₉₅ = 11.66 (ideal Gaussian = 1.96).

**Charts:**
- `fig2_uq_ranking.pdf` — Spearman ρ across all UQ methods
- `fig6_with_without_uq.pdf` — Deterministic vs MC Dropout accuracy comparison
- `t8_s_convergence.pdf` — S-convergence of ρ and mean σ
- `t8_selective_prediction_curve.pdf` — MAE vs retention fraction

---

### 2. CONFORMAL PREDICTION (Post-Hoc, Marginal Coverage)

**What it is:** A distribution-free method under exchangeability assumptions that targets marginal prediction-interval coverage. Nonconformity score = |y - ŷ|. Uses 50 calibration graphs to find quantile q, applies ŷ ± q to 50 evaluation graphs.

**Applied to:** T8 (primary), T7 (cross-check)

#### 2a. Standard (Global) Conformal Prediction

**Split:** 50 calibration + 50 evaluation graphs (1,581,750 nodes each)

| Level | Quantile q | Achieved Coverage | Interval Width |
|---|---|---|---|
| 90% | 9.92 veh/h | **90.02%** | ±9.92 veh/h |
| 95% | 14.68 veh/h | **95.01%** | ±14.68 veh/h |

**Calibration Audit (100 test graphs, 3,163,500 nodes — source: calibration_audit.json):**

| Nominal | Raw MC Coverage | k_emp | Global Conformal | Adaptive Conformal |
|---|---|---|---|---|
| 50% | 23.4% | **1.713** | 50.4% | 50.2% |
| 70% | 33.7% | **3.088** | 70.4% | 70.2% |
| 80% | 40.1% | **4.545** | 80.3% | 80.2% |
| 90% | 48.6% | **7.737** | 90.2% | 90.1% |
| 95% | 54.9% | **11.66** | 95.1% | 95.0% |

**Key Finding:** Raw MC Dropout severely undercovers at every level (54.9% at 95% nominal). Conformal prediction achieved near-nominal empirical marginal coverage on this evaluated split; it does not guarantee conditional or per-scenario coverage.

#### 2b. Conditional Coverage Analysis (Deciles by σ)

- Standard conformal over-covers low-uncertainty nodes (D1: **98.1%**) and under-covers high-uncertainty nodes (D10: **59.0%**)
- This is theoretically expected — Barber et al. proved distribution-free conditional coverage is impossible without additional assumptions

#### 2c. Adaptive Conformal Prediction

Normalises nonconformity score by MC Dropout σ → node-specific interval widths.
- Conditional coverage range: **[83.7%, 96.4%]** across deciles (vs [59.0%, 98.1%] for standard) — source: adaptive_conformal_decile.json
- Adaptive conformal quantile q₉₀_adapt = **7.71** *(source: adaptive_conformal_results.json)*

**Charts:**
- `fig3_conformal_coverage.pdf` — Nominal vs achieved coverage + interval widths
- `fig14_conformal_workflow.pdf` — Split conformal workflow diagram
- `t8_conformal_conditional.pdf` — Conditional coverage by uncertainty decile (standard vs adaptive)
- `t8_calibration_curve.pdf` — Nominal vs achieved coverage across all levels
- `t8_interval_width_comparison.pdf` — Interval width comparison at each level

---

### 3. REGRESSION σ-SCALING / TEMPERATURE SCALING (Post-Hoc Calibration)

> **Terminology:** the submitted thesis (Ch. 5.4) calls this method **"post-hoc regression σ-scaling"**; the saved JSON artefact is named `temperature_scaling_results.json`. Same method, two names.

**What it is:** Single-parameter post-hoc recalibration. Learns scalar T on calibration set, scales all σ: σ_scaled = σ_raw × T. Minimises Kuleshov ECE (1σ coverage per percentile bin).

**Calibration set:** 30% of node-level predictions (949,050 nodes, seed=42 random split)
**Evaluation:** Remaining 70% (2,214,450 nodes)
**Optimal T = 2.887** *(verified 2026-04-24, source: temperature_scaling_results.json)*

| Metric | Before (T=1.0) | After (T=2.887) |
|---|---|---|
| ECE | 0.356 | **0.034** (−90.5%) |
| 1σ coverage | 32.7% | **68.0%** (expected 68.3%) ✅ |
| 2σ coverage | 55.6% | 85.0% (expected 95.4%) |
| 3σ coverage | 69.1% | 91.6% (expected 99.7%) |
| k₉₅ | 11.66 | **4.04** |

**Key Finding:** Temperature scaling reduces ECE by 90.5% and achieves near-perfect 1σ calibration (68.0% vs 68.3% target). However, 2σ and 3σ coverage remains below theoretical Gaussian targets — residual miscalibration persists because scaling only optimises 1σ. It cannot replace conformal prediction for empirical marginal coverage under the stated exchangeability assumptions.

**Note on earlier hardcoded values:** Prior summaries cited T=2.70, ECE 0.269→0.048 — these were from an earlier unrecorded run with a different split. The values above are the verified numbers from the saved JSON.

**Charts:**
- `fig_temp_scaling_4panel.png` — T optimisation curve + coverage bars + reliability diagrams (before/after)
- `fig_temp_scaling_reliability.png` — Thesis-ready before/after reliability diagram (ECE annotated)

---

### 4. ENSEMBLE METHODS

**What it is:** Comparing MC Dropout vs ensemble-based uncertainty signals.

#### Experiment A — 5 Seeded MC Dropout Runs on T8

5 independent seeds (42, 142, 242, 342, 442), S=30 each, same T8 model.

| Method | Spearman ρ | Mean σ (veh/h) |
|---|---|---|
| MC Dropout (S=30) | 0.4908 | 1.369 |
| Ensemble Variance (5 runs) | 0.4370 | 0.217 |
| Combined (quadrature) | **0.4909** | 1.388 |

MC Dropout outperforms seed ensemble variance by 12.3%. Combined adds negligible benefit.

#### Experiment B — Multi-Model Ensemble (T2, T5, T6, T7, T8)

Weighted ensemble by individual R², deterministic forward passes.

| Metric | Value |
|---|---|
| Ensemble Spearman ρ | 0.4333 |
| Ensemble R² | 0.5656 |
| Ensemble MAE | 3.99 veh/h |
| Best individual (T8) R² | 0.5957 |

Multi-model ensemble is weaker than MC Dropout (ρ=0.4333 vs 0.4908). Averaging uneven-quality models dilutes the best predictor.

**Important Note:** Original ensemble scripts had a PyG GATConv API version mismatch (checkpoint stores `lin.weight`, PyG 2.3.1 expects `lin_src.weight` + `lin_dst.weight`). Fixed by key remapping before loading with strict=True.

**Charts:**
- `fig2_uq_ranking.pdf` — Spearman ρ across all UQ methods (bar chart)

#### Experiment C — True Deep Ensemble (5 Independently Trained Models)

5 models trained from scratch with different random seeds (42, 137, 256, 389, 512). Same architecture as T8 (PointNetTransfGAT, dropout=0.2). Same dataset split (seed=42). Deterministic inference (dropout OFF during evaluation).

| Metric | Value | vs T8 Baseline |
|---|---|---|
| Ensemble R² | **0.6841** | +0.088 (+14.8%) — **best R² in project** |
| Ensemble MAE | **3.485 veh/h** | −0.472 (−11.9%) |
| Ensemble RMSE | **6.293 veh/h** | −0.825 (−11.6%) |
| Spearman ρ (UQ quality) | 0.3997 | — |
| Mean σ (veh/h) | 1.258 | — |
| k₉₅ | 15.18 | — |
| Coverage @ 1.96σ | 51.0% | — |
| Per-graph ρ | 0.389 | — |

**Individual member R²:** seed_42=0.640, seed_137=0.646, seed_256=0.650, seed_389=0.647, seed_512=0.649. Ensemble mean (0.684) exceeds all individual members — classic variance reduction from averaging.

**MC Dropout vs True Deep Ensemble — head-to-head:**

| Metric | MC Dropout | Deep Ensemble | Winner |
|---|---|---|---|
| R² | 0.5857 | **0.6841** | Ensemble |
| MAE (veh/h) | 3.948 | **3.485** | Ensemble |
| RMSE (veh/h) | 7.207 | **6.293** | Ensemble |
| Spearman ρ | **0.4817** | 0.3997 | MC Dropout |
| k₉₅ (sharpness) | **11.64** | 15.18 | MC Dropout |
| Coverage @ 1.96σ | 54.9% | 51.0% | Both poor |

**Key finding:** Deep Ensemble dominates on point prediction accuracy (+14.8% R²). MC Dropout dominates on uncertainty quality (ρ +0.082, sharper intervals). Both are severely overconfident — k₉₅ ≫ 1.96 — requiring post-hoc calibration for valid intervals.

**Source:** `uq_verification_run/comparison_verified.json` (verified 2026-04-24, 100 test graphs, 3,163,500 nodes)

---

### 5. PROPER SCORING RULES

**Applied to:** T8 MC Dropout (S=30, 100 test graphs, 3,163,500 nodes)

| Metric | Value | Interpretation |
|---|---|---|
| CRPS | 3.38 veh/h | 14.3% below MAE (CRPS/MAE = 0.857) |
| CRPS/MAE ratio | 0.857 | Ideal calibrated Gaussian = 0.707; our ratio shows calibration cost |
| PIT KS statistic | 0.245 | Severe underdispersion (ideal = 0) |
| PIT mean | 0.433 | Ideal = 0.500; model overpredicts more than underpredicts |
| Winkler score (90%) | **49.68** | Raw MC Gaussian intervals |
| Winkler score (90%, std conformal) | **35.78** | **28.0%** improvement with standard conformal |
| Winkler score (90%, adaptive conformal) | **32.32** | **35.0%** improvement with adaptive conformal |

**Charts:**
- `t8_pit_histogram.pdf` — PIT histogram showing underdispersion
- `t8_reliability_diagram.pdf` — Expected vs observed Gaussian coverage

---

### 6. SELECTIVE PREDICTION + ERROR DETECTION

**Selective Prediction (T8):**
- 50% retention → MAE 3.95 → 2.32 veh/h **(−41.2%)**
- 90% retention → MAE 3.95 → 3.23 veh/h **(−18.3%)**
- Source: mc_dropout_full_100graphs_mc30.npz

**Error Detection AUROC (T8):**
- Top-10% error threshold: AUROC = **0.7548** *(source: auroc_corrected.json)*
- Top-20% error threshold: AUROC = **0.7324** *(source: auroc_corrected.json)*
- T7 cross-check (top-10%): AUROC = **0.7416** *(source: t7_auroc.json)*

**Tiered Deployment Workflow:**
- ACCEPT: Bottom 50% σ → expected MAE ≈ 2.32 veh/h
- FLAG: 50–90% σ → manual review / sensitivity analysis
- REJECT: Top 10% σ → full MATSim re-simulation

**Charts:**
- `t8_selective_prediction_curve.pdf` — MAE vs retention fraction
- `fig9_policy_explanation.pdf` — Uncertainty-guided decision framework

---

### 7. UNCERTAINTY-AWARE TRAINING — Trial 9: Heteroscedastic (Partial Positive)

**What it is:** Instead of post-hoc UQ, directly predict mean + variance from the model. T8 backbone FROZEN (1,416,768 params). New GATConv(64→2) head (134 trainable params) predicts μ and log σ² per node. Loss = heteroscedastic NLL.

**Training:** 315 epochs, best checkpoint at epoch 290, val_NLL=3.2489, ~873 minutes

**Results (from t9_evaluation_results.json — VERIFIED):**

| Metric | T8 Baseline | Trial 9 | Gate | Status |
|---|---|---|---|---|
| R² | 0.5957 | **0.4991** | ≥ 0.55 | **FAIL** |
| MAE (veh/h) | 3.957 | 4.053 | — | — |
| RMSE (veh/h) | 7.118 | 7.924 | — | — |
| Spearman ρ | 0.4820 | 0.4797 | — | — |
| PICP₉₀ | 90.02% | 86.90% | ≥ 85% | PASS |
| PICP₉₅ | 95.01% | 90.01% | ≥ 90% | PASS |
| **k₉₅** | 11.66 | **2.84** | (lower better) | **4× improvement** |

**Uncertainty Decomposition (from JSON):**

| Component | Mean σ (veh/h) | Std σ (veh/h) |
|---|---|---|
| Aleatoric (σ_alea) | **4.657** | 5.283 |
| Epistemic (σ_epi) | 1.099 | 1.164 |
| Total (σ_tot) | 4.823 | 5.376 |
| Ratio alea/epi | **4.238** | — |
| Frac aleatoric dominant | **99.85%** | — |

**Verdict: Partial Positive**
- Good UQ calibration: k₉₅ = 2.84 (vs T8's 11.65 — 4× better), approaching ideal Gaussian (1.96)
- Insufficient point accuracy: R² = 0.499 < 0.55 gate
- Root cause: Head-only training ceiling under NLL–MSE tradeoff (Seitzer 2022). 134-parameter head can't fully recover R² from frozen backbone representations.
- Aleatoric dominates at 99.85% of nodes — traffic volume change is primarily data-level variability

**Charts:**
- `t9v2_point_metrics.pdf` — R², MAE vs T8 baseline (with gate line)
- `t9v2_uncertainty_decomposition.pdf` — Aleatoric vs epistemic vs total σ
- `t9v2_k95_comparison.pdf` — k₉₅: T8 (11.65) vs T9 (2.84) vs ideal Gaussian (1.96)
- `t9v2_error_vs_sigma.pdf` — Spearman correlation |error| vs σ_tot
- `t9v2_calibration.pdf` — Coverage reliability diagram (PICP₉₀=86.9%, PICP₉₅=90.0%)

---

### 8. UNCERTAINTY-AWARE TRAINING — Trial 10: CQR Full Backbone (Negative Result)

**What it is:** Conformalized Quantile Regression. Full T8 backbone UNFROZEN. GATConv(64→2) head predicts q̂₀.₀₅ and q̂₀.₉₅ per node. Loss = joint pinball loss. Conformal correction applied post-training.

**Training (v2):** 340 epochs, best val_pinball=1.4857, ~52 hours. Backbone lr = 5×10⁻⁵ (reduced from v1's 5×10⁻⁴)

**Comparison T10-v1 vs T10-v2 vs T11:**

| Version | R² | PICP₉₅ | Epochs | Notes |
|---|---|---|---|---|
| T10-v1 | 0.315 | 94.90% | 885 | Full backbone, original lr |
| T10-v2 | 0.406 | 91.78% | 340 | Reduced backbone lr — better R², worse PICP₉₅ |
| T11 | **0.5835** | **94.91%** | 1000 | Frozen backbone — PASS |

**Results (from cqr_metrics.json — T10 — VERIFIED):**

| Metric | T8 Baseline | Trial 10 | Gate | Status |
|---|---|---|---|---|
| R² (midpoint) | 0.5957 | **0.4057** | ≥ 0.57 | **FAIL** |
| MAE (veh/h) | 3.957 | 4.130 | — | — |
| RMSE (veh/h) | 7.118 | 8.631 | — | — |
| Spearman ρ | 0.4820 | 0.3177 | — | — |
| PICP₉₀ | 90.02% | 89.47% | ≥ 88% | PASS |
| PICP₉₅ | 95.01% | **91.78%** | ≥ 93% | **FAIL** |
| Width₉₀ (veh/h) | 19.84 | 17.776 | < Width₉₅ | PASS |
| Width₉₅ (veh/h) | 29.35 | 20.021 | strictly wider | PASS |
| Q̂₉₀ | — | **−0.00107** | > 0 | **FAIL** |
| Q̂₉₅ | — | 1.1212 | > 0 | PASS |

**Gate Failures: 3 out of 6** → Negative Result

**Root Cause:** Pinball loss applied to all layers reshapes backbone representations built under MSE training. Midpoint prediction loses point-accuracy. Even with reduced backbone lr, PICP₉₅ degraded further (94.9% → 91.8%).

**Charts:**
- `t10v2_scatter.pdf` — R² progression: T8 → T10-v1 → T10-v2 → T11
- `t10v2_coverage_bars.pdf` — PICP₉₀ and PICP₉₅ for T10-v1, T10-v2, T11
- `t10v2_interval_widths.pdf` — Prediction interval widths comparison

---

### 9. UNCERTAINTY-AWARE TRAINING — Trial 11: CQR Frozen Backbone (POSITIVE RESULT)

**What it is:** Same as T10 but T8 backbone FROZEN (1,416,768 params, requires_grad=False). Only GATConv(64→2) quantile head (134 params) trained with pinball loss. CQR conformal correction applied post-training.

**Training:** 1000 epochs (early stopping did not trigger), best checkpoint epoch 999, val_pinball=1.4862, ~39.7 hours. Zero monotonicity crossings on both calibration and test sets.

**Results (from cqr_metrics.json — T11 — VERIFIED):**

| Metric | T8 Baseline | Trial 11 | Gate | Status |
|---|---|---|---|---|
| R² (midpoint) | 0.5957 | **0.5835** | ≥ 0.57 | **PASS** |
| MAE (veh/h) | 3.957 | 4.302 | — | — |
| RMSE (veh/h) | 7.118 | 7.225 | — | — |
| Spearman ρ | 0.4820 | 0.2943 | — | — |
| PICP₉₀ | 90.02% | **89.82%** | ≥ 88% | **PASS** |
| PICP₉₅ | 95.01% | **94.91%** | ≥ 93% | **PASS** |
| Width₉₀ (veh/h) | 19.84 | 17.825 | < Width₉₅ | **PASS** |
| Width₉₅ (veh/h) | 29.35 | 24.822 | strictly wider | **PASS** |
| Q̂₉₀ | — | **0.1244** | > 0 | **PASS** |
| Q̂₉₅ | — | **3.6226** | > 0 | **PASS** |

**All 6 Gates: PASS** → Positive Result ✓

**Key finding:** Freezing the backbone is the critical design decision. R² drops by only 1.2% from T8 (0.5957 → 0.5835) while providing native asymmetric prediction intervals with empirical marginal coverage under the stated split assumptions.

**Charts:**
- `t10v2_scatter.pdf` — R² progression across T8, T10-v1, T10-v2, T11
- `t10v2_coverage_bars.pdf` — Coverage comparison across trials
- `t10v2_interval_widths.pdf` — Interval width comparison

---

## STRATIFIED AND CROSS-TRIAL ANALYSIS

### T7 Cross-Check (Replication on Trial 7)

| Metric | T7 | T8 |
|---|---|---|
| R² | 0.5471 | 0.5957 |
| Spearman ρ | 0.4437 | 0.4820 |
| Selective pred (50% ret.) | −38.3% MAE | −41.2% MAE |
| k₉₅ | 16.15 | 11.66 |
| AUROC (top 10% errors) | **0.7416** | **0.7548** |

Same qualitative conclusions hold across both trials.

**Charts:**
- `t7_calibration_curve.pdf` — T7 calibration curve
- `t7_selective_prediction_curve.pdf` — T7 selective prediction
- `t7_vs_t8_uq_comparison.pdf` — T7 vs T8 UQ comparison
- `t7_interval_width_comparison.pdf` — T7 interval widths

### Stratified UQ by |Δv| Quartile (submitted thesis §5.10)

Rank-based quartiles of |Δv| (790,875 nodes each). Q1 = segments with zero policy effect (|Δv| = 0); Q4 = largest responses (up to ~230 veh/h).

| Quartile | MAE (veh/h) | Spearman ρ |
|---|---|---|
| Q1 (smallest \|Δv\|, all zero-effect) | 1.24 | 0.721 |
| Q4 (largest \|Δv\|) | 10.08 | **0.100** |

The high Q1 ρ is partly mechanical (when y = 0, both |error| and σ depend on the model's small output magnitude); the Q4 degradation is real — **the uncertainty signal is weakest exactly where policy effects are largest.** This is the thesis's headline caveat for deployment.

**Charts:**
- `t8_stratified_uq.pdf` — Stratified UQ by road feature quartiles
- `t8_per_graph_variation.pdf` — Per-graph ρ distribution across 100 test graphs
- `t8_error_detection_auroc.pdf` — AUROC for error detection

---

## MASTER RESULTS SUMMARY TABLE

| Method | R² | PICP₉₅ | k₉₅ | Spearman ρ | Gate | Result |
|---|---|---|---|---|---|---|
| T8 MSE Baseline | 0.5957 | 95.01% | 11.66 | 0.4820 | Locked | Baseline |
| T8 + MC Dropout | 0.5856 | — | 11.66 | 0.4820 | — | Primary UQ |
| T8 + σ-Scaling (T=2.887) | 0.5856 | — | 4.04 | 0.4820 | — | Post-hoc calib (ECE −90.5%) |
| T8 + Std Conformal | 0.5957 | **95.01%** | — | — | — | Near-nominal marginal coverage |
| T8 + Adaptive Conformal | 0.5957 | [83.7–96.4%] | — | — | — | Best conditional coverage |
| Ensemble (5 runs avg) | 0.5865 | — | — | **0.4908** | — | Marginal gain |
| Multi-model Ensemble | 0.5656 | — | — | 0.4333 | — | Weaker UQ |
| **T9 Heteroscedastic** | **0.4991** | **90.01%** | **2.84** | 0.4797 | 1 FAIL | Partial Positive |
| **T10 CQR Full** | **0.4057** | **91.78%** | — | 0.3177 | 3 FAIL | **Negative** |
| **T11 CQR Frozen** | **0.5835** | **94.91%** | — | 0.2943 | **6 PASS** | **Positive** |

---

## CROSS-VERIFICATION — JSON vs THESIS (ALL MATCH)

| Number | Source (JSON) | Thesis Value | Match |
|---|---|---|---|
| T9 R² | 0.4990585... | 0.4991 | ✓ |
| T9 MAE | 4.053110... | 4.053 | ✓ |
| T9 PICP₉₀ | 86.9001% | 86.90% | ✓ |
| T9 PICP₉₅ | 90.0144% | 90.01% | ✓ |
| T9 k₉₅ | 2.8373... | 2.84 | ✓ |
| T9 σ_alea | 4.6573... | 4.657 | ✓ |
| T9 σ_epi | 1.0988... | 1.099 | ✓ |
| T9 alea/epi ratio | 4.2383... | 4.24 | ✓ |
| T9 frac aleatoric | 0.9985... | 99.85% | ✓ |
| T10 R² | 0.4056635... | 0.4057 | ✓ |
| T10 MAE | 4.1304... | 4.130 | ✓ |
| T10 PICP₉₀ | 89.4734% | 89.47% | ✓ |
| T10 PICP₉₅ | 91.7792% | 91.78% | ✓ |
| T10 Q̂₉₀ | -0.00106... | -0.0011 | ✓ |
| T10 Q̂₉₅ | 1.12120... | 1.1212 | ✓ |
| T10 Width₉₀ | 17.7763... | 17.776 | ✓ |
| T11 R² | 0.5835253... | 0.5835 | ✓ |
| T11 MAE | 4.3015... | 4.302 | ✓ |
| T11 PICP₉₀ | 89.8224% | 89.82% | ✓ |
| T11 PICP₉₅ | 94.9078% | 94.91% | ✓ |
| T11 Q̂₉₀ | 0.12436... | 0.1244 | ✓ |
| T11 Q̂₉₅ | 3.62258... | 3.6226 | ✓ |
| T11 Width₉₀ | 17.8250... | 17.825 | ✓ |
| T11 Width₉₅ | 24.8215... | 24.822 | ✓ |
| T8 AUROC top-10% | 0.7548 | 0.7548 | ✓ |
| T8 AUROC top-20% | 0.7324 | 0.7324 | ✓ |
| T7 AUROC top-10% | 0.7416 | 0.7416 | ✓ |
| Winkler MC 90% | 49.68 | 49.68 | ✓ |
| Winkler std conformal 90% | 35.78 | 35.78 | ✓ |
| Winkler adaptive 90% | 32.32 | 32.32 | ✓ |
| k_emp @ 50% | 1.7130 | 1.713 | ✓ |
| k_emp @ 70% | 3.0879 | 3.088 | ✓ |
| k_emp @ 80% | 4.5452 | 4.545 | ✓ |
| k_emp @ 90% | 7.7371 | 7.737 | ✓ |
| k_emp @ 95% | 11.6616 | 11.66 | ✓ |
| Adp conformal D1 coverage | 89.45% | 89.45% | ✓ |
| Adp conformal D10 coverage | 96.44% | 96.44% | ✓ |
| Std conformal D1 coverage | 98.08% | 98.1% | ✓ |
| Std conformal D10 coverage | 59.04% | 59.0% | ✓ |
| q₉₀_adapt | 7.7086 | 7.71 | ✓ |
| S-conv ρ @ S=30 | 0.4658 | 0.4658 | ✓ |
| S-conv ρ @ S=50 | 0.4706 | 0.4706 | ✓ |
| S-conv ρ gain S=30→50 | 1.03% | 1.03% | ✓ |
| S-conv ρ gain S=5→30 | 10.8% | 10.8% | ✓ |
| Deep Ensemble R² | 0.6840808... | 0.6841 | ✓ | comparison_verified.json |
| Deep Ensemble MAE | 3.4853... | 3.485 | ✓ | comparison_verified.json |
| Deep Ensemble RMSE | 6.2927... | 6.293 | ✓ | comparison_verified.json |
| Deep Ensemble ρ | 0.3997361... | 0.3997 | ✓ | comparison_verified.json |
| Deep Ensemble k₉₅ | 15.183622... | 15.18 | ✓ | comparison_verified.json |

**STATUS: ALL 49 NUMBERS VERIFIED — 100% MATCH** ✓ *(Sources: auroc_corrected.json, winkler_scores.json, calibration_audit.json, adaptive_conformal_decile.json, adaptive_conformal_results.json, t7_auroc.json, s_convergence_with_rho.json, comparison_verified.json)*

**Note on MC Dropout ρ:** Two slightly different values appear in this document — 0.4820 (main T8 table, source: mc_dropout_metrics_mc30.json) and 0.4817 (head-to-head comparison table, source: comparison_verified.json). Both are from independent computation runs of the same model; the 0.0003 difference is noise from different random orderings in Spearman computation. All thesis claims use whichever source is cited in context.

---

## KEY DESIGN PRINCIPLE DISCOVERED

> **Freezing the backbone is the critical decision.**

| Approach | Backbone | R² | UQ Quality |
|---|---|---|---|
| T9 (Heteroscedastic) | FROZEN | 0.499 | Excellent (k₉₅=2.84) |
| T10 (CQR full) | UNFROZEN | 0.406 | Poor (3 gate failures) |
| T11 (CQR frozen) | FROZEN | **0.584** | **Good (all 6 PASS)** |

Unfreezing the backbone lets pinball/NLL gradients reshape MSE-trained representations → R² degrades. Frozen backbone preserves accuracy while head learns uncertainty structure.

---

## FIGURES DIRECTORY (document/figures/new/)

The figures shipped with the submitted thesis (PDF + PNG pairs in `document/figures/new/`):

| File | Content |
|---|---|
| `fig00_all_models_summary.pdf` | Master summary: R² ranking, accuracy/UQ trade-off, k₉₅ across all 12 models |
| `fig07_selective_prediction_curve.pdf` | MAE vs retention fraction (50%/90% annotated) |
| `fig12_sigma_scaling_ece.pdf` | ECE vs scaling factor T; optimum T* = 2.887 |
| `fig13_conformal_coverage_nominal_vs_achieved.pdf` | Nominal vs achieved coverage |
| `fig15_conditional_coverage_by_decile.pdf` | Conditional coverage by σ decile (standard vs adaptive) |
| `fig19_t9_uncertainty_decomposition.pdf` | T9 aleatoric vs epistemic decomposition |
| `fig22_cqr_r2_progression.pdf` | R² progression T8 → T10 → T11 |
| `fig26_deep_ensemble_member_r2.pdf` | Deep Ensemble mean exceeds every member |
| `fig28_stratified_uq_quartiles.pdf` | Stratified UQ by \|Δv\| quartile (MAE ↑, ρ ↓) |
| `fig29_pointnet_architecture.pdf` | PointNetTransfGAT architecture diagram |
| `fig31_network_intro.pdf` | Paris network introduction figure |
| `fig34_feature_distributions.pdf` | Input feature distributions |
| `fig35_policy_decision_framework.pdf` | Three-tier accept/review/reject decision framework |

*(Generation scripts: `generate_thesis_figures.py` and `scripts/misc/gen_batch*.py`.)*

---

## DATA SOURCES

| File | Description |
|---|---|
| `code/data/TR-C_Benchmarks/.../t9_evaluation_results.json` | T9 ground truth metrics |
| `code/data/TR-C_Benchmarks/.../10th.../cqr_metrics.json` | T10 ground truth metrics |
| `code/data/TR-C_Benchmarks/.../11th.../cqr_metrics.json` | T11 ground truth metrics |
| `../analysis_outputs/THESIS_INTELLIGENCE_REPORT.md` | Post-submission aggregate audit and limitations |

---

*Generated: 2026-04-10 | Cross-verified against JSON sources | Last audit: 2026-04-24 (49 numbers verified, 7 key JSONs: auroc_corrected.json, winkler_scores.json, calibration_audit.json, adaptive_conformal_decile.json, t7_auroc.json, s_convergence_with_rho.json, comparison_verified.json)*
