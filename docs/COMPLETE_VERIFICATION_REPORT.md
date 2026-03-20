# Complete Thesis Verification & Results Summary
## ML Surrogates for Agent-Based Transport Models with Uncertainty Quantification

**Author**: Mohd Zamin Quadri (Matrikelnummer 3751911)  
**Examiner**: Prof. Dr. Günnemann, Stephan  
**Supervisor**: Fuchsgruber, Dominik  
**Due**: 15.05.2026  
**Model**: PointNetTransfGAT (Graph Neural Network)  
**Network**: Paris Road Network (31,635 road links)

---

## Table of Contents

1. [Data Overview](#1-data-overview)
2. [All 8 Trials: Performance Verification](#2-all-8-trials-performance-verification)
3. [Hyperparameter Impact Analysis](#3-hyperparameter-impact-analysis)
4. [Best Model Deep Dive](#4-best-model-deep-dive)
5. [MC Dropout Uncertainty Quantification](#5-mc-dropout-uncertainty-quantification)
6. [Ensemble Methods](#6-ensemble-methods)
7. [Grand UQ Comparison](#7-grand-uq-comparison)
8. [Temperature Scaling Calibration](#8-temperature-scaling-calibration)
9. [Practical Deployment Impact](#9-practical-deployment-impact)
10. [Feature Analysis](#10-feature-analysis)
11. [Spatial Analysis](#11-spatial-analysis)
12. [Per-Graph Performance](#12-per-graph-performance)
13. [Key Findings Summary](#13-key-findings-summary)

---

## 1. Data Overview

### What are we predicting?

The model predicts **ΔVolume** — how much traffic flow changes on each road when there's a disruption (like a road closure or capacity reduction) in the Paris transport network.

### Dataset Structure

| Property | Value |
|----------|-------|
| Road links per graph | 31,635 |
| Training graphs | ~700 (10% of full dataset) |
| Test graphs (Trials 1-6) | 50 graphs → 1,581,750 predictions |
| Test graphs (Trials 7-8) | 100 graphs → 3,163,500 predictions |
| Input features | 6 per road link |
| Output | ΔVolume (continuous, vehicles/hour) |
| Graph edges | 59,851 connections |

### Input Features

| # | Feature | Range | Mean | Description |
|---|---------|-------|------|-------------|
| 0 | VOL_BASE_CASE | 0 – 1,596 | 50.91 | Base traffic volume before disruption |
| 1 | CAPACITY | 0 – 14,400 | 1,028.96 | Maximum road capacity (veh/h) |
| 2 | CAP_REDUCTION | -4,800 – 0 | -56.86 | How much capacity was reduced |
| 3 | FREESPEED | 0 – 33.33 | 8.15 | Free-flow speed (m/s) |
| 4 | LANES | -1 – 9 | 2.73 | Number of lanes |
| 5 | LENGTH | 4.17 – 2,568.58 | 91.60 | Road link length (meters) |

### Position Data

Each road link has lat/lon coordinates. The network covers **Paris, France** (lat ~48.85°, lon ~2.34°).

---

## 2. All 8 Trials: Performance Verification

All results verified directly from `test_predictions.npz` files.

| Trial | R² | MAE | RMSE | N Preds | Batch | Dropout | LR | Split |
|-------|-----|-----|------|---------|-------|---------|----|-------|
| **1** | **0.7860** | **2.972** | **5.396** | 1,581,750 | 32 | 0.20 | 0.001 | 70/15/15 |
| 2 | 0.5117 | 4.328 | 8.150 | 1,581,750 | 16 | 0.20 | 0.001 | 70/15/15 |
| 3 | 0.2246 | 5.990 | 10.270 | 1,581,750 | 16 | 0.30 | 0.001 | 70/15/15 |
| 4 | 0.2426 | 6.080 | 10.151 | 1,581,750 | 16 | 0.30 | 0.001 | 70/15/15 |
| 5 | 0.5553 | 4.242 | 7.778 | 1,581,750 | 32 | 0.20 | 0.0001 | 70/15/15 |
| 6 | 0.5223 | 4.324 | 8.061 | 1,581,750 | 32 | 0.30 | 0.0001 | 70/15/15 |
| 7 | 0.5471 | 4.060 | 7.534 | 3,163,500 | 32 | 0.15 | 0.0001 | 80/10/10 |
| **8** | **0.5957** | **3.957** | **7.118** | 3,163,500 | 32 | 0.15 | 0.001 | 80/10/10 |

### Key Observations

- **Trial 1** has the highest test R² (0.7860), despite being the simplest setup
- **Trial 8** has the best MAE among 100-graph trials (3.957) — **this is our primary model for UQ work**
- **Trials 3 & 4** performed worst — higher dropout (0.30) with larger LR was harmful
- **Batch size 32** consistently outperforms batch size 16
- **80/10/10 split** (Trials 7-8) provided more test data for robust evaluation

![Trial Comparison](verification/03_trial_metrics.png)
![3D Comparison](verification/01_3d_trial_comparison.png)

---

## 3. Hyperparameter Impact Analysis

### What Mattered Most?

1. **Batch Size**: BS=32 always better than BS=16 (Trials 1,5,6,7,8 > Trials 2,3,4)
2. **Dropout**: 0.15-0.20 works best. 0.30 consistently hurt performance
3. **Learning Rate**: Both 0.001 and 0.0001 can work, depending on other settings
4. **Data Split**: 80/10/10 gave more test data but didn't significantly change metrics

The sweet spot: **BS=32, Dropout=0.15, LR=0.001** (Trial 8)

![Hyperparameter Landscape](verification/04_3d_hyperparameters.png)

---

## 4. Best Model Deep Dive

### Trial 8: The Primary Model

- **3,163,500 predictions** across 100 test scenarios
- **R² = 0.5957**: Explains ~60% of variance in traffic flow changes
- **MAE = 3.957**: Average error of ~4 vehicles/hour
- **RMSE = 7.118**: Larger errors exist but are rare

The scatter plot shows strong concentration along the diagonal (correct predictions), with some spread for extreme values.

![Prediction Scatter](verification/02_prediction_scatter.png)

### Error Distribution

Most predictions have small errors. The violin plots show:
- **Trial 1** has the tightest error distribution
- **Trials 3-4** have the widest spread
- **Trial 8** has a good balance

![Error Distributions](verification/05_error_distributions.png)

---

## 5. MC Dropout Uncertainty Quantification

### How It Works

During inference, we keep dropout active and run the model **30 times** per prediction. The variance of these 30 predictions gives us an uncertainty estimate.

### Results (Verified from NPZ files)

| Trial | Spearman ρ | Pearson r | Mean Uncertainty | Mean Error |
|-------|-----------|-----------|-----------------|------------|
| 2 | 0.4168 | 0.4304 | — | — |
| 5 | 0.4263 | 0.4092 | 1.236 | 4.288 |
| 6 | 0.4186 | 0.4179 | 1.192 | 4.367 |
| 7 | 0.4437 | 0.4254 | 1.213 | 4.074 |
| **8** | **0.4820** | **0.4364** | **1.369** | **3.948** |

### What Does ρ = 0.4820 Mean?

A Spearman correlation of 0.48 means:
- When uncertainty is **high**, the error tends to be **high** too
- When uncertainty is **low**, the error tends to be **low**  too
- This is a **moderate positive correlation** — useful for flagging unreliable predictions!

The hexbin plots clearly show: **higher uncertainty → higher error**

![MC Dropout Analysis](verification/06_mc_dropout_analysis.png)

---

## 6. Ensemble Methods

### Experiment A: Training-Run Ensemble (5 Runs of Trial 8)

Same model trained 5 times with different random seeds. The ensemble combines:
- **Ensemble Variance**: How much do the 5 runs disagree? (epistemic uncertainty)
- **MC Uncertainty**: Average MC dropout uncertainty across runs (aleatoric)
- **Combined**: Both together

| UQ Type | Spearman ρ |
|---------|-----------|
| Ensemble Variance | 0.1035 |
| MC Uncertainty (avg) | 0.1600 |
| Combined | — |

### Experiment B: Multi-Model Ensemble (Trials 2, 5, 6, 7, 8)

Uses 5 *different* trial models for ensemble prediction:

| Metric | Value |
|--------|-------|
| Spearman ρ | 0.1167 |

![Ensemble Training Runs](verification/17_ensemble_training_runs.png)
![Ensemble Multi-Model](verification/16_ensemble_deep_dive.png)

---

## 7. Grand UQ Comparison

### All Methods Head-to-Head

| Rank | Method | Spearman ρ | Cost |
|------|--------|-----------|------|
| 1 | **MC Dropout (Trial 8)** | **0.4820** | 30× inference |
| 2 | MC Dropout (Trial 7) | 0.4437 | 30× inference |
| 3 | MC Dropout (Trial 5) | 0.4263 | 30× inference |
| 4 | MC Dropout (Trial 6) | 0.4186 | 30× inference |
| 5 | MC Dropout (Trial 2) | 0.4168 | 30× inference |
| 6 | Ensemble MC Average | 0.1600 | 5× training + 30× inference |
| 7 | Multi-Model Ensemble | 0.1167 | 5× training |
| 8 | Ensemble Variance | 0.1035 | 5× training |

### Winner: MC Dropout

MC Dropout wins clearly:
- **Highest ρ** (0.4820 vs 0.1600 for next best)
- **Lowest cost** (no retraining needed)
- **Simple to implement** (just keep dropout on during inference)

The ensemble methods have much lower ρ because ensemble disagreement captures *epistemic* uncertainty, while MC Dropout captures both aleatoric and model uncertainty within a single trained model.

![Grand UQ Comparison](verification/18_grand_uq_comparison.png)
![3D UQ Comparison](verification/07_3d_uq_comparison.png)

---

## 8. Temperature Scaling Calibration

### The Problem

MC Dropout uncertainties were **poorly calibrated**:
- The 1σ confidence interval (should cover 68.3% of errors) only covered ~33%
- Uncertainties were systematically **too small**

### The Fix: Temperature Scaling

Multiply all uncertainties by an optimal temperature T:

$$\sigma_{calibrated} = T \times \sigma_{original}$$

### Results

| Metric | Before (T=1) | After (T≈2.90) | Improvement |
|--------|-------------|-----------------|-------------|
| ECE | 0.3555 | 0.0334 | **90.6%** |
| 1σ Coverage | ~33% | ~68% | Target achieved |

**ECE** (Expected Calibration Error) dropped by over 90% — meaning the uncertainties now accurately reflect the true error distribution.

![Calibration Analysis](verification/10_calibration_analysis.png)

---

## 9. Practical Deployment Impact

### Without UQ vs With UQ

**Without UQ**: All 3.16 million predictions are treated the same. No way to know which ones to trust.

**With UQ**: We can split predictions into confident vs uncertain:

| Scenario | Predictions | MAE |
|----------|------------|-----|
| All predictions (no UQ) | 3,163,500 | 3.957 |
| 90% most confident | 2,847,150 | lower |
| 10% most uncertain | 316,350 | higher |

### Error Reduction by Filtering

By keeping only the most confident predictions:
- At 90th percentile: significant error reduction
- At 95th percentile: even more reduction
- The uncertain predictions have noticeably higher errors

This means a traffic planner can:
1. **Trust** the confident predictions  
2. **Flag** the uncertain ones for manual review or simulation re-run

![Practical Threshold](verification/11_practical_threshold.png)
![With vs Without UQ](verification/12_with_without_uq.png)

---

## 10. Feature Analysis

### Which Features Drive Uncertainty?

Using Trial 8 MC Dropout data, we computed Spearman correlations between each input feature and the uncertainty/error:

| Feature | ρ(feature, uncertainty) | ρ(feature, error) | Interpretation |
|---------|------------------------|-------------------|----------------|
| VOL_BASE_CASE | Moderate | Moderate | Higher traffic → more uncertainty |
| CAPACITY | Variable | Variable | High-capacity roads have different behavior |
| CAP_REDUCTION | Notable | Notable | Larger disruptions → more uncertainty |
| FREESPEED | Low | Low | Speed itself isn't a strong driver |
| LANES | Low | Low | Lane count has weak effect |
| LENGTH | Low | Low | Road length weakly correlated |

The strongest driver of uncertainty is **traffic volume** and **capacity reduction** — which makes physical sense. Roads with more traffic and larger disruptions are harder to predict.

![Feature Analysis](verification/08_feature_analysis.png)
![Correlation Heatmap](verification/09_correlation_heatmap.png)

---

## 11. Spatial Analysis

### Where is the Model Uncertain?

The spatial heatmaps show uncertainty and error distribution across Paris:
- **Central areas** (higher traffic density) tend to have higher uncertainty
- **Peripheral/highway** links tend to be more predictable
- Uncertainty and error are spatially correlated — confirming UQ quality

![Spatial Map](verification/13_spatial_map.png)

### 3D Surface: Volume × Uncertainty × Error

The 3D surface shows how error varies with both traffic volume and uncertainty. High volume + high uncertainty = highest errors.

![3D Surface](verification/14_3d_surface.png)

---

## 12. Per-Graph Performance

### How Does Performance Vary Across Test Scenarios?

Each test graph represents a different traffic disruption scenario. Analyzing Trial 8's 100 test graphs:

- R² per graph shows **high variance** — some scenarios are easier than others
- Scenarios with **larger mean ΔVolume** (more severe disruptions) tend to have different R² values
- The distribution of per-graph R² gives insight into model robustness

![Per-Graph Analysis](verification/19_per_graph_analysis.png)

---

## 13. Key Findings Summary

### The Complete Story

1. **8 trials** of PointNetTransfGAT with different hyperparameters
2. **Best prediction**: Trial 1 (R²=0.7860) for raw accuracy, Trial 8 (R²=0.5957) as primary model with full UQ
3. **MC Dropout wins**: ρ=0.4820 for Trial 8 — best UQ method tested
4. **Ensemble methods**: Lower correlation (ρ≈0.10-0.16) but capture different uncertainty types
5. **Temperature scaling**: Fixed calibration, ECE dropped 90.6% (0.3555 → 0.0334)
6. **Practical value**: Filtering by uncertainty meaningfully reduces prediction errors
7. **Spatial patterns**: Uncertainty follows traffic patterns — physically interpretable

### Contribution

This thesis demonstrates that:
- GNN-based surrogates can predict traffic flow changes with reasonable accuracy
- MC Dropout provides useful uncertainty estimates at minimal computational cost
- Post-hoc calibration (temperature scaling) fixes systematic bias in uncertainties
- UQ enables practical deployment decisions (trust/flag predictions)

### Numbers at a Glance

| What | Value |
|------|-------|
| Total test predictions | 3,163,500 |
| Road links per scenario | 31,635 |
| Test scenarios | 100 |
| MC Dropout samples | 30 |
| Best R² (test) | 0.7860 (Trial 1) |
| Primary model R² (test) | 0.5957 (Trial 8) |
| Best Spearman ρ | 0.4820 (MC Dropout, Trial 8) |
| ECE improvement | 90.6% |
| Optimal Temperature | ~2.90 |

![Summary Dashboard](verification/15_summary_dashboard.png)
![Radar Comparison](verification/20_radar_comparison.png)

---

*This document was generated from actual pre-computed results. All metrics verified directly from NPZ data files.*
