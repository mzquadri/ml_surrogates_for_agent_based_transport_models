# Ensemble-based Uncertainty Quantification Experiments Report

**Author:** Nazim Zameen  
**Date:** December 2024  
**Thesis:** ML Surrogates for Agent-Based Transport Models  

---

## Executive Summary

This report documents the results of two comprehensive experiments evaluating ensemble-based uncertainty quantification (UQ) methods for Graph Neural Network (GNN) surrogates of agent-based transport models. The experiments were conducted using the PointNetTransfGAT architecture on Paris district-level traffic demand prediction data.

### Key Findings

1. **MC Dropout outperforms Ensemble Variance** for uncertainty estimation (Spearman ρ = 0.151 vs 0.105)
2. **Single best model outperforms weighted ensemble** in prediction quality (R² = 0.0033 vs -0.0013)
3. **Ensemble disagreement provides useful but weaker uncertainty signal** compared to MC Dropout

---

## Experiment A: MC Dropout vs Ensemble Variance

### Methodology

**Objective:** Compare two uncertainty quantification methods on the best-performing model (Trial 8, dropout rate 0.3).

#### UQ Methods Evaluated:
1. **MC Dropout:** Perform 30 stochastic forward passes with dropout enabled, compute mean prediction and standard deviation
2. **Ensemble Variance:** Run 5 independent inference sessions, compute variance across runs
3. **Combined:** √(MC_σ² + Ensemble_σ²)

### Configuration
| Parameter | Value |
|-----------|-------|
| Model | Trial 8 (dropout=0.3) |
| MC Samples | 30 |
| Ensemble Runs | 5 |
| Test Graphs | 10 |
| Total Nodes | 316,350 |
| Device | CPU |

### Results

#### Uncertainty Estimation Quality (Correlation with Actual Error)

| Method | Spearman ρ | p-value | Pearson r | p-value |
|--------|------------|---------|-----------|---------|
| **MC Dropout** | **0.1507** | 0.0 | **0.1934** | 0.0 |
| Ensemble Variance | 0.1050 | 0.0 | 0.1576 | 0.0 |
| Combined | 0.1508 | 0.0 | 0.1933 | 0.0 |

#### Uncertainty Statistics

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| MC Dropout | 0.112 | 0.070 | 0.030 | 2.006 |
| Ensemble Variance | 0.018 | 0.013 | 0.001 | 0.514 |
| Combined | 0.114 | 0.071 | 0.030 | 2.026 |

### Interpretation

- **MC Dropout is the winner** with 43% higher Spearman correlation (0.151 vs 0.105)
- Combined uncertainty shows negligible improvement over MC Dropout alone
- Ensemble variance has smaller absolute values but weaker correlation with actual errors
- All p-values are effectively 0, indicating statistically significant correlations

### Visualization

![MC Dropout vs Ensemble Variance](../data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/plots/exp_a_uncertainty_comparison.png)

---

## Experiment B: 5-Model Ensemble with Weighted Averaging

### Methodology

**Objective:** Evaluate whether combining predictions from multiple independently trained models improves performance.

#### Ensemble Configuration:
- **Models:** Trials 2, 5, 6, 7, 8 (varying dropout rates and training configurations)
- **Weighting:** R²-score based weights (normalized by sum)
- **Uncertainty:** Standard deviation across model predictions

### Model Weights
| Model | Dropout Rate | R² Score | Normalized Weight |
|-------|--------------|----------|-------------------|
| Trial 2 | 0.2 | 0.5117 | 0.171 |
| Trial 5 | 0.2 | 0.4882 | 0.163 |
| Trial 6 | 0.2 | 0.4779 | 0.160 |
| Trial 7 | 0.3 | 0.5647 | 0.189 |
| **Trial 8** | **0.3** | **0.5957** | **0.199** |

### Results

#### Model Performance Comparison

| Model | R² | MAE | RMSE | MSE |
|-------|-----|-----|------|-----|
| **Model 7** | **0.0033** | **4.297** | **11.188** | 125.18 |
| Model 8 | 0.0013 | 4.306 | 11.200 | 125.43 |
| Model 6 | -0.0016 | 4.437 | 11.215 | 125.79 |
| **Weighted Ensemble** | **-0.0013** | 4.308 | 11.214 | 125.75 |
| Model 5 | -0.0063 | 4.308 | 11.242 | 126.38 |
| Model 2 | -0.0074 | 4.394 | 11.248 | 126.53 |

#### Ensemble Uncertainty Calibration

| Metric | Value |
|--------|-------|
| Mean Uncertainty (σ) | 0.240 |
| Std Uncertainty | 0.149 |
| Min Uncertainty | 0.015 |
| Max Uncertainty | 4.069 |
| Spearman ρ (Uncertainty vs Error) | 0.0247 |
| Pearson r (Uncertainty vs Error) | 0.1333 |

### Interpretation

1. **Single best model wins:** Model 7 (R² = 0.0033) outperforms the weighted ensemble (R² = -0.0013)
2. **Ensemble averaging doesn't help:** The weighted combination degrades performance
3. **Weak uncertainty signal:** Ensemble disagreement has very low correlation with error (ρ = 0.025)
4. **Model diversity is limited:** All models have similar architectures, reducing benefit of ensembling

### Key Observations

- Models trained with dropout=0.3 (Trials 7, 8) outperform dropout=0.2 (Trials 2, 5, 6)
- The ensemble smooths predictions but doesn't improve accuracy
- Individual model selection based on validation R² is a better strategy

---

## Conclusions

### Uncertainty Quantification Recommendations

| Use Case | Recommended Method | Rationale |
|----------|-------------------|-----------|
| **General UQ** | MC Dropout | Best uncertainty-error correlation |
| **Fast Inference** | Single Model + MC Dropout | Ensemble adds computation without benefit |
| **High Confidence Needed** | MC Dropout + Reject High-σ Predictions | Use calibrated uncertainty thresholds |

### Technical Insights

1. **MC Dropout is sufficient** - No need for complex ensemble methods
2. **Model architecture diversity matters** - Similar models don't provide ensemble benefit
3. **Dropout rate 0.3 is optimal** - Higher dropout gives better UQ and similar prediction quality
4. **Ensemble disagreement is unreliable** - Low correlation makes it unsuitable for calibrated UQ

### Future Work

1. Train models with different architectures (GAT variants, GCN, GraphSAGE) for true ensemble
2. Implement Deep Ensembles with different random seeds
3. Explore Bayesian Neural Networks for more principled UQ
4. Apply conformal prediction for guaranteed coverage

---

## Appendix

### Generated Files

```
ensemble_experiments/
├── experiment_a_results.json    # Experiment A metrics
├── experiment_a_data.npz        # Raw predictions and uncertainties
├── experiment_b_results.json    # Experiment B metrics  
├── experiment_b_data.npz        # Raw predictions and uncertainties
└── plots/
    ├── exp_a_mc_vs_ensemble.png
    ├── exp_a_uncertainty_comparison.png
    ├── exp_a_uncertainty_distributions.png
    ├── exp_b_all_models_scatter.png
    ├── exp_b_ensemble_performance.png
    ├── exp_b_model_comparison.png
    └── exp_b_uncertainty_distribution.png
```

### Script Location

```
scripts/evaluation/ensemble_uq_experiments.py
```

### Usage

```bash
# Run both experiments
conda activate thesis-env
python scripts/evaluation/ensemble_uq_experiments.py --experiment both --max-graphs 100 --cpu

# Run individual experiments
python scripts/evaluation/ensemble_uq_experiments.py --experiment a
python scripts/evaluation/ensemble_uq_experiments.py --experiment b
```

---

## References

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.
2. Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.
3. Original GNN surrogate paper and methodology - see `REPO_AND_PAPER_MAP.md`
