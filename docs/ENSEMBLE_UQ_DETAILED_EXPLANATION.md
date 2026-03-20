# Ensemble Uncertainty Quantification Experiments - Detailed Report

**Author:** Nazim Zameen  
**Date:** February 2026  
**Runtime:** ~13 hours on CPU  
**Test Data:** 100 graphs, 3.16 million nodes  

---

## 📚 Background: Uncertainty Quantification (UQ) Kya Hai?

### Problem Statement
Jab GNN model koi prediction deta hai (e.g., "is node par 15 trips hongi"), hum kaise jaanein ke yeh prediction kitni **reliable** hai?

**Example:**
```
Node A: Prediction = 15 trips, Uncertainty = 0.5  → High confidence ✅
Node B: Prediction = 15 trips, Uncertainty = 5.0  → Low confidence ⚠️
```

### UQ Ka Matlab
- **Low uncertainty** = Model confident hai → Prediction trust karo
- **High uncertainty** = Model unsure hai → Prediction pe rely mat karo

### Thesis Mein Kyun Important?
Transport planning mein galat predictions costly hain. Agar hum jaanein ke model kahan uncertain hai, toh:
1. High-uncertainty areas mein extra validation kar sakte hain
2. Decision makers ko confidence level bata sakte hain
3. Model ki limitations samajh sakte hain

---

## 🔬 Experiment A: MC Dropout vs Ensemble Variance

### Objective
**Single model (Model 8) par 2 UQ methods compare karna**

### Method 1: MC Dropout (Monte Carlo Dropout)

**Concept:**
Training mein dropout neurons randomly OFF karta hai. Inference mein bhi dropout ON rakh ke, har baar different neurons OFF honge → Different predictions milenge.

```
┌──────────────────────────────────────────────────────────┐
│                    SAME INPUT GRAPH                       │
└──────────────────────────────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
┌─────────┐          ┌─────────┐          ┌─────────┐
│ Forward │          │ Forward │          │ Forward │
│ Pass 1  │          │ Pass 2  │   ...    │ Pass 30 │
│(Random  │          │(Random  │          │(Random  │
│ Dropout)│          │ Dropout)│          │ Dropout)│
└────┬────┘          └────┬────┘          └────┬────┘
     │                    │                    │
     ▼                    ▼                    ▼
  Pred: 14.2           Pred: 15.8           Pred: 14.9
     │                    │                    │
     └────────────────────┼────────────────────┘
                          ▼
              ┌───────────────────────┐
              │ Mean = 15.0 (Final)   │
              │ Std = 0.8 (Uncertainty)│
              └───────────────────────┘
```

**Hamara Setup:**
- 30 forward passes (MC samples)
- 5 different random seeds (ensemble runs)
- Total: 100 graphs × 5 runs × 30 samples = **15,000 forward passes**

### Method 2: Ensemble Variance

**Concept:**
Same model ko 5 different random seeds ke saath run karo. Predictions ka variance = uncertainty.

```
┌──────────────────────────────────────────────────────────┐
│                    SAME INPUT GRAPH                       │
└──────────────────────────────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
┌─────────┐          ┌─────────┐          ┌─────────┐
│  Run 1  │          │  Run 2  │   ...    │  Run 5  │
│(Seed 42)│          │(Seed 142│          │(Seed 442│
└────┬────┘          └────┬────┘          └────┬────┘
     │                    │                    │
     ▼                    ▼                    ▼
Mean: 15.02           Mean: 15.08           Mean: 14.98
     │                    │                    │
     └────────────────────┼────────────────────┘
                          ▼
              ┌───────────────────────┐
              │ Variance = 0.02       │
              │ (Ensemble Uncertainty) │
              └───────────────────────┘
```

### Method 3: Combined Uncertainty

**Formula:** `√(MC_Dropout² + Ensemble_Variance²)`

Dono uncertainties combine - total uncertainty.

---

### 📊 Experiment A Results

| UQ Method | Spearman ρ | Pearson r | Mean σ | Interpretation |
|-----------|-----------|-----------|--------|----------------|
| **MC Dropout** | **0.1600** ✅ | 0.2117 | 0.130 | **Best correlation** |
| Ensemble Variance | 0.1035 | 0.1433 | 0.021 | Weaker signal |
| Combined | 0.1601 | 0.2117 | 0.132 | ~Same as MC Dropout |

### Spearman ρ Kya Hai?

**Spearman correlation** measure karta hai ke uncertainty aur actual error mein **monotonic relationship** hai ya nahi.

```
ρ = 1.0  → Perfect: High uncertainty = High error (ALWAYS)
ρ = 0.0  → No relationship: Uncertainty random hai
ρ = -1.0 → Inverse: High uncertainty = Low error (BAD!)

Hamara result: ρ = 0.16 
→ Positive but weak correlation
→ Uncertainty somewhat useful but not perfect
```

### Visualization

```
        High │    ·  ·
             │  ·  ·  · ·
   Actual    │ · ·  · · · ·
   Error     │· · · · · · · ·
             │· · · · · · · · ·
        Low  │· · · · · · · · · ·
             └────────────────────
              Low              High
                  Uncertainty (σ)

Ideal case: Points follow diagonal line (high σ = high error)
Our case: Weak positive trend (ρ = 0.16)
```

### Why MC Dropout > Ensemble Variance?

| Factor | MC Dropout | Ensemble Variance |
|--------|------------|-------------------|
| Samples | 30 per run | 5 runs total |
| Captures | Epistemic uncertainty (model uncertainty) | Run-to-run randomness |
| Range | Higher variance (0.03 - 2.0) | Lower variance (0.001 - 0.5) |
| Result | ρ = 0.16 | ρ = 0.10 |

**Conclusion:** MC Dropout **55% better** correlation with actual error!

---

## 🔬 Experiment B: Multi-Model Ensemble

### Objective
**5 alag trained models combine karke ensemble banana**

### Approach

```
┌──────────────────────────────────────────────────────────┐
│                    TEST GRAPH                             │
└──────────────────────────────────────────────────────────┘
                           │
     ┌──────────┬──────────┼──────────┬──────────┐
     ▼          ▼          ▼          ▼          ▼
┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
│ Model 2 ││ Model 5 ││ Model 6 ││ Model 7 ││ Model 8 │
│(dropout ││(dropout ││(dropout ││(dropout ││(dropout │
│  0.3)   ││  0.3)   ││  0.3)   ││  0.3)   ││  0.2)   │
│         ││         ││         ││         ││         │
│Weight:  ││Weight:  ││Weight:  ││Weight:  ││Weight:  │
│  19.4%  ││  18.5%  ││  18.1%  ││  21.4%  ││  22.6%  │
└────┬────┘└────┬────┘└────┬────┘└────┬────┘└────┬────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
  Pred: 14.5  Pred: 15.2  Pred: 14.8  Pred: 15.5  Pred: 14.9
     │          │          │          │          │
     └──────────┴──────────┼──────────┴──────────┘
                           ▼
              ┌────────────────────────────┐
              │ Weighted Average = 15.0    │
              │ Std = 0.23 (Uncertainty)   │
              └────────────────────────────┘
```

### Weighted Averaging Kaise Kaam Karta Hai?

Models ka weight unke **training R² score** par based hai:

| Model | Training R² | Normalized Weight | Contribution |
|-------|-------------|-------------------|--------------|
| Model 8 | 0.5957 | 22.6% | Highest (best model) |
| Model 7 | 0.5647 | 21.4% | Second best |
| Model 2 | 0.5117 | 19.4% | |
| Model 5 | 0.4882 | 18.5% | |
| Model 6 | 0.4779 | 18.1% | Lowest |

**Final Prediction:**
```
ŷ = 0.226×M8 + 0.214×M7 + 0.194×M2 + 0.185×M5 + 0.181×M6
```

### Ensemble Uncertainty

**Formula:** Standard deviation across 5 model predictions

```python
ensemble_uncertainty = std([M2_pred, M5_pred, M6_pred, M7_pred, M8_pred])
```

**Intuition:** Agar models agree karte hain → Low uncertainty
             Agar models disagree karte hain → High uncertainty

---

### 📊 Experiment B Results

| Model/Ensemble | R² Score | MAE | RMSE | Rank |
|----------------|----------|-----|------|------|
| **Model 7** | **0.0057** ✅ | 4.278 | 11.164 | 🥇 Best |
| Model 6 | -0.0009 | 4.365 | 11.201 | 2nd |
| **Weighted Ensemble** | **-0.0021** | 4.331 | 11.207 | 3rd |
| Model 5 | -0.0023 | 4.287 | 11.208 | 4th |
| Model 8 | -0.0059 | 4.337 | 11.229 | 5th |
| Model 2 | -0.0101 | 4.495 | 11.252 | 6th |

### Surprising Finding! 🤔

**Expected:** Ensemble should beat single models (wisdom of crowds)
**Actual:** Single best model (Model 7) outperforms ensemble!

### Why Ensemble Didn't Help?

```
┌─────────────────────────────────────────────────────────────┐
│                 ENSEMBLE FAILURE ANALYSIS                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. LOW MODEL DIVERSITY                                      │
│     ────────────────────                                     │
│     All 5 models use SAME architecture (PointNetTransfGAT)  │
│     Only difference: dropout rate (0.2 vs 0.3)              │
│     Similar training → Similar errors → No diversity benefit │
│                                                              │
│  2. AVERAGING DILUTES BEST MODEL                            │
│     ────────────────────────────                             │
│     Model 7 (best): R² = 0.0057                             │
│     Model 2 (worst): R² = -0.0101                           │
│     Ensemble: R² = -0.0021                                  │
│                                                              │
│     → Worse models pull down the average!                    │
│                                                              │
│  3. WRONG TEST SET                                           │
│     ──────────────────                                       │
│     Using Model 8's test split for all models               │
│     Other models never saw this data during training        │
│     → May perform worse on unfamiliar data                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Ensemble Uncertainty Quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean σ | 0.229 | Models disagree moderately |
| Spearman ρ | 0.117 | Weak correlation with error |

**Conclusion:** Ensemble disagreement provides **some** signal, but weaker than MC Dropout.

---

## 🎯 Final Conclusions

### Ranking of UQ Methods

| Rank | Method | Spearman ρ | Recommendation |
|------|--------|-----------|----------------|
| 🥇 | MC Dropout (30 samples) | 0.160 | **USE THIS** |
| 🥈 | Ensemble Variance (5 runs) | 0.117 | Okay but slower |
| 🥉 | Combined | 0.160 | Overkill - same as MC |
| 4th | Multi-Model Ensemble | 0.117 | Not worth the complexity |

### Practical Recommendations

```
┌─────────────────────────────────────────────────────────────┐
│              THESIS RECOMMENDATIONS                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ USE:    Single best model (Model 7 or 8)                │
│             + MC Dropout with 30 samples                     │
│                                                              │
│  ❌ AVOID:  Multi-model ensemble                            │
│             (complexity without benefit)                     │
│                                                              │
│  📝 NOTE:   ρ = 0.16 is weak but statistically significant  │
│             (p-value = 0.0)                                  │
│             This means UQ provides SOME useful signal        │
│             but should not be solely relied upon             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Generated Files

```
ensemble_experiments/
├── experiment_a_results.json    # Metrics for Exp A
├── experiment_a_data.npz        # Raw predictions (500MB+)
├── experiment_b_results.json    # Metrics for Exp B  
├── experiment_b_data.npz        # Raw predictions
└── plots/
    ├── exp_a_uncertainty_comparison.png
    ├── exp_a_uncertainty_distributions.png
    ├── exp_a_mc_vs_ensemble.png
    ├── exp_b_ensemble_performance.png
    ├── exp_b_model_comparison.png
    ├── exp_b_uncertainty_distribution.png
    └── exp_b_all_models_scatter.png
```

---

## 📖 References

1. Gal, Y., & Ghahramani, Z. (2016). **Dropout as a Bayesian Approximation**: Representing Model Uncertainty in Deep Learning. ICML.

2. Lakshminarayanan, B., et al. (2017). **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles**. NeurIPS.

3. Kendall, A., & Gal, Y. (2017). **What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?** NeurIPS.

---

## 🔧 How to Reproduce

```bash
# Activate environment
conda activate thesis-env

# Run both experiments (100 graphs, CPU)
python scripts/evaluation/ensemble_uq_experiments.py \
    --experiment both \
    --max-graphs 100 \
    --cpu

# Run only Experiment A (faster)
python scripts/evaluation/ensemble_uq_experiments.py \
    --experiment A \
    --max-graphs 10

# Run only Experiment B
python scripts/evaluation/ensemble_uq_experiments.py \
    --experiment B \
    --max-graphs 100
```

---

*Document generated: February 2026*
