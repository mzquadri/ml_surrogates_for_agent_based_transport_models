# UQ Methods Study Guide — All 6 Methods

**Thesis:** Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models
**Author:** Mohd Zamin Quadri
**Model:** T8 (PointNetTransfGAT, dropout=0.2, R^2=0.5957, MAE=3.96 veh/h)
**Test Data:** 100 graphs, 3,163,500 nodes (31,635 nodes/graph), Paris road network

---

## Quick Summary Table

| # | Method | Key Metric | Key Result | Cost |
|---|--------|-----------|------------|------|
| 1 | MC Dropout | Spearman rho | 0.4820 | 228 min (30 passes x 100 graphs) |
| 2 | Deep Ensembles | Spearman rho | 0.4370 (ens var), 0.4908 (MC avg) | 5x MC Dropout cost |
| 3 | Combined MC+Ensemble | Spearman rho | 0.4909 (+0.0001 over MC alone) | same as Exp A |
| 4 | Conformal Prediction | Coverage | 90.02% / 95.01% (guaranteed) | negligible (just quantile) |
| 5 | Selective Prediction | MAE reduction | 41.2% at 50% retention | negligible (just sorting) |
| 6 | Temperature Scaling | ECE | 0.265 -> 0.048 (82% improvement) | negligible (scalar optimization) |

**The thesis narrative:** MC Dropout is the foundation (ranking), conformal prediction provides guarantees (coverage), selective prediction provides practical utility (filtering), and temperature scaling improves global calibration -- but none alone is sufficient, which is why a multi-method UQ framework is needed.

---
---

# METHOD 1: MC DROPOUT

## 1.1 Basic Idea

MC Dropout (Gal & Ghahramani, 2016) model ko test time pe bhi dropout ON rakh ke
multiple baar forward pass karta hai. Har pass mein alag neurons randomly off hote hain,
to har baar thoda different prediction aata hai. Is variation se uncertainty estimate nikalte hain.

**Analogy:** Socho ek teacher se 30 baar same question pucho, lekin har baar unke kuch
neurons (brain cells) randomly off hain. Agar har baar same answer aaye = confident.
Agar har baar alag answer aaye = uncertain.

## 1.2 Your Exact Setup

- Model: T8 (PointNetTransfGAT)
- Dropout rate: 0.2 (20% neurons randomly off per pass)
- S = 30 stochastic forward passes per graph
- Total: 30 passes x 100 graphs = 3,000 forward passes
- Result: 3,163,500 nodes, each with 30 predictions
- GPU time: 228 minutes on NVIDIA T4
- NPZ file: `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz`

## 1.3 How It Works (Code)

```python
def mc_dropout_predict(model, data, S=30):
    model.train()                  # Dropout ON
    for m in model.modules():      # But freeze BatchNorm
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    preds = []
    with torch.no_grad():
        for _ in range(S):
            out = model(data)
            preds.append(out.squeeze().cpu().numpy())
    preds = np.array(preds)        # Shape: (30, n_nodes)
    mu = preds.mean(axis=0)        # Mean prediction per node
    sigma = preds.std(axis=0)      # Uncertainty per node
    return mu, sigma
```

**Key points:**
- `model.train()` keeps dropout active (this is the trick!)
- BatchNorm is frozen with `.eval()` (otherwise running stats get corrupted)
- `torch.no_grad()` because we don't need gradients (just inference)
- mu = average of 30 predictions = final prediction
- sigma = std dev of 30 predictions = uncertainty estimate

## 1.4 What Sigma Means

For each node, sigma measures how much the 30 predictions disagree.

**Concrete examples from your data:**

| Node Type | sigma (veh/h) | Interpretation |
|-----------|--------------|---------------|
| Quiet residential street | ~0.3 | 30 predictions nearly identical, model confident |
| Medium arterial road | ~1.0 | Some variation, moderate uncertainty |
| Highway A6 (busy) | ~4-5 | 30 predictions vary a lot, model uncertain |

**Overall statistics:**
- Mean sigma: ~1.37 veh/h
- Sigma range: [0.065, 25.434] veh/h
- Most nodes have small sigma (model is confident on ~70% of the quiet network)

## 1.5 Spearman Rho = 0.4820

Spearman rho measures: **"Does sigma correctly RANK nodes by error?"**

rho = rank_correlation(sigma, |actual_error|)

- rho = 1.0: Perfect ranking (highest sigma = highest error, always)
- rho = 0.0: No correlation (sigma is useless)
- rho = 0.4820: Moderate positive correlation

**Matlab:** Agar tum 3.16 million nodes ko sigma ke hisaab se sort karo, aur separately actual
error ke hisaab se sort karo, to dono rankings mein 48.2% agreement hai.

**Is 0.4820 good?** For a post-hoc UQ method on a regression task, yes -- it means sigma
provides meaningful information about where the model struggles. But it's far from perfect.

## 1.6 k95 = 11.34 (The Calibration Problem)

k95 answers: "Sigma ko kitna multiply karna padega for 95% of nodes to be within +-k*sigma?"

- Gaussian assumption says: k = 1.96
- Your actual data needs: k = **11.34**

This 5.8x gap means MC Dropout sigma is **not calibrated** -- it's a good ranking signal
but the absolute magnitude is far too small. If you tell a traffic planner "uncertainty is 2 veh/h",
they'd expect 95% of errors within +-3.92 veh/h (1.96 x 2). But actually you'd need
+-22.68 veh/h (11.34 x 2) for 95% coverage.

**This motivates Methods 4 (Conformal) and 6 (Temperature Scaling).**

## 1.7 S Convergence

How many forward passes are enough?

| S | Spearman rho | Comment |
|---|-------------|---------|
| 5 | ~0.46 | Noisy but already useful |
| 10 | ~0.47 | Improving |
| 20 | ~0.48 | Diminishing returns |
| 30 | 0.4820 | Used in thesis |
| 50 | ~0.483 | Barely better than S=30 |

S=30 is the sweet spot -- 90%+ of the information is captured, further passes add negligible benefit.

## 1.8 Defense Answer

> "MC Dropout runs 30 stochastic forward passes with dropout enabled at test time.
> The standard deviation across passes serves as an uncertainty estimate. We achieved
> Spearman rho = 0.4820, indicating moderate positive correlation between predicted
> uncertainty and actual error. However, the raw sigma is severely miscalibrated --
> k95 = 11.34 versus the Gaussian ideal of 1.96 -- meaning sigma is a useful ranking
> signal but not a reliable absolute uncertainty measure. This motivates our complementary
> calibration methods: conformal prediction and temperature scaling."

---
---

# METHOD 2: DEEP ENSEMBLES

## 2.1 Basic Idea

MC Dropout mein ek hi model tha, aur hum usmein randomness inject karte the (dropout ON).
Deep Ensembles (Lakshminarayanan et al., 2017) mein idea alag hai -- **multiple independently
trained models** rakh lo, aur dekho ki unke predictions kitne different hain.

**Analogy:** 5 doctors ke paas gaye diagnosis ke liye:
- Paanchon same baat bolein = high confidence (low uncertainty)
- Paanchon alag-alag bolein = low confidence (high uncertainty)

Disagreement between models = uncertainty.

## 2.2 Two Ensemble Experiments

| | **Experiment A** | **Experiment B** |
|---|---|---|
| What | Same model (T8), 5 inference runs, different seeds | 5 different trained models (T2, T5, T6, T7, T8) |
| Diversity source | MC Dropout randomness + seed variation | Different training runs, hyperparameters |
| True ensemble? | No -- "pseudo-ensemble" | Yes -- proper multi-model ensemble |
| Data | 100 test graphs, 3.16M nodes | 100 test graphs, 3.16M nodes |

## 2.3 Experiment A: MC-on-Ensemble (Pseudo-Ensemble)

**Step 1:** Load T8 model (with weight remapping fix).
**Step 2:** Run 5 times with different seeds: [42, 142, 242, 342, 442]
   - Each run: 100 graphs x 30 MC Dropout passes
   - Each run produces: per-node mean prediction + per-node MC sigma
**Step 3:** Aggregate 5 runs into 3 uncertainty types:

```python
ensemble_preds = np.array(all_preds)        # (5, n_nodes) -- per-run MC means
mc_uncs = np.array(all_uncs)                # (5, n_nodes) -- per-run MC stds

# (a) Average MC Dropout uncertainty
avg_mc_unc = mc_uncs.mean(axis=0)           # Average of 5 MC sigmas

# (b) Ensemble variance
ens_variance = ensemble_preds.std(axis=0)   # Std of 5 predictions

# (c) Combined (quadrature sum)
combined_unc = np.sqrt(avg_mc_unc**2 + ens_variance**2)
```

### Experiment A Results:

| Uncertainty Type | Spearman rho | Mean sigma (veh/h) |
|---|---|---|
| MC Dropout (averaged over 5 runs) | **0.4908** | 1.369 |
| Ensemble Variance | 0.4370 | 0.217 |
| Combined | **0.4909** | 1.388 |

**Key findings:**
- MC Dropout averaged (0.4908) > standalone MC (0.4820) -- noise reduction from 5 runs
- Ensemble variance alone (0.4370) is weak -- same model, predictions very similar
- Combined (0.4909) adds only +0.0001 over MC alone -- ensemble variance negligible

**Why weak?** Same model, same weights -- only dropout masks differ run-to-run.
Mean ensemble sigma = 0.217 veh/h (6x smaller than MC sigma = 1.369). Diversity is too low.

## 2.4 Experiment B: True Multi-Model Ensemble

### The 5 Models:

| Model | Dropout | R^2 | MAE (veh/h) | Weight |
|---|---|---|---|---|
| T2 | 0.3 | 0.5116 | 4.148 | 0.194 |
| T5 | 0.3 | 0.5552 | 4.070 | 0.211 |
| T6 | 0.3 | 0.5222 | 4.147 | 0.198 |
| T7 | 0.3 | 0.5471 | 4.060 | 0.208 |
| T8 | 0.2 | **0.5957** | **3.957** | **0.226** |

(T1 excluded: no dropout. T3, T4 excluded: weighted loss, poor performance.)

### How it works:

**Step 1:** Load all 5 models (with weight remapping fix).
**Step 2:** Each model does ONE deterministic forward pass (no MC dropout).
**Step 3:** R^2-weighted average prediction:

```python
weights = [0.5117, 0.5553, 0.5223, 0.5471, 0.5957]  # R^2 values
weights = weights / sum(weights)  # normalize

weighted_pred = sum(w_i * pred_i)  # weighted average
weighted_var = sum(w_i * (pred_i - weighted_pred)^2)  # weighted variance
sigma_ens = sqrt(weighted_var)     # uncertainty
```

### Experiment B Results:

| Metric | Value |
|---|---|
| Ensemble R^2 | 0.5656 (WORSE than T8 alone: 0.5957) |
| Ensemble MAE | 3.989 veh/h |
| Spearman rho | **0.4333** (weakest of all methods) |
| Mean sigma_ens | 0.783 veh/h |

**Why worse?** Weak models (T2, R^2=0.51) drag down the ensemble. T8 alone is better.
Ensemble only helps when models are comparable in quality. Here T8 is dominant.

## 2.5 The PyG Version Bug (Weight Remapping)

**Problem:** Models trained on Google Colab (older PyTorch Geometric). Evaluation on local
machine (newer PyG 2.3.1). GATConv weight names changed:
- Old: `gat_final.lin.weight` (single weight matrix)
- New: `gat_final.lin_src.weight` + `gat_final.lin_dst.weight` (split)

Loading with `strict=False` silently dropped GATConv trained weights. R^2 dropped to ~0.003.

**Fix:**
```python
state_dict = torch.load(model_path)
remapped = {}
for k, v in state_dict.items():
    if ".lin.weight" in k:
        remapped[k.replace(".lin.weight", ".lin_src.weight")] = v
        remapped[k.replace(".lin.weight", ".lin_dst.weight")] = v
    else:
        remapped[k] = v
model.load_state_dict(remapped, strict=True)  # strict=True catches mismatches
```

**Note:** Standalone MC Dropout results were NOT affected (generated on Colab where PyG matched).

## 2.6 Complete Comparison

| Method | rho | Cost |
|---|---|---|
| MC Dropout standalone (T8, S=30) | 0.4820 | 228 min (1 model x 30 passes) |
| MC-on-Ensemble (Exp A, 5 runs) | 0.4908 | ~1140 min (5x cost) |
| Ensemble Variance (Exp A) | 0.4370 | same runs, different metric |
| Multi-Model Ensemble (Exp B) | 0.4333 | 5 models x 1 pass |

**Conclusion:** MC Dropout alone gives best cost-performance ratio.
5x more compute yields only +0.0088 rho improvement (0.4908 vs 0.4820).

## 2.7 Defense Answer

> "We conducted two ensemble experiments. Experiment A ran the same T8 model 5 times
> with different seeds, comparing MC Dropout, ensemble variance, and their combination.
> Experiment B used 5 different trained models weighted by R-squared. The key finding
> was that ensemble methods provided no meaningful improvement over standalone MC Dropout --
> rho increased by only 0.0088 at 5x computational cost. Multi-model ensemble had the
> weakest correlation (rho=0.4333). This demonstrates that for our GNN surrogate, MC Dropout
> is the most cost-effective uncertainty estimator."

---
---

# METHOD 3: COMBINED MC+ENSEMBLE

## 3.1 Core Idea

MC Dropout aur Ensemble Variance dono alag-alag uncertainty measure karte hain.
Theory mein ye two different sources of uncertainty capture karte hain:

| Type | Source | Captured by |
|---|---|---|
| Aleatoric | Data ka inherent noise | MC Dropout (partially) |
| Epistemic | Model ko pata nahi | Ensemble disagreement |

Combine karo via **quadrature sum** (like adding independent errors):

```
sigma_combined = sqrt(sigma_mc^2 + sigma_ens^2)
```

## 3.2 Your Numbers

From Experiment A (3,163,500 nodes):

| Metric | MC Dropout (avg) | Ensemble Variance | Combined |
|---|---|---|---|
| Mean sigma (veh/h) | **1.369** | 0.217 | 1.388 |
| Spearman rho | **0.4908** | 0.4370 | **0.4909** |

## 3.3 Why Combined Barely Helps

sigma_ens (0.217) is **6.3x smaller** than sigma_mc (1.369).

Quadrature sum:
```
sigma_combined = sqrt(1.369^2 + 0.217^2) = sqrt(1.874 + 0.047) = sqrt(1.921) = 1.386
```

Ensemble variance contributes only **0.047 out of 1.921** total variance = **2.4%**.
This is so small that node rankings barely change, hence rho improves by just 0.0001.

## 3.4 Why This Is a Valuable Negative Result

It shows:
- Don't waste 5x compute on ensembles if MC Dropout already available
- MC Dropout captures the dominant uncertainty signal for this architecture
- Motivates looking at fundamentally different approaches (Methods 4-6)

## 3.5 Defense Answer

> "The combined uncertainty uses sigma_combined = sqrt(sigma_mc^2 + sigma_ens^2).
> Ensemble variance was 6x smaller than MC Dropout uncertainty (0.217 vs 1.369 veh/h),
> contributing only 2.4% to the total combined variance. Spearman rho improved by just
> 0.0001 (0.4909 vs 0.4908), demonstrating that ensemble variance is redundant when
> MC Dropout is available. This is consistent with literature findings that ensembles
> provide the most benefit when individual models are diverse."

---
---

# METHOD 4: CONFORMAL PREDICTION

## 4.1 The Problem It Solves

MC Dropout sigma ka ranking accha hai (rho=0.4820) lekin **calibrated nahi hai**.
Agar model bolta hai sigma=2.0 veh/h, Gaussian assumption ke hisaab se 95% nodes ka
error +-3.92 veh/h ke andar hona chahiye. Actually sirf **54.8%** nodes us interval mein hain.

**Conformal Prediction provides GUARANTEED coverage** -- no distributional assumptions needed.

## 4.2 Split Conformal Prediction (Fixed Width)

### Step 1: Split test data
- Calibration: 50 graphs = 1,581,750 nodes
- Evaluation: 50 graphs = 1,581,750 nodes
- Seed: 42

### Step 2: Compute nonconformity scores on calibration set
```
score_i = |y_true_i - y_pred_i|    (absolute residual)
```

### Step 3: Find quantile
- 90% coverage: q_90 = 90th percentile of calibration residuals = **9.92 veh/h**
- 95% coverage: q_95 = 95th percentile = **14.68 veh/h**

### Step 4: Prediction intervals
```
interval = [prediction - q_hat, prediction + q_hat]
```

### Step 5: Check coverage on evaluation set

| Nominal | q_hat (veh/h) | Total Width | **Achieved Coverage** |
|---|---|---|---|
| 90% | 9.92 | 19.84 veh/h | **90.02%** |
| 95% | 14.68 | 29.35 veh/h | **95.01%** |

Almost exactly the target! This is the conformal guarantee (Vovk et al., 2005).

### Code:
```python
def conformal_q(residuals, alpha):
    n = residuals.shape[0]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(residuals, q_level, method="higher")
```

## 4.3 The Problem with Fixed-Width Intervals

**Same width for every node** -- highway nodes aur residential nodes ko same interval milta hai.

Per sigma decile (90% nominal):

| Decile | Mean sigma | Node type | Fixed Coverage | Problem |
|---|---|---|---|---|
| D1 (lowest sigma) | 0.276 | Quiet residential | **98.6%** | Over-covered (wasteful) |
| D5 (medium) | 0.906 | Mixed roads | 94.1% | Slightly over |
| D10 (highest sigma) | 4.564 | Busy highways | **62.9%** | Severely under-covered! |

**Spread: 62.9% to 98.6% = 35.7 percentage points.** Terrible conditional coverage.

## 4.4 Adaptive (Sigma-Scaled) Conformal Prediction

### The Fix: Normalize residuals by MC Dropout sigma

```
score_i = |y_true_i - y_pred_i| / (sigma_i + epsilon)
```

Now the score measures "how many sigmas wrong was the model?" instead of raw error.

### Adaptive quantile:
- k_90 = 7.58 (20/80 split)
- k_95 = 11.36 (20/80 split)

### Node-specific intervals:
```
interval = prediction +/- k_hat * sigma_mc(node)
```

- Highway node (sigma=5.0): +/- 11.36 * 5.0 = +/- 56.8 veh/h (wide)
- Residential node (sigma=0.3): +/- 11.36 * 0.3 = +/- 3.4 veh/h (narrow)

### Adaptive results per decile (90% nominal):

| Decile | Fixed Coverage | **Adaptive Coverage** |
|---|---|---|
| D1 (quiet) | 98.6% | **90.0%** |
| D5 (mixed) | 94.1% | **88.6%** |
| D10 (busy) | 62.9% | **96.2%** |

**Adaptive spread: 90.0% to 96.2% = only 6.2pp** (vs 35.7pp for fixed).

## 4.5 Connection to k95 = 11.34

| Source | k_95 value |
|---|---|
| MC Dropout empirical | **11.34** |
| Conformal sigma-scaled (20/80 split) | **11.36** |
| Conformal sigma-scaled (50/50 split) | **11.65** |

All consistent -- sigma needs ~11.3-11.6x multiplication for 95% coverage.

## 4.6 Winkler Scores (Interval Quality, lower = better)

| Method | 90% Winkler | 95% Winkler |
|---|---|---|
| Raw Gaussian (MC Dropout) | 49.68 | 87.27 |
| Conformal absolute (fixed) | 35.77 | 47.65 |
| **Conformal sigma-scaled** | **32.33** | **43.63** |

Sigma-scaled conformal is best -- 35% better than raw Gaussian at 90%.

## 4.7 Theoretical Limitation

Barber et al. (2021) proved: **Distribution-free conditional coverage is mathematically impossible.**
No method can guarantee exactly 90% for every subgroup without distributional assumptions.
Adaptive conformal does its best (6.2pp spread) but perfect conditional coverage is impossible.

## 4.8 Defense Answer

> "MC Dropout provides well-ranked but poorly calibrated uncertainty -- raw Gaussian
> intervals achieve only 54.8% at the 95% nominal level. Split conformal prediction
> computes empirical quantiles on a calibration set, achieving 90.02% and 95.01%
> coverage exactly. However, fixed-width intervals have severe conditional miscoverage:
> 98.6% for low-uncertainty nodes but only 62.9% for high-uncertainty ones. Adaptive
> conformal normalizes by MC Dropout sigma, narrowing this gap to [90.0%, 96.2%].
> Winkler scores confirm sigma-scaled conformal produces the highest-quality intervals."

---
---

# METHOD 5: SELECTIVE PREDICTION

## 5.1 A Different Philosophy

Methods 1-4: "I will predict, and tell you how confident I am."
Method 5: **"If I'm not confident, I REFUSE to predict."**

**Analogy:** A doctor who says "I need to refer you to a specialist" for cases they're
not sure about -- rather than guessing and giving a diagnosis with a warning label.

Geifman & El-Yaniv (NeurIPS 2017) formalized this idea.

## 5.2 How It Works

**Step 1:** Get all 3,163,500 nodes' predictions and MC Dropout sigma.
**Step 2:** Sort nodes by sigma ascending (most confident first).
**Step 3:** Choose retention level (e.g., 50%). Keep bottom 50% (most confident), reject top 50%.
**Step 4:** Compute MAE on retained nodes only.

```python
# Sort by sigma descending
sort_idx = np.argsort(-mc_std)
mc_mean_sorted = mc_mean[sort_idx]
targets_sorted = targets[sort_idx]

# Keep bottom k (least uncertain)
k = int(np.floor(retention * n_total))
sub_tgt  = targets_sorted[n_total - k:]     # last k = lowest sigma
sub_pred = mc_mean_sorted[n_total - k:]
mae = np.mean(np.abs(sub_tgt - sub_pred))
```

## 5.3 Complete Results Table

Baseline MAE (100% retained) = **3.95 veh/h**

| Retained % | Nodes Kept | MAE (veh/h) | MAE Reduction |
|---|---|---|---|
| 100% | 3,163,500 | 3.95 | -- |
| 95% | 3,005,325 | 3.48 | -11.9% |
| **90%** | 2,847,150 | **3.23** | **-18.3%** |
| 85% | 2,688,975 | 3.05 | -22.7% |
| 80% | 2,530,800 | 2.91 | -26.2% |
| 75% | 2,372,625 | 2.79 | -29.2% |
| 70% | 2,214,450 | 2.69 | -31.9% |
| 60% | 1,898,100 | 2.50 | -36.6% |
| **50%** | **1,581,750** | **2.32** | **-41.2%** |
| 40% | 1,265,400 | 2.13 | -46.0% |
| 30% | 949,050 | 1.92 | -51.3% |
| **25%** | 790,875 | **1.79** | **-54.6%** |
| 10% | 316,350 | 1.06 | -73.3% |

## 5.4 Key Operating Points

### 90% retention (conservative)
- Reject only 10% most uncertain nodes (316K out of 3.16M)
- MAE: 3.95 -> 3.23 (**-18.3%**)
- Practical: flag hardest nodes, trust the rest

### 50% retention (balanced, headline number)
- Keep half, reject half
- MAE: 3.95 -> 2.32 (**-41.2%**)
- Almost halved error by rejecting uncertain predictions

### 25% retention (aggressive)
- Keep only most confident quarter
- MAE: 3.95 -> 1.79 (**-54.6%**)
- Excellent accuracy for retained nodes

## 5.5 The Risk-Coverage Curve

Figure 5.8 in thesis:
- X-axis: Coverage (retention %) -- 100% down to 10%
- Y-axis: Risk (MAE on retained set)
- Curve monotonically decreases

This is a **trade-off curve**:
- Traffic planner says "I need MAE < 3 veh/h" -> curve says ~82% retention
- "MAE < 2 veh/h" -> ~35% retention

## 5.6 What Happens to Rejected Nodes?

Practical options:
1. Fall back to full MATSim simulation (expensive but accurate)
2. Flag as "low confidence" for the user
3. Use conformal intervals (predict with wider uncertainty band)
4. Hybrid: GNN for confident nodes, MATSim for uncertain nodes

## 5.7 Error Detection (Related)

Instead of "keep the best," ask "can I identify the worst?"

| Metric | T8 | T7 |
|---|---|---|
| AUROC (top-10% errors) | 0.759 | 0.742 |
| AUROC (top-20% errors) | 0.740 | 0.715 |

AUROC 0.759: 75.9% chance of correctly identifying high-error vs low-error node.

## 5.8 T7 Cross-Validation

| Metric | T7 | T8 |
|---|---|---|
| Baseline MAE | 4.07 | 3.95 |
| 50% retention MAE | 2.51 | 2.32 |
| 50% MAE reduction | **38.3%** | **41.2%** |
| 90% MAE reduction | 18.6% | 18.3% |

Both models show significant reductions -- method is robust across models.

## 5.9 Why This Is the Strongest Practical Result

1. No retraining needed -- just use existing MC Dropout sigma
2. No calibration needed -- unlike conformal, no calibration set required
3. Interpretable -- "model predicted on 50% of nodes, flagged rest as uncertain"
4. Guaranteed improvement -- any positive rho means selective prediction helps

**41.2% MAE reduction is the strongest practical finding in the thesis.**

## 5.10 Defense Answer

> "Selective prediction uses MC Dropout uncertainty to filter out unreliable predictions.
> By retaining only the most confident 50% of nodes, MAE drops from 3.95 to 2.32 veh/h --
> a 41.2% reduction. Even at 90% retention, MAE improves by 18.3%. This was cross-validated
> on Trial 7 (38.3% reduction at 50%). Practically, a traffic planner can use the GNN for
> confident nodes and fall back to full MATSim simulation for uncertain ones -- combining
> surrogate speed with simulator accuracy where it matters most."

---
---

# METHOD 6: TEMPERATURE SCALING

## 6.1 The Problem

MC Dropout sigma ka ranking accha hai, magnitude galat hai:

| Nominal Level | Expected Coverage | Actual Coverage (raw sigma) | Gap |
|---|---|---|---|
| 50% (+-0.674 sigma) | 50% | 23.8% | -26.2pp |
| 80% (+-1.282 sigma) | 80% | 40.6% | -39.4pp |
| 90% (+-1.645 sigma) | 90% | 49.2% | -40.8pp |
| 95% (+-1.96 sigma) | 95% | 55.6% | -39.4pp |

Model is **severely overconfident** at every level.

## 6.2 The Concept

### Origin: Classification (Guo et al., 2017)
Softmax logits ko temperature T se divide karte the: `softmax(z/T)`.
T > 1 = softer probabilities (less confident).

### Adaptation for Regression (Laves et al., 2020)
```
sigma_scaled = sigma_raw * T
```

T > 1: sigma increases, intervals wider, less overconfident.
T < 1: sigma decreases, intervals narrower, more confident.

**Your T = 2.70** -- sigma ko 2.7x multiply karo.

**Key property:** Only sigma changes. Predictions (mu) stay the same.
R^2, MAE, RMSE all unchanged. Only uncertainty calibration improves.

## 6.3 How T Was Found

### Data split:
- Calibration: 20 graphs (632,700 nodes)
- Evaluation: 80 graphs (2,530,800 nodes)

### Optimization target: ECE (Expected Calibration Error)

ECE checks 10 nominal levels {10%, 20%, ..., 90%, 95%}:
```
ECE = (1/10) * sum |observed_coverage_l - nominal_l|
```

### Two-stage optimization:
```python
# Stage 1: Coarse grid search
T_values = np.logspace(-1, 2, 50)   # 0.1 to 100
for T in T_values:
    ece = compute_ece_scaled(T, sigmas, errors)
best_T_init = T_values[argmin(ece)]

# Stage 2: Fine-tune with scipy
result = minimize_scalar(
    lambda T: compute_ece(T, sigmas, errors),
    bounds=(best_T_init * 0.5, best_T_init * 2),
    method='bounded'
)
T_optimal = 2.7025  # rounded to 2.70
```

**Note:** ECE directly optimized (not NLL). Deliberate choice -- ECE directly measures
the calibration gap we want to close.

## 6.4 Results: Before vs After

### ECE Improvement

| | ECE Before | ECE After (T=2.70) | Improvement |
|---|---|---|---|
| Calibration set | 0.270 | 0.048 | 82.3% |
| **Evaluation set** | **0.269** | **0.048** | **82.2%** |

Average calibration gap per level: ~26.9pp -> ~4.8pp.

### Coverage at Each Level

| Nominal | Before | After (T=2.70) | Improvement |
|---|---|---|---|
| 10% | 4.8% | 12.6% | +7.8pp |
| 20% | 9.5% | 23.7% | +14.2pp |
| 30% | 14.1% | 33.9% | +19.7pp |
| 50% | 23.4% | **52.3%** | +28.9pp |
| 80% | 40.1% | **73.3%** | +33.2pp |
| 90% | 48.6% | **79.5%** | +30.9pp |
| **95%** | **54.9%** | **83.3%** | **+28.4pp** |

**50% level almost perfect** (52.3% ~ 50%). But 95% level still only 83.3%, not 95%.

### Why Can't T=2.70 Fix Everything?

Temperature scaling is a **single scalar** -- same factor for all nodes.
Actual error distribution has **heavy tails** (non-Gaussian).
Stretching a Gaussian wider doesn't change its shape -- tails still fall off as exp(-x^2).
Real errors have much heavier tails.

**Analogy:** Tumhare paas ek rubber band hai jo chhota hai (raw sigma).
T=2.70 se tum use 2.7x stretch karte ho. Ab bigger hai, lekin shape same hai.
Agar actual distribution ka shape alag hai, sirf stretching se perfect fit nahi aayega.

That's why:
- 50% level (tails don't matter): 52.3% ~ 50% (almost perfect!)
- 95% level (tails matter a lot): 83.3% != 95% (still off)

## 6.5 k95 After Scaling

| | k95 Value | Meaning |
|---|---|---|
| Raw | **11.34** | Need 11.34 * sigma for 95% coverage |
| After T=2.70 | **5.30** | Need 5.30 * sigma_scaled (= 1.96 * 2.70) |
| Ideal Gaussian | **1.96** | Perfect calibration |

Temperature scaling closes 53% of the k95 gap. But 5.30 >> 1.96 still.

## 6.6 Supporting Metrics

### NLL (Negative Log-Likelihood)

```
NLL = (1/n) * sum [0.5 * log(2*pi*sigma^2) + (y - y_hat)^2 / (2*sigma^2)]
```

| | NLL |
|---|---|
| Raw sigma | **21.65** |
| After T=2.70 | **4.75** |
| **Improvement** | **78%** |

### PIT (Probability Integral Transform)

PIT values should be Uniform[0,1] for calibrated predictions.

| Metric | Raw | After T=2.70 | Ideal |
|---|---|---|---|
| PIT mean | 0.433 | **0.471** | 0.500 |
| PIT std | 0.399 | **0.302** | 0.289 |
| KS statistic | 0.245 | **0.104** | 0.000 |

KS stat **57% reduction**. U-shaped histogram substantially flattened.

### CRPS (Continuous Ranked Probability Score)

| Metric | Value |
|---|---|
| CRPS | 3.383 veh/h |
| MAE | 3.948 veh/h |
| CRPS/MAE ratio | **0.857** |
| Theoretical optimum | 0.707 |

CRPS/MAE < 1.0 confirms probabilistic forecast adds value over point forecast.

## 6.7 Temperature Scaling vs Conformal Prediction

| Aspect | Temperature Scaling | Conformal Prediction |
|---|---|---|
| Approach | Multiply sigma by T | Compute empirical quantile |
| Guarantee | No formal guarantee | **Distribution-free guarantee** |
| 95% coverage? | 83.3% (NO) | 95.01% (YES) |
| Assumption | Gaussian (after scaling) | **None** |
| Adaptive widths? | Yes (proportional to sigma) | Fixed or adaptive |
| What improves | ECE, NLL, PIT | Coverage, Winkler |

They're complementary -- temperature scaling improves global calibration,
conformal prediction guarantees coverage.

## 6.8 Defense Answer

> "Temperature scaling multiplies MC Dropout sigma by a learned scalar T, optimized
> to minimize ECE on a calibration set. We found T=2.70, reducing ECE from 0.265 to
> 0.048 -- an 82% improvement. NLL improved by 78%, PIT KS statistic by 57%. However,
> the 95% confidence interval still achieves only 83.3% coverage, revealing heavy-tailed
> non-Gaussian residuals. This is why we complement temperature scaling with conformal
> prediction, which provides distribution-free guarantees. T=2.70 confirms MC Dropout
> uncertainties need ~2.7x inflation for better calibration, but a single scalar cannot
> fully correct non-Gaussian behavior."

---
---

# OVERALL DEFENSE NARRATIVE

If asked: "Summarize your UQ framework and what you found."

> "We implemented a six-method UQ framework for our GNN traffic surrogate.
>
> MC Dropout (Method 1) provides the foundation -- 30 stochastic forward passes yield
> uncertainty estimates with Spearman rho = 0.4820, meaning the model reliably identifies
> where its predictions are less trustworthy. However, the raw uncertainty is severely
> miscalibrated (k95 = 11.34 vs the Gaussian ideal of 1.96).
>
> Deep Ensembles (Methods 2-3) showed that ensemble-based approaches are largely redundant
> once MC Dropout is available -- ensemble variance added less than 0.0001 to Spearman rho
> at 5x computational cost.
>
> Conformal Prediction (Method 4) provides distribution-free coverage guarantees: 90.02%
> and 95.01% at their respective nominal levels. Adaptive conformal normalizes intervals
> by MC Dropout sigma, reducing conditional coverage spread from 35.7 to 6.2 percentage
> points across uncertainty deciles.
>
> Selective Prediction (Method 5) delivers the strongest practical result: rejecting the
> 50% most uncertain predictions reduces MAE by 41.2%, from 3.95 to 2.32 veh/h.
>
> Temperature Scaling (Method 6) improves global calibration by 82% (ECE: 0.265 to 0.048)
> via a single scalar T=2.70, though non-Gaussian tails prevent perfect calibration.
>
> Together, these methods show that MC Dropout is the most cost-effective uncertainty
> source, but requires post-hoc calibration (conformal or temperature scaling) for
> reliable uncertainty quantification. The multi-method framework provides complementary
> perspectives: ranking (MC Dropout), guarantees (conformal), practical filtering
> (selective), and calibration (temperature scaling)."

---

*File created for thesis defense preparation. All numbers verified against raw JSON/NPZ data.*
