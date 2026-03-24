# Meeting Prep: Answer Cards for March 30 Thesis Defense
## Mohd Zamin Quadri — UQ for GNN Surrogates of Agent-Based Transport Models

---

## DOMINIK'S 10 PREDICTED QUESTIONS

### Q1: How do you distinguish epistemic from aleatoric uncertainty?

**Short answer:** We do not decompose them. MC Dropout captures a mixture of both: the predictive variance includes epistemic uncertainty (model uncertainty from limited data/architecture) and aleatoric uncertainty (irreducible noise in the MATSim target values). We discuss this explicitly in Section 6.5.4.

**Key numbers:**
- Mean sigma = 1.37 veh/h across 3.16M predictions
- This is small relative to MAE = 3.95, suggesting the model underestimates total uncertainty
- Temperature scaling (T=2.70) inflates sigma by 2.7x, which partially compensates

**If pressed:** "Decomposition would require either heteroscedastic output heads (for aleatoric) or ensembles (for epistemic), both of which we identify as future work. Our conformal prediction layer provides coverage guarantees regardless of the uncertainty source."

**Thesis reference:** Section 2.3 (background), Section 6.5.4 (discussion)

---

### Q2: What does PIT look like after temperature scaling?

**Short answer:** Temperature scaling substantially improves PIT uniformity. The KS statistic drops from 0.245 to 0.104 (57% reduction). The U-shape flattens significantly — first bin density drops from 0.284 to 0.088, last bin from 0.231 to 0.117.

**Key numbers:**
- KS before: 0.245, KS after: 0.104 (−57.4%)
- PIT mean shifts from 0.433 toward 0.471 (ideal: 0.500)
- PIT std drops from 0.399 to 0.302 (ideal: 0.289)
- Residual non-uniformity: still excess mass in extreme bins (tails), confirming the Gaussian assumption is imperfect even after scaling

**If pressed:** "Temperature scaling is a single-parameter correction (sigma_scaled = sigma * T). It can fix the scale of the distribution but not its shape. The residual PIT non-uniformity reflects that the true error distribution has heavier tails than Gaussian — consistent with the extreme predictions on high-uncertainty road segments."

**Thesis reference:** Section 6.5.4, Figure in discussion chapter

---

### Q3: Why only MC Dropout? What about GEBM or other methods?

**Short answer:** MC Dropout was chosen because (1) it requires no architectural changes or retraining — just enabling dropout at test time; (2) it's well-established in the BNN literature; (3) it's computationally feasible with our existing pipeline. We cite Fuchsgruber et al.'s GEBM (NeurIPS 2024) as a promising post-hoc alternative in our related work (Section 2.4.3) and future directions (Section 6.5.5).

**Key context about GEBM:**
- GEBM is a post-hoc energy-based method for epistemic uncertainty on GNNs
- Unlike MC Dropout, it requires only a single forward pass → much cheaper
- It specifically targets epistemic uncertainty, whereas MC Dropout conflates epistemic + aleatoric
- Applying GEBM to our setting would be a natural extension

**If pressed:** "We explicitly recommend GEBM as future work because it addresses two limitations of our approach: (1) the S-fold computational cost, and (2) the lack of epistemic/aleatoric decomposition."

**Thesis reference:** Section 2.4.3 (related work), Section 6.5.5 (future directions)

---

### Q4: Spearman rho = 0.48 — is that actually useful?

**Short answer:** Yes, demonstrably useful through downstream tasks. While 0.48 is a moderate correlation, it is sufficient for:

**Key numbers showing practical utility:**
- Selective prediction: retaining 50% most certain → MAE drops 41.2% (3.95 → 2.32)
- Retaining 90% → MAE drops 18.3% (3.95 → 3.23)
- Error detection AUROC: 0.79 (top-10% errors), 0.76 (top-20% errors) — well above random (0.50)
- Bootstrap 95% CI for rho: [0.4599, 0.4689] — statistically significantly above zero

**If pressed:** "The selective prediction curve (Figure 5.8) is the best evidence. A rho of 0.48 means uncertainty successfully separates high-error from low-error predictions — not perfectly, but enough to enable meaningful risk-aware decision support. The AUROC numbers confirm this independently."

**Thesis reference:** Section 5.5 (selective prediction), Section 5.7 (error detection)

---

### Q5: Does uncertainty correlate with graph structure (degree, centrality)?

**Short answer:** We perform stratified analysis by feature quartiles. Uncertainty is higher on road segments with higher base-case volume (VOL_BASE_CASE) and higher capacity (CAPACITY). The top uncertainty decile has mean MAE of 10.44 veh/h vs 1.06 for the bottom decile, confirming that uncertainty tracks with prediction difficulty.

**Key numbers:**
- Per-graph variation: mean rho = 0.4643, std = 0.03 across 100 graphs
- Conditional coverage (global conformal) ranges from 98.6% (low-sigma) to 62.9% (high-sigma)
- Adaptive conformal narrows this to [90.0%, 96.2%]

**If pressed:** "We did not analyze graph-theoretic features (degree, betweenness) directly. The GATConv architecture implicitly aggregates neighbourhood information, so the learned uncertainty already incorporates local topology. A formal graph-structural analysis would be interesting future work."

**Thesis reference:** Section 5.6 (stratified analysis), Section 6 (discussion)

---

### Q6: Temperature T=2.70 is large — what does that mean?

**Short answer:** T=2.70 means the raw MC Dropout standard deviations need to be inflated by a factor of 2.7 to achieve proper calibration. This confirms that MC Dropout with our architecture severely underestimates uncertainty — the variational posterior is too concentrated around the MAP estimate.

**Key numbers:**
- Raw sigma mean: 1.37 veh/h → scaled sigma: 3.70 veh/h
- Even after scaling, k_95 ≈ 5.30 (vs ideal 1.96) — still not perfectly Gaussian
- ECE drops from 0.269 to 0.048 (82% improvement)
- This is consistent with known behavior of MC Dropout: Gal & Ghahramani (2016) note that dropout rate and network architecture strongly affect calibration

**If pressed:** "A high T is expected for approximate Bayesian methods. The dropout rate (p=0.1 in our architecture) was optimized for predictive accuracy, not calibration. Higher dropout would produce wider uncertainty bands but worse point predictions — there's a fundamental trade-off."

**Thesis reference:** Section 5.4 (temperature scaling), Section 6.1 (discussion)

---

### Q7: The ensemble experiments show near-zero R2 — but you call it "inconclusive"?

**Short answer:** We traced the near-zero R2 to a PyTorch Geometric API version mismatch. The checkpoint stores `lin.weight` (old PyG format) while PyG 2.3.1 expects `lin_src.weight` and `lin_dst.weight`. Loading with `strict=False` silently drops these keys, leaving the final two GATConv layers randomly initialized. Remapping the keys restores R2 ≈ 0.57. Therefore, the ensemble results are artifacts of a bug, not evidence that ensembles don't work.

**Key evidence:**
- Predictions compressed to [-1.6, 3.0] instead of [-186, 107] veh/h
- Mean sigma = 0.13 (ensemble) vs 1.37 (correct MC Dropout)
- Weight remapping on a 3-graph subset restores R2 to 0.57
- The standalone MC Dropout NPZ (produced Jan 26 with matching PyG version) is correct

**If pressed:** "We label this RQ3 as 'Inconclusive' rather than claiming ensembles fail. The correct experiment requires re-running ensemble inference with the weight remapping fix, which we identify as future work. Within the degraded context, MC Dropout still outperforms ensemble variance by 55% (rho 0.16 vs 0.10), but this comparison is not scientifically valid for drawing conclusions about ensembles in general."

**Thesis reference:** Section 4.4 (experimental setup), Section 5.3 (results), Section 6.4 (discussion)

---

### Q8: How does your CRPS/MAE compare to other papers?

**Short answer:** Our CRPS/MAE ratio of 0.857 is above the theoretical optimum of 1/√2 ≈ 0.707 (for a perfectly calibrated Gaussian). The 21.2% excess over the optimum quantifies the miscalibration cost.

**Key context:**
- Theoretical: For N(mu, sigma^2), CRPS = sigma/sqrt(pi), MAE = sigma*sqrt(2/pi), ratio = 1/sqrt(2) = 0.707
- Our 0.857: sigma is too small relative to true errors → higher CRPS relative to MAE
- Weather forecasting (Gneiting et al. 2005): CRPS used as primary evaluation metric
- GCN + MC Dropout (Murad et al. 2021): applied to molecular property prediction

**If pressed:** "A ratio above 0.707 means the Gaussian predictive distribution is underdispersed — it concentrates too much mass near the mean. Temperature scaling partially addresses this (the post-TS PIT is closer to uniform). The ratio provides a single number summary of how far the predictive distribution is from ideal calibration."

**Thesis reference:** Section 6.3 (CRPS/MAE benchmark paragraph)

---

### Q9: S=30 forward passes — what's the computational cost?

**Short answer:** 228 minutes (3.8 hours) for the full 100-graph test set with S=30. The S-convergence study shows diminishing returns beyond S≈25-30.

**Key numbers:**
- Full S=30 inference: 228 min on CPU
- S-convergence (10-graph subset, 26.4 min for S=5 to S=50):
  - S=5: rho=0.410, S=10: rho=0.439, S=30: rho=0.458, S=50: rho=0.463
  - S=30 to S=50: <1% rho improvement, not worth the 67% extra compute
- By comparison: one MATSim simulation = hours per scenario; GNN inference = seconds per scenario

**If pressed:** "The S=30 cost is a one-time evaluation cost, not per-scenario. In deployment, you would run S=30 passes for each new scenario — still seconds to minutes, vs hours for MATSim. The S-convergence curve (Figure 5.10) provides practical guidance: S=20-30 captures >95% of the asymptotic uncertainty quality."

**Thesis reference:** Section 5.8 (S-convergence), Figure 5.10

---

### Q10: What about conditional coverage? Does adaptive conformal actually help?

**Short answer:** Yes, dramatically. Standard conformal prediction achieves exact marginal coverage (95.01% overall) but has severe conditional coverage variation: 98.6% for low-uncertainty nodes vs 62.9% for high-uncertainty nodes. Adaptive conformal (normalizing nonconformity scores by sigma) narrows this to [90.0%, 96.2%] — much closer to uniform conditional coverage.

**Key numbers (decile analysis):**
- Global conformal 90% coverage by decile: D1=98.6%, D5=94.1%, D10=62.9%
- Adaptive conformal 90% coverage by decile: D1=90.0%, D5=88.6%, D10=96.2%
- The variance in conditional coverage drops dramatically
- Theory: Barber et al. (2021) prove that marginal coverage ≥ (1-alpha) holds unconditionally, but conditional coverage requires additional structure

**If pressed:** "The high-uncertainty decile (D10, mean sigma=4.56 veh/h) has mean MAE of 10.48 veh/h — these are the hardest predictions. Standard conformal under-covers them because the fixed quantile q=9.92 is too small relative to their errors. Adaptive conformal widens their intervals proportionally, restoring conditional coverage."

**Thesis reference:** Section 5.4.3 (conditional coverage), Section 6.2 (discussion)

---

## ELENA'S PREDICTED QUESTIONS (Transport Domain Expert)

### E1: How does R2=0.60 compare to the full-dataset R2=0.91 in Natterer et al.?

**Short answer:** The difference is entirely explained by dataset size. We use 1,000 scenarios (10% subset) vs the full 10,000 scenarios in Natterer et al. (2025). This is explicitly stated throughout the thesis. The 10% subset was chosen for computational feasibility of the UQ analysis (3.16M predictions × S=30 = 94.9M forward-pass outputs).

**If pressed:** "The UQ methods themselves are independent of dataset size. Our contributions are about quantifying and calibrating uncertainty, not about achieving state-of-the-art point prediction. A larger dataset would improve R2 but the calibration findings would likely hold."

---

### E2: Are the policy scenarios (capacity reductions) representative of real urban planning?

**Short answer:** The scenarios represent synthetic capacity reductions on road segments, generated by MATSim for the Paris road network. They are representative of a specific class of interventions (e.g., lane closures, construction). The thesis focuses on UQ methodology rather than policy realism.

**If pressed:** "Real urban planning involves multi-modal, time-varying interventions. Our framework is agnostic to the intervention type — it quantifies uncertainty on the surrogate's Delta_v predictions regardless of what generates the scenarios."

---

### E3: How would uncertainty information be used in practice by a planner?

**Short answer:** Three practical use cases:
1. **Selective prediction:** Only trust predictions where uncertainty is low (the 50% most certain cover MAE < 2.32 veh/h)
2. **Confidence intervals:** Conformal intervals give coverage-guaranteed error bounds per road segment
3. **Risk flagging:** High-uncertainty segments (top decile, sigma > 2.65) should trigger full MATSim simulation for verification

**Thesis reference:** Section 6.3 (practical implications), Section 7 (conclusion)

---

### E4: Why Delta_v (change in volume) instead of absolute volume?

**Short answer:** Delta_v = V_policy - V_baseline isolates the marginal effect of the policy intervention. Predicting absolute volume would conflate the baseline traffic pattern with the policy effect, making it harder to assess which roads are most affected by the intervention.

**If pressed:** "This is the formulation from Natterer et al. (2025). The uncertainty quantification applies equally to either formulation — the methods are target-agnostic."

---

### E5: Could you apply this to other cities / networks?

**Short answer:** Yes, the methodology is transferable. The GNN architecture operates on graph structure (nodes = road segments, edges = connections) which generalizes across networks. However, the model would need retraining on the new network's MATSim simulations. The UQ methods (MC Dropout, conformal prediction, temperature scaling) are model-agnostic post-hoc methods that would transfer directly.

---

### E6: What about the temporal dimension — time-of-day effects?

**Short answer:** There is no temporal dimension in this dataset. Each scenario produces a single equilibrium traffic state (VOL_BASE_CASE is the converged MATSim rate, not an hourly time series). The problem is a static, cross-sectional regression with no temporal component. Extending to time-varying predictions would be interesting future work.

**Thesis reference:** Section 3 (methodology) — explicitly states "static, cross-sectional regression"

---

### E7: Why did you choose GATConv specifically?

**Short answer:** GATConv (Graph Attention Network convolution) was the architecture from Natterer et al.'s pipeline. The attention mechanism allows the model to learn different importance weights for different neighboring road segments, which is well-suited to traffic networks where influence between roads is heterogeneous. We evaluated 8 trial configurations with varying hyperparameters; Trial 8 (GATConv with dropout=0.1, lr=5e-4, batch_size=8) performed best.

---

## CURVEBALL QUESTIONS (Unexpected Deep Dives)

### C1: "Isn't MC Dropout just a poor man's ensemble?"

**Answer:** Technically yes — MC Dropout can be viewed as an implicit ensemble over sub-networks (Gal & Ghahramani, 2016). However, it has practical advantages: (1) no additional training (just flip dropout on at test time), (2) single model checkpoint, (3) memory-efficient. Our S-convergence study shows S=30 sub-networks are sufficient. A proper ensemble of independently trained models would likely give better uncertainty estimates, but we couldn't test this due to the PyG loading bug.

### C2: "Your conformal intervals are very wide (29.35 veh/h at 95%). Is that useful?"

**Answer:** The width reflects the model's actual prediction quality. With MAE ≈ 3.95 veh/h on average but a long tail of large errors, the 95% conformal quantile must be wide to cover 95% of cases. The adaptive conformal intervals are more useful in practice because they are narrower for confident predictions and wider for uncertain ones. The Winkler scores show that sigma-scaled conformal intervals are 50% better than naive Gaussian at the 95% level.

### C3: "What would you do differently if you started over?"

**Answer:** Three things: (1) Use the full 10,000-scenario dataset for better point prediction. (2) Implement heteroscedastic regression (predict both mu and sigma) to disentangle aleatoric and epistemic uncertainty. (3) Try GEBM (Fuchsgruber et al., 2024) as a single-pass alternative to MC Dropout, avoiding the S-fold computational cost.

### C4: "How sensitive are the conformal results to the calibration/evaluation split?"

**Answer:** We tested two splits: 50/50 (primary) and 20/80 (audit). The results are very consistent: q_95 = 14.68 (50/50) vs 14.77 (20/80), and both achieve near-exact 95% coverage. This stability is expected from conformal prediction theory — the coverage guarantee holds for any exchangeable split.

### C5: "Why not use Bayesian GNNs with proper priors?"

**Answer:** Full Bayesian inference on our architecture (4 GATConv layers, ~170K parameters, 31,635-node graphs) would require variational inference or MCMC, both of which are computationally prohibitive and would require significant architectural changes. MC Dropout is a practical approximation that scales to our problem size. This is a well-recognized trade-off in the Bayesian deep learning literature (Gal, 2016; Wilson & Izmailov, 2020).
