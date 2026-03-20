# SUPERVISOR_CONTEXT_NOTES.md
# Supervisor and Examiner Context — For Meeting Preparation
# IMPORTANT: Academic profiles based on publicly available research records.
# Repo-specific context [INFERRED FROM REPO].
# Last verified: 2026-03-14

---

## Disclaimer

This document is for your personal meeting preparation only.
Supervisor profiles are based on publicly available academic records (papers, lab pages).
Their likely questions are inferences, not guarantees.

---

## SUPERVISOR: Dominik Fuchsgruber
**Role:** PhD candidate at TUM, Chair for AI Methodology (DAML)
**Supervised you on:** GNN + UQ methodology

### Known Research (relevant to your thesis)

| Paper | Venue | Relevance to Your Thesis |
|-------|-------|--------------------------|
| graph-ebm (Energy-Based Models on graphs) | NeurIPS 2024 | OOD detection, energy-based UQ; he may ask if EBMs are an alternative to MC Dropout |
| magnetic_edge_gnn (Magnetic edge GNN) | ICLR 2025 | Edge-level GNNs — directly related to your road network edge prediction problem |

### What Dominik Cares About (inferred from research focus)

1. **Rigorous UQ methodology** — He will check if you understand the assumptions behind MC Dropout as a Bayesian approximation. Know Gal & Ghahramani 2016.

2. **OOD awareness** — His NeurIPS paper is about detecting when inputs are out-of-distribution. He will ask: "What happens when you apply this to a new city or a policy you haven't seen?" 

3. **Calibration vs. uncertainty ranking** — He knows the difference between a well-calibrated model and one that just ranks uncertainty. Be precise: MC Dropout sigma is NOT calibrated; conformal IS.

4. **Spearman ρ interpretation** — He will ask: is ρ=0.48 good? Be ready to say: "It means 23% shared variance between uncertainty and error (ρ²=0.232). It's meaningful but not strong — and for practical purposes, selective prediction works well: 39.9% MAE reduction by filtering top 50% uncertain."

5. **Ensemble diversity** — He will probe why ensemble (ρ=0.10) is worse than MC Dropout (ρ=0.48). See MEETING_GUIDED_NOTES_HINGLISH.md Q10.

### Questions Dominik Is Likely to Ask

- "Why not energy-based uncertainty instead of MC Dropout?"
  → "EBMs are a valid alternative — they can model uncertainty through energy scoring. For this thesis, MC Dropout was chosen as a computationally tractable approximation that is well-established in the literature. Testing EBMs is a concrete direction for future work."

- "Your model uses edge-level prediction (road segments). How does your approach compare to edge-GNN methods like magnetic edge GNNs?"
  → "I used the LineGraph transform to convert edge prediction to node prediction, which lets me use standard node-level layers. A direct edge-GNN approach (like magnetic GNN) could potentially avoid this transform and operate natively on edges. This is an interesting architectural alternative I did not explore."

- "What is the posterior over which MC Dropout is performing variational inference?"
  → "MC Dropout approximates a posterior over network weights using a Bernoulli variational family. Each dropped sub-network is a sample from this approximate posterior. The prior is implicit — it is not explicitly specified. This is a known limitation of the method."

---

## ADVISOR: Elena Natterer
**Role:** TUM Chair of Traffic Engineering
**Role in your thesis:** Created the PointNetTransfGAT architecture; transport domain expert

### What Elena Cares About

1. **Practical usefulness** — Does the prediction help a transport planner? Is the error of 3.96 veh/h acceptable for policy decisions? She will think in terms of use case, not just R².

2. **The architecture** — She designed PointNetTransfGAT. Be careful to credit her when describing it (e.g., "building on the architecture proposed by Elena Natterer...").

3. **Feature choices** — She may ask about the HIGHWAY feature that is in the code but not used (use_all_features=False). Know why it was excluded.

4. **Policy interpretation** — She understands what CAPACITY_REDUCTION means for different policy scenarios (road closures, lane reductions, etc.). Be ready to explain your policy context clearly.

### Questions Elena Is Likely to Ask

- "How would a planner actually use this system?"
  → "A planner specifies a policy scenario: e.g., close a specific lane on Boulevard Haussmann. The scenario is encoded as a CAPACITY_REDUCTION value. The model predicts the resulting change in car volume on every road in Paris in seconds. The conformal intervals tell the planner: 'I'm 95% confident this road will see between X and Y change.' This supports rapid scenario exploration before running expensive MATSim simulations."

- "Why did you not use the HIGHWAY feature?"
  → "The code has use_all_features=False, which selects the 5 verified features. The HIGHWAY feature is categorical and would require one-hot encoding. The decision to exclude it may have been made to keep the feature set simple. I note this as a potential improvement — including road type could help the model distinguish motorways from residential streets."

- "The weighted loss approach (T3/T4) hurt performance — why?"
  → "Weighted loss amplifies the contribution of high-error examples during training. In this case, the high-error examples may be genuinely hard cases (unusual road types, extreme policies) that the model cannot fit well. Forcing the model to upweight these may have caused it to overfit to noise rather than learn general patterns."

---

## EXAMINER: Prof. Dr. Stephan Guennemann
**Role:** Full Professor, leads TUM DAML group
**Role in your thesis:** Thesis examiner

### Known Research (relevant to your thesis)

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Graph Posterior Networks (GPN) | NeurIPS 2021 | Bayesian uncertainty on graphs — direct competitor to MC Dropout for UQ on GNNs |
| BGNN, APPNP variants | Various | Graph propagation, semi-supervised node classification |

### What Prof. Guennemann Cares About

1. **Bayesian principled approaches** — He created Graph Posterior Networks, which do UQ properly on graphs. He may ask why you didn't use GPN instead of MC Dropout.
   → "GPN is designed for node classification, not regression. Adapting it to continuous regression on 31,635-node graphs with 1,000 training graphs is non-trivial. MC Dropout is the established approach for regression UQ and was the practical choice."

2. **Theoretical soundness** — He will check whether you understand the theoretical guarantees (or lack thereof) of your methods. Conformal prediction has a formal coverage guarantee; MC Dropout does not.

3. **Generalization** — Will the surrogate generalize to a new city? New policy types? He will ask about the boundaries of your claims.

4. **Comparison to baselines** — He may ask: did you compare against a simple baseline (e.g., linear regression, MLP without graph structure)?
   → [If no baseline was run]: "A direct MLP baseline was not included in this thesis, which is a limitation. The ablation across architectures (T2–T8) provides some evidence of what matters, but a non-GNN baseline would have strengthened the argument."

### Questions Prof. Guennemann Is Likely to Ask

- "Why not use Graph Posterior Networks for UQ?"
  → Answer as above — it's classification-focused; MC Dropout is practical for regression.

- "What are the coverage guarantees of your MC Dropout intervals?"
  → "None — the MC Dropout sigma is not calibrated (k95=11.65 vs 1.96). Only the conformal prediction intervals have a formal coverage guarantee, by construction."

- "How does your surrogate handle extrapolation?"
  → "It does not explicitly — GNNs in general interpolate between training distribution scenarios. A policy scenario that is qualitatively different from all training scenarios will likely give high MC Dropout variance (the model is uncertain) but may still give wrong predictions. Explicit OOD detection was not implemented."

---

## Meeting Dynamics — Practical Tips

- **Dominik** will likely lead the technical questions. Prepare for depth on UQ methodology.
- **Elena** will likely focus on architecture justification and practical transport utility.
- **Prof. Guennemann** will ask big-picture theoretical questions during the defense.
- If you don't know something, say: "That's a great point — I haven't fully explored this but I think X would be the right direction."
- Do NOT guess numbers — always say the exact verified number or say you need to check.

---

## Key Papers to Know (for defense)

| Paper | Why You Need It |
|-------|----------------|
| Gal & Ghahramani 2016 (Dropout as Bayesian approx.) | Your MC Dropout theoretical basis |
| Angelopoulos & Bates 2023 (Conformal Prediction tutorial) | Your conformal theoretical basis |
| Lim et al. / PyTorch Geometric LineGraph | Your graph transform basis |
| Wang et al. PointNet/PointNet++ | PointNet layer origin |
| Shi et al. (GraphTransformer) | TransformerConv basis |
| Velickovic et al. (GAT) | GATConv basis |
| Graph Posterior Networks (Stadler et al. NeurIPS 2021) | Prof. Guennemann's paper — know it |
