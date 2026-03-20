# THESIS STORY FROM ZERO
> The complete narrative arc of this thesis — what was done, why, and what it means.
> Written for explaining to someone unfamiliar with the topic.
> All numbers verified from JSON files.

---

## THE STORY IN ONE PARAGRAPH

Paris has 31,635 road segments. When a city planner changes a road (e.g., adds a bike lane), the whole traffic network rearranges — people reroute, congestion shifts, and effects ripple across the city. Figuring out these effects requires running MATSim, a full agent-based traffic simulation that takes hours per scenario. This thesis trains a Graph Neural Network (GNN) to act as a fast surrogate: given a road network and a policy change, predict the delta traffic volume on every road in under a second. The GNN achieves R2=0.596 (explains 60% of traffic change variance). But a fast prediction is not enough for planning — planners also need to know how much to trust each prediction. This thesis therefore adds Uncertainty Quantification (UQ): MC Dropout provides a per-road uncertainty estimate (Spearman rank correlation rho=0.48 with actual error), and Conformal Prediction provides guaranteed statistical coverage intervals (90.02% coverage at 90% target; 95.01% at 95% target). Together, these tools turn the GNN from a black box into a plannable decision support system.

---

## CHAPTER 1: THE PROBLEM

### Why traffic simulation is hard
- Paris has thousands of roads, millions of daily trips
- When one road changes, traffic reroutes everywhere — a nonlinear, city-wide effect
- Traditional simulation (MATSim): accurate but slow (hours per scenario)
- City planners need to evaluate hundreds of policy options — infeasible with MATSim alone

### The surrogate idea
- Train a neural network on existing MATSim outputs
- Network learns: given a policy change, predict its city-wide effect
- Inference: seconds instead of hours
- Enables rapid policy screening

### Why a GNN?
- Road networks are graphs (roads connected at intersections)
- GNNs naturally handle graph-structured data
- Message passing allows each road segment to "learn from" its neighbors
- Captures the rerouting cascade that makes traffic prediction hard

---

## CHAPTER 2: THE DATA

### What we have
- 10,000 MATSim policy scenarios for Paris (1,000 used in this thesis)
- Each scenario: Paris road network + one set of capacity reductions
- For each scenario: MATSim output with car volumes per road per hour

### The line graph trick
- Standard graphs: nodes=intersections, edges=roads
- We want to predict PER ROAD — easier as node-level output
- Apply LineGraph() transform: roads become nodes (31,635), adjacency = shared junctions
- Now each "node" is a road segment with 5 features and one prediction target

### The target
- delta_volume = volume_in_policy_scenario - volume_in_base_case (veh/h)
- Positive = more traffic (rerouting onto this road)
- Negative = less traffic (capacity reduced or rerouting away)
- Zero = unaffected road

### The 5 features
1. Baseline car volume (how busy normally)
2. Road capacity (maximum flow)
3. Capacity reduction (the policy signal — varies per scenario)
4. Free-flow speed (road type proxy)
5. Segment length

---

## CHAPTER 3: THE MODEL

### Architecture: PointNetTransfGAT
A custom GNN combining three types of layers:
1. **PointNetConv** (×2): spatially-aware feature aggregation using road segment coordinates
2. **TransformerConv** (×2): attention-based message passing (128→512 channels)
3. **GATConv** (×2): graph attention, final output 512→64→1 (the prediction)

### Training: 8 trials
The model was trained 8 times with different hyperparameters:
- T1: excluded (different architecture)
- T2-T4: exploring loss functions and batch sizes
- T5-T6: batch size=8 confirmed as best
- T7-T8: larger test set (80/10/10 split), fine-tuning dropout

**Best model: T8** (batch=8, LR=0.0005, dropout=0.2)
- Test R2 = 0.5957
- Test MAE = 3.96 veh/h
- Test RMSE = 7.12 veh/h

### Key finding: weighted loss hurts
Trials T3 and T4 used weighted MSE loss (higher weight for roads with large volume changes). Both performed worse (R2 ≈ 0.22-0.24) than standard MSE (R2 ≈ 0.51-0.60). Likely because weighted loss focuses too much on a few high-change roads at the expense of overall network accuracy.

---

## CHAPTER 4: THE UNCERTAINTY PROBLEM

A GNN giving predictions without uncertainty is like a weather forecast without a confidence level. The planner needs to know: "Can I trust this prediction? For which roads is the model most uncertain?"

### Two problems:
1. **Ranking**: Can we identify which predictions are likely to be wrong?
2. **Calibration**: Can we give statistically guaranteed prediction intervals?

---

## CHAPTER 5: MC DROPOUT — THE UNCERTAINTY ESTIMATOR

### How it works
- Dropout (randomly deactivating neurons during training) is a regularization technique
- Gal & Ghahramani (2016): if you keep dropout active at test time, you get approximate Bayesian inference
- Each forward pass with different random dropout mask = one approximate posterior sample

### What we measured
- For T8: 30 forward passes per test graph = 30 predictions per road
- Standard deviation across 30 predictions = sigma (uncertainty estimate)
- Spearman rank correlation between sigma and |error| = **rho = 0.4820**
- Interpretation: roads with higher MC Dropout uncertainty tend to have higher actual prediction error

### The calibration problem
- Sigma is informative as a **risk ranking** signal (rho=0.48)
- Sigma is NOT a calibrated standard deviation
- Naive 95% interval: mu ± 1.96*sigma gives only ~55% actual coverage (k95 = 11.65)
- We need something better for guaranteed intervals

---

## CHAPTER 6: CONFORMAL PREDICTION — THE COVERAGE GUARANTEE

### How it works
Split-conformal prediction (Angelopoulos & Bates, 2022):
1. Split test set: 50 calibration graphs, 50 evaluation graphs
2. On calibration set: compute nonconformity scores = |y_true - mu_hat|
3. Find q = (1-alpha) quantile of nonconformity scores
4. For evaluation: interval = [mu_hat - q, mu_hat + q]
5. Guaranteed: at least (1-alpha) fraction of y_true falls in interval

### Results for T8
- 90% target: q = 9.9196 veh/h, actual coverage = 90.02% ✓
- 95% target: q = 14.6766 veh/h, actual coverage = 95.01% ✓

### The sigma-normalized enhancement
Using MC Dropout sigma to create adaptive intervals:
- interval = [mu - q*sigma, mu + q*sigma]
- Low-uncertainty roads: narrow interval (66% narrower on average)
- High-uncertainty roads: wider interval
- Still maintains coverage guarantee

---

## CHAPTER 7: ENSEMBLE EXPERIMENTS

Two additional experiments explored multi-model uncertainty:

**Experiment A:** 5 independently-retrained T8 models
- MC Dropout averaged: rho = 0.1600
- Ensemble variance: rho = 0.1035
- Why low? Models trained on different data subsets → disagreement reflects distribution differences, not prediction uncertainty

**Experiment B:** Multi-model ensemble
- rho = 0.1167
- Similar issue: cross-distribution artifact

**Conclusion:** For this task, single-model MC Dropout (rho=0.48) outperforms multi-model ensemble UQ (rho=0.10-0.16) because the ensemble members are trained on different data, not just different initializations.

---

## CHAPTER 8: WHAT THIS MEANS

### For city planning
- A planner can now get a full Paris traffic forecast in seconds (vs hours)
- The prediction comes with a reliability map: which roads to trust, which to double-check with MATSim
- For roads with low sigma: trust the prediction, proceed with planning
- For roads with high sigma: run actual MATSim before committing

### Contributions
1. First GNN surrogate for Paris-scale traffic policy analysis (R2=0.60 at 31,635-node resolution)
2. MC Dropout uncertainty quantification achieving rho=0.48 (Spearman) with actual errors
3. Conformal prediction with guaranteed coverage (90.02% at 90% target)
4. Adaptive intervals (sigma-normalized) 66% narrower for confident predictions
5. Empirical study of ensemble vs single-model UQ for graph-level traffic surrogates

---

## WHAT SENTENCE IS SAFE TO SAY IN THE THESIS

**Safe:**
> "The best model (T8) achieves R2=0.596 and MAE=3.96 veh/h on 100 unseen policy scenarios."

> "MC Dropout uncertainty estimates achieve Spearman rank correlation rho=0.482 with absolute prediction errors, enabling reliable identification of high-uncertainty road segments."

> "Conformal prediction intervals achieve 90.02% empirical coverage at the 90% nominal level, providing distribution-free statistical guarantees."

> "Sigma-normalized conformal intervals are 65.8% narrower for low-uncertainty predictions compared to fixed-width intervals, while maintaining coverage guarantees."

**Do NOT say:**
> "Temperature scaling reduced ECE from 0.356 to 0.033" — not verified, no source file found.
> "The model achieves R2=0.786" — this is T1 (different architecture), not the comparable best model.
> "We used batch sizes of 32/64" — wrong, T5-T8 all used batch size 8.
