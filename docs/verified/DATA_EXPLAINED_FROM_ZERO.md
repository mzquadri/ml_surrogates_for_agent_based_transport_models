# DATA EXPLAINED FROM ZERO
> Everything you need to understand the data in this thesis, from scratch.
> No prior knowledge assumed. All facts verified from repository code and JSON files.

---

## 1. THE REAL-WORLD PROBLEM

A city planner asks: "If I close one lane on a major road in Paris, what happens to traffic across the entire city?"

This is not a simple question. Reducing capacity on one road:
- Forces some drivers onto alternative routes
- Those routes get more congested
- Drivers on those routes also reroute
- The effect ripples across thousands of roads

To answer this properly requires a **traffic simulation**.

---

## 2. MATSIM: THE SIMULATION ENGINE

MATSim (Multi-Agent Transport Simulation) simulates every individual driver in a city:

- Each agent has a daily plan: home → work → shopping → home
- Each agent chooses routes based on current congestion
- Agents interact on the road network
- The simulation runs in repeated iterations until patterns stabilize

**One MATSim run = one simulated day of traffic in Paris**

Output: for every road segment, the number of cars that used it per hour (car volume, veh/h).

MATSim runs are the **ground truth** in this thesis. They are expensive (hours per run) but accurate.

---

## 3. WHAT IS A POLICY SCENARIO?

A **policy** = a change applied to the road network before simulation.

In this thesis, policies are implemented as **capacity reductions** on specific road segments:
- A road normally handles 1000 veh/h
- After policy: capacity reduced to 500 veh/h (e.g., lane closure, bike lane addition)

**Base case**: Paris with NO policy changes (normal operation, capacity everywhere unchanged)

**Policy scenario**: Paris with capacity reductions applied to some roads

Each of the 10,000 scenarios has a different set of roads affected and different reduction amounts.

---

## 4. THE DATASET: 10,000 SCENARIOS, 1,000 USED

The full dataset:
- 10,000 different random policy scenarios, each fully simulated with MATSim
- Also: one base case simulation (Paris with no policy)

**This thesis uses 1,000 of the 10,000 scenarios.**

Why only 1,000? Each graph has 31,635 nodes and is expensive to store/process. 1,000 graphs is a practical subset that still provides meaningful training data.

---

## 5. THE TARGET VARIABLE: DELTA VOLUME

For each road segment in each scenario, we compute:

```
delta_volume[road_i, scenario_j] = volume_car[road_i, scenario_j] - volume_base_case[road_i]
```

Units: **veh/h** (vehicles per hour)

Examples:
- Road whose capacity was reduced: delta_volume = -600 veh/h (cars forced off)
- Parallel road nearby: delta_volume = +200 veh/h (cars rerouted here)
- Road far from intervention: delta_volume ≈ 0 (not affected)

**This delta volume is what the GNN predicts for every road segment.**

Verified from: `help_functions.py` lines 121-123:
```python
edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
```

---

## 6. THE PARIS ROAD NETWORK AS A GRAPH

A road network is naturally a graph. But the representation used here is non-standard.

### Standard representation:
- Nodes = intersections (junctions)
- Edges = road segments between junctions

### This thesis uses a LINE GRAPH transformation:

After `LineGraph()` transform (from PyTorch Geometric):
- **Nodes = road segments** (31,635)
- **Edges = pairs of road segments that share an intersection**

Two road segments are connected if they meet at the same junction.

### Why use line graph?

The prediction target (delta volume) is defined per road segment, not per intersection.
In a GNN, it is cleaner to predict node outputs than edge outputs.
The line graph makes road segments into nodes, enabling natural node-level prediction.

**Verified from:** `process_simulations_for_gnn.py` — `LineGraph()` transform applied explicitly.

---

## 7. ONE DATA SAMPLE = ONE GRAPH

Each of the 1,000 scenarios becomes one PyTorch Geometric Data object:

```
data.x          shape: [31635, 5]    — 5 features per road segment
data.y          shape: [31635, 1]    — delta volume (TARGET) per road segment
data.edge_index shape: [2, E]        — connectivity (which segments share an intersection)
data.pos        shape: [31635, 3, 2] — coordinates: start, end, midpoint of each segment
```

The model takes one graph, processes all 31,635 nodes simultaneously, and outputs 31,635 predictions.

---

## 8. DATASET SPLITS

**Group A (Trials T1-T6): 80/15/5 split**
- Train: 800 graphs (800 × 31,635 = 25,308,000 training nodes)
- Validation: 150 graphs
- Test: 50 graphs → 1,581,750 test predictions

**Group B (Trials T7-T8): 80/10/10 split**
- Train: 800 graphs
- Validation: 100 graphs
- Test: 100 graphs → 3,163,500 test predictions

**Important:** Group A and Group B have different test sets. R2 scores cannot be directly compared between T1-T6 and T7-T8. The improvement from T6 (R2=0.522) to T7 (R2=0.547) may partly reflect the different test distribution.

---

## 9. WHAT THE METRICS MEAN

### R2 = 0.5957 (T8 best model)
- R2 = 1.0: perfect prediction
- R2 = 0.0: model predicts the mean delta volume everywhere (useless)
- R2 < 0: worse than the mean
- **R2 = 0.60 means the model explains 60% of variance in traffic volume changes**

In context: predicting city-scale traffic rerouting from a 5-feature input in <1 second (vs hours of MATSim simulation) while explaining 60% of the variance is the core engineering contribution.

### MAE = 3.96 veh/h (T8)
- Average absolute prediction error across all road segments
- Most Paris streets carry 100-2000+ veh/h
- ~4 veh/h average error is very small for major roads; more significant for small side streets

### RMSE = 7.12 veh/h (T8)
- Root mean squared error — penalizes large errors more than MAE
- Higher than MAE because some segments have large errors (high-traffic near the policy intervention)

---

## 10. TIME-OF-DAY NOTE

**There is no time-of-day data.** The MATSim output used here is a daily aggregate (or peak-hour aggregate). The model does not have morning/evening/night as separate inputs or targets. All delta volumes are for a single time period per scenario.

---

## 11. KEY VERIFIED FACTS

| Fact | Value | Source |
|---|---|---|
| Road segments (nodes after line graph) | 31,635 | process_simulations_for_gnn.py |
| Features per road segment | 5 | point_net_transf_gat.py |
| Target variable | Delta car volume (veh/h) | help_functions.py |
| Total MATSim scenarios available | 10,000 | repo docs |
| Scenarios used | 1,000 | verified from splits |
| Best model R2 (T8) | 0.5957 | test_evaluation_complete.json |
| Best model MAE (T8) | 3.96 veh/h | test_evaluation_complete.json |
| Graph transform | LineGraph() from PyTorch Geometric | process_simulations_for_gnn.py |
| T1-T6 test set size | 50 graphs / 1,581,750 nodes | verified from 80/15/5 split |
| T7-T8 test set size | 100 graphs / 3,163,500 nodes | verified from 80/10/10 split |
