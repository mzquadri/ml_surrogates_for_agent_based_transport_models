# FEATURES EXPLAINED FROM ZERO
> Every input feature to the GNN, explained from scratch.
> All facts verified from repository code (point_net_transf_gat.py, process_simulations_for_gnn.py).

---

## OVERVIEW

The GNN takes **5 features per road segment** as input (`data.x`, shape [31635, 5]).

These features describe the road segment itself — not the policy outcome.
The policy effect is encoded via CAPACITY_REDUCTION (Feature 2).

**Verified from:** `EdgeFeatures` enum in `process_simulations_for_gnn.py`

---

## THE 5 FEATURES

### Feature 0: VOL_BASE_CASE — Baseline Car Volume

**What it is:** The number of cars that use this road segment per hour in the baseline (no-policy) scenario.

**Units:** veh/h (vehicles per hour)

**Why it matters:** Roads with high baseline volume are likely to experience larger absolute changes when a policy is applied nearby. The model uses this as a reference point for how busy a road normally is.

**Correlation with prediction error:** +0.3316 (verified from T8 feature_analysis_report.txt)
High-baseline-volume roads are harder to predict accurately — makes sense since they carry more traffic that could reroute in complex ways.

**Example values:** ~50 veh/h (quiet side street) to ~3000+ veh/h (major boulevard)

---

### Feature 1: CAPACITY_BASE_CASE — Road Capacity

**What it is:** The maximum traffic flow this road can sustain (in the base case, before any policy change).

**Units:** veh/h (vehicles per hour)

**Why it matters:** A road carrying 900 veh/h on a 1000 veh/h capacity is near saturation. A road carrying 900 veh/h on a 3000 veh/h capacity has plenty of spare capacity. The model needs to understand congestion levels.

**Correlation with prediction error:** +0.2615
Higher-capacity roads (typically major roads) also tend to have larger errors.

**Example values:** ~200 veh/h (residential street) to ~4000+ veh/h (urban motorway)

**Note:** VOL_BASE_CASE / CAPACITY_BASE_CASE gives the volume-to-capacity ratio (utilization), a key traffic engineering metric. Though this ratio isn't a separate feature, the model can implicitly compute it from features 0 and 1.

---

### Feature 2: CAPACITY_REDUCTION — Policy Intervention

**What it is:** The change in capacity applied by the policy scenario. This is what makes each scenario different.

**Units:** veh/h (negative = reduction)

**Why it matters:** This is the primary signal about WHERE the policy intervention is. For roads not affected by the policy, CAPACITY_REDUCTION = 0. For roads where a lane is closed or a bike lane added, CAPACITY_REDUCTION < 0.

**Correlation with prediction error:** -0.2286
Roads with large capacity reductions have more predictable effects (direct causal relationship). Interestingly, the sign is negative — large reductions are associated with more negative errors or the model systematically overestimates these.

**Example values:** 0 (most roads — unaffected), -500 (moderate lane closure), -2000 (major road blockage)

**This is the KEY differentiating feature between scenarios.** All other features are the same for a given road segment across all 1,000 scenarios. Only CAPACITY_REDUCTION varies between scenarios for a given road.

---

### Feature 3: FREESPEED — Free-Flow Speed

**What it is:** The speed of traffic on this road when there is no congestion (free-flow conditions).

**Units:** m/s (meters per second) — note: NOT km/h in the raw data

**Why it matters:** Encodes road type. Highways have high freespeed (~30 m/s ≈ 108 km/h). Residential streets have low freespeed (~5-8 m/s ≈ 18-30 km/h). This implicitly encodes whether a road is a major artery or a small side street.

**Correlation with prediction error:** +0.2110
Fast roads (motorways/boulevards) are harder to predict — consistent with VOL_BASE_CASE and CAPACITY_BASE_CASE correlations.

---

### Feature 4: LENGTH — Road Segment Length

**What it is:** The physical length of the road segment.

**Units:** meters

**Why it matters:** Longer road segments carry more total traffic and take longer to traverse. Also partially encodes road type (motorway segments tend to be longer than urban street segments).

**Correlation with prediction error:** -0.0695
Weakest predictor of error. Length has minimal direct relationship with prediction difficulty.

**Example values:** ~20m (short urban intersection link) to ~2000m+ (long motorway segment)

---

## WHAT IS NOT A FEATURE (but exists in the code)

- **HIGHWAY type** (motorway, primary, secondary, etc.): This categorical feature exists as `EdgeFeatures.HIGHWAY` in the preprocessing code but is **NOT used** in the 5-feature model. The `use_all_features=False` setting excludes it.
- **Time of day**: Not available. No morning/evening/night split.
- **Travel time**: Not a feature (though freespeed and length together imply free-flow travel time).

---

## FEATURE SPACE SUMMARY

```
Feature index   Name                  Type      Varies per scenario?
0               VOL_BASE_CASE         Float     No  (same across all 1000 scenarios)
1               CAPACITY_BASE_CASE    Float     No  (same across all 1000 scenarios)
2               CAPACITY_REDUCTION    Float     YES (the policy signal — changes per scenario)
3               FREESPEED             Float     No  (same across all 1000 scenarios)
4               LENGTH                Float     No  (same across all 1000 scenarios)
```

**Key insight:** 4 out of 5 features are the same for a given road segment regardless of which scenario is being evaluated. Only CAPACITY_REDUCTION changes. The model must use this single signal + the graph structure (message passing from neighbors) to predict traffic rerouting across 31,635 roads.

---

## POSITIONAL FEATURES (not in data.x but used by model)

In addition to the 5 node features, the model uses **coordinate positions**:

`data.pos` shape: [31635, 3, 2]
- `pos[:, 0, :]` = start coordinates of each road segment (used in PointNetConv Layer 1)
- `pos[:, 1, :]` = end coordinates of each road segment (used in PointNetConv Layer 2)
- `pos[:, 2, :]` = midpoint coordinates (computed, not directly used as separate input)

These allow the PointNetConv layers to perform spatially-aware feature aggregation.

---

## THESIS-SAFE SENTENCES

**Safe to say:**
> "The model takes five road-segment features as input: baseline car volume, road capacity, policy-induced capacity reduction, free-flow speed, and segment length."

> "Feature correlation analysis shows that baseline car volume (r=0.33) and road capacity (r=0.26) are most strongly correlated with prediction error, suggesting that high-traffic roads are harder to predict."

> "Capacity reduction is the only feature that varies between scenarios; all other features are fixed properties of the road network."

**Do NOT say:**
> "Highway type is used as a feature" — it is in the code but not in the 5-feature model.
> "Time-of-day features are included" — they are not.
