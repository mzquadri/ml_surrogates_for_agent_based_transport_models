# POLICY EXPLAINED SIMPLY
> What "transport policy" means in this thesis, explained simply.
> No traffic engineering background needed.

---

## WHAT IS A TRANSPORT POLICY?

A transport policy is a decision that changes how a road network operates.

In this thesis, policies are simple: **reduce the capacity of one or more road segments**.

Examples of real-world actions that translate to capacity reductions:
- Adding a bike lane (removes one car lane)
- Implementing a bus lane (removes one car lane)
- Road works / construction (blocks one lane temporarily)
- Speed table or traffic calming (reduces throughput)
- Closing a road entirely to cars

---

## HOW POLICIES ARE REPRESENTED IN THE DATA

Each MATSim scenario has a **capacity reduction vector**: for every road segment, by how much is its capacity reduced?

```
Road segment 1:  capacity reduced by   0 veh/h  (unaffected)
Road segment 2:  capacity reduced by   0 veh/h  (unaffected)
Road segment 3:  capacity reduced by -500 veh/h  (one lane closed)
...
Road segment 14200: capacity reduced by -1200 veh/h (major road blocked)
...
Road segment 31635: capacity reduced by   0 veh/h  (unaffected)
```

This vector IS Feature 2 (CAPACITY_REDUCTION) in the GNN input — the key signal.

Most roads in any given scenario have CAPACITY_REDUCTION = 0. Only a few roads are directly affected by the policy. But those affected roads create ripple effects across the whole network.

---

## WHAT CHANGES AND WHAT DOESN'T

**Changes between scenarios:**
- Which roads are affected (CAPACITY_REDUCTION)
- The resulting traffic volumes (output from MATSim)
- The target delta volumes

**Stays the same across all scenarios:**
- The Paris road network geometry
- Road capacities (base case)
- Baseline traffic volumes
- Free-flow speeds
- Road lengths

This is why Feature 2 (CAPACITY_REDUCTION) is the GNN's primary signal about "what policy is this?"

---

## THE CAUSAL CHAIN

```
Policy decision:
  "Reduce capacity of Road X from 1200 to 400 veh/h"
        |
        | (MATSim simulation / GNN prediction)
        v
Direct effect:
  Road X carries fewer cars (-800 veh/h)
        |
        v
Secondary effects (rerouting):
  Roads parallel to X carry more cars (+200, +350, +100 veh/h)
        |
        v
Tertiary effects (congestion cascades):
  Roads feeding into those parallel routes also change
  (+50, -30, +20 veh/h, etc.)
        |
        v
Network-wide effect:
  Most roads: near zero change
  Roads near the policy: moderate changes
  The specific affected road: large negative change
```

The GNN must capture ALL of these effects simultaneously for all 31,635 roads.

---

## WHY THIS IS HARD

1. **Nonlinearity**: The rerouting cascade is highly nonlinear. A small capacity change can trigger large downstream effects or very small ones depending on the network state.

2. **Sparsity**: In most scenarios, only 1-3 roads are directly affected. The GNN must infer effects on all 31,635 roads from a signal affecting only a tiny fraction.

3. **Spatial extent**: Effects can propagate far from the intervention. A change in one suburb can affect commute patterns across the entire city.

4. **Varying magnitude**: Some interventions cause large city-wide effects (blocking a key arterial); others cause very localized effects (closing a small side street).

---

## WHAT THE GNN IS ACTUALLY LEARNING

The GNN is learning a function:

```
f(road_features, policy_signal, network_topology) -> delta_volume per road
```

It learns this from 800 training examples (800 different policies and their MATSim outcomes).

It then generalizes to predict the traffic effects of 100 NEW policies it has never seen.

The fact that it achieves R2=0.60 means it has learned meaningful patterns about how capacity reductions propagate through Paris road networks — without knowing the underlying traffic assignment equations.

---

## WHAT A CITY PLANNER WOULD DO WITH THIS TOOL

1. "I'm considering adding a bike lane on Boulevard Voltaire."
2. Encode this as a capacity reduction on the Boulevard Voltaire road segments.
3. Run GNN: 31,635 predictions in <1 second.
4. Examine the uncertainty map: which roads have low sigma (trustworthy predictions)?
5. For high-sigma roads: run actual MATSim before committing.
6. For low-sigma roads: accept GNN prediction for planning purposes.
7. Use conformal intervals to quantify worst-case traffic impact for safety-critical roads.

---

## KEY NUMBERS FOR POLICY CONTEXT

| Metric | Value | Meaning for planners |
|---|---|---|
| Model R2 | 0.596 | 60% of traffic change variance explained — good for planning screening |
| MAE | 3.96 veh/h | Average error of ~4 cars/hour — negligible on major roads, acceptable on minor ones |
| 95% conformal interval | ±14.68 veh/h | Worst-case error bound with 95% statistical guarantee |
| Speedup vs MATSim | seconds vs hours | Enables real-time policy screening |
| Uncertainty ranking | rho=0.48 | Reliable identification of roads needing verification |
