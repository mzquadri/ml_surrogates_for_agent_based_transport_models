# NODE VS GRAPH LEVEL EXPLAINED
> What is a node-level task vs a graph-level task, and why this thesis uses node-level.
> Critical concept for understanding the GNN design choices.

---

## THE TWO MAIN GNN PREDICTION TASKS

In Graph Neural Networks, there are two main types of prediction:

### Graph-Level Prediction
- **Output**: ONE prediction per graph (one number per scenario)
- **Example**: "Is this molecule toxic? (yes/no)" — one answer per molecule
- **Example**: "What is the energy of this crystal?" — one number per crystal
- **How**: Pool all node representations into a single vector, then predict

### Node-Level Prediction
- **Output**: ONE prediction per NODE (per road segment)
- **Example**: "What is the traffic flow on each road in this scenario?" — 31,635 numbers per scenario
- **How**: Each node's representation directly predicts that node's value

---

## WHICH DOES THIS THESIS USE?

**Node-level prediction.**

For each of the 100 test scenarios (graphs), the model outputs 31,635 predictions — one per road segment.

This is because the TARGET is delta volume per road, not a single aggregate number per scenario.

---

## THE ORIGINAL GRAPH VS THE LINE GRAPH

### Original graph of Paris road network:
```
Nodes = intersections (~17,000)
Edges = road segments (~31,635)
```

If we used the original graph directly, predicting delta volume per road would be an **EDGE-LEVEL** prediction task. GNNs don't natively do edge prediction — it requires extra steps.

### After LineGraph() transform:
```
Nodes = road segments (31,635)  <- each original edge becomes a node
Edges = pairs of road segments sharing an intersection
```

Now predicting delta volume per road is a **NODE-LEVEL** prediction task — the natural output of a GNN.

This is why the LineGraph() transform is applied in preprocessing.

---

## WHAT MESSAGE PASSING DOES IN NODE-LEVEL PREDICTION

In a GNN, each node receives "messages" from its neighbors and updates its own representation.

For this thesis:
- Road segment X receives messages from all road segments that share a junction with X
- It updates its representation based on: its own features + neighbors' features + the graph structure
- After multiple rounds of message passing, X's representation encodes not just its own features but information about the nearby network

This is how the GNN captures the rerouting cascade:
- The capacity-reduced road (Feature 2 signal) propagates information to its neighbors
- Neighbors propagate to their neighbors
- After 4-6 GNN layers, distant roads are also informed of the intervention

---

## WHY NOT GRAPH-LEVEL?

One might ask: instead of predicting 31,635 numbers per scenario, why not predict a single aggregate (e.g., total network delay increase)?

**Reason 1:** City planners need spatial detail. "Total delay increases by 10 minutes" is less useful than "Road X gets 200 more veh/h, Road Y gets 100 less veh/h."

**Reason 2:** Node-level prediction is more informative and allows uncertainty analysis per road.

**Reason 3:** The MATSim ground truth provides per-road outputs — node-level is the natural supervision signal.

---

## METRICS AT NODE LEVEL

Because predictions are at the node level, all metrics are computed aggregated over nodes:

```
R2  = r2_score(y_true_all_nodes, y_pred_all_nodes)
MAE = mean_absolute_error(y_true_all_nodes, y_pred_all_nodes)
```

For T8 test evaluation:
- y_true: 3,163,500 values (100 graphs × 31,635 nodes)
- y_pred: 3,163,500 values
- R2 = 0.5957 (computed over all 3.16M values)
- MAE = 3.96 veh/h (average over all 3.16M values)

---

## METRICS AT GRAPH LEVEL (for reference)

In some applications you might also compute per-graph R2:
- R2 computed separately for each of the 100 test graphs
- Some graphs easier to predict than others
- Average per-graph R2 would differ from global R2

This thesis uses **global R2** (pooling all nodes across all test graphs). This is consistent with standard practice for regression GNNs.

---

## UNCERTAINTY IS ALSO NODE-LEVEL

MC Dropout sigma is computed per node:
```
sigma[i] = std of 30 MC forward passes for node i
```

This gives a different uncertainty value for every road segment in every scenario.

The Spearman correlation (rho=0.4820) measures whether sigma ranks nodes correctly by their actual error:
- High sigma nodes → tend to have high |error|
- Low sigma nodes → tend to have low |error|

This is a node-level ranking evaluation — consistent with the prediction being node-level.

---

## CONFORMAL PREDICTION IS NODE-LEVEL TOO

The nonconformity score is computed per node:
```
score[i] = |y_true[i] - mu_pred[i]|
```

The calibration step uses all 50 calibration graphs × 31,635 nodes = 1,581,750 calibration scores.

The resulting interval [mu - q, mu + q] is then applied at the node level in evaluation.

So the 90.02% coverage means: across all 50×31,635 = 1,581,750 evaluation nodes, 90.02% have y_true inside the predicted interval.

---

## SUMMARY TABLE

| Aspect | This Thesis | Why |
|---|---|---|
| Prediction granularity | Node-level (per road) | Need spatial detail per road segment |
| Output size per scenario | 31,635 numbers | One per road segment |
| Graph transform | LineGraph() | Makes roads into nodes for natural node-level output |
| Metrics computed | Over all nodes × all test graphs | 3.16M values for T7/T8 |
| UQ (sigma) | Per node | Different uncertainty per road |
| Conformal coverage | Marginal over all nodes | 90%/95% of nodes covered |
