# GRAPH_EXPLANATION_GUIDE.md
# How to Explain Your Graph Neural Network — From First Principles
# For thesis writing, defense, and meeting explanations
# Last verified: 2026-03-14

---

## Level 0: The One-Sentence Explanation

"We model the Paris road network as a graph, where roads are nodes and intersections
are edges, and use a GNN to predict how car volume on each road changes under a
given transport policy."

---

## Level 1: For Non-Technical Audience (e.g., transport planner)

Imagine Paris as a map of roads. Each road segment is a box. Each box knows:
- How busy it normally is (baseline volume)
- How wide it is (capacity)
- Whether the policy has closed or narrowed it (capacity reduction)
- How fast cars normally go (free speed)
- How long the road is (length)

A graph neural network is a model where each box talks to its neighbors — the roads
that connect to the same intersection. By sharing information between neighboring
roads, the model predicts: after this policy change, how much will car volume on
each road increase or decrease?

---

## Level 2: For Technical Audience (e.g., supervisor meeting)

### The Original Problem

Road traffic simulation produces values on **edges** (road segments).
We want to predict `edge_car_volume_difference` per road segment.

### Dataset Scale (Verified)

> **All results in this thesis use 1,000 of 10,000 available MATSim scenarios (10% subset).**
> The full dataset (10,000 scenarios, Elena Natterer / original paper) was not accessible
> at the same scale. When citing results, always note this 10% subset context.
>
> Verified from: `feature_analysis_report.txt` line 15 (Elena=10,000, Model 8=1,000);
> `point_net_transf_gat.py` line 14 ("paper conducted using 10,000 simulations").

| Property | This thesis | Original paper |
|---|---|---|
| Total scenarios | 1,000 | 10,000 |
| Training scenarios | ~800 (80%) | ~8,000 (80%) |
| Test scenarios (T7/T8) | 100 (10%) | 1,000 (10%) |
| Test scenarios (T1–T6) | 50 (5%) | 500 (5%) |
| Nodes per graph | 31,635 | 31,635 |
| Total test nodes (T8) | 3,163,500 | 31,635,000 |

### Why a GNN?

A feed-forward network (MLP) treats each road independently. But roads are spatially
correlated: if a main road is closed, traffic redistributes to neighboring streets.
A GNN explicitly models this by passing messages between adjacent nodes.

### The Line Graph Transform

PyTorch Geometric predicts **node-level** values. Road volume is an **edge-level** value.
Solution: apply the **line graph transform** (LineGraph() in PyG).

```
ORIGINAL GRAPH:
  Nodes = road intersections
  Edges = road segments

AFTER LINE GRAPH TRANSFORM:
  Nodes = road segments  (each original edge becomes a node)
  Edges = shared intersections  (two roads are connected if they share a junction)
```

This converts the edge-prediction problem into a node-prediction problem.

**Result:** 31,635 nodes per graph (one per road segment in Paris)

### The Architecture (PointNetTransfGAT)

Three types of GNN layers are combined:

**1. PointNetConv — Spatial geometric encoding**
- PointNet was originally designed for 3D point cloud data
- Here, each road segment has a START coordinate and END coordinate (lat/lon)
- `pos` shape: `[N, 3, 2]` — 3 points per node: START (`pos[:,0,:]`), END (`pos[:,1,:]`), MIDPOINT (`pos[:,2,:]`)
- Only START and END are used in the architecture; MIDPOINT is stored in data but not passed to any layer
- PointNetConv layer 1 uses START position `pos[:,0,:]`
- PointNetConv layer 2 uses END position `pos[:,1,:]`
- This encodes the spatial geometry of each road into the node representation

**2. TransformerConv — Attention-based message passing**
- Inspired by Transformer attention (Vaswani et al. 2017)
- Each node attends to its neighbors with a learned attention weight
- Layer 1: 128 → 256 hidden dimensions, 4 attention heads
- Layer 2: 256 → 512 hidden dimensions, 4 attention heads
- Allows the model to learn which neighboring roads are most relevant
- Dropout applied here (for MC Dropout UQ)

**3. GATConv — Graph Attention Network final layers**
- Layer 1: 512 → 64
- Layer 2: 64 → 1  (final output: predicted Δvol_car)
- No dropout in GATConv layers

### Information Flow (one forward pass)

```
Road segment i
  └─ Features: [VOL_BASE, CAP_BASE, CAP_REDUC, FREESPEED, LENGTH]
  └─ Position: pos shape [N,3,2] — [START(x,y), END(x,y), MIDPOINT(x,y)]
               Only START and END are used in the model

Step 1: PointNetConv(START) + PointNetConv(END)
        → 128-dim node embedding (spatial geometry encoded)

Step 2: TransformerConv (128→256, 4 heads)
        → Each road aggregates info from neighboring roads via attention
        → 256-dim embedding

Step 3: TransformerConv (256→512, 4 heads)
        → Deeper aggregation
        → 512-dim embedding

Step 4: GATConv (512→64)
        → Final local aggregation

Step 5: GATConv (64→1)
        → Scalar output: predicted Δvol_car for road i (veh/h)
```

---

## Level 3: Explaining Specific Design Choices

### Why PointNetConv at the start?

Roads are line segments with a start and end point. Their geometric properties
(direction, length, position relative to city center) matter for traffic. PointNetConv
encodes this spatial information before the message-passing layers see the graph structure.

This is similar to how PointNet processes 3D objects: it encodes local geometry first,
then reasons about structure.

### Why TransformerConv in the middle?

Uniform message passing (like GCN: average all neighbors) treats all neighboring roads
equally. But not all neighbors are equally important — a major boulevard matters more
than a small side street. TransformerConv uses learned attention weights, so the model
can learn which neighbors to attend to.

### Why GATConv at the end?

GATConv is a simpler attention mechanism than TransformerConv. After the heavy
representation learning in the Transformer layers, GAT provides a final lightweight
aggregation step before the scalar output.

### Why no dropout in the final GATConv layers?

MC Dropout requires test-time dropout. If dropout were in the final layer (64→1),
the output itself would be stochastically zeroed, producing trivially high variance.
The meaningful uncertainty comes from the intermediate representations, not the
final linear projection.

---

## Level 4: Common Questions and Answers

**Q: How many graph layers deep is your model?**
A: "The PointNetConv layers don't perform full graph message passing in the traditional
sense — they're local geometric aggregators. The true graph message-passing depth is
2 TransformerConv layers + 2 GATConv layers = 4 message-passing steps. Each step
aggregates 1-hop neighborhood information, so the model has an effective receptive
field of 4 hops."

**Q: How large is one graph?**
A: "31,635 nodes, each with 5 features. Edges depend on the network topology but
are on the order of ~60,000–80,000 edges (each node has ~2 neighbors on average in
a road-like line graph). One training graph is approximately 5–10 MB in memory.
Note: all results in this thesis are based on 1,000 of 10,000 available scenarios
(10% subset)."

**Q: What is the batch size and how does batching work in PyG?**
A: "The best model (T8) uses batch_size=8. In PyTorch Geometric, graph batching
combines multiple graphs into one large disconnected graph. So a batch of 8 graphs
= one graph with 8×31,635 = 253,080 nodes. Mini-batch gradient descent operates
on this combined graph."

**Q: How many parameters does the model have?**
A: "Not directly verified from a saved file. The TransformerConv 128→256 with 4 heads
and the GATConv layers suggest roughly O(100K–500K) parameters. This would need to
be measured directly from the model checkpoint."

**Q: Why not a deeper GNN?**
A: "Deeper GNNs suffer from over-smoothing — after many hops, all node representations
converge to similar values (the graph's dominant eigenvector). With 4 message-passing
steps and attention mechanisms to preserve local information, this architecture avoids
over-smoothing while still capturing medium-range dependencies."

---

## Level 5: Whiteboard Sketches for Defense

### Sketch 1: LineGraph Transform
```
Original network:              After LineGraph():
  A─────B─────C                 [AB]─────[BC]
  |     |     |        →             \   /
  D─────E─────F                    [BE]─────[EF]
                                         ...
  (intersections = nodes)    (roads = nodes, junctions = edges)
```

### Sketch 2: Message Passing
```
Road X at time step t:
  h_X^(t) = UPDATE(h_X^(t-1), AGGREGATE({h_Y^(t-1) : Y neighbor of X}))

In TransformerConv:
  AGGREGATE uses attention: α_XY = softmax(W_q h_X · W_k h_Y)
  h_X^(t) = ReLU(W_v Σ_Y α_XY h_Y)
```

### Sketch 3: MC Dropout UQ
```
T=30 forward passes with dropout enabled:
  ŷ_1, ŷ_2, ..., ŷ_30   (different sub-networks each time)
  
  Prediction: mean(ŷ_1...ŷ_30)
  Uncertainty: var(ŷ_1...ŷ_30)   = σ²
  
  High σ = model disagrees with itself = uncertain
  Low σ  = model consistent = more confident
```

---

## Safe Thesis Sentences About the Architecture

**SAFE:** "The PointNetTransfGAT architecture combines spatial geometric encoding (PointNetConv), attention-based message passing (TransformerConv), and a final graph attention readout (GATConv) to predict node-level traffic volume changes."

**SAFE:** "Dropout is applied in the PointNet and TransformerConv layers but not in the final GATConv layers, allowing MC Dropout uncertainty estimation from the intermediate representations."

**SAFE:** "Results shown are based on 1,000 of 10,000 available MATSim scenarios (10% subset), unless stated otherwise."

**UNSAFE:** "The architecture achieves state-of-the-art performance." — No comparison to other architectures was done.

**UNSAFE:** "Our architecture is novel." — The architecture was designed by Elena Natterer; describe it accurately without overclaiming novelty.

**UNSAFE:** "Results are comparable to the full 10,000-scenario dataset." — The 10% subset may lead to different generalization; always note the subset size.
