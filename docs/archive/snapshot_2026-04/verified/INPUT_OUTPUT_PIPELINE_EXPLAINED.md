# INPUT-OUTPUT PIPELINE EXPLAINED
> The complete journey from raw MATSim simulation output to GNN prediction and uncertainty estimate.
> All steps verified from repository code.

---

## BIRD'S EYE VIEW

```
MATSim simulation output (XML/CSV)
        |
        v
[PREPROCESSING] process_simulations_for_gnn.py
        |
        v
PyTorch Geometric Data objects (.pt files)
        |
        v
[TRAINING] run_models.py
        |
        v
Trained model checkpoint (.pth)
        |
        v
[INFERENCE] predict on test graphs
        |
        v
Predictions: delta_volume per road segment
        |
        v
[UQ PIPELINE] MC Dropout + Conformal Prediction
        |
        v
Uncertainty estimates: sigma per node + calibrated intervals
```

---

## STEP 1: RAW DATA (MATSim Output)

**Input:** MATSim simulation output files
- One file per scenario (1,000 scenarios used + base case)
- Format: link statistics CSV/XML
- Contains: for each road link (segment), the traffic volume (veh/h) during the simulation

**Also needed:** GeoDataFrame of the Paris road network with geometry (start/end coordinates, length, capacity, freespeed for each road segment)

---

## STEP 2: PREPROCESSING (`process_simulations_for_gnn.py`)

For each scenario:

### 2a. Load road network GeoDataFrame
- 31,635 road segments
- Each has: geometry (start/end coords), capacity, freespeed, length, highway type

### 2b. Merge simulation output with road network
- Join MATSim volumes to the road network GeoDataFrame on link ID
- Result: each road segment now has its car volume for this scenario

### 2c. Compute target variable
```python
edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
```
This is `data.y` — the delta volume for every road segment in this scenario.

### 2d. Construct features (`data.x`)
```python
features = [VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITY_REDUCTION, FREESPEED, LENGTH]
data.x = torch.tensor(features)  # shape [31635, 5]
```

### 2e. Construct positional features (`data.pos`)
```python
data.pos = coords  # shape [31635, 3, 2] — start/end/midpoint per segment
```

### 2f. Build initial graph (road network topology)
- At this stage: nodes = intersections, edges = road segments
- This is the "natural" graph of Paris road network

### 2g. Apply LineGraph() transform
```python
from torch_geometric.transforms import LineGraph
transform = LineGraph()
data = transform(data)
```

After transform:
- Nodes = road segments (31,635)
- Edges = pairs of adjacent road segments sharing an intersection
- `data.edge_index` encodes this adjacency

### 2h. Save as .pt file
One `.pt` file per scenario, all stored in the processed data directory.

---

## STEP 3: DATA LOADING AND SPLITTING

During training:
```python
dataset = [load(f"graph_{i}.pt") for i in range(1000)]
# T7/T8 split:
train = dataset[:800]   # 800 graphs
val   = dataset[800:900] # 100 graphs
test  = dataset[900:]   # 100 graphs
```

The DataLoader batches multiple graphs together for GPU-efficient training.

---

## STEP 4: MODEL FORWARD PASS (`point_net_transf_gat.py`)

For one graph with 31,635 nodes:

```
Input: data.x [31635, 5], data.edge_index, data.pos [31635, 3, 2], data.batch

Layer 1: PointNetConv(pos=start_coords, x=features)   -> [31635, 128]
         Uses spatial relationship between segment start points

Layer 2: PointNetConv(pos=end_coords, x=features)     -> [31635, 128]
         Uses spatial relationship between segment end points
         (concatenated with Layer 1 output -> [31635, 256])

Layer 3: TransformerConv(256 -> 256, heads=4)          -> [31635, 256]
         + ReLU + [optional dropout]

Layer 4: TransformerConv(256 -> 512, heads=4) -> [31635, 512]
         (Actually: 64 head_channels x 4 heads = 256 channels per step)
         + ReLU + [optional dropout]

Layer 5: GATConv(512 -> 64)                            -> [31635, 64]
         Attention-weighted aggregation from neighbors

Layer 6: GATConv(64 -> 1)   [gat_final]               -> [31635, 1]
         Final prediction — NO dropout here

Output: [31635, 1] — predicted delta volume per road segment
```

**Note on dropout:** Dropout is applied in PointNetConv and TransformerConv layers only. The final GATConv layers have no dropout. This is important for MC Dropout: dropout must remain active at inference time to generate stochastic samples.

---

## STEP 5: LOSS FUNCTION AND TRAINING

```python
loss = MSE(y_pred, y_true)  # standard mean squared error
# For T3/T4: weighted MSE where weights proportional to |y_true|
optimizer = Adam(lr=0.0005)  # T8 learning rate
```

Training runs for ~200 epochs (approximate, verified from checkpoint files).
Best checkpoint selected by validation loss.

---

## STEP 6: TEST EVALUATION

```python
model.eval()
with torch.no_grad():
    predictions = model(test_graph)

# Compute metrics
r2  = r2_score(y_true, y_pred)      # = 0.5957 for T8
mae = mean_absolute_error(y_true, y_pred)  # = 3.96 veh/h for T8
rmse = sqrt(mean_squared_error(...))       # = 7.12 veh/h for T8
```

Results saved to `test_evaluation_complete.json`.

---

## STEP 7: MC DROPOUT UQ

After training, the model is run in **MC Dropout mode**:
- Dropout layers remain active at inference time (NOT disabled)
- Run inference T=30 times per graph
- Each run produces a different prediction (stochastic due to active dropout)

```python
model.train()  # keeps dropout active
samples = [model(graph) for _ in range(30)]  # 30 forward passes
mu  = mean(samples)    # per-node mean prediction
sigma = std(samples)   # per-node uncertainty estimate
```

For T8: 100 graphs × 30 samples = 3,000 inference runs = 228.25 minutes total.

**Evaluation:** Spearman rank correlation between sigma (uncertainty) and |error| (actual error).
T8 achieves rho = 0.4820 — nodes with higher sigma tend to have higher actual error.

---

## STEP 8: CONFORMAL PREDICTION

Split-conformal procedure on T8 test set:

```
100 test graphs split into:
  - Calibration set: 50 graphs
  - Evaluation set:  50 graphs

For calibration set:
  nonconformity score for node i = |y_true[i] - mu[i]|

Find q = (1-alpha)(1 + 1/n_cal) quantile of scores

Prediction interval for new node: [mu - q, mu + q]
```

Results for T8:
- 90% coverage: q = 9.9196, achieved = 90.02% ✓
- 95% coverage: q = 14.6766, achieved = 95.01% ✓

**Coverage guarantee:** By construction, conformal intervals have guaranteed marginal coverage. The 90.02% and 95.01% numbers confirm the procedure is working correctly.

---

## STEP 9: SIGMA-NORMALIZED CONFORMAL

An enhanced version uses MC Dropout sigma to create adaptive interval widths:

```
nonconformity score = |y_true[i] - mu[i]| / sigma[i]

Interval: [mu - q*sigma, mu + q*sigma]
```

This gives narrower intervals for low-uncertainty predictions and wider intervals for high-uncertainty ones.

Result: **65.8% narrower** intervals for low-uncertainty nodes vs fixed-width intervals.

---

## DATA FLOW SUMMARY TABLE

| Stage | Input | Output | Key File |
|---|---|---|---|
| Preprocessing | MATSim CSVs + road network | .pt graph files | process_simulations_for_gnn.py |
| Training | .pt files | model.pth checkpoint | run_models.py |
| Test evaluation | model.pth + test graphs | test_evaluation_complete.json | run_models.py |
| MC Dropout | model.pth + test graphs | mc_dropout_full_metrics*.json | ensemble_uq_experiments.py |
| Conformal | mc_dropout predictions + y_true | conformal_standard.json | ensemble_uq_experiments.py |
| Feature analysis | predictions + features | feature_analysis_report.txt | evaluation scripts |
