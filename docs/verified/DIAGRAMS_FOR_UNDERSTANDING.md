# DIAGRAMS FOR UNDERSTANDING
> Text-based diagrams to visualize the system architecture, data flow, and key concepts.
> All diagrams based on verified code and JSON data.

---

## DIAGRAM 1: The Full System at a Glance

```
PROBLEM: City planner asks "What happens to Paris traffic if I reduce capacity on Road X?"

SLOW ANSWER (hours):
  Road Network + Policy --> [MATSim simulation] --> Traffic volumes for all 31,635 roads

FAST ANSWER (seconds):
  Road Network + Policy --> [GNN surrogate] --> Predicted delta volumes for all 31,635 roads

THIS THESIS: Build the GNN surrogate + quantify how uncertain its predictions are
```

---

## DIAGRAM 2: From Road Network to Line Graph

```
REAL PARIS ROAD NETWORK                    AFTER LINE GRAPH TRANSFORM
(standard graph representation)            (what the GNN actually sees)

     A ----road1---- B                     road1 ---[share junction B]--- road2
     |               |                        \
   road4           road2                       ---[share junction A]--- road4
     |               |
     D ----road3---- C                     road2 ---[share junction C]--- road3

Nodes = intersections (A,B,C,D)            Nodes = road SEGMENTS (road1,road2,road3,road4)
Edges = roads                              Edges = pairs sharing a junction

Result: 31,635 nodes (one per road segment)
```

---

## DIAGRAM 3: One Data Sample (One Graph)

```
For scenario_042 (one of 1,000 policy scenarios):

data.x  [31635 x 5]:               data.y  [31635 x 1]:
+---------+----------+-----+        +----------+
| road_1  | 500 1000 |...  |        | road_1   | +150  <- 150 more veh/h
| road_2  | 800 1200 |...  |        | road_2   | -600  <- capacity reduced
| road_3  | 200  400 |...  |        | road_3   | +80
| ...     | ...  ... |...  |        | ...      | ...
| road_31635 | ...   |...  |        | road_31635| +5
+---------+----------+-----+        +----------+
  VOL_BASE CAP_BASE                   TARGET: delta volume
  CAP_RED  FREESPEED LENGTH           (what GNN must predict)
```

---

## DIAGRAM 4: Model Architecture (PointNetTransfGAT)

```
Input: data.x [31635, 5], data.pos [31635, 3, 2], data.edge_index

     [31635, 5]
         |
         v
  PointNetConv(pos=start_coords)   <-- spatial awareness at segment starts
         |
     [31635, 128]
         |
  PointNetConv(pos=end_coords)     <-- spatial awareness at segment ends
         |
     [31635, 256]   (concatenated)
         |
  TransformerConv(->256, 4 heads)  <-- attention-weighted message passing
  + ReLU + [dropout if enabled]
         |
     [31635, 256]
         |
  TransformerConv(->512, 4 heads)  <-- deeper feature extraction
  + ReLU + [dropout if enabled]
         |
     [31635, 512]
         |
  GATConv(512->64)                 <-- graph attention aggregation
         |
     [31635, 64]
         |
  GATConv(64->1)  [gat_final]      <-- final prediction (NO dropout here)
         |
     [31635, 1]
         |
         v
  Output: predicted delta_volume per road segment
```

---

## DIAGRAM 5: Training History (8 Trials)

```
R2 on test set:

T1  [LINEAR ARCH - excluded] ████████████████████  0.786
    -------- architecture change (GATConv final) --------
T2  [batch=16, drop=0.3]     ████████████          0.512
T3  [WEIGHTED LOSS, drop=0.0] ██████               0.225  <-- weighted loss hurts
T4  [WEIGHTED LOSS, drop=0.3] ██████               0.243  <-- weighted loss hurts
    -------- batch size change to 8 --------
T5  [batch=8, drop=0.3]      █████████████         0.555  <-- improvement
T6  [LR=0.0003]              ████████████          0.522  <-- lower LR hurts
    -------- train/val/test split change to 80/10/10 --------
T7  [LR=0.0006]              █████████████         0.547
T8  [dropout=0.2]            ██████████████        0.596  <-- BEST
                                                   ^
                                              Goal: R2 as high as possible
```

---

## DIAGRAM 6: MC Dropout — How Uncertainty is Generated

```
NORMAL INFERENCE (model.eval()):
  Input graph -> [model, dropout OFF] -> one prediction per node

MC DROPOUT INFERENCE:
  Input graph -> [model, dropout ON] -> prediction_1 per node
  Input graph -> [model, dropout ON] -> prediction_2 per node
  ...
  Input graph -> [model, dropout ON] -> prediction_30 per node

                         ┌────────────────┐
                         │  30 predictions │  <- stochastic due to random dropout masks
                         │  per node       │
                         └────────────────┘
                                 |
                    ┌────────────┴──────────────┐
                    |                           |
                 MEAN (mu)                 STD (sigma)
               = best prediction         = uncertainty estimate
```

---

## DIAGRAM 7: Conformal Prediction — Coverage Guarantee

```
TEST SET: 100 graphs
    |
    ├── CALIBRATION (50 graphs)
    │       |
    │       | For each of the 1,581,750 nodes in calibration:
    │       | nonconformity_score = |y_true - mu|
    │       |
    │       | Find q = 90th percentile of all scores
    │       | q_90 = 9.9196 veh/h
    │       
    └── EVALUATION (50 graphs)
            |
            | Prediction interval: [mu - 9.92, mu + 9.92]
            |
            | Check: what % of y_true falls inside interval?
            | Answer: 90.02% ✓
            |
            | (target was 90%, achieved 90.02% = guarantee holds!)
```

---

## DIAGRAM 8: Why Ensemble Experiment rho=0.16 is NOT a failure

```
SINGLE MODEL MC DROPOUT (T8):
  - One model trained on graphs 0-799
  - Test on graphs 900-999
  - All 30 MC samples come from same model
  - sigma measures WITHIN-model uncertainty
  - Spearman rho = 0.48 (meaningful!)

ENSEMBLE of 5 MODELS (Experiment A):
  - Model 1: trained on random 80% subset
  - Model 2: trained on different random 80% subset
  - ...
  - Model 5: trained on yet another 80% subset
  - Each model has different training data
  - Test on graphs from MIXED distributions
  - sigma measures BETWEEN-model disagreement due to DATA differences
  - Spearman rho = 0.16 (low, but expected)

The low rho in Experiment A is not a failure of the method.
It reflects that models trained on different data subsets
disagree for distribution-shift reasons, not prediction uncertainty reasons.
```

---

## DIAGRAM 9: What sigma tells you vs what it doesn't

```
SIGMA IS GOOD FOR:
  Ranking predictions by reliability
  "I'm 90% confident in THIS prediction but less in THAT one"
  Triage: skip high-sigma predictions, report low-sigma predictions confidently

  Example: reject top 50% uncertain -> MAE drops from ~3.96 to ~2.4 veh/h (-39.9%)

SIGMA IS NOT GOOD FOR:
  Treating as a calibrated standard deviation
  Saying "I'm 95% confident the true value is in mu +/- 1.96*sigma"
  (k95 = 11.65 means you need mu +/- 11.65*sigma for actual 95% coverage)

CONFORMAL PREDICTION fixes the calibration:
  Replaces naive +/-1.96*sigma with calibrated +/-q_95 = +/-14.68 veh/h
  Provides GUARANTEED 95% coverage by construction
```
