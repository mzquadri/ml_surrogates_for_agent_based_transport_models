# COLAB_RUNBOOK.md
# Google Colab Runbook — How to Run Inference and UQ on Colab
# Last verified: 2026-03-14

---

## Overview

This runbook describes how to reproduce the key inference and UQ experiments
in Google Colab. Training is NOT covered (expensive, not needed — use existing weights).

**Prerequisite:** Trained model checkpoint for Trial 8 must be uploaded to Colab or
mounted from Google Drive.

---

## Environment Setup (Cell 1)

```python
# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install torch_geometric
!pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
!pip install geopandas pandas numpy matplotlib scipy scikit-learn

# Verify
import torch
import torch_geometric
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Note:** PyG version compatibility matters. The original training used a specific
version — if you get errors, check the training environment's requirements.txt.

---

## Mount Google Drive (Cell 2)

```python
from google.colab import drive
drive.mount('/content/drive')

# Set your repo path — adjust to your Drive structure
REPO_PATH = '/content/drive/MyDrive/Nazim_thesis/ml_surrogates_for_agent_based_transport_models'
T8_PATH = f'{REPO_PATH}/data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout'
```

---

## Load the Model (Cell 3)

```python
import sys
sys.path.insert(0, f'{REPO_PATH}/scripts')

import torch
from gnn.models.point_net_transf_gat import PointNetTransfGAT

# T8 hyperparameters (verified from test_evaluation_complete.json)
model = PointNetTransfGAT(
    in_channels=5,
    dropout=0.2,   # T8 verified dropout
    # additional params as required by your model constructor
)

# Load checkpoint
checkpoint_path = f'{T8_PATH}/model_checkpoint.pth'  # adjust filename if different
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded successfully")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Important:** The exact checkpoint filename may differ. Check what `.pth` or `.pt`
files exist in the T8 folder.

---

## Load Test Data (Cell 4)

```python
import torch
from torch_geometric.data import DataLoader

# Adjust to your data path
TEST_DATA_PATH = f'{REPO_PATH}/data/processed/test_graphs/'  # adjust as needed

# Load PyG data objects
test_dataset = torch.load(f'{TEST_DATA_PATH}/test_data.pt')  # adjust filename
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Test graphs: {len(test_dataset)}")
print(f"First graph: {test_dataset[0]}")
print(f"Nodes per graph: {test_dataset[0].num_nodes}")
```

**Note:** The exact data format depends on how `process_simulations_for_gnn.py`
serializes the graphs. Check what files are produced by the preprocessing script.

---

## Standard Inference (Cell 5)

```python
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, batch.pos)
        # out may be (predictions, embeddings) — check model forward signature
        if isinstance(out, tuple):
            preds = out[0]
        else:
            preds = out
        all_preds.append(preds.squeeze().numpy())
        all_targets.append(batch.y.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

r2 = r2_score(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(np.mean((all_preds - all_targets)**2))

print(f"R²:   {r2:.4f}  (expected: 0.5957)")
print(f"MAE:  {mae:.2f}  (expected: 3.96)")
print(f"RMSE: {rmse:.2f}  (expected: 7.12)")
```

**Expected outputs (verified from JSON):**
- R²: 0.5957
- MAE: 3.96 veh/h
- RMSE: 7.12 veh/h

If your results differ significantly: check that you're using T8 weights, T8 test data,
and the same preprocessing pipeline.

---

## MC Dropout Inference (Cell 6)

```python
def enable_dropout(model):
    """Enable dropout layers at test time."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

T_SAMPLES = 30  # number of MC samples (verified: 30 used in original experiments)

model.eval()
enable_dropout(model)  # keep dropout active

mc_preds = []  # shape: [T_SAMPLES, n_test_nodes]

for t in range(T_SAMPLES):
    batch_preds = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.pos)
            if isinstance(out, tuple):
                preds = out[0]
            else:
                preds = out
            batch_preds.append(preds.squeeze().numpy())
    mc_preds.append(np.concatenate(batch_preds))
    if (t+1) % 5 == 0:
        print(f"MC sample {t+1}/{T_SAMPLES}")

mc_preds = np.array(mc_preds)  # [T, N]
mc_mean = mc_preds.mean(axis=0)  # [N]
mc_std = mc_preds.std(axis=0)    # [N] = uncertainty

print(f"Mean prediction shape: {mc_mean.shape}")
print(f"Std uncertainty shape: {mc_std.shape}")
```

---

## UQ Evaluation — Spearman Correlation (Cell 7)

```python
from scipy.stats import spearmanr

abs_error = np.abs(mc_mean - all_targets)
rho, pval = spearmanr(mc_std, abs_error)

print(f"Spearman ρ (uncertainty vs |error|): {rho:.4f}  (expected: 0.4820)")
print(f"p-value: {pval:.2e}")
print(f"n_nodes: {len(abs_error):,}  (expected: 3,163,500)")
```

**Expected:** ρ = 0.4820 (verified from mc_dropout_full_metrics_model8_mc30_100graphs.json)

---

## Conformal Prediction (Cell 8)

```python
# Conformal prediction uses 50/50 split of 100 test graphs
# 50 calibration graphs, 50 evaluation graphs

n_test = len(test_dataset)
n_calib = n_test // 2  # 50 calibration graphs

# Collect per-graph predictions and targets for calibration set
calib_errors = []

model.eval()
model_no_dropout = model  # use deterministic predictions for conformal

with torch.no_grad():
    for i, data in enumerate(test_dataset[:n_calib]):
        from torch_geometric.data import DataLoader as DL
        loader = DL([data], batch_size=1)
        for batch in loader:
            out = model_no_dropout(batch.x, batch.edge_index, batch.pos)
            if isinstance(out, tuple):
                preds = out[0].squeeze().numpy()
            else:
                preds = out.squeeze().numpy()
            targets = batch.y.numpy()
            calib_errors.extend(np.abs(preds - targets))

calib_errors = np.array(calib_errors)

# Compute quantiles
alpha_90 = 0.10
alpha_95 = 0.05
n = len(calib_errors)
q_90 = np.quantile(calib_errors, np.ceil((1-alpha_90)*(n+1))/n)
q_95 = np.quantile(calib_errors, np.ceil((1-alpha_95)*(n+1))/n)

print(f"q (90%): {q_90:.4f}  (expected: 9.9196)")
print(f"q (95%): {q_95:.4f}  (expected: 14.6766)")
```

---

## Runtime Estimates on Colab

| Task | Colab CPU | Colab GPU (T4) |
|------|-----------|----------------|
| Standard inference (100 graphs) | ~10 min | ~2 min |
| MC Dropout 30 samples (100 graphs) | ~300 min | ~60 min |
| Conformal calibration (50 graphs) | ~5 min | ~1 min |

**Note:** Original MC Dropout experiments took 228.25 minutes — this was likely on CPU
or a limited GPU. Use GPU runtime in Colab for speed.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| ImportError: torch_geometric | Wrong PyG version | Pin to exact version from training env |
| Model forward signature error | Wrong number of args | Check point_net_transf_gat.py forward() signature |
| R² much lower than 0.59 | Wrong test set / wrong model | Verify T8 checkpoint, T8 test graphs |
| Spearman ρ near 0 | Dropout not enabled | Call enable_dropout(model) before MC sampling |
| OOM on GPU | Too large batch | Use batch_size=1 for inference |
| Results differ from JSON | Preprocessing mismatch | Use same preprocessing as training |

---

## File Checklist Before Running

```
[ ] T8 model checkpoint (.pth or .pt file) uploaded to Drive
[ ] Processed test graph data (.pt or .pkl) available
[ ] scripts/gnn/models/point_net_transf_gat.py accessible
[ ] scripts/data_preprocessing/ accessible for data loading utilities
[ ] GPU runtime enabled in Colab (Runtime → Change runtime type → T4 GPU)
```

---

## Quick Verification (run this first)

After loading model and test data, run this sanity check:

```python
# Single graph forward pass
sample = test_dataset[0]
from torch_geometric.data import DataLoader as DL
loader = DL([sample], batch_size=1)
for batch in loader:
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.pos)
    print(f"Output shape: {out[0].shape if isinstance(out, tuple) else out.shape}")
    print(f"Expected: torch.Size([31635]) or torch.Size([31635, 1])")
    break
```
