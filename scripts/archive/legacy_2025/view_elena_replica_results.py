"""
COMPLETE ANALYSIS - YOUR ELENA REPLICA TRAINING
WandB: olive-shadow-2 | Architecture: PointNetTransfGAT
"""

# Mount Drive
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("✓ Drive mounted")
except:
    pass

import torch
import json
import joblib
import os

print("="*80)
print("ZAMIN'S ELENA REPLICA MODEL - COMPLETE ANALYSIS")
print("="*80)

# ============================================================================
# TRAINING INFO
# ============================================================================

print("\n📊 WandB Training Information:")
print("="*80)
print(f"  User: mohdzaminquadri-technical-university-of-munich")
print(f"  Project: zamin_elena_replica")
print(f"  Run: olive-shadow-2")
print(f"  Architecture: PointNetTransfGAT")
print(f"  URL: https://wandb.ai/mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/runs/olive-shadow-2")

# ============================================================================
# LOAD 8 FILES
# ============================================================================

base = "/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts"

files = ["test_dl.pt", "test_loader_params.json", "test_pos_scaler.pkl", 
         "test_x_scaler.pkl", "train_pos_scaler.pkl", "train_x_scaler.pkl",
         "validation_pos_scaler.pkl", "validation_x_scaler.pkl"]

print(f"\n📁 Files at: {base}")
for i, f in enumerate(files, 1):
    size = os.path.getsize(f"{base}/{f}") / (1024*1024)
    print(f"  {i}. ✓ {f:30s} {size:6.2f} MB")

# Load test params
with open(f"{base}/test_loader_params.json") as f:
    params = json.load(f)

# Load scalers
train_x = joblib.load(f"{base}/train_x_scaler.pkl")
test_dl = torch.load(f"{base}/test_dl.pt", weights_only=False)

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)

features = ['VOL_BASE', 'CAPACITY_BASE', 'CAP_RED', 'SPEED', 'LENGTH']
print("\nFeature Normalization (Training):")
for i, name in enumerate(features):
    print(f"  {name:15s}: μ={train_x.mean_[i]:9.2f}, σ={train_x.scale_[i]:9.2f}")

sample = next(iter(test_dl))
print(f"\nTest Dataset:")
print(f"  Test samples: {len(test_dl.dataset):,}")
print(f"  Batches: {len(test_dl)}")
print(f"  Batch size: {test_dl.batch_size}")
print(f"  Roads/scenario: {sample.x.shape[0]:,}")
print(f"  Features: {sample.x.shape[1]}")
print(f"  Edges: {sample.edge_index.shape[1]:,}")

# ============================================================================
# WANDB RESULTS
# ============================================================================

print("\n" + "="*80)
print("WANDB RESULTS - HOW TO VIEW")
print("="*80)

print("""
🌐 View Online Dashboard:
   https://wandb.ai/mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/runs/olive-shadow-2

📊 Fetch Metrics in Colab:
   !pip install wandb -q
   import wandb
   wandb.login()
   
   api = wandb.Api()
   run = api.run("mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/olive-shadow-2")
   
   # Summary metrics
   print(f"Best R²: {run.summary.get('r^2')}")
   print(f"Best Loss: {run.summary.get('best_val_loss')}")
   
   # Full history
   history = run.history()
   import matplotlib.pyplot as plt
   plt.plot(history['epoch'], history['r^2'])
   plt.xlabel('Epoch')
   plt.ylabel('R²')
   plt.title('Training Progress')
   plt.show()
""")

print("\n" + "="*80)
print("✅ ALL DATA LOADED - CHECK WANDB FOR TRAINING METRICS")
print("="*80)
