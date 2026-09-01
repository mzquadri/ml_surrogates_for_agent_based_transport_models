# ============================================================================
# QUICK RESULTS - NO WANDB LOGIN NEEDED
# Shows everything from your Drive files
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

import torch
import joblib
import json

BASE = "/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts"

print("="*80)
print("YOUR TRAINING RESULTS - QUICK VIEW")
print("="*80)

# ============================================================================
# 1. DATASET INFO
# ============================================================================
print("\n📊 DATASET INFORMATION")
print("="*80)

test_dl = torch.load(f"{BASE}/test_dl.pt", weights_only=False)
sample = test_dl[0]

print(f"✓ Test scenarios: {len(test_dl):,}")
print(f"✓ Roads per scenario: {sample.x.shape[0]:,}")
print(f"✓ Total test roads: {len(test_dl) * sample.x.shape[0]:,}")
print(f"✓ Features per road: {sample.x.shape[1]}")
print(f"✓ Network connections: {sample.edge_index.shape[1]:,}")

# ============================================================================
# 2. FEATURE STATISTICS
# ============================================================================
print("\n📈 TRAINING DATA STATISTICS")
print("="*80)

scaler = joblib.load(f"{BASE}/train_x_scaler.pkl")
features = ['VOL_BASE', 'CAPACITY', 'CAP_RED', 'SPEED', 'LENGTH']

print("\nFeature Normalization:")
for i, name in enumerate(features):
    mean = scaler.mean_[i]
    std = scaler.scale_[i]
    print(f"  {name:12s}: Mean={mean:8.2f}, Std={std:8.2f}")

# ============================================================================
# 3. SAMPLE DATA
# ============================================================================
print("\n🔍 SAMPLE TEST DATA")
print("="*80)

print(f"\nFirst scenario, first 5 roads:")
print(f"  Features (normalized): {sample.x[:5].tolist()}")
print(f"  Targets: {sample.y[:5].squeeze().tolist()}")

# ============================================================================
# 4. WANDB RESULTS (Manual check)
# ============================================================================
print("\n🌐 YOUR WANDB TRAINING RESULTS")
print("="*80)

print(f"""
To see your actual R², Loss, and training metrics:

Option 1 - WEB BROWSER (EASIEST):
  1. Open this link:
     https://wandb.ai/mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/runs/olive-shadow-2
  
  2. You'll see graphs showing:
     ✓ R² score over epochs
     ✓ Training & validation loss
     ✓ Pearson & Spearman correlation
     ✓ Learning rate schedule

Option 2 - MOBILE APP:
  1. Download "Weights & Biases" app
  2. Login
  3. Go to "zamin_elena_replica" project
  4. Open "olive-shadow-2" run

Option 3 - API (if needed):
  Run this in NEW cell:
  
  import wandb
  wandb.login()  # Enter your API key from https://wandb.ai/authorize
  
  api = wandb.Api()
  run = api.run("mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/olive-shadow-2")
  
  print(f"R²: {run.summary.get('r^2', 'N/A')}")
  print(f"Loss: {run.summary.get('best_val_loss', 'N/A')}")
""")

# ============================================================================
# 5. COMPARISON
# ============================================================================
print("\n📊 COMPARISON WITH ELENA'S PAPER")
print("="*80)

print("""
Elena Boreale's Results (Paper):
  R² Score:     0.76 (overall)
  MSE:          24.95
  MAE:          2.74
  Pearson:      0.87
  Spearman:     0.85

Your Results:
  Check WandB dashboard at the link above!
  
Expected Performance:
  ✓ Primary roads: R² ~ 0.95
  ✓ Residential:   R² ~ 0.68
  ✓ Overall:       R² ~ 0.76
""")

print("\n" + "="*80)
print("✅ DATA ANALYSIS COMPLETE")
print("="*80)
print("\n💡 For training metrics, open WandB link in browser (no coding needed!)")
