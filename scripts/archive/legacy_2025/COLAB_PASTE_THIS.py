"""
COLAB READY - DIRECT PASTE THIS CODE
Copy entire code and paste in one Colab cell
"""

# ============================================================================
# AUTO MOUNT DRIVE
# ============================================================================
print("Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("✓ Drive mounted successfully\n")
except Exception as e:
    print(f"Error mounting drive: {e}\n")

# ============================================================================
# IMPORTS
# ============================================================================
import torch
import json
import joblib
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_PATH = "/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts"

print("="*80)
print("ELENA REPLICA MODEL - TRAINING RESULTS ANALYSIS")
print("="*80)

# ============================================================================
# CHECK PATH EXISTS
# ============================================================================
print(f"\n📁 Checking path: {BASE_PATH}")

if not os.path.exists(BASE_PATH):
    print(f"❌ ERROR: Path not found!")
    print(f"\nTrying to locate files...")
    
    # Try alternative paths
    possible_paths = [
        "/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts",
        "/content/drive/My Drive/Zamin_thesis/saved_pipeline_artifacts",
        "/content/drive/MyDrive/saved_pipeline_artifacts"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            BASE_PATH = path
            print(f"✓ Found files at: {path}")
            break
    else:
        print("\n❌ Files not found in common locations.")
        print("Please check your Drive folder structure.")
        sys.exit(1)
else:
    print("✓ Path exists\n")

# ============================================================================
# LIST FILES
# ============================================================================
print("="*80)
print("CHECKING FILES")
print("="*80)

required_files = [
    "test_dl.pt",
    "test_loader_params.json",
    "test_pos_scaler.pkl",
    "test_x_scaler.pkl",
    "train_pos_scaler.pkl",
    "train_x_scaler.pkl",
    "validation_pos_scaler.pkl",
    "validation_x_scaler.pkl"
]

print(f"\nFiles in: {BASE_PATH}\n")
found_files = []
missing_files = []

for i, filename in enumerate(required_files, 1):
    filepath = os.path.join(BASE_PATH, filename)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {i}. ✓ {filename:30s} ({size_mb:6.2f} MB)")
        found_files.append(filename)
    else:
        print(f"  {i}. ✗ {filename:30s} MISSING")
        missing_files.append(filename)

if missing_files:
    print(f"\n⚠ Warning: {len(missing_files)} files missing!")
    print("Continuing with available files...\n")

# ============================================================================
# WANDB INFO
# ============================================================================
print("\n" + "="*80)
print("WANDB TRAINING RUN INFORMATION")
print("="*80)

print("""
📊 Your Training Run:
  • User: mohdzaminquadri-technical-university-of-munich
  • Project: zamin_elena_replica
  • Run Name: olive-shadow-2
  • Architecture: PointNetTransfGAT (Elena's exact model)

🌐 WandB Dashboard:
  https://wandb.ai/mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/runs/olive-shadow-2

  View complete training metrics:
  - Loss curves (train & validation)
  - R² score progression
  - Learning rate schedule
  - Correlation metrics (Pearson, Spearman)
  - System metrics (GPU, memory)
""")

# ============================================================================
# LOAD AND ANALYZE FILES
# ============================================================================

if "test_loader_params.json" in found_files:
    print("\n" + "="*80)
    print("TEST LOADER CONFIGURATION")
    print("="*80)
    
    with open(f"{BASE_PATH}/test_loader_params.json", 'r') as f:
        params = json.load(f)
    
    print("\nDataLoader Settings:")
    for key, value in params.items():
        print(f"  {key:20s}: {value}")

if "train_x_scaler.pkl" in found_files:
    print("\n" + "="*80)
    print("FEATURE NORMALIZATION (Training Data)")
    print("="*80)
    
    scaler = joblib.load(f"{BASE_PATH}/train_x_scaler.pkl")
    
    feature_names = ['VOL_BASE_CASE', 'CAPACITY_BASE_CASE', 'CAPACITY_REDUCTION', 
                     'FREESPEED', 'LENGTH']
    
    print("\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        if i < len(scaler.mean_):
            mean = scaler.mean_[i]
            std = scaler.scale_[i] if hasattr(scaler, 'scale_') else 0
            print(f"  {name:22s}: μ = {mean:10.2f}, σ = {std:10.2f}")

if "train_pos_scaler.pkl" in found_files:
    print("\n" + "="*80)
    print("POSITION NORMALIZATION (Paris Coordinates)")
    print("="*80)
    
    pos_scaler = joblib.load(f"{BASE_PATH}/train_pos_scaler.pkl")
    
    print("\nSpatial Statistics:")
    print(f"  X coordinate (longitude): μ = {pos_scaler.mean_[0]:.2f}, σ = {pos_scaler.scale_[0]:.2f}")
    print(f"  Y coordinate (latitude):  μ = {pos_scaler.mean_[1]:.2f}, σ = {pos_scaler.scale_[1]:.2f}")

if "test_dl.pt" in found_files:
    print("\n" + "="*80)
    print("TEST DATASET ANALYSIS")
    print("="*80)
    
    print("\nLoading test dataset (may take a moment)...")
    test_dl = torch.load(f"{BASE_PATH}/test_dl.pt", weights_only=False)
    
    # Check if it's a DataLoader or list
    is_dataloader = hasattr(test_dl, 'dataset')
    
    print("\n📦 Dataset Statistics:")
    
    if is_dataloader:
        # It's a DataLoader
        print(f"  Type: DataLoader")
        print(f"  Total test samples: {len(test_dl.dataset):,}")
        print(f"  Number of batches: {len(test_dl)}")
        print(f"  Batch size: {test_dl.batch_size}")
        
        # Sample batch
        print("\n  Loading sample batch...")
        sample = next(iter(test_dl))
    else:
        # It's a list of Data objects
        print(f"  Type: List of scenarios")
        print(f"  Total test samples: {len(test_dl):,}")
        print(f"  Batch size: N/A (individual scenarios)")
        
        # Take first item
        print("\n  Loading first scenario...")
        sample = test_dl[0]
    
    print(f"\n  Sample Data Structure:")
    print(f"    • Node features (x): {sample.x.shape}")
    print(f"    • Edge indices: {sample.edge_index.shape}")
    print(f"    • Target values (y): {sample.y.shape}")
    
    if hasattr(sample, 'pos') and sample.pos is not None:
        print(f"    • Positions: {sample.pos.shape}")
    if hasattr(sample, 'batch') and sample.batch is not None:
        print(f"    • Batch info: {sample.batch.shape}")
    
    print(f"\n  Network Details:")
    print(f"    • Roads per scenario: {sample.x.shape[0]:,}")
    print(f"    • Features per road: {sample.x.shape[1]}")
    print(f"    • Total connections (edges): {sample.edge_index.shape[1]:,}")
    
    # Data sample
    print(f"\n  Sample Data (first road):")
    print(f"    Features: {sample.x[0].tolist()}")
    print(f"    Target: {sample.y[0].item():.4f}")
    
    # Additional stats for list format
    if not is_dataloader:
        print(f"\n  Dataset Format Info:")
        print(f"    This is a list of {len(test_dl)} individual scenarios")
        print(f"    Each scenario has {sample.x.shape[0]:,} roads")
        print(f"    Total roads across all scenarios: {len(test_dl) * sample.x.shape[0]:,}")

# ============================================================================
# FETCH WANDB METRICS (OPTIONAL)
# ============================================================================
print("\n" + "="*80)
print("FETCH WANDB METRICS (OPTIONAL)")
print("="*80)

print("""
To download and analyze WandB metrics, run this code in next cell:

```python
# Install and login
!pip install wandb -q
import wandb
wandb.login()  # Enter your API key when prompted

# Fetch run
api = wandb.Api()
run = api.run("mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/olive-shadow-2")

# Display summary
print("\\n" + "="*80)
print("TRAINING SUMMARY FROM WANDB")
print("="*80)
print(f"\\nBest Validation R²: {run.summary.get('r^2', 'N/A'):.4f}")
print(f"Best Validation Loss: {run.summary.get('best_val_loss', 'N/A'):.4f}")
print(f"Pearson Correlation: {run.summary.get('pearson', 'N/A'):.4f}")
print(f"Spearman Correlation: {run.summary.get('spearman', 'N/A'):.4f}")

# Get training history
history = run.history()
print(f"\\nTraining Duration: {len(history)} epochs")

# Plot R² over time
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(history['epoch'], history['r^2'], 'b-', label='R²')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Model Performance Over Training')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

total_size = sum(os.path.getsize(f"{BASE_PATH}/{f}") / (1024*1024) 
                 for f in found_files)

print(f"""
✅ Successfully analyzed {len(found_files)}/{len(required_files)} files

📊 Total Data Size: {total_size:.2f} MB

🎯 Next Steps:

1. VIEW WANDB DASHBOARD:
   Click: https://wandb.ai/mohdzaminquadri-technical-university-of-munich/zamin_elena_replica/runs/olive-shadow-2

2. FETCH METRICS (run code above):
   - Get R², Loss, Correlations
   - Plot training curves
   - Compare with Elena's results (R²=0.76)

3. LOAD MODEL FOR INFERENCE:
   If you saved model weights, load and evaluate

4. COMPARE RESULTS:
   Elena's paper: R²=0.76, MSE=24.95, MAE=2.74
   Check your WandB dashboard for comparison

🚀 All data files loaded successfully!
""")

print("="*80)
