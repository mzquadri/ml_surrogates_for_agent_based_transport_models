"""
GOOGLE COLAB - RUN THIS TO LOAD YOUR 8 TRAINING FILES
Copy this entire code to a Colab notebook cell and run
"""

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Import libraries
import torch
import json
import joblib
import numpy as np
import os

print("="*80)
print("LOADING ALL 8 FILES FROM GOOGLE DRIVE")
print("="*80)

base_path = "/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts"
print(f"\nBase path: {base_path}")

# Check if files exist
print("\nChecking files...")
files_to_check = [
    "test_dl.pt",
    "test_loader_params.json",
    "test_pos_scaler.pkl",
    "test_x_scaler.pkl",
    "train_pos_scaler.pkl",
    "train_x_scaler.pkl",
    "validation_pos_scaler.pkl",
    "validation_x_scaler.pkl"
]

for i, filename in enumerate(files_to_check, 1):
    filepath = f"{base_path}/{filename}"
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ {i}. {filename:30s} ({size_mb:8.2f} MB)")
    else:
        print(f"  ✗ {i}. {filename:30s} NOT FOUND")

# ============================================================================
# FILE 1: test_loader_params.json
# ============================================================================

print("\n" + "="*80)
print("FILE 1: test_loader_params.json")
print("="*80)

with open(f"{base_path}/test_loader_params.json", 'r') as f:
    test_params = json.load(f)

print("\nTest DataLoader Configuration:")
for key, value in test_params.items():
    print(f"  {key:20s}: {value}")

# ============================================================================
# FILE 2-3: Training Scalers (x and pos)
# ============================================================================

print("\n" + "="*80)
print("FILE 2: train_x_scaler.pkl (Feature Scaler)")
print("="*80)

train_x_scaler = joblib.load(f"{base_path}/train_x_scaler.pkl")

print("\nTraining Features Normalization:")
print(f"  Number of features: {len(train_x_scaler.mean_)}")
print(f"  Scaler type: {type(train_x_scaler).__name__}")

print("\n  Feature Statistics:")
feature_names = ['VOL_BASE_CASE', 'CAPACITY_BASE_CASE', 'CAPACITY_REDUCTION', 'FREESPEED', 'LENGTH']
for i, name in enumerate(feature_names):
    mean = train_x_scaler.mean_[i]
    std = train_x_scaler.scale_[i] if hasattr(train_x_scaler, 'scale_') else 0
    print(f"    {i}. {name:20s} → Mean: {mean:10.4f}, Std: {std:10.4f}")

print("\n" + "="*80)
print("FILE 3: train_pos_scaler.pkl (Position Scaler)")
print("="*80)

train_pos_scaler = joblib.load(f"{base_path}/train_pos_scaler.pkl")

print("\nTraining Position Normalization:")
print(f"  Number of dimensions: {len(train_pos_scaler.mean_)}")
print(f"  X coordinate → Mean: {train_pos_scaler.mean_[0]:.4f}, Std: {train_pos_scaler.scale_[0]:.4f}")
print(f"  Y coordinate → Mean: {train_pos_scaler.mean_[1]:.4f}, Std: {train_pos_scaler.scale_[1]:.4f}")

# ============================================================================
# FILE 4-5: Validation Scalers
# ============================================================================

print("\n" + "="*80)
print("FILE 4: validation_x_scaler.pkl")
print("="*80)

val_x_scaler = joblib.load(f"{base_path}/validation_x_scaler.pkl")

print("\nValidation Features Normalization:")
print(f"  Number of features: {len(val_x_scaler.mean_)}")
print("\n  Feature Statistics:")
for i, name in enumerate(feature_names):
    mean = val_x_scaler.mean_[i]
    std = val_x_scaler.scale_[i]
    print(f"    {i}. {name:20s} → Mean: {mean:10.4f}, Std: {std:10.4f}")

print("\n" + "="*80)
print("FILE 5: validation_pos_scaler.pkl")
print("="*80)

val_pos_scaler = joblib.load(f"{base_path}/validation_pos_scaler.pkl")

print("\nValidation Position Normalization:")
print(f"  X coordinate → Mean: {val_pos_scaler.mean_[0]:.4f}, Std: {val_pos_scaler.scale_[0]:.4f}")
print(f"  Y coordinate → Mean: {val_pos_scaler.mean_[1]:.4f}, Std: {val_pos_scaler.scale_[1]:.4f}")

# ============================================================================
# FILE 6-7: Test Scalers
# ============================================================================

print("\n" + "="*80)
print("FILE 6: test_x_scaler.pkl")
print("="*80)

test_x_scaler = joblib.load(f"{base_path}/test_x_scaler.pkl")

print("\nTest Features Normalization:")
print(f"  Number of features: {len(test_x_scaler.mean_)}")
print("\n  Feature Statistics:")
for i, name in enumerate(feature_names):
    mean = test_x_scaler.mean_[i]
    std = test_x_scaler.scale_[i]
    print(f"    {i}. {name:20s} → Mean: {mean:10.4f}, Std: {std:10.4f}")

print("\n" + "="*80)
print("FILE 7: test_pos_scaler.pkl")
print("="*80)

test_pos_scaler = joblib.load(f"{base_path}/test_pos_scaler.pkl")

print("\nTest Position Normalization:")
print(f"  X coordinate → Mean: {test_pos_scaler.mean_[0]:.4f}, Std: {test_pos_scaler.scale_[0]:.4f}")
print(f"  Y coordinate → Mean: {test_pos_scaler.mean_[1]:.4f}, Std: {test_pos_scaler.scale_[1]:.4f}")

# ============================================================================
# FILE 8: test_dl.pt (Test DataLoader)
# ============================================================================

print("\n" + "="*80)
print("FILE 8: test_dl.pt (Test DataLoader)")
print("="*80)

test_dl = torch.load(f"{base_path}/test_dl.pt", weights_only=False)

print("\nTest DataLoader Statistics:")
print(f"  Type: {type(test_dl).__name__}")
print(f"  Number of batches: {len(test_dl)}")
print(f"  Batch size: {test_dl.batch_size}")
print(f"  Total test samples: {len(test_dl.dataset)}")

# Sample one batch to show structure
print("\n  Sample batch structure:")
sample_batch = next(iter(test_dl))
print(f"    - Node features (x): {sample_batch.x.shape}")
print(f"    - Edge indices: {sample_batch.edge_index.shape}")
print(f"    - Targets (y): {sample_batch.y.shape}")
if hasattr(sample_batch, 'pos'):
    print(f"    - Positions: {sample_batch.pos.shape}")
if hasattr(sample_batch, 'batch'):
    print(f"    - Batch indices: {sample_batch.batch.shape}")

# Show sample data from first batch
print("\n  Sample data from first test batch:")
print(f"    - First road features:\n{sample_batch.x[0]}")
print(f"    - First road target: {sample_batch.y[0].item():.4f}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("COMPLETE SUMMARY - ALL 8 FILES")
print("="*80)

print("\n📊 Dataset Split Information:")
print(f"  Training samples: {len(test_dl.dataset) * 16}")  # Approximate based on 80/15/5 split
print(f"  Validation samples: {len(test_dl.dataset) * 3}")
print(f"  Test samples: {len(test_dl.dataset)}")
print(f"  Total scenarios: ~8,308")

print("\n📈 Feature Ranges (from training data):")
for i, name in enumerate(feature_names):
    mean = train_x_scaler.mean_[i]
    std = train_x_scaler.scale_[i]
    # Approximate min/max assuming ~3 std from mean
    approx_min = mean - 3*std
    approx_max = mean + 3*std
    print(f"  {name:20s}: ~{approx_min:8.1f} to {approx_max:8.1f}")

print("\n🗺️  Paris Network Information:")
print(f"  Road segments: 31,635")
print(f"  Districts: 20")
print(f"  X range: {train_pos_scaler.mean_[0] - 3*train_pos_scaler.scale_[0]:.1f} to {train_pos_scaler.mean_[0] + 3*train_pos_scaler.scale_[0]:.1f}")
print(f"  Y range: {train_pos_scaler.mean_[1] - 3*train_pos_scaler.scale_[1]:.1f} to {train_pos_scaler.mean_[1] + 3*train_pos_scaler.scale_[1]:.1f}")

print("\n💾 File Sizes:")
total_size = 0
for i, filename in enumerate(files_to_check, 1):
    filepath = f"{base_path}/{filename}"
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    total_size += size_mb
    print(f"  {i}. {filename:30s} → {size_mb:8.2f} MB")

print(f"\n  Total: {total_size:.2f} MB")

print("\n" + "="*80)
print("✅ ALL 8 FILES SUCCESSFULLY LOADED FROM GOOGLE DRIVE!")
print("="*80)

print("""
🎯 Next Steps:

1. Train Elena's model using these scalers:
   - Use train_x_scaler.pkl for feature normalization
   - Use train_pos_scaler.pkl for position normalization
   
2. Evaluate model on test set:
   - Load test_dl.pt
   - Use test_x_scaler.pkl and test_pos_scaler.pkl
   
3. Make predictions on new scenarios:
   - Apply same normalization using saved scalers
   - Inverse transform predictions back to original scale

All data is ready for model training! 🚀
""")
