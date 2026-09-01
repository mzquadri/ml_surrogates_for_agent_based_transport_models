"""
EXTRACT AND DISPLAY ALL 8 TRAINING FILES
Shows contents and statistics of all files created during training
"""

import torch
import json
import joblib
import numpy as np
import os
import sys

# Detect Colab environment
try:
    import google.colab
    IS_COLAB = True
except:
    IS_COLAB = False

# Mount Google Drive if on Colab
if IS_COLAB:
    print("="*80)
    print("GOOGLE COLAB DETECTED - MOUNTING DRIVE")
    print("="*80)
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

print("\n" + "="*80)
print("LOADING ALL 8 FILES FROM LAST WEEK'S TRAINING")
print("="*80)

# Set base path based on environment
if IS_COLAB:
    base_path = "/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts"
    print("\n✓ Running on Google Colab - using Drive path")
else:
    base_path = "data/TR-C_Benchmarks/trans_conv_5_features/data_created_during_training"
    print("\n✓ Running locally - using local path")

print(f"Base path: {base_path}")

# Check if path exists
if not os.path.exists(base_path):
    print(f"\n❌ ERROR: Path does not exist: {base_path}")
    print("\nPlease check:")
    if IS_COLAB:
        print("  1. Drive is mounted correctly")
        print("  2. Files are in: /content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts/")
        print("  3. You have access to the Drive folder")
    else:
        print("  1. You are in the correct directory")
        print("  2. Training data exists in data/TR-C_Benchmarks/")
    sys.exit(1)

# List files in directory
print("\n📁 Files found:")
try:
    files = os.listdir(base_path)
    for f in sorted(files):
        print(f"  - {f}")
except Exception as e:
    print(f"❌ ERROR listing files: {e}")
    sys.exit(1)

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

print("\n  Feature Means:")
feature_names = ['VOL_BASE_CASE', 'CAPACITY_BASE_CASE', 'CAPACITY_REDUCTION', 'FREESPEED', 'LENGTH']
for i, (name, mean) in enumerate(zip(feature_names, train_x_scaler.mean_)):
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
print("\n  Feature Means:")
for i, (name, mean) in enumerate(zip(feature_names, val_x_scaler.mean_)):
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
print("\n  Feature Means:")
for i, (name, mean) in enumerate(zip(feature_names, test_x_scaler.mean_)):
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

# ============================================================================
# SUMMARY - ALL 8 FILES
# ============================================================================

print("\n" + "="*80)
print("COMPLETE SUMMARY - ALL 8 FILES")
print("="*80)

file_sizes = {}
for filename in os.listdir(base_path):
    filepath = os.path.join(base_path, filename)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    file_sizes[filename] = size_mb

print("\nFile Sizes:")
for i, (filename, size) in enumerate(sorted(file_sizes.items()), 1):
    print(f"  {i}. {filename:30s} → {size:8.2f} MB")

print(f"\nTotal size: {sum(file_sizes.values()):.2f} MB")

# ============================================================================
# EXPORT SUMMARY TO JSON
# ============================================================================

summary_data = {
    "training_date": "October 7, 2024",
    "model_architecture": "TransConv",
    "features_used": 5,
    "test_loader_params": test_params,
    "scalers": {
        "train_features": {
            "means": train_x_scaler.mean_.tolist(),
            "stds": train_x_scaler.scale_.tolist()
        },
        "train_positions": {
            "means": train_pos_scaler.mean_.tolist(),
            "stds": train_pos_scaler.scale_.tolist()
        },
        "validation_features": {
            "means": val_x_scaler.mean_.tolist(),
            "stds": val_x_scaler.scale_.tolist()
        },
        "test_features": {
            "means": test_x_scaler.mean_.tolist(),
            "stds": test_x_scaler.scale_.tolist()
        }
    },
    "test_dataset": {
        "num_samples": len(test_dl.dataset),
        "batch_size": test_dl.batch_size,
        "num_batches": len(test_dl)
    },
    "file_sizes_mb": file_sizes
}

output_file = "training_files_summary.json"
with open(output_file, 'w') as f:
    json.dump(summary_data, f, indent=2)

print(f"\n Summary exported to: {output_file}")

print("\n" + "="*80)
print("ALL 8 FILES SUCCESSFULLY LOADED AND ANALYZED!")
print("="*80)
print("""
These files contain:
  ✓ Feature normalization statistics (train/val/test)
  ✓ Position normalization statistics (train/val/test)  
  ✓ Complete test dataset (ready for evaluation)
  ✓ DataLoader configuration

You can use these to:
  1. Make predictions on new data (using same normalization)
  2. Continue training from checkpoint
  3. Evaluate model performance on test set
  4. Reproduce exact training conditions
""")
