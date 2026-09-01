"""
VIEW YOUR TRAINED MODEL RESULTS
Script to load and analyze your trained model from last week
"""

import torch
import json
import joblib
import os
import pandas as pd
from pathlib import Path

# ============================================================================
# 1. CHECK SAVED MODEL FILES
# ============================================================================

print("="*80)
print("YOUR TRAINED MODEL FILES")
print("="*80)

base_path = "data/TR-C_Benchmarks/trans_conv_5_features"

# Check what files exist
print("\n📁 Files in your trained model directory:")
print(f"\nBase path: {base_path}/")

# Data created during training
data_path = f"{base_path}/data_created_during_training"
if os.path.exists(data_path):
    print(f"\n✅ Data files found in: {data_path}/")
    for file in os.listdir(data_path):
        file_size = os.path.getsize(os.path.join(data_path, file)) / (1024 * 1024)  # MB
        print(f"   - {file:30s} ({file_size:.2f} MB)")

# Trained model
model_path = f"{base_path}/trained_model"
if os.path.exists(model_path):
    print(f"\n✅ Model directory: {model_path}/")
    if os.path.exists(f"{model_path}/model.pth"):
        model_size = os.path.getsize(f"{model_path}/model.pth") / (1024 * 1024)
        print(f"   - model.pth ({model_size:.2f} MB) ✓")
    else:
        print(f"   ⚠️  model.pth NOT FOUND (training may not have completed)")
    
    # Checkpoints
    checkpoint_path = f"{model_path}/checkpoints"
    if os.path.exists(checkpoint_path):
        checkpoints = os.listdir(checkpoint_path)
        if checkpoints:
            print(f"\n   📦 Checkpoints ({len(checkpoints)} files):")
            for ckpt in sorted(checkpoints):
                print(f"      - {ckpt}")
        else:
            print(f"\n   ⚠️  No checkpoints saved (training was short or failed)")

# ============================================================================
# 2. LOAD MODEL CHECKPOINT (if exists)
# ============================================================================

print("\n" + "="*80)
print("LOAD MODEL CHECKPOINT")
print("="*80)

checkpoint_files = []
if os.path.exists(f"{model_path}/checkpoints"):
    checkpoint_files = [f for f in os.listdir(f"{model_path}/checkpoints") if f.endswith('.pt')]

if checkpoint_files:
    # Load the latest checkpoint
    latest_checkpoint = sorted(checkpoint_files)[-1]
    checkpoint_full_path = f"{model_path}/checkpoints/{latest_checkpoint}"
    
    print(f"\n📦 Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(checkpoint_full_path, map_location='cpu', weights_only=False)
    
    print(f"\n✓ Checkpoint loaded successfully!")
    print(f"\nCheckpoint details:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"  Current validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    if 'model_state_dict' in checkpoint:
        print(f"  Model state: ✓ Present")
    if 'optimizer_state_dict' in checkpoint:
        print(f"  Optimizer state: ✓ Present")
    if 'scaler_state_dict' in checkpoint:
        print(f"  Scaler state: ✓ Present")
else:
    print("\n⚠️  No checkpoints found. Model may not have been trained.")

# ============================================================================
# 3. LOAD SCALERS
# ============================================================================

print("\n" + "="*80)
print("DATA SCALERS")
print("="*80)

scalers_info = {}
scaler_files = {
    'train_x_scaler.pkl': 'Training features',
    'train_pos_scaler.pkl': 'Training positions',
    'test_x_scaler.pkl': 'Test features',
    'test_pos_scaler.pkl': 'Test positions',
}

for file, description in scaler_files.items():
    scaler_path = f"{data_path}/{file}"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"\n✓ {description}: {file}")
        if hasattr(scaler, 'mean_'):
            print(f"  Mean shape: {scaler.mean_.shape}")
            print(f"  Features normalized: {len(scaler.mean_)}")
            scalers_info[file] = {
                'mean': scaler.mean_,
                'scale': scaler.scale_ if hasattr(scaler, 'scale_') else None
            }

# ============================================================================
# 4. CHECK WANDB LOGS
# ============================================================================

print("\n" + "="*80)
print("WANDB TRAINING LOGS")
print("="*80)

wandb_path = "wandb"
if os.path.exists(wandb_path):
    runs = [d for d in os.listdir(wandb_path) if d.startswith('run-')]
    print(f"\n✓ Found {len(runs)} WandB runs:")
    
    for run in sorted(runs):
        run_path = f"{wandb_path}/{run}"
        
        # Get run timestamp from folder name
        run_date = run.split('-')[1].split('_')[0]  # YYYYMMDD
        run_time = run.split('-')[1].split('_')[1]  # HHMMSS
        run_id = run.split('-')[2]
        
        formatted_date = f"20{run_date[2:4]}-{run_date[4:6]}-{run_date[6:8]}"
        formatted_time = f"{run_time[:2]}:{run_time[2:4]}:{run_time[4:6]}"
        
        print(f"\n  📊 Run: {run_id}")
        print(f"     Date: {formatted_date} {formatted_time}")
        print(f"     Path: {run_path}/")
        
        # Check for history file
        files_dir = f"{run_path}/files"
        if os.path.exists(files_dir):
            history_file = f"{files_dir}/wandb-history.jsonl"
            summary_file = f"{files_dir}/wandb-summary.json"
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    if 'best_val_loss' in summary:
                        print(f"     Best val loss: {summary['best_val_loss']:.4f}")
                    if 'r^2' in summary:
                        print(f"     R²: {summary['r^2']:.4f}")

print("\n" + "="*80)
print("HOW TO VIEW YOUR RESULTS")
print("="*80)

print("""
1️⃣  VIEW WANDB DASHBOARD (Recommended):
   
   Option A - Online Dashboard:
   1. Visit: https://wandb.ai
   2. Login with your credentials
   3. Go to project: "TR-C_Benchmarks"
   4. View all metrics, graphs, and comparisons
   
   Option B - Local CLI:
   In terminal, run:
   > wandb sync wandb/run-<your-run-id>
   
2️⃣  LOAD MODEL FOR INFERENCE:

   ```python
   import torch
   from scripts.gnn.models.trans_conv import TransConv
   
   # Load model
   model = TransConv(in_channels=5, out_channels=1)
   model.load_state_dict(torch.load('data/TR-C_Benchmarks/trans_conv_5_features/trained_model/model.pth'))
   model.eval()
   
   # Use for predictions
   with torch.no_grad():
       predictions = model(your_data)
   ```

3️⃣  CONTINUE TRAINING:

   Run command with --continue_training flag:
   ```bash
   python scripts/training/run_models.py \\
     --gnn_arch trans_conv \\
     --continue_training True \\
     --base_checkpoint_path "data/TR-C_Benchmarks/trans_conv_5_features/trained_model/checkpoints/checkpoint_epoch_XXX.pt" \\
     --unique_model_description trans_conv_5_features_continued
   ```

4️⃣  EXTRACT METRICS HISTORY:

   ```python
   import json
   
   # Read WandB history
   with open('wandb/run-XXXX/files/wandb-history.jsonl', 'r') as f:
       history = [json.loads(line) for line in f]
   
   # Extract specific metrics
   epochs = [h.get('epoch') for h in history if 'epoch' in h]
   val_losses = [h.get('val_loss') for h in history if 'val_loss' in h]
   r2_scores = [h.get('r^2') for h in history if 'r^2' in h]
   ```

5️⃣  ANALYZE TEST SET PERFORMANCE:

   ```python
   # Load test dataloader
   test_dl = torch.load('data/TR-C_Benchmarks/trans_conv_5_features/data_created_during_training/test_dl.pt')
   
   # Load model and evaluate
   # (Use scripts/evaluation/ folder functions)
   ```
""")

print("="*80)
print("\n✅ Results summary script completed!")
print("\nYour training from last week is saved and accessible.")
print("Check WandB dashboard for detailed metrics and graphs.\n")
