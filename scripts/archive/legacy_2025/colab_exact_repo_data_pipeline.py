"""
 ACTUAL ELENA REPO DATA PIPELINE - DIRECT FUNCTION USE
========================================================
This script uses Elena's ACTUAL repository functions directly.
NO reimplementation - just wrapper + usage instructions.

Why this approach?
------------------
1. EXACT match guaranteed (using actual repo code)
2. No bugs from reimplementation
3. Handles all edge cases that repo handles
4. Future-proof if repo updates

Based on:
- scripts/training/help_functions.py::prepare_data_with_graph_features()
- scripts/gnn/gnn_io.py::split_into_subsets() (called internally)
"""

import os
import sys
import torch
import joblib

# ============================================================================
# SETUP: REPOSITORY PATH & IMPORTS
# ============================================================================

BASE_PATH = "/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models"
sys.path.insert(0, os.path.join(BASE_PATH, "scripts"))

# Import ACTUAL repo functions (already verified working)
from training.help_functions import (
    prepare_data_with_graph_features,
    set_random_seeds
)

# ============================================================================
# WRAPPER FUNCTION FOR CLARITY
# ============================================================================

def prepare_data_exact_repo_pipeline(
    datalist, 
    batch_size=8,
    save_path=None,
    use_bootstrapping=False,
    seed=42
):
    """
    Uses Elena's ACTUAL repo function: prepare_data_with_graph_features()
    
    This is the EXACT pipeline used in the paper/repo:
    1. Set random seeds (reproducibility)
    2. Split data (80/15/5 by default, or bootstrap)
    3. Select 5 features (VOL, CAP, CAP_RED, SPEED, LENGTH) - skip HIGHWAY
    4. Normalize x features (StandardScaler, fit on train only)
    5. Normalize pos features (StandardScaler, fit on train only)
    6. Create DataLoaders with collate_fn + seed_worker
    7. Save scalers to disk
    
    Args:
        datalist: List of PyG Data objects (your 1000 scenarios)
        batch_size: Batch size (paper uses 8)
        save_path: Where to save dataloaders/scalers
        use_bootstrapping: False for standard split, True for bootstrap+OOB
        seed: Random seed (default 42)
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader (returned by repo, but you need to extract from function)
        scalers_train: Dict with fitted scalers
        scalers_validation: Dict with validation scalers (for verification)
    """
    print("="*80)
    print(" USING ACTUAL ELENA REPO PIPELINE")
    print("="*80)
    print(f"Total scenarios: {len(datalist)}")
    print(f"Batch size: {batch_size}")
    print(f"Use bootstrapping: {use_bootstrapping}")
    print(f"Random seed: {seed}")
    print("="*80)
    
    # Set random seeds (IMPORTANT for reproducibility)
    set_random_seeds(seed)
    
    # Setup save path
    if save_path is None:
        save_path = f"{BASE_PATH}/saved_pipeline_artifacts"
    os.makedirs(save_path, exist_ok=True)
    
    # CRITICAL: Repo concatenates path + filename, so path MUST end with /
    if not save_path.endswith('/'):
        save_path = save_path + '/'
    
    print(f"\n Save path: {save_path}")
    
    # Call ACTUAL repo function
    print("\n Calling prepare_data_with_graph_features()...")
    print("   (This is Elena's EXACT function from the repo)")
    
    # NOTE: Repo returns 4 values (train, val, scalers_train, scalers_val)
    #       Test loader needs to be loaded separately from saved file
    train_loader, val_loader, scalers_train, scalers_validation = \
        prepare_data_with_graph_features(
            datalist=datalist,
            batch_size=batch_size,
            path_to_save_dataloader=save_path,
            use_all_features=False,      # False = use 5 features from ablation study
            use_bootstrapping=use_bootstrapping,
            is_eign=False                # False = use pos features (not eigenvalues)
        )
    
    # Load test loader from saved file
    print("\n Loading test loader from saved file...")
    test_loader = torch.load(os.path.join(save_path, 'test_dl.pt'), weights_only=False)
    
    # Load test scalers
    scalers_test = {
        'x_scaler': joblib.load(os.path.join(save_path, 'test_x_scaler.pkl')),
        'pos_scaler': joblib.load(os.path.join(save_path, 'test_pos_scaler.pkl'))
    }
    
    print("\n Pipeline complete!")
    print("="*80)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("="*80)
    
    # Sanity check
    batch = next(iter(train_loader))
    print(f"\n Sanity check (first batch):")
    print(f"   x shape: {batch.x.shape}  (last dim should be 5 features)")
    print(f"   pos shape: {batch.pos.shape}  (should be [N, 2, 2])")
    print(f"   y shape: {batch.y.shape}  (should be [N])")
    print(f"   batch size: {batch.num_graphs} graphs")
    
    print(f"\n Saved artifacts:")
    print(f"   - Scalers: {save_path}/train_x_scaler.pkl")
    print(f"   - Scalers: {save_path}/train_pos_scaler.pkl")
    print(f"   - Test loader: {save_path}/test_dl.pt")
    print(f"   - Loader params: {save_path}/test_loader_params.json")
    
    return train_loader, val_loader, test_loader, scalers_train, scalers_validation, scalers_test


# ============================================================================
# OPTIONAL: LOAD PREVIOUSLY SAVED SCALERS
# ============================================================================

def load_saved_scalers(save_path):
    """
    Load previously saved scalers for inference/evaluation
    """
    import joblib
    x_scaler = joblib.load(os.path.join(save_path, 'train_x_scaler.pkl'))
    pos_scaler = joblib.load(os.path.join(save_path, 'train_pos_scaler.pkl'))
    print(f" Loaded scalers from {save_path}")
    return x_scaler, pos_scaler


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" COLAB USAGE INSTRUCTIONS")
    print("="*80)
    print("""
    
STEP 1: Load this script in Colab
----------------------------------
%run colab_exact_repo_data_pipeline.py


STEP 2: Prepare your data (already loaded 1000 scenarios)
----------------------------------------------------------
# Your 'all_data' list with 1000 scenarios should be ready


STEP 3: Run the EXACT repo pipeline
------------------------------------
train_loader, val_loader, test_loader, scalers_train, scalers_val, scalers_test = \\
    prepare_data_exact_repo_pipeline(
        datalist=all_data,
        batch_size=8,
        save_path='/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts',
        use_bootstrapping=False,  # False = standard 80/15/5 split
        seed=42
    )


STEP 4: Verify the output
--------------------------
# Check batch
for batch in train_loader:
    print(f"x: {batch.x.shape}")      # [total_nodes, 5] features
    print(f"pos: {batch.pos.shape}")  # [total_nodes, 2, 2] positions
    print(f"y: {batch.y.shape}")      # [total_nodes] targets
    break


STEP 5: Load the complete training script
-------------------------------------------
%run colab_train_elena_model.py


STEP 6: Train model with EXACT paper configuration
---------------------------------------------------
model, history = setup_and_train(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=750,              # Paper: 750 epochs
    lr=5e-4,                     # Paper: 5e-4
    weight_decay=1e-4,           # Paper: 1e-4
    accumulation_steps=3,        # Paper: effective batch = 24
    early_stopping_patience=40,  # Paper: patience = 40
    save_path='/content/drive/MyDrive/Zamin_thesis/trained_models',
    seed=42
)

# Training includes:
# ✓ LR Scheduler (Linear Warmup 5% + Cosine Decay to 5e-6)
# ✓ Early Stopping (patience=40)
# ✓ Gradient Accumulation (steps=3)
# ✓ Mixed Precision (AMP)
# ✓ Gradient Clipping
# ✓ Progress bars + logging
# ✓ Automatic test evaluation with R² score
# ✓ Best model saved automatically
          

    """)
    print("="*80)
    print("\n This approach uses ACTUAL repo functions - EXACT match guaranteed!")
    print("📊 Next: Run colab_train_elena_model.py for complete training setup!")
    print("="*80)
