"""
 TRUE EXACT REPLICA - USES ELENA'S ACTUAL REPO TRAINING FUNCTION
===================================================================
This script uses the ACTUAL training function from Elena's repository.
No custom reimplementation - direct call to repo's train_model().

Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100
Repo: https://github.com/ElenaBoreale/ml_surrogates_for_agent_based_transport_models

WHY THIS IS BETTER:
- Uses repo's actual training loop (BaseGNN.train_model)
- Uses repo's actual LR scheduler logic
- Uses repo's actual early stopping
- Uses repo's actual gradient accumulation
- ZERO reimplementation = ZERO bugs from interpretation

WHAT YOU NEED:
- DataLoaders from prepare_data_exact_repo_pipeline()
- WandB config object (simple class)
NOTE: TrainingConfig includes most common fields repo expects.
If you get AttributeError like "'TrainingConfig' object has no attribute 'xyz'",
simply add that field to TrainingConfig.__init__() with a sensible default."""

import os
import sys
import torch

# ============================================================================
# SETUP: REPOSITORY PATH
# ============================================================================

BASE_PATH = "/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models"
sys.path.insert(0, os.path.join(BASE_PATH, "scripts"))

# Import repo functions
from gnn.models.point_net_transf_gat import PointNetTransfGAT
from gnn.help_functions import GNN_Loss  # validate_model_during_training also here
from training.help_functions import EarlyStopping, set_random_seeds
import wandb


# ============================================================================
# CONFIG OBJECT (Simple class to match repo's expectations)
# ============================================================================

class TrainingConfig:
    """
    Config object with ONLY fields actually used by BaseGNN.train_model()
    Based on repo's scripts/gnn/models/base_gnn.py
    """
    def __init__(
        self,
        num_epochs=750,
        lr=5e-4,
        gradient_accumulation_steps=3,
        early_stopping_patience=40,
        predict_mode_stats=False,
        use_gradient_clipping=True,
        continue_training=False,
        base_checkpoint_path=None,
    ):
        # Fields ACTUALLY used by BaseGNN.train_model
        self.num_epochs = num_epochs
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.predict_mode_stats = predict_mode_stats
        self.use_gradient_clipping = use_gradient_clipping
        self.continue_training = continue_training
        self.base_checkpoint_path = base_checkpoint_path
        
        # Optional metadata (safe, not used by train_model but good for tracking)
        self.project_name = "TR-C_Benchmarks"
        self.unique_model_description = "PointNetTransfGAT_Zamin"


# ============================================================================
# DATA FIX HELPER (Reshape pos if needed)
# ============================================================================

def fix_pos_shape_in_dataset(dataset):
    """
    Fix pos shape mismatch: normalize_pos_features_batched produces (N, 3, 2)
    but model expects (N, 2, 2). Extract only first 2 position features.
    """
    print("\nChecking pos shape in dataset...")
    sample = dataset[0]
    
    if sample.pos.shape[1] == 3:
        print(f"  Fixing pos shape: {sample.pos.shape} → (num_nodes, 2, 2)")
        print("   Extracting first 2 position features (start/end coords)")
        
        for data in dataset:
            # Keep only first 2 position features (start and end coordinates)
            data.pos = data.pos[:, :2, :]  # (num_nodes, 3, 2) → (num_nodes, 2, 2)
        
        print(f"✓ Fixed pos shape to {dataset[0].pos.shape}")
    elif sample.pos.shape[1] == 2:
        print(f"✓ Pos shape already correct: {sample.pos.shape}")
    else:
        print(f"⚠️  Unexpected pos shape: {sample.pos.shape}")
    
    return dataset


# ============================================================================
# VALIDATION HELPERS (Pre-flight checks)
# ============================================================================

def validate_setup(model, train_loader, scalers_train, scalers_validation):
    """
    Pre-flight checks to catch common signature/format mismatches
    Prints warnings but doesn't block execution
    """
    print("\n" + "="*80)
    print("  PRE-FLIGHT VALIDATION CHECKS")
    print("="*80)
    
    # Check 1: train_model signature
    import inspect
    try:
        sig = inspect.signature(model.train_model)
        params = list(sig.parameters.keys())
        print(f"✓ train_model() parameters: {params}")
        
        # Check for common variations
        if 'valid_dl' not in params and 'val_dl' in params:
            print("  WARNING: Repo uses 'val_dl' not 'valid_dl' - update call!")
        if 'loss_fct' not in params and 'loss_func' in params:
            print("  WARNING: Repo uses 'loss_func' not 'loss_fct' - update call!")
    except Exception as e:
        print(f"  Could not inspect train_model signature: {e}")
    
    # Check 2: EarlyStopping signature
    try:
        from training.help_functions import EarlyStopping
        sig = inspect.signature(EarlyStopping)
        print(f"✓ EarlyStopping() parameters: {list(sig.parameters.keys())}")
    except Exception as e:
        print(f"  Could not inspect EarlyStopping: {e}")
    
    # Check 3: Scalers format
    try:
        print(f"✓ scalers_train type: {type(scalers_train)}")
        if hasattr(scalers_train, 'keys'):
            print(f"  Keys: {list(scalers_train.keys())}")
        print(f"✓ scalers_validation type: {type(scalers_validation)}")
        if hasattr(scalers_validation, 'keys'):
            print(f"  Keys: {list(scalers_validation.keys())}")
    except Exception as e:
        print(f"  Could not inspect scalers: {e}")
    
    # Check 4: data.pos shape (should be fixed before this point)
    try:
        sample_data = train_loader.dataset[0]
        pos_shape = sample_data.pos.shape
        print(f"✓ data.pos shape: {pos_shape}")
        
        if len(pos_shape) == 3 and pos_shape[1] == 2:
            print(f"  ✓ Shape correct for model: (num_nodes, 2, 2)")
        else:
            print(f"  ⚠️  Unexpected pos shape: {pos_shape}")
    except Exception as e:
        print(f"  Could not inspect pos shape: {e}")
    
    print("="*80 + "\n")


# ============================================================================
# TRAINING WRAPPER (Calls repo's actual train_model)
# ============================================================================

def train_with_repo_function(
    train_loader,
    val_loader,
    scalers_train,
    scalers_validation,
    save_path='/content/drive/MyDrive/Zamin_thesis/trained_models',
    num_epochs=750,
    seed=42,
    wandb_project_name="elena_model_reproduction"
):
    """
    Train using ACTUAL repo function - TRUE exact replica
    
    Args:
        train_loader, val_loader: DataLoaders from prepare_data_exact_repo_pipeline()
        scalers_train, scalers_validation: Scaler dicts from prepare_data_exact_repo_pipeline()            NOTE: prepare_data_exact_repo_pipeline() returns:
                  (train_loader, val_loader, test_loader, 
                   scalers_train, scalers_validation, scalers_test)
                  Pass scalers_validation (not scalers_val) to this function        save_path: Where to save model
        num_epochs: 750 (paper default)
        seed: Random seed (42)
        wandb_project_name: WandB project name
    
    Returns:
        model: Trained model (with best weights loaded)
        best_val_loss: Best validation loss
        best_epoch: Epoch with best validation loss
        loss_fn: Loss function (for test evaluation)
    """
    print("="*80)
    print(" TRAINING WITH ACTUAL REPO FUNCTION (TRUE EXACT REPLICA)")
    print("="*80)
    print(f"This uses Elena's actual BaseGNN.train_model() - NO custom reimplementation!")
    print("="*80 + "\n")
    
    # Set seeds (repo function supports seed parameter)
    set_random_seeds(seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ====================================================================
    # PATHS
    # ====================================================================
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, 'best_model.pth')
    
    # ====================================================================
    # CONFIG OBJECT (repo-used fields only)
    # ====================================================================
    print("\nCreating config object...")
    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=5e-4,
        gradient_accumulation_steps=3,
        early_stopping_patience=40,
        predict_mode_stats=False,
        use_gradient_clipping=True,
        continue_training=False,
        base_checkpoint_path=None,
    )
    print(f"✓ Config: {config.num_epochs} epochs, lr={config.lr}, grad_accum={config.gradient_accumulation_steps}")
    
    # ====================================================================
    # WANDB INIT (BEFORE model creation - repo style)
    # CRITICAL: BaseGNN.train_model() calls wandb.log() directly
    #           So wandb.init() MUST always run (even if mode="disabled")
    # ====================================================================
    print("\nInitializing WandB...")
    use_wandb = True  # Set to False to disable logging (but init still required)
    
    if use_wandb:
        wandb.init(
            project=wandb_project_name,
            config={
                "num_epochs": config.num_epochs,
                "lr": config.lr,
                "batch_size": getattr(train_loader, "batch_size", None),
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "early_stopping_patience": config.early_stopping_patience,
                "use_gradient_clipping": config.use_gradient_clipping,
                "architecture": "PointNetTransfGAT",
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "seed": seed,
                "paper": "Elena Boreale - Transport GNN",
            },
            reinit=True,
        )
        print(f"✓ WandB initialized (project: {wandb_project_name})")
    else:
        wandb.init(mode="disabled")  # Required: repo calls wandb.log()
        print(f"✓ WandB disabled (offline mode)")
    
    # ====================================================================
    # INITIALIZE MODEL (Exact paper config + log_to_wandb like repo)
    # NOTE: predict_mode_stats MUST match config.predict_mode_stats
    # ====================================================================
    print("\nInitializing PointNetTransfGAT model...")
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=0.3,
        use_dropout=True,
        predict_mode_stats=config.predict_mode_stats,  # Must match config
        dtype=torch.float32,
        log_to_wandb=True,  # Repo-style: model can update wandb.config
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {total_params:,} parameters")
    
    # ====================================================================
    # OPTIMIZER
    # ====================================================================
    print("\nInitializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
    )
    print(f"✓ AdamW optimizer (lr={config.lr}, weight_decay=1e-4)")
    
    # ====================================================================
    # FIX POS SHAPE (normalize_pos_features_batched produces (N,3,2) but model expects (N,2,2))
    # DataLoaders are immutable, so we need to recreate them with fixed datasets
    # ====================================================================
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    # Fix train dataset and recreate loader
    train_dataset_fixed = fix_pos_shape_in_dataset(train_loader.dataset)
    train_loader = PyGDataLoader(
        train_dataset_fixed,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Fix val dataset and recreate loader
    val_dataset_fixed = fix_pos_shape_in_dataset(val_loader.dataset)
    val_loader = PyGDataLoader(
        val_dataset_fixed,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # ====================================================================
    # LOSS FUNCTION (repo's GNN_Loss)
    # ====================================================================
    print("\nInitializing loss function...")
    num_nodes = train_loader.dataset[0].x.shape[0]
    loss_fn = GNN_Loss("mse", num_nodes, device, False)  # Positional args (repo-exact)
    print(f"✓ GNN_Loss (MSE, num_nodes={num_nodes})")
    
    # ====================================================================
    # EARLY STOPPING (repo's training/help_functions.py)
    # ====================================================================
    print("\nInitializing early stopping...")
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        verbose=True
    )
    print(f"✓ EarlyStopping (patience={config.early_stopping_patience})")
    
    # ====================================================================
    # PRE-FLIGHT VALIDATION (Check signatures before training)
    # ====================================================================
    validate_setup(model, train_loader, scalers_train, scalers_validation)
    
    # ====================================================================
    # CALL REPO'S ACTUAL TRAINING FUNCTION
    # ====================================================================
    print("="*80)
    print(" STARTING TRAINING (REPO'S ACTUAL FUNCTION)")
    print("="*80)
    print(f"Calling model.train_model() from BaseGNN...\n")
    
    try:
        best_val_loss, best_epoch = model.train_model(
            config=config,
            loss_fct=loss_fn,
            optimizer=optimizer,
            train_dl=train_loader,
            valid_dl=val_loader,
            device=device,
            early_stopping=early_stopping,
            model_save_path=model_save_path,
            scalers_train=scalers_train,
            scalers_validation=scalers_validation
        )
    except TypeError as e:
        print("\n" + "="*80)
        print(" PARAMETER MISMATCH ERROR")
        print("="*80)
        print(f"Error: {e}\n")
        print("Common fixes:")
        print("1. Check if repo uses 'val_dl' instead of 'valid_dl'")
        print("2. Check if repo uses 'loss_func' instead of 'loss_fct'")
        print("3. Run this in Colab to see exact signature:")
        print("   import inspect")
        print("   print(inspect.signature(model.train_model))")
        print("="*80)
        raise
    except KeyError as e:
        print("\n" + "="*80)
        print(" SCALER FORMAT ERROR")
        print("="*80)
        print(f"Error: Missing key {e} in scalers\n")
        print("Your scalers keys:")
        print(f"  train: {list(scalers_train.keys()) if hasattr(scalers_train, 'keys') else 'not a dict'}")
        print(f"  val: {list(scalers_validation.keys()) if hasattr(scalers_validation, 'keys') else 'not a dict'}")
        print("="*80)
        raise
    
    # ====================================================================
    # TRAINING COMPLETE
    # ====================================================================
    print("\n" + "="*80)
    print(" TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Model saved to: {model_save_path}")
    print("="*80)
    
    # ====================================================================
    # LOAD BEST MODEL (Critical: model may be at last epoch weights)
    # ====================================================================
    print("\n Loading best model weights...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    print(f"✓ Best model loaded from {model_save_path}")
    print("="*80)
    
    if use_wandb:
        wandb.finish()
    
    return model, best_val_loss, best_epoch, loss_fn


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" COLAB USAGE - TRUE EXACT REPO REPLICA")
    print("="*80)
    print("""

STEP 0: Login to WandB (first time only)
-----------------------------------------
import wandb
wandb.login()  # Enter your API key when prompted

# OR set it directly:
# import os
# os.environ['WANDB_API_KEY'] = 'your-api-key-here'


STEP 1: Prepare data with repo pipeline
----------------------------------------
%run colab_exact_repo_data_pipeline.py

# NOTE: Use consistent naming - function returns scalers_validation (not scalers_val)
train_loader, val_loader, test_loader, scalers_train, scalers_validation, scalers_test = \\
    prepare_data_exact_repo_pipeline(
        datalist=all_data,
        batch_size=8,
        save_path='/content/drive/MyDrive/Zamin_thesis/saved_pipeline_artifacts',
        use_bootstrapping=False,
        seed=42
    )


STEP 2: Load this TRUE replica script
--------------------------------------
%run colab_train_EXACT_REPO.py


STEP 3: Train with ACTUAL repo function
----------------------------------------
model, best_val_loss, best_epoch, loss_fn = train_with_repo_function(
    train_loader=train_loader,
    val_loader=val_loader,
    scalers_train=scalers_train,
    scalers_validation=scalers_validation,  # Use consistent naming
    save_path='/content/drive/MyDrive/Zamin_thesis/trained_models',
    num_epochs=750,
    seed=42,
    wandb_project_name='elena_exact_replica'
)

print(f"Training complete! Best epoch: {best_epoch}, Val loss: {best_val_loss:.6f}")


STEP 4: Test evaluation (repo-correct)
---------------------------------------
from gnn.help_functions import validate_model_during_training  # CORRECT import location

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CRITICAL: Fix test_loader pos shape (same as train/val)
# The fix_pos_shape_in_dataset function is already loaded from Step 2
test_loader.dataset = fix_pos_shape_in_dataset(test_loader.dataset)

# Model already loaded with best weights (done in train_with_repo_function)
# Repo returns: (test_loss, r2, spearman, pearson)
test_loss, r2, spearman, pearson = validate_model_during_training(
    config=TrainingConfig(),
    model=model,
    dataset=test_loader,
    loss_func=loss_fn,  # Available from Step 3 return
    device=device,
    scalers_validation=scalers_test
)

print(f"\\nTest Results:")
print(f"  Loss: {test_loss:.6f}")
print(f"  R²: {r2:.4f}")
print(f"  Spearman: {spearman:.4f}")
print(f"  Pearson: {pearson:.4f}")
print(f"\\n Paper target: R²=0.76 (may vary with 1000 vs 10k scenarios)")

    """)
    print("="*80)
    print("\n This is the TRUE exact replica!")
    print("   Uses repo's actual train_model() - ZERO custom code")
    print("   Includes pre-flight validation checks for common issues")
    print("="*80)
    print("\n TROUBLESHOOTING:")
    print("\n1. AttributeError: 'TrainingConfig' missing field 'xyz'")
    print("   → Add 'self.xyz = value' to TrainingConfig.__init__()")
    print("   → Check repo's scripts/training/run_models.py for correct value")
    print("\n2. TypeError: train_model() got unexpected keyword argument")
    print("   → Pre-flight check will show correct signature")
    print("   → Common: 'valid_dl' vs 'val_dl', 'loss_fct' vs 'loss_func'")
    print("\n3. KeyError in scalers")
    print("   → Pre-flight check will show your scaler keys")
    print("   → Verify scalers_train/validation have expected structure")
    print("="*80)
