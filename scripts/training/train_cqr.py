"""
CQR GNN Training Script
========================
Trains PointNetTransfGATQuantile using exact T8 hyperparameters.
The model outputs (q_lo, q_hi) at quantile levels tau=0.05 and tau=0.95,
trained with PinballLoss (joint pinball at both quantiles).

After training completes, a calibration pass is run over the validation set
(= calibration set for CQR) and val_predictions.npz is saved.
This file is consumed by evaluate_cqr.py to compute the conformal correction.

CQR approach: Romano, Patterson & Candes (2019) NeurIPS, arXiv:1905.03222
Reference impl: yromano/cqr (github.com/yromano/cqr)

T8 Hyperparameters (EXACT MATCH):
  - dropout:                 0.2
  - lr:                      0.0005
  - batch_size:              8
  - gradient_accumulation:   3
  - split:                   80/10/10
  - in_channels:             5
  - early_stopping_patience: 25
  - num_epochs:              1000
  - seed:                    42

New (CQR-specific):
  - loss:   PinballLoss(alpha=0.10)  [tau_lo=0.05, tau_hi=0.95]
  - output: (q_lo, q_hi) per node

Output directory:
  data/TR-C_Benchmarks/point_net_transf_gat_10th_trial_cqr/

IMPORTANT: T8 artifacts are NEVER touched by this script.

Author: Mohd Zamin Quadri
"""

import os
import sys
import json
import time
import math
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---- Path setup ----
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ---- Hyperparameters (exact T8 values) ----
DROPOUT       = 0.2
LR            = 0.0005
BATCH_SIZE    = 8
GRAD_ACCUM    = 3
NUM_EPOCHS    = 1000
PATIENCE      = 25
ALPHA         = 0.10     # Miscoverage level: tau_lo=0.05, tau_hi=0.95
SEED          = 42
IN_CHANNELS   = 5
USE_GRAD_CLIP = True

DATASET_PATH    = os.path.join(PROJECT_ROOT, 'data', 'train_data', 'dist_not_connected_10k_1pct')
BASE_DIR        = os.path.join(PROJECT_ROOT, 'data', 'TR-C_Benchmarks')
EXP_NAME        = 'point_net_transf_gat_10th_trial_cqr'
EXP_DIR         = os.path.join(BASE_DIR, EXP_NAME)
MODEL_SAVE_PATH = os.path.join(EXP_DIR, 'trained_model', 'model.pth')
DATA_SAVE_PATH  = os.path.join(EXP_DIR, 'data_created_during_training')
LOG_FILE        = os.path.join(EXP_DIR, 'training_log.json')
VAL_PRED_PATH   = os.path.join(EXP_DIR, 'val_predictions.npz')


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset():
    """Load all batch files in order (same as run_models.py and train_heteroscedastic.py)."""
    from data_preprocessing.process_simulations_for_gnn import EdgeFeatures
    datalist = []
    batch_num = 1
    while True:
        batch_file = os.path.join(DATASET_PATH, f'datalist_batch_{batch_num}.pt')
        if not os.path.exists(batch_file):
            break
        print(f"  Loading batch {batch_num}...")
        batch_data = torch.load(batch_file, map_location='cpu', weights_only=False)
        if isinstance(batch_data, list):
            datalist.extend(batch_data)
        batch_num += 1
    # Fix num_nodes (same temp-fix as in run_models.py)
    for data in datalist:
        data.num_nodes = data.x.shape[0]
    print(f"  Loaded {len(datalist)} graphs total.")
    return datalist


def split_dataset(datalist, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Deterministic 80/10/10 split (same as T8 via gnn_io)."""
    from gnn.gnn_io import split_into_subsets
    train_set, valid_set, test_set = split_into_subsets(
        dataset=datalist, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    return train_set, valid_set, test_set


def normalize_and_build_loaders(train_set, valid_set, test_set, batch_size, save_dir):
    """
    Normalize node features using per-split fitted StandardScalers (same as T8).
    Saves scalers and test_dl to save_dir.
    Returns: train_loader, val_loader
    """
    from training.help_functions import normalize_dataset
    import joblib
    from gnn.gnn_io import save_dataloader, save_dataloader_params, collate_fn

    node_features = ["VOL_BASE_CASE", "CAPACITY_BASE_CASE", "CAPACITY_REDUCTION", "FREESPEED", "LENGTH"]

    print("  Normalizing train set...")
    train_normalized, scalers_train = normalize_dataset(dataset_input=train_set, node_features=node_features)
    print("  Normalizing validation set...")
    valid_normalized, scalers_val   = normalize_dataset(dataset_input=valid_set, node_features=node_features)
    print("  Normalizing test set...")
    test_normalized, scalers_test   = normalize_dataset(dataset_input=test_set,  node_features=node_features)

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, collate_fn=collate_fn
        )

    train_loader = make_loader(train_normalized, shuffle=True)
    val_loader   = make_loader(valid_normalized, shuffle=False)
    test_loader  = make_loader(test_normalized,  shuffle=False)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scalers_train['x_scaler'],   os.path.join(save_dir, 'train_x_scaler.pkl'))
    joblib.dump(scalers_train['pos_scaler'], os.path.join(save_dir, 'train_pos_scaler.pkl'))
    joblib.dump(scalers_val['x_scaler'],     os.path.join(save_dir, 'validation_x_scaler.pkl'))
    joblib.dump(scalers_val['pos_scaler'],   os.path.join(save_dir, 'validation_pos_scaler.pkl'))
    joblib.dump(scalers_test['x_scaler'],    os.path.join(save_dir, 'test_x_scaler.pkl'))
    joblib.dump(scalers_test['pos_scaler'],  os.path.join(save_dir, 'test_pos_scaler.pkl'))
    save_dataloader(test_loader, os.path.join(save_dir, 'test_dl.pt'))
    save_dataloader_params(test_loader, os.path.join(save_dir, 'test_loader_params.json'))
    print("  Scalers and test_dl saved.")

    return train_loader, val_loader


def compute_r2(preds, targets):
    """Compute R-squared."""
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def validate(model, val_loader, loss_fn, device):
    """
    CQR validation loop.
    Primary metric: val_pinball (for best-model selection).
    Also reports R^2 from prediction midpoints (monitoring only).

    Returns: (val_pinball, r2, spearman_r, pearson_r)
    """
    model.eval()
    total_pinball = 0.0
    n_batches     = 0
    all_midpoints = []
    all_targets   = []

    with torch.no_grad():
        for data in val_loader:
            data   = data.to(device)
            target = data.y.to(device)

            q_lo, q_hi = model(data)
            loss = loss_fn(q_lo, q_hi, target)

            total_pinball += loss.item()
            n_batches     += 1

            # Midpoint of raw quantile outputs (no ordering needed for monitoring R^2)
            midpoint = (q_lo.squeeze() + q_hi.squeeze()) / 2.0
            all_midpoints.append(midpoint.cpu().numpy())
            all_targets.append(target.squeeze().cpu().numpy())

    all_midpoints = np.concatenate(all_midpoints)
    all_targets   = np.concatenate(all_targets)

    val_pinball = total_pinball / n_batches if n_batches > 0 else float('inf')
    r2          = compute_r2(all_midpoints, all_targets)
    spearman_r, _ = spearmanr(all_midpoints, all_targets)
    pearson_r,  _ = pearsonr(all_midpoints,  all_targets)

    return val_pinball, r2, spearman_r, pearson_r


def run_calibration_pass(model, val_loader, device, save_path):
    """
    Run inference over the validation set (= calibration set for CQR)
    using the best saved model. Applies min/max monotonicity ordering
    following LearnerOptimizedCrossing.predict() from yromano/cqr.

    Saves val_predictions.npz with keys: q_lo, q_hi, targets (all 1D).

    Args:
        model (nn.Module): Best model (state dict already loaded).
        val_loader:        Validation DataLoader.
        device:            Torch device.
        save_path (str):   Path for .npz output.
    """
    model.eval()
    all_q_lo    = []
    all_q_hi    = []
    all_targets = []

    with torch.no_grad():
        for data in val_loader:
            data   = data.to(device)
            target = data.y

            q_lo, q_hi = model(data)

            all_q_lo.append(q_lo.squeeze().cpu().numpy())
            all_q_hi.append(q_hi.squeeze().cpu().numpy())
            all_targets.append(target.squeeze().cpu().numpy())

    all_q_lo    = np.concatenate(all_q_lo)
    all_q_hi    = np.concatenate(all_q_hi)
    all_targets = np.concatenate(all_targets)

    # Apply min/max monotonicity ordering
    # Follows LearnerOptimizedCrossing.predict() in yromano/cqr
    q_lo_ord = np.minimum(all_q_lo, all_q_hi)
    q_hi_ord = np.maximum(all_q_lo, all_q_hi)

    np.savez(save_path, q_lo=q_lo_ord, q_hi=q_hi_ord, targets=all_targets)

    n_crossings = int(np.sum(all_q_lo > all_q_hi))
    print(f"  Calibration pass complete: {len(all_targets)} samples")
    print(f"  Monotonicity crossings corrected: {n_crossings} / {len(all_q_lo)}")
    print(f"  q_lo range: [{q_lo_ord.min():.4f}, {q_lo_ord.max():.4f}]")
    print(f"  q_hi range: [{q_hi_ord.min():.4f}, {q_hi_ord.max():.4f}]")
    print(f"  targets range: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
    print(f"  Saved: {save_path}")


def train():
    """Main training function."""
    set_seeds(SEED)

    # Create output directories
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    checkpoint_dir = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("CQR GNN TRAINING")
    print(f"Experiment: {EXP_NAME}")
    print("=" * 60)
    print(f"  dropout={DROPOUT}, lr={LR}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"  alpha={ALPHA} (tau_lo={ALPHA/2}, tau_hi={1-ALPHA/2})")
    print(f"  max_epochs={NUM_EPOCHS}, patience={PATIENCE}")
    print()

    # Device: prefer XPU (Intel Arc), then CUDA, then CPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"  Using XPU (Intel Arc): {torch.xpu.get_device_name(0)}")
        amp_device_type = 'xpu'
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        amp_device_type = 'cuda'
    else:
        device = torch.device('cpu')
        print("  No GPU available, using CPU")
        amp_device_type = 'cpu'

    # Load data
    print("\nLoading dataset...")
    datalist = load_dataset()

    # Split
    print("Splitting dataset (80/10/10)...")
    train_set, valid_set, test_set = split_dataset(datalist)
    print(f"  Train: {len(train_set.indices)}, Val: {len(valid_set.indices)}, Test: {len(test_set.indices)}")

    # Normalize and build loaders
    print("Normalizing and building data loaders...")
    train_loader, val_loader = normalize_and_build_loaders(
        train_set, valid_set, test_set, BATCH_SIZE, DATA_SAVE_PATH
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    from gnn.models.point_net_transf_gat_quantile import PointNetTransfGATQuantile
    model = PointNetTransfGATQuantile(
        in_channels=IN_CHANNELS,
        out_channels=2,
        dropout=DROPOUT,
        use_dropout=True,
        log_to_wandb=False
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Loss and optimizer
    from gnn.losses.quantile_loss import PinballLoss
    loss_fn   = PinballLoss(alpha=ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # LR Scheduler (same as T8: linear warmup + cosine decay)
    from gnn.help_functions import LinearWarmupCosineDecayScheduler
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler   = LinearWarmupCosineDecayScheduler(initial_lr=LR, total_steps=total_steps)

    # AMP Scaler
    grad_scaler = torch.amp.GradScaler(amp_device_type)

    # Training state
    best_val_pinball = float('inf')
    best_epoch       = 0
    patience_counter = 0
    log_history      = []

    print("\nStarting training...\n")
    train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        epoch_train_loss = 0.0
        n_train_batches  = 0

        for idx, data in enumerate(tqdm(train_loader,
                                        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                                        leave=False, disable=True)):
            step = epoch * len(train_loader) + idx
            # Update LR
            lr = scheduler.get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            data   = data.to(device)
            target = data.y.to(device)

            with torch.amp.autocast(amp_device_type):
                q_lo, q_hi = model(data)
                loss = loss_fn(q_lo, q_hi, target)

            epoch_train_loss += loss.item()
            n_train_batches  += 1

            grad_scaler.scale(loss).backward()

            if (idx + 1) % GRAD_ACCUM == 0:
                if USE_GRAD_CLIP:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()

        # Handle remaining accumulation steps
        if len(train_loader) % GRAD_ACCUM != 0:
            if USE_GRAD_CLIP:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        avg_train_pinball = epoch_train_loss / n_train_batches if n_train_batches > 0 else float('inf')

        # Validation
        val_pinball, r2, spearman_r, pearson_r = validate(
            model, val_loader, loss_fn, device
        )

        # Log
        row = {
            'epoch':          epoch + 1,
            'train_pinball':  avg_train_pinball,
            'val_pinball':    val_pinball,
            'r2_midpoint':    r2,
            'spearman':       spearman_r,
            'pearson':        pearson_r,
            'lr':             lr
        }
        log_history.append(row)

        print(
            f"Epoch {epoch+1:4d} | train_pinball={avg_train_pinball:.4f} | "
            f"val_pinball={val_pinball:.4f} | r2_mid={r2:.4f} | "
            f"spearman={spearman_r:.4f} | lr={lr:.6f}",
            flush=True
        )

        # Save best model (lowest val pinball loss)
        if val_pinball < best_val_pinball:
            best_val_pinball = val_pinball
            best_epoch       = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [BEST] Model saved (val_pinball={val_pinball:.4f})", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

        # Periodic checkpoint (every 20 epochs)
        if (epoch + 1) % 20 == 0:
            cp_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pinball':      val_pinball,
                'best_val_pinball': best_val_pinball,
            }, cp_path)

    elapsed = time.time() - train_start
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")
    print(f"Best val_pinball={best_val_pinball:.4f} at epoch {best_epoch}")

    # Save training log
    training_summary = {
        'experiment': EXP_NAME,
        'hyperparameters': {
            'dropout':              DROPOUT,
            'lr':                   LR,
            'batch_size':           BATCH_SIZE,
            'gradient_accumulation': GRAD_ACCUM,
            'num_epochs':           NUM_EPOCHS,
            'patience':             PATIENCE,
            'alpha':                ALPHA,
            'tau_lo':               ALPHA / 2.0,
            'tau_hi':               1.0 - ALPHA / 2.0,
            'in_channels':          IN_CHANNELS,
            'split':                '80/10/10',
            'seed':                 SEED
        },
        'results': {
            'best_val_pinball':   best_val_pinball,
            'best_epoch':         best_epoch,
            'total_epochs_run':   epoch + 1,
            'training_time_minutes': elapsed / 60
        },
        'history': log_history
    }
    with open(LOG_FILE, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training log saved to {LOG_FILE}")

    # -------------------------------------------------------------------
    # Calibration pass: run best model on val set, save val_predictions.npz
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CALIBRATION PASS (val set = CQR calibration set)")
    print("=" * 60)

    # Load best model weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"Loaded best model from epoch {best_epoch}")

    run_calibration_pass(model, val_loader, device, VAL_PRED_PATH)

    return best_val_pinball, best_epoch


if __name__ == '__main__':
    best_pinball, best_ep = train()
    print(f"\nDone. Best val pinball = {best_pinball:.4f} at epoch {best_ep}")
