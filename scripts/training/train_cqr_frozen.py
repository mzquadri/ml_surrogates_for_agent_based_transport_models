"""
CQR Frozen Backbone Training Script (Trial 11)
===============================================
Trains only the quantile head of PointNetTransfGATFrozenCQR.
The T8 backbone (point_net_conv_1, point_net_conv_2, gat_graph_layers) is
loaded from the T8 checkpoint and permanently frozen.

Key differences from Trial 10 (train_cqr.py):
  - Model:     PointNetTransfGATFrozenCQR (frozen backbone + new GATConv(64->2) head)
  - Optimizer: AdamW targets ONLY gat_quantile_head.parameters()
  - Data:      Reuses T8's saved scalers (no re-fitting); test_dl.pt from T8
  - No new scaler files are saved (T8's scalers are the ground truth)

Preprocessing rule (same as all CQR trials):
  Per-split fitted normalization, reusing T8's scalers:
    train split  <- T8's train_x_scaler.pkl / train_pos_scaler.pkl
    val split    <- T8's validation_x_scaler.pkl / validation_pos_scaler.pkl

T8 Hyperparameters (EXACT MATCH):
  - lr:                      0.0005
  - batch_size:              8
  - gradient_accumulation:   3
  - split:                   80/10/10  (same seed=42 -> same split as T8)
  - in_channels:             5
  - early_stopping_patience: 25
  - min_delta:               0.001
  - num_epochs:              1000
  - seed:                    42

CQR-specific:
  - loss:   PinballLoss(alpha=0.10)  [tau_lo=0.05, tau_hi=0.95]
  - output: (q_lo, q_hi) per node

Output directory:
  data/TR-C_Benchmarks/point_net_transf_gat_11th_trial_cqr_frozen/

IMPORTANT: T8 artifacts are NEVER modified by this script.

CQR reference: Romano, Patterson & Candes (2019) NeurIPS, arXiv:1905.03222
Reference impl: yromano/cqr (github.com/yromano/cqr)

Author: Mohd Zamin Quadri
"""

import os
import sys
import json
import copy
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# ---- Path setup ----
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ---- Hyperparameters (exact T8 values) ----
LR            = 0.0005
BATCH_SIZE    = 8
GRAD_ACCUM    = 3
NUM_EPOCHS    = 1000
PATIENCE      = 25
MIN_DELTA     = 0.001   # minimum improvement to reset patience counter
ALPHA         = 0.10    # tau_lo=0.05, tau_hi=0.95
SEED          = 42
USE_GRAD_CLIP = True

# ---- Paths ----
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'train_data', 'dist_not_connected_10k_1pct')
BASE_DIR     = os.path.join(PROJECT_ROOT, 'data', 'TR-C_Benchmarks')

# T8 (READ-ONLY source of backbone weights and scalers)
T8_EXP_NAME  = 'point_net_transf_gat_8th_trial_lower_dropout'
T8_EXP_DIR   = os.path.join(BASE_DIR, T8_EXP_NAME)
T8_DATA_DIR  = os.path.join(T8_EXP_DIR, 'data_created_during_training')
T8_MODEL_PATH = os.path.join(T8_EXP_DIR, 'trained_model', 'model.pth')
T8_TEST_DL_PATH = os.path.join(T8_DATA_DIR, 'test_dl.pt')

# Trial 11 output
EXP_NAME        = 'point_net_transf_gat_11th_trial_cqr_frozen'
EXP_DIR         = os.path.join(BASE_DIR, EXP_NAME)
MODEL_SAVE_PATH  = os.path.join(EXP_DIR, 'trained_model', 'model.pth')
DATA_SAVE_PATH   = os.path.join(EXP_DIR, 'data_created_during_training')
LOG_FILE         = os.path.join(EXP_DIR, 'training_log.json')
VAL_PRED_PATH    = os.path.join(EXP_DIR, 'val_predictions.npz')
CHECKPOINT_PATH  = os.path.join(EXP_DIR, 'checkpoint.pt')

# Node features (same as T8/T10)
NODE_FEATURES = [
    "VOL_BASE_CASE",
    "CAPACITY_BASE_CASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "LENGTH",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_r2(preds, targets):
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def save_checkpoint(path, epoch, model, optimizer, best_val_pinball,
                    patience_counter, best_epoch, log_history):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_pinball':     best_val_pinball,
        'patience_counter':     patience_counter,
        'best_epoch':           best_epoch,
        'log_history':          log_history,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return (
        ckpt['epoch'],
        ckpt['best_val_pinball'],
        ckpt['patience_counter'],
        ckpt.get('best_epoch', 0),
        ckpt['log_history'],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset():
    """Load all batch files in order (same as run_models.py / train_cqr.py)."""
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
    for data in datalist:
        data.num_nodes = data.x.shape[0]
    print(f"  Loaded {len(datalist)} graphs total.")
    return datalist


def split_dataset(datalist):
    """Deterministic 80/10/10 split (same seed as T8 via gnn_io)."""
    from gnn.gnn_io import split_into_subsets
    train_set, valid_set, test_set = split_into_subsets(
        dataset=datalist, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    return train_set, valid_set, test_set


# ---------------------------------------------------------------------------
# Normalization with T8's pre-fitted scalers (no re-fitting)
# ---------------------------------------------------------------------------

def apply_t8_scalers(dataset_subset, x_scaler, pos_scaler, batch_size=100):
    """
    Normalize a dataset subset using T8's pre-fitted scalers.
    x_scaler  : sklearn StandardScaler fitted on T8's split x features
    pos_scaler: sklearn StandardScaler fitted on T8's split pos features

    Returns a plain list of (deep-copied, normalized) Data objects.
    This mirrors train_cqr.py's normalize_and_build_loaders but uses
    transform() instead of fit_transform() -- no new scaler is fitted.
    """
    from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

    continuous_feat = [
        EdgeFeatures.VOL_BASE_CASE,
        EdgeFeatures.CAPACITY_BASE_CASE,
        EdgeFeatures.CAPACITY_REDUCTION,
        EdgeFeatures.FREESPEED,
        EdgeFeatures.LENGTH,
    ]
    node_feature_filter = [EdgeFeatures[f].value for f in NODE_FEATURES]

    data_list = [
        copy.deepcopy(dataset_subset.dataset[idx])
        for idx in dataset_subset.indices
    ]

    num_nodes = data_list[0].x.shape[0]

    # Apply x_scaler (transform only, scaler already fitted on T8's split)
    for i in tqdm(range(0, len(data_list), batch_size), desc="  Normalizing x", leave=False):
        batch = data_list[i:i + batch_size]
        batch_x = np.vstack([d.x[:, continuous_feat].numpy() for d in batch])
        batch_x_norm = x_scaler.transform(batch_x)
        for j, d in enumerate(batch):
            d.x[:, continuous_feat] = torch.tensor(
                batch_x_norm[j * num_nodes:(j + 1) * num_nodes],
                dtype=d.x.dtype,
            )

    # Filter to the 5 used features
    for d in data_list:
        d.x = d.x[:, node_feature_filter]

    # Apply pos_scaler (transform only)
    for d in data_list:
        n = d.pos.shape[0]
        pos_flat = d.pos.numpy().reshape(-1, 6)
        pos_norm = pos_scaler.transform(pos_flat)
        d.pos = torch.tensor(pos_norm.reshape(n, 3, 2), dtype=d.pos.dtype)

    return data_list


def build_loaders(train_set, valid_set, t8_data_dir):
    """
    Build train and val DataLoaders using T8's saved scalers.
    Returns (train_loader, val_loader).
    """
    import joblib
    from gnn.gnn_io import collate_fn

    print("  Loading T8 scalers...")
    train_x_scaler   = joblib.load(os.path.join(t8_data_dir, 'train_x_scaler.pkl'))
    train_pos_scaler = joblib.load(os.path.join(t8_data_dir, 'train_pos_scaler.pkl'))
    val_x_scaler     = joblib.load(os.path.join(t8_data_dir, 'validation_x_scaler.pkl'))
    val_pos_scaler   = joblib.load(os.path.join(t8_data_dir, 'validation_pos_scaler.pkl'))
    print("  T8 scalers loaded.")

    print("  Normalizing train set with T8 train scalers...")
    train_norm = apply_t8_scalers(train_set, train_x_scaler, train_pos_scaler)
    print(f"  Train normalized: {len(train_norm)} graphs.")

    print("  Normalizing val set with T8 validation scalers...")
    val_norm = apply_t8_scalers(valid_set, val_x_scaler, val_pos_scaler)
    print(f"  Val normalized: {len(val_norm)} graphs.")

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset=dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=0, collate_fn=collate_fn,
        )

    train_loader = make_loader(train_norm, shuffle=True)
    val_loader   = make_loader(val_norm,   shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loader, loss_fn, device):
    """
    CQR validation loop.
    Primary metric: val_pinball (for best-model selection).
    Also reports R^2 from midpoints (monitoring only).
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

            midpoint = (q_lo.squeeze() + q_hi.squeeze()) / 2.0
            all_midpoints.append(midpoint.cpu().numpy())
            all_targets.append(target.squeeze().cpu().numpy())

    all_midpoints = np.concatenate(all_midpoints)
    all_targets   = np.concatenate(all_targets)

    val_pinball = total_pinball / n_batches if n_batches > 0 else float('inf')
    r2          = compute_r2(all_midpoints, all_targets)
    spearman_r, _ = spearmanr(all_midpoints, all_targets)
    pearson_r,  _ = pearsonr(all_midpoints,  all_targets)

    return val_pinball, r2, float(spearman_r), float(pearson_r)


# ---------------------------------------------------------------------------
# Calibration pass
# ---------------------------------------------------------------------------

def run_calibration_pass(model, val_loader, device, save_path):
    """
    Run inference over val set (= CQR calibration set) with the best model.
    Applies min/max monotonicity ordering (LearnerOptimizedCrossing.predict
    from yromano/cqr).
    Saves val_predictions.npz with keys: q_lo, q_hi, targets (all 1D).
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

    # Monotonicity ordering (LearnerOptimizedCrossing.predict in yromano/cqr)
    q_lo_ord = np.minimum(all_q_lo, all_q_hi)
    q_hi_ord = np.maximum(all_q_lo, all_q_hi)

    np.savez(save_path, q_lo=q_lo_ord, q_hi=q_hi_ord, targets=all_targets)

    n_crossings = int(np.sum(all_q_lo > all_q_hi))
    print(f"  Calibration pass: {len(all_targets)} samples")
    print(f"  Monotonicity crossings corrected: {n_crossings} / {len(all_q_lo)}")
    print(f"  q_lo range: [{q_lo_ord.min():.4f}, {q_lo_ord.max():.4f}]")
    print(f"  q_hi range: [{q_hi_ord.min():.4f}, {q_hi_ord.max():.4f}]")
    print(f"  targets range: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train():
    set_seeds(SEED)

    # Create output directories
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)

    print("=" * 70)
    print("CQR FROZEN BACKBONE TRAINING (Trial 11)")
    print(f"  Experiment: {EXP_NAME}")
    print(f"  T8 backbone: {T8_MODEL_PATH}")
    print("=" * 70)
    print(f"  lr={LR}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"  alpha={ALPHA} (tau_lo={ALPHA/2}, tau_hi={1-ALPHA/2})")
    print(f"  max_epochs={NUM_EPOCHS}, patience={PATIENCE}")
    print()

    # Device
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

    # ---- Load and split dataset ----
    print("\nLoading dataset...")
    datalist = load_dataset()

    print("Splitting dataset (80/10/10, seed=42)...")
    train_set, valid_set, test_set = split_dataset(datalist)
    print(f"  Train: {len(train_set.indices)}, Val: {len(valid_set.indices)}, Test: {len(test_set.indices)}")

    # ---- Build loaders using T8 scalers ----
    print("\nBuilding data loaders (reusing T8 scalers)...")
    train_loader, val_loader = build_loaders(train_set, valid_set, T8_DATA_DIR)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Model ----
    print("\nBuilding PointNetTransfGATFrozenCQR...")
    from gnn.models.point_net_transf_gat_frozen_cqr import PointNetTransfGATFrozenCQR
    model = PointNetTransfGATFrozenCQR(
        t8_model_path=T8_MODEL_PATH,
        device=device,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {n_trainable:,} / {n_total:,} total")

    # ---- Loss, optimizer (HEAD ONLY) ----
    from gnn.losses.quantile_loss import PinballLoss
    loss_fn   = PinballLoss(alpha=ALPHA)
    optimizer = torch.optim.AdamW(
        model.gat_quantile_head.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    # LR scheduler (same as T8: linear warmup + cosine decay)
    from gnn.help_functions import LinearWarmupCosineDecayScheduler
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler   = LinearWarmupCosineDecayScheduler(
        initial_lr=LR, total_steps=total_steps
    )

    # AMP scaler (disabled on CPU: GradScaler/autocast not meaningful without GPU)
    use_amp = (amp_device_type != 'cpu')
    if use_amp:
        grad_scaler = torch.amp.GradScaler(amp_device_type)
    else:
        grad_scaler = None

    # ---- Training loop ----
    best_val_pinball = float('inf')
    best_epoch       = 0
    patience_counter = 0
    log_history      = []
    start_epoch      = 0

    # Resume from checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nCheckpoint found: {CHECKPOINT_PATH}")
        start_epoch, best_val_pinball, patience_counter, best_epoch, log_history = \
            load_checkpoint(CHECKPOINT_PATH, model, optimizer, device)
        print(f"  Resumed from epoch {start_epoch}, best_val_pinball={best_val_pinball:.4f}, "
              f"patience_counter={patience_counter}")
    else:
        print("\nNo checkpoint found, starting from epoch 1.")

    print("\nStarting training...\n")
    train_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()   # backbone stays in eval() via overridden train()
        optimizer.zero_grad()
        epoch_train_loss = 0.0
        n_train_batches  = 0

        for idx, data in enumerate(tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            leave=False,
            disable=True,
        )):
            step = epoch * len(train_loader) + idx
            lr = scheduler.get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            data   = data.to(device)
            target = data.y.to(device)

            if use_amp:
                with torch.amp.autocast(amp_device_type):
                    q_lo, q_hi = model(data)
                    loss = loss_fn(q_lo, q_hi, target)
            else:
                q_lo, q_hi = model(data)
                loss = loss_fn(q_lo, q_hi, target)

            epoch_train_loss += loss.item()
            n_train_batches  += 1

            if use_amp:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (idx + 1) % GRAD_ACCUM == 0:
                if use_amp:
                    if USE_GRAD_CLIP:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.gat_quantile_head.parameters(), max_norm=1.0
                        )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    if USE_GRAD_CLIP:
                        torch.nn.utils.clip_grad_norm_(
                            model.gat_quantile_head.parameters(), max_norm=1.0
                        )
                    optimizer.step()
                optimizer.zero_grad()

        # Handle remaining accumulation steps
        if len(train_loader) % GRAD_ACCUM != 0:
            if use_amp:
                if USE_GRAD_CLIP:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.gat_quantile_head.parameters(), max_norm=1.0
                    )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                if USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(
                        model.gat_quantile_head.parameters(), max_norm=1.0
                    )
                optimizer.step()
            optimizer.zero_grad()

        avg_train_pinball = (
            epoch_train_loss / n_train_batches if n_train_batches > 0 else float('inf')
        )

        # Validation
        val_pinball, r2, spearman_r, pearson_r = validate(
            model, val_loader, loss_fn, device
        )

        row = {
            'epoch':         epoch + 1,
            'train_pinball': float(avg_train_pinball),
            'val_pinball':   float(val_pinball),
            'r2_midpoint':   float(r2),
            'spearman':      float(spearman_r),
            'pearson':       float(pearson_r),
            'lr':            float(lr),
        }
        log_history.append(row)

        print(
            f"Epoch {epoch+1:4d} | train_pinball={avg_train_pinball:.4f} | "
            f"val_pinball={val_pinball:.4f} | r2_mid={r2:.4f} | "
            f"spearman={spearman_r:.4f} | lr={lr:.6f}",
            flush=True,
        )

        # Save best model (full state dict: backbone + head)
        if val_pinball < best_val_pinball - MIN_DELTA:
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

        # Save checkpoint at end of every epoch (overwrites previous)
        save_checkpoint(
            CHECKPOINT_PATH, epoch + 1, model, optimizer,
            best_val_pinball, patience_counter, best_epoch, log_history,
        )

    elapsed = time.time() - train_start
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")
    print(f"Best val_pinball={best_val_pinball:.4f} at epoch {best_epoch}")

    # Remove checkpoint (training finished cleanly)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint removed (training complete).")

    # ---- Save training log ----
    training_summary = {
        'experiment':     EXP_NAME,
        'trial':          11,
        'strategy':       'frozen_backbone_cqr',
        't8_backbone':    T8_MODEL_PATH,
        't8_test_dl':     T8_TEST_DL_PATH,
        'hyperparameters': {
            'lr':                   float(LR),
            'batch_size':           BATCH_SIZE,
            'gradient_accumulation': GRAD_ACCUM,
            'num_epochs':           NUM_EPOCHS,
            'patience':             PATIENCE,
            'alpha':                float(ALPHA),
            'tau_lo':               float(ALPHA / 2.0),
            'tau_hi':               float(1.0 - ALPHA / 2.0),
            'in_channels':          5,
            'split':                '80/10/10',
            'seed':                 SEED,
            'optimizer_scope':      'gat_quantile_head only',
        },
        'results': {
            'best_val_pinball':        float(best_val_pinball),
            'best_epoch':              best_epoch,
            'total_epochs_run':        epoch + 1,
            'training_time_minutes':   float(elapsed / 60),
        },
        'history': log_history,
    }
    with open(LOG_FILE, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training log saved to {LOG_FILE}")

    # ---- Calibration pass ----
    print("\n" + "=" * 70)
    print("CALIBRATION PASS (val set = CQR calibration set)")
    print("=" * 70)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"Loaded best model from epoch {best_epoch}")

    run_calibration_pass(model, val_loader, device, VAL_PRED_PATH)

    return float(best_val_pinball), best_epoch


if __name__ == '__main__':
    best_pinball, best_ep = train()
    print(f"\nDone. Best val pinball = {best_pinball:.4f} at epoch {best_ep}")
