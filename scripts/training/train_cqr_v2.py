"""
CQR Pretrained Backbone Training Script v2 (Trial 10)
======================================================
Fine-tunes PointNetTransfGATQuantile starting from T8 pretrained weights.
Backbone is UNFROZEN (differential LR: backbone 5e-5, head 5e-4).

Root causes of T10-v1 failure (R2=0.315) fixed here:
  1. Start from T8 pretrained weights (not random init) -- backbone already
     learned MSE-quality features; pinball loss only refines quantile head
  2. Multi-task loss: alpha*MSE(midpoint,y) + (1-alpha)*PinballLoss -- MSE
     anchor prevents backbone from drifting away from mean prediction quality
  3. Non-crossing soft penalty: lambda_cross * mean(relu(q_lo - q_hi))
  4. T8 scalers reused -- no data leakage from re-fitting
  5. MIN_DELTA=0.001 + patience=25 prevents overfit gap (val=0.589, test=0.315)
  6. Differential LR: backbone at 5e-5 (fine-tune), head at 5e-4 (learn new)

Architecture:
  - Model: PointNetTransfGATQuantile (existing class, GATConv(64->2) head)
  - Weight loading: T8 backbone keys loaded; gat_final.* skipped (shape mismatch)
  - Optimizer: backbone params lr=5e-5, gat_final params lr=5e-4

Multi-task loss:
  ALPHA_MSE * MSE(midpoint, y) + (1-ALPHA_MSE) * PinballLoss(q_lo, q_hi, y)
  + LAMBDA_CROSS * mean(relu(q_lo - q_hi))  [non-crossing penalty]

  ALPHA_MSE = 0.3  (30% MSE anchor, 70% pinball)
  LAMBDA_CROSS = 1.0

Calibration pass:
  After training, runs val-set inference and saves val_predictions.npz
  (required by evaluate_cqr.py).

Output directory:
  data/TR-C_Benchmarks/point_net_transf_gat_10th_trial_cqr/

T8 artifacts are NEVER modified by this script.

CQR reference: Romano, Patterson & Candes (2019) NeurIPS, arXiv:1905.03222

Author: Mohd Zamin Quadri
"""

import os
import sys
import json
import copy
import time
import shutil
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

# ---- Hyperparameters ----
LR_BACKBONE   = 5e-5    # fine-tune backbone (10x smaller than head)
LR_HEAD       = 5e-4    # train new quantile head
BATCH_SIZE    = 8
GRAD_ACCUM    = 3
NUM_EPOCHS    = 500
PATIENCE      = 25
MIN_DELTA     = 0.001   # minimum improvement in val_pinball to reset patience
ALPHA_PINBALL = 0.10    # tau_lo=0.05, tau_hi=0.95
ALPHA_MSE     = 0.3     # weight of MSE anchor in multi-task loss
LAMBDA_CROSS  = 1.0     # weight of non-crossing soft penalty
SEED          = 42
USE_GRAD_CLIP = True

# ---- Paths ----
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'train_data', 'dist_not_connected_10k_1pct')
BASE_DIR     = os.path.join(PROJECT_ROOT, 'data', 'TR-C_Benchmarks')

# T8 (READ-ONLY source of backbone weights and scalers)
T8_EXP_NAME   = 'point_net_transf_gat_8th_trial_lower_dropout'
T8_EXP_DIR    = os.path.join(BASE_DIR, T8_EXP_NAME)
T8_DATA_DIR   = os.path.join(T8_EXP_DIR, 'data_created_during_training')
T8_MODEL_PATH = os.path.join(T8_EXP_DIR, 'trained_model', 'model.pth')
T8_TEST_DL_PATH = os.path.join(T8_DATA_DIR, 'test_dl.pt')

# Trial 10 output (OVERWRITES old T10 results)
EXP_NAME        = 'point_net_transf_gat_10th_trial_cqr'
EXP_DIR         = os.path.join(BASE_DIR, EXP_NAME)
MODEL_SAVE_PATH  = os.path.join(EXP_DIR, 'trained_model', 'model.pth')
DATA_SAVE_PATH   = os.path.join(EXP_DIR, 'data_created_during_training')
LOG_FILE         = os.path.join(EXP_DIR, 'training_log.json')
VAL_PRED_PATH    = os.path.join(EXP_DIR, 'val_predictions.npz')
CHECKPOINT_PATH  = os.path.join(EXP_DIR, 'checkpoint.pt')

# Node features (same as T8)
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
    """Load all batch files in order (same as T8/T11)."""
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
    Uses transform() -- no new scaler is fitted.
    Mirrors train_cqr_frozen.py exactly.
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

    # Apply x_scaler (transform only)
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
# Load T8 weights into PointNetTransfGATQuantile (backbone only)
# ---------------------------------------------------------------------------

def load_t8_backbone_weights(model, t8_model_path, device):
    """
    Load T8 checkpoint weights into the quantile model backbone.

    Strategy:
      - Load T8 state_dict
      - Filter OUT keys starting with 'gat_final.*' (T8: GATConv(64,1),
        T10 needs GATConv(64,2) -- shape mismatch, skip these)
      - Load remaining backbone keys with strict=False

    No PyG key remapping needed for backbone keys (already in correct format
    for PyG 2.3.1). gat_final.* keys are skipped entirely so no remapping
    is required for them either.
    """
    t8_state = torch.load(t8_model_path, map_location=device, weights_only=False)

    # Filter out gat_final keys (shape mismatch: T8 output 1 channel, T10 needs 2)
    backbone_state = {
        k: v for k, v in t8_state.items()
        if not k.startswith('gat_final.')
    }

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)

    # Expected missing: gat_final.* (new head, randomly initialized)
    bad_missing = [k for k in missing if not k.startswith('gat_final.')]
    if bad_missing:
        raise RuntimeError(
            f"Backbone keys NOT loaded from T8 checkpoint: {bad_missing}"
        )

    # Expected unexpected: none (we already filtered gat_final.*)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading T8 backbone: {unexpected}"
        )

    n_loaded  = len(backbone_state)
    n_new_head = len(missing)
    print(f"  T8 backbone loaded: {n_loaded} key tensors from {t8_model_path}")
    print(f"  New gat_final keys (randomly initialised): {n_new_head}")


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

class MultiTaskCQRLoss(nn.Module):
    """
    Multi-task loss for CQR fine-tuning.

    total = ALPHA_MSE * MSE(midpoint, y)
          + (1 - ALPHA_MSE) * PinballLoss(q_lo, q_hi, y)
          + LAMBDA_CROSS * mean(relu(q_lo - q_hi))

    The MSE anchor on the midpoint prevents the backbone from losing the
    mean-prediction quality it learned in T8. The non-crossing penalty
    encourages q_lo <= q_hi without hard ordering (which would zero gradients).
    """

    def __init__(self, alpha_mse, pinball_loss_fn, lambda_cross):
        super().__init__()
        self.alpha_mse      = alpha_mse
        self.pinball_loss_fn = pinball_loss_fn
        self.lambda_cross   = lambda_cross
        self.mse_fn         = nn.MSELoss()

    def forward(self, q_lo, q_hi, target):
        q_lo = q_lo.squeeze()
        q_hi = q_hi.squeeze()
        target = target.squeeze()

        midpoint = (q_lo + q_hi) / 2.0
        mse_loss = self.mse_fn(midpoint, target)

        pinball_loss = self.pinball_loss_fn(
            q_lo.unsqueeze(-1), q_hi.unsqueeze(-1), target
        )

        # Non-crossing soft penalty: relu(q_lo - q_hi) > 0 only when q_lo > q_hi
        crossing_penalty = torch.mean(torch.relu(q_lo - q_hi))

        total = (
            self.alpha_mse * mse_loss
            + (1.0 - self.alpha_mse) * pinball_loss
            + self.lambda_cross * crossing_penalty
        )
        return total, mse_loss, pinball_loss, crossing_penalty


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loader, loss_fn_pinball, device):
    """
    CQR validation loop.
    Primary metric: val_pinball (for best-model selection).
    Also reports R2 from midpoints (monitoring only).
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
            loss = loss_fn_pinball(q_lo, q_hi, target)

            total_pinball += loss.item()
            n_batches     += 1

            midpoint = (q_lo.squeeze() + q_hi.squeeze()) / 2.0
            all_midpoints.append(midpoint.cpu().numpy())
            all_targets.append(target.squeeze().cpu().numpy())

    all_midpoints = np.concatenate(all_midpoints)
    all_targets   = np.concatenate(all_targets)

    val_pinball   = total_pinball / n_batches if n_batches > 0 else float('inf')
    r2            = compute_r2(all_midpoints, all_targets)
    spearman_r, _ = spearmanr(all_midpoints, all_targets)
    pearson_r,  _ = pearsonr(all_midpoints,  all_targets)

    return val_pinball, r2, float(spearman_r), float(pearson_r)


# ---------------------------------------------------------------------------
# Calibration pass
# ---------------------------------------------------------------------------

def run_calibration_pass(model, val_loader, device, save_path):
    """
    Run inference over val set (= CQR calibration set) with the best model.
    Applies min/max monotonicity ordering.
    Saves val_predictions.npz with keys: q_lo, q_hi, targets (all 1D).
    Required by evaluate_cqr.py.
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
    print("CQR PRETRAINED BACKBONE TRAINING v2 (Trial 10)")
    print(f"  Experiment: {EXP_NAME}")
    print(f"  T8 backbone: {T8_MODEL_PATH}")
    print("=" * 70)
    print(f"  lr_backbone={LR_BACKBONE}, lr_head={LR_HEAD}")
    print(f"  batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"  alpha_mse={ALPHA_MSE}, alpha_pinball={ALPHA_PINBALL}, lambda_cross={LAMBDA_CROSS}")
    print(f"  max_epochs={NUM_EPOCHS}, patience={PATIENCE}, min_delta={MIN_DELTA}")
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
    print("\nBuilding PointNetTransfGATQuantile (loading T8 backbone weights)...")
    from gnn.models.point_net_transf_gat_quantile import PointNetTransfGATQuantile
    model = PointNetTransfGATQuantile(
        in_channels=5,
        out_channels=2,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=0.2,
        use_dropout=True,
        log_to_wandb=False,
    )

    # Load T8 backbone weights (gat_final skipped due to shape mismatch)
    load_t8_backbone_weights(model, T8_MODEL_PATH, device)
    model = model.to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_total:,}  Trainable: {n_trainable:,}")

    # ---- Loss functions ----
    from gnn.losses.quantile_loss import PinballLoss
    pinball_loss_fn = PinballLoss(alpha=ALPHA_PINBALL)
    multi_task_loss = MultiTaskCQRLoss(
        alpha_mse=ALPHA_MSE,
        pinball_loss_fn=pinball_loss_fn,
        lambda_cross=LAMBDA_CROSS,
    )

    # ---- Optimizer with differential LR ----
    # Backbone: all params NOT in gat_final
    # Head: gat_final params
    backbone_params = [
        p for n, p in model.named_parameters()
        if not n.startswith('gat_final.')
    ]
    head_params = list(model.gat_final.parameters())

    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': LR_BACKBONE},
            {'params': head_params,     'lr': LR_HEAD},
        ],
        weight_decay=1e-4,
    )

    n_backbone_params = sum(p.numel() for p in backbone_params)
    n_head_params     = sum(p.numel() for p in head_params)
    print(f"  Backbone params (lr={LR_BACKBONE}): {n_backbone_params:,}")
    print(f"  Head params     (lr={LR_HEAD}):     {n_head_params:,}")

    # ---- LR scheduler ----
    # LinearWarmupCosineDecayScheduler scales by initial_lr.
    # We apply it multiplicatively: backbone gets LR_BACKBONE * scale,
    # head gets LR_HEAD * scale.
    from gnn.help_functions import LinearWarmupCosineDecayScheduler
    total_steps = NUM_EPOCHS * len(train_loader)
    # Two schedulers -- one per param group
    sched_backbone = LinearWarmupCosineDecayScheduler(
        initial_lr=LR_BACKBONE, total_steps=total_steps
    )
    sched_head = LinearWarmupCosineDecayScheduler(
        initial_lr=LR_HEAD, total_steps=total_steps
    )

    # AMP scaler
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

    epoch = start_epoch  # ensure defined for training_summary if loop does not run
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        epoch_train_loss    = 0.0
        epoch_mse_loss      = 0.0
        epoch_pinball_loss  = 0.0
        epoch_cross_penalty = 0.0
        n_train_batches     = 0

        for idx, data in enumerate(tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            leave=False,
            disable=True,
        )):
            step = epoch * len(train_loader) + idx
            lr_bb   = sched_backbone.get_lr(step)
            lr_head = sched_head.get_lr(step)
            optimizer.param_groups[0]['lr'] = lr_bb
            optimizer.param_groups[1]['lr'] = lr_head

            data   = data.to(device)
            target = data.y.to(device)

            if use_amp:
                with torch.amp.autocast(amp_device_type):
                    q_lo, q_hi = model(data)
                    total, mse, pb, cross = multi_task_loss(q_lo, q_hi, target)
            else:
                q_lo, q_hi = model(data)
                total, mse, pb, cross = multi_task_loss(q_lo, q_hi, target)

            epoch_train_loss    += total.item()
            epoch_mse_loss      += mse.item()
            epoch_pinball_loss  += pb.item()
            epoch_cross_penalty += cross.item()
            n_train_batches     += 1

            if use_amp:
                grad_scaler.scale(total).backward()
            else:
                total.backward()

            if (idx + 1) % GRAD_ACCUM == 0:
                if use_amp:
                    if USE_GRAD_CLIP:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0
                        )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    if USE_GRAD_CLIP:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0
                        )
                    optimizer.step()
                optimizer.zero_grad()

        # Handle remaining accumulation steps
        if len(train_loader) % GRAD_ACCUM != 0:
            if use_amp:
                if USE_GRAD_CLIP:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                if USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        avg_train_loss   = epoch_train_loss    / n_train_batches if n_train_batches > 0 else float('inf')
        avg_mse          = epoch_mse_loss      / n_train_batches if n_train_batches > 0 else float('inf')
        avg_pinball      = epoch_pinball_loss  / n_train_batches if n_train_batches > 0 else float('inf')
        avg_cross        = epoch_cross_penalty / n_train_batches if n_train_batches > 0 else float('inf')

        # Validation (pure pinball loss for selection metric)
        val_pinball, r2, spearman_r, pearson_r = validate(
            model, val_loader, pinball_loss_fn, device
        )

        row = {
            'epoch':         epoch + 1,
            'train_total':   float(avg_train_loss),
            'train_mse':     float(avg_mse),
            'train_pinball': float(avg_pinball),
            'train_cross':   float(avg_cross),
            'val_pinball':   float(val_pinball),
            'r2_midpoint':   float(r2),
            'spearman':      float(spearman_r),
            'pearson':       float(pearson_r),
            'lr_backbone':   float(lr_bb),
            'lr_head':       float(lr_head),
        }
        log_history.append(row)

        print(
            f"Epoch {epoch+1:4d} | total={avg_train_loss:.4f} "
            f"mse={avg_mse:.4f} pb={avg_pinball:.4f} cross={avg_cross:.4f} | "
            f"val_pb={val_pinball:.4f} | r2_mid={r2:.4f} | "
            f"lr_bb={lr_bb:.2e} lr_hd={lr_head:.2e}",
            flush=True,
        )

        # Save best model (full state dict)
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

        # Save checkpoint at end of every epoch
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
        'trial':          10,
        'version':        'v2_pretrained_backbone',
        'strategy':       'pretrained_backbone_cqr_multitask',
        't8_backbone':    T8_MODEL_PATH,
        't8_test_dl':     T8_TEST_DL_PATH,
        'fixes_applied': [
            'pretrained_backbone_not_random_init',
            'multitask_loss_mse_anchor',
            'non_crossing_soft_penalty',
            'reuse_t8_scalers_no_data_leakage',
            'min_delta_0.001_prevents_overfit_gap',
            'differential_lr_backbone_5e-5_head_5e-4',
        ],
        'hyperparameters': {
            'lr_backbone':           float(LR_BACKBONE),
            'lr_head':               float(LR_HEAD),
            'batch_size':            BATCH_SIZE,
            'gradient_accumulation': GRAD_ACCUM,
            'num_epochs':            NUM_EPOCHS,
            'patience':              PATIENCE,
            'min_delta':             float(MIN_DELTA),
            'alpha_pinball':         float(ALPHA_PINBALL),
            'tau_lo':                float(ALPHA_PINBALL / 2.0),
            'tau_hi':                float(1.0 - ALPHA_PINBALL / 2.0),
            'alpha_mse':             float(ALPHA_MSE),
            'lambda_cross':          float(LAMBDA_CROSS),
            'in_channels':           5,
            'split':                 '80/10/10',
            'seed':                  SEED,
            'optimizer_scope':       'full model (backbone + head, differential lr)',
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

    # ---- Copy T8 test_dl.pt to T10 data dir ----
    # evaluate_cqr.py looks for test_dl.pt in T10's data_created_during_training/
    t10_test_dl_dest = os.path.join(DATA_SAVE_PATH, 'test_dl.pt')
    if not os.path.exists(t10_test_dl_dest):
        print(f"\nCopying T8 test_dl.pt -> {t10_test_dl_dest}")
        shutil.copy2(T8_TEST_DL_PATH, t10_test_dl_dest)
        print("  Done.")
    else:
        print(f"\ntest_dl.pt already exists at {t10_test_dl_dest}, skipping copy.")

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
