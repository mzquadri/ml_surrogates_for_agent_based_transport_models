"""
Heteroscedastic GNN Training Script v2 (Trial 9 -- Frozen Backbone)
====================================================================
Trains ONLY the heteroscedastic head of PointNetTransfGATFrozenHeteroscedastic.
The T8 backbone is loaded once and permanently frozen.

Root causes of T9-v1 failure (R2=0.02) fixed here:
  1. Frozen backbone (no variance inflation via backbone NLL collapse)
  2. Stronger variance regularization: lambda=0.1 (was 0.01)
  3. T8 scalers reused -- no data leakage from re-fitting
  4. MIN_DELTA=0.001 guard prevents premature stopping

Architecture:
  - Backbone: T8 (point_net_conv_1, point_net_conv_2, gat_graph_layers) -- FROZEN
  - Head:     gat_heteroscedastic_head GATConv(64->2) -- TRAINABLE only
  - Output:   (mu, log_var) per node, shape [N,1] each
  - Backbone dropout STAYS ON during train() for MC Dropout (epistemic UQ)

Key differences from train_cqr_frozen.py (T11):
  - Model: PointNetTransfGATFrozenHeteroscedastic (head name: gat_heteroscedastic_head)
  - Loss:  HeteroscedasticGaussianLoss(var_reg_lambda=0.1, var_reg_type='log')
  - Validation metric: val_nll (primary), r2 from means (monitoring)
  - No calibration pass needed (heteroscedastic eval uses MC Dropout, not CQR)

Hyperparameters:
  - lr=5e-4, batch_size=8, grad_accum=3, max_epochs=1000, patience=25, min_delta=0.001
  - seed=42 (same 80/10/10 split as T8)

Output directory:
  data/TR-C_Benchmarks/point_net_transf_gat_9th_trial_heteroscedastic/

T8 artifacts are NEVER modified by this script.

References:
  - Kendall & Gal (2017) NIPS: "What Uncertainties Do We Need in Bayesian DL?"
  - Seitzer (2022): "On the Pitfalls of Heteroscedastic Uncertainty Estimation"

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

# ---- Hyperparameters ----
LR            = 5e-4
BATCH_SIZE    = 8
GRAD_ACCUM    = 3
NUM_EPOCHS    = 1000
PATIENCE      = 25
MIN_DELTA     = 0.001   # minimum improvement in NLL to reset patience
VAR_REG_LAMBDA = 0.1    # stronger than T9-v1 (was 0.01) -- prevents variance inflation
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

# Trial 9 output (OVERWRITES old T9 results)
EXP_NAME        = 'point_net_transf_gat_9th_trial_heteroscedastic'
EXP_DIR         = os.path.join(BASE_DIR, EXP_NAME)
MODEL_SAVE_PATH  = os.path.join(EXP_DIR, 'trained_model', 'model.pth')
DATA_SAVE_PATH   = os.path.join(EXP_DIR, 'data_created_during_training')
LOG_FILE         = os.path.join(EXP_DIR, 'training_log.json')
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


def save_checkpoint(path, epoch, model, optimizer, best_val_nll,
                    patience_counter, best_epoch, log_history):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_nll':         best_val_nll,
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
        ckpt['best_val_nll'],
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
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loader, loss_fn, device):
    """
    Heteroscedastic validation loop.
    Primary metric: val_nll (for best-model selection).
    Also reports R2 from means (monitoring only).
    """
    model.eval()
    total_nll = 0.0
    n_batches  = 0
    all_means  = []
    all_targets = []

    with torch.no_grad():
        for data in val_loader:
            data   = data.to(device)
            target = data.y.to(device)

            mu, log_var = model(data)
            loss = loss_fn(mu, log_var, target)

            total_nll += loss.item()
            n_batches  += 1

            all_means.append(mu.squeeze().cpu().numpy())
            all_targets.append(target.squeeze().cpu().numpy())

    all_means   = np.concatenate(all_means)
    all_targets = np.concatenate(all_targets)

    val_nll    = total_nll / n_batches if n_batches > 0 else float('inf')
    r2         = compute_r2(all_means, all_targets)
    spearman_r, _ = spearmanr(all_means, all_targets)
    pearson_r,  _ = pearsonr(all_means,  all_targets)

    return val_nll, r2, float(spearman_r), float(pearson_r)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train():
    set_seeds(SEED)

    # Create output directories
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)

    print("=" * 70)
    print("HETEROSCEDASTIC FROZEN BACKBONE TRAINING v2 (Trial 9)")
    print(f"  Experiment: {EXP_NAME}")
    print(f"  T8 backbone: {T8_MODEL_PATH}")
    print("=" * 70)
    print(f"  lr={LR}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"  var_reg_lambda={VAR_REG_LAMBDA} (stronger regularization vs T9-v1)")
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
    print("\nBuilding PointNetTransfGATFrozenHeteroscedastic...")
    from gnn.models.point_net_transf_gat_frozen_heteroscedastic import (
        PointNetTransfGATFrozenHeteroscedastic,
    )
    model = PointNetTransfGATFrozenHeteroscedastic(
        t8_model_path=T8_MODEL_PATH,
        device=device,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {n_trainable:,} / {n_total:,} total")

    # ---- Loss, optimizer (HEAD ONLY) ----
    from gnn.losses.heteroscedastic_loss import HeteroscedasticGaussianLoss
    loss_fn   = HeteroscedasticGaussianLoss(var_reg_lambda=VAR_REG_LAMBDA, var_reg_type='log')
    optimizer = torch.optim.AdamW(
        model.gat_heteroscedastic_head.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    # LR scheduler (same as T8: linear warmup + cosine decay)
    from gnn.help_functions import LinearWarmupCosineDecayScheduler
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler   = LinearWarmupCosineDecayScheduler(
        initial_lr=LR, total_steps=total_steps
    )

    # AMP scaler
    use_amp = (amp_device_type != 'cpu')
    if use_amp:
        grad_scaler = torch.amp.GradScaler(amp_device_type)
    else:
        grad_scaler = None

    # ---- Training loop ----
    best_val_nll     = float('inf')
    best_epoch       = 0
    patience_counter = 0
    log_history      = []
    start_epoch      = 0

    # Resume from checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nCheckpoint found: {CHECKPOINT_PATH}")
        start_epoch, best_val_nll, patience_counter, best_epoch, log_history = \
            load_checkpoint(CHECKPOINT_PATH, model, optimizer, device)
        print(f"  Resumed from epoch {start_epoch}, best_val_nll={best_val_nll:.4f}, "
              f"patience_counter={patience_counter}")
    else:
        print("\nNo checkpoint found, starting from epoch 1.")

    print("\nStarting training...\n")
    train_start = time.time()

    epoch = start_epoch  # ensure defined for training_summary if loop does not run
    for epoch in range(start_epoch, NUM_EPOCHS):
        # model.train() enables backbone dropout (required for MC Dropout later)
        # Note: backbone weights are still frozen (requires_grad=False)
        model.train()
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
                    mu, log_var = model(data)
                    loss = loss_fn(mu, log_var, target)
            else:
                mu, log_var = model(data)
                loss = loss_fn(mu, log_var, target)

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
                            model.gat_heteroscedastic_head.parameters(), max_norm=1.0
                        )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    if USE_GRAD_CLIP:
                        torch.nn.utils.clip_grad_norm_(
                            model.gat_heteroscedastic_head.parameters(), max_norm=1.0
                        )
                    optimizer.step()
                optimizer.zero_grad()

        # Handle remaining accumulation steps
        if len(train_loader) % GRAD_ACCUM != 0:
            if use_amp:
                if USE_GRAD_CLIP:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.gat_heteroscedastic_head.parameters(), max_norm=1.0
                    )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                if USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(
                        model.gat_heteroscedastic_head.parameters(), max_norm=1.0
                    )
                optimizer.step()
            optimizer.zero_grad()

        avg_train_nll = (
            epoch_train_loss / n_train_batches if n_train_batches > 0 else float('inf')
        )

        # Validation
        val_nll, r2, spearman_r, pearson_r = validate(
            model, val_loader, loss_fn, device
        )

        row = {
            'epoch':      epoch + 1,
            'train_nll':  float(avg_train_nll),
            'val_nll':    float(val_nll),
            'r2_mean':    float(r2),
            'spearman':   float(spearman_r),
            'pearson':    float(pearson_r),
            'lr':         float(lr),
        }
        log_history.append(row)

        print(
            f"Epoch {epoch+1:4d} | train_nll={avg_train_nll:.4f} | "
            f"val_nll={val_nll:.4f} | r2_mean={r2:.4f} | "
            f"spearman={spearman_r:.4f} | lr={lr:.6f}",
            flush=True,
        )

        # Save best model
        if val_nll < best_val_nll - MIN_DELTA:
            best_val_nll = val_nll
            best_epoch   = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [BEST] Model saved (val_nll={val_nll:.4f})", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

        # Save checkpoint at end of every epoch
        save_checkpoint(
            CHECKPOINT_PATH, epoch + 1, model, optimizer,
            best_val_nll, patience_counter, best_epoch, log_history,
        )

    elapsed = time.time() - train_start
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")
    print(f"Best val_nll={best_val_nll:.4f} at epoch {best_epoch}")

    # Remove checkpoint (training finished cleanly)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint removed (training complete).")

    # ---- Save training log ----
    training_summary = {
        'experiment':     EXP_NAME,
        'trial':          9,
        'version':        'v2_frozen_backbone',
        'strategy':       'frozen_backbone_heteroscedastic',
        't8_backbone':    T8_MODEL_PATH,
        't8_test_dl':     T8_TEST_DL_PATH,
        'fixes_applied': [
            'frozen_backbone_prevents_nll_collapse',
            'stronger_var_reg_lambda_0.1',
            'reuse_t8_scalers_no_data_leakage',
            'min_delta_0.001_prevents_premature_stop',
        ],
        'hyperparameters': {
            'lr':                    float(LR),
            'batch_size':            BATCH_SIZE,
            'gradient_accumulation': GRAD_ACCUM,
            'num_epochs':            NUM_EPOCHS,
            'patience':              PATIENCE,
            'min_delta':             float(MIN_DELTA),
            'var_reg_lambda':        float(VAR_REG_LAMBDA),
            'var_reg_type':          'log',
            'in_channels':           5,
            'split':                 '80/10/10',
            'seed':                  SEED,
            'optimizer_scope':       'gat_heteroscedastic_head only',
        },
        'results': {
            'best_val_nll':            float(best_val_nll),
            'best_epoch':              best_epoch,
            'total_epochs_run':        epoch + 1,
            'training_time_minutes':   float(elapsed / 60),
        },
        'history': log_history,
    }
    with open(LOG_FILE, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training log saved to {LOG_FILE}")

    # ---- Write test_dl reference ----
    # The evaluation script will use T8's test_dl.pt directly.
    # Write a README-style JSON so evaluate_heteroscedastic.py knows where to find it.
    test_dl_ref_path = os.path.join(DATA_SAVE_PATH, 'test_dl_source.json')
    with open(test_dl_ref_path, 'w') as f:
        json.dump({'test_dl_path': T8_TEST_DL_PATH, 'source': 'T8 (reused)'}, f, indent=2)
    print(f"test_dl reference written to {test_dl_ref_path}")
    print(f"(eval script will load T8 test_dl.pt directly: {T8_TEST_DL_PATH})")

    return float(best_val_nll), best_epoch


if __name__ == '__main__':
    best_nll, best_ep = train()
    print(f"\nDone. Best val NLL = {best_nll:.4f} at epoch {best_ep}")
