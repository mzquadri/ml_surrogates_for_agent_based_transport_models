"""
run_cal_pass.py
===============
Standalone script to regenerate val_predictions.npz for the CQR trial.

Background
----------
train_cqr.py completed 885 epochs successfully (best model at epoch 860,
saved to trained_model/model.pth).  After training, the script crashed at
json.dump (numpy float32 not JSON serializable) before it could call
run_calibration_pass().  This left val_predictions.npz unsaved.

This script reproduces the exact pipeline used in train_cqr.py up to and
including the calibration pass, then exits.  No training is performed.
No files are modified except val_predictions.npz.

Author: Mohd Zamin Quadri
"""

import os
import sys

import torch

# ---- Path setup (identical to train_cqr.py) ----
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ---- Import constants and functions from train_cqr (safe: __main__ guard) ----
from train_cqr import (
    SEED, BATCH_SIZE, IN_CHANNELS, DROPOUT,
    MODEL_SAVE_PATH, DATA_SAVE_PATH, VAL_PRED_PATH,
    set_seeds, load_dataset, split_dataset,
    normalize_and_build_loaders, run_calibration_pass,
)


def main():
    print("=" * 60)
    print("CQR CALIBRATION PASS (val_predictions.npz recovery)")
    print("=" * 60)
    print()
    print("  Model  :", MODEL_SAVE_PATH)
    print("  Output :", VAL_PRED_PATH)
    print()

    # 1. Reproducibility
    set_seeds(SEED)

    # 2. Device
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print("  Device: XPU (Intel Arc) --", torch.xpu.get_device_name(0))
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("  Device: GPU --", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("  Device: CPU")

    # 3. Load dataset
    print("\nLoading dataset...")
    datalist = load_dataset()

    # 4. Split (deterministic seed=42, same as training run)
    print("Splitting dataset (80/10/10)...")
    train_set, valid_set, test_set = split_dataset(datalist)
    print("  Train:", len(train_set.indices),
          " Val:", len(valid_set.indices),
          " Test:", len(test_set.indices))

    # 5. Normalize and build loaders
    #    Scalers are re-fitted deterministically and will overwrite the existing
    #    pkl files with byte-for-byte identical values (same data, same seed).
    print("Normalizing and building data loaders...")
    train_loader, val_loader = normalize_and_build_loaders(
        train_set, valid_set, test_set, BATCH_SIZE, DATA_SAVE_PATH
    )
    print("  Val batches:", len(val_loader))

    # 6. Load best model
    from gnn.models.point_net_transf_gat_quantile import PointNetTransfGATQuantile

    model = PointNetTransfGATQuantile(
        in_channels=IN_CHANNELS,
        out_channels=2,
        dropout=DROPOUT,
        use_dropout=True,
        log_to_wandb=False
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
    model = model.to(device)
    print("\nLoaded model from:", MODEL_SAVE_PATH)

    # 7. Calibration pass
    print("\n" + "=" * 60)
    print("Running calibration pass...")
    print("=" * 60)
    run_calibration_pass(model, val_loader, device, VAL_PRED_PATH)

    # 8. Quick verification
    import numpy as np
    data = np.load(VAL_PRED_PATH)
    assert data['q_lo'].shape == (len(valid_set.indices),), \
        "Shape mismatch: q_lo"
    assert data['q_hi'].shape == (len(valid_set.indices),), \
        "Shape mismatch: q_hi"
    assert data['targets'].shape == (len(valid_set.indices),), \
        "Shape mismatch: targets"
    print("\n[OK] val_predictions.npz verified:")
    print("     q_lo    shape:", data['q_lo'].shape, " dtype:", data['q_lo'].dtype)
    print("     q_hi    shape:", data['q_hi'].shape, " dtype:", data['q_hi'].dtype)
    print("     targets shape:", data['targets'].shape, " dtype:", data['targets'].dtype)
    print("\nDone. Run evaluate_cqr.py next.")


if __name__ == '__main__':
    main()
