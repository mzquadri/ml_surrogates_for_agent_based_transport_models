"""
CQR Evaluation Script
======================
Applies Conformalized Quantile Regression (CQR) post-training.

Pipeline:
  1. Load val_predictions.npz (calibration set -- saved by train_cqr.py)
  2. Compute nonconformity scores: E_i = max(q_lo_i - y_i, y_i - q_hi_i)
  3. Compute conformal quantiles Q_hat_90 and Q_hat_95 using the exact
     QuantileRegErrFunc.apply_inverse() formula from yromano/cqr
  4. Load test set, run trained model, apply min/max monotonicity ordering
  5. Construct CQR intervals:
       lo = q_lo - Q_hat,   hi = q_hi + Q_hat
  6. Compute and report: PICP90, PICP95, width90, width95, R2, MAE, RMSE, Spearman
  7. Save results to cqr_results/cqr_metrics.json

Gate criteria (must all pass to proceed to thesis write-up):
  - R^2 (midpoint) >= 0.57
  - PICP90         >= 88%  (theorem guarantees >=90%, +-2pp tolerance)
  - PICP95         >= 93%
  - width90        <  width95  (sanity: narrower at lower coverage)
  - Q_hat values are finite and positive

CQR reference: Romano, Patterson & Candes (2019) NeurIPS, arXiv:1905.03222
Reference impl: yromano/cqr (github.com/yromano/cqr)

Author: Mohd Zamin Quadri
"""

import os
import sys
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# ---- Path setup ----
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ---- Paths ----
BASE_DIR     = os.path.join(PROJECT_ROOT, 'data', 'TR-C_Benchmarks')
EXP_NAME     = 'point_net_transf_gat_10th_trial_cqr'
EXP_DIR      = os.path.join(BASE_DIR, EXP_NAME)

VAL_PRED_PATH   = os.path.join(EXP_DIR, 'val_predictions.npz')
MODEL_PATH      = os.path.join(EXP_DIR, 'trained_model', 'model.pth')
TEST_DL_PATH    = os.path.join(EXP_DIR, 'data_created_during_training', 'test_dl.pt')
RESULTS_DIR     = os.path.join(EXP_DIR, 'cqr_results')
RESULTS_PATH    = os.path.join(RESULTS_DIR, 'cqr_metrics.json')

# Baseline locked values for comparison
BASELINE = {
    'r2':       0.5957,
    'mae':      3.957,
    'rmse':     7.118,
    'picp90':   90.02,
    'picp95':   95.01,
    'k90':      7.563,
    'k95':      11.344,
    'conformal_q90': 9.920,
    'conformal_q95': 14.677,
}


# ---------------------------------------------------------------------------
# Exact CQR conformal quantile (QuantileRegErrFunc.apply_inverse in nc.py)
# ---------------------------------------------------------------------------

def cqr_conformal_q(scores, alpha):
    """
    Exact conformal quantile following QuantileRegErrFunc.apply_inverse()
    from yromano/cqr/nonconformist/nc.py.

    index = int(ceil((1 - alpha) * (n + 1))) - 1
    index = clamp(index, 0, n-1)
    Q_hat = scores_sorted[index]

    Args:
        scores (np.ndarray): 1D nonconformity scores (length n).
        alpha  (float):      Miscoverage level (e.g. 0.10 for 90% coverage).

    Returns:
        float: Conformal quantile Q_hat.
    """
    n = len(scores)
    scores_sorted = np.sort(scores)
    index = int(math.ceil((1.0 - alpha) * (n + 1))) - 1
    index = min(max(index, 0), n - 1)
    return float(scores_sorted[index])


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_r2(preds, targets):
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_picp(lo, hi, y):
    """Prediction Interval Coverage Probability (%)."""
    covered = (y >= lo) & (y <= hi)
    return float(np.mean(covered) * 100.0)


def compute_mean_width(lo, hi):
    return float(np.mean(hi - lo))


# ---------------------------------------------------------------------------
# Model inference on test set
# ---------------------------------------------------------------------------

def run_test_inference(model, test_loader, device):
    """
    Run the trained model on the test set.
    Returns raw q_lo, q_hi, targets (all 1D numpy arrays, no ordering applied).
    Min/max ordering is applied by the caller.
    """
    model.eval()
    all_q_lo    = []
    all_q_hi    = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader:
            data   = data.to(device)
            target = data.y

            q_lo, q_hi = model(data)

            all_q_lo.append(q_lo.squeeze().cpu().numpy())
            all_q_hi.append(q_hi.squeeze().cpu().numpy())
            all_targets.append(target.squeeze().cpu().numpy())

    return (
        np.concatenate(all_q_lo),
        np.concatenate(all_q_hi),
        np.concatenate(all_targets)
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate():
    """
    Full CQR evaluation pipeline.
    """
    print("=" * 70)
    print("CQR EVALUATION")
    print(f"Experiment: {EXP_NAME}")
    print("=" * 70)

    # ---- Check required files ----
    for path, label in [
        (VAL_PRED_PATH, 'val_predictions.npz'),
        (MODEL_PATH,    'model.pth'),
        (TEST_DL_PATH,  'test_dl.pt'),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] Required file not found: {path}")
            print(f"        ({label} must be produced by train_cqr.py first)")
            return 1

    # ---- Device ----
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"Device: XPU (Intel Arc) -- {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: GPU -- {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Device: CPU")

    # ================================================================
    # Step 1: Load calibration predictions (val set)
    # ================================================================
    print("\n--- Step 1: Load calibration predictions ---")
    cal = np.load(VAL_PRED_PATH)
    q_lo_cal = cal['q_lo']    # already min/max ordered by train_cqr.py
    q_hi_cal = cal['q_hi']    # already min/max ordered by train_cqr.py
    y_cal    = cal['targets']
    n_cal    = len(y_cal)

    # Re-apply ordering (idempotent) for safety
    q_lo_cal = np.minimum(cal['q_lo'], cal['q_hi'])
    q_hi_cal = np.maximum(cal['q_lo'], cal['q_hi'])

    print(f"  Calibration samples: {n_cal}")
    print(f"  q_lo_cal range: [{q_lo_cal.min():.4f}, {q_lo_cal.max():.4f}]")
    print(f"  q_hi_cal range: [{q_hi_cal.min():.4f}, {q_hi_cal.max():.4f}]")
    print(f"  y_cal    range: [{y_cal.min():.4f},    {y_cal.max():.4f}]")

    # ================================================================
    # Step 2: Compute nonconformity scores
    # Exact QuantileRegErrFunc.apply() from yromano/cqr/nonconformist/nc.py:
    #   error_low  = y_lower - y        (positive when q_lo > y)
    #   error_high = y - y_upper        (positive when y > q_hi)
    #   E = maximum(error_high, error_low)
    # ================================================================
    print("\n--- Step 2: Nonconformity scores ---")
    error_low  = q_lo_cal - y_cal    # positive when lower bound too high
    error_high = y_cal - q_hi_cal    # positive when y above upper bound
    E_cal = np.maximum(error_high, error_low)

    print(f"  E_cal range: [{E_cal.min():.4f}, {E_cal.max():.4f}]")
    print(f"  E_cal mean:  {E_cal.mean():.4f}")
    print(f"  Fraction E > 0 (uncovered): {(E_cal > 0).mean():.4f}")

    # ================================================================
    # Step 3: Conformal quantiles Q_hat (exact apply_inverse formula)
    # ================================================================
    print("\n--- Step 3: Conformal quantiles ---")
    Q_hat_90 = cqr_conformal_q(E_cal, alpha=0.10)
    Q_hat_95 = cqr_conformal_q(E_cal, alpha=0.05)

    # Index diagnostics
    n = len(E_cal)
    idx_90 = int(math.ceil((1.0 - 0.10) * (n + 1))) - 1
    idx_90 = min(max(idx_90, 0), n - 1)
    idx_95 = int(math.ceil((1.0 - 0.05) * (n + 1))) - 1
    idx_95 = min(max(idx_95, 0), n - 1)

    print(f"  n_cal = {n}")
    print(f"  Q_hat_90 (alpha=0.10): {Q_hat_90:.4f}  [index={idx_90}]")
    print(f"  Q_hat_95 (alpha=0.05): {Q_hat_95:.4f}  [index={idx_95}]")

    # Sanity: Q_hat_95 >= Q_hat_90 (more conservative for higher coverage)
    if Q_hat_95 < Q_hat_90:
        print(f"  [WARNING] Q_hat_95 < Q_hat_90 ({Q_hat_95:.4f} < {Q_hat_90:.4f})")
        print(f"            This is unexpected. Check calibration scores.")
    else:
        print(f"  [OK] Q_hat_95 >= Q_hat_90 (as expected)")

    # ================================================================
    # Step 4: Load model and test dataloader
    # ================================================================
    print("\n--- Step 4: Load model and test set ---")

    from gnn.models.point_net_transf_gat_quantile import PointNetTransfGATQuantile
    from gnn.gnn_io import collate_fn

    model = PointNetTransfGATQuantile(
        in_channels=5,
        out_channels=2,
        dropout=0.2,
        use_dropout=True,
        log_to_wandb=False
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded: {n_params:,} parameters")

    test_dataset = torch.load(TEST_DL_PATH, weights_only=False)
    test_loader  = DataLoader(
        dataset=test_dataset, batch_size=8, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    print(f"  Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")

    # ================================================================
    # Step 5: Test inference + monotonicity ordering
    # ================================================================
    print("\n--- Step 5: Test inference ---")
    raw_q_lo, raw_q_hi, y_test = run_test_inference(model, test_loader, device)
    n_test = len(y_test)

    # Apply min/max monotonicity ordering
    # LearnerOptimizedCrossing.predict() from yromano/cqr
    q_lo_test = np.minimum(raw_q_lo, raw_q_hi)
    q_hi_test = np.maximum(raw_q_lo, raw_q_hi)

    n_crossings = int(np.sum(raw_q_lo > raw_q_hi))
    print(f"  Test samples: {n_test}")
    print(f"  Monotonicity crossings corrected: {n_crossings} / {n_test}")
    print(f"  q_lo_test range: [{q_lo_test.min():.4f}, {q_lo_test.max():.4f}]")
    print(f"  q_hi_test range: [{q_hi_test.min():.4f}, {q_hi_test.max():.4f}]")
    print(f"  y_test    range: [{y_test.min():.4f},    {y_test.max():.4f}]")

    # ================================================================
    # Step 6: Construct CQR intervals
    # Exact RegressorNc.predict() from yromano/cqr:
    #   intervals[:, 0] = prediction[:, 0]  - Q_hat   (q_lo - Q_hat)
    #   intervals[:, 1] = prediction[:, -1] + Q_hat   (q_hi + Q_hat)
    # ================================================================
    print("\n--- Step 6: Construct CQR intervals ---")

    lo_90 = q_lo_test - Q_hat_90
    hi_90 = q_hi_test + Q_hat_90
    lo_95 = q_lo_test - Q_hat_95
    hi_95 = q_hi_test + Q_hat_95

    print(f"  90% interval sample: lo=[{lo_90.min():.2f}, {lo_90.max():.2f}]")
    print(f"  90% interval sample: hi=[{hi_90.min():.2f}, {hi_90.max():.2f}]")

    # ================================================================
    # Step 7: Compute metrics
    # ================================================================
    print("\n--- Step 7: Compute metrics ---")

    PICP_90 = compute_picp(lo_90, hi_90, y_test)
    PICP_95 = compute_picp(lo_95, hi_95, y_test)
    width_90 = compute_mean_width(lo_90, hi_90)
    width_95 = compute_mean_width(lo_95, hi_95)

    # Point metrics from midpoint of ordered quantiles
    midpoint = (q_lo_test + q_hi_test) / 2.0
    r2   = compute_r2(midpoint, y_test)
    mae  = float(np.mean(np.abs(y_test - midpoint)))
    rmse = float(np.sqrt(np.mean((y_test - midpoint) ** 2)))
    spearman_rho = float(spearmanr(midpoint, y_test).correlation)

    # Coverage on calibration set (sanity: should be close to 1-alpha)
    picp_cal_90 = compute_picp(q_lo_cal - Q_hat_90, q_hi_cal + Q_hat_90, y_cal)
    picp_cal_95 = compute_picp(q_lo_cal - Q_hat_95, q_hi_cal + Q_hat_95, y_cal)

    print(f"\n  === CQR RESULTS ===")
    print(f"  PICP 90% (test):  {PICP_90:.2f}%  (baseline MC: {BASELINE['picp90']:.2f}%)")
    print(f"  PICP 95% (test):  {PICP_95:.2f}%  (baseline MC: {BASELINE['picp95']:.2f}%)")
    print(f"  Width 90% (test): {width_90:.4f}")
    print(f"  Width 95% (test): {width_95:.4f}")
    print(f"  Q_hat_90:         {Q_hat_90:.4f}")
    print(f"  Q_hat_95:         {Q_hat_95:.4f}")
    print(f"  R2 (midpoint):    {r2:.4f}  (baseline T8: {BASELINE['r2']:.4f})")
    print(f"  MAE (midpoint):   {mae:.4f}  (baseline T8: {BASELINE['mae']:.4f})")
    print(f"  RMSE (midpoint):  {rmse:.4f}  (baseline T8: {BASELINE['rmse']:.4f})")
    print(f"  Spearman rho:     {spearman_rho:.4f}")
    print(f"  PICP 90% (cal):   {picp_cal_90:.2f}%  (sanity check)")
    print(f"  PICP 95% (cal):   {picp_cal_95:.2f}%  (sanity check)")

    # ================================================================
    # Gate check
    # ================================================================
    print("\n--- Gate Check ---")
    gates = {
        'r2_ge_0.57':        (r2 >= 0.57,          f"R2={r2:.4f} >= 0.57"),
        'picp90_ge_88':      (PICP_90 >= 88.0,     f"PICP90={PICP_90:.2f}% >= 88%"),
        'picp95_ge_93':      (PICP_95 >= 93.0,     f"PICP95={PICP_95:.2f}% >= 93%"),
        'width90_lt_width95': (width_90 < width_95, f"width90={width_90:.4f} < width95={width_95:.4f}"),
        'q_hat_90_finite_pos': (np.isfinite(Q_hat_90) and Q_hat_90 > 0,
                                f"Q_hat_90={Q_hat_90:.4f} finite+positive"),
        'q_hat_95_finite_pos': (np.isfinite(Q_hat_95) and Q_hat_95 > 0,
                                f"Q_hat_95={Q_hat_95:.4f} finite+positive"),
    }

    all_pass = True
    for key, (passed, msg) in gates.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {msg}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("*** ALL GATE CRITERIA MET ***")
        gate_status = "PASS"
    else:
        print("*** SOME GATE CRITERIA NOT MET -- SEE ABOVE ***")
        gate_status = "FAIL"

    # ================================================================
    # Save results
    # ================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        'experiment':     EXP_NAME,
        'reference':      'Romano, Patterson & Candes (2019) NeurIPS arXiv:1905.03222',
        'implementation': 'yromano/cqr (github.com/yromano/cqr)',
        'alpha':          0.10,
        'tau_lo':         0.05,
        'tau_hi':         0.95,
        'n_cal':          int(n_cal),
        'n_test':         int(n_test),
        'conformal_quantiles': {
            'Q_hat_90': Q_hat_90,
            'Q_hat_95': Q_hat_95,
            'Q_hat_90_index': int(idx_90),
            'Q_hat_95_index': int(idx_95),
        },
        'test_metrics': {
            'PICP_90_pct':    PICP_90,
            'PICP_95_pct':    PICP_95,
            'width_90':       width_90,
            'width_95':       width_95,
            'r2_midpoint':    r2,
            'mae_midpoint':   mae,
            'rmse_midpoint':  rmse,
            'spearman_rho':   spearman_rho,
            'n_crossings_corrected': int(n_crossings),
        },
        'calibration_sanity': {
            'PICP_90_cal_pct': picp_cal_90,
            'PICP_95_cal_pct': picp_cal_95,
        },
        'baseline_comparison': {
            'baseline_r2':       BASELINE['r2'],
            'baseline_picp90':   BASELINE['picp90'],
            'baseline_picp95':   BASELINE['picp95'],
            'baseline_k90':      BASELINE['k90'],
            'baseline_k95':      BASELINE['k95'],
            'baseline_conformal_q90': BASELINE['conformal_q90'],
            'baseline_conformal_q95': BASELINE['conformal_q95'],
        },
        'gate_results': {k: bool(v[0]) for k, v in gates.items()},
        'gate_status':  gate_status,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(evaluate())
