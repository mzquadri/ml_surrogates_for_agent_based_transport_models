#!/usr/bin/env python3
"""
Phase 0: Lock T8 Baseline
==========================
Verifies that existing T8 evaluation can be reproduced before heteroscedastic extension.

This script:
1. Loads existing verified metrics from JSON artifacts
2. Runs deterministic forward pass to verify R², MAE, RMSE
3. Compares with thesis-verified values
4. Creates locked baseline artifact

Usage:
    python scripts/evaluation/lock_baseline_t8.py
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

# Add scripts to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from gnn.models.point_net_transf_gat import PointNetTransfGAT

# Paths
MODEL_FOLDER = REPO_ROOT / 'data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout'
OUTPUT_FILE = MODEL_FOLDER / 'baseline_locked.json'

# Thesis-verified values (tolerance: 0.01 for metrics, 0.1 for MAE/RMSE)
THESIS_VERIFIED = {
    'deterministic': {
        'r2': 0.5957,  # From trial8_uq_thesis_analysis or thesis Section 5.1
        'mae': 3.96,   # From thesis
        'rmse': 7.21,  # From thesis
    },
    'mc_dropout': {
        'spearman_rho': 0.4820,  # From mc_dropout_full_metrics_model8_mc30_100graphs.json
        'k_95': 11.34,           # From trial8_uq_diagnostics.json
    },
    'conformal': {
        'q_95': 14.68,           # From conformal_standard.json (absolute_q_95)
        'picp_95': 95.01,        # From conformal_standard.json (absolute_picp_95)
    },
    'selective_prediction': {
        'mae_reduction_50pct': 41.2,  # From thesis Section 5.6
    }
}

def load_model(device):
    """Load T8 model with exact configuration."""
    print("\n--- Loading T8 Model ---")
    
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        use_dropout=True,
        dropout=0.2  # T8 uses lower dropout
    )
    
    model_path = MODEL_FOLDER / 'trained_model/model.pth'
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"    Model loaded from: {model_path}")
    print(f"    Device: {device}")
    return model


def load_test_data():
    """Load test dataloader."""
    print("\n--- Loading Test Data ---")
    
    test_dl_path = MODEL_FOLDER / 'data_created_during_training/test_dl.pt'
    test_dl = torch.load(test_dl_path, weights_only=False)
    
    print(f"    Test graphs: {len(test_dl)}")
    print(f"    Nodes per graph: {test_dl[0].x.shape[0]}")
    print(f"    Total samples: {len(test_dl) * test_dl[0].x.shape[0]:,}")
    
    return test_dl


def run_deterministic_inference(model, test_dl, device):
    """Run deterministic forward pass (dropout off)."""
    print("\n--- Running Deterministic Inference ---")
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, graph in enumerate(test_dl):
            graph = graph.to(device)
            output = model(graph)
            
            all_predictions.append(output.squeeze().cpu().numpy())
            all_targets.append(graph.y.squeeze().cpu().numpy())
            
            if (i + 1) % 20 == 0:
                print(f"    Processed {i+1}/{len(test_dl)} graphs")
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    print(f"    Total predictions: {len(predictions):,}")
    
    return predictions, targets


def compute_metrics(predictions, targets):
    """Compute deterministic metrics."""
    print("\n--- Computing Deterministic Metrics ---")
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # MAE, RMSE
    mae = np.mean(np.abs(targets - predictions))
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    
    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'n_samples': len(predictions)
    }
    
    print(f"    R²:   {r2:.4f} (thesis: {THESIS_VERIFIED['deterministic']['r2']:.4f})")
    print(f"    MAE:  {mae:.4f} (thesis: {THESIS_VERIFIED['deterministic']['mae']:.2f})")
    print(f"    RMSE: {rmse:.4f} (thesis: {THESIS_VERIFIED['deterministic']['rmse']:.2f})")
    
    return metrics


def load_existing_uq_metrics():
    """Load existing verified UQ metrics from JSON artifacts."""
    print("\n--- Loading Existing UQ Metrics ---")
    
    # MC Dropout metrics
    mc_path = MODEL_FOLDER / 'uq_results/mc_dropout_full_metrics_model8_mc30_100graphs.json'
    with open(mc_path, 'r') as f:
        mc_metrics = json.load(f)
    
    # Calibration metrics
    diag_path = MODEL_FOLDER / 'trial8_uq_diagnostics.json'
    with open(diag_path, 'r') as f:
        diag_metrics = json.load(f)
    
    # Conformal metrics
    conf_path = MODEL_FOLDER / 'uq_results/conformal_standard.json'
    with open(conf_path, 'r') as f:
        conf_metrics = json.load(f)
    
    uq_metrics = {
        'mc_dropout': {
            'spearman_rho': mc_metrics['spearman'],
            'spearman_pval': mc_metrics['spearman_pval'],
            'r2': mc_metrics['r2'],
            'mae': mc_metrics['mae'],
            'rmse': mc_metrics['rmse'],
        },
        'calibration': {
            'k_90': diag_metrics['calibration_factors']['k_90'],
            'k_95': diag_metrics['calibration_factors']['k_95'],
            'k_99': diag_metrics['calibration_factors']['k_99'],
        },
        'conformal': {
            'q_90': conf_metrics['absolute_q_90'],
            'picp_90': conf_metrics['absolute_picp_90'],
            'q_95': conf_metrics['absolute_q_95'],
            'picp_95': conf_metrics['absolute_picp_95'],
        }
    }
    
    print(f"    MC Dropout ρ: {uq_metrics['mc_dropout']['spearman_rho']:.4f} "
          f"(thesis: {THESIS_VERIFIED['mc_dropout']['spearman_rho']:.4f})")
    print(f"    k₉₅: {uq_metrics['calibration']['k_95']:.2f} "
          f"(thesis: {THESIS_VERIFIED['mc_dropout']['k_95']:.2f})")
    print(f"    Conformal q₉₅: {uq_metrics['conformal']['q_95']:.2f} "
          f"(thesis: {THESIS_VERIFIED['conformal']['q_95']:.2f})")
    print(f"    Conformal PICP₉₅: {uq_metrics['conformal']['picp_95']:.2f}% "
          f"(thesis: {THESIS_VERIFIED['conformal']['picp_95']:.2f}%)")
    
    return uq_metrics


def verify_baseline(det_metrics, uq_metrics):
    """Verify metrics match thesis values within tolerance."""
    print("\n--- Verifying Baseline ---")
    
    checks = []
    
    # Deterministic checks
    r2_match = abs(det_metrics['r2'] - THESIS_VERIFIED['deterministic']['r2']) < 0.01
    mae_match = abs(det_metrics['mae'] - THESIS_VERIFIED['deterministic']['mae']) < 0.2
    rmse_match = abs(det_metrics['rmse'] - THESIS_VERIFIED['deterministic']['rmse']) < 0.2
    
    checks.append(('R²', r2_match, det_metrics['r2'], THESIS_VERIFIED['deterministic']['r2']))
    checks.append(('MAE', mae_match, det_metrics['mae'], THESIS_VERIFIED['deterministic']['mae']))
    checks.append(('RMSE', rmse_match, det_metrics['rmse'], THESIS_VERIFIED['deterministic']['rmse']))
    
    # UQ checks
    rho_match = abs(uq_metrics['mc_dropout']['spearman_rho'] - THESIS_VERIFIED['mc_dropout']['spearman_rho']) < 0.01
    k95_match = abs(uq_metrics['calibration']['k_95'] - THESIS_VERIFIED['mc_dropout']['k_95']) < 0.5
    q95_match = abs(uq_metrics['conformal']['q_95'] - THESIS_VERIFIED['conformal']['q_95']) < 0.5
    picp95_match = abs(uq_metrics['conformal']['picp_95'] - THESIS_VERIFIED['conformal']['picp_95']) < 1.0
    
    checks.append(('Spearman ρ', rho_match, uq_metrics['mc_dropout']['spearman_rho'], 
                   THESIS_VERIFIED['mc_dropout']['spearman_rho']))
    checks.append(('k₉₅', k95_match, uq_metrics['calibration']['k_95'], 
                   THESIS_VERIFIED['mc_dropout']['k_95']))
    checks.append(('Conformal q₉₅', q95_match, uq_metrics['conformal']['q_95'], 
                   THESIS_VERIFIED['conformal']['q_95']))
    checks.append(('Conformal PICP₉₅', picp95_match, uq_metrics['conformal']['picp_95'], 
                   THESIS_VERIFIED['conformal']['picp_95']))
    
    # Print results
    all_pass = True
    for name, passed, actual, expected in checks:
        status = "✓" if passed else "✗"
        print(f"    {status} {name:20s}: {actual:8.4f} (expected: {expected:8.4f})")
        if not passed:
            all_pass = False
    
    return all_pass, checks


def save_locked_baseline(det_metrics, uq_metrics, checks):
    """Save locked baseline artifact."""
    print("\n--- Saving Locked Baseline ---")
    
    baseline = {
        'deterministic': det_metrics,
        'uq': uq_metrics,
        'verification': {
            'all_checks_passed': all(c[1] for c in checks),
            'checks': [
                {
                    'metric': name,
                    'passed': passed,
                    'actual': actual,
                    'expected': expected,
                    'diff': actual - expected
                }
                for name, passed, actual, expected in checks
            ]
        },
        'metadata': {
            'model': 'T8 (point_net_transf_gat_8th_trial_lower_dropout)',
            'dropout': 0.2,
            'test_graphs': 100,
            'locked_for': 'heteroscedastic_extension'
        }
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"    Saved to: {OUTPUT_FILE}")
    return baseline


def main():
    """Main baseline locking function."""
    print("=" * 70)
    print("PHASE 0: LOCK T8 BASELINE")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model and data
    model = load_model(device)
    test_dl = load_test_data()
    
    # Run deterministic inference
    predictions, targets = run_deterministic_inference(model, test_dl, device)
    det_metrics = compute_metrics(predictions, targets)
    
    # Load existing UQ metrics
    uq_metrics = load_existing_uq_metrics()
    
    # Verify baseline
    all_pass, checks = verify_baseline(det_metrics, uq_metrics)
    
    # Save locked baseline
    baseline = save_locked_baseline(det_metrics, uq_metrics, checks)
    
    # Final summary
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ BASELINE LOCKED SUCCESSFULLY")
        print("=" * 70)
        print("\nAll metrics match thesis values within tolerance.")
        print("Proceed to Phase 1A: Implement heteroscedastic code.")
        return 0
    else:
        print("✗ BASELINE VERIFICATION FAILED")
        print("=" * 70)
        print("\nSome metrics do not match thesis values.")
        print("STOP: Debug environment before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
