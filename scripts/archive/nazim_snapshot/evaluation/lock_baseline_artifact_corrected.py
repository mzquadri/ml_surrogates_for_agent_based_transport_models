#!/usr/bin/env python3
"""
Phase 0: T8 Baseline Artifact Lock (CORRECTED)
================================================
Verifies existing T8 metrics from authoritative artifact sources.

IMPORTANT: This is an **artifact lock**, not a full baseline reproduction.
- Loads deterministic metrics from test_predictions.npz (authoritative source)
- Loads MC Dropout metrics from mc_dropout JSON
- Loads calibration metrics from diagnostics JSON
- Does NOT re-run inference (environment missing some dependencies)

This confirms:
- Artifact files exist and are readable
- Metrics match thesis-verified values
- Artifacts are internally consistent

This does NOT confirm:
- Current environment can reproduce metrics from scratch
- Model/dataloader/evaluation pipeline still works end-to-end

For a true baseline reproduction, would need full environment + re-run evaluation.

Author: Mohd Zamin Quadri
"""

import json
import numpy as np
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_FOLDER = REPO_ROOT / 'data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout'
OUTPUT_FILE = MODEL_FOLDER / 'baseline_artifact_lock.json'

# Thesis-verified values (STRICT tolerances - investigate if exceeded)
THESIS_VERIFIED = {
    'deterministic': {
        'r2': 0.5957,   # From thesis Section 5.1 / Table 5.1
        'mae': 3.96,    # From thesis Table 5.1
        'rmse': 7.12,   # From thesis (corrected from 7.21)
    },
    'mc_dropout': {
        'spearman_rho': 0.4820,  # From thesis Section 5.5
        'r2_mc': 0.5857,         # MC Dropout R^2 (slightly lower than deterministic)
        'mae_mc': 3.95,          # MC Dropout MAE
    },
    'calibration': {
        'k_95': 11.34,           # From thesis Section 5.9
        'k_90': 7.56,
    },
    'conformal': {
        'q_95': 14.68,           # From thesis Section 5.8
        'picp_95': 95.01,
        'q_90': 9.92,
        'picp_90': 90.02,
    },
}

# STRICT tolerances (no artificial loosening)
TOLERANCES = {
    'r2': 0.001,        # 0.1% for R^2 (very strict)
    'mae': 0.05,        # 0.05 veh/h for MAE
    'rmse': 0.05,       # 0.05 veh/h for RMSE
    'spearman_rho': 0.001,  # 0.1% for correlation
    'k_95': 0.1,        # 0.1 for calibration factors
    'k_90': 0.1,
    'q_95': 0.1,        # 0.1 veh/h for conformal quantiles
    'q_90': 0.1,
    'picp_95': 0.5,     # 0.5% for coverage
    'picp_90': 0.5,
}


def load_deterministic_metrics():
    """
    Load AUTHORITATIVE deterministic metrics from test_predictions.npz.
    
    This is the CORRECT source for deterministic baseline, not MC Dropout JSON.
    """
    print("\n--- Loading Deterministic Metrics (AUTHORITATIVE SOURCE) ---")
    
    npz_path = MODEL_FOLDER / 'test_predictions.npz'
    print(f"    Source: {npz_path.name}")
    
    data = np.load(npz_path)
    predictions = data['predictions']
    targets = data['targets']
    
    # Compute deterministic metrics
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    mae = np.mean(np.abs(targets - predictions))
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    
    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'n_samples': len(predictions),
        'source_file': 'test_predictions.npz'
    }
    
    print(f"    R^2:  {r2:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    Samples: {len(predictions):,}")
    
    return metrics


def load_mc_dropout_metrics():
    """Load MC Dropout metrics from JSON."""
    print("\n--- Loading MC Dropout Metrics ---")
    
    mc_path = MODEL_FOLDER / 'uq_results/mc_dropout_full_metrics_model8_mc30_100graphs.json'
    print(f"    Source: {mc_path.name}")
    
    with open(mc_path, 'r') as f:
        mc_data = json.load(f)
    
    metrics = {
        'r2_mc': mc_data['r2'],
        'mae_mc': mc_data['mae'],
        'rmse_mc': mc_data['rmse'],
        'spearman_rho': mc_data['spearman'],
        'spearman_pval': mc_data['spearman_pval'],
        'n_graphs': mc_data['n_graphs'],
        'n_nodes': mc_data['n_nodes'],
        'num_samples': mc_data['num_samples'],
        'source_file': 'mc_dropout_full_metrics_model8_mc30_100graphs.json'
    }
    
    print(f"    Spearman rho: {metrics['spearman_rho']:.6f}")
    print(f"    R^2 (MC):     {metrics['r2_mc']:.6f}")
    print(f"    MAE (MC):     {metrics['mae_mc']:.6f}")
    
    return metrics


def load_calibration_metrics():
    """Load calibration diagnostics from JSON."""
    print("\n--- Loading Calibration Metrics ---")
    
    diag_path = MODEL_FOLDER / 'trial8_uq_diagnostics.json'
    print(f"    Source: {diag_path.name}")
    
    with open(diag_path, 'r') as f:
        diag_data = json.load(f)
    
    metrics = {
        'k_90': diag_data['calibration_factors']['k_90'],
        'k_95': diag_data['calibration_factors']['k_95'],
        'k_99': diag_data['calibration_factors']['k_99'],
        'source_file': 'trial8_uq_diagnostics.json'
    }
    
    print(f"    k_90: {metrics['k_90']:.4f}")
    print(f"    k_95: {metrics['k_95']:.4f}")
    
    return metrics


def load_conformal_metrics():
    """Load conformal prediction metrics from JSON."""
    print("\n--- Loading Conformal Metrics ---")
    
    conf_path = MODEL_FOLDER / 'uq_results/conformal_standard.json'
    print(f"    Source: {conf_path.name}")
    
    with open(conf_path, 'r') as f:
        conf_data = json.load(f)
    
    metrics = {
        'q_90': conf_data['absolute_q_90'],
        'picp_90': conf_data['absolute_picp_90'],
        'q_95': conf_data['absolute_q_95'],
        'picp_95': conf_data['absolute_picp_95'],
        'n_calibration': conf_data['n_calibration'],
        'n_test': conf_data['n_test'],
        'source_file': 'conformal_standard.json'
    }
    
    print(f"    q_95:      {metrics['q_95']:.4f}")
    print(f"    PICP_95:   {metrics['picp_95']:.2f}%")
    
    return metrics


def verify_metrics(det_metrics, mc_metrics, cal_metrics, conf_metrics):
    """Verify all metrics match thesis values within STRICT tolerances."""
    print("\n--- Verification Summary ---")
    print(f"\n{'Category':<20} {'Metric':<20} {'Actual':<12} {'Thesis':<12} {'Diff':<10} {'Status':<10}")
    print("-" * 84)
    
    checks = []
    
    # Deterministic checks (from test_predictions.npz)
    for key, thesis_key in [('r2', 'r2'), ('mae', 'mae'), ('rmse', 'rmse')]:
        actual = det_metrics[key]
        expected = THESIS_VERIFIED['deterministic'][thesis_key]
        diff = abs(actual - expected)
        passed = diff < TOLERANCES[key]
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{'Deterministic':<20} {key.upper():<20} {actual:<12.6f} {expected:<12.4f} {diff:<10.6f} {status:<10}")
        checks.append((f'deterministic_{key}', passed, actual, expected, diff, 'test_predictions.npz'))
    
    # MC Dropout checks
    for key, thesis_key in [('spearman_rho', 'spearman_rho'), ('r2_mc', 'r2_mc'), ('mae_mc', 'mae_mc')]:
        actual = mc_metrics[key]
        expected = THESIS_VERIFIED['mc_dropout'][thesis_key]
        diff = abs(actual - expected)
        tol_key = 'spearman_rho' if key == 'spearman_rho' else 'mae'
        passed = diff < TOLERANCES[tol_key]
        status = "[PASS]" if passed else "[FAIL]"
        label = key.upper().replace('_MC', ' (MC)')
        print(f"{'MC Dropout':<20} {label:<20} {actual:<12.6f} {expected:<12.4f} {diff:<10.6f} {status:<10}")
        checks.append((f'mc_{key}', passed, actual, expected, diff, 'mc_dropout_full_metrics.json'))
    
    # Calibration checks
    for key in ['k_90', 'k_95']:
        actual = cal_metrics[key]
        expected = THESIS_VERIFIED['calibration'][key]
        diff = abs(actual - expected)
        passed = diff < TOLERANCES[key]
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{'Calibration':<20} {key.upper():<20} {actual:<12.4f} {expected:<12.2f} {diff:<10.4f} {status:<10}")
        checks.append((f'calibration_{key}', passed, actual, expected, diff, 'trial8_uq_diagnostics.json'))
    
    # Conformal checks
    for key in ['q_90', 'q_95', 'picp_90', 'picp_95']:
        actual = conf_metrics[key]
        expected = THESIS_VERIFIED['conformal'][key]
        diff = abs(actual - expected)
        passed = diff < TOLERANCES[key]
        status = "[PASS]" if passed else "[FAIL]"
        label = key.upper().replace('_', ' ')
        print(f"{'Conformal':<20} {label:<20} {actual:<12.4f} {expected:<12.2f} {diff:<10.4f} {status:<10}")
        checks.append((f'conformal_{key}', passed, actual, expected, diff, 'conformal_standard.json'))
    
    return checks


def analyze_failures(checks):
    """Analyze and report any verification failures."""
    failures = [c for c in checks if not c[1]]
    
    if not failures:
        return True
    
    print("\n--- FAILURES DETECTED ---")
    for name, _, actual, expected, diff, source in failures:
        print(f"\n  {name}:")
        print(f"    Actual:   {actual:.6f}")
        print(f"    Expected: {expected:.6f}")
        print(f"    Diff:     {diff:.6f}")
        print(f"    Source:   {source}")
        print(f"    Tolerance: {TOLERANCES.get(name.split('_')[-1], 'N/A')}")
    
    return False


def save_artifact_lock(det_metrics, mc_metrics, cal_metrics, conf_metrics, checks, all_pass):
    """Save artifact lock with corrected sources."""
    print("\n--- Saving Artifact Lock ---")
    
    artifact_lock = {
        'lock_type': 'artifact_verification',
        'warning': 'This is an ARTIFACT LOCK, not a full baseline reproduction. '
                   'Verifies existing artifacts match thesis, but does NOT re-run evaluation.',
        'deterministic': det_metrics,
        'mc_dropout': mc_metrics,
        'calibration': cal_metrics,
        'conformal': conf_metrics,
        'verification': {
            'all_checks_passed': all_pass,
            'total_checks': len(checks),
            'passed_checks': sum(1 for c in checks if c[1]),
            'failed_checks': sum(1 for c in checks if not c[1]),
            'checks': [
                {
                    'name': name,
                    'passed': passed,
                    'actual': actual,
                    'expected': expected,
                    'diff': diff,
                    'source_file': source,
                    'tolerance': TOLERANCES.get(name.split('_')[-1], None)
                }
                for name, passed, actual, expected, diff, source in checks
            ]
        },
        'metadata': {
            'model': 'T8 (point_net_transf_gat_8th_trial_lower_dropout)',
            'dropout': 0.2,
            'test_samples': det_metrics['n_samples'],
            'locked_for': 'heteroscedastic_extension',
            'deterministic_source': 'test_predictions.npz (AUTHORITATIVE)',
            'mc_source': 'mc_dropout_full_metrics_model8_mc30_100graphs.json',
            'calibration_source': 'trial8_uq_diagnostics.json',
            'conformal_source': 'conformal_standard.json'
        }
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(artifact_lock, f, indent=2)
    
    print(f"    Saved to: {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    return artifact_lock


def main():
    """Main artifact lock function."""
    print("=" * 84)
    print("PHASE 0: T8 BASELINE ARTIFACT LOCK (CORRECTED)")
    print("=" * 84)
    print("\nIMPORTANT: This is an ARTIFACT LOCK, not a full reproduction.")
    print("Verifies existing artifacts match thesis, does NOT re-run evaluation.")
    
    # Load metrics from authoritative sources
    det_metrics = load_deterministic_metrics()
    mc_metrics = load_mc_dropout_metrics()
    cal_metrics = load_calibration_metrics()
    conf_metrics = load_conformal_metrics()
    
    # Verify with STRICT tolerances
    checks = verify_metrics(det_metrics, mc_metrics, cal_metrics, conf_metrics)
    
    # Analyze failures
    all_pass = analyze_failures(checks)
    
    # Save artifact lock
    artifact_lock = save_artifact_lock(det_metrics, mc_metrics, cal_metrics, conf_metrics, checks, all_pass)
    
    # Final summary
    print("\n" + "=" * 84)
    if all_pass:
        print("*** ARTIFACT LOCK SUCCESSFUL ***")
        print("=" * 84)
        print("\nAll metrics match thesis values within STRICT tolerances.")
        print("\n[NEXT] Run smoke tests before training")
    else:
        print("*** ARTIFACT LOCK FAILED ***")
        print("=" * 84)
        print("\nSome metrics do not match thesis values.")
        print("[STOP] Investigate discrepancies before proceeding")
    
    print(f"\nArtifact lock saved to:")
    print(f"  {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
