#!/usr/bin/env python3
"""
Conformal Prediction + UQ Comparison Script
============================================
Creates conformal prediction intervals and compares with/without UQ for Model 8.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from scipy.stats import spearmanr

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
os.chdir(REPO_ROOT)

def main():
    # Load MC Dropout results
    data_path = 'data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz'
    data = np.load(data_path)
    preds = data['predictions']
    uncs = data['uncertainties']
    targets = data['targets']
    errors = np.abs(targets - preds)

    print('=' * 70)
    print('CONFORMAL PREDICTION + UQ COMPARISON')
    print('=' * 70)
    print(f'Total nodes: {len(preds):,}')

    # ============================================
    # CONFORMAL PREDICTION (Split Conformal)
    # ============================================
    print('\n--- CONFORMAL PREDICTION ---')

    # Split: 50% calibration, 50% test
    np.random.seed(42)
    n = len(preds)
    indices = np.random.permutation(n)
    cal_idx = indices[:n // 2]
    test_idx = indices[n // 2:]

    cal_errors = errors[cal_idx]
    test_preds = preds[test_idx]
    test_targets = targets[test_idx]
    test_errors = errors[test_idx]
    test_uncs = uncs[test_idx]

    print(f'Calibration set: {len(cal_idx):,} nodes')
    print(f'Test set: {len(test_idx):,} nodes')

    # Compute conformal intervals at multiple coverage levels
    coverage_levels = [0.50, 0.80, 0.90, 0.95]
    conformal_results = {}

    print('\nConformal Prediction Results:')
    print('-' * 60)
    print(f'{"Level":<10} {"Q (radius)":<15} {"Actual PICP":<15} {"MPIW":<12}')
    print('-' * 60)

    for alpha in coverage_levels:
        # Find quantile of calibration residuals
        q = np.quantile(cal_errors, alpha)

        # Prediction intervals on test set
        lower = test_preds - q
        upper = test_preds + q

        # Coverage (PICP)
        covered = (test_targets >= lower) & (test_targets <= upper)
        picp = covered.mean()

        # Width (MPIW)
        mpiw = 2 * q

        conformal_results[f'{int(alpha * 100)}%'] = {
            'target': alpha,
            'q': float(q),
            'picp': float(picp),
            'mpiw': float(mpiw)
        }

        status = 'OK' if abs(picp - alpha) < 0.02 else '!'
        print(f'{int(alpha * 100)}% {status:<6} {q:<15.3f} {picp:<15.4f} {mpiw:<12.3f}')

    print('-' * 60)

    # ============================================
    # MC DROPOUT INTERVALS (using sigma)
    # ============================================
    print('\nMC Dropout Prediction Intervals:')
    print('-' * 60)
    print(f'{"Level":<10} {"k × sigma":<15} {"Actual PICP":<15} {"Avg Width":<12}')
    print('-' * 60)

    mc_results = {}
    # For Gaussian: 50%~0.67σ, 80%~1.28σ, 90%~1.64σ, 95%~1.96σ
    k_values = {0.50: 0.67, 0.80: 1.28, 0.90: 1.645, 0.95: 1.96}

    for alpha, k in k_values.items():
        lower = test_preds - k * test_uncs
        upper = test_preds + k * test_uncs

        covered = (test_targets >= lower) & (test_targets <= upper)
        picp = covered.mean()
        avg_width = (2 * k * test_uncs).mean()

        mc_results[f'{int(alpha * 100)}%'] = {
            'target': alpha,
            'k': k,
            'picp': float(picp),
            'avg_width': float(avg_width)
        }

        status = 'OK' if abs(picp - alpha) < 0.05 else '!'
        print(f'{int(alpha * 100)}% {status:<6} {k:<15.2f} {picp:<15.4f} {avg_width:<12.3f}')

    print('-' * 60)

    # ============================================
    # WITH UQ vs WITHOUT UQ COMPARISON
    # ============================================
    print('\n' + '=' * 70)
    print('WITH UQ vs WITHOUT UQ COMPARISON')
    print('=' * 70)

    # Baseline metrics (without UQ)
    ss_res = np.sum((test_targets - test_preds) ** 2)
    ss_tot = np.sum((test_targets - test_targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    mae = np.mean(test_errors)
    rmse = np.sqrt(np.mean(test_errors ** 2))

    spearman_corr, _ = spearmanr(test_uncs, test_errors)

    print('\n' + '=' * 70)
    print('MODEL 8 - WITH UQ vs WITHOUT UQ')
    print('=' * 70)
    print(f'\n{"Aspect":<25} {"Without UQ":<20} {"With UQ (MC Dropout)":<25}')
    print('-' * 70)
    print(f'{"R-squared":<25} {r2:<20.4f} {r2:<25.4f}')
    print(f'{"MAE":<25} {mae:<20.2f} {mae:<25.2f}')
    print(f'{"RMSE":<25} {rmse:<20.2f} {rmse:<25.2f}')
    print('-' * 70)
    print(f'{"Uncertainty Output":<25} {"None":<20} {"sigma per node":<25}')
    spearman_str = f"Spearman rho = {spearman_corr:.2f}"
    print(f'{"Unc-Error Correlation":<25} {"N/A":<20} {spearman_str:<25}')
    print('-' * 70)
    conf_90 = conformal_results['90%']
    mc_90 = mc_results['90%']
    conf_str = f"PICP={conf_90['picp']:.2f}, W={conf_90['mpiw']:.1f}"
    mc_str = f"PICP={mc_90['picp']:.2f}, W={mc_90['avg_width']:.1f}"
    print(f'{"90% Interval (Conformal)":<25} {"N/A":<20} {conf_str:<25}')
    print(f'{"90% Interval (MC)":<25} {"N/A":<20} {mc_str:<25}')
    print('-' * 70)
    print(f'{"High-Unc Flagging":<25} {"Not possible":<20} {"Flag top 10% nodes":<25}')
    print(f'{"Decision Support":<25} {"Single value":<20} {"Value + confidence":<25}')
    print('=' * 70)

    # Save comparison results
    output_folder = 'data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results'

    comparison = {
        'model': 8,
        'test_nodes': int(len(test_idx)),
        'baseline': {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse)
        },
        'mc_dropout': {
            'spearman': float(spearman_corr),
            'unc_mean': float(test_uncs.mean()),
            'unc_std': float(test_uncs.std()),
            'intervals': mc_results
        },
        'conformal': conformal_results
    }

    with open(f'{output_folder}/uq_comparison_model8.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f'\nResults saved to: {output_folder}/uq_comparison_model8.json')

    # ============================================
    # COVERAGE COMPARISON PLOT
    # ============================================
    plot_folder = f'{output_folder}/uq_plots'
    os.makedirs(plot_folder, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Coverage comparison
    ax1 = axes[0]
    levels = [50, 80, 90, 95]
    conformal_picp = [conformal_results[f'{l}%']['picp'] * 100 for l in levels]
    mc_picp = [mc_results[f'{l}%']['picp'] * 100 for l in levels]

    x = np.arange(len(levels))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, conformal_picp, width, label='Conformal', color='steelblue', edgecolor='black')
    bars2 = ax1.bar(x + width / 2, mc_picp, width, label='MC Dropout', color='coral', edgecolor='black')
    ax1.plot(x, levels, 'k--', marker='o', label='Target', linewidth=2)

    ax1.set_xlabel('Nominal Coverage Level', fontsize=12)
    ax1.set_ylabel('Actual Coverage (%)', fontsize=12)
    ax1.set_title('Coverage: Conformal vs MC Dropout', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{l}%' for l in levels])
    ax1.legend()
    ax1.set_ylim(0, 105)

    # Plot 2: Interval width comparison
    ax2 = axes[1]
    conformal_width = [conformal_results[f'{l}%']['mpiw'] for l in levels]
    mc_width = [mc_results[f'{l}%']['avg_width'] for l in levels]

    bars1 = ax2.bar(x - width / 2, conformal_width, width, label='Conformal', color='steelblue', edgecolor='black')
    bars2 = ax2.bar(x + width / 2, mc_width, width, label='MC Dropout', color='coral', edgecolor='black')

    ax2.set_xlabel('Nominal Coverage Level', fontsize=12)
    ax2.set_ylabel('Mean Interval Width (vehicles)', fontsize=12)
    ax2.set_title('Sharpness: Conformal vs MC Dropout', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{l}%' for l in levels])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{plot_folder}/coverage_comparison.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {plot_folder}/coverage_comparison.png')

    plt.close('all')
    print('\n✅ All comparisons complete!')


if __name__ == '__main__':
    main()
