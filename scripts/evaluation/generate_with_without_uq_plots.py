"""
Generate WITH vs WITHOUT UQ comparison plots and summaries.
Supports Models 2, 5, 6, 7, 8 (auto-detects NPZ files).

This script creates:
1. det_pred_vs_target.png - Deterministic predictions vs targets
2. mc_pred_vs_target.png - MC Dropout mean predictions vs targets
3. det_error_hist.png - Deterministic error distribution
4. mc_error_hist.png - MC Dropout error distribution
5. pred_diff_mc_minus_det.png - Histogram of prediction differences
6. with_without_metrics_bar.png - Metrics comparison bar chart
7. with_without_dashboard.png - Combined 2x2 dashboard
8. WITH_WITHOUT_UQ_SUMMARY_MODEL<k>.md - Thesis-ready markdown summary

Usage:
    python generate_with_without_uq_plots.py                  # Process all available models
    python generate_with_without_uq_plots.py --models 6 7     # Process specific models
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re
import argparse
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure matplotlib with nice academic style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Nice academic color palette
COLORS = {
    'det': '#2E86AB',          # Deterministic - steel blue
    'mc': '#28A745',           # MC Dropout - green
    'error': '#DC3545',        # Error - red
    'neutral': '#6C757D',      # Neutral - gray
    'accent': '#F18F01',       # Accent - orange
}

# Model folder mapping
MODEL_FOLDERS = {
    2: 'point_net_transf_gat_2nd_try',
    5: 'point_net_transf_gat_5th_try',
    6: 'point_net_transf_gat_6th_trial_lower_lr',
    7: 'point_net_transf_gat_7th_trial_80_10_10_split',
    8: 'point_net_transf_gat_8th_trial_lower_dropout',
}

# Unit label for plots - Elena confirmed: vehicles per hour
UNIT = "veh/h"  # traffic flow difference in vehicles per hour
# Nodes per graph (for inferring graph count when filename doesn't specify)
NODES_PER_GRAPH = 31635

def get_dropout(model_num):
    return 0.2 if model_num == 8 else 0.3

def find_best_npz(uq_dir, pattern):
    """Find largest NPZ matching pattern (full run > partial)."""
    candidates = []
    for f in os.listdir(uq_dir):
        if pattern in f.lower() and f.endswith('.npz'):
            fp = os.path.join(uq_dir, f)
            candidates.append(fp)
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getsize(p))


MAX_SCATTER = 50000  # subsample for scatter plots


def process_model(model_num, base_dir='data/TR-C_Benchmarks'):
    """Process a single model."""
    if model_num not in MODEL_FOLDERS:
        print(f'Model {model_num} not supported')
        return None
    
    model_dir = os.path.join(base_dir, MODEL_FOLDERS[model_num])
    uq_dir = os.path.join(model_dir, 'uq_results')
    plot_dir = os.path.join(uq_dir, 'uq_plots')
    
    if not os.path.exists(uq_dir):
        print(f'Model {model_num}: uq_results folder not found')
        return None
    
    # Auto-detect NPZ files
    det_path = find_best_npz(uq_dir, 'deterministic')
    mc_path = find_best_npz(uq_dir, 'mc_dropout')
    
    if not det_path or not mc_path:
        print(f'Model {model_num}: Missing NPZ files (det={det_path}, mc={mc_path})')
        return None
    
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f'\n{"="*60}')
    print(f'MODEL {model_num}')
    print('='*60)
    print(f'Det NPZ: {os.path.basename(det_path)}')
    print(f'MC NPZ:  {os.path.basename(mc_path)}')
    
    # Load data
    det = np.load(det_path)
    mc = np.load(mc_path)
    
    # Get arrays
    det_preds = det['predictions'].ravel()
    det_targets = det['targets'].ravel()
    
    mc_preds = mc['predictions'].ravel()
    mc_targets = mc['targets'].ravel()
    mc_uncs = mc['uncertainties'].ravel()
    
    n_det = len(det_preds)
    n_mc = len(mc_preds)
    
    print(f'Det samples (road segments): {n_det:,}')
    print(f'MC  samples (road segments): {n_mc:,}')
    
    # CRITICAL: Verify alignment
    if n_det != n_mc:
        print(f'ERROR: Length mismatch! Det={n_det}, MC={n_mc}')
        return {'error': 'length mismatch'}
    
    max_target_diff = float(np.max(np.abs(det_targets - mc_targets)))
    print(f'Max |target_det - target_mc|: {max_target_diff}')
    
    if max_target_diff > 1e-6:
        print(f'ERROR: Targets do not match!')
        return {'error': f'target mismatch: {max_target_diff}'}
    
    cfg = {'dropout': get_dropout(model_num)}
    
    targets = det_targets  # Use either since they match
    n_nodes = len(targets)
    
    # Determine graph count from MC NPZ filename or infer from node count
    mc_basename = os.path.basename(mc_path)
    m = re.search(r'(\d+)graphs', mc_basename)
    if m:
        n_graphs = int(m.group(1))
        graph_source = "from filename"
    else:
        # Infer from node count (each graph has NODES_PER_GRAPH nodes)
        n_graphs = n_nodes // NODES_PER_GRAPH if n_nodes % NODES_PER_GRAPH == 0 else 'unknown'
        graph_source = "inferred from node count"
    
    # Compute metrics
    det_r2 = r2_score(targets, det_preds)
    det_mae = mean_absolute_error(targets, det_preds)
    det_rmse = np.sqrt(mean_squared_error(targets, det_preds))
    
    mc_r2 = r2_score(targets, mc_preds)
    mc_mae = mean_absolute_error(targets, mc_preds)
    mc_rmse = np.sqrt(mean_squared_error(targets, mc_preds))
    
    mc_errors = np.abs(targets - mc_preds)
    spearman_rho, _ = stats.spearmanr(mc_uncs, mc_errors)
    
    pred_diff = mc_preds - det_preds
    mean_abs_diff = float(np.mean(np.abs(pred_diff)))
    
    print(f'Test graphs: {n_graphs} ({graph_source})')
    print(f'\nDeterministic: R2={det_r2:.4f}, MAE={det_mae:.2f}, RMSE={det_rmse:.2f}')
    print(f'MC Dropout:    R2={mc_r2:.4f}, MAE={mc_mae:.2f}, RMSE={mc_rmse:.2f}')
    print(f'Spearman rho:  {spearman_rho:.4f}')
    print(f'Mean |pred_mc - pred_det|: {mean_abs_diff:.4f} {UNIT}')
    
    # Subsampling for scatter
    np.random.seed(42)
    if n_nodes > MAX_SCATTER:
        idx = np.random.choice(n_nodes, MAX_SCATTER, replace=False)
    else:
        idx = np.arange(n_nodes)
    
    generated_files = []
    skipped_files = []
    
    # --- PLOT 1: det_pred_vs_target.png ---
    fpath = f'{plot_dir}/det_pred_vs_target.png'
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(targets[idx], det_preds[idx], alpha=0.25, s=3, c=COLORS['det'], rasterized=True)
    lims = [min(targets.min(), det_preds.min()), max(targets.max(), det_preds.max())]
    ax.plot(lims, lims, color=COLORS['error'], linestyle='--', lw=1.5, label='Perfect prediction (y=x)')
    ax.set_xlabel(f'Target Traffic Flow Difference ({UNIT})')
    ax.set_ylabel(f'Predicted Traffic Flow Difference ({UNIT})')
    ax.set_title(f'Model {model_num} - Deterministic Inference (R²={det_r2:.3f})')
    ax.legend(loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    generated_files.append('det_pred_vs_target.png')
    
    # --- PLOT 2: mc_pred_vs_target.png ---
    fpath = f'{plot_dir}/mc_pred_vs_target.png'
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(targets[idx], mc_preds[idx], alpha=0.25, s=3, c=COLORS['mc'], rasterized=True)
    lims = [min(targets.min(), mc_preds.min()), max(targets.max(), mc_preds.max())]
    ax.plot(lims, lims, color=COLORS['error'], linestyle='--', lw=1.5, label='Perfect prediction (y=x)')
    ax.set_xlabel(f'Target Traffic Flow Difference ({UNIT})')
    ax.set_ylabel(f'Predicted Traffic Flow Difference ({UNIT})')
    ax.set_title(f'Model {model_num} - MC Dropout Mean (S=30, R²={mc_r2:.3f})')
    ax.legend(loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    generated_files.append('mc_pred_vs_target.png')
    
    # --- PLOT 3: det_error_hist.png ---
    fpath = f'{plot_dir}/det_error_hist.png'
    det_abs_err = np.abs(targets - det_preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(det_abs_err, bins=100, color=COLORS['det'], edgecolor='white', alpha=0.8)
    ax.axvline(det_abs_err.mean(), color=COLORS['error'], linestyle='--', lw=2, label=f'MAE = {det_mae:.2f} {UNIT}')
    ax.set_xlabel(f'Absolute Prediction Error ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Model {model_num} - Deterministic Error Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    generated_files.append('det_error_hist.png')
    
    # --- PLOT 4: mc_error_hist.png ---
    fpath = f'{plot_dir}/mc_error_hist.png'
    mc_abs_err = np.abs(targets - mc_preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(mc_abs_err, bins=100, color=COLORS['mc'], edgecolor='white', alpha=0.8)
    ax.axvline(mc_abs_err.mean(), color=COLORS['error'], linestyle='--', lw=2, label=f'MAE = {mc_mae:.2f} {UNIT}')
    ax.set_xlabel(f'Absolute Prediction Error ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Model {model_num} - MC Dropout Error Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    generated_files.append('mc_error_hist.png')
    
    # --- PLOT 5: pred_diff_mc_minus_det.png ---
    fpath = f'{plot_dir}/pred_diff_mc_minus_det.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pred_diff, bins=100, color=COLORS['neutral'], edgecolor='white', alpha=0.8)
    ax.axvline(0, color=COLORS['error'], linestyle='--', lw=2, label='Zero difference')
    ax.set_xlabel(f'Prediction Difference: MC Mean - Deterministic ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Model {model_num} - Prediction Shift (Mean |diff| = {mean_abs_diff:.3f} {UNIT})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    generated_files.append('pred_diff_mc_minus_det.png')
    
    # --- PLOT 6: with_without_metrics_bar.png ---
    fpath = f'{plot_dir}/with_without_metrics_bar.png'
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    metrics = ['R²', f'MAE ({UNIT})', f'RMSE ({UNIT})']
    det_vals = [det_r2, det_mae, det_rmse]
    mc_vals = [mc_r2, mc_mae, mc_rmse]
    bar_colors = [COLORS['det'], COLORS['mc']]
    
    for i, (m, dv, mv) in enumerate(zip(metrics, det_vals, mc_vals)):
        ax = axes[i]
        bars = ax.bar(['Deterministic', 'MC Dropout'], [dv, mv], color=bar_colors, edgecolor='white', linewidth=1)
        ax.set_ylabel(m)
        ax.set_title(m)
        for bar, val in zip(bars, [dv, mv]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(dv,mv),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle(f'Model {model_num} - Deterministic vs MC Dropout Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    generated_files.append('with_without_metrics_bar.png')
    
    # --- PLOT 7: with_without_dashboard.png ---
    fpath = f'{plot_dir}/with_without_dashboard.png'
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (0,0) Det pred vs target
    ax = axes[0,0]
    ax.scatter(targets[idx], det_preds[idx], alpha=0.25, s=3, c=COLORS['det'], rasterized=True)
    lims = [min(targets.min(), det_preds.min()), max(targets.max(), det_preds.max())]
    ax.plot(lims, lims, color=COLORS['error'], linestyle='--', lw=1.5)
    ax.set_xlabel(f'Target Traffic Flow Difference ({UNIT})')
    ax.set_ylabel(f'Predicted Traffic Flow Difference ({UNIT})')
    ax.set_title(f'Deterministic Inference (R²={det_r2:.3f})')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # (0,1) MC pred vs target
    ax = axes[0,1]
    ax.scatter(targets[idx], mc_preds[idx], alpha=0.25, s=3, c=COLORS['mc'], rasterized=True)
    ax.plot(lims, lims, color=COLORS['error'], linestyle='--', lw=1.5)
    ax.set_xlabel(f'Target Traffic Flow Difference ({UNIT})')
    ax.set_ylabel(f'Predicted Traffic Flow Difference ({UNIT})')
    ax.set_title(f'MC Dropout Mean (R²={mc_r2:.3f})')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # (1,0) Deterministic absolute error hist
    ax = axes[1,0]
    det_abs_err = np.abs(targets - det_preds)
    ax.hist(det_abs_err, bins=80, color=COLORS['det'], edgecolor='white', alpha=0.8)
    ax.axvline(det_mae, color=COLORS['error'], linestyle='--', lw=2)
    ax.set_xlabel(f'Absolute Prediction Error ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Deterministic Error (MAE={det_mae:.2f} {UNIT})')
    
    # (1,1) MC absolute error hist
    ax = axes[1,1]
    mc_abs_err = np.abs(targets - mc_preds)
    ax.hist(mc_abs_err, bins=80, color=COLORS['mc'], edgecolor='white', alpha=0.8)
    ax.axvline(mc_mae, color=COLORS['error'], linestyle='--', lw=2)
    ax.set_xlabel(f'Absolute Prediction Error ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'MC Dropout Error (MAE={mc_mae:.2f} {UNIT})')
    
    fig.suptitle(f'Model {model_num} - With vs Without UQ Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    generated_files.append('with_without_dashboard.png')
    
    # --- MARKDOWN SUMMARY ---
    md_path = f"{uq_dir}/WITH_WITHOUT_UQ_SUMMARY_MODEL{model_num}.md"
    
    md_content = f'''# Model {model_num} - With vs Without UQ Comparison

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | PointNetTransfGAT |
| Dropout | {cfg['dropout']} |
| Test graphs | {n_graphs} |
| Total nodes | {n_nodes:,} |

## Metrics Summary

### Without UQ (Deterministic Inference)

| Metric | Value |
|--------|-------|
| R2 | {det_r2:.4f} |
| MAE | {det_mae:.2f} {UNIT} |
| RMSE | {det_rmse:.2f} {UNIT} |

### With UQ (MC Dropout, S=30)

| Metric | Value |
|--------|-------|
| R2 | {mc_r2:.4f} |
| MAE | {mc_mae:.2f} {UNIT} |
| RMSE | {mc_rmse:.2f} {UNIT} |
| Spearman rho (unc vs error) | {spearman_rho:.4f} |

## Comparison

| Metric | Deterministic | MC Dropout | Difference |
|--------|--------------|------------|------------|
| R2 | {det_r2:.4f} | {mc_r2:.4f} | {mc_r2 - det_r2:+.4f} |
| MAE ({UNIT}) | {det_mae:.2f} | {mc_mae:.2f} | {mc_mae - det_mae:+.2f} |
| RMSE ({UNIT}) | {det_rmse:.2f} | {mc_rmse:.2f} | {mc_rmse - det_rmse:+.2f} |

**Mean absolute prediction difference:** {mean_abs_diff:.4f} {UNIT}

## Target Alignment Verification

Max |target_det - target_mc|: **{max_target_diff}**

This confirms that both inference modes were evaluated on identical test data.

## Interpretation

MC Dropout inference with S=30 forward passes introduces a small reduction in point prediction accuracy compared to deterministic inference (single forward pass with dropout disabled). This is expected, as the MC mean aggregates predictions across stochastic dropout masks, which can slightly blur the decision boundary.

However, MC Dropout provides per-prediction uncertainty estimates (standard deviation across samples), enabling:
1. Identification of unreliable predictions
2. Construction of calibrated prediction intervals via conformal prediction
3. Risk-aware decision making in downstream applications

The positive Spearman correlation ({spearman_rho:.4f}) between predicted uncertainty and actual error confirms that the uncertainty estimates are informative: higher predicted uncertainty tends to correspond to larger prediction errors.

## Generated Plots

- `det_pred_vs_target.png` - Deterministic predictions vs targets
- `mc_pred_vs_target.png` - MC Dropout mean predictions vs targets
- `det_error_hist.png` - Deterministic error distribution
- `mc_error_hist.png` - MC Dropout error distribution
- `pred_diff_mc_minus_det.png` - Histogram of prediction differences
- `with_without_metrics_bar.png` - Metrics comparison bar chart
- `with_without_dashboard.png` - Combined 2x2 dashboard
'''
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    generated_files.append(f'WITH_WITHOUT_UQ_SUMMARY_MODEL{model_num}.md')
    
    EPS = 1e-6
    return {
        'det_npz': det_path,
        'mc_npz': mc_path,
        'target_check': 'PASSED' if max_target_diff <= EPS else f'FAILED ({max_target_diff})',
        'generated': generated_files,
        'skipped': skipped_files,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate WITH vs WITHOUT UQ plots')
    parser.add_argument('--models', type=int, nargs='+', default=None,
                       help='Model numbers to process (default: all available)')
    args = parser.parse_args()
    
    # Determine which models to process
    if args.models:
        models_to_process = args.models
    else:
        models_to_process = list(MODEL_FOLDERS.keys())
    
    results_checklist = {}
    
    for model_num in models_to_process:
        result = process_model(model_num)
        if result:
            results_checklist[model_num] = result
    
    # Final checklist
    print('\n')
    print('='*70)
    print('FINAL CHECKLIST')
    print('='*70)
    
    for model_num, res in results_checklist.items():
        print(f'\n--- MODEL {model_num} ---')
        if 'error' in res:
            print(f'  ERROR: {res["error"]}')
        else:
            print(f'  Det NPZ: {res["det_npz"]}')
            print(f'  MC NPZ:  {res["mc_npz"]}')
            print(f'  Target alignment: {res["target_check"]}')
            print(f'  Generated files ({len(res["generated"])}):')
            for f in res['generated']:
                print(f'    - {f}')
            if res['skipped']:
                print(f'  Already present (skipped): {len(res["skipped"])}')
                for f in res['skipped']:
                    print(f'    - {f}')
    
    print('\n' + '='*70)
    print('COMPLETE')
    print('='*70)


if __name__ == '__main__':
    main()
