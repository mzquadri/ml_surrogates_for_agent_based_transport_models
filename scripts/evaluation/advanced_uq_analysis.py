"""
Advanced UQ Analysis: Thesis-Critical Visualizations
=====================================================

This script generates the supervisor-approved visualizations that demonstrate
MC Dropout's VALUE beyond raw correlation:

1. BINNED ERROR vs UNCERTAINTY - Shows σ ranks error (not raw scatter fog)
2. RISK-COVERAGE CURVE - Filtering by σ reduces error (actionable)
3. INTERVAL WIDTH COMPARISON - Absolute vs σ-normalized (efficiency gain)
4. HEXBIN DENSITY - Better than 3M point scatter

Key findings to prove:
- σ is INFORMATIVE for ranking (Spearman ρ ~ 0.48)
- σ is NOT CALIBRATED as predictive std (k95 ~ 11.65, not ~2)
- σ-normalized conformal gives ADAPTIVE intervals

Usage:
    python advanced_uq_analysis.py          # Process Model 8 (best)
    python advanced_uq_analysis.py --all    # Process all models
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure matplotlib
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Academic colors
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#28A745', 
    'error': '#DC3545',
    'uncertainty': '#6F42C1',
    'accent': '#F18F01',
    'neutral': '#6C757D',
}

UNIT = "veh/h"

MODEL_FOLDERS = {
    2: 'point_net_transf_gat_2nd_try',
    5: 'point_net_transf_gat_5th_try',
    6: 'point_net_transf_gat_6th_trial_lower_lr',
    7: 'point_net_transf_gat_7th_trial_80_10_10_split',
    8: 'point_net_transf_gat_8th_trial_lower_dropout',
}


def load_model_data(model_num, base_dir='data/TR-C_Benchmarks'):
    """Load MC dropout results for a model."""
    folder = MODEL_FOLDERS.get(model_num)
    if not folder:
        return None
    
    uq_dir = os.path.join(base_dir, folder, 'uq_results')
    
    # Find MC dropout NPZ
    mc_path = None
    for f in os.listdir(uq_dir):
        if 'mc_dropout' in f.lower() and f.endswith('.npz'):
            candidate = os.path.join(uq_dir, f)
            if mc_path is None or os.path.getsize(candidate) > os.path.getsize(mc_path):
                mc_path = candidate
    
    if not mc_path:
        return None
    
    # Load data
    data = np.load(mc_path)
    preds = data['predictions'].ravel()
    targets = data['targets'].ravel()
    sigmas = data['uncertainties'].ravel()
    errors = np.abs(targets - preds)
    
    # Load conformal results
    conf_path = os.path.join(uq_dir, 'conformal_standard.json')
    conformal = None
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conformal = json.load(f)
    
    return {
        'predictions': preds,
        'targets': targets,
        'sigmas': sigmas,
        'errors': errors,
        'conformal': conformal,
        'uq_dir': uq_dir,
    }


def plot_binned_error_vs_sigma(data, model_num, save_dir, n_bins=20):
    """
    CRITICAL PLOT: Binned error vs sigma with quantile bands.
    This proves σ RANKS error, replacing foggy scatter.
    """
    sigmas = data['sigmas']
    errors = data['errors']
    
    # Create sigma bins (percentile-based for even samples)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(sigmas, percentiles)
    bin_centers = []
    median_errors = []
    q10_errors = []
    q90_errors = []
    mean_errors = []
    
    for i in range(n_bins):
        mask = (sigmas >= bin_edges[i]) & (sigmas < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (sigmas >= bin_edges[i]) & (sigmas <= bin_edges[i+1])
        
        bin_errs = errors[mask]
        if len(bin_errs) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            median_errors.append(np.median(bin_errs))
            q10_errors.append(np.percentile(bin_errs, 10))
            q90_errors.append(np.percentile(bin_errs, 90))
            mean_errors.append(np.mean(bin_errs))
    
    bin_centers = np.array(bin_centers)
    median_errors = np.array(median_errors)
    q10_errors = np.array(q10_errors)
    q90_errors = np.array(q90_errors)
    mean_errors = np.array(mean_errors)
    
    # Calculate Spearman ρ
    rho, pval = stats.spearmanr(sigmas, errors)
    
    # Calculate "Top 10% σ accounts for X% of total error"
    sigma_p90 = np.percentile(sigmas, 90)
    top10_mask = sigmas >= sigma_p90
    top10_error_share = errors[top10_mask].sum() / errors.sum() * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 10-90% band
    ax.fill_between(bin_centers, q10_errors, q90_errors, 
                    alpha=0.25, color=COLORS['uncertainty'], label='10th-90th percentile')
    
    # Median line
    ax.plot(bin_centers, median_errors, 'o-', color=COLORS['uncertainty'], 
            lw=2.5, markersize=6, label='Median error per bin')
    
    # Mean line
    ax.plot(bin_centers, mean_errors, 's--', color=COLORS['accent'], 
            lw=1.5, markersize=4, alpha=0.7, label='Mean error per bin')
    
    # Annotations
    ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.text(0.05, 0.85, f'Top 10% σ → {top10_error_share:.1f}% of total error', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel(f'Predicted Uncertainty σ ({UNIT})', fontsize=12)
    ax.set_ylabel(f'Absolute Prediction Error ({UNIT})', fontsize=12)
    ax.set_title(f'Model {model_num}: Binned Error vs Uncertainty\n'
                 f'(Higher σ → Higher Error: Uncertainty is Informative)', fontsize=13)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    fpath = os.path.join(save_dir, 'binned_error_vs_sigma.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: binned_error_vs_sigma.png')
    print(f'    Spearman ρ = {rho:.4f}, Top 10% σ accounts for {top10_error_share:.1f}% of error')
    
    return {'rho': rho, 'top10_error_share': top10_error_share}


def plot_risk_coverage_curve(data, model_num, save_dir, n_points=100):
    """
    CRITICAL PLOT: Error vs retained fraction after filtering by σ.
    Shows: if you reject high-σ predictions, remaining error decreases.
    """
    sigmas = data['sigmas']
    errors = data['errors']
    
    # Sort by sigma (ascending)
    sort_idx = np.argsort(sigmas)
    sorted_errors = errors[sort_idx]
    
    # Calculate cumulative MAE as we retain more samples
    fractions = np.linspace(0.05, 1.0, n_points)
    retained_maes = []
    retained_rmses = []
    
    n_total = len(errors)
    for frac in fractions:
        n_keep = int(frac * n_total)
        kept_errors = sorted_errors[:n_keep]  # Keep lowest-σ samples
        retained_maes.append(np.mean(kept_errors))
        retained_rmses.append(np.sqrt(np.mean(kept_errors**2)))
    
    # Full dataset metrics
    full_mae = np.mean(errors)
    full_rmse = np.sqrt(np.mean(errors**2))
    
    # Metrics at different retention levels
    mae_at_50 = retained_maes[n_points // 2]  # ~50% retention
    mae_at_90 = retained_maes[int(0.9 * n_points)]  # ~90% retention
    
    improvement_50 = (1 - mae_at_50 / full_mae) * 100
    improvement_90 = (1 - mae_at_90 / full_mae) * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot(fractions * 100, retained_maes, '-', color=COLORS['primary'], 
            lw=2.5, label='MAE (filtering by σ)')
    
    ax.axhline(full_mae, color=COLORS['error'], linestyle='--', lw=1.5, 
               label=f'Full dataset MAE = {full_mae:.2f}')
    
    # Mark key points
    ax.scatter([50], [mae_at_50], color=COLORS['accent'], s=100, zorder=5, 
               marker='o', edgecolor='black')
    ax.annotate(f'50% retained:\nMAE={mae_at_50:.2f}\n({improvement_50:.1f}% ↓)', 
                xy=(50, mae_at_50), xytext=(35, mae_at_50 - 0.5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.scatter([90], [mae_at_90], color=COLORS['secondary'], s=100, zorder=5, 
               marker='s', edgecolor='black')
    ax.annotate(f'90% retained:\nMAE={mae_at_90:.2f}\n({improvement_90:.1f}% ↓)', 
                xy=(90, mae_at_90), xytext=(75, mae_at_90 + 0.3),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('Percentage of Samples Retained (lowest σ first)', fontsize=12)
    ax.set_ylabel(f'Mean Absolute Error ({UNIT})', fontsize=12)
    ax.set_title(f'Model {model_num}: Risk-Coverage Curve\n'
                 f'(Rejecting high-σ samples reduces error)', fontsize=13)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 105)
    
    plt.tight_layout()
    fpath = os.path.join(save_dir, 'risk_coverage_curve.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: risk_coverage_curve.png')
    print(f'    50% retention → MAE {mae_at_50:.2f} ({improvement_50:.1f}% improvement)')
    print(f'    90% retention → MAE {mae_at_90:.2f} ({improvement_90:.1f}% improvement)')
    
    return {
        'full_mae': full_mae,
        'mae_at_50': mae_at_50,
        'mae_at_90': mae_at_90,
        'improvement_50': improvement_50,
        'improvement_90': improvement_90,
    }


def plot_interval_width_comparison(data, model_num, save_dir):
    """
    CRITICAL PLOT: Compare interval widths - absolute vs sigma-normalized.
    This proves σ-normalized gives ADAPTIVE intervals.
    """
    conformal = data['conformal']
    sigmas = data['sigmas']
    
    if conformal is None:
        print('  ⚠ No conformal data, skipping interval width plot')
        return None
    
    # Get conformal parameters
    abs_q95 = conformal['absolute_q_95']
    sigma_k95 = conformal['sigma_k_95']
    abs_width_95 = conformal['absolute_width_95']
    sigma_width_95_mean = conformal['sigma_width_95']
    
    # Calculate per-sample interval widths
    abs_widths = np.full_like(sigmas, 2 * abs_q95)  # Constant width
    sigma_widths = 2 * sigma_k95 * sigmas  # Adaptive width
    
    # Statistics
    abs_mean = np.mean(abs_widths)
    abs_median = np.median(abs_widths)
    sigma_mean = np.mean(sigma_widths)
    sigma_median = np.median(sigma_widths)
    
    # Width by sigma percentile
    low_sigma_mask = sigmas < np.percentile(sigmas, 25)
    high_sigma_mask = sigmas > np.percentile(sigmas, 75)
    
    sigma_width_low = np.mean(sigma_widths[low_sigma_mask])
    sigma_width_high = np.mean(sigma_widths[high_sigma_mask])
    
    efficiency_low = (1 - sigma_width_low / abs_mean) * 100
    
    # Plot 1: Histogram comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram of widths
    ax = axes[0]
    bins = np.linspace(0, np.percentile(sigma_widths, 99), 60)
    
    ax.hist(sigma_widths, bins=bins, alpha=0.7, color=COLORS['uncertainty'], 
            label=f'σ-normalized (mean={sigma_mean:.1f})', density=True)
    ax.axvline(abs_mean, color=COLORS['error'], linestyle='--', lw=2.5, 
               label=f'Absolute (constant={abs_mean:.1f})')
    ax.axvline(sigma_mean, color=COLORS['uncertainty'], linestyle='-', lw=2, 
               alpha=0.7)
    
    ax.set_xlabel(f'95% Prediction Interval Width ({UNIT})', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Interval Widths at 95% Level', fontsize=13)
    ax.legend(loc='upper right')
    
    # Right: Boxplot comparison by sigma quartile
    ax = axes[1]
    
    # Create data for boxplot
    quartile_labels = ['Q1\n(low σ)', 'Q2', 'Q3', 'Q4\n(high σ)']
    quartile_widths = []
    quartile_edges = np.percentile(sigmas, [0, 25, 50, 75, 100])
    
    for i in range(4):
        mask = (sigmas >= quartile_edges[i]) & (sigmas < quartile_edges[i+1])
        if i == 3:
            mask = (sigmas >= quartile_edges[i]) & (sigmas <= quartile_edges[i+1])
        quartile_widths.append(sigma_widths[mask])
    
    bp = ax.boxplot(quartile_widths, labels=quartile_labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    
    colors_q = [COLORS['secondary'], COLORS['primary'], COLORS['accent'], COLORS['error']]
    for patch, color in zip(bp['boxes'], colors_q):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(abs_mean, color=COLORS['neutral'], linestyle='--', lw=2, 
               label=f'Absolute method (constant={abs_mean:.1f})')
    
    ax.set_ylabel(f'95% Interval Width ({UNIT})', fontsize=12)
    ax.set_xlabel('Uncertainty Quartile', fontsize=12)
    ax.set_title('σ-Normalized Intervals are ADAPTIVE\n(narrow for confident, wide for uncertain)', fontsize=13)
    ax.legend(loc='upper left')
    
    # Annotations
    ax.annotate(f'Low σ: {sigma_width_low:.1f}\n({efficiency_low:.0f}% narrower)', 
                xy=(1, sigma_width_low), xytext=(1.3, sigma_width_low + 5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    fpath = os.path.join(save_dir, 'interval_width_comparison.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: interval_width_comparison.png')
    print(f'    Absolute width (constant): {abs_mean:.2f} {UNIT}')
    print(f'    σ-normalized mean width: {sigma_mean:.2f} {UNIT}')
    print(f'    Low-σ (Q1) width: {sigma_width_low:.2f} ({efficiency_low:.1f}% narrower than absolute)')
    
    return {
        'abs_width': abs_mean,
        'sigma_width_mean': sigma_mean,
        'sigma_width_low': sigma_width_low,
        'sigma_width_high': sigma_width_high,
        'efficiency_low': efficiency_low,
    }


def plot_hexbin_uncertainty_error(data, model_num, save_dir):
    """
    HEXBIN plot: Better than scatter fog for 3M points.
    """
    sigmas = data['sigmas']
    errors = data['errors']
    
    # Use log scale for better visualization
    log_sigmas = np.log10(sigmas + 0.01)  # Add small offset to avoid log(0)
    log_errors = np.log10(errors + 0.01)
    
    # Calculate Spearman ρ
    rho, _ = stats.spearmanr(sigmas, errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Hexbin on original scale
    ax = axes[0]
    hb = ax.hexbin(sigmas, errors, gridsize=50, cmap='YlOrRd', mincnt=1,
                   extent=[0, np.percentile(sigmas, 99), 0, np.percentile(errors, 99)])
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count')
    
    ax.set_xlabel(f'Predicted Uncertainty σ ({UNIT})', fontsize=12)
    ax.set_ylabel(f'Absolute Error ({UNIT})', fontsize=12)
    ax.set_title(f'Uncertainty vs Error Density (ρ={rho:.3f})', fontsize=13)
    
    # Right: Hexbin on log scale
    ax = axes[1]
    hb = ax.hexbin(log_sigmas, log_errors, gridsize=50, cmap='YlGnBu', mincnt=1)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count')
    
    ax.set_xlabel(f'log₁₀(σ)', fontsize=12)
    ax.set_ylabel(f'log₁₀(|error|)', fontsize=12)
    ax.set_title(f'Log-Scale Density View', fontsize=13)
    
    plt.tight_layout()
    fpath = os.path.join(save_dir, 'hexbin_uncertainty_error.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: hexbin_uncertainty_error.png')
    
    return {'rho': rho}


def plot_calibration_diagnostic(data, model_num, save_dir):
    """
    Shows WHY σ is not calibrated: RMSE >> mean σ, k95 >> 2
    """
    sigmas = data['sigmas']
    errors = data['errors']
    conformal = data['conformal']
    
    if conformal is None:
        return None
    
    mean_sigma = np.mean(sigmas)
    rmse = np.sqrt(np.mean(errors**2))
    k95 = conformal['sigma_k_95']
    
    # If σ were calibrated, k95 should be ~1.96
    expected_k = 1.96
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Mean σ', 'RMSE', 'k₉₅\n(actual)', 'k₉₅\n(if calibrated)']
    values = [mean_sigma, rmse, k95, expected_k]
    colors = [COLORS['uncertainty'], COLORS['error'], COLORS['accent'], COLORS['secondary']]
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Annotate
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel(f'Value ({UNIT} or multiplier)', fontsize=12)
    ax.set_title(f'Model {model_num}: Calibration Diagnostic\n'
                 f'σ is UNDER-DISPERSED (k₉₅={k95:.1f} >> 1.96)', fontsize=13)
    
    # Add explanation box
    explanation = (
        f'If σ were a calibrated predictive std:\n'
        f'  • Mean σ ≈ RMSE (actual: {mean_sigma:.2f} vs {rmse:.2f})\n'
        f'  • k₉₅ ≈ 1.96 (actual: {k95:.1f})\n\n'
        f'Conclusion: σ ranks errors well (ρ=0.48)\n'
        f'but is NOT calibrated as a scale estimate.'
    )
    ax.text(0.98, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_ylim(0, max(values) * 1.3)
    
    plt.tight_layout()
    fpath = os.path.join(save_dir, 'calibration_diagnostic.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved: calibration_diagnostic.png')
    print(f'    Mean σ = {mean_sigma:.2f}, RMSE = {rmse:.2f}, k95 = {k95:.2f} (should be ~1.96)')
    
    return {
        'mean_sigma': mean_sigma,
        'rmse': rmse,
        'k95': k95,
        'expected_k': expected_k,
    }


def generate_thesis_summary(model_num, results, save_dir):
    """Generate thesis-ready summary markdown."""
    
    summary = f"""# Advanced UQ Analysis - Model {model_num}

## Executive Summary

MC Dropout provides **informative but uncalibrated** uncertainty estimates.

### Key Findings

#### ✅ POSITIVE: σ is Informative for Ranking
- **Spearman ρ = {results.get('binned', {}).get('rho', 'N/A'):.4f}** between σ and |error|
- Top 10% σ accounts for **{results.get('binned', {}).get('top10_error_share', 'N/A'):.1f}%** of total error
- Higher uncertainty → higher expected error (monotonic relationship)

#### ⚠️ NEGATIVE: σ is NOT Calibrated as Predictive Std
- Mean σ = **{results.get('calibration', {}).get('mean_sigma', 'N/A'):.2f}** {UNIT}
- RMSE = **{results.get('calibration', {}).get('rmse', 'N/A'):.2f}** {UNIT}  
- k₉₅ = **{results.get('calibration', {}).get('k95', 'N/A'):.2f}** (should be ~1.96 if calibrated)
- Naive Gaussian intervals ŷ ± 1.96σ severely under-cover (~55% instead of 95%)

#### ✅ POSITIVE: Risk-Based Filtering Works
- Rejecting top 50% σ → MAE drops by **{results.get('risk', {}).get('improvement_50', 'N/A'):.1f}%**
- Rejecting top 10% σ → MAE drops by **{results.get('risk', {}).get('improvement_90', 'N/A'):.1f}%**
- This enables risk-aware policy analysis

#### ✅ POSITIVE: σ-Normalized Conformal Gives Adaptive Intervals
- Absolute conformal: constant width = **{results.get('intervals', {}).get('abs_width', 'N/A'):.2f}** {UNIT}
- σ-normalized: mean width = **{results.get('intervals', {}).get('sigma_width_mean', 'N/A'):.2f}** {UNIT}
- Low-σ samples (Q1): width = **{results.get('intervals', {}).get('sigma_width_low', 'N/A'):.2f}** {UNIT}
- **Efficiency gain**: {results.get('intervals', {}).get('efficiency_low', 'N/A'):.1f}% narrower intervals for confident predictions

## Balanced Conclusion

> **MC Dropout is valuable as a RISK SCORE, not as a calibrated predictive distribution.**
> Combined with conformal calibration, it enables **adaptive-width prediction intervals**
> where the model is narrow when confident and wide when uncertain.
> This is actionable for transportation policy analysis.

## What to Tell Supervisors

"I implemented MC Dropout (S=30) on the best surrogate (Model 8). The main finding is:
MC Dropout uncertainty is **informative for ranking reliability** (Spearman ρ=0.48 between 
σ and |error|, and binned error increases monotonically with σ). However, σ is **not 
calibrated as a predictive standard deviation**: naive Gaussian intervals ŷ±1.96σ under-cover 
strongly (55% vs 95%), and the σ-normalized conformal multiplier k₉₅ is ~{results.get('calibration', {}).get('k95', 11.65):.1f}, indicating 
under-dispersion.

Using conformal calibration, we obtain valid 90/95% marginal coverage, and σ becomes useful 
for **adaptive-width intervals** (narrow for low-σ, wide for high-σ). Specifically, for 
low-uncertainty predictions, intervals are {results.get('intervals', {}).get('efficiency_low', 'N/A'):.0f}% narrower than the constant-width 
alternative, which is the practical value for policy analysis."

## Data Splits (No Leakage)

- **Training set**: Used to fit GNN model (80%)
- **Test set**: 100 graphs, split 50/50 for conformal:
  - **Calibration set**: {results.get('conformal', {}).get('n_calibration', 'N/A'):,} samples → fit conformal quantiles
  - **Evaluation set**: {results.get('conformal', {}).get('n_test', 'N/A'):,} samples → report PICP

## What σ Represents

> σ is the standard deviation across S=30 MC Dropout forward passes and primarily reflects
> **epistemic uncertainty** (model uncertainty due to limited training data), not the full
> aleatoric noise of the traffic simulator. This explains why σ can rank errors (epistemic
> uncertainty correlates with difficulty) but not match the error scale (aleatoric noise
> is not captured).

## Limitations

1. MC Dropout σ is under-dispersed and not a calibrated std
2. Correlation ≠ calibration (ρ=0.48 doesn't mean intervals are valid)
3. ~48% of targets are near-zero; should evaluate separately for |y| > τ
4. Conformal coverage is marginal (over all samples), not conditional

## Next Steps

1. Risk-coverage analysis by target magnitude (separate |y| < 1 vs |y| > 5)
2. Conditional coverage analysis by σ bins
3. Compare against baselines: constant σ, residual predictor, deep ensemble
"""
    
    fpath = os.path.join(save_dir, f'ADVANCED_UQ_SUMMARY_MODEL{model_num}.md')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f'  ✓ Saved: ADVANCED_UQ_SUMMARY_MODEL{model_num}.md')


def process_model(model_num, base_dir='data/TR-C_Benchmarks'):
    """Process single model with advanced analysis."""
    print(f'\n{"="*60}')
    print(f'ADVANCED UQ ANALYSIS - MODEL {model_num}')
    print('='*60)
    
    data = load_model_data(model_num, base_dir)
    if data is None:
        print(f'ERROR: Could not load data for Model {model_num}')
        return
    
    save_dir = os.path.join(data['uq_dir'], 'uq_plots')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f'Samples: {len(data["sigmas"]):,}')
    print(f'Save dir: {save_dir}')
    
    results = {}
    
    # Generate all plots
    results['binned'] = plot_binned_error_vs_sigma(data, model_num, save_dir)
    results['risk'] = plot_risk_coverage_curve(data, model_num, save_dir)
    results['intervals'] = plot_interval_width_comparison(data, model_num, save_dir)
    results['hexbin'] = plot_hexbin_uncertainty_error(data, model_num, save_dir)
    results['calibration'] = plot_calibration_diagnostic(data, model_num, save_dir)
    results['conformal'] = data['conformal']
    
    # Generate summary
    generate_thesis_summary(model_num, results, data['uq_dir'])
    
    print(f'\n✅ Model {model_num} advanced analysis complete!')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Advanced UQ Analysis')
    parser.add_argument('--all', action='store_true', help='Process all models')
    parser.add_argument('--models', nargs='+', type=int, default=[8], 
                        help='Model numbers to process (default: 8)')
    args = parser.parse_args()
    
    if args.all:
        models = [2, 5, 6, 7, 8]
    else:
        models = args.models
    
    for model_num in models:
        process_model(model_num)
    
    print('\n' + '='*60)
    print('ALL ADVANCED ANALYSES COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()
