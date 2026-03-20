#!/usr/bin/env python
"""
Standardized UQ Plotting Script for GNN Models
==============================================

Creates consistent plots across all models for thesis/publication.

Canonical Plot Set:
  1. pred_vs_target.png              - Prediction scatter with 1:1 line
  2. uncertainty_hist.png            - Distribution of predictive uncertainty (sigma)
  3. error_hist.png                  - Distribution of absolute errors
  4. uncertainty_vs_error_scatter.png - Scatter showing uncertainty-error correlation
  5. binned_error_vs_uncertainty.png - Binned analysis of error vs uncertainty
  6. coverage_curve_k_sigma.png      - Coverage vs k (for intervals y_hat +/- k*sigma)
  7. coverage_comparison.png         - Bar chart comparing 90%/95% coverage (both methods)
  8. uq_dashboard.png                - Combined 2x2 summary dashboard

Usage:
  python plot_uq_standard.py --model-dir <path> --model-num <N>
  python plot_uq_standard.py --model-dir data/TR-C_Benchmarks/point_net_transf_gat_2nd_try --model-num 2

Author: Auto-generated for thesis standardization
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Unit label for plots - Elena confirmed: vehicles per hour
UNIT = "veh/h"  # traffic flow difference in vehicles per hour

# Nice academic color palette
COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Raspberry
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red-orange
    'neutral': '#3B3B3B',      # Dark gray
    'light': '#E8E8E8',        # Light gray
    'det': '#2E86AB',          # Deterministic - blue
    'mc': '#28A745',           # MC Dropout - green
    'error': '#DC3545',        # Error - red
    'uncertainty': '#6F42C1',  # Uncertainty - purple
}

# Academic style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def find_best_mc_npz(uq_dir):
    """
    Find the best MC dropout NPZ file (largest file = full run).
    Avoids accidentally picking partial runs.
    """
    candidates = []
    for f in os.listdir(uq_dir):
        if 'mc_dropout' in f.lower() and f.endswith('.npz'):
            fp = os.path.join(uq_dir, f)
            candidates.append(fp)
    
    if not candidates:
        return None
    
    # Prefer largest file (full runs are bigger than partial)
    best = max(candidates, key=lambda p: os.path.getsize(p))
    print(f"  Selected NPZ: {os.path.basename(best)} ({os.path.getsize(best)/1e6:.1f} MB)")
    return best


def find_deterministic_npz(uq_dir):
    """Find deterministic results NPZ."""
    for f in os.listdir(uq_dir):
        if 'deterministic' in f.lower() and f.endswith('.npz'):
            return os.path.join(uq_dir, f)
    return None


def compute_conformal_metrics(targets, predictions, uncertainties, split_ratio=0.5, seed=42):
    """
    Compute conformal prediction metrics using BOTH methods:
      1. Absolute residual: q = quantile(|y - y_hat|)  -> interval [y_hat - q, y_hat + q]
      2. Sigma-normalized: k = quantile(|y - y_hat| / sigma) -> interval [y_hat - k*sigma, y_hat + k*sigma]
    
    Returns dict with both methods' results.
    """
    np.random.seed(seed)
    n = len(targets)
    idx = np.random.permutation(n)
    n_cal = int(n * split_ratio)
    
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]
    
    # Calibration set
    cal_targets = targets[cal_idx]
    cal_preds = predictions[cal_idx]
    cal_sigma = uncertainties[cal_idx]
    cal_errors = np.abs(cal_targets - cal_preds)
    
    # Test set
    test_targets = targets[test_idx]
    test_preds = predictions[test_idx]
    test_sigma = uncertainties[test_idx]
    test_errors = np.abs(test_targets - test_preds)
    
    results = {
        'n_calibration': n_cal,
        'n_test': len(test_idx),
        'split_ratio': split_ratio,
        'seed': seed,
    }
    
    # ===== Method 1: Absolute Residual (q) =====
    for alpha, name in [(0.90, '90'), (0.95, '95')]:
        q = np.quantile(cal_errors, alpha)
        covered = test_errors <= q
        picp = covered.mean() * 100
        avg_width = 2 * q  # interval width = 2q
        
        results[f'absolute_q_{name}'] = float(q)
        results[f'absolute_picp_{name}'] = float(picp)
        results[f'absolute_width_{name}'] = float(avg_width)
    
    # ===== Method 2: Sigma-Normalized (k*sigma) =====
    eps = 1e-6
    cal_normalized = cal_errors / (cal_sigma + eps)
    
    for alpha, name in [(0.90, '90'), (0.95, '95')]:
        k = np.quantile(cal_normalized, alpha)
        test_intervals = k * test_sigma
        covered = test_errors <= test_intervals
        picp = covered.mean() * 100
        avg_width = 2 * k * test_sigma.mean()
        
        results[f'sigma_k_{name}'] = float(k)
        results[f'sigma_picp_{name}'] = float(picp)
        results[f'sigma_width_{name}'] = float(avg_width)
    
    # For coverage curve: store test set info
    results['test_errors'] = test_errors
    results['test_sigma'] = test_sigma
    results['cal_errors'] = cal_errors
    results['cal_sigma'] = cal_sigma
    
    return results


def plot_pred_vs_target(targets, predictions, model_num, save_path, sample_size=50000):
    """Plot 1: Predictions vs Targets scatter."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Subsample for visualization
    if len(targets) > sample_size:
        idx = np.random.choice(len(targets), sample_size, replace=False)
        t_sub, p_sub = targets[idx], predictions[idx]
    else:
        t_sub, p_sub = targets, predictions
    
    ax.scatter(t_sub, p_sub, alpha=0.2, s=3, c=COLORS['primary'], rasterized=True)
    
    # 1:1 line
    lims = [min(t_sub.min(), p_sub.min()), max(t_sub.max(), p_sub.max())]
    ax.plot(lims, lims, color=COLORS['error'], linestyle='--', lw=1.5, label='Perfect prediction (y=x)')
    
    # Metrics
    r2 = 1 - np.sum((targets - predictions)**2) / np.sum((targets - targets.mean())**2)
    mae = np.mean(np.abs(targets - predictions))
    
    ax.set_xlabel(f'Target Traffic Flow Difference ({UNIT})')
    ax.set_ylabel(f'Predicted Traffic Flow Difference ({UNIT})')
    ax.set_title(f'Model {model_num}: Predictions vs Targets')
    ax.legend(loc='upper left')
    
    # Text box with metrics
    textstr = f'$R^2$ = {r2:.4f}\nMAE = {mae:.2f} {UNIT}\nN = {len(targets):,}'
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['light']))
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_uncertainty_hist(uncertainties, model_num, save_path):
    """Plot 2: Uncertainty (sigma) distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.hist(uncertainties, bins=100, color=COLORS['uncertainty'], alpha=0.7, edgecolor='white')
    ax.axvline(uncertainties.mean(), color=COLORS['error'], linestyle='--', lw=2, 
               label=f'Mean = {uncertainties.mean():.3f} {UNIT}')
    ax.axvline(np.median(uncertainties), color=COLORS['accent'], linestyle=':', lw=2,
               label=f'Median = {np.median(uncertainties):.3f} {UNIT}')
    
    ax.set_xlabel(f'Predictive Uncertainty $\\sigma$ ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Model {model_num}: Uncertainty Distribution (MC Dropout)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_error_hist(targets, predictions, model_num, save_path):
    """Plot 3: Absolute error distribution histogram."""
    errors = np.abs(targets - predictions)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.hist(errors, bins=100, color=COLORS['error'], alpha=0.7, edgecolor='white')
    ax.axvline(errors.mean(), color=COLORS['neutral'], linestyle='--', lw=2,
               label=f'MAE = {errors.mean():.3f} {UNIT}')
    ax.axvline(np.median(errors), color=COLORS['accent'], linestyle=':', lw=2,
               label=f'Median = {np.median(errors):.3f} {UNIT}')
    
    ax.set_xlabel(f'Absolute Prediction Error ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Model {model_num}: Prediction Error Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_uncertainty_vs_error_scatter(targets, predictions, uncertainties, model_num, save_path, sample_size=50000):
    """Plot 4: Uncertainty vs Error scatter (the key correlation plot)."""
    errors = np.abs(targets - predictions)
    
    # Subsample
    if len(errors) > sample_size:
        idx = np.random.choice(len(errors), sample_size, replace=False)
        e_sub, u_sub = errors[idx], uncertainties[idx]
    else:
        e_sub, u_sub = errors, uncertainties
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.scatter(u_sub, e_sub, alpha=0.15, s=3, c=COLORS['uncertainty'], rasterized=True)
    
    # Trend line (flatten arrays to ensure 1D)
    z = np.polyfit(uncertainties.ravel(), errors.ravel(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(uncertainties.min(), uncertainties.max(), 100)
    ax.plot(x_line, p(x_line), color=COLORS['error'], lw=2, label=f'Linear fit (slope={z[0]:.2f})')
    
    # Spearman correlation
    rho, pval = stats.spearmanr(uncertainties, errors)
    
    ax.set_xlabel(f'Predictive Uncertainty $\\sigma$ ({UNIT})')
    ax.set_ylabel(f'Absolute Prediction Error ({UNIT})')
    ax.set_title(f'Model {model_num}: Uncertainty vs Error Correlation')
    ax.legend(loc='upper left')
    
    textstr = f'Spearman $\\rho$ = {rho:.4f}\np-value < {max(pval, 1e-10):.1e}'
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['light']))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_binned_error_vs_uncertainty(targets, predictions, uncertainties, model_num, save_path, n_bins=20):
    """Plot 5: Binned error statistics by uncertainty quantile."""
    errors = np.abs(targets - predictions)
    
    # Create uncertainty bins
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_mean_errors = []
    bin_std_errors = []
    
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if i == n_bins - 1:  # Include upper edge for last bin
            mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1])
        
        if mask.sum() > 0:
            bin_centers.append(uncertainties[mask].mean())
            bin_mean_errors.append(errors[mask].mean())
            bin_std_errors.append(errors[mask].std())
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.errorbar(bin_centers, bin_mean_errors, yerr=bin_std_errors, 
                fmt='o-', color='steelblue', capsize=3, capthick=1.5,
                markersize=6, label='Mean error per bin')
    
    # Perfect calibration line (error = uncertainty)
    xlims = ax.get_xlim()
    ax.plot(xlims, xlims, 'r--', alpha=0.7, label='Perfect calibration')
    ax.set_xlim(xlims)
    
    ax.set_xlabel(f'Mean Uncertainty $\\sigma$ in Bin ({UNIT})')
    ax.set_ylabel(f'Mean Absolute Error ({UNIT})')
    ax.set_title(f'Model {model_num}: Binned Error vs Uncertainty ({n_bins} bins)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_coverage_curve_k_sigma(conformal_results, model_num, save_path):
    """
    Plot 6: Coverage vs k curve (for sigma-normalized method).
    
    This plot shows how coverage changes with k in intervals [y_hat - k*sigma, y_hat + k*sigma].
    Critical for demonstrating Gaussian assumption failure.
    """
    test_errors = conformal_results['test_errors']
    test_sigma = conformal_results['test_sigma']
    cal_errors = conformal_results['cal_errors']
    cal_sigma = conformal_results['cal_sigma']
    
    eps = 1e-6
    
    # Compute coverage for range of k values
    k_values = np.linspace(0.5, 4.0, 50)
    empirical_coverages = []
    gaussian_coverages = []
    
    for k in k_values:
        # Empirical coverage on test set
        covered = test_errors <= (k * test_sigma)
        empirical_coverages.append(covered.mean() * 100)
        
        # Theoretical Gaussian coverage: 2*Phi(k) - 1
        gaussian_cov = 2 * stats.norm.cdf(k) - 1
        gaussian_coverages.append(gaussian_cov * 100)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(k_values, empirical_coverages, 'b-', lw=2, label='Empirical coverage')
    ax.plot(k_values, gaussian_coverages, 'r--', lw=2, label='Gaussian assumption')
    
    # Mark key points
    for k_ref, label in [(1.0, 'k=1'), (1.65, 'k=1.65'), (1.96, 'k=1.96'), (2.0, 'k=2')]:
        emp_cov = np.interp(k_ref, k_values, empirical_coverages)
        gauss_cov = (2 * stats.norm.cdf(k_ref) - 1) * 100
        ax.axvline(k_ref, color='gray', alpha=0.3, linestyle=':')
        ax.plot(k_ref, emp_cov, 'bo', markersize=6)
        ax.plot(k_ref, gauss_cov, 'r^', markersize=6)
    
    ax.axhline(90, color='green', alpha=0.5, linestyle='--', label='90% target')
    ax.axhline(95, color='purple', alpha=0.5, linestyle='--', label='95% target')
    
    ax.set_xlabel('Multiplier $k$')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(f'Model {model_num}: Coverage Curve ($\\hat{{y}} \\pm k\\sigma$)')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 100])
    
    # Add gap annotation
    k_196 = 1.96
    emp_at_196 = np.interp(k_196, k_values, empirical_coverages)
    gap = 95 - emp_at_196
    ax.annotate(f'Gap at k=1.96: {gap:.1f}%', xy=(k_196, emp_at_196),
                xytext=(k_196 + 0.3, emp_at_196 - 10),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_coverage_comparison(conformal_results, model_num, save_path):
    """Plot 7: Bar chart comparing coverage for both methods at 90% and 95%."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['Absolute (q)', 'Sigma-norm (k*σ)']
    x = np.arange(len(methods))
    width = 0.35
    
    # 90% coverage
    cov_90 = [
        conformal_results.get('absolute_picp_90', np.nan),
        conformal_results.get('sigma_picp_90', np.nan)
    ]
    # 95% coverage
    cov_95 = [
        conformal_results.get('absolute_picp_95', np.nan),
        conformal_results.get('sigma_picp_95', np.nan)
    ]
    
    bars1 = ax.bar(x - width/2, cov_90, width, label='90% nominal', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, cov_95, width, label='95% nominal', color='coral', alpha=0.8)
    
    # Target lines
    ax.axhline(90, color='steelblue', linestyle='--', alpha=0.7)
    ax.axhline(95, color='coral', linestyle='--', alpha=0.7)
    
    # Value labels
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Conformal Method')
    ax.set_ylabel('PICP (%)')
    ax.set_title(f'Model {model_num}: Conformal Coverage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_uq_dashboard(targets, predictions, uncertainties, conformal_results, model_num, save_path, sample_size=30000):
    """Plot 8: Combined 2x2 dashboard for quick overview."""
    errors = np.abs(targets - predictions)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subsample
    if len(errors) > sample_size:
        idx = np.random.choice(len(errors), sample_size, replace=False)
        t_sub, p_sub, u_sub, e_sub = targets[idx], predictions[idx], uncertainties[idx], errors[idx]
    else:
        t_sub, p_sub, u_sub, e_sub = targets, predictions, uncertainties, errors
    
    # (a) Predictions vs Targets
    ax = axes[0, 0]
    ax.scatter(t_sub, p_sub, alpha=0.1, s=2, c='steelblue', rasterized=True)
    lims = [min(t_sub.min(), p_sub.min()), max(t_sub.max(), p_sub.max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    r2 = 1 - np.sum((targets - predictions)**2) / np.sum((targets - targets.mean())**2)
    ax.set_xlabel(f'Target ({UNIT})')
    ax.set_ylabel(f'Predicted ({UNIT})')
    ax.set_title(f'(a) Predictions vs Targets ($R^2$={r2:.4f})')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # (b) Uncertainty vs Error
    ax = axes[0, 1]
    ax.scatter(u_sub, e_sub, alpha=0.1, s=2, c='steelblue', rasterized=True)
    rho = stats.spearmanr(uncertainties, errors)[0]
    ax.set_xlabel(f'Uncertainty $\\sigma$ ({UNIT})')
    ax.set_ylabel(f'Absolute Error ({UNIT})')
    ax.set_title(f'(b) Uncertainty vs Error (Spearman={rho:.4f})')
    
    # (c) Uncertainty histogram
    ax = axes[1, 0]
    ax.hist(uncertainties, bins=80, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(uncertainties.mean(), color='red', linestyle='--', lw=2)
    ax.set_xlabel(f'Uncertainty $\\sigma$ ({UNIT})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'(c) Uncertainty Distribution (mean={uncertainties.mean():.3f})')
    
    # (d) Coverage comparison
    ax = axes[1, 1]
    methods = ['Abs-90%', 'Abs-95%', 'Sig-90%', 'Sig-95%']
    coverages = [
        conformal_results.get('absolute_picp_90', np.nan),
        conformal_results.get('absolute_picp_95', np.nan),
        conformal_results.get('sigma_picp_90', np.nan),
        conformal_results.get('sigma_picp_95', np.nan)
    ]
    colors = ['steelblue', 'steelblue', 'coral', 'coral']
    bars = ax.bar(methods, coverages, color=colors, alpha=0.8)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(95, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('PICP (%)')
    ax.set_title('(d) Conformal Coverage')
    ax.set_ylim([0, 105])
    for bar, cov in zip(bars, coverages):
        if not np.isnan(cov):
            ax.annotate(f'{cov:.1f}%', xy=(bar.get_x() + bar.get_width()/2, cov),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f'Model {model_num}: UQ Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def generate_standard_uq_plots(model_dir, model_num, sample_size=50000, split_ratio=0.5, seed=42):
    """
    Generate all 8 canonical UQ plots for a model.
    
    Args:
        model_dir: Path to model directory (e.g., 'data/TR-C_Benchmarks/point_net_transf_gat_2nd_try')
        model_num: Model number for titles
        sample_size: Max points to plot in scatter plots
        split_ratio: Calibration/test split for conformal (default 50/50)
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"GENERATING STANDARD UQ PLOTS - MODEL {model_num}")
    print(f"{'='*60}")
    
    uq_dir = os.path.join(model_dir, 'uq_results')
    plot_dir = os.path.join(uq_dir, 'uq_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Find MC dropout NPZ (prefer largest = full run)
    print("\n[1/4] Loading MC Dropout results...")
    mc_file = find_best_mc_npz(uq_dir)
    if mc_file is None:
        print(f"ERROR: No MC dropout NPZ found in {uq_dir}")
        return False
    
    mc_data = np.load(mc_file)
    predictions = mc_data['predictions']
    uncertainties = mc_data['uncertainties']
    targets = mc_data['targets']
    
    print(f"  Total nodes: {len(predictions):,}")
    print(f"  Mean sigma: {uncertainties.mean():.4f} {UNIT}")
    print(f"  Mean error: {np.abs(targets - predictions).mean():.4f} {UNIT}")
    
    # Compute conformal metrics (both methods)
    print("\n[2/4] Computing conformal prediction metrics...")
    conformal = compute_conformal_metrics(targets, predictions, uncertainties, 
                                          split_ratio=split_ratio, seed=seed)
    
    print(f"  Absolute method - 90%: q={conformal['absolute_q_90']:.4f}, PICP={conformal['absolute_picp_90']:.2f}%")
    print(f"  Absolute method - 95%: q={conformal['absolute_q_95']:.4f}, PICP={conformal['absolute_picp_95']:.2f}%")
    print(f"  Sigma-norm method - 90%: k={conformal['sigma_k_90']:.4f}, PICP={conformal['sigma_picp_90']:.2f}%")
    print(f"  Sigma-norm method - 95%: k={conformal['sigma_k_95']:.4f}, PICP={conformal['sigma_picp_95']:.2f}%")
    
    # Save conformal metrics
    conformal_save = {k: v for k, v in conformal.items() if not isinstance(v, np.ndarray)}
    with open(os.path.join(uq_dir, 'conformal_standard.json'), 'w') as f:
        json.dump(conformal_save, f, indent=2)
    print(f"  Saved: conformal_standard.json")
    
    # Generate all 8 plots
    print("\n[3/4] Generating plots...")
    np.random.seed(seed)
    
    plot_pred_vs_target(targets, predictions, model_num,
                        os.path.join(plot_dir, 'pred_vs_target.png'), sample_size)
    
    plot_uncertainty_hist(uncertainties, model_num,
                          os.path.join(plot_dir, 'uncertainty_hist.png'))
    
    plot_error_hist(targets, predictions, model_num,
                    os.path.join(plot_dir, 'error_hist.png'))
    
    plot_uncertainty_vs_error_scatter(targets, predictions, uncertainties, model_num,
                                      os.path.join(plot_dir, 'uncertainty_vs_error_scatter.png'), sample_size)
    
    plot_binned_error_vs_uncertainty(targets, predictions, uncertainties, model_num,
                                     os.path.join(plot_dir, 'binned_error_vs_uncertainty.png'))
    
    plot_coverage_curve_k_sigma(conformal, model_num,
                                os.path.join(plot_dir, 'coverage_curve_k_sigma.png'))
    
    plot_coverage_comparison(conformal, model_num,
                             os.path.join(plot_dir, 'coverage_comparison.png'))
    
    plot_uq_dashboard(targets, predictions, uncertainties, conformal, model_num,
                      os.path.join(plot_dir, 'uq_dashboard.png'), sample_size)
    
    print("\n[4/4] Summary")
    print(f"  Output directory: {plot_dir}")
    print(f"  8 canonical plots generated successfully!")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate standardized UQ plots for GNN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_uq_standard.py --model-dir data/TR-C_Benchmarks/point_net_transf_gat_2nd_try --model-num 2
  python plot_uq_standard.py --model-dir data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout --model-num 8
        """
    )
    parser.add_argument('--model-dir', required=True, help='Path to model directory')
    parser.add_argument('--model-num', type=int, required=True, help='Model number for plot titles')
    parser.add_argument('--sample-size', type=int, default=50000, help='Max points in scatter plots')
    parser.add_argument('--split', type=float, default=0.5, help='Calibration/test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    success = generate_standard_uq_plots(
        args.model_dir, 
        args.model_num,
        sample_size=args.sample_size,
        split_ratio=args.split,
        seed=args.seed
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
