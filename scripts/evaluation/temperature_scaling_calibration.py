"""
Temperature Scaling Calibration for MC Dropout Uncertainty

This script implements post-hoc calibration using temperature scaling
to improve the Expected Calibration Error (ECE) from ~0.38 to ~0.10

Method:
1. Learn a single temperature parameter T on validation data
2. Scale uncertainties: calibrated_σ = raw_σ * T
3. Evaluate on test data

Author: Nazim Zaman
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
from pathlib import Path

# Paths
_REPO = Path(__file__).resolve().parent.parent.parent
MODEL_FOLDER = str(_REPO / 'data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout')
FIGURES_DIR = str(_REPO / 'thesis/latex_tum_official/figures')

# Light pastel color scheme (consistent with thesis)
COLORS = {
    'primary': '#89CFF0',      # Baby blue
    'secondary': '#FFD1DC',    # Pastel pink
    'success': '#98FB98',      # Pale green
    'accent': '#E6E6FA',       # Lavender
    'warning': '#FFEAA7',      # Pastel yellow
    'dark': '#2C3E50',         # Dark blue-gray for text
}


def load_data():
    """Load pre-computed MC Dropout data"""
    npz_path = os.path.join(MODEL_FOLDER, 'uq_results/mc_dropout_full_100graphs_mc30.npz')
    data = np.load(npz_path)
    
    predictions = data['predictions'].flatten()
    uncertainties = data['uncertainties'].flatten()
    targets = data['targets'].flatten()
    
    errors = np.abs(predictions - targets)
    
    print(f"Loaded {len(predictions):,} predictions")
    print(f"Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
    print(f"Error range: [{errors.min():.2f}, {errors.max():.2f}]")
    
    return predictions, uncertainties, targets, errors


def compute_ece(uncertainties, errors, n_bins=10):
    """
    Compute Expected Calibration Error for regression.
    
    For regression, we check if the error falls within uncertainty intervals.
    We define calibration as: for predictions with uncertainty σ,
    about 68% should have |error| < σ (1-sigma rule)
    """
    # Bin by uncertainty
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # Remove duplicates
    
    ece = 0.0
    total_samples = len(uncertainties)
    calibration_data = []
    
    for i in range(len(bin_edges) - 1):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if i == len(bin_edges) - 2:  # Include last edge
            mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1])
        
        if mask.sum() == 0:
            continue
            
        bin_unc = uncertainties[mask]
        bin_err = errors[mask]
        
        # For each prediction, check if error < uncertainty (scaled by coverage factor)
        # For 68% coverage (1 sigma), we check |error| < σ
        observed_coverage = np.mean(bin_err < bin_unc)
        expected_coverage = 0.68  # 1-sigma coverage
        
        bin_weight = mask.sum() / total_samples
        ece += bin_weight * np.abs(observed_coverage - expected_coverage)
        
        calibration_data.append({
            'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
            'observed': observed_coverage,
            'expected': expected_coverage,
            'n_samples': mask.sum()
        })
    
    return ece, calibration_data


def compute_ece_scaled(T, uncertainties, errors):
    """Compute ECE with scaled uncertainties"""
    scaled_unc = uncertainties * T
    ece, _ = compute_ece(scaled_unc, errors)
    return ece


def find_optimal_temperature(uncertainties, errors):
    """Find optimal temperature T using grid search then fine-tuning"""
    print("\n" + "="*60)
    print("Finding Optimal Temperature")
    print("="*60)
    
    # Grid search
    T_values = np.logspace(-1, 2, 50)  # 0.1 to 100
    ece_values = []
    
    for T in T_values:
        ece = compute_ece_scaled(T, uncertainties, errors)
        ece_values.append(ece)
    
    best_idx = np.argmin(ece_values)
    T_init = T_values[best_idx]
    
    print(f"Grid search best: T={T_init:.4f}, ECE={ece_values[best_idx]:.4f}")
    
    # Fine-tune with optimization
    result = minimize_scalar(
        lambda T: compute_ece_scaled(T, uncertainties, errors),
        bounds=(T_init * 0.5, T_init * 2),
        method='bounded'
    )
    
    T_optimal = result.x
    ece_optimal = result.fun
    
    print(f"Optimized: T={T_optimal:.4f}, ECE={ece_optimal:.4f}")
    
    return T_optimal, T_values, ece_values


def evaluate_calibration(uncertainties, errors, T=1.0, label=""):
    """Evaluate calibration with coverage analysis"""
    scaled_unc = uncertainties * T
    
    # Compute coverage at different sigma levels
    coverages = {}
    expected = {1: 0.683, 2: 0.954, 3: 0.997}  # Normal distribution
    
    for sigma in [1, 2, 3]:
        coverage = np.mean(errors < sigma * scaled_unc)
        coverages[sigma] = coverage
    
    print(f"\n{label} Calibration Results (T={T:.4f}):")
    print("-" * 50)
    for sigma in [1, 2, 3]:
        diff = coverages[sigma] - expected[sigma]
        status = "✓" if abs(diff) < 0.05 else ("↑ over" if diff > 0 else "↓ under")
        print(f"  {sigma}σ coverage: {coverages[sigma]:.3f} (expected {expected[sigma]:.3f}) {status}")
    
    ece, _ = compute_ece(scaled_unc, errors)
    print(f"  ECE: {ece:.4f}")
    
    return coverages, ece


def create_calibration_figure(uncertainties, errors, T_optimal, T_values, ece_values):
    """Create figure showing calibration improvement"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Temperature vs ECE
    ax1 = axes[0, 0]
    ax1.semilogx(T_values, ece_values, linewidth=2, color=COLORS['dark'])
    ax1.axvline(T_optimal, color=COLORS['primary'], linestyle='--', linewidth=2, 
                label=f'Optimal T={T_optimal:.2f}')
    ax1.axvline(1.0, color=COLORS['warning'], linestyle=':', linewidth=2, 
                label='Original T=1.0')
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Expected Calibration Error', fontsize=12)
    ax1.set_title('Temperature Scaling Optimization', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # 2. Coverage comparison
    ax2 = axes[0, 1]
    sigmas = [1, 2, 3]
    expected = [0.683, 0.954, 0.997]
    
    # Original coverage
    original_cov = [np.mean(errors < s * uncertainties) for s in sigmas]
    # Calibrated coverage
    calibrated_cov = [np.mean(errors < s * uncertainties * T_optimal) for s in sigmas]
    
    x = np.arange(len(sigmas))
    width = 0.25
    
    bars1 = ax2.bar(x - width, expected, width, label='Expected (Normal)', 
                    color=COLORS['accent'], edgecolor=COLORS['dark'])
    bars2 = ax2.bar(x, original_cov, width, label='Original (T=1)', 
                    color=COLORS['warning'], edgecolor=COLORS['dark'])
    bars3 = ax2.bar(x + width, calibrated_cov, width, label=f'Calibrated (T={T_optimal:.2f})', 
                    color=COLORS['success'], edgecolor=COLORS['dark'])
    
    ax2.set_xlabel('Confidence Level (σ)', fontsize=12)
    ax2.set_ylabel('Coverage Probability', fontsize=12)
    ax2.set_title('Coverage Before vs After Calibration', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['1σ (68.3%)', '2σ (95.4%)', '3σ (99.7%)'])
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 3. Reliability diagram (before)
    ax3 = axes[1, 0]
    n_bins = 10
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    
    bin_centers = []
    bin_coverages = []
    
    for i in range(len(bin_edges) - 1):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if mask.sum() > 0:
            bin_unc = uncertainties[mask].mean()
            bin_cov = np.mean(errors[mask] < uncertainties[mask])
            bin_centers.append(bin_unc)
            bin_coverages.append(bin_cov)
    
    ax3.plot([0, max(bin_centers)*1.1], [0.683, 0.683], 'k--', linewidth=2, 
             label='Perfect Calibration (68.3%)')
    ax3.scatter(bin_centers, bin_coverages, s=100, c=COLORS['warning'], 
               edgecolors=COLORS['dark'], linewidth=2, zorder=5)
    ax3.plot(bin_centers, bin_coverages, '-', color=COLORS['warning'], 
             linewidth=2, alpha=0.7, label='Original (T=1)')
    
    ax3.set_xlabel('Mean Uncertainty (σ)', fontsize=12)
    ax3.set_ylabel('Observed 1σ Coverage', fontsize=12)
    ax3.set_title('Reliability Diagram (Before)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')
    
    original_ece, _ = compute_ece(uncertainties, errors)
    ax3.text(0.95, 0.05, f'ECE = {original_ece:.4f}', transform=ax3.transAxes,
            fontsize=12, fontweight='bold', ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor=COLORS['warning'], alpha=0.8))
    
    # 4. Reliability diagram (after)
    ax4 = axes[1, 1]
    scaled_unc = uncertainties * T_optimal
    
    bin_edges_scaled = np.percentile(scaled_unc, np.linspace(0, 100, n_bins + 1))
    
    bin_centers_scaled = []
    bin_coverages_scaled = []
    
    for i in range(len(bin_edges_scaled) - 1):
        mask = (scaled_unc >= bin_edges_scaled[i]) & (scaled_unc < bin_edges_scaled[i+1])
        if mask.sum() > 0:
            bin_unc = scaled_unc[mask].mean()
            bin_cov = np.mean(errors[mask] < scaled_unc[mask])
            bin_centers_scaled.append(bin_unc)
            bin_coverages_scaled.append(bin_cov)
    
    ax4.plot([0, max(bin_centers_scaled)*1.1], [0.683, 0.683], 'k--', linewidth=2, 
             label='Perfect Calibration (68.3%)')
    ax4.scatter(bin_centers_scaled, bin_coverages_scaled, s=100, c=COLORS['success'], 
               edgecolors=COLORS['dark'], linewidth=2, zorder=5)
    ax4.plot(bin_centers_scaled, bin_coverages_scaled, '-', color=COLORS['success'], 
             linewidth=2, alpha=0.7, label=f'Calibrated (T={T_optimal:.2f})')
    
    ax4.set_xlabel('Mean Uncertainty (σ × T)', fontsize=12)
    ax4.set_ylabel('Observed 1σ Coverage', fontsize=12)
    ax4.set_title('Reliability Diagram (After)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')
    
    calibrated_ece, _ = compute_ece(scaled_unc, errors)
    ax4.text(0.95, 0.05, f'ECE = {calibrated_ece:.4f}', transform=ax4.transAxes,
            fontsize=12, fontweight='bold', ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.8))
    
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, 'uq_temperature_scaling.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {save_path}")
    plt.close()


def create_combined_calibration_figure(uncertainties, errors, T_optimal):
    """Create a single figure suitable for thesis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scaled_unc = uncertainties * T_optimal
    n_bins = 10
    
    # Before calibration
    ax1 = axes[0]
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    
    bin_centers = []
    bin_coverages = []
    
    for i in range(len(bin_edges) - 1):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if mask.sum() > 0:
            bin_unc = uncertainties[mask].mean()
            bin_cov = np.mean(errors[mask] < uncertainties[mask])
            bin_centers.append(bin_unc)
            bin_coverages.append(bin_cov)
    
    # Perfect calibration line
    ax1.axhline(0.683, color=COLORS['dark'], linestyle='--', linewidth=2, 
                label='Perfect (68.3%)')
    
    # Gap region
    for bc, cov in zip(bin_centers, bin_coverages):
        ax1.fill_between([bc-5, bc+5], [0.683, 0.683], [cov, cov], 
                        alpha=0.3, color=COLORS['secondary'])
    
    ax1.scatter(bin_centers, bin_coverages, s=120, c=COLORS['warning'], 
               edgecolors=COLORS['dark'], linewidth=2, zorder=5, label='Observed')
    ax1.plot(bin_centers, bin_coverages, '-', color=COLORS['warning'], linewidth=2, alpha=0.7)
    
    original_ece, _ = compute_ece(uncertainties, errors)
    ax1.set_xlabel('Predicted Uncertainty (σ)', fontsize=13)
    ax1.set_ylabel('Observed 1σ Coverage', fontsize=13)
    ax1.set_title(f'Before Calibration (ECE = {original_ece:.3f})', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    ax1.set_ylim(0, 1.0)
    
    # After calibration
    ax2 = axes[1]
    bin_edges_scaled = np.percentile(scaled_unc, np.linspace(0, 100, n_bins + 1))
    
    bin_centers_scaled = []
    bin_coverages_scaled = []
    
    for i in range(len(bin_edges_scaled) - 1):
        mask = (scaled_unc >= bin_edges_scaled[i]) & (scaled_unc < bin_edges_scaled[i+1])
        if mask.sum() > 0:
            bin_unc = scaled_unc[mask].mean()
            bin_cov = np.mean(errors[mask] < scaled_unc[mask])
            bin_centers_scaled.append(bin_unc)
            bin_coverages_scaled.append(bin_cov)
    
    ax2.axhline(0.683, color=COLORS['dark'], linestyle='--', linewidth=2, 
                label='Perfect (68.3%)')
    
    ax2.scatter(bin_centers_scaled, bin_coverages_scaled, s=120, c=COLORS['success'], 
               edgecolors=COLORS['dark'], linewidth=2, zorder=5, label='Observed')
    ax2.plot(bin_centers_scaled, bin_coverages_scaled, '-', color=COLORS['success'], 
             linewidth=2, alpha=0.7)
    
    calibrated_ece, _ = compute_ece(scaled_unc, errors)
    ax2.set_xlabel(f'Calibrated Uncertainty (σ × {T_optimal:.2f})', fontsize=13)
    ax2.set_ylabel('Observed 1σ Coverage', fontsize=13)
    ax2.set_title(f'After Calibration (ECE = {calibrated_ece:.3f})', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_ylim(0, 1.0)
    
    # Add improvement annotation
    improvement = (original_ece - calibrated_ece) / original_ece * 100
    fig.text(0.5, 0.02, f'Temperature T = {T_optimal:.2f}  |  ECE Improvement: {improvement:.1f}%', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=COLORS['accent'], alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    save_path = os.path.join(FIGURES_DIR, 'uq_calibration_improvement.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 70)
    
    # Load data
    predictions, uncertainties, targets, errors = load_data()
    
    # Split into validation (30%) and test (70%)
    np.random.seed(42)
    n_samples = len(predictions)
    indices = np.random.permutation(n_samples)
    
    val_size = int(0.3 * n_samples)
    val_idx, test_idx = indices[:val_size], indices[val_size:]
    
    val_unc = uncertainties[val_idx]
    val_err = errors[val_idx]
    test_unc = uncertainties[test_idx]
    test_err = errors[test_idx]
    
    print(f"\nValidation set: {len(val_idx):,} samples")
    print(f"Test set: {len(test_idx):,} samples")
    
    # Evaluate original calibration
    evaluate_calibration(test_unc, test_err, T=1.0, label="Original")
    
    # Find optimal temperature on validation set
    T_optimal, T_values, ece_values = find_optimal_temperature(val_unc, val_err)
    
    # Evaluate calibrated on test set
    evaluate_calibration(test_unc, test_err, T=T_optimal, label="Calibrated")
    
    # Create figures (using all data for visualization)
    print("\n" + "="*60)
    print("Creating Figures")
    print("="*60)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    create_calibration_figure(uncertainties, errors, T_optimal, T_values, ece_values)
    create_combined_calibration_figure(uncertainties, errors, T_optimal)
    
    # Summary
    original_ece, _ = compute_ece(uncertainties, errors)
    calibrated_ece, _ = compute_ece(uncertainties * T_optimal, errors)
    improvement = (original_ece - calibrated_ece) / original_ece * 100
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Optimal Temperature: T = {T_optimal:.4f}")
    print(f"  Original ECE: {original_ece:.4f}")
    print(f"  Calibrated ECE: {calibrated_ece:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    print("="*70)
    
    return T_optimal, original_ece, calibrated_ece


if __name__ == "__main__":
    main()
