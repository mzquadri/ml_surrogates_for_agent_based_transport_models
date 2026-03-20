#!/usr/bin/env python3
"""
Comprehensive Uncertainty Quantification Analysis
==================================================

This script implements 4 key analyses for thesis:
1. Threshold-based Decision Making - Flag high-uncertainty predictions
2. Uncertainty Heat Maps - Visualize uncertainty on Paris road network
3. Feature-wise Error Analysis - Which features correlate with uncertainty
4. Calibration Curves - Are uncertainty estimates trustworthy?

Usage:
    python scripts/evaluation/comprehensive_uq_analysis.py
"""

import os
import sys
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Add scripts to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from gnn.models.point_net_transf_gat import PointNetTransfGAT

# ============================================================================
# Configuration
# ============================================================================

MODEL_FOLDER = os.path.join(REPO_ROOT, 'data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'thesis/latex_tum_official/figures')

# Light pastel colors (consistent with thesis)
COLORS = {
    'sky_blue': '#B8D4E8',
    'mint': '#B8E8D4', 
    'peach': '#E8D4B8',
    'lavender': '#D4B8E8',
    'rose': '#E8B8D4',
    'cream': '#E8E8B8',
    'aqua': '#B8E8E8',
    'coral': '#E8C8B8',
    'text': '#4A5568',
    'border': '#A0AEC0'
}

# Feature names
FEATURE_NAMES = [
    'VOL_BASE_CASE',
    'CAPACITY_BASE_CASE', 
    'CAPACITY_REDUCTION',
    'FREESPEED',
    'LENGTH'
]

FEATURE_LABELS = [
    'Baseline Volume\n(veh/h)',
    'Road Capacity\n(veh/h)',
    'Capacity Reduction\n(fraction)',
    'Free-flow Speed\n(km/h)',
    'Segment Length\n(m)'
]


# ============================================================================
# Helper Functions
# ============================================================================

def mc_dropout_predict(model, data, num_samples=50, device=None):
    """MC Dropout inference."""
    model = model.to(device)
    data = data.to(device)
    model.train()
    
    # Freeze BatchNorm
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0).squeeze()
    uncertainty = predictions.std(axis=0).squeeze()
    
    return mean_pred, uncertainty


def load_model(device):
    """Load trained Model 8."""
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=0.2,
        use_dropout=True,
        predict_mode_stats=False
    )
    
    model_path = os.path.join(MODEL_FOLDER, 'trained_model/model.pth')
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False),
        strict=False
    )
    return model.to(device)


def run_inference(model, test_data, device, n_graphs=None, mc_samples=50):
    """Run MC Dropout inference on test data."""
    if n_graphs is None:
        n_graphs = len(test_data)
    n_graphs = min(n_graphs, len(test_data))
    
    all_predictions = []
    all_uncertainties = []
    all_targets = []
    all_features = []
    all_positions = []
    
    print(f"Running MC Dropout inference on {n_graphs} graphs with {mc_samples} samples...")
    
    for i in tqdm(range(n_graphs)):
        data = test_data[i].to(device)
        
        mean_pred, uncertainty = mc_dropout_predict(model, data, mc_samples, device)
        
        all_predictions.append(mean_pred)
        all_uncertainties.append(uncertainty)
        all_targets.append(data.y.cpu().numpy().squeeze())
        all_features.append(data.x.cpu().numpy())
        
        if hasattr(data, 'pos') and data.pos is not None:
            # pos shape: (N, 3, 2) - [start, mid, end] x [x, y]
            # Use middle position for visualization
            pos = data.pos[:, 1, :].cpu().numpy()  # middle point
            all_positions.append(pos)
    
    return {
        'predictions': np.concatenate(all_predictions),
        'uncertainties': np.concatenate(all_uncertainties),
        'targets': np.concatenate(all_targets),
        'features': np.vstack(all_features),
        'positions': np.vstack(all_positions) if all_positions else None,
        'n_graphs': n_graphs
    }


# ============================================================================
# Analysis 1: Threshold-based Decision Making
# ============================================================================

def analysis_1_threshold_decision(results, output_dir):
    """
    Threshold-based decision making analysis.
    Determine what percentage of predictions need manual review at different confidence levels.
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: Threshold-based Decision Making")
    print("="*80)
    
    uncertainties = results['uncertainties']
    predictions = results['predictions']
    targets = results['targets']
    abs_errors = np.abs(predictions - targets)
    
    # Compute percentiles
    percentiles = [50, 75, 90, 95, 99]
    thresholds = np.percentile(uncertainties, percentiles)
    
    print("\nUncertainty Percentile Thresholds:")
    print("-" * 60)
    
    analysis_results = []
    for p, thresh in zip(percentiles, thresholds):
        high_unc_mask = uncertainties > thresh
        n_flagged = np.sum(high_unc_mask)
        pct_flagged = 100 * n_flagged / len(uncertainties)
        
        # Error statistics for flagged vs not flagged
        flagged_mae = np.mean(abs_errors[high_unc_mask]) if n_flagged > 0 else 0
        unflagged_mae = np.mean(abs_errors[~high_unc_mask])
        
        print(f"  {100-p}% confidence (unc > {thresh:.2f}):")
        print(f"    → {pct_flagged:.1f}% predictions flagged for review ({n_flagged:,} nodes)")
        print(f"    → Flagged MAE: {flagged_mae:.2f} vs Unflagged MAE: {unflagged_mae:.2f}")
        
        analysis_results.append({
            'confidence': 100-p,
            'threshold': thresh,
            'pct_flagged': pct_flagged,
            'n_flagged': n_flagged,
            'flagged_mae': flagged_mae,
            'unflagged_mae': unflagged_mae
        })
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Threshold vs % Flagged
    ax = axes[0]
    confidence_levels = [r['confidence'] for r in analysis_results]
    pct_flagged = [r['pct_flagged'] for r in analysis_results]
    bars = ax.bar(range(len(confidence_levels)), pct_flagged, 
                  color=[COLORS['sky_blue'], COLORS['mint'], COLORS['peach'], 
                         COLORS['lavender'], COLORS['rose']],
                  edgecolor=COLORS['border'], linewidth=1.5)
    ax.set_xticks(range(len(confidence_levels)))
    ax.set_xticklabels([f"{c}%" for c in confidence_levels])
    ax.set_xlabel('Confidence Level', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('% Predictions Flagged', fontweight='bold', color=COLORS['text'])
    ax.set_title('(a) Predictions Requiring Review', fontweight='bold', color=COLORS['text'])
    for bar, pct in zip(bars, pct_flagged):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(pct_flagged) * 1.2)
    
    # Plot 2: MAE comparison
    ax = axes[1]
    x = np.arange(len(confidence_levels))
    width = 0.35
    flagged_mae = [r['flagged_mae'] for r in analysis_results]
    unflagged_mae = [r['unflagged_mae'] for r in analysis_results]
    
    bars1 = ax.bar(x - width/2, flagged_mae, width, label='Flagged (High Uncertainty)',
                   color=COLORS['coral'], edgecolor=COLORS['border'])
    bars2 = ax.bar(x + width/2, unflagged_mae, width, label='Unflagged (Confident)',
                   color=COLORS['mint'], edgecolor=COLORS['border'])
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}%" for c in confidence_levels])
    ax.set_xlabel('Confidence Level', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Mean Absolute Error (veh/h)', fontweight='bold', color=COLORS['text'])
    ax.set_title('(b) Error by Flagging Status', fontweight='bold', color=COLORS['text'])
    ax.legend(loc='upper right')
    
    # Plot 3: Trade-off curve
    ax = axes[2]
    all_percentiles = np.arange(0, 100, 1)
    all_thresholds = np.percentile(uncertainties, all_percentiles)
    
    pct_kept = []
    mae_kept = []
    
    for p, t in zip(all_percentiles, all_thresholds):
        keep_mask = uncertainties <= t
        if np.sum(keep_mask) > 0:
            pct_kept.append(p)
            mae_kept.append(np.mean(abs_errors[keep_mask]))
    
    ax.plot(pct_kept, mae_kept, color=COLORS['sky_blue'], linewidth=2.5)
    ax.fill_between(pct_kept, mae_kept, alpha=0.3, color=COLORS['sky_blue'])
    
    # Mark key points
    for p in [90, 95]:
        idx = pct_kept.index(p) if p in pct_kept else None
        if idx:
            ax.scatter([p], [mae_kept[idx]], s=100, color=COLORS['rose'], 
                      edgecolor='white', linewidth=2, zorder=5)
            ax.annotate(f'{p}%\nMAE={mae_kept[idx]:.1f}', 
                       xy=(p, mae_kept[idx]), xytext=(p-8, mae_kept[idx]+10),
                       fontsize=9, fontweight='bold')
    
    ax.set_xlabel('% Predictions Kept (Confident)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('MAE of Kept Predictions', fontweight='bold', color=COLORS['text'])
    ax.set_title('(c) Accuracy-Coverage Trade-off', fontweight='bold', color=COLORS['text'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: uq_threshold_analysis.png")
    
    return analysis_results


# ============================================================================
# Analysis 2: Uncertainty Heat Map
# ============================================================================

def analysis_2_uncertainty_heatmap(results, output_dir):
    """
    Visualize uncertainty on the Paris road network.
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: Uncertainty Heat Map")
    print("="*80)
    
    positions = results['positions']
    uncertainties = results['uncertainties']
    abs_errors = np.abs(results['predictions'] - results['targets'])
    
    if positions is None:
        print("  ⚠ Position data not available, skipping heat map")
        return None
    
    print(f"  Visualizing {len(positions):,} road segments...")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Normalize positions for plotting
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Sample for faster plotting if too many points
    if len(x) > 50000:
        sample_idx = np.random.choice(len(x), 50000, replace=False)
        x = x[sample_idx]
        y = y[sample_idx]
        uncertainties_plot = uncertainties[sample_idx]
        errors_plot = abs_errors[sample_idx]
    else:
        uncertainties_plot = uncertainties
        errors_plot = abs_errors
    
    # Custom colormap (light to dark)
    colors_cmap = ['#E8F4F8', '#B8D4E8', '#7EB5D6', '#4A90B8', '#2C5F7C']
    cmap_unc = LinearSegmentedColormap.from_list('uncertainty', colors_cmap)
    
    colors_err = ['#E8F8E8', '#B8E8B8', '#7ED67E', '#4AB84A', '#2C7C2C']
    cmap_err = LinearSegmentedColormap.from_list('error', colors_err)
    
    # Plot 1: Uncertainty map
    ax = axes[0]
    unc_99 = np.percentile(uncertainties_plot, 99)
    scatter1 = ax.scatter(x, y, c=np.clip(uncertainties_plot, 0, unc_99), 
                         cmap=cmap_unc, s=1, alpha=0.6)
    ax.set_xlabel('X Coordinate (normalized)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Y Coordinate (normalized)', fontweight='bold', color=COLORS['text'])
    ax.set_title('(a) Uncertainty Heat Map - Paris Network', fontweight='bold', 
                 color=COLORS['text'], fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=ax, shrink=0.8)
    cbar1.set_label('Uncertainty (σ)', fontweight='bold')
    ax.set_aspect('equal')
    
    # Plot 2: Error map
    ax = axes[1]
    err_99 = np.percentile(errors_plot, 99)
    scatter2 = ax.scatter(x, y, c=np.clip(errors_plot, 0, err_99),
                         cmap=cmap_err, s=1, alpha=0.6)
    ax.set_xlabel('X Coordinate (normalized)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Y Coordinate (normalized)', fontweight='bold', color=COLORS['text'])
    ax.set_title('(b) Prediction Error Heat Map - Paris Network', fontweight='bold',
                 color=COLORS['text'], fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=ax, shrink=0.8)
    cbar2.set_label('|Error| (veh/h)', fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_spatial_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional: High uncertainty zones
    fig, ax = plt.subplots(figsize=(10, 10))
    
    unc_threshold = np.percentile(uncertainties_plot, 90)
    low_unc_mask = uncertainties_plot <= unc_threshold
    high_unc_mask = uncertainties_plot > unc_threshold
    
    ax.scatter(x[low_unc_mask], y[low_unc_mask], c=COLORS['mint'], s=1, 
               alpha=0.4, label='Confident (90%)')
    ax.scatter(x[high_unc_mask], y[high_unc_mask], c=COLORS['coral'], s=5, 
               alpha=0.8, label='Uncertain (top 10%)')
    
    ax.set_xlabel('X Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Y Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_title('High Uncertainty Road Segments (Top 10%)', fontweight='bold',
                 color=COLORS['text'], fontsize=14)
    ax.legend(loc='upper right', markerscale=5)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_high_uncertainty_zones.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: uq_spatial_heatmap.png")
    print(f"  ✓ Saved: uq_high_uncertainty_zones.png")
    
    # Statistics
    spearman_spatial = spearmanr(uncertainties_plot, errors_plot)[0]
    print(f"\n  Spatial uncertainty-error correlation: ρ = {spearman_spatial:.4f}")
    
    return {'spearman_spatial': spearman_spatial}


# ============================================================================
# Analysis 3: Feature-wise Error Analysis
# ============================================================================

def analysis_3_feature_error(results, output_dir):
    """
    Analyze which feature values correlate with high uncertainty/error.
    """
    print("\n" + "="*80)
    print("ANALYSIS 3: Feature-wise Error Analysis")
    print("="*80)
    
    features = results['features']
    uncertainties = results['uncertainties']
    abs_errors = np.abs(results['predictions'] - results['targets'])
    
    n_features = features.shape[1]
    
    # Compute correlations
    print("\nFeature-Uncertainty Correlations (Spearman ρ):")
    print("-" * 50)
    
    feature_stats = []
    for i in range(n_features):
        feat_vals = features[:, i]
        corr_unc = spearmanr(feat_vals, uncertainties)[0]
        corr_err = spearmanr(feat_vals, abs_errors)[0]
        
        print(f"  {FEATURE_NAMES[i]:<25}: ρ_unc = {corr_unc:+.4f}, ρ_error = {corr_err:+.4f}")
        
        feature_stats.append({
            'feature': FEATURE_NAMES[i],
            'corr_uncertainty': corr_unc,
            'corr_error': corr_err
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors_list = [COLORS['sky_blue'], COLORS['mint'], COLORS['peach'], 
                   COLORS['lavender'], COLORS['rose']]
    
    # Sample for faster plotting
    if len(features) > 20000:
        sample_idx = np.random.choice(len(features), 20000, replace=False)
    else:
        sample_idx = np.arange(len(features))
    
    for i in range(n_features):
        ax = axes[i]
        feat_vals = features[sample_idx, i]
        unc_vals = uncertainties[sample_idx]
        
        # Bin the feature values and compute mean uncertainty per bin
        n_bins = 20
        bins = np.percentile(feat_vals, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        
        if len(bins) < 3:
            ax.text(0.5, 0.5, 'Insufficient variance', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(FEATURE_LABELS[i], fontweight='bold', color=COLORS['text'])
            continue
        
        bin_indices = np.digitize(feat_vals, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        bin_centers = []
        bin_mean_unc = []
        bin_std_unc = []
        
        for b in range(len(bins) - 1):
            mask = bin_indices == b
            if np.sum(mask) > 10:
                bin_centers.append((bins[b] + bins[b+1]) / 2)
                bin_mean_unc.append(np.mean(unc_vals[mask]))
                bin_std_unc.append(np.std(unc_vals[mask]) / np.sqrt(np.sum(mask)))
        
        if len(bin_centers) > 0:
            ax.errorbar(bin_centers, bin_mean_unc, yerr=bin_std_unc,
                       fmt='o-', color=colors_list[i], linewidth=2, markersize=6,
                       capsize=3, capthick=1.5, ecolor=COLORS['border'])
            ax.fill_between(bin_centers, 
                           np.array(bin_mean_unc) - np.array(bin_std_unc),
                           np.array(bin_mean_unc) + np.array(bin_std_unc),
                           alpha=0.2, color=colors_list[i])
        
        corr = feature_stats[i]['corr_uncertainty']
        ax.set_xlabel(FEATURE_LABELS[i], fontweight='bold', color=COLORS['text'])
        ax.set_ylabel('Mean Uncertainty (σ)', fontweight='bold', color=COLORS['text'])
        ax.set_title(f'ρ = {corr:+.3f}', fontweight='bold', color=COLORS['text'])
        ax.grid(True, alpha=0.3)
    
    # Summary bar chart in 6th subplot
    ax = axes[5]
    corrs = [fs['corr_uncertainty'] for fs in feature_stats]
    bars = ax.barh(range(n_features), corrs, color=colors_list, edgecolor=COLORS['border'], linewidth=1.5)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_xlabel('Correlation with Uncertainty (ρ)', fontweight='bold', color=COLORS['text'])
    ax.set_title('Feature-Uncertainty Correlations', fontweight='bold', color=COLORS['text'])
    ax.axvline(x=0, color=COLORS['border'], linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Color bars by sign
    for bar, corr in zip(bars, corrs):
        if corr < 0:
            bar.set_color(COLORS['mint'])
        else:
            bar.set_color(COLORS['coral'])
    
    plt.suptitle('Feature-wise Uncertainty Analysis', fontsize=14, fontweight='bold', 
                 color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_feature_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: uq_feature_analysis.png")
    
    return feature_stats


# ============================================================================
# Analysis 4: Calibration Curves
# ============================================================================

def analysis_4_calibration(results, output_dir):
    """
    Calibration analysis - Are uncertainty estimates trustworthy?
    """
    print("\n" + "="*80)
    print("ANALYSIS 4: Calibration Analysis")
    print("="*80)
    
    uncertainties = results['uncertainties']
    predictions = results['predictions']
    targets = results['targets']
    abs_errors = np.abs(predictions - targets)
    
    # 1. Reliability Diagram (Expected Calibration Error)
    # Bin predictions by uncertainty and compute expected vs observed error
    
    n_bins = 10
    percentile_edges = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(uncertainties, percentile_edges)
    
    calibration_data = []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1])
        else:
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        
        if np.sum(mask) > 0:
            mean_unc = np.mean(uncertainties[mask])
            mean_err = np.mean(abs_errors[mask])
            std_err = np.std(abs_errors[mask])
            n_samples = np.sum(mask)
            
            calibration_data.append({
                'bin': i,
                'mean_uncertainty': mean_unc,
                'mean_error': mean_err,
                'std_error': std_err,
                'n_samples': n_samples
            })
    
    # Compute Expected Calibration Error (ECE)
    total_samples = len(uncertainties)
    ece = 0
    for cd in calibration_data:
        weight = cd['n_samples'] / total_samples
        # Difference between uncertainty and observed error (normalized)
        diff = abs(cd['mean_uncertainty'] - cd['mean_error'] / np.mean(abs_errors))
        ece += weight * diff
    
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Reliability Diagram
    ax = axes[0, 0]
    mean_uncs = [cd['mean_uncertainty'] for cd in calibration_data]
    mean_errs = [cd['mean_error'] for cd in calibration_data]
    
    # Normalize for comparison
    max_val = max(max(mean_uncs), max(mean_errs)) * 1.1
    
    ax.bar(range(len(mean_uncs)), mean_errs, color=COLORS['sky_blue'], 
           edgecolor=COLORS['border'], linewidth=1.5, label='Observed Error')
    ax.plot(range(len(mean_uncs)), mean_uncs, 'o-', color=COLORS['coral'], 
            linewidth=2, markersize=8, label='Predicted Uncertainty')
    
    ax.set_xlabel('Uncertainty Bin (Low → High)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Value', fontweight='bold', color=COLORS['text'])
    ax.set_title(f'(a) Reliability Diagram (ECE = {ece:.3f})', fontweight='bold', color=COLORS['text'])
    ax.legend()
    ax.set_xticks(range(len(mean_uncs)))
    ax.set_xticklabels([f'{i+1}' for i in range(len(mean_uncs))])
    
    # Plot 2: Error Distribution by Uncertainty Quantile
    ax = axes[0, 1]
    
    # Split into confidence quartiles
    quartiles = [25, 50, 75, 100]
    quartile_labels = ['Most Confident\n(0-25%)', '25-50%', '50-75%', 'Least Confident\n(75-100%)']
    quartile_colors = [COLORS['mint'], COLORS['sky_blue'], COLORS['peach'], COLORS['coral']]
    
    prev_q = 0
    violin_data = []
    for q in quartiles:
        threshold_low = np.percentile(uncertainties, prev_q)
        threshold_high = np.percentile(uncertainties, q)
        mask = (uncertainties >= threshold_low) & (uncertainties <= threshold_high)
        violin_data.append(abs_errors[mask])
        prev_q = q
    
    parts = ax.violinplot(violin_data, positions=range(len(quartiles)), showmeans=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(quartile_colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(quartiles)))
    ax.set_xticklabels(quartile_labels, fontsize=9)
    ax.set_xlabel('Uncertainty Quartile', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('|Error| Distribution (veh/h)', fontweight='bold', color=COLORS['text'])
    ax.set_title('(b) Error Distribution by Confidence', fontweight='bold', color=COLORS['text'])
    
    # Plot 3: Cumulative Calibration
    ax = axes[1, 0]
    
    sorted_idx = np.argsort(uncertainties)
    sorted_unc = uncertainties[sorted_idx]
    sorted_err = abs_errors[sorted_idx]
    
    cumulative_pct = np.arange(1, len(sorted_unc) + 1) / len(sorted_unc) * 100
    cumulative_mae = np.cumsum(sorted_err) / np.arange(1, len(sorted_err) + 1)
    
    # Subsample for plotting
    sample_idx = np.linspace(0, len(cumulative_pct) - 1, 500).astype(int)
    
    ax.plot(cumulative_pct[sample_idx], cumulative_mae[sample_idx], 
            color=COLORS['sky_blue'], linewidth=2)
    ax.fill_between(cumulative_pct[sample_idx], cumulative_mae[sample_idx], 
                    alpha=0.3, color=COLORS['sky_blue'])
    
    # Mark key percentiles
    for p in [50, 90]:
        idx = int(len(sorted_unc) * p / 100)
        ax.scatter([p], [cumulative_mae[idx]], s=100, color=COLORS['rose'], 
                  edgecolor='white', linewidth=2, zorder=5)
        ax.annotate(f'MAE={cumulative_mae[idx]:.1f}', 
                   xy=(p, cumulative_mae[idx]), xytext=(p+5, cumulative_mae[idx]+5),
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('% Most Confident Predictions', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Cumulative MAE', fontweight='bold', color=COLORS['text'])
    ax.set_title('(c) Cumulative Calibration Curve', fontweight='bold', color=COLORS['text'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    # Plot 4: Summary Statistics Box
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute statistics
    spearman_corr = spearmanr(uncertainties, abs_errors)[0]
    pearson_corr = pearsonr(uncertainties, abs_errors)[0]
    
    # Error reduction at different confidence levels
    top50_mask = uncertainties <= np.percentile(uncertainties, 50)
    top90_mask = uncertainties <= np.percentile(uncertainties, 90)
    
    mae_all = np.mean(abs_errors)
    mae_top50 = np.mean(abs_errors[top50_mask])
    mae_top90 = np.mean(abs_errors[top90_mask])
    
    summary_text = f"""
    ╔══════════════════════════════════════════╗
    ║     CALIBRATION SUMMARY                  ║
    ╠══════════════════════════════════════════╣
    ║  Expected Calibration Error:  {ece:.4f}     ║
    ║                                          ║
    ║  Uncertainty-Error Correlation:          ║
    ║    • Spearman ρ:  {spearman_corr:.4f}                 ║
    ║    • Pearson r:   {pearson_corr:.4f}                 ║
    ║                                          ║
    ║  Error at Confidence Levels:             ║
    ║    • All predictions:    MAE = {mae_all:.1f}     ║
    ║    • Top 90% confident:  MAE = {mae_top90:.1f}     ║
    ║    • Top 50% confident:  MAE = {mae_top50:.1f}     ║
    ║                                          ║
    ║  Error Reduction:                        ║
    ║    • At 90% conf: {100*(1-mae_top90/mae_all):.1f}% lower error     ║
    ║    • At 50% conf: {100*(1-mae_top50/mae_all):.1f}% lower error     ║
    ╚══════════════════════════════════════════╝
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, fontfamily='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=COLORS['cream'], edgecolor=COLORS['border']))
    
    plt.suptitle('Uncertainty Calibration Analysis', fontsize=14, fontweight='bold',
                 color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_calibration_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Spearman correlation: ρ = {spearman_corr:.4f}")
    print(f"  Pearson correlation:  r = {pearson_corr:.4f}")
    print(f"\n  Error reduction at 90% confidence: {100*(1-mae_top90/mae_all):.1f}%")
    print(f"  Error reduction at 50% confidence: {100*(1-mae_top50/mae_all):.1f}%")
    print(f"\n  ✓ Saved: uq_calibration_analysis.png")
    
    return {
        'ece': ece,
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'mae_all': mae_all,
        'mae_top90': mae_top90,
        'mae_top50': mae_top50
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE UNCERTAINTY QUANTIFICATION ANALYSIS")
    print("="*80)
    print(f"Model: Trial 8 (PointNetTransfGAT)")
    print(f"Output: {OUTPUT_DIR}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(device)
    print("  ✓ Model loaded")
    
    # Load test data
    print("\nLoading test data...")
    test_dl_path = os.path.join(MODEL_FOLDER, 'data_created_during_training/test_dl.pt')
    test_data = torch.load(test_dl_path, weights_only=False)
    print(f"  ✓ Loaded {len(test_data)} test graphs")
    
    # Run inference
    results = run_inference(model, test_data, device, n_graphs=100, mc_samples=50)
    
    print(f"\nTotal predictions: {len(results['predictions']):,}")
    print(f"Features shape: {results['features'].shape}")
    if results['positions'] is not None:
        print(f"Positions shape: {results['positions'].shape}")
    
    # Run all analyses
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    analysis_1_threshold_decision(results, OUTPUT_DIR)
    analysis_2_uncertainty_heatmap(results, OUTPUT_DIR)
    analysis_3_feature_error(results, OUTPUT_DIR)
    analysis_4_calibration(results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nGenerated figures in: {OUTPUT_DIR}")
    print("  • uq_threshold_analysis.png")
    print("  • uq_spatial_heatmap.png")
    print("  • uq_high_uncertainty_zones.png")
    print("  • uq_feature_analysis.png")
    print("  • uq_calibration_analysis.png")


if __name__ == '__main__':
    main()
