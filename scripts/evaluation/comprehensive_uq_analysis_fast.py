#!/usr/bin/env python3
"""
Comprehensive UQ Analysis - Using Pre-computed Data
====================================================

Uses existing MC Dropout inference results to generate 4 key analyses:
1. Threshold-based Decision Making
2. Uncertainty Heat Maps (needs position data)
3. Feature-wise Error Analysis
4. Calibration Curves

This version loads pre-computed data from mc_dropout_full_100graphs_mc30.npz
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

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

MODEL_FOLDER = os.path.join(REPO_ROOT, 'data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'thesis/latex_tum_official/figures')

# Light pastel colors
COLORS = {
    'sky_blue': '#B8D4E8', 'mint': '#B8E8D4', 'peach': '#E8D4B8',
    'lavender': '#D4B8E8', 'rose': '#E8B8D4', 'cream': '#E8E8B8',
    'aqua': '#B8E8E8', 'coral': '#E8C8B8', 'text': '#4A5568', 'border': '#A0AEC0'
}

FEATURE_NAMES = ['VOL_BASE_CASE', 'CAPACITY_BASE_CASE', 'CAPACITY_REDUCTION', 'FREESPEED', 'LENGTH']
FEATURE_LABELS = ['Baseline Volume\n(veh/h)', 'Road Capacity\n(veh/h)', 
                  'Capacity Reduction\n(fraction)', 'Free-flow Speed\n(km/h)', 'Segment Length\n(m)']


def load_precomputed_data():
    """Load pre-computed MC Dropout inference results and features."""
    print("Loading pre-computed data...")
    
    # Load MC Dropout results
    npz_path = os.path.join(MODEL_FOLDER, 'uq_results/mc_dropout_full_100graphs_mc30.npz')
    data = np.load(npz_path)
    
    print(f"  Available keys: {list(data.keys())}")
    
    predictions = data['predictions']
    uncertainties = data['uncertainties'] 
    targets = data['targets']
    
    print(f"  Predictions: {predictions.shape}")
    print(f"  Uncertainties: {uncertainties.shape}")
    print(f"  Targets: {targets.shape}")
    
    # Load test data for features and positions
    test_dl_path = os.path.join(MODEL_FOLDER, 'data_created_during_training/test_dl.pt')
    test_data = torch.load(test_dl_path, weights_only=False)
    
    # Extract features and positions
    features_list = []
    positions_list = []
    
    n_graphs = min(100, len(test_data))
    total_nodes = 0
    
    for i in range(n_graphs):
        data = test_data[i]
        n_nodes = data.x.shape[0]
        
        # Only take nodes up to the number we have predictions for
        if total_nodes + n_nodes > len(predictions):
            n_nodes = len(predictions) - total_nodes
        
        if n_nodes <= 0:
            break
            
        features_list.append(data.x[:n_nodes].numpy())
        
        if hasattr(data, 'pos') and data.pos is not None:
            # pos shape: (N, 3, 2) - use middle position
            positions_list.append(data.pos[:n_nodes, 1, :].numpy())
        
        total_nodes += n_nodes
    
    features = np.vstack(features_list) if features_list else None
    positions = np.vstack(positions_list) if positions_list else None
    
    # Align shapes
    min_len = min(len(predictions), len(targets), len(uncertainties))
    if features is not None:
        min_len = min(min_len, len(features))
    if positions is not None:
        min_len = min(min_len, len(positions))
    
    print(f"\n  Using {min_len:,} aligned predictions")
    
    return {
        'predictions': predictions[:min_len],
        'uncertainties': uncertainties[:min_len],
        'targets': targets[:min_len],
        'features': features[:min_len] if features is not None else None,
        'positions': positions[:min_len] if positions is not None else None
    }


def analysis_1_threshold_decision(results, output_dir):
    """Threshold-based decision making analysis."""
    print("\n" + "="*80)
    print("ANALYSIS 1: Threshold-based Decision Making")
    print("="*80)
    
    uncertainties = results['uncertainties']
    predictions = results['predictions']
    targets = results['targets']
    abs_errors = np.abs(predictions - targets)
    
    percentiles = [50, 75, 90, 95, 99]
    thresholds = np.percentile(uncertainties, percentiles)
    
    print("\nConfidence Level Analysis:")
    print("-" * 60)
    
    analysis_results = []
    for p, thresh in zip(percentiles, thresholds):
        high_unc_mask = uncertainties > thresh
        n_flagged = np.sum(high_unc_mask)
        pct_flagged = 100 * n_flagged / len(uncertainties)
        flagged_mae = np.mean(abs_errors[high_unc_mask]) if n_flagged > 0 else 0
        unflagged_mae = np.mean(abs_errors[~high_unc_mask])
        
        print(f"  {100-p}% confidence → {pct_flagged:.1f}% flagged | MAE: {unflagged_mae:.1f} (kept) vs {flagged_mae:.1f} (flagged)")
        
        analysis_results.append({
            'confidence': 100-p, 'threshold': thresh, 'pct_flagged': pct_flagged,
            'flagged_mae': flagged_mae, 'unflagged_mae': unflagged_mae
        })
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: % Flagged
    ax = axes[0]
    conf_levels = [r['confidence'] for r in analysis_results]
    pct_flagged = [r['pct_flagged'] for r in analysis_results]
    colors_bar = [COLORS['sky_blue'], COLORS['mint'], COLORS['peach'], COLORS['lavender'], COLORS['rose']]
    bars = ax.bar(range(len(conf_levels)), pct_flagged, color=colors_bar, edgecolor=COLORS['border'], linewidth=1.5)
    ax.set_xticks(range(len(conf_levels)))
    ax.set_xticklabels([f"{c}%" for c in conf_levels])
    ax.set_xlabel('Confidence Level', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('% Predictions Flagged', fontweight='bold', color=COLORS['text'])
    ax.set_title('(a) Predictions Requiring Review', fontweight='bold', color=COLORS['text'])
    for bar, pct in zip(bars, pct_flagged):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(pct_flagged) * 1.25)
    
    # Plot 2: MAE comparison
    ax = axes[1]
    x = np.arange(len(conf_levels))
    width = 0.35
    flagged_mae = [r['flagged_mae'] for r in analysis_results]
    unflagged_mae = [r['unflagged_mae'] for r in analysis_results]
    ax.bar(x - width/2, flagged_mae, width, label='Flagged (Uncertain)', color=COLORS['coral'], edgecolor=COLORS['border'])
    ax.bar(x + width/2, unflagged_mae, width, label='Kept (Confident)', color=COLORS['mint'], edgecolor=COLORS['border'])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}%" for c in conf_levels])
    ax.set_xlabel('Confidence Level', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('MAE (veh/h)', fontweight='bold', color=COLORS['text'])
    ax.set_title('(b) Error by Flagging Status', fontweight='bold', color=COLORS['text'])
    ax.legend()
    
    # Plot 3: Trade-off curve
    ax = axes[2]
    all_percentiles = np.arange(0, 100, 2)
    all_thresholds = np.percentile(uncertainties, all_percentiles)
    pct_kept, mae_kept = [], []
    for p, t in zip(all_percentiles, all_thresholds):
        keep_mask = uncertainties <= t
        if np.sum(keep_mask) > 0:
            pct_kept.append(p)
            mae_kept.append(np.mean(abs_errors[keep_mask]))
    
    ax.plot(pct_kept, mae_kept, color=COLORS['sky_blue'], linewidth=2.5)
    ax.fill_between(pct_kept, mae_kept, alpha=0.3, color=COLORS['sky_blue'])
    for p in [90, 95]:
        if p in pct_kept:
            idx = pct_kept.index(p)
            ax.scatter([p], [mae_kept[idx]], s=100, color=COLORS['rose'], edgecolor='white', linewidth=2, zorder=5)
            ax.annotate(f'{p}%: MAE={mae_kept[idx]:.1f}', xy=(p, mae_kept[idx]), 
                       xytext=(p-15, mae_kept[idx]+8), fontsize=9, fontweight='bold')
    ax.set_xlabel('% Predictions Kept', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('MAE of Kept Predictions', fontweight='bold', color=COLORS['text'])
    ax.set_title('(c) Accuracy-Coverage Trade-off', fontweight='bold', color=COLORS['text'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: uq_threshold_analysis.png")
    return analysis_results


def analysis_2_uncertainty_heatmap(results, output_dir):
    """Uncertainty heat map visualization."""
    print("\n" + "="*80)
    print("ANALYSIS 2: Uncertainty Heat Map")
    print("="*80)
    
    positions = results['positions']
    uncertainties = results['uncertainties']
    abs_errors = np.abs(results['predictions'] - results['targets'])
    
    if positions is None:
        print("  ⚠ Position data not available")
        return None
    
    print(f"  Visualizing {len(positions):,} road segments...")
    
    # Sample for faster plotting
    if len(positions) > 50000:
        idx = np.random.choice(len(positions), 50000, replace=False)
        x, y = positions[idx, 0], positions[idx, 1]
        unc_plot, err_plot = uncertainties[idx], abs_errors[idx]
    else:
        x, y = positions[:, 0], positions[:, 1]
        unc_plot, err_plot = uncertainties, abs_errors
    
    cmap_unc = LinearSegmentedColormap.from_list('unc', ['#E8F4F8', '#B8D4E8', '#7EB5D6', '#4A90B8', '#2C5F7C'])
    cmap_err = LinearSegmentedColormap.from_list('err', ['#E8F8E8', '#B8E8B8', '#7ED67E', '#4AB84A', '#2C7C2C'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Uncertainty map
    ax = axes[0]
    scatter = ax.scatter(x, y, c=np.clip(unc_plot, 0, np.percentile(unc_plot, 99)), 
                        cmap=cmap_unc, s=1, alpha=0.6)
    ax.set_xlabel('X Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Y Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_title('(a) Uncertainty Map - Paris Network', fontweight='bold', color=COLORS['text'], fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Uncertainty (σ)', fontweight='bold')
    ax.set_aspect('equal')
    
    # Error map
    ax = axes[1]
    scatter = ax.scatter(x, y, c=np.clip(err_plot, 0, np.percentile(err_plot, 99)), 
                        cmap=cmap_err, s=1, alpha=0.6)
    ax.set_xlabel('X Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Y Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_title('(b) Error Map - Paris Network', fontweight='bold', color=COLORS['text'], fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('|Error| (veh/h)', fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_spatial_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # High uncertainty zones
    fig, ax = plt.subplots(figsize=(10, 10))
    unc_thresh = np.percentile(unc_plot, 90)
    low_mask = unc_plot <= unc_thresh
    high_mask = unc_plot > unc_thresh
    ax.scatter(x[low_mask], y[low_mask], c=COLORS['mint'], s=1, alpha=0.4, label='Confident (90%)')
    ax.scatter(x[high_mask], y[high_mask], c=COLORS['coral'], s=5, alpha=0.8, label='Uncertain (top 10%)')
    ax.set_xlabel('X Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Y Coordinate', fontweight='bold', color=COLORS['text'])
    ax.set_title('High Uncertainty Road Segments (Top 10%)', fontweight='bold', color=COLORS['text'], fontsize=14)
    ax.legend(loc='upper right', markerscale=5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_high_uncertainty_zones.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: uq_spatial_heatmap.png")
    print(f"  ✓ Saved: uq_high_uncertainty_zones.png")
    return {'spatial_corr': spearmanr(unc_plot, err_plot)[0]}


def analysis_3_feature_error(results, output_dir):
    """Feature-wise error analysis."""
    print("\n" + "="*80)
    print("ANALYSIS 3: Feature-wise Error Analysis")
    print("="*80)
    
    features = results['features']
    uncertainties = results['uncertainties']
    abs_errors = np.abs(results['predictions'] - results['targets'])
    
    if features is None:
        print("  ⚠ Feature data not available")
        return None
    
    n_features = features.shape[1]
    feature_stats = []
    
    print("\nFeature-Uncertainty Correlations:")
    print("-" * 50)
    
    for i in range(n_features):
        corr_unc = spearmanr(features[:, i], uncertainties)[0]
        corr_err = spearmanr(features[:, i], abs_errors)[0]
        print(f"  {FEATURE_NAMES[i]:<25}: ρ_unc = {corr_unc:+.4f}, ρ_err = {corr_err:+.4f}")
        feature_stats.append({'feature': FEATURE_NAMES[i], 'corr_uncertainty': corr_unc, 'corr_error': corr_err})
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors_list = [COLORS['sky_blue'], COLORS['mint'], COLORS['peach'], COLORS['lavender'], COLORS['rose']]
    
    # Sample for speed
    if len(features) > 20000:
        idx = np.random.choice(len(features), 20000, replace=False)
    else:
        idx = np.arange(len(features))
    
    for i in range(n_features):
        ax = axes[i]
        feat_vals = features[idx, i]
        unc_vals = uncertainties[idx]
        
        # Binned analysis
        n_bins = 20
        bins = np.percentile(feat_vals, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 3:
            ax.text(0.5, 0.5, 'Low variance', ha='center', va='center', transform=ax.transAxes)
            continue
        
        bin_idx = np.clip(np.digitize(feat_vals, bins) - 1, 0, len(bins) - 2)
        bin_centers, bin_means, bin_stds = [], [], []
        
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if np.sum(mask) > 10:
                bin_centers.append((bins[b] + bins[b+1]) / 2)
                bin_means.append(np.mean(unc_vals[mask]))
                bin_stds.append(np.std(unc_vals[mask]) / np.sqrt(np.sum(mask)))
        
        if len(bin_centers) > 0:
            ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', color=colors_list[i],
                       linewidth=2, markersize=6, capsize=3, ecolor=COLORS['border'])
            ax.fill_between(bin_centers, np.array(bin_means) - np.array(bin_stds),
                           np.array(bin_means) + np.array(bin_stds), alpha=0.2, color=colors_list[i])
        
        ax.set_xlabel(FEATURE_LABELS[i], fontweight='bold', color=COLORS['text'])
        ax.set_ylabel('Mean Uncertainty', fontweight='bold', color=COLORS['text'])
        ax.set_title(f'ρ = {feature_stats[i]["corr_uncertainty"]:+.3f}', fontweight='bold', color=COLORS['text'])
        ax.grid(True, alpha=0.3)
    
    # Summary bar chart
    ax = axes[5]
    corrs = [fs['corr_uncertainty'] for fs in feature_stats]
    bar_colors = [COLORS['mint'] if c < 0 else COLORS['coral'] for c in corrs]
    ax.barh(range(n_features), corrs, color=bar_colors, edgecolor=COLORS['border'], linewidth=1.5)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_xlabel('Correlation with Uncertainty (ρ)', fontweight='bold', color=COLORS['text'])
    ax.set_title('Feature-Uncertainty Summary', fontweight='bold', color=COLORS['text'])
    ax.axvline(x=0, color=COLORS['border'], linestyle='--')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Feature-wise Uncertainty Analysis', fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_feature_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: uq_feature_analysis.png")
    return feature_stats


def analysis_4_calibration(results, output_dir):
    """Calibration analysis."""
    print("\n" + "="*80)
    print("ANALYSIS 4: Calibration Analysis")
    print("="*80)
    
    uncertainties = results['uncertainties']
    predictions = results['predictions']
    targets = results['targets']
    abs_errors = np.abs(predictions - targets)
    
    # Reliability diagram data
    n_bins = 10
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    calibration_data = []
    
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1] if i < n_bins-1 else uncertainties <= bin_edges[i+1])
        if np.sum(mask) > 0:
            calibration_data.append({
                'mean_uncertainty': np.mean(uncertainties[mask]),
                'mean_error': np.mean(abs_errors[mask]),
                'n_samples': np.sum(mask)
            })
    
    # ECE
    total = len(uncertainties)
    ece = sum(cd['n_samples']/total * abs(cd['mean_uncertainty'] - cd['mean_error']/np.mean(abs_errors)) for cd in calibration_data)
    
    # Correlations
    spearman_corr = spearmanr(uncertainties, abs_errors)[0]
    pearson_corr = pearsonr(uncertainties, abs_errors)[0]
    
    # Error at confidence levels
    mae_all = np.mean(abs_errors)
    mae_top90 = np.mean(abs_errors[uncertainties <= np.percentile(uncertainties, 90)])
    mae_top50 = np.mean(abs_errors[uncertainties <= np.percentile(uncertainties, 50)])
    
    print(f"\n  ECE: {ece:.4f}")
    print(f"  Spearman ρ: {spearman_corr:.4f}")
    print(f"  Error reduction @90%: {100*(1-mae_top90/mae_all):.1f}%")
    print(f"  Error reduction @50%: {100*(1-mae_top50/mae_all):.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reliability diagram
    ax = axes[0, 0]
    mean_uncs = [cd['mean_uncertainty'] for cd in calibration_data]
    mean_errs = [cd['mean_error'] for cd in calibration_data]
    ax.bar(range(len(mean_uncs)), mean_errs, color=COLORS['sky_blue'], edgecolor=COLORS['border'], label='Observed Error')
    ax.plot(range(len(mean_uncs)), mean_uncs, 'o-', color=COLORS['coral'], linewidth=2, markersize=8, label='Uncertainty')
    ax.set_xlabel('Uncertainty Bin (Low → High)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Value', fontweight='bold', color=COLORS['text'])
    ax.set_title(f'(a) Reliability Diagram (ECE = {ece:.3f})', fontweight='bold', color=COLORS['text'])
    ax.legend()
    ax.set_xticks(range(len(mean_uncs)))
    
    # Violin plot
    ax = axes[0, 1]
    quartiles = [25, 50, 75, 100]
    quartile_colors = [COLORS['mint'], COLORS['sky_blue'], COLORS['peach'], COLORS['coral']]
    violin_data = []
    prev_q = 0
    for q in quartiles:
        mask = (uncertainties >= np.percentile(uncertainties, prev_q)) & (uncertainties <= np.percentile(uncertainties, q))
        violin_data.append(abs_errors[mask])
        prev_q = q
    parts = ax.violinplot(violin_data, positions=range(4), showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(quartile_colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Q1\n(Most Conf)', 'Q2', 'Q3', 'Q4\n(Least Conf)'])
    ax.set_ylabel('|Error| (veh/h)', fontweight='bold', color=COLORS['text'])
    ax.set_title('(b) Error by Confidence Quartile', fontweight='bold', color=COLORS['text'])
    
    # Cumulative curve
    ax = axes[1, 0]
    sorted_idx = np.argsort(uncertainties)
    sorted_err = abs_errors[sorted_idx]
    cum_pct = np.arange(1, len(sorted_err)+1) / len(sorted_err) * 100
    cum_mae = np.cumsum(sorted_err) / np.arange(1, len(sorted_err)+1)
    sample = np.linspace(0, len(cum_pct)-1, 500).astype(int)
    ax.plot(cum_pct[sample], cum_mae[sample], color=COLORS['sky_blue'], linewidth=2)
    ax.fill_between(cum_pct[sample], cum_mae[sample], alpha=0.3, color=COLORS['sky_blue'])
    for p in [50, 90]:
        i = int(len(sorted_err) * p / 100)
        ax.scatter([p], [cum_mae[i]], s=100, color=COLORS['rose'], edgecolor='white', zorder=5)
        ax.annotate(f'{p}%: {cum_mae[i]:.1f}', xy=(p, cum_mae[i]), xytext=(p+3, cum_mae[i]+5), fontweight='bold')
    ax.set_xlabel('% Most Confident Predictions', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Cumulative MAE', fontweight='bold', color=COLORS['text'])
    ax.set_title('(c) Cumulative Calibration', fontweight='bold', color=COLORS['text'])
    ax.grid(True, alpha=0.3)
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
╔══════════════════════════════════════╗
║     CALIBRATION SUMMARY              ║
╠══════════════════════════════════════╣
║  ECE:            {ece:.4f}             ║
║  Spearman ρ:     {spearman_corr:.4f}             ║
║  Pearson r:      {pearson_corr:.4f}             ║
║                                      ║
║  MAE (all):      {mae_all:.1f} veh/h          ║
║  MAE (90% conf): {mae_top90:.1f} veh/h          ║
║  MAE (50% conf): {mae_top50:.1f} veh/h          ║
║                                      ║
║  Error Reduction:                    ║
║    @90%: {100*(1-mae_top90/mae_all):.1f}% lower                 ║
║    @50%: {100*(1-mae_top50/mae_all):.1f}% lower                 ║
╚══════════════════════════════════════╝
"""
    ax.text(0.15, 0.5, summary, fontsize=11, fontfamily='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['cream'], edgecolor=COLORS['border']))
    
    plt.suptitle('Uncertainty Calibration Analysis', fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uq_calibration_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: uq_calibration_analysis.png")
    return {'ece': ece, 'spearman': spearman_corr, 'mae_all': mae_all, 'mae_top90': mae_top90}


def main():
    print("="*80)
    print("COMPREHENSIVE UQ ANALYSIS (Using Pre-computed Data)")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = load_precomputed_data()
    
    analysis_1_threshold_decision(results, OUTPUT_DIR)
    analysis_2_uncertainty_heatmap(results, OUTPUT_DIR)
    analysis_3_feature_error(results, OUTPUT_DIR)
    analysis_4_calibration(results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nGenerated figures:")
    print(f"  • {OUTPUT_DIR}/uq_threshold_analysis.png")
    print(f"  • {OUTPUT_DIR}/uq_spatial_heatmap.png")
    print(f"  • {OUTPUT_DIR}/uq_high_uncertainty_zones.png")
    print(f"  • {OUTPUT_DIR}/uq_feature_analysis.png")
    print(f"  • {OUTPUT_DIR}/uq_calibration_analysis.png")


if __name__ == '__main__':
    main()
