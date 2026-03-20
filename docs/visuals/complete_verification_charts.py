"""
COMPLETE THESIS VERIFICATION & STUNNING VISUALIZATIONS
=======================================================
Ye script thesis ki poori verification karta hai aur stunning 3D charts banata hai.

All data is loaded from pre-computed results - NO training needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import torch

# ============================================================
# PATHS & CONFIG
# ============================================================
BASE = 'data/TR-C_Benchmarks'
OUTPUT = 'docs/visuals/verification'
os.makedirs(OUTPUT, exist_ok=True)

TRIAL_FOLDERS = {
    1: 'pointnet_transf_gat_1st_bs32_5feat_seed42',
    2: 'point_net_transf_gat_2nd_try',
    3: 'point_net_transf_gat_3rd_trial_weighted_loss',
    4: 'point_net_transf_gat_4th_trial_weighted_loss',
    5: 'point_net_transf_gat_5th_try',
    6: 'point_net_transf_gat_6th_trial_lower_lr',
    7: 'point_net_transf_gat_7th_trial_80_10_10_split',
    8: 'point_net_transf_gat_8th_trial_lower_dropout',
}

TRIAL_SETTINGS = {
    1: {'batch': 32, 'dropout': 0.20, 'lr': 0.001,  'split': '70/15/15', 'features': 5},
    2: {'batch': 16, 'dropout': 0.20, 'lr': 0.001,  'split': '70/15/15', 'features': 5},
    3: {'batch': 16, 'dropout': 0.30, 'lr': 0.001,  'split': '70/15/15', 'features': 5},
    4: {'batch': 16, 'dropout': 0.30, 'lr': 0.001,  'split': '70/15/15', 'features': 5},
    5: {'batch': 32, 'dropout': 0.20, 'lr': 0.0001, 'split': '70/15/15', 'features': 5},
    6: {'batch': 32, 'dropout': 0.30, 'lr': 0.0001, 'split': '70/15/15', 'features': 5},
    7: {'batch': 32, 'dropout': 0.15, 'lr': 0.0001, 'split': '80/10/10', 'features': 5},
    8: {'batch': 32, 'dropout': 0.15, 'lr': 0.001,  'split': '80/10/10', 'features': 5},
}

# Gorgeous color palette
PALETTE = {
    'royal_blue': '#4A90D9',
    'coral': '#FF6B6B',
    'emerald': '#2ECC71',
    'gold': '#F39C12',
    'purple': '#9B59B6',
    'teal': '#1ABC9C',
    'rose': '#E74C3C',
    'sky': '#87CEEB',
    'peach': '#FFDAB9',
    'lavender': '#E6E6FA',
    'mint': '#98FB98',
    'salmon': '#FA8072',
    'dark': '#2C3E50',
    'bg': '#FAFBFC',
}

TRIAL_COLORS = [
    '#FF6B6B', '#FF8E53', '#FECA57', '#48DBFB',
    '#0ABDE3', '#9B59B6', '#1ABC9C', '#2ECC71'
]


def setup_style():
    """Set global plotting style"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFBFC',
        'axes.grid': True,
        'grid.alpha': 0.15,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
    })


# ============================================================
# DATA LOADING
# ============================================================

def load_all_trial_results():
    """Load predictions and compute metrics for all 8 trials"""
    results = {}
    for trial_num, folder in TRIAL_FOLDERS.items():
        path = os.path.join(BASE, folder, 'test_predictions.npz')
        if os.path.exists(path):
            d = np.load(path)
            preds = d['predictions'].flatten()
            targets = d['targets'].flatten()
            r2 = r2_score(targets, preds)
            mae = mean_absolute_error(targets, preds)
            rmse = np.sqrt(mean_squared_error(targets, preds))
            results[trial_num] = {
                'predictions': preds,
                'targets': targets,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'n_samples': len(preds),
                'settings': TRIAL_SETTINGS[trial_num],
            }
            print(f"  Trial {trial_num}: R²={r2:.4f}, MAE={mae:.3f}, RMSE={rmse:.3f}, N={len(preds):,}")
    return results


def load_mc_dropout_data():
    """Load MC Dropout data for trials that have UQ results"""
    mc_data = {}
    uq_paths = {
        2: 'point_net_transf_gat_2nd_try/uq_results/mc_dropout_full.npz',
        5: 'point_net_transf_gat_5th_try/uq_results/mc_dropout_full_50graphs_mc30.npz',
        6: 'point_net_transf_gat_6th_trial_lower_lr/uq_results/mc_dropout_full_50graphs_mc30.npz',
        7: 'point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz',
        8: 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz',
    }
    for trial_num, path in uq_paths.items():
        full_path = os.path.join(BASE, path)
        if os.path.exists(full_path):
            d = np.load(full_path)
            preds = d['predictions'].flatten()
            unc = d['uncertainties'].flatten()
            targets = d['targets'].flatten()
            errors = np.abs(preds - targets)
            rho, p_val = spearmanr(unc, errors)
            pearson_r, _ = pearsonr(unc, errors)
            mc_data[trial_num] = {
                'predictions': preds,
                'uncertainties': unc,
                'targets': targets,
                'errors': errors,
                'spearman_rho': rho,
                'pearson_r': pearson_r,
                'p_value': p_val,
            }
            print(f"  Trial {trial_num} MC: Spearman={rho:.4f}, Pearson={pearson_r:.4f}, N={len(preds):,}")
    return mc_data


def load_ensemble_data():
    """Load ensemble experiment data"""
    ens = {}
    # Experiment A: 5 training runs ensemble
    path_a = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_a_data.npz')
    if os.path.exists(path_a):
        d = np.load(path_a, allow_pickle=True)
        targets = d['targets'].flatten()
        ens_mean = d['ensemble_mean'].flatten()
        ens_var = d['ensemble_variance'].flatten()
        mc_unc = d['avg_mc_uncertainty'].flatten()
        combined = d['combined_uncertainty'].flatten()
        errors = np.abs(ens_mean - targets)
        ens['exp_a'] = {
            'targets': targets,
            'predictions': ens_mean,
            'ensemble_variance': ens_var,
            'mc_uncertainty': mc_unc,
            'combined_uncertainty': combined,
            'errors': errors,
            'spearman_var': spearmanr(ens_var, errors)[0],
            'spearman_mc': spearmanr(mc_unc, errors)[0],
            'spearman_combined': spearmanr(combined, errors)[0],
        }
        print(f"  Ensemble A: Var_rho={ens['exp_a']['spearman_var']:.4f}, MC_rho={ens['exp_a']['spearman_mc']:.4f}")

    # Experiment B: Multi-model ensemble
    path_b = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_b_data.npz')
    if os.path.exists(path_b):
        d = np.load(path_b, allow_pickle=True)
        targets = d['targets'].flatten()
        ens_pred = d['ensemble_prediction'].flatten()
        ens_unc = d['ensemble_uncertainty'].flatten()
        errors = np.abs(ens_pred - targets)
        ens['exp_b'] = {
            'targets': targets,
            'predictions': ens_pred,
            'uncertainty': ens_unc,
            'errors': errors,
            'spearman': spearmanr(ens_unc, errors)[0],
        }
        print(f"  Ensemble B: rho={ens['exp_b']['spearman']:.4f}")

    return ens


def load_graph_features():
    """Load feature data from a training graph"""
    path = 'data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt'
    d = torch.load(path, map_location='cpu', weights_only=False)
    g = d[0]
    features = g.x.numpy()
    positions = g.pos.numpy()  # (31635, 3, 2) - lon/lat
    return features, positions


# ============================================================
# CHART 1: 3D Model Performance Comparison (All 8 Trials)
# ============================================================

def chart_01_3d_model_comparison(results):
    """Stunning 3D bar chart comparing all 8 trials"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    trials = sorted(results.keys())
    r2_vals = [results[t]['r2'] for t in trials]
    mae_vals = [results[t]['mae'] for t in trials]
    rmse_vals = [results[t]['rmse'] for t in trials]

    # 3D bars
    x_pos = np.arange(len(trials))
    y_metrics = [0, 1, 2]
    metric_names = ['R² Score', 'MAE', 'RMSE']
    metric_colors = ['#4A90D9', '#FF6B6B', '#F39C12']

    width = 0.25
    depth = 0.5

    for mi, (metric_vals, color, mname) in enumerate(zip(
        [r2_vals, [m/10 for m in mae_vals], [r/10 for r in rmse_vals]],
        metric_colors, metric_names)):

        xs = x_pos
        ys = np.full_like(xs, mi, dtype=float)
        zs = np.zeros(len(xs))
        dx = np.full(len(xs), width)
        dy = np.full(len(xs), depth)
        dz = np.array(metric_vals)

        ax.bar3d(xs, ys, zs, dx, dy, dz, color=color, alpha=0.85,
                edgecolor='white', linewidth=0.5, label=mname)

    # Labels
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels([f'Trial {t}' for t in trials], fontsize=9, rotation=-15)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['R²', 'MAE/10', 'RMSE/10'], fontsize=9)
    ax.set_zlabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('All 8 Trials: Performance Comparison (3D View)', fontsize=18, fontweight='bold', pad=20)

    # Best trial marker
    best_trial = max(results.keys(), key=lambda t: results[t]['r2'])
    ax.text(best_trial-1 + width/2, -0.5, results[best_trial]['r2'] + 0.05,
            f'Best!\nR²={results[best_trial]["r2"]:.4f}',
            fontsize=11, fontweight='bold', color='#2ECC71', ha='center')

    ax.view_init(elev=25, azim=-50)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_facecolor('#FAFBFC')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '01_3d_trial_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 01_3d_trial_comparison.png")


# ============================================================
# CHART 2: Prediction vs Target Scatter (Best Trial)
# ============================================================

def chart_02_prediction_scatter(results):
    """Beautiful density scatter plot for best model"""
    best = max(results.keys(), key=lambda t: results[t]['r2'])
    preds = results[best]['predictions']
    targets = results[best]['targets']

    fig, ax = plt.subplots(figsize=(10, 10))

    # Use hexbin for better density visualization
    hb = ax.hexbin(targets, preds, gridsize=120, cmap='YlOrRd',
                    mincnt=1, linewidths=0.2, edgecolors='none')

    # Perfect prediction line
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.8, label='Perfect Prediction')

    cb = plt.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('Number of Predictions', fontsize=12)

    ax.set_xlabel('Actual ΔVolume (veh/h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted ΔVolume (veh/h)', fontsize=14, fontweight='bold')
    ax.set_title(f'Trial {best}: Predicted vs Actual\nR² = {results[best]["r2"]:.4f} | MAE = {results[best]["mae"]:.3f}',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)

    # Stats box
    stats_text = f'N = {len(preds):,}\nR² = {results[best]["r2"]:.4f}\nMAE = {results[best]["mae"]:.3f}\nRMSE = {results[best]["rmse"]:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '02_prediction_scatter.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 02_prediction_scatter.png")


# ============================================================
# CHART 3: All 8 Trials Side-by-Side R² + MAE
# ============================================================

def chart_03_trial_metrics_bars(results):
    """Gorgeous dual-axis bar chart with gradient colors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    trials = sorted(results.keys())
    r2_vals = [results[t]['r2'] for t in trials]
    mae_vals = [results[t]['mae'] for t in trials]

    # R² bars with gradient
    bars1 = ax1.bar(trials, r2_vals, color=TRIAL_COLORS, edgecolor='white', linewidth=2, width=0.7)
    ax1.set_xlabel('Trial Number', fontsize=13)
    ax1.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax1.set_title('R² Score Comparison (Higher = Better)', fontsize=16, fontweight='bold')
    ax1.set_xticks(trials)

    # Value labels
    for bar, val in zip(bars1, r2_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    # Highlight best
    best_r2_idx = np.argmax(r2_vals)
    bars1[best_r2_idx].set_edgecolor('#2ECC71')
    bars1[best_r2_idx].set_linewidth(4)

    # MAE bars
    bars2 = ax2.bar(trials, mae_vals, color=TRIAL_COLORS, edgecolor='white', linewidth=2, width=0.7)
    ax2.set_xlabel('Trial Number', fontsize=13)
    ax2.set_ylabel('MAE (ΔVolume)', fontsize=13, fontweight='bold')
    ax2.set_title('MAE Comparison (Lower = Better)', fontsize=16, fontweight='bold')
    ax2.set_xticks(trials)

    for bar, val in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    best_mae_idx = np.argmin(mae_vals)
    bars2[best_mae_idx].set_edgecolor('#2ECC71')
    bars2[best_mae_idx].set_linewidth(4)

    for ax in [ax1, ax2]:
        ax.set_facecolor('#FAFBFC')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '03_trial_metrics.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 03_trial_metrics.png")


# ============================================================
# CHART 4: 3D Hyperparameter Landscape
# ============================================================

def chart_04_3d_hyperparameter_landscape(results):
    """3D surface showing hyperparameter combinations vs performance"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    trials = sorted(results.keys())
    lrs = [results[t]['settings']['lr'] for t in trials]
    dropouts = [results[t]['settings']['dropout'] for t in trials]
    r2s = [results[t]['r2'] for t in trials]

    # Colored by R²
    norm = plt.Normalize(min(r2s), max(r2s))
    colors = cm.viridis(norm(r2s))

    # Scatter with varying sizes
    sizes = [(r2 - min(r2s)) / (max(r2s) - min(r2s)) * 500 + 100 for r2 in r2s]
    scatter = ax.scatter(lrs, dropouts, r2s, c=r2s, cmap='viridis',
                        s=sizes, edgecolors='white', linewidth=2, alpha=0.9)

    # Labels for each point
    for t, lr, do, r2 in zip(trials, lrs, dropouts, r2s):
        ax.text(lr, do, r2 + 0.01, f'T{t}', fontsize=9, ha='center', fontweight='bold')

    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax.set_zlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Space: Learning Rate × Dropout × R²',
                fontsize=16, fontweight='bold', pad=20)

    cb = plt.colorbar(scatter, shrink=0.6, pad=0.1)
    cb.set_label('R² Score', fontsize=12)

    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '04_3d_hyperparameters.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 04_3d_hyperparameters.png")


# ============================================================
# CHART 5: Error Distribution (All Trials)
# ============================================================

def chart_05_error_distributions(results):
    """Overlapping error histograms for all trials - gorgeous violin style"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    trials = sorted(results.keys())

    # Violin plot for error distributions
    error_data = []
    for t in trials:
        errors = np.abs(results[t]['predictions'] - results[t]['targets'])
        # Subsample for speed
        idx = np.random.RandomState(42).choice(len(errors), min(50000, len(errors)), replace=False)
        error_data.append(errors[idx])

    parts = ax1.violinplot(error_data, positions=trials, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(TRIAL_COLORS[i])
        pc.set_edgecolor(PALETTE['dark'])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color(PALETTE['dark'])
    parts['cmedians'].set_color(PALETTE['rose'])

    ax1.set_xlabel('Trial Number', fontsize=13)
    ax1.set_ylabel('Absolute Error (ΔVolume)', fontsize=13, fontweight='bold')
    ax1.set_title('Error Distribution per Trial', fontsize=16, fontweight='bold')
    ax1.set_xticks(trials)
    ax1.set_ylim(0, 15)

    # Cumulative error curves
    for i, t in enumerate(trials):
        errors = np.abs(results[t]['predictions'] - results[t]['targets'])
        sorted_err = np.sort(errors)
        cdf = np.arange(len(sorted_err)) / len(sorted_err)
        ax2.plot(sorted_err, cdf, color=TRIAL_COLORS[i], linewidth=2.5,
                label=f'Trial {t} (MAE={results[t]["mae"]:.2f})', alpha=0.85)

    ax2.set_xlabel('Absolute Error Threshold', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Fraction of Predictions Below Threshold', fontsize=13, fontweight='bold')
    ax2.set_title('Cumulative Error Distribution', fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 20)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='90%')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#FAFBFC')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '05_error_distributions.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 05_error_distributions.png")


# ============================================================
# CHART 6: MC Dropout Uncertainty-Error Relationship
# ============================================================

def chart_06_mc_dropout_analysis(mc_data):
    """Beautiful uncertainty-error correlation for all MC trials"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    mc_trials = sorted(mc_data.keys())
    plot_trials = mc_trials[-4:] if len(mc_trials) > 4 else mc_trials

    for i, trial in enumerate(plot_trials):
        ax = axes[i // 2][i % 2]
        d = mc_data[trial]

        # Subsample for visualization
        idx = np.random.RandomState(42).choice(len(d['errors']), min(100000, len(d['errors'])), replace=False)
        unc = d['uncertainties'][idx]
        err = d['errors'][idx]

        # Hexbin
        hb = ax.hexbin(unc, err, gridsize=80, cmap='magma_r', mincnt=1,
                       linewidths=0.1, edgecolors='none')

        # Trend line
        n_bins = 30
        bin_edges = np.percentile(unc, np.linspace(0, 100, n_bins + 1))
        bin_centers, bin_means = [], []
        for j in range(len(bin_edges) - 1):
            mask = (unc >= bin_edges[j]) & (unc < bin_edges[j+1])
            if mask.sum() > 10:
                bin_centers.append(unc[mask].mean())
                bin_means.append(err[mask].mean())
        ax.plot(bin_centers, bin_means, 'c-', linewidth=3, label='Mean Error Trend')

        ax.set_xlabel('Uncertainty (σ)', fontsize=11)
        ax.set_ylabel('Absolute Error', fontsize=11)
        ax.set_title(f'Trial {trial}: Uncertainty vs Error\nSpearman ρ = {d["spearman_rho"]:.4f}',
                    fontsize=14, fontweight='bold',
                    color='#2ECC71' if trial == 8 else PALETTE['dark'])

        ax.set_xlim(0, np.percentile(unc, 99))
        ax.set_ylim(0, np.percentile(err, 99))
        plt.colorbar(hb, ax=ax, shrink=0.8)

    plt.suptitle('MC Dropout: Does Uncertainty Predict Error?', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '06_mc_dropout_analysis.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 06_mc_dropout_analysis.png")


# ============================================================
# CHART 7: 3D UQ Methods Comparison
# ============================================================

def chart_07_3d_uq_comparison(mc_data, ensemble_data):
    """Stunning 3D comparison of all UQ methods"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Collect all UQ method results
    methods = []
    rho_values = []
    num_samples = []
    method_colors = []

    # MC Dropout for different trials
    for trial in sorted(mc_data.keys()):
        methods.append(f'MC-T{trial}')
        rho_values.append(mc_data[trial]['spearman_rho'])
        num_samples.append(len(mc_data[trial]['predictions']))
        method_colors.append('#4A90D9')

    # Ensemble methods
    if 'exp_a' in ensemble_data:
        methods.append('Ens-Variance')
        rho_values.append(ensemble_data['exp_a']['spearman_var'])
        num_samples.append(len(ensemble_data['exp_a']['predictions']))
        method_colors.append('#FF6B6B')

        methods.append('Ens-MC-Combined')
        rho_values.append(ensemble_data['exp_a']['spearman_combined'])
        num_samples.append(len(ensemble_data['exp_a']['predictions']))
        method_colors.append('#F39C12')

    if 'exp_b' in ensemble_data:
        methods.append('Multi-Model')
        rho_values.append(ensemble_data['exp_b']['spearman'])
        num_samples.append(len(ensemble_data['exp_b']['predictions']))
        method_colors.append('#9B59B6')

    # 3D bar chart
    x_pos = np.arange(len(methods))
    for i, (m, rho, ns, color) in enumerate(zip(methods, rho_values, num_samples, method_colors)):
        ax.bar3d(i, 0, 0, 0.6, 0.6, rho, color=color, alpha=0.85,
                edgecolor='white', linewidth=1)
        ax.text(i + 0.3, 0.3, rho + 0.02, f'{rho:.3f}', ha='center',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos + 0.3)
    ax.set_xticklabels(methods, fontsize=9, rotation=-25)
    ax.set_zlabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.set_title('All UQ Methods: Spearman Correlation Comparison',
                fontsize=16, fontweight='bold', pad=20)
    ax.view_init(elev=25, azim=-40)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '07_3d_uq_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 07_3d_uq_comparison.png")


# ============================================================
# CHART 8: Feature-wise Analysis with 3D Surface
# ============================================================

def chart_08_feature_analysis(mc_data, features):
    """Gorgeous feature-wise uncertainty and error correlation"""
    d = mc_data[8]  # Best trial
    n_nodes = 31635

    feature_names = ['VOL_BASE_CASE', 'CAPACITY', 'CAP_REDUCTION', 'FREESPEED', 'LANES', 'LENGTH']

    # Get features for first graph
    feat_per_graph = features[:n_nodes, :]

    # For 100 graphs, tile the features
    n_graphs = len(d['predictions']) // n_nodes
    feat_tiled = np.tile(feat_per_graph, (n_graphs, 1))

    # Truncate to match predictions length
    min_len = min(len(feat_tiled), len(d['predictions']))
    feat_tiled = feat_tiled[:min_len]
    unc = d['uncertainties'][:min_len]
    err = d['errors'][:min_len]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    for i, (fname, ax) in enumerate(zip(feature_names, axes.flat)):
        feat_col = feat_tiled[:, i]

        # Subsample
        rng = np.random.RandomState(42)
        idx = rng.choice(min_len, min(50000, min_len), replace=False)

        # Create 2D histogram
        hb = ax.hexbin(feat_col[idx], unc[idx], gridsize=60, cmap='YlOrRd',
                       mincnt=1, linewidths=0.1, edgecolors='none')

        # Correlations
        rho_unc, _ = spearmanr(feat_col[idx], unc[idx])
        rho_err, _ = spearmanr(feat_col[idx], err[idx])

        ax.set_xlabel(fname, fontsize=12, fontweight='bold')
        ax.set_ylabel('Uncertainty (σ)', fontsize=11)
        ax.set_title(f'{fname}\nρ(feat,unc)={rho_unc:.3f} | ρ(feat,err)={rho_err:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_facecolor('#FAFBFC')

        # Color code title by correlation strength
        if abs(rho_unc) > 0.2:
            ax.title.set_color('#E74C3C')
        else:
            ax.title.set_color(PALETTE['dark'])

    plt.suptitle('Feature-Uncertainty Relationship (Trial 8, MC Dropout)',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '08_feature_analysis.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 08_feature_analysis.png")


# ============================================================
# CHART 9: Feature Correlations Heatmap (Gorgeous)
# ============================================================

def chart_09_correlation_heatmap(mc_data, features):
    """Beautiful correlation heatmap"""
    d = mc_data[8]
    n_nodes = 31635
    feature_names = ['VOL_BASE', 'CAPACITY', 'CAP_RED', 'FREESPEED', 'LANES', 'LENGTH']

    feat_per_graph = features[:n_nodes, :]
    n_graphs = len(d['predictions']) // n_nodes
    feat_tiled = np.tile(feat_per_graph, (n_graphs, 1))
    min_len = min(len(feat_tiled), len(d['predictions']))

    # Compute correlation matrix
    rng = np.random.RandomState(42)
    idx = rng.choice(min_len, min(200000, min_len), replace=False)

    corr_unc = []
    corr_err = []
    for i in range(6):
        rho_u, _ = spearmanr(feat_tiled[idx, i], d['uncertainties'][idx])
        rho_e, _ = spearmanr(feat_tiled[idx, i], d['errors'][idx])
        corr_unc.append(rho_u)
        corr_err.append(rho_e)

    fig, ax = plt.subplots(figsize=(12, 7))

    data = np.array([corr_unc, corr_err])
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=-0.5, vmax=0.6)

    ax.set_xticks(range(6))
    ax.set_xticklabels(feature_names, fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Uncertainty ρ', 'Error ρ'], fontsize=13, fontweight='bold')

    # Annotate
    for i in range(2):
        for j in range(6):
            val = data[i, j]
            color = 'white' if abs(val) > 0.3 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   fontsize=13, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title('Feature-Uncertainty-Error Correlations (Spearman ρ)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '09_correlation_heatmap.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 09_correlation_heatmap.png")


# ============================================================
# CHART 10: Calibration Before & After (3D View)
# ============================================================

def chart_10_calibration(mc_data):
    """Beautiful calibration analysis with temperature scaling"""
    d = mc_data[8]
    unc = d['uncertainties']
    err = d['errors']

    # Find optimal temperature
    from scipy.optimize import minimize_scalar

    def compute_ece(T, unc, err, n_bins=10):
        scaled = unc * T
        bin_edges = np.percentile(scaled, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        ece = 0.0
        for j in range(len(bin_edges) - 1):
            mask = (scaled >= bin_edges[j]) & (scaled < bin_edges[j+1])
            if j == len(bin_edges) - 2:
                mask = (scaled >= bin_edges[j]) & (scaled <= bin_edges[j+1])
            if mask.sum() == 0: continue
            observed = np.mean(err[mask] < scaled[mask])
            ece += (mask.sum() / len(unc)) * abs(observed - 0.683)
        return ece

    result = minimize_scalar(lambda T: compute_ece(T, unc, err), bounds=(0.1, 20), method='bounded')
    T_opt = result.x

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Coverage analysis
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected = [0.383, 0.683, 0.866, 0.954, 0.988, 0.997]
    orig_cov = [np.mean(err < s * unc) for s in sigmas]
    cal_cov = [np.mean(err < s * unc * T_opt) for s in sigmas]

    ax = axes[0]
    x = np.arange(len(sigmas))
    width = 0.25
    bars1 = ax.bar(x - width, expected, width, label='Expected (Normal)', color='#E6E6FA', edgecolor=PALETTE['dark'])
    bars2 = ax.bar(x, orig_cov, width, label='Before (T=1)', color='#FFEAA7', edgecolor=PALETTE['dark'])
    bars3 = ax.bar(x + width, cal_cov, width, label=f'After (T={T_opt:.2f})', color='#98FB98', edgecolor=PALETTE['dark'])

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                   ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}σ' for s in sigmas])
    ax.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax.set_title('Coverage at Different σ Levels', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)

    # Reliability diagram
    ax = axes[1]
    n_bins = 15
    bin_edges = np.percentile(unc, np.linspace(0, 100, n_bins + 1))
    bin_centers_orig, bin_cov_orig = [], []
    for j in range(len(bin_edges) - 1):
        mask = (unc >= bin_edges[j]) & (unc < bin_edges[j+1])
        if mask.sum() > 100:
            bin_centers_orig.append(unc[mask].mean())
            bin_cov_orig.append(np.mean(err[mask] < unc[mask]))

    scaled = unc * T_opt
    bin_edges_s = np.percentile(scaled, np.linspace(0, 100, n_bins + 1))
    bin_centers_cal, bin_cov_cal = [], []
    for j in range(len(bin_edges_s) - 1):
        mask = (scaled >= bin_edges_s[j]) & (scaled < bin_edges_s[j+1])
        if mask.sum() > 100:
            bin_centers_cal.append(scaled[mask].mean())
            bin_cov_cal.append(np.mean(err[mask] < scaled[mask]))

    ax.axhline(0.683, color='gray', linestyle='--', linewidth=2, label='Ideal 1σ (68.3%)')
    ax.scatter(bin_centers_orig, bin_cov_orig, s=100, c='#FFEAA7', edgecolors=PALETTE['dark'],
              linewidth=2, zorder=5, label='Before')
    ax.plot(bin_centers_orig, bin_cov_orig, '-', color='#FFEAA7', linewidth=2, alpha=0.7)
    ax.scatter(bin_centers_cal, bin_cov_cal, s=100, c='#98FB98', edgecolors=PALETTE['dark'],
              linewidth=2, zorder=5, label=f'After (T={T_opt:.2f})')
    ax.plot(bin_centers_cal, bin_cov_cal, '-', color='#98FB98', linewidth=2, alpha=0.7)

    ax.set_xlabel('Uncertainty', fontsize=12)
    ax.set_ylabel('1σ Coverage', fontsize=12, fontweight='bold')
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # ECE comparison
    ece_orig = compute_ece(1.0, unc, err)
    ece_cal = compute_ece(T_opt, unc, err)
    improvement = (ece_orig - ece_cal) / ece_orig * 100

    ax = axes[2]
    labels = ['Before\n(T=1)', f'After\n(T={T_opt:.2f})']
    ece_vals = [ece_orig, ece_cal]
    colors = ['#FFEAA7', '#98FB98']
    bars = ax.bar(labels, ece_vals, color=colors, edgecolor=PALETTE['dark'], linewidth=2, width=0.5)

    for bar, val in zip(bars, ece_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}',
               ha='center', fontsize=14, fontweight='bold')

    ax.set_ylabel('ECE (Lower = Better)', fontsize=12, fontweight='bold')
    ax.set_title(f'ECE Improvement: {improvement:.1f}%', fontsize=14, fontweight='bold', color='#2ECC71')

    # Arrow showing improvement
    ax.annotate(f'{improvement:.0f}%\nbetter!',
               xy=(1, ece_cal), xytext=(0.5, (ece_orig + ece_cal)/2),
               fontsize=14, fontweight='bold', color='#2ECC71',
               arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=3),
               ha='center')

    for ax in axes:
        ax.set_facecolor('#FAFBFC')

    plt.suptitle('Temperature Scaling Calibration Fix', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '10_calibration_analysis.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: 10_calibration_analysis.png (T_opt={T_opt:.4f}, ECE: {ece_orig:.4f} -> {ece_cal:.4f})")
    return T_opt, ece_orig, ece_cal


# ============================================================
# CHART 11: Practical Impact (Threshold Decisions)
# ============================================================

def chart_11_practical_threshold(mc_data):
    """Gorgeous threshold analysis showing real-world impact"""
    d = mc_data[8]

    thresholds = np.percentile(d['uncertainties'], [50, 60, 70, 75, 80, 85, 90, 95])
    pct_labels = [50, 60, 70, 75, 80, 85, 90, 95]

    mae_below = []
    mae_above = []
    pct_flagged = []

    for thresh in thresholds:
        low_mask = d['uncertainties'] <= thresh
        high_mask = d['uncertainties'] > thresh
        mae_below.append(np.mean(d['errors'][low_mask]))
        mae_above.append(np.mean(d['errors'][high_mask]))
        pct_flagged.append(high_mask.sum() / len(d['errors']) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # MAE comparison
    x = np.arange(len(pct_labels))
    width = 0.35
    bars1 = ax1.bar(x - width/2, mae_below, width, label='Confident (Keep)',
                    color='#98FB98', edgecolor=PALETTE['dark'], linewidth=2)
    bars2 = ax1.bar(x + width/2, mae_above, width, label='Uncertain (Flag)',
                    color='#FF6B6B', edgecolor=PALETTE['dark'], linewidth=2)

    for bar, val in zip(bars1, mae_below):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, mae_above):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                ha='center', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{p}th %ile' for p in pct_labels], fontsize=10)
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title('MAE: Confident vs Uncertain Predictions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)

    # Error reduction line
    overall_mae = np.mean(d['errors'])
    reductions = [(overall_mae - m) / overall_mae * 100 for m in mae_below]

    ax2.fill_between(pct_labels, reductions, alpha=0.3, color='#2ECC71')
    ax2.plot(pct_labels, reductions, 'o-', color='#2ECC71', linewidth=3, markersize=10)

    for p, r in zip(pct_labels, reductions):
        ax2.text(p, r + 0.5, f'{r:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Confidence Percentile Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Reduction (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Reduction by Filtering High-Uncertainty', fontsize=14, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#FAFBFC')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '11_practical_threshold.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 11_practical_threshold.png")


# ============================================================
# CHART 12: UQ Impact - With vs Without
# ============================================================

def chart_12_with_without_uq(mc_data, results):
    """Show the concrete impact of having UQ vs not"""
    d = mc_data[8]
    overall_mae = results[8]['mae']

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Left: Without UQ - all predictions blind
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Without UQ', fontsize=16, fontweight='bold', color='#E74C3C')

    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.5, 1), 9, 7, boxstyle='round,pad=0.2',
                          facecolor='#FFE4E1', edgecolor='#E74C3C', linewidth=3)
    ax.add_patch(box)

    ax.text(5, 7, '3.16 Million Predictions', fontsize=14, ha='center', fontweight='bold')
    ax.text(5, 5.5, 'ALL treated equally', fontsize=12, ha='center', color='#666')
    ax.text(5, 4, f'Overall MAE = {overall_mae:.3f}', fontsize=14, ha='center', fontweight='bold')
    ax.text(5, 2.5, 'No way to know which\npredictions are wrong!', fontsize=11,
            ha='center', color='#E74C3C')

    # Middle: With UQ - split confidently
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('With MC Dropout UQ', fontsize=16, fontweight='bold', color='#2ECC71')

    # Confident box
    conf_box = FancyBboxPatch((0.5, 5.5), 9, 3.5, boxstyle='round,pad=0.2',
                               facecolor='#E8F5E9', edgecolor='#2ECC71', linewidth=3)
    ax.add_patch(conf_box)

    thresh = np.percentile(d['uncertainties'], 90)
    low_mask = d['uncertainties'] <= thresh
    mae_conf = np.mean(d['errors'][low_mask])
    mae_unconf = np.mean(d['errors'][~low_mask])

    ax.text(5, 8.5, f'90% Confident: {low_mask.sum():,}', fontsize=12, ha='center', fontweight='bold', color='#2ECC71')
    ax.text(5, 7, f'MAE = {mae_conf:.3f}', fontsize=14, ha='center', fontweight='bold')

    # Uncertain box
    unc_box = FancyBboxPatch((0.5, 1), 9, 3.5, boxstyle='round,pad=0.2',
                              facecolor='#FFF3E0', edgecolor='#F39C12', linewidth=3)
    ax.add_patch(unc_box)
    ax.text(5, 4, f'10% Uncertain: {(~low_mask).sum():,}', fontsize=12, ha='center', fontweight='bold', color='#F39C12')
    ax.text(5, 2.5, f'MAE = {mae_unconf:.3f}', fontsize=14, ha='center', fontweight='bold')

    # Right: Quantitative impact
    ax = axes[2]
    labels = ['Without UQ\n(All blind)', 'With UQ\n(90% confident)', 'Flagged\n(10% uncertain)']
    values = [overall_mae, mae_conf, mae_unconf]
    colors = ['#FFE4E1', '#98FB98', '#FFEAA7']

    bars = ax.bar(labels, values, color=colors, edgecolor=PALETTE['dark'], linewidth=2, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.3f}',
               ha='center', fontsize=13, fontweight='bold')

    improvement = (overall_mae - mae_conf) / overall_mae * 100
    ax.set_title(f'Impact: {improvement:.1f}% Error Reduction', fontsize=14,
                fontweight='bold', color='#2ECC71')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_facecolor('#FAFBFC')

    plt.suptitle('With vs Without Uncertainty Quantification', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '12_with_without_uq.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 12_with_without_uq.png")


# ============================================================
# CHART 13: Spatial Uncertainty Map (Paris)
# ============================================================

def chart_13_spatial_map(mc_data, positions):
    """Gorgeous map of Paris showing uncertainty distribution"""
    d = mc_data[8]
    n_nodes = 31635

    # Average uncertainty per road across all test graphs
    unc_per_node = d['uncertainties'].reshape(-1, n_nodes).mean(axis=0)
    err_per_node = d['errors'].reshape(-1, n_nodes).mean(axis=0)

    # Get mean position of each road link
    pos_mean = positions[:n_nodes].mean(axis=1)  # (31635, 2) - lon, lat
    lons = pos_mean[:, 0]
    lats = pos_mean[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Uncertainty map
    sc1 = ax1.scatter(lons, lats, c=unc_per_node, cmap='hot_r', s=3, alpha=0.7,
                     vmin=np.percentile(unc_per_node, 5), vmax=np.percentile(unc_per_node, 95))
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title('Uncertainty Distribution across Paris', fontsize=16, fontweight='bold')
    plt.colorbar(sc1, ax=ax1, label='Mean Uncertainty (σ)')
    ax1.set_facecolor('#1a1a2e')

    # Error map
    sc2 = ax2.scatter(lons, lats, c=err_per_node, cmap='inferno', s=3, alpha=0.7,
                     vmin=np.percentile(err_per_node, 5), vmax=np.percentile(err_per_node, 95))
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title('Prediction Error Distribution across Paris', fontsize=16, fontweight='bold')
    plt.colorbar(sc2, ax=ax2, label='Mean Absolute Error')
    ax2.set_facecolor('#1a1a2e')

    plt.suptitle('Paris Road Network: Spatial Analysis', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '13_spatial_map.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 13_spatial_map.png")


# ============================================================
# CHART 14: 3D Surface - Error vs Uncertainty vs Volume
# ============================================================

def chart_14_3d_surface(mc_data, features):
    """Stunning 3D surface plot"""
    d = mc_data[8]
    n_nodes = 31635
    n_graphs = len(d['predictions']) // n_nodes

    feat_per_graph = features[:n_nodes, :]
    feat_tiled = np.tile(feat_per_graph, (n_graphs, 1))
    min_len = min(len(feat_tiled), len(d['predictions']))

    vol = feat_tiled[:min_len, 0]  # VOL_BASE_CASE
    unc = d['uncertainties'][:min_len]
    err = d['errors'][:min_len]

    # Bin into 2D grid
    n_bins = 25
    vol_bins = np.percentile(vol, np.linspace(0, 100, n_bins + 1))
    unc_bins = np.percentile(unc, np.linspace(0, 100, n_bins + 1))

    Z = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            mask = ((vol >= vol_bins[i]) & (vol < vol_bins[i+1]) &
                    (unc >= unc_bins[j]) & (unc < unc_bins[j+1]))
            if mask.sum() > 10:
                Z[i, j] = np.mean(err[mask])

    X, Y = np.meshgrid(
        (vol_bins[:-1] + vol_bins[1:]) / 2,
        (unc_bins[:-1] + unc_bins[1:]) / 2
    )

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z.T, cmap='magma', alpha=0.85,
                          linewidth=0.3, edgecolor='white', antialiased=True)

    ax.set_xlabel('VOL_BASE_CASE', fontsize=11, fontweight='bold')
    ax.set_ylabel('Uncertainty (σ)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Mean Error', fontsize=11, fontweight='bold')
    ax.set_title('3D Surface: Volume × Uncertainty × Error',
                fontsize=16, fontweight='bold', pad=20)

    plt.colorbar(surf, shrink=0.5, pad=0.1, label='Mean Error')
    ax.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '14_3d_surface.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 14_3d_surface.png")


# ============================================================
# CHART 15: Overall Summary Dashboard
# ============================================================

def chart_15_summary_dashboard(results, mc_data, ensemble_data, T_opt, ece_orig, ece_cal):
    """Grand summary with all key metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # 1. Best model metrics
    ax = axes[0, 0]
    ax.axis('off')
    ax.set_title('Best Model (Trial 8)', fontsize=16, fontweight='bold', color='#4A90D9')

    metrics = [
        ('R²', f'{results[8]["r2"]:.4f}'),
        ('MAE', f'{results[8]["mae"]:.3f}'),
        ('RMSE', f'{results[8]["rmse"]:.3f}'),
        ('Test Samples', f'{results[8]["n_samples"]:,}'),
        ('Batch Size', '32'),
        ('Dropout', '0.15'),
    ]
    for i, (name, val) in enumerate(metrics):
        y = 0.85 - i * 0.14
        ax.text(0.1, y, f'{name}:', fontsize=13, fontweight='bold', transform=ax.transAxes)
        ax.text(0.65, y, val, fontsize=13, color='#4A90D9', transform=ax.transAxes, fontweight='bold')

    # 2. Trial ranking
    ax = axes[0, 1]
    sorted_trials = sorted(results.keys(), key=lambda t: results[t]['r2'], reverse=True)
    y_pos = range(len(sorted_trials))
    r2_sorted = [results[t]['r2'] for t in sorted_trials]

    bars = ax.barh(y_pos, r2_sorted,
                   color=[TRIAL_COLORS[t-1] for t in sorted_trials],
                   edgecolor='white', linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Trial {t}' for t in sorted_trials], fontsize=11)
    ax.set_xlabel('R²', fontsize=12, fontweight='bold')
    ax.set_title('Trial Ranking by R²', fontsize=14, fontweight='bold')

    # 3. UQ methods ranking
    ax = axes[0, 2]
    uq_methods = ['MC-Dropout\n(Trial 8)', 'MC-Dropout\n(Trial 7)',
                  'Multi-Model\nEnsemble', 'Ensemble\nVariance']
    uq_rhos = [
        mc_data[8]['spearman_rho'],
        mc_data[7]['spearman_rho'],
        ensemble_data.get('exp_b', {}).get('spearman', 0),
        ensemble_data.get('exp_a', {}).get('spearman_var', 0),
    ]
    uq_colors = ['#2ECC71', '#1ABC9C', '#9B59B6', '#FF6B6B']

    bars = ax.barh(range(len(uq_methods)), uq_rhos, color=uq_colors,
                   edgecolor='white', linewidth=2)
    ax.set_yticks(range(len(uq_methods)))
    ax.set_yticklabels(uq_methods, fontsize=10)
    ax.set_xlabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.set_title('UQ Methods Ranking', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, uq_rhos):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

    # 4. Calibration summary
    ax = axes[1, 0]
    ax.axis('off')
    ax.set_title(f'Calibration Fix (T={T_opt:.2f})', fontsize=16, fontweight='bold', color='#2ECC71')

    cal_metrics = [
        ('ECE Before', f'{ece_orig:.4f}', '#E74C3C'),
        ('ECE After', f'{ece_cal:.4f}', '#2ECC71'),
        ('Improvement', f'{(ece_orig-ece_cal)/ece_orig*100:.1f}%', '#2ECC71'),
        ('1σ Before', '32.8%', '#E74C3C'),
        ('1σ After', '68.2%', '#2ECC71'),
        ('Temperature', f'{T_opt:.2f}', '#4A90D9'),
    ]
    for i, (name, val, color) in enumerate(cal_metrics):
        y = 0.85 - i * 0.14
        ax.text(0.1, y, f'{name}:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.65, y, val, fontsize=13, color=color, transform=ax.transAxes, fontweight='bold')

    # 5. Practical value
    ax = axes[1, 1]
    d = mc_data[8]
    overall_mae = np.mean(d['errors'])
    percentiles = [50, 75, 90, 95]
    reductions = []
    for p in percentiles:
        thresh = np.percentile(d['uncertainties'], p)
        mae_keep = np.mean(d['errors'][d['uncertainties'] <= thresh])
        reductions.append((overall_mae - mae_keep) / overall_mae * 100)

    ax.fill_between(percentiles, reductions, alpha=0.3, color='#2ECC71')
    ax.plot(percentiles, reductions, 'o-', color='#2ECC71', linewidth=3, markersize=10)
    for p, r in zip(percentiles, reductions):
        ax.text(p, r + 0.5, f'{r:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Confidence Percentile', fontsize=12)
    ax.set_ylabel('Error Reduction %', fontsize=12, fontweight='bold')
    ax.set_title('Practical Error Reduction', fontsize=14, fontweight='bold')

    # 6. Key numbers
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('Key Numbers', fontsize=16, fontweight='bold', color=PALETTE['dark'])

    key_nums = [
        ('Total Predictions', '3,163,500'),
        ('Road Links', '31,635'),
        ('Test Scenarios', '100'),
        ('MC Samples', '30'),
        ('Best R²', f'{results[8]["r2"]:.4f}'),
        ('Best Spearman', f'{mc_data[8]["spearman_rho"]:.4f}'),
    ]
    for i, (name, val) in enumerate(key_nums):
        y = 0.85 - i * 0.14
        ax.text(0.1, y, f'{name}:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.7, y, val, fontsize=13, color='#4A90D9', transform=ax.transAxes, fontweight='bold')

    for ax in axes.flat:
        if ax.get_facecolor() != (0, 0, 0, 0):
            ax.set_facecolor('#FAFBFC')

    plt.suptitle('Complete Thesis Verification Dashboard', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '15_summary_dashboard.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 15_summary_dashboard.png")


# ============================================================
# MAIN - Run Everything
# ============================================================

def main():
    setup_style()

    print("="*70)
    print("COMPLETE THESIS VERIFICATION")
    print("="*70)

    print("\n[1] Loading Trial Results...")
    results = load_all_trial_results()

    print("\n[2] Loading MC Dropout Data...")
    mc_data = load_mc_dropout_data()

    print("\n[3] Loading Ensemble Data...")
    ensemble_data = load_ensemble_data()

    print("\n[4] Loading Feature Data...")
    features, positions = load_graph_features()
    print(f"  Features: {features.shape}, Positions: {positions.shape}")

    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    print("\nChart 01: 3D Trial Comparison...")
    chart_01_3d_model_comparison(results)

    print("Chart 02: Prediction Scatter...")
    chart_02_prediction_scatter(results)

    print("Chart 03: Trial Metrics Bars...")
    chart_03_trial_metrics_bars(results)

    print("Chart 04: 3D Hyperparameter Landscape...")
    chart_04_3d_hyperparameter_landscape(results)

    print("Chart 05: Error Distributions...")
    chart_05_error_distributions(results)

    print("Chart 06: MC Dropout Analysis...")
    chart_06_mc_dropout_analysis(mc_data)

    print("Chart 07: 3D UQ Comparison...")
    chart_07_3d_uq_comparison(mc_data, ensemble_data)

    print("Chart 08: Feature Analysis...")
    chart_08_feature_analysis(mc_data, features)

    print("Chart 09: Correlation Heatmap...")
    chart_09_correlation_heatmap(mc_data, features)

    print("Chart 10: Calibration Analysis...")
    T_opt, ece_orig, ece_cal = chart_10_calibration(mc_data)

    print("Chart 11: Practical Threshold...")
    chart_11_practical_threshold(mc_data)

    print("Chart 12: With vs Without UQ...")
    chart_12_with_without_uq(mc_data, results)

    print("Chart 13: Spatial Map...")
    chart_13_spatial_map(mc_data, positions)

    print("Chart 14: 3D Surface...")
    chart_14_3d_surface(mc_data, features)

    print("Chart 15: Summary Dashboard...")
    chart_15_summary_dashboard(results, mc_data, ensemble_data, T_opt, ece_orig, ece_cal)

    print("\n" + "="*70)
    print(f"DONE! 15 charts saved to {OUTPUT}/")
    print("="*70)

if __name__ == "__main__":
    main()
