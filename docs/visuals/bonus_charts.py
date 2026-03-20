"""
BONUS CHARTS: Ensemble Deep-Dive + Additional 3D/Polished Visuals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

BASE = 'data/TR-C_Benchmarks'
OUTPUT = 'docs/visuals/verification'
os.makedirs(OUTPUT, exist_ok=True)

PALETTE = {
    'royal_blue': '#4A90D9', 'coral': '#FF6B6B', 'emerald': '#2ECC71',
    'gold': '#F39C12', 'purple': '#9B59B6', 'teal': '#1ABC9C',
    'rose': '#E74C3C', 'dark': '#2C3E50',
}

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFBFC',
    'axes.grid': True, 'grid.alpha': 0.15, 'font.size': 11,
    'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.labelsize': 13,
})


# ============================================================
# CHART 16: Multi-Model Ensemble Deep Dive
# ============================================================

def chart_16_ensemble_multi_model():
    """Deep dive into multi-model ensemble (Experiment B)"""
    path = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_b_data.npz')
    d = np.load(path, allow_pickle=True)

    targets = d['targets'].flatten()
    ens_pred = d['ensemble_prediction'].flatten()
    ens_unc = d['ensemble_uncertainty'].flatten()

    # Individual model predictions
    model_keys = ['model_2_predictions', 'model_5_predictions',
                  'model_6_predictions', 'model_7_predictions', 'model_8_predictions']
    model_names = ['Trial 2', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8']
    model_colors = ['#FF8E53', '#0ABDE3', '#9B59B6', '#1ABC9C', '#2ECC71']

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # Individual model performance
    model_r2s = []
    model_maes = []
    for mk, mn in zip(model_keys, model_names):
        preds = d[mk].flatten()
        r2 = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        model_r2s.append(r2)
        model_maes.append(mae)

    # Ensemble performance
    ens_r2 = r2_score(targets, ens_pred)
    ens_mae = mean_absolute_error(targets, ens_pred)

    # 1. R² comparison
    ax = axes[0, 0]
    names = model_names + ['Ensemble']
    r2s = model_r2s + [ens_r2]
    colors = model_colors + ['#FFD700']
    bars = ax.bar(names, r2s, color=colors, edgecolor='white', linewidth=2)
    bars[-1].set_edgecolor('#FFD700')
    bars[-1].set_linewidth(4)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
               f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('R² Score: Individual vs Ensemble', fontsize=14, fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')

    # 2. MAE comparison
    ax = axes[0, 1]
    maes = model_maes + [ens_mae]
    bars = ax.bar(names, maes, color=colors, edgecolor='white', linewidth=2)
    bars[-1].set_edgecolor('#FFD700')
    bars[-1].set_linewidth(4)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
               f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('MAE: Individual vs Ensemble', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE', fontweight='bold')

    # 3. Model agreement visualization
    ax = axes[0, 2]
    rng = np.random.RandomState(42)
    idx = rng.choice(len(targets), 50000, replace=False)

    model_preds_all = np.array([d[mk].flatten()[idx] for mk in model_keys])
    std_per_point = model_preds_all.std(axis=0)
    mean_err = np.abs(ens_pred[idx] - targets[idx])

    hb = ax.hexbin(std_per_point, mean_err, gridsize=60, cmap='viridis',
                   mincnt=1, linewidths=0.1, edgecolors='none')
    rho, _ = spearmanr(std_per_point, mean_err)
    ax.set_xlabel('Model Disagreement (std)', fontsize=12)
    ax.set_ylabel('Ensemble Error', fontsize=12)
    ax.set_title(f'Disagreement vs Error (ρ = {rho:.3f})', fontsize=14, fontweight='bold')
    plt.colorbar(hb, ax=ax, shrink=0.8)

    # 4. Prediction spread across models
    ax = axes[1, 0]
    sample_idx = rng.choice(len(targets), 200, replace=False)
    sample_idx = np.sort(sample_idx)

    for mi, (mk, mn, mc) in enumerate(zip(model_keys, model_names, model_colors)):
        preds = d[mk].flatten()[sample_idx]
        ax.scatter(range(200), preds, c=mc, s=10, alpha=0.5, label=mn)

    ax.scatter(range(200), targets[sample_idx], c='black', s=30, marker='x', label='Target', zorder=10)
    ax.scatter(range(200), ens_pred[sample_idx], c='#FFD700', s=20, marker='D', label='Ensemble', zorder=10)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('ΔVolume', fontsize=12)
    ax.set_title('Model Predictions Spread (200 Random Samples)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=4)

    # 5. Uncertainty distribution
    ax = axes[1, 1]
    ax.hist(ens_unc, bins=100, color='#9B59B6', alpha=0.7, edgecolor='white')
    ax.axvline(np.median(ens_unc), color='#E74C3C', linewidth=2, linestyle='--',
               label=f'Median: {np.median(ens_unc):.3f}')
    ax.axvline(np.mean(ens_unc), color='#4A90D9', linewidth=2, linestyle='--',
               label=f'Mean: {np.mean(ens_unc):.3f}')
    ax.set_xlabel('Ensemble Uncertainty', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Multi-Model Uncertainty Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    # 6. Error reduction with ensemble filtering
    ax = axes[1, 2]
    errors = np.abs(ens_pred - targets)
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    overall_mae = np.mean(errors)
    reductions = []
    for p in percentiles:
        thresh = np.percentile(ens_unc, p)
        mask = ens_unc <= thresh
        mae_keep = np.mean(errors[mask])
        reductions.append((overall_mae - mae_keep) / overall_mae * 100)

    ax.fill_between(percentiles, reductions, alpha=0.3, color='#9B59B6')
    ax.plot(percentiles, reductions, 'o-', color='#9B59B6', linewidth=3, markersize=10)
    for p, r in zip(percentiles, reductions):
        ax.text(p, r + 0.3, f'{r:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Confidence Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Reduction %', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble Error Reduction', fontsize=14, fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#FAFBFC')

    plt.suptitle('Multi-Model Ensemble: Trials 2, 5, 6, 7, 8', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '16_ensemble_deep_dive.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 16_ensemble_deep_dive.png")


# ============================================================
# CHART 17: Training Run Ensemble (Experiment A)
# ============================================================

def chart_17_ensemble_training_runs():
    """Experiment A: 5 training runs of the same model"""
    path = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_a_data.npz')
    d = np.load(path, allow_pickle=True)

    targets = d['targets'].flatten()
    ens_mean = d['ensemble_mean'].flatten()
    ens_var = d['ensemble_variance'].flatten()
    mc_unc = d['avg_mc_uncertainty'].flatten()
    combined = d['combined_uncertainty'].flatten()
    errors = np.abs(ens_mean - targets)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # 1. Ensemble predictions vs targets
    ax = axes[0, 0]
    rng = np.random.RandomState(42)
    idx = rng.choice(len(targets), 100000, replace=False)
    hb = ax.hexbin(targets[idx], ens_mean[idx], gridsize=80, cmap='YlOrRd',
                   mincnt=1, linewidths=0.1, edgecolors='none')
    lims = [targets[idx].min(), targets[idx].max()]
    ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.7)
    r2 = r2_score(targets, ens_mean)
    ax.set_title(f'5-Run Ensemble: R² = {r2:.4f}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    plt.colorbar(hb, ax=ax, shrink=0.8)

    # 2. Three uncertainty types comparison
    ax = axes[0, 1]
    types = ['Ensemble\nVariance', 'MC\nUncertainty', 'Combined']
    rhos = [
        spearmanr(ens_var, errors)[0],
        spearmanr(mc_unc, errors)[0],
        spearmanr(combined, errors)[0],
    ]
    colors = ['#FF6B6B', '#4A90D9', '#F39C12']
    bars = ax.bar(types, rhos, color=colors, edgecolor='white', linewidth=2, width=0.5)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
               f'{val:.4f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.set_title('UQ Type Comparison', fontsize=14, fontweight='bold')

    # 3. Epistemic vs Aleatoric
    ax = axes[0, 2]
    hb = ax.hexbin(ens_var[idx], mc_unc[idx], gridsize=80, cmap='magma_r',
                   mincnt=1, linewidths=0.1, edgecolors='none')
    ax.set_xlabel('Ensemble Variance (Epistemic)', fontsize=12)
    ax.set_ylabel('MC Uncertainty (Aleatoric)', fontsize=12)
    ax.set_title('Epistemic vs Aleatoric Uncertainty', fontsize=14, fontweight='bold')
    plt.colorbar(hb, ax=ax, shrink=0.8)

    # 4. Individual model predictions
    ax = axes[1, 0]
    ens_preds = d['ensemble_predictions']  # (5, N)
    model_r2s = []
    for i in range(5):
        preds_i = ens_preds[i].flatten()
        r2_i = r2_score(targets, preds_i)
        model_r2s.append(r2_i)

    ens_r2 = r2_score(targets, ens_mean)
    run_colors = ['#FF6B6B', '#FF8E53', '#FECA57', '#48DBFB', '#0ABDE3']
    names = [f'Run {i+1}' for i in range(5)] + ['Ensemble']
    all_r2s = model_r2s + [ens_r2]
    all_colors = run_colors + ['#2ECC71']

    bars = ax.bar(names, all_r2s, color=all_colors, edgecolor='white', linewidth=2)
    bars[-1].set_edgecolor('#2ECC71')
    bars[-1].set_linewidth(4)
    for bar, val in zip(bars, all_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
               f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('5 Training Runs + Ensemble R²', fontsize=14, fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')

    # 5. Combined uncertainty vs error
    ax = axes[1, 1]
    hb = ax.hexbin(combined[idx], errors[idx], gridsize=80, cmap='inferno_r',
                   mincnt=1, linewidths=0.1, edgecolors='none')

    # Trend line
    n_bins = 30
    bin_edges = np.percentile(combined[idx], np.linspace(0, 100, n_bins + 1))
    bc, bm = [], []
    for j in range(len(bin_edges) - 1):
        mask = (combined[idx] >= bin_edges[j]) & (combined[idx] < bin_edges[j+1])
        if mask.sum() > 10:
            bc.append(combined[idx][mask].mean())
            bm.append(errors[idx][mask].mean())
    ax.plot(bc, bm, 'c-', linewidth=3, label='Mean Trend')

    rho_c = spearmanr(combined, errors)[0]
    ax.set_xlabel('Combined Uncertainty', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(f'Combined UQ vs Error (ρ = {rho_c:.4f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.colorbar(hb, ax=ax, shrink=0.8)

    # 6. Error reduction
    ax = axes[1, 2]
    overall_mae = np.mean(errors)
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    methods = [
        ('Ensemble Var', ens_var, '#FF6B6B'),
        ('MC Unc', mc_unc, '#4A90D9'),
        ('Combined', combined, '#F39C12'),
    ]
    for mname, unc_arr, color in methods:
        reductions = []
        for p in percentiles:
            thresh = np.percentile(unc_arr, p)
            mask = unc_arr <= thresh
            mae_keep = np.mean(errors[mask])
            reductions.append((overall_mae - mae_keep) / overall_mae * 100)
        ax.plot(percentiles, reductions, 'o-', color=color, linewidth=2.5, markersize=8, label=mname)

    ax.set_xlabel('Confidence Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Reduction %', fontsize=12, fontweight='bold')
    ax.set_title('Error Reduction by UQ Method', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    for ax in axes.flat:
        ax.set_facecolor('#FAFBFC')

    plt.suptitle('Training-Run Ensemble (Experiment A): 5 Runs of Trial 8', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '17_ensemble_training_runs.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 17_ensemble_training_runs.png")


# ============================================================
# CHART 18: Grand Comparison - MC vs Ensemble vs Multi-Model
# ============================================================

def chart_18_grand_uq_comparison():
    """Compare all UQ approaches head-to-head"""
    # Load MC Dropout from Trial 8
    mc_path = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz')
    mc = np.load(mc_path)

    # Load Ensemble A
    ea_path = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_a_data.npz')
    ea = np.load(ea_path, allow_pickle=True)

    # Load Ensemble B
    eb_path = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_b_data.npz')
    eb = np.load(eb_path, allow_pickle=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # ---- Method 1: MC Dropout ----
    mc_errors = np.abs(mc['predictions'].flatten() - mc['targets'].flatten())
    mc_unc = mc['uncertainties'].flatten()

    # ---- Method 2: Ensemble Variance (Exp A) ----
    ea_errors = np.abs(ea['ensemble_mean'].flatten() - ea['targets'].flatten())
    ea_var = ea['ensemble_variance'].flatten()
    ea_combined = ea['combined_uncertainty'].flatten()

    # ---- Method 3: Multi-Model (Exp B) ----
    eb_errors = np.abs(eb['ensemble_prediction'].flatten() - eb['targets'].flatten())
    eb_unc = eb['ensemble_uncertainty'].flatten()

    # 1. Spearman comparison bar
    ax = axes[0, 0]
    methods = [
        'MC Dropout\n(T8, 30 samples)',
        'Ens. Variance\n(5 runs)',
        'Ens. Combined\n(MC+Var)',
        'MC Avg.\n(in Ensemble)',
        'Multi-Model\n(T2,5,6,7,8)'
    ]
    rhos = [
        spearmanr(mc_unc, mc_errors)[0],
        spearmanr(ea_var, ea_errors)[0],
        spearmanr(ea_combined, ea_errors)[0],
        spearmanr(ea['avg_mc_uncertainty'].flatten(), ea_errors)[0],
        spearmanr(eb_unc, eb_errors)[0],
    ]
    colors = ['#4A90D9', '#FF6B6B', '#F39C12', '#1ABC9C', '#9B59B6']

    bars = ax.bar(range(len(methods)), rhos, color=colors, edgecolor='white', linewidth=2, width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.set_title('All UQ Methods: Spearman ρ', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
               f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    # Highlight best
    best_idx = np.argmax(rhos)
    bars[best_idx].set_edgecolor('#2ECC71')
    bars[best_idx].set_linewidth(4)

    # 2. Error reduction curves for all methods
    ax = axes[0, 1]
    percentiles = np.arange(50, 96, 2)

    method_data = [
        ('MC Dropout', mc_unc, mc_errors, '#4A90D9'),
        ('Ens. Variance', ea_var, ea_errors, '#FF6B6B'),
        ('Ens. Combined', ea_combined, ea_errors, '#F39C12'),
        ('Multi-Model', eb_unc, eb_errors, '#9B59B6'),
    ]

    for mname, unc, err, color in method_data:
        overall = np.mean(err)
        reductions = []
        for p in percentiles:
            thresh = np.percentile(unc, p)
            mask = unc <= thresh
            if mask.sum() > 0:
                reductions.append((overall - np.mean(err[mask])) / overall * 100)
            else:
                reductions.append(0)
        ax.plot(percentiles, reductions, '-', color=color, linewidth=2.5, label=mname, alpha=0.85)

    ax.set_xlabel('Confidence Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Reduction %', fontsize=12, fontweight='bold')
    ax.set_title('Error Reduction by Filtering', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # 3. 3D comparison
    ax = axes[1, 0]
    ax = fig.add_subplot(223, projection='3d')

    method_short = ['MC Drop', 'Ens Var', 'Ens Comb', 'MultiModel']
    rho_3d = rhos[:2] + [rhos[2]] + [rhos[4]]
    method_r2 = [
        r2_score(mc['targets'].flatten(), mc['predictions'].flatten()),
        r2_score(ea['targets'].flatten(), ea['ensemble_mean'].flatten()),
        r2_score(ea['targets'].flatten(), ea['ensemble_mean'].flatten()),
        r2_score(eb['targets'].flatten(), eb['ensemble_prediction'].flatten()),
    ]
    method_mae = [
        mean_absolute_error(mc['targets'].flatten(), mc['predictions'].flatten()),
        mean_absolute_error(ea['targets'].flatten(), ea['ensemble_mean'].flatten()),
        mean_absolute_error(ea['targets'].flatten(), ea['ensemble_mean'].flatten()),
        mean_absolute_error(eb['targets'].flatten(), eb['ensemble_prediction'].flatten()),
    ]
    colors_3d = ['#4A90D9', '#FF6B6B', '#F39C12', '#9B59B6']

    for i, (m, rho, r2, mae, c) in enumerate(zip(method_short, rho_3d, method_r2, method_mae, colors_3d)):
        ax.scatter(rho, r2, mae, c=c, s=300, alpha=0.85, edgecolors='white', linewidth=2, label=m)
        ax.text(rho, r2, mae + 0.1, m, fontsize=9, ha='center', fontweight='bold')

    ax.set_xlabel('Spearman ρ', fontsize=10)
    ax.set_ylabel('R²', fontsize=10)
    ax.set_zlabel('MAE', fontsize=10)
    ax.set_title('3D: ρ × R² × MAE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.view_init(elev=25, azim=-45)

    # 4. Winner summary
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('UQ Methods Summary', fontsize=16, fontweight='bold', color='#2C3E50')

    summary = [
        ('MC Dropout (30 samples)', f'ρ = {rhos[0]:.4f}', 'Best correlation', '#4A90D9',
         'Single model, fast inference, best error prediction'),
        ('Ensemble Variance (5 runs)', f'ρ = {rhos[1]:.4f}', 'Captures epistemic', '#FF6B6B',
         '5x training cost, lower ρ alone'),
        ('Combined (MC+Ens)', f'ρ = {rhos[2]:.4f}', 'Epistemic+Aleatoric', '#F39C12',
         'Comprehensive but costly'),
        ('Multi-Model Ensemble', f'ρ = {rhos[4]:.4f}', 'Cross-architecture', '#9B59B6',
         'Uses 5 different trials'),
    ]

    for i, (name, rho_str, note, color, desc) in enumerate(summary):
        y = 0.88 - i * 0.22
        ax.text(0.02, y, '●', fontsize=22, color=color, transform=ax.transAxes)
        ax.text(0.06, y, f'{name}: {rho_str}', fontsize=12, fontweight='bold',
               transform=ax.transAxes, color='#2C3E50')
        ax.text(0.06, y - 0.06, desc, fontsize=10, transform=ax.transAxes, color='#666')

    ax.text(0.02, 0.02, 'Winner: MC Dropout (best rho, lowest cost)', fontsize=14,
           fontweight='bold', color='#2ECC71', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '18_grand_uq_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 18_grand_uq_comparison.png")


# ============================================================
# CHART 19: Prediction Quality by Graph Scenario
# ============================================================

def chart_19_per_graph_analysis():
    """Analyze performance per test graph"""
    # Trial 8: 100 graphs × 31635 nodes
    path = os.path.join(BASE, 'point_net_transf_gat_8th_trial_lower_dropout/test_predictions.npz')
    d = np.load(path)
    preds = d['predictions'].flatten()
    targets = d['targets'].flatten()

    n_nodes = 31635
    n_graphs = len(preds) // n_nodes

    graph_r2s = []
    graph_maes = []
    graph_mean_vol = []

    for g in range(n_graphs):
        s, e = g * n_nodes, (g + 1) * n_nodes
        p, t = preds[s:e], targets[s:e]
        graph_r2s.append(r2_score(t, p))
        graph_maes.append(mean_absolute_error(t, p))
        graph_mean_vol.append(np.mean(np.abs(t)))

    graph_r2s = np.array(graph_r2s)
    graph_maes = np.array(graph_maes)
    graph_mean_vol = np.array(graph_mean_vol)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. R² per graph
    ax = axes[0, 0]
    colors = cm.viridis(np.linspace(0, 1, n_graphs))
    bars = ax.bar(range(n_graphs), graph_r2s, color=colors, edgecolor='none', width=0.9)
    ax.axhline(np.mean(graph_r2s), color='#E74C3C', linewidth=2, linestyle='--',
               label=f'Mean: {np.mean(graph_r2s):.4f}')
    ax.set_xlabel('Test Graph Index', fontsize=12)
    ax.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax.set_title(f'R² per Test Scenario (Trial 8, {n_graphs} graphs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    # 2. MAE per graph
    ax = axes[0, 1]
    ax.bar(range(n_graphs), graph_maes, color=cm.plasma(np.linspace(0, 1, n_graphs)),
           edgecolor='none', width=0.9)
    ax.axhline(np.mean(graph_maes), color='#E74C3C', linewidth=2, linestyle='--',
               label=f'Mean: {np.mean(graph_maes):.3f}')
    ax.set_xlabel('Test Graph Index', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title(f'MAE per Test Scenario', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    # 3. R² vs Mean Volume Change
    ax = axes[1, 0]
    sc = ax.scatter(graph_mean_vol, graph_r2s, c=graph_maes, cmap='RdYlGn',
                    s=80, edgecolors='white', linewidth=1, alpha=0.85)
    ax.set_xlabel('Mean |ΔVolume| in Graph', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² per Graph', fontsize=12, fontweight='bold')
    ax.set_title('R² vs Traffic Disruption Severity', fontsize=14, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='MAE')

    rho_graph, _ = spearmanr(graph_mean_vol, graph_r2s)
    ax.text(0.05, 0.05, f'ρ = {rho_graph:.3f}', transform=ax.transAxes,
           fontsize=13, fontweight='bold', color='#4A90D9',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 4. Distribution of per-graph R²
    ax = axes[1, 1]
    ax.hist(graph_r2s, bins=30, color='#4A90D9', edgecolor='white', alpha=0.8, linewidth=2)
    ax.axvline(np.median(graph_r2s), color='#E74C3C', linewidth=2, linestyle='--',
               label=f'Median: {np.median(graph_r2s):.4f}')
    ax.axvline(np.mean(graph_r2s), color='#2ECC71', linewidth=2, linestyle='--',
               label=f'Mean: {np.mean(graph_r2s):.4f}')
    ax.set_xlabel('R²', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Per-Graph R²', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    for ax in axes.flat:
        ax.set_facecolor('#FAFBFC')

    plt.suptitle(f'Per-Graph Analysis: {n_graphs} Test Scenarios', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '19_per_graph_analysis.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 19_per_graph_analysis.png")


# ============================================================
# CHART 20: 3D Radar/Spider Comparison
# ============================================================

def chart_20_radar_comparison():
    """Gorgeous radar/spider chart comparing all trials"""
    from matplotlib.patches import FancyArrowPatch

    trials_data = {}
    for trial_num, folder in {
        1: 'pointnet_transf_gat_1st_bs32_5feat_seed42',
        2: 'point_net_transf_gat_2nd_try',
        5: 'point_net_transf_gat_5th_try',
        7: 'point_net_transf_gat_7th_trial_80_10_10_split',
        8: 'point_net_transf_gat_8th_trial_lower_dropout',
    }.items():
        path = os.path.join(BASE, folder, 'test_predictions.npz')
        d = np.load(path)
        preds = d['predictions'].flatten()
        targets = d['targets'].flatten()
        errors = np.abs(preds - targets)

        trials_data[trial_num] = {
            'r2': r2_score(targets, preds),
            'mae': mean_absolute_error(targets, preds),
            'rmse': np.sqrt(mean_squared_error(targets, preds)),
            'p90_err': np.percentile(errors, 90),
            'pct_under_5': np.mean(errors < 5) * 100,
        }

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    categories = ['R² (×100)', 'Low MAE\n(10-MAE)', 'Low RMSE\n(15-RMSE)', '<5 Error %', 'Low P90\n(20-P90)']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    trial_colors = {1: '#FF6B6B', 2: '#FF8E53', 5: '#0ABDE3', 7: '#1ABC9C', 8: '#2ECC71'}

    for trial_num, data in sorted(trials_data.items()):
        values = [
            data['r2'] * 100,
            10 - data['mae'],
            15 - data['rmse'],
            data['pct_under_5'],
            20 - data['p90_err'],
        ]
        # Normalize to 0-100 for visualization
        values_norm = [max(0, v) for v in values]
        values_norm += values_norm[:1]

        ax.fill(angles, values_norm, alpha=0.15, color=trial_colors[trial_num])
        ax.plot(angles, values_norm, 'o-', linewidth=2.5, markersize=8,
               color=trial_colors[trial_num], label=f'Trial {trial_num}')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_title('Multi-Axis Performance Comparison', fontsize=18, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, '20_radar_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 20_radar_comparison.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("BONUS CHARTS")
    print("="*60)

    print("\nChart 16: Ensemble Multi-Model Deep Dive...")
    chart_16_ensemble_multi_model()

    print("Chart 17: Training-Run Ensemble...")
    chart_17_ensemble_training_runs()

    print("Chart 18: Grand UQ Comparison...")
    chart_18_grand_uq_comparison()

    print("Chart 19: Per-Graph Analysis...")
    chart_19_per_graph_analysis()

    print("Chart 20: Radar Comparison...")
    chart_20_radar_comparison()

    print(f"\n{'='*60}")
    print(f"5 bonus charts saved to {OUTPUT}/")
    print(f"{'='*60}")
