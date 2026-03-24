#!/usr/bin/env python3
"""
Heteroscedastic UQ Full Test Runner — Trial 9
===============================================
Implements the full uncertainty decomposition from Kendall & Gal (NeurIPS 2017, Section 3.2):

    Per node, for S MC Dropout forward passes each returning (mu_t, log_var_t):

        sigma^2_aleatoric  = mean_t( exp(log_var_t) )
        sigma^2_epistemic  = Var_t( mu_t )
        sigma^2_total      = sigma^2_epistemic + sigma^2_aleatoric

    sigma_aleatoric: irreducible data noise, predicted by the model
    sigma_epistemic: model uncertainty, reduced with more data
    sigma_total    : total predictive uncertainty

Usage:
    python code/evaluation/run_hetero_uq_t9.py --samples 30
    python code/evaluation/run_hetero_uq_t9.py --samples 30 --cpu
    python code/evaluation/run_hetero_uq_t9.py --samples 30 --max-graphs 10  # quick check
"""

import os
import sys
import argparse
import time
import json
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

# Add scripts to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from gnn.help_functions import mc_dropout_predict_hetero
from gnn.models.point_net_transf_gat import PointNetTransfGAT

# T9 trial folder name
T9_FOLDER_NAME = 'point_net_transf_gat_9th_trial_heteroscedastic'


def to_numpy(x):
    """Safely convert tensor or array to 1D float32 numpy."""
    if isinstance(x, torch.Tensor):
        return x.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.asarray(x).squeeze().astype(np.float32)


def load_model(model_folder: str, device: torch.device):
    """Load trained T9 heteroscedastic model."""
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=0.2,
        use_dropout=True,
        heteroscedastic=True,
        predict_mode_stats=False
    )

    model_path = os.path.join(model_folder, 'trained_model', 'model.pth')
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False),
        strict=True
    )
    return model.to(device)


def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def run_hetero_uq_full(
    num_samples: int = 30,
    force_cpu:   bool = False,
    resume:      bool = True,
    max_graphs:  int  = None
):
    """
    Run MC Dropout inference on T9 with heteroscedastic output decomposition.

    Args:
        num_samples: Number of MC Dropout forward passes S.
        force_cpu:   Force CPU even if GPU is available.
        resume:      Resume from per-graph checkpoints if they exist.
        max_graphs:  Limit to first N graphs (None = full test set).
    """
    # Device setup
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('HETEROSCEDASTIC UQ FULL TEST — MODEL T9')
    print('Kendall & Gal (NeurIPS 2017): aleatoric + epistemic decomposition')
    print('=' * 70)
    print(f'Device:     {device}')
    if device.type == 'cuda':
        print(f'GPU:        {torch.cuda.get_device_name(0)}')
        print(f'VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'MC samples: {num_samples}')
    print()

    # Paths
    model_folder   = os.path.join(REPO_ROOT, 'data', 'TR-C_Benchmarks', T9_FOLDER_NAME)
    output_folder  = os.path.join(model_folder, 'uq_results')
    checkpoint_dir = os.path.join(output_folder, f'checkpoints_hetero_mc{num_samples}')
    plots_dir      = os.path.join(model_folder, 'uq_plots')

    os.makedirs(output_folder,  exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir,      exist_ok=True)

    # Load test data
    test_dl_path = os.path.join(model_folder, 'data_created_during_training', 'test_dl.pt')
    test_dl      = torch.load(test_dl_path, weights_only=False)
    n_total      = len(test_dl)
    n_graphs     = min(n_total, max_graphs) if max_graphs else n_total

    print(f'Test set: {n_total} graphs total')
    if max_graphs:
        print(f'Running on first {n_graphs} graphs (--max-graphs={max_graphs})')

    # Load model
    print('Loading model...')
    model = load_model(model_folder, device)
    print('Model loaded successfully.')
    print()

    # Resume from checkpoints
    completed_graphs = set()
    if resume:
        for fn in os.listdir(checkpoint_dir):
            if fn.startswith('graph_') and fn.endswith('.npz'):
                idx = int(fn.split('_')[1].split('.')[0])
                if idx < n_graphs:
                    completed_graphs.add(idx)
        if completed_graphs:
            print(f'Found {len(completed_graphs)} completed graphs, resuming...')

    # Enable dropout for stochastic passes; freeze BatchNorm if present
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    start_time = time.time()
    processed_this_run = 0

    print(f'Started at: {time.strftime("%H:%M:%S")}')
    print()

    for i in tqdm(range(n_graphs), desc='Hetero MC Dropout'):
        if i in completed_graphs:
            continue

        graph = test_dl[i].to(device)

        # mc_dropout_predict_hetero returns:
        #   mean_mu, sigma_aleatoric, sigma_epistemic, sigma_total
        mean_mu, sigma_al, sigma_ep, sigma_tot = mc_dropout_predict_hetero(
            model, graph, num_samples=num_samples, device=device
        )

        actual = to_numpy(graph.y)

        np.savez_compressed(
            os.path.join(checkpoint_dir, f'graph_{i:04d}.npz'),
            mean_mu=mean_mu.astype(np.float32),
            sigma_aleatoric=sigma_al.astype(np.float32),
            sigma_epistemic=sigma_ep.astype(np.float32),
            sigma_total=sigma_tot.astype(np.float32),
            targets=actual.astype(np.float32)
        )

        completed_graphs.add(i)
        processed_this_run += 1

        if processed_this_run % 10 == 0:
            elapsed   = time.time() - start_time
            remaining = n_graphs - len(completed_graphs)
            eta       = remaining * (elapsed / processed_this_run)
            tqdm.write(f'  {len(completed_graphs)}/{n_graphs} | '
                       f'Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min')

    total_time = time.time() - start_time
    print()
    print(f'Total time: {total_time/60:.1f} min  ({total_time/3600:.2f} h)')
    print()

    # Aggregate all checkpoints
    print('Aggregating results...')
    files = sorted(glob.glob(os.path.join(checkpoint_dir, 'graph_*.npz')))
    if not files:
        raise RuntimeError('No checkpoint files found to aggregate.')

    all_mean_mu   = []
    all_sigma_al  = []
    all_sigma_ep  = []
    all_sigma_tot = []
    all_targets   = []

    for fp in files:
        d = np.load(fp)
        all_mean_mu.append(d['mean_mu'])
        all_sigma_al.append(d['sigma_aleatoric'])
        all_sigma_ep.append(d['sigma_epistemic'])
        all_sigma_tot.append(d['sigma_total'])
        all_targets.append(d['targets'])

    all_mean_mu   = np.concatenate(all_mean_mu)
    all_sigma_al  = np.concatenate(all_sigma_al)
    all_sigma_ep  = np.concatenate(all_sigma_ep)
    all_sigma_tot = np.concatenate(all_sigma_tot)
    all_targets   = np.concatenate(all_targets)
    n_completed   = len(files)

    print(f'Aggregated {n_completed} graphs, {len(all_mean_mu):,} nodes total')

    # Compute regression metrics on the mean prediction
    errors = np.abs(all_targets - all_mean_mu)

    ss_res = np.sum((all_targets - all_mean_mu) ** 2)
    ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
    r2     = 1 - (ss_res / ss_tot)
    mae    = np.mean(errors)
    rmse   = np.sqrt(np.mean((all_targets - all_mean_mu) ** 2))

    # Spearman correlations for each uncertainty type
    rho_al,  pval_al  = spearmanr(all_sigma_al,  errors)
    rho_ep,  pval_ep  = spearmanr(all_sigma_ep,  errors)
    rho_tot, pval_tot = spearmanr(all_sigma_tot, errors)

    # Print results
    print()
    print('=' * 70)
    print(f'HETERO UQ RESULTS — T9  ({n_completed} graphs, {num_samples} MC samples)')
    print('=' * 70)
    print(f'Nodes total:     {len(all_mean_mu):,}')
    print(f'R2:              {r2:.4f}')
    print(f'MAE:             {mae:.4f} veh/hr')
    print(f'RMSE:            {rmse:.4f} veh/hr')
    print()
    print('Uncertainty decomposition (Kendall & Gal 2017):')
    print(f'  sigma_aleatoric:  mean={all_sigma_al.mean():.4f}  std={all_sigma_al.std():.4f}')
    print(f'  sigma_epistemic:  mean={all_sigma_ep.mean():.4f}  std={all_sigma_ep.std():.4f}')
    print(f'  sigma_total:      mean={all_sigma_tot.mean():.4f}  std={all_sigma_tot.std():.4f}')
    print()
    print('Spearman rho (sigma vs |error|):')
    print(f'  aleatoric:  rho={rho_al:.4f}  p={pval_al:.2e}')
    print(f'  epistemic:  rho={rho_ep:.4f}  p={pval_ep:.2e}')
    print(f'  total:      rho={rho_tot:.4f}  p={pval_tot:.2e}')
    print('=' * 70)

    # Save arrays
    arrays_path = os.path.join(output_folder,
                               f'hetero_uq_mc{num_samples}_{n_completed}graphs.npz')
    np.savez_compressed(
        arrays_path,
        mean_mu=all_mean_mu.astype(np.float32),
        sigma_aleatoric=all_sigma_al.astype(np.float32),
        sigma_epistemic=all_sigma_ep.astype(np.float32),
        sigma_total=all_sigma_tot.astype(np.float32),
        targets=all_targets.astype(np.float32)
    )
    print(f'\nArrays saved to: {arrays_path}')

    # Save metrics JSON
    metrics = {
        'trial': 9,
        'mc_samples': num_samples,
        'n_graphs': n_completed,
        'n_nodes': int(len(all_mean_mu)),
        'r2':   float(r2),
        'mae':  float(mae),
        'rmse': float(rmse),
        'spearman_aleatoric':     float(rho_al),
        'spearman_aleatoric_p':   float(pval_al),
        'spearman_epistemic':     float(rho_ep),
        'spearman_epistemic_p':   float(pval_ep),
        'spearman_total':         float(rho_tot),
        'spearman_total_p':       float(pval_tot),
        'sigma_aleatoric_mean':   float(all_sigma_al.mean()),
        'sigma_aleatoric_std':    float(all_sigma_al.std()),
        'sigma_epistemic_mean':   float(all_sigma_ep.mean()),
        'sigma_epistemic_std':    float(all_sigma_ep.std()),
        'sigma_total_mean':       float(all_sigma_tot.mean()),
        'sigma_total_std':        float(all_sigma_tot.std()),
        'total_time_minutes':     float(total_time / 60),
    }

    metrics_path = os.path.join(
        output_folder,
        f'hetero_uq_metrics_t9_mc{num_samples}_{n_completed}graphs.json'
    )
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved to: {metrics_path}')

    # Generate plots
    setup_style()
    _generate_uq_plots(
        all_sigma_al, all_sigma_ep, all_sigma_tot,
        errors, all_targets, all_mean_mu,
        rho_al, rho_ep, rho_tot,
        n_completed, num_samples,
        plots_dir
    )

    return metrics


def _generate_uq_plots(
    sigma_al, sigma_ep, sigma_tot,
    errors, targets, preds,
    rho_al, rho_ep, rho_tot,
    n_graphs, num_samples,
    plots_dir
):
    """Generate UQ diagnostic plots for T9."""

    # ------------------------------------------------------------------ #
    # Plot 1: Uncertainty decomposition — histograms of all three sigmas  #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, sigma, label, color, rho in [
        (axes[0], sigma_al,  'sigma_aleatoric', '#C00000', rho_al),
        (axes[1], sigma_ep,  'sigma_epistemic', '#4472C4', rho_ep),
        (axes[2], sigma_tot, 'sigma_total',     '#7030A0', rho_tot),
    ]:
        ax.hist(sigma, bins=80, color=color, alpha=0.75, edgecolor='white', density=True)
        ax.axvline(sigma.mean(), color='black', linestyle='--', linewidth=1.5,
                   label=f'mean={sigma.mean():.3f}')
        ax.set_xlabel(f'{label} (veh/hr)')
        ax.set_ylabel('Density')
        ax.set_title(f'{label}\nSpearman rho={rho:.3f}')
        ax.legend(fontsize=8)

    fig.suptitle(f'T9 Uncertainty Decomposition — {n_graphs} graphs, S={num_samples}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fp = os.path.join(plots_dir, f'hetero_uq_01_decomposition_histograms_mc{num_samples}.png')
    fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved: hetero_uq_01_decomposition_histograms_mc{num_samples}.png')

    # ------------------------------------------------------------------ #
    # Plot 2: sigma vs |error| scatter (subsample) — all three types      #
    # ------------------------------------------------------------------ #
    n_sub = min(100000, len(errors))
    idx   = np.random.choice(len(errors), n_sub, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, sigma, label, color, rho in [
        (axes[0], sigma_al,  'sigma_aleatoric', '#C00000', rho_al),
        (axes[1], sigma_ep,  'sigma_epistemic', '#4472C4', rho_ep),
        (axes[2], sigma_tot, 'sigma_total',     '#7030A0', rho_tot),
    ]:
        ax.scatter(sigma[idx], errors[idx], alpha=0.05, s=1, c=color)
        ax.set_xlabel(label)
        ax.set_ylabel('|Error| (veh/hr)')
        ax.set_title(f'{label} vs |error|\nSpearman rho={rho:.3f}')
        ax.text(0.05, 0.92, f'rho={rho:.3f}', transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    fig.suptitle(f'T9 Uncertainty vs Error — Calibration Check (S={num_samples})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fp = os.path.join(plots_dir, f'hetero_uq_02_sigma_vs_error_mc{num_samples}.png')
    fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved: hetero_uq_02_sigma_vs_error_mc{num_samples}.png')

    # ------------------------------------------------------------------ #
    # Plot 3: Aleatoric fraction of total variance                        #
    # ------------------------------------------------------------------ #
    var_al  = sigma_al  ** 2
    var_ep  = sigma_ep  ** 2
    var_tot = sigma_tot ** 2

    # Clamp to avoid div-by-zero at nodes where total variance is tiny
    frac_al = np.where(var_tot > 1e-8, var_al / var_tot, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.hist(frac_al, bins=60, color='#ED7D31', alpha=0.75, edgecolor='white', density=True)
    ax1.axvline(frac_al.mean(), color='black', linestyle='--', linewidth=1.5,
                label=f'mean={frac_al.mean():.3f}')
    ax1.set_xlabel('Aleatoric fraction = sigma2_al / sigma2_total')
    ax1.set_ylabel('Density')
    ax1.set_title('Aleatoric Fraction of Total Variance\n'
                  '(1 = pure aleatoric, 0 = pure epistemic)')
    ax1.legend()

    ax2 = axes[1]
    ax2.scatter(sigma_al[idx], sigma_ep[idx], alpha=0.05, s=1, c='gray')
    ax2.set_xlabel('sigma_aleatoric')
    ax2.set_ylabel('sigma_epistemic')
    ax2.set_title('Aleatoric vs Epistemic Uncertainty\n(per node)')
    rho_al_ep, _ = spearmanr(sigma_al, sigma_ep)
    ax2.text(0.05, 0.92, f'Spearman rho={rho_al_ep:.3f}', transform=ax2.transAxes,
             fontsize=9, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    fig.suptitle(f'T9 Uncertainty Decomposition — Aleatoric vs Epistemic (S={num_samples})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fp = os.path.join(plots_dir, f'hetero_uq_03_aleatoric_fraction_mc{num_samples}.png')
    fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved: hetero_uq_03_aleatoric_fraction_mc{num_samples}.png')

    # ------------------------------------------------------------------ #
    # Plot 4: Summary bar chart — Spearman rho across uncertainty types   #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 5))

    labels  = ['aleatoric\n(Kendall & Gal)', 'epistemic\n(MC Dropout)', 'total']
    rhos    = [rho_al, rho_ep, rho_tot]
    colors  = ['#C00000', '#4472C4', '#7030A0']

    bars = ax.bar(labels, rhos, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{rho:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Spearman rho (sigma vs |error|)')
    ax.set_ylim(0, max(rhos) * 1.2 + 0.05)
    ax.set_title(f'T9 UQ Quality: Spearman Rank Correlation\n'
                 f'({n_graphs} graphs, S={num_samples} MC passes)')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    fp = os.path.join(plots_dir, f'hetero_uq_04_spearman_bar_mc{num_samples}.png')
    fig.savefig(fp, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved: hetero_uq_04_spearman_bar_mc{num_samples}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Heteroscedastic UQ full test for T9 (Kendall & Gal 2017)'
    )
    parser.add_argument('--samples',    type=int,  default=30,
                        help='Number of MC Dropout forward passes (default: 30)')
    parser.add_argument('--cpu',        action='store_true',
                        help='Force CPU even if GPU available')
    parser.add_argument('--no-resume',  action='store_true',
                        help='Start fresh, ignore existing checkpoints')
    parser.add_argument('--max-graphs', type=int,  default=None,
                        help='Limit to first N graphs (for quick testing)')

    args = parser.parse_args()

    run_hetero_uq_full(
        num_samples=args.samples,
        force_cpu=args.cpu,
        resume=not args.no_resume,
        max_graphs=args.max_graphs
    )
