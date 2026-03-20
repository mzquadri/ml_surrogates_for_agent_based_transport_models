#!/usr/bin/env python3
"""
MC Dropout Full Test Runner with Checkpointing
===============================================
Runs MC Dropout on full test set with:
- GPU/CPU auto-detection
- Per-graph checkpointing (resume capability)
- Progress tracking and ETA
- Final aggregation and metrics

Usage:
    python scripts/evaluation/run_mc_dropout_full.py --model 8 --samples 30
"""

import os
import sys
import argparse
import time
import json
import glob
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

# Add scripts to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from gnn.help_functions import mc_dropout_predict
from gnn.models.point_net_transf_gat import PointNetTransfGAT


def to_numpy(x):
    """Safely convert tensor or array to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.asarray(x).squeeze().astype(np.float32)


# Model folder mapping
MODEL_FOLDERS = {
    2: 'point_net_transf_gat_2nd_try',
    3: 'point_net_transf_gat_3rd_trial_weighted_loss',
    4: 'point_net_transf_gat_4th_trial_weighted_loss',
    5: 'point_net_transf_gat_5th_try',
    6: 'point_net_transf_gat_6th_trial_lower_lr',
    7: 'point_net_transf_gat_7th_trial_80_10_10_split',
    8: 'point_net_transf_gat_8th_trial_lower_dropout',
}


def get_model_config(model_num: int) -> dict:
    """Get model configuration based on model number."""
    if model_num == 8:
        return {'dropout': 0.2}
    else:
        return {'dropout': 0.3}


def load_model(model_folder: str, model_num: int, device: torch.device):
    """Load trained model."""
    config = get_model_config(model_num)
    
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=config['dropout'],
        use_dropout=True,
        predict_mode_stats=False
    )
    
    model_path = os.path.join(model_folder, 'trained_model/model.pth')
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False),
        strict=False
    )
    
    model = model.to(device)
    return model


def run_mc_dropout_full(
    model_num: int,
    num_samples: int = 30,
    force_cpu: bool = False,
    resume: bool = True,
    max_graphs: int = None
):
    """
    Run MC Dropout on full test set with checkpointing.
    
    Args:
        model_num: Model number (2-8)
        num_samples: Number of MC samples per graph
        max_graphs: Limit number of graphs (None = all)
        force_cpu: Force CPU even if GPU available
        resume: Resume from checkpoint if exists
    """
    
    # Setup device
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('=' * 70)
    print(f'MC DROPOUT FULL TEST - MODEL {model_num}')
    print('=' * 70)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'MC Samples: {num_samples}')
    print()
    
    # Paths
    model_folder_name = MODEL_FOLDERS[model_num]
    model_folder = os.path.join(REPO_ROOT, 'data/TR-C_Benchmarks', model_folder_name)
    
    # Output folder for checkpoints (sample-specific to avoid mixing different runs)
    output_folder = os.path.join(model_folder, 'uq_results')
    checkpoint_folder = os.path.join(output_folder, f'checkpoints_mc{num_samples}')
    os.makedirs(checkpoint_folder, exist_ok=True)
    
    # Load test data
    test_dl_path = os.path.join(model_folder, 'data_created_during_training/test_dl.pt')
    test_set_dl = torch.load(test_dl_path, weights_only=False)
    n_graphs_total = len(test_set_dl)
    n_graphs = min(n_graphs_total, max_graphs) if max_graphs else n_graphs_total
    print(f'Test set: {n_graphs_total} graphs total')
    if max_graphs:
        print(f'Running on first {n_graphs} graphs (--max-graphs={max_graphs})')
    
    # Load model
    print('Loading model...')
    model = load_model(model_folder, model_num, device)
    print('Model loaded successfully!')
    print()
    
    # Check existing checkpoints
    completed_graphs = set()
    if resume:
        for f in os.listdir(checkpoint_folder):
            if f.startswith('graph_') and f.endswith('.npz'):
                idx = int(f.split('_')[1].split('.')[0])
                completed_graphs.add(idx)
        # Fix #2: Filter to current range only
        completed_graphs = {i for i in completed_graphs if i < n_graphs}
        if completed_graphs:
            print(f'Found {len(completed_graphs)} completed graphs, resuming...')
    
    # Run MC Dropout
    start_time = time.time()
    graphs_processed_this_session = 0  # Track graphs processed THIS run (for ETA)
    print(f'Started at: {time.strftime("%H:%M:%S")}')
    print()
    
    # Ensure dropout is ON and BatchNorm is frozen (if any)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    
    for i in tqdm(range(n_graphs), desc='MC Dropout'):
        # Skip if already done
        if i in completed_graphs:
            continue
        
        # Get data and move to device (Bug #2 fix)
        test_data = test_set_dl[i].to(device)
        
        # Run MC Dropout
        mean_pred, uncertainty = mc_dropout_predict(
            model, test_data, num_samples=num_samples, device=device
        )
        
        # Convert to numpy SAFELY using helper (works for both tensor and numpy)
        pred = to_numpy(mean_pred)
        unc = to_numpy(uncertainty)
        actual = to_numpy(test_data.y)
        
        # Save checkpoint with numpy arrays
        checkpoint_path = os.path.join(checkpoint_folder, f'graph_{i:04d}.npz')
        np.savez_compressed(
            checkpoint_path,
            predictions=pred,
            uncertainties=unc,
            targets=actual
        )
        
        completed_graphs.add(i)
        graphs_processed_this_session += 1
        
        # Progress update every 10 graphs (Bug #5 fix: correct ETA)
        if graphs_processed_this_session % 10 == 0:
            elapsed = time.time() - start_time
            remaining_graphs = n_graphs - len(completed_graphs)
            avg_time_per_graph = elapsed / graphs_processed_this_session
            eta = remaining_graphs * avg_time_per_graph
            tqdm.write(f'  Progress: {len(completed_graphs)}/{n_graphs} | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min')
    
    total_time = time.time() - start_time
    print()
    print(f'Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')
    print()
    
    # Aggregate results (Fix #3: use glob to handle partial runs)
    print('Aggregating results...')
    files = sorted(glob.glob(os.path.join(checkpoint_folder, 'graph_*.npz')))
    if not files:
        raise RuntimeError('No checkpoints found to aggregate.')
    
    all_preds = []
    all_uncertainties = []
    all_actuals = []
    
    for fp in files:
        data = np.load(fp)
        all_preds.append(data['predictions'])
        all_uncertainties.append(data['uncertainties'])
        all_actuals.append(data['targets'])
    
    all_preds = np.concatenate(all_preds)
    all_uncertainties = np.concatenate(all_uncertainties)
    all_actuals = np.concatenate(all_actuals)
    n_graphs_completed = len(files)
    print(f'Aggregated {n_graphs_completed} graphs')
    
    # Compute metrics
    ss_res = np.sum((all_actuals - all_preds) ** 2)
    ss_tot = np.sum((all_actuals - all_actuals.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    mae = np.mean(np.abs(all_actuals - all_preds))
    rmse = np.sqrt(np.mean((all_actuals - all_preds) ** 2))
    
    errors = np.abs(all_actuals - all_preds)
    spearman_corr, pval = spearmanr(all_uncertainties, errors)
    
    # Print results
    print()
    print('=' * 70)
    print(f'FULL TEST RESULTS - MODEL {model_num} ({n_graphs_completed} graphs, {num_samples} MC samples)')
    print('=' * 70)
    print(f'Total nodes:   {len(all_preds):,}')
    print(f'R-squared:     {r2:.4f}')
    print(f'MAE:           {mae:.4f}')
    print(f'RMSE:          {rmse:.4f}')
    print()
    print('Uncertainty Stats:')
    print(f'  Mean:        {all_uncertainties.mean():.4f}')
    print(f'  Std:         {all_uncertainties.std():.4f}')
    print(f'  Min:         {all_uncertainties.min():.4f}')
    print(f'  Max:         {all_uncertainties.max():.4f}')
    print()
    print('Uncertainty-Error Correlation:')
    print(f'  Spearman:    {spearman_corr:.4f}')
    print(f'  p-value:     {pval:.2e}')
    print('=' * 70)
    
    # Save final aggregated results (arrays only in npz)
    final_path = os.path.join(output_folder, f'mc_dropout_full_{n_graphs_completed}graphs_mc{num_samples}.npz')
    np.savez_compressed(
        final_path,
        predictions=all_preds.astype(np.float32),
        uncertainties=all_uncertainties.astype(np.float32),
        targets=all_actuals.astype(np.float32)
    )
    print(f'\nArrays saved to: {final_path}')
    
    # Save metrics separately as JSON (Bug #4 fix: avoid pickle in npz)
    metrics_dict = {
        'model': model_num,
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'spearman': float(spearman_corr),
        'spearman_pval': float(pval),
        'unc_mean': float(all_uncertainties.mean()),
        'unc_std': float(all_uncertainties.std()),
        'unc_min': float(all_uncertainties.min()),
        'unc_max': float(all_uncertainties.max()),
        'n_graphs': n_graphs_completed,
        'n_nodes': len(all_preds),
        'num_samples': num_samples,
        'total_time_minutes': total_time / 60
    }
    metrics_path = os.path.join(output_folder, f'mc_dropout_full_metrics_model{model_num}_mc{num_samples}_{n_graphs_completed}graphs.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f'Metrics saved to: {metrics_path}')
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'spearman': spearman_corr,
        'n_graphs': n_graphs_completed,
        'n_nodes': len(all_preds)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MC Dropout on full test set')
    parser.add_argument('--model', type=int, default=8, choices=[2, 3, 4, 5, 6, 7, 8],
                        help='Model number (2-8)')
    parser.add_argument('--samples', type=int, default=30,
                        help='Number of MC samples')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU available')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore checkpoints')
    parser.add_argument('--max-graphs', type=int, default=None,
                        help='Limit number of graphs (for quick demo)')
    
    args = parser.parse_args()
    
    run_mc_dropout_full(
        model_num=args.model,
        num_samples=args.samples,
        force_cpu=args.cpu,
        resume=not args.no_resume,
        max_graphs=args.max_graphs
    )
