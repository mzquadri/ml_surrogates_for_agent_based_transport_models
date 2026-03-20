#!/usr/bin/env python3
"""
Ensemble Uncertainty Quantification Experiments
================================================
Two main experiments:

Experiment A: Best Model (Trial 8) - MC Dropout vs Ensemble Variance
    - Run multiple MC Dropout inference runs with different seeds
    - Compare MC Dropout uncertainty with ensemble variance across runs
    
Experiment B: All Models Ensemble
    - Combine 5 models (2, 5, 6, 7, 8) 
    - Weighted average predictions based on model performance
    - Ensemble uncertainty from prediction variance across models

Usage:
    python scripts/evaluation/ensemble_uq_experiments.py --experiment A
    python scripts/evaluation/ensemble_uq_experiments.py --experiment B  
    python scripts/evaluation/ensemble_uq_experiments.py --experiment both
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add scripts to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from gnn.models.point_net_transf_gat import PointNetTransfGAT


def mc_dropout_predict_safe(model, data, num_samples: int = 50, device=None):
    """
    Safe MC Dropout inference that avoids torch.inference_mode issues.
    """
    was_training = model.training
    model = model.to(device)
    data = data.to(device)
    
    # Activate dropout
    model.train()
    
    # Freeze BatchNorm
    bn_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            bn_layers.append((m, m.training))
            m.eval()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)  # (num_samples, num_nodes, 1)
    mean_prediction = predictions.mean(axis=0).squeeze()
    uncertainty = predictions.std(axis=0).squeeze()
    
    model.train(was_training)
    for m, was_train in bn_layers:
        m.train(was_train)
    
    return mean_prediction, uncertainty

# ============================================================================
# Configuration
# ============================================================================

# Model folder mapping (only models with working dropout)
MODEL_FOLDERS = {
    2: 'point_net_transf_gat_2nd_try',
    5: 'point_net_transf_gat_5th_try',
    6: 'point_net_transf_gat_6th_trial_lower_lr',
    7: 'point_net_transf_gat_7th_trial_80_10_10_split',
    8: 'point_net_transf_gat_8th_trial_lower_dropout',
}

# Model performance weights (based on R² from deterministic evaluation)
# Higher R² = higher weight
MODEL_WEIGHTS = {
    2: 0.5117,  # R² from Model 2
    5: 0.4882,  # R² from Model 5
    6: 0.4779,  # R² from Model 6
    7: 0.5647,  # R² from Model 7
    8: 0.5957,  # R² from Model 8 (best)
}


def to_numpy(x):
    """Safely convert tensor or array to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.asarray(x).squeeze().astype(np.float32)


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


def compute_metrics(predictions, targets, uncertainties=None):
    """Compute regression metrics."""
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'n_samples': int(len(targets))
    }
    
    if uncertainties is not None:
        abs_errors = np.abs(predictions - targets)
        spearman_corr, spearman_pval = spearmanr(uncertainties, abs_errors)
        pearson_corr, pearson_pval = pearsonr(uncertainties, abs_errors)
        
        metrics.update({
            'uncertainty_mean': float(np.mean(uncertainties)),
            'uncertainty_std': float(np.std(uncertainties)),
            'uncertainty_min': float(np.min(uncertainties)),
            'uncertainty_max': float(np.max(uncertainties)),
            'spearman_corr': float(spearman_corr),
            'spearman_pval': float(spearman_pval),
            'pearson_corr': float(pearson_corr),
            'pearson_pval': float(pearson_pval),
        })
    
    return metrics


# ============================================================================
# EXPERIMENT A: Best Model Ensemble UQ (MC Dropout vs Ensemble Variance)
# ============================================================================

def run_experiment_a(device, num_ensemble_runs=5, mc_samples=30, max_graphs=10):
    """
    Experiment A: Compare MC Dropout uncertainty with Ensemble variance on Model 8.
    
    Approach:
    1. Run MC Dropout multiple times with different random seeds
    2. Compute:
       - MC Dropout uncertainty (std across MC samples within each run)
       - Ensemble variance (std across multiple MC Dropout runs)
    3. Compare both uncertainty estimates
    """
    print('=' * 80)
    print('EXPERIMENT A: Best Model (Trial 8) - MC Dropout vs Ensemble Variance')
    print('=' * 80)
    print(f'Device: {device}')
    print(f'Ensemble runs: {num_ensemble_runs}')
    print(f'MC samples per run: {mc_samples}')
    print(f'Max graphs: {max_graphs}')
    print()
    
    # Setup paths
    model_folder = os.path.join(REPO_ROOT, 'data/TR-C_Benchmarks', MODEL_FOLDERS[8])
    output_folder = os.path.join(model_folder, 'uq_results', 'ensemble_experiments')
    os.makedirs(output_folder, exist_ok=True)
    
    # Load test data
    test_dl_path = os.path.join(model_folder, 'data_created_during_training/test_dl.pt')
    test_set_dl = torch.load(test_dl_path, weights_only=False)
    n_graphs = min(len(test_set_dl), max_graphs)
    print(f'Using {n_graphs} test graphs')
    
    # Load model
    print('Loading Model 8...')
    model = load_model(model_folder, 8, device)
    print('Model loaded!')
    print()
    
    # Storage for ensemble predictions
    all_ensemble_predictions = []  # Shape: (num_runs, total_nodes)
    all_mc_uncertainties = []      # Shape: (num_runs, total_nodes)
    all_targets = None
    
    # Run multiple MC Dropout inference passes
    for run_idx in range(num_ensemble_runs):
        print(f'--- Ensemble Run {run_idx + 1}/{num_ensemble_runs} ---')
        
        # Set different random seed for each run
        torch.manual_seed(42 + run_idx * 100)
        np.random.seed(42 + run_idx * 100)
        
        run_predictions = []
        run_uncertainties = []
        run_targets = []
        
        for graph_idx in tqdm(range(n_graphs), desc=f'Run {run_idx + 1}'):
            test_data = test_set_dl[graph_idx].to(device)
            
            # Run MC Dropout
            mean_pred, uncertainty = mc_dropout_predict_safe(
                model, test_data, num_samples=mc_samples, device=device
            )
            
            run_predictions.append(to_numpy(mean_pred))
            run_uncertainties.append(to_numpy(uncertainty))
            
            if run_idx == 0:
                run_targets.append(to_numpy(test_data.y))
        
        all_ensemble_predictions.append(np.concatenate(run_predictions))
        all_mc_uncertainties.append(np.concatenate(run_uncertainties))
        
        if run_idx == 0:
            all_targets = np.concatenate(run_targets)
    
    # Convert to arrays
    ensemble_preds = np.array(all_ensemble_predictions)  # (num_runs, total_nodes)
    mc_uncertainties = np.array(all_mc_uncertainties)    # (num_runs, total_nodes)
    
    print()
    print('Computing uncertainty estimates...')
    
    # Compute uncertainty estimates
    # 1. Average MC Dropout uncertainty (avg across runs)
    avg_mc_uncertainty = np.mean(mc_uncertainties, axis=0)
    
    # 2. Ensemble variance (std across runs)
    ensemble_variance = np.std(ensemble_preds, axis=0)
    
    # 3. Combined uncertainty (sqrt(mc² + ensemble²))
    combined_uncertainty = np.sqrt(avg_mc_uncertainty**2 + ensemble_variance**2)
    
    # 4. Ensemble mean prediction
    ensemble_mean_pred = np.mean(ensemble_preds, axis=0)
    
    # Compute metrics for each uncertainty type
    results = {
        'config': {
            'model': 8,
            'num_ensemble_runs': num_ensemble_runs,
            'mc_samples': mc_samples,
            'num_graphs': n_graphs,
            'total_nodes': int(len(all_targets))
        },
        'mc_dropout_uncertainty': compute_metrics(ensemble_mean_pred, all_targets, avg_mc_uncertainty),
        'ensemble_variance': compute_metrics(ensemble_mean_pred, all_targets, ensemble_variance),
        'combined_uncertainty': compute_metrics(ensemble_mean_pred, all_targets, combined_uncertainty),
    }
    
    # Print results
    print()
    print('=' * 60)
    print('EXPERIMENT A RESULTS')
    print('=' * 60)
    print(f"Prediction Metrics (Ensemble Mean):")
    print(f"  R²:   {results['mc_dropout_uncertainty']['r2']:.4f}")
    print(f"  MAE:  {results['mc_dropout_uncertainty']['mae']:.4f}")
    print(f"  RMSE: {results['mc_dropout_uncertainty']['rmse']:.4f}")
    print()
    print('Uncertainty-Error Correlation (Spearman ρ):')
    print(f"  MC Dropout σ:       {results['mc_dropout_uncertainty']['spearman_corr']:.4f}")
    print(f"  Ensemble Variance:  {results['ensemble_variance']['spearman_corr']:.4f}")
    print(f"  Combined:           {results['combined_uncertainty']['spearman_corr']:.4f}")
    print()
    print('Uncertainty Statistics:')
    print(f"  MC Dropout σ:       mean={results['mc_dropout_uncertainty']['uncertainty_mean']:.4f}, std={results['mc_dropout_uncertainty']['uncertainty_std']:.4f}")
    print(f"  Ensemble Variance:  mean={results['ensemble_variance']['uncertainty_mean']:.4f}, std={results['ensemble_variance']['uncertainty_std']:.4f}")
    print(f"  Combined:           mean={results['combined_uncertainty']['uncertainty_mean']:.4f}, std={results['combined_uncertainty']['uncertainty_std']:.4f}")
    print()
    
    # Save results
    results_path = os.path.join(output_folder, 'experiment_a_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {results_path}')
    
    # Save arrays
    npz_path = os.path.join(output_folder, 'experiment_a_data.npz')
    np.savez(npz_path,
             ensemble_predictions=ensemble_preds,
             mc_uncertainties=mc_uncertainties,
             targets=all_targets,
             ensemble_mean=ensemble_mean_pred,
             avg_mc_uncertainty=avg_mc_uncertainty,
             ensemble_variance=ensemble_variance,
             combined_uncertainty=combined_uncertainty)
    print(f'Data saved to: {npz_path}')
    
    # Generate plots
    plot_experiment_a(output_folder, all_targets, ensemble_mean_pred, 
                      avg_mc_uncertainty, ensemble_variance, combined_uncertainty)
    
    return results


def plot_experiment_a(output_folder, targets, predictions, mc_unc, ens_var, combined_unc):
    """Generate plots for Experiment A."""
    plot_folder = os.path.join(output_folder, 'plots')
    os.makedirs(plot_folder, exist_ok=True)
    
    abs_errors = np.abs(predictions - targets)
    
    # 1. Comparison of uncertainty methods
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sample_idx = np.random.choice(len(targets), size=min(10000, len(targets)), replace=False)
    
    for ax, unc, title in zip(axes, 
                               [mc_unc, ens_var, combined_unc],
                               ['MC Dropout σ', 'Ensemble Variance', 'Combined']):
        ax.scatter(unc[sample_idx], abs_errors[sample_idx], alpha=0.3, s=1)
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('|Error|')
        ax.set_title(f'{title}\nρ = {spearmanr(unc, abs_errors)[0]:.4f}')
        ax.set_xlim(0, np.percentile(unc, 99))
        ax.set_ylim(0, np.percentile(abs_errors, 99))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'exp_a_uncertainty_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Uncertainty distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(mc_unc, bins=100, alpha=0.5, label='MC Dropout σ', density=True)
    ax.hist(ens_var, bins=100, alpha=0.5, label='Ensemble Variance', density=True)
    ax.hist(combined_unc, bins=100, alpha=0.5, label='Combined', density=True)
    
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Density')
    ax.set_title('Experiment A: Uncertainty Distributions')
    ax.legend()
    ax.set_xlim(0, np.percentile(combined_unc, 99))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'exp_a_uncertainty_distributions.png'), dpi=150)
    plt.close()
    
    # 3. MC vs Ensemble scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mc_unc[sample_idx], ens_var[sample_idx], alpha=0.3, s=1)
    ax.set_xlabel('MC Dropout σ')
    ax.set_ylabel('Ensemble Variance')
    ax.set_title(f'MC Dropout vs Ensemble Variance\nCorrelation: {pearsonr(mc_unc, ens_var)[0]:.4f}')
    max_val = max(np.percentile(mc_unc, 99), np.percentile(ens_var, 99))
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'exp_a_mc_vs_ensemble.png'), dpi=150)
    plt.close()
    
    print(f'Plots saved to: {plot_folder}')


# ============================================================================
# EXPERIMENT B: All Models Ensemble
# ============================================================================

def run_experiment_b(device, max_graphs=10, use_weighted=True):
    """
    Experiment B: Multi-model ensemble with weighted averaging.
    
    Approach:
    1. Load all 5 models (2, 5, 6, 7, 8)
    2. Run deterministic inference on shared test data
    3. Compute weighted average predictions (weights based on R²)
    4. Compute ensemble uncertainty (std across models)
    """
    print('=' * 80)
    print('EXPERIMENT B: All Models Ensemble (5 Models)')
    print('=' * 80)
    print(f'Device: {device}')
    print(f'Max graphs: {max_graphs}')
    print(f'Weighted averaging: {use_weighted}')
    print()
    
    # Models to include (all with working dropout for fair comparison)
    model_nums = [2, 5, 6, 7, 8]
    print(f'Models: {model_nums}')
    print(f'Weights (R²): {[f"M{m}: {MODEL_WEIGHTS[m]:.4f}" for m in model_nums]}')
    print()
    
    # Use Model 8's test data as reference (100 graphs)
    # Note: Different models have different test splits, so we need common graphs
    # For simplicity, we'll use model 8's test data and run all models on it
    ref_model_folder = os.path.join(REPO_ROOT, 'data/TR-C_Benchmarks', MODEL_FOLDERS[8])
    output_folder = os.path.join(ref_model_folder, 'uq_results', 'ensemble_experiments')
    os.makedirs(output_folder, exist_ok=True)
    
    # Load test data from Model 8
    test_dl_path = os.path.join(ref_model_folder, 'data_created_during_training/test_dl.pt')
    test_set_dl = torch.load(test_dl_path, weights_only=False)
    n_graphs = min(len(test_set_dl), max_graphs)
    print(f'Using {n_graphs} test graphs from Model 8 test set')
    print()
    
    # Load all models
    models = {}
    for model_num in model_nums:
        print(f'Loading Model {model_num}...')
        model_folder = os.path.join(REPO_ROOT, 'data/TR-C_Benchmarks', MODEL_FOLDERS[model_num])
        models[model_num] = load_model(model_folder, model_num, device)
        models[model_num].eval()  # Set to eval mode for deterministic inference
    print('All models loaded!')
    print()
    
    # Storage for predictions
    all_model_predictions = {m: [] for m in model_nums}
    all_targets = []
    
    # Run inference for each model
    print('Running ensemble inference...')
    for graph_idx in tqdm(range(n_graphs), desc='Graphs'):
        test_data = test_set_dl[graph_idx].to(device)
        
        # Get target
        target = to_numpy(test_data.y)
        all_targets.append(target)
        
        # Run each model
        with torch.no_grad():
            for model_num in model_nums:
                model = models[model_num]
                pred = model(test_data)
                if isinstance(pred, tuple):
                    pred = pred[0]
                all_model_predictions[model_num].append(to_numpy(pred))
    
    # Concatenate all predictions
    targets = np.concatenate(all_targets)
    model_preds = {m: np.concatenate(all_model_predictions[m]) for m in model_nums}
    
    # Stack predictions: (num_models, total_nodes)
    pred_stack = np.stack([model_preds[m] for m in model_nums], axis=0)
    
    print()
    print('Computing ensemble predictions...')
    
    # Compute ensemble predictions
    if use_weighted:
        # Normalize weights to sum to 1
        weights = np.array([MODEL_WEIGHTS[m] for m in model_nums])
        weights = weights / weights.sum()
        print(f'Normalized weights: {dict(zip(model_nums, weights))}')
        
        # Weighted average
        weighted_pred = np.average(pred_stack, axis=0, weights=weights)
        
        # Weighted variance: Var = sum(w_i * (x_i - mean)^2)
        weighted_var = np.average((pred_stack - weighted_pred)**2, axis=0, weights=weights)
        ensemble_uncertainty = np.sqrt(weighted_var)
    else:
        # Simple average
        weighted_pred = np.mean(pred_stack, axis=0)
        ensemble_uncertainty = np.std(pred_stack, axis=0)
    
    # Compute metrics for each model and ensemble
    results = {
        'config': {
            'models': model_nums,
            'weights': {m: float(MODEL_WEIGHTS[m]) for m in model_nums},
            'use_weighted': use_weighted,
            'num_graphs': n_graphs,
            'total_nodes': int(len(targets))
        },
        'individual_models': {},
        'ensemble': {}
    }
    
    # Individual model metrics
    print()
    print('Individual Model Performance:')
    print('-' * 50)
    for model_num in model_nums:
        metrics = compute_metrics(model_preds[model_num], targets)
        results['individual_models'][str(model_num)] = metrics
        print(f'  Model {model_num}: R²={metrics["r2"]:.4f}, MAE={metrics["mae"]:.4f}, RMSE={metrics["rmse"]:.4f}')
    
    # Ensemble metrics
    results['ensemble'] = compute_metrics(weighted_pred, targets, ensemble_uncertainty)
    
    print()
    print('=' * 60)
    print('EXPERIMENT B RESULTS')
    print('=' * 60)
    print(f"Ensemble Prediction Metrics:")
    print(f"  R²:   {results['ensemble']['r2']:.4f}")
    print(f"  MAE:  {results['ensemble']['mae']:.4f}")
    print(f"  RMSE: {results['ensemble']['rmse']:.4f}")
    print()
    print(f"Ensemble Uncertainty:")
    print(f"  Mean: {results['ensemble']['uncertainty_mean']:.4f}")
    print(f"  Std:  {results['ensemble']['uncertainty_std']:.4f}")
    print(f"  Spearman ρ: {results['ensemble']['spearman_corr']:.4f}")
    print()
    
    # Compare with best single model
    best_model = max(model_nums, key=lambda m: results['individual_models'][str(m)]['r2'])
    best_r2 = results['individual_models'][str(best_model)]['r2']
    ensemble_r2 = results['ensemble']['r2']
    print(f"Comparison with Best Single Model (Model {best_model}):")
    print(f"  Best Model R²:    {best_r2:.4f}")
    print(f"  Ensemble R²:      {ensemble_r2:.4f}")
    print(f"  Improvement:      {(ensemble_r2 - best_r2) * 100:.2f}%")
    print()
    
    # Save results
    results_path = os.path.join(output_folder, 'experiment_b_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {results_path}')
    
    # Save arrays
    npz_path = os.path.join(output_folder, 'experiment_b_data.npz')
    save_dict = {
        'targets': targets,
        'ensemble_prediction': weighted_pred,
        'ensemble_uncertainty': ensemble_uncertainty,
    }
    for m in model_nums:
        save_dict[f'model_{m}_predictions'] = model_preds[m]
    np.savez(npz_path, **save_dict)
    print(f'Data saved to: {npz_path}')
    
    # Generate plots
    plot_experiment_b(output_folder, targets, weighted_pred, ensemble_uncertainty, 
                      model_preds, model_nums, results)
    
    return results


def plot_experiment_b(output_folder, targets, ensemble_pred, ensemble_unc, 
                       model_preds, model_nums, results):
    """Generate plots for Experiment B."""
    plot_folder = os.path.join(output_folder, 'plots')
    os.makedirs(plot_folder, exist_ok=True)
    
    abs_errors = np.abs(ensemble_pred - targets)
    sample_idx = np.random.choice(len(targets), size=min(10000, len(targets)), replace=False)
    
    # 1. Ensemble prediction vs target
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.scatter(targets[sample_idx], ensemble_pred[sample_idx], alpha=0.3, s=1)
    max_val = max(np.percentile(targets, 99), np.percentile(ensemble_pred, 99))
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.set_xlabel('Target')
    ax.set_ylabel('Ensemble Prediction')
    ax.set_title(f'Ensemble Prediction vs Target\nR² = {results["ensemble"]["r2"]:.4f}')
    ax.legend()
    
    ax = axes[1]
    ax.scatter(ensemble_unc[sample_idx], abs_errors[sample_idx], alpha=0.3, s=1)
    ax.set_xlabel('Ensemble Uncertainty')
    ax.set_ylabel('|Error|')
    ax.set_title(f'Ensemble Uncertainty vs Error\nρ = {results["ensemble"]["spearman_corr"]:.4f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'exp_b_ensemble_performance.png'), dpi=150)
    plt.close()
    
    # 2. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models_labels = [f'M{m}' for m in model_nums] + ['Ensemble']
    r2_values = [results['individual_models'][str(m)]['r2'] for m in model_nums] + [results['ensemble']['r2']]
    colors = ['steelblue'] * len(model_nums) + ['darkgreen']
    
    bars = ax.bar(models_labels, r2_values, color=colors, edgecolor='black')
    ax.set_ylabel('R²')
    ax.set_title('Model Comparison: Individual vs Ensemble')
    ax.set_ylim(0, max(r2_values) * 1.1)
    
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'exp_b_model_comparison.png'), dpi=150)
    plt.close()
    
    # 3. Uncertainty distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ensemble_unc, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(ensemble_unc), color='red', linestyle='--', 
               label=f'Mean: {np.mean(ensemble_unc):.4f}')
    ax.set_xlabel('Ensemble Uncertainty (Std across models)')
    ax.set_ylabel('Count')
    ax.set_title('Experiment B: Ensemble Uncertainty Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'exp_b_uncertainty_distribution.png'), dpi=150)
    plt.close()
    
    # 4. Individual model predictions scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, model_num in enumerate(model_nums):
        ax = axes[idx]
        preds = model_preds[model_num]
        r2 = results['individual_models'][str(model_num)]['r2']
        ax.scatter(targets[sample_idx], preds[sample_idx], alpha=0.3, s=1)
        max_val = max(np.percentile(targets, 99), np.percentile(preds, 99))
        ax.plot([0, max_val], [0, max_val], 'r--')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.set_title(f'Model {model_num}\nR² = {r2:.4f}')
    
    # Ensemble in last subplot
    ax = axes[5]
    ax.scatter(targets[sample_idx], ensemble_pred[sample_idx], alpha=0.3, s=1, c='green')
    max_val = max(np.percentile(targets, 99), np.percentile(ensemble_pred, 99))
    ax.plot([0, max_val], [0, max_val], 'r--')
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title(f'Ensemble\nR² = {results["ensemble"]["r2"]:.4f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'exp_b_all_models_scatter.png'), dpi=150)
    plt.close()
    
    print(f'Plots saved to: {plot_folder}')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ensemble UQ Experiments')
    parser.add_argument('--experiment', type=str, choices=['A', 'B', 'both'], 
                        default='both', help='Which experiment to run')
    parser.add_argument('--max-graphs', type=int, default=10,
                        help='Maximum number of graphs to process (default: 10)')
    parser.add_argument('--ensemble-runs', type=int, default=5,
                        help='Number of ensemble runs for Exp A (default: 5)')
    parser.add_argument('--mc-samples', type=int, default=30,
                        help='MC samples per run (default: 30)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU available')
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('=' * 80)
    print('ENSEMBLE UNCERTAINTY QUANTIFICATION EXPERIMENTS')
    print('=' * 80)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()
    
    results = {}
    
    if args.experiment in ['A', 'both']:
        results['experiment_a'] = run_experiment_a(
            device=device,
            num_ensemble_runs=args.ensemble_runs,
            mc_samples=args.mc_samples,
            max_graphs=args.max_graphs
        )
        print()
    
    if args.experiment in ['B', 'both']:
        results['experiment_b'] = run_experiment_b(
            device=device,
            max_graphs=args.max_graphs,
            use_weighted=True
        )
        print()
    
    print('=' * 80)
    print('ALL EXPERIMENTS COMPLETED!')
    print('=' * 80)
    
    return results


if __name__ == '__main__':
    main()
