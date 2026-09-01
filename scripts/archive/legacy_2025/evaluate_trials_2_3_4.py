"""
COMPLETE EVALUATION SCRIPT - ALL TRIALS (2-8)
This script evaluates ALL trained models on their test sets
Generates comprehensive metrics for visualization script update
"""

import torch
import numpy as np
from tqdm import tqdm
import json
import sys
import os
from scipy.stats import pearsonr, spearmanr

# Setup paths
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
sys.path.insert(0, os.path.join(BASE_PATH, "scripts"))

from torch_geometric.loader import DataLoader
from gnn.models.point_net_transf_gat import PointNetTransfGAT
from gnn.help_functions import GNN_Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Complete trial configurations
TRIALS = {
    'Trial 2': {
        'name': 'Trial 2 (dropout=0.3, BS=16)',
        'dropout': 0.3,
        'batch_size': 16,
        'model_dir': 'point_net_transf_gat_2nd_try',
        'learning_rate': '5e-4',
        'weighted_loss': False,
        'description': 'First trial with dropout=0.3',
        'val_r2': 0.5841  # From training history
    },
    'Trial 3': {
        'name': 'Trial 3 (No dropout, Weighted Loss, BS=16)',
        'dropout': 0.0,
        'batch_size': 16,
        'model_dir': 'point_net_transf_gat_3rd_trial_weighted_loss',
        'learning_rate': '5e-4',
        'weighted_loss': True,
        'description': 'No dropout with weighted loss',
        'val_r2': 0.5953  # From training history
    },
    'Trial 4': {
        'name': 'Trial 4 (No dropout, Weighted Loss, BS=16)',
        'dropout': 0.0,
        'batch_size': 16,
        'model_dir': 'point_net_transf_gat_4th_trial_weighted_loss',
        'learning_rate': '5e-4',
        'weighted_loss': True,
        'description': 'Same as Trial 3, repeated run',
        'val_r2': 0.6097  # From training history
    },
    'Trial 5': {
        'name': 'Trial 5 (dropout=0.3, BS=8)',
        'dropout': 0.3,
        'batch_size': 8,
        'model_dir': 'point_net_transf_gat_5th_try',
        'learning_rate': '5e-4',
        'weighted_loss': False,
        'description': 'Baseline - smaller batch size',
        'val_r2': 0.5500  # From training history
    },
    'Trial 6': {
        'name': 'Trial 6 (LR=3e-4, dropout=0.3, BS=8)',
        'dropout': 0.3,
        'batch_size': 8,
        'model_dir': 'point_net_transf_gat_6th_trial_lower_lr',
        'learning_rate': '3e-4',
        'weighted_loss': False,
        'description': 'Lower learning rate (too slow)',
        'val_r2': 0.5224  # From training history
    },
    'Trial 7': {
        'name': 'Trial 7 (LR=6e-4, dropout=0.3, BS=8)',
        'dropout': 0.3,
        'batch_size': 8,
        'model_dir': 'point_net_transf_gat_7th_trial_80_10_10_split',
        'learning_rate': '6e-4',
        'weighted_loss': False,
        'description': 'Higher learning rate (overshoots)',
        'val_r2': 0.5497  # From training history
    },
    'Trial 8': {
        'name': 'Trial 8 (dropout=0.2, BS=8)',
        'dropout': 0.2,
        'batch_size': 8,
        'model_dir': 'point_net_transf_gat_8th_trial_lower_dropout',
        'learning_rate': '5e-4',
        'weighted_loss': False,
        'description': 'Best model - lower dropout',
        'val_r2': 0.5970  # From training history
    }
}

def evaluate_trial(trial_name, trial_config):
    """Evaluate a single trial on test set with comprehensive metrics"""
    
    print("\n" + "="*80)
    print(f" EVALUATING {trial_name.upper()}")
    print("="*80)
    print(f"Description: {trial_config['description']}")
    print(f"Learning Rate: {trial_config['learning_rate']}")
    print(f"Dropout: {trial_config['dropout']}")
    print(f"Batch Size: {trial_config['batch_size']}")
    print(f"Weighted Loss: {trial_config['weighted_loss']}\n")
    
    MODEL_DIR = f"{BASE_PATH}/data/TR-C_Benchmarks/{trial_config['model_dir']}"
    
    # Check if model exists
    model_path = f"{MODEL_DIR}/trained_model/model.pth"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print(f"        Trying alternative paths...")
        
        # Try without trained_model folder
        alt_path = f"{MODEL_DIR}/model.pth"
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"[OK] Found model at: {alt_path}")
        else:
            print(f"     Skipping {trial_name}\n")
            return None
    
    # Load model
    try:
        model = PointNetTransfGAT(
            in_channels=5,
            out_channels=1,
            point_net_conv_layer_structure_local_mlp=[256],
            point_net_conv_layer_structure_global_mlp=[512],
            gat_conv_layer_structure=[128, 256, 512],
            dropout=trial_config['dropout'],
            use_dropout=(trial_config['dropout'] > 0),
            predict_mode_stats=False,
            dtype=torch.float32,
            log_to_wandb=False
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print(f"        Skipping {trial_name}\n")
        return None
    
    # Load test data
    try:
        # Try loading test dataset
        test_data_path = f"{MODEL_DIR}/data_created_during_training/test_dl.pt"
        if not os.path.exists(test_data_path):
            print(f"[ERROR] Test data not found: {test_data_path}")
            return None
            
        test_dataset = torch.load(test_data_path, weights_only=False)
        print(f"[OK] Test dataset loaded: {len(test_dataset)} samples")
        
        # Load validation dataset for comparison (if needed)
        val_data_path = f"{MODEL_DIR}/data_created_during_training/validation_dl.pt"
        val_dataset = None
        if os.path.exists(val_data_path):
            val_dataset = torch.load(val_data_path, weights_only=False)
            print(f"[OK] Validation dataset loaded: {len(val_dataset)} samples")
        
        # Create DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=trial_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        print(f"[OK] Test loader created: batch_size={test_loader.batch_size}\n")
    except Exception as e:
        print(f"[ERROR] Error loading test data: {e}")
        return None
    
    # Evaluate on TEST set
    print("Running TEST set evaluation...")
    num_nodes = test_dataset[0].x.shape[0]
    loss_fn = GNN_Loss("mse", num_nodes, device, False)
    
    all_targets = []
    all_predictions = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {trial_name} (Test)", leave=False):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Handle different output formats
            if isinstance(output, dict):
                predictions = output['predictions'].squeeze()
            else:
                predictions = output.squeeze()
            
            targets = batch.y.squeeze()
            
            # Calculate loss
            loss = loss_fn(predictions, targets, batch.batch)
            total_loss += loss.item()
            
            # Store for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Convert to numpy
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    # Calculate TEST metrics
    test_metrics = calculate_metrics(all_targets, all_predictions, "TEST")
    
    # Use validation R² from training history
    val_r2 = trial_config.get('val_r2', None)
    
    # Calculate val-test gap if validation R² is available
    if val_r2 is not None:
        val_test_gap = ((val_r2 - test_metrics['r2']) / val_r2 * 100)
        test_metrics['val_test_gap'] = val_test_gap
        test_metrics['val_r2'] = val_r2
    
    # Print results
    print("\n" + "="*80)
    print(f" {trial_name.upper()} - COMPLETE RESULTS")
    print("="*80)
    
    if val_r2 is not None:
        print(f"\nVALIDATION SET (from training history):")
        print(f"  R² Score:          {val_r2:.4f}")
        
    print(f"\nTEST SET:")
    print(f"  R² Score:          {test_metrics['r2']:.4f}")
    print(f"  Pearson Corr:      {test_metrics['pearson']:.4f}")
    print(f"  Spearman Corr:     {test_metrics['spearman']:.4f}")
    print(f"  MAE:               {test_metrics['mae']:.4f} vehicles/hour")
    print(f"  RMSE:              {test_metrics['rmse']:.4f} vehicles/hour")
    print(f"  MSE:               {test_metrics['mse']:.2f}")
    
    print(f"\nSTATISTICS:")
    print(f"  Target Mean:       {test_metrics['target_mean']:.4f}")
    print(f"  Target Std:        {test_metrics['target_std']:.4f}")
    print(f"  Prediction Mean:   {test_metrics['pred_mean']:.4f}")
    print(f"  Prediction Std:    {test_metrics['pred_std']:.4f}")
    print(f"  Variance Coverage: {test_metrics['variance_coverage']:.1f}%")
    
    if val_r2 is not None:
        print(f"\nGENERALIZATION:")
        print(f"  Val-Test Gap:      {test_metrics['val_test_gap']:+.2f}%")
        if test_metrics['val_test_gap'] > 10:
            print(f"  [WARNING] Severe overfitting detected!")
        elif test_metrics['val_test_gap'] > 5:
            print(f"  [WARNING] Moderate overfitting detected")
        elif test_metrics['val_test_gap'] < -2:
            print(f"  [OK] Good generalization (test better than val)")
        else:
            print(f"  [OK] Excellent generalization")
    
    # Model Quality Diagnostics
    print(f"\nMODEL DIAGNOSTICS:")
    
    # Check 1: Overall Performance (R² Score)
    if test_metrics['r2'] < 0.3:
        print(f"  [ERROR] Severe underfitting (R²={test_metrics['r2']:.4f})")
        print(f"          Model is not learning patterns effectively")
    elif test_metrics['r2'] < 0.5:
        print(f"  [WARNING] Underfitting detected (R²={test_metrics['r2']:.4f})")
        print(f"            Model performance is below acceptable threshold")
    elif test_metrics['r2'] > 0.7:
        print(f"  [OK] Excellent model performance (R²={test_metrics['r2']:.4f})")
    else:
        print(f"  [OK] Acceptable model performance (R²={test_metrics['r2']:.4f})")
    
    # Check 2: Variance Coverage Analysis
    var_cov = test_metrics['variance_coverage']
    if var_cov > 100:
        print(f"  [WARNING] Overpredicting variance ({var_cov:.1f}%)")
        print(f"            Model may be fitting noise (overfitting)")
    elif var_cov < 60:
        print(f"  [WARNING] Underpredicting variance ({var_cov:.1f}%)")
        print(f"            Model is too conservative (underfitting)")
    elif 70 <= var_cov <= 85:
        print(f"  [OK] Optimal variance coverage ({var_cov:.1f}%)")
    else:
        print(f"  [OK] Acceptable variance coverage ({var_cov:.1f}%)")
    
    # Check 3: Mean Prediction Bias
    mean_bias = test_metrics['pred_mean'] - test_metrics['target_mean']
    mean_bias_pct = (abs(mean_bias) / test_metrics['target_std']) * 100 if test_metrics['target_std'] > 0 else 0
    if mean_bias_pct > 10:
        print(f"  [WARNING] Significant prediction bias ({mean_bias:+.4f})")
        print(f"            Model systematically {'over' if mean_bias > 0 else 'under'}predicts")
    else:
        print(f"  [OK] Low prediction bias ({mean_bias:+.4f})")
    
    # Check 4: Correlation Quality
    if test_metrics['pearson'] < 0.5:
        print(f"  [ERROR] Poor correlation (Pearson={test_metrics['pearson']:.4f})")
        print(f"          Model predictions weakly correlated with targets")
    elif test_metrics['pearson'] < 0.7:
        print(f"  [WARNING] Moderate correlation (Pearson={test_metrics['pearson']:.4f})")
    else:
        print(f"  [OK] Strong correlation (Pearson={test_metrics['pearson']:.4f})")
    
    # Check 5: Spearman vs Pearson (Non-linearity Check)
    corr_diff = abs(test_metrics['pearson'] - test_metrics['spearman'])
    if corr_diff > 0.15:
        print(f"  [WARNING] Large correlation discrepancy ({corr_diff:.4f})")
        print(f"            Model may have non-linear prediction errors")
    else:
        print(f"  [OK] Consistent correlation metrics")
    
    # Overall Assessment
    print(f"\nOVERALL ASSESSMENT:")
    issues = []
    if test_metrics['r2'] < 0.5:
        issues.append("underfitting")
    if val_r2 is not None and test_metrics['val_test_gap'] > 10:
        issues.append("severe overfitting")
    if var_cov > 100:
        issues.append("variance overprediction")
    if var_cov < 60:
        issues.append("variance underprediction")
    if mean_bias_pct > 10:
        issues.append("prediction bias")
    if test_metrics['pearson'] < 0.7:
        issues.append("weak correlation")
    
    if not issues:
        print(f"  [OK] Model is healthy - no critical issues detected")
    else:
        print(f"  [WARNING] Issues detected: {', '.join(issues)}")
    
    print("="*80 + "\n")
    
    return test_metrics

def calculate_metrics(targets, predictions, set_name):
    """Calculate comprehensive evaluation metrics"""
    
    # Basic error metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Correlations
    pearson_corr, _ = pearsonr(targets, predictions)
    spearman_corr, _ = spearmanr(targets, predictions)
    
    # Variance coverage
    target_std = np.std(targets)
    pred_std = np.std(predictions)
    variance_coverage = (pred_std / target_std) * 100 if target_std > 0 else 0
    
    # Statistics
    target_mean = np.mean(targets)
    pred_mean = np.mean(predictions)
    
    # Convert all numpy types to Python native types for JSON serialization
    return {
        'r2': float(r2),
        'pearson': float(pearson_corr),
        'spearman': float(spearman_corr),
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'target_mean': float(target_mean),
        'target_std': float(target_std),
        'pred_mean': float(pred_mean),
        'pred_std': float(pred_std),
        'variance_coverage': float(variance_coverage)
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" COMPREHENSIVE EVALUATION - ALL 7 TRIALS (Trials 2-8)")
    print("="*80)
    print(" This script evaluates all trained models on test sets")
    print(" and generates complete metrics for visualization update")
    print("="*80 + "\n")
    
    results = {}
    failed_trials = []
    
    for trial_name, trial_config in TRIALS.items():
        result = evaluate_trial(trial_name, trial_config)
        if result is not None:
            results[trial_name] = result
        else:
            failed_trials.append(trial_name)
        print("\n")
    
    # Final Summary
    print("\n" + "="*80)
    print(" FINAL SUMMARY - ALL TRIALS")
    print("="*80)
    
    if results:
        print(f"\n{'Trial':<10} {'Val R²':<10} {'Test R²':<10} {'Pearson':<10} {'MAE':<10} {'Var Cov':<12} {'Gap %':<10}")
        print("-"*80)
        
        for trial_name, metrics in results.items():
            val_r2_str = f"{metrics.get('val_r2', 0):.4f}" if 'val_r2' in metrics and metrics.get('val_r2') else "N/A"
            gap_str = f"{metrics.get('val_test_gap', 0):+.2f}%" if 'val_test_gap' in metrics and metrics.get('val_test_gap') is not None else "N/A"
            
            print(f"{trial_name:<10} "
                  f"{val_r2_str:<10} "
                  f"{metrics['r2']:<10.4f} "
                  f"{metrics['pearson']:<10.4f} "
                  f"{metrics['mae']:<10.4f} "
                  f"{metrics['variance_coverage']:<12.1f}% "
                  f"{gap_str:<10}")
        
        print("="*80)
        
        # Generate Health Report
        print("\n" + "="*80)
        print(" MODEL HEALTH SUMMARY")
        print("="*80)
        
        healthy_models = []
        overfitting_models = []
        underfitting_models = []
        
        for trial_name, metrics in results.items():
            issues = []
            
            # Check for underfitting
            if metrics['r2'] < 0.5:
                issues.append("underfitting")
                underfitting_models.append(trial_name)
            
            # Check for overfitting
            if 'val_test_gap' in metrics and metrics['val_test_gap'] is not None:
                if metrics['val_test_gap'] > 10:
                    issues.append("overfitting")
                    overfitting_models.append(trial_name)
            
            # Check variance coverage
            if metrics['variance_coverage'] > 100:
                issues.append("variance overprediction")
            elif metrics['variance_coverage'] < 60:
                issues.append("variance underprediction")
            
            if not issues:
                healthy_models.append(trial_name)
        
        print(f"\n[OK] Healthy Models ({len(healthy_models)}):")
        if healthy_models:
            for model in healthy_models:
                r2 = results[model]['r2']
                pearson = results[model]['pearson']
                print(f"     {model}: R²={r2:.4f}, Pearson={pearson:.4f}")
        else:
            print(f"     None")
        
        print(f"\n[WARNING] Overfitting Models ({len(overfitting_models)}):")
        if overfitting_models:
            for model in overfitting_models:
                gap = results[model].get('val_test_gap', 0)
                print(f"     {model}: Val-Test Gap={gap:+.2f}%")
        else:
            print(f"     None")
        
        print(f"\n[ERROR] Underfitting Models ({len(underfitting_models)}):")
        if underfitting_models:
            for model in underfitting_models:
                r2 = results[model]['r2']
                print(f"     {model}: Test R²={r2:.4f}")
        else:
            print(f"     None")
        
        # Best Model Recommendation
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        best_name = best_model[0]
        best_r2 = best_model[1]['r2']
        best_pearson = best_model[1]['pearson']
        best_gap = best_model[1].get('val_test_gap', 0)
        
        print(f"\n[OK] RECOMMENDED MODEL: {best_name}")
        print(f"     Test R²: {best_r2:.4f}")
        print(f"     Pearson: {best_pearson:.4f}")
        if best_gap is not None:
            print(f"     Val-Test Gap: {best_gap:+.2f}%")
        print(f"     MAE: {best_model[1]['mae']:.4f} vehicles/hour")
        print(f"     Variance Coverage: {best_model[1]['variance_coverage']:.1f}%")
        
        print("="*80)
        
        # Save results to JSON
        output_file = f"{BASE_PATH}/all_trials_complete_evaluation.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {output_file}")
        
        # Generate Python dict for easy copy-paste into visualization script
        print("\n" + "="*80)
        print(" COPY-PASTE THIS INTO generate_all_visualizations.py")
        print("="*80)
        print("\nTRIALS_DATA = {")
        
        for trial_name, metrics in results.items():
            trial_num = trial_name.split()[1]
            config = TRIALS[trial_name]
            
            print(f"    '{trial_name}': {{")
            print(f"        'val_r2': {metrics.get('val_r2', 0):.4f},")
            print(f"        'test_r2': {metrics['r2']:.4f},")
            print(f"        'test_pearson': {metrics['pearson']:.4f},")
            print(f"        'test_spearman': {metrics['spearman']:.4f},")
            print(f"        'test_mae': {metrics['mae']:.4f},")
            print(f"        'test_rmse': {metrics['rmse']:.4f},")
            print(f"        'test_mse': {metrics['mse']:.2f},")
            print(f"        'target_mean': {metrics['target_mean']:.4f},")
            print(f"        'target_std': {metrics['target_std']:.4f},")
            print(f"        'pred_mean': {metrics['pred_mean']:.4f},")
            print(f"        'pred_std': {metrics['pred_std']:.4f},")
            print(f"        'variance_coverage': {metrics['variance_coverage']:.1f},")
            if 'val_test_gap' in metrics and metrics['val_test_gap'] is not None:
                print(f"        'val_test_gap': {metrics['val_test_gap']:.2f},")
            else:
                print(f"        'val_test_gap': None,")
            if 'val_r2' in metrics and metrics['val_r2']:
                print(f"        'val_r2': {metrics['val_r2']:.4f},")
            else:
                print(f"        'val_r2': None,")
            print(f"        'learning_rate': '{config['learning_rate']}',")
            print(f"        'dropout': {config['dropout']},")
            print(f"        'batch_size': {config['batch_size']},")
            print(f"        'weighted_loss': {str(config['weighted_loss'])},")
            print(f"        'architecture': 'PointNetTransfGAT'")
            print(f"    }},")
        
        print("}")
        print("="*80)
        
    else:
        print("[ERROR] No trials could be evaluated. Check model paths.")
    
    if failed_trials:
        print(f"\n[WARNING] Failed to evaluate: {', '.join(failed_trials)}")
        print("          Check if model files exist in the specified directories.")
    
    print("\n" + "="*80)
    print(" EVALUATION COMPLETE!")
    print("="*80)
