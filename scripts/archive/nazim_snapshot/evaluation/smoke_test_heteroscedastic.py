"""
Smoke Tests for Heteroscedastic Extension
==========================================
Tests the heteroscedastic model, loss, and inference on a single graph/batch
before committing to full training.

Tests:
1. Model instantiation
2. Forward pass (shapes, NaN/Inf checks)
3. Loss computation
4. MC Dropout inference (S=5 samples)
5. Uncertainty decomposition verification

Author: Mohd Zamin Quadri
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directories to path
code_dir = Path(__file__).parent.parent.parent
scripts_dir = code_dir / 'scripts'
sys.path.insert(0, str(scripts_dir))

from gnn.models.point_net_transf_gat_heteroscedastic import PointNetTransfGATHeteroscedastic
from gnn.losses.heteroscedastic_loss import HeteroscedasticGaussianLoss
from gnn.heteroscedastic_mc_dropout import heteroscedastic_mc_dropout_single_graph


def smoke_test_model_instantiation():
    """Test 1: Instantiate heteroscedastic model"""
    print("\n" + "="*80)
    print("TEST 1: Model Instantiation")
    print("="*80)
    
    try:
        model = PointNetTransfGATHeteroscedastic(
            dropout=0.2,  # T8 dropout
            use_dropout=True,
            predict_mode_stats=False
        )
        print("[PASS] Model instantiated successfully")
        print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"[FAIL] Model instantiation failed: {e}")
        raise


def smoke_test_forward_pass(model, graph):
    """Test 2: Forward pass on single graph"""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass")
    print("="*80)
    
    print(f"Input graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Features: {graph.x.shape}")
    
    try:
        model.eval()
        with torch.no_grad():
            mean, log_var = model(graph)
        
        print(f"[PASS] Forward pass successful")
        print(f"       Mean shape: {mean.shape} (expected: [{graph.num_nodes}, 1])")
        print(f"       Log_var shape: {log_var.shape} (expected: [{graph.num_nodes}, 1])")
        print(f"       Mean range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
        print(f"       Log_var range: [{log_var.min().item():.4f}, {log_var.max().item():.4f}]")
        
        # Check shapes
        assert mean.shape == (graph.num_nodes, 1), f"Mean shape mismatch: {mean.shape}"
        assert log_var.shape == (graph.num_nodes, 1), f"Log_var shape mismatch: {log_var.shape}"
        
        # Check for NaN/Inf
        assert not torch.isnan(mean).any(), "Mean contains NaN"
        assert not torch.isinf(mean).any(), "Mean contains Inf"
        assert not torch.isnan(log_var).any(), "Log_var contains NaN"
        assert not torch.isinf(log_var).any(), "Log_var contains Inf"
        
        print("[PASS] All shape and numerical checks passed")
        return mean, log_var
        
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        raise


def smoke_test_loss_computation(model, graph):
    """Test 3: Loss computation on single batch"""
    print("\n" + "="*80)
    print("TEST 3: Loss Computation")
    print("="*80)
    
    try:
        model.train()
        mean, log_var = model(graph)
        
        loss_fn = HeteroscedasticGaussianLoss(var_reg_lambda=0.01, var_reg_type='log')
        loss = loss_fn(mean, log_var, graph.y)
        
        print(f"[PASS] Loss computed successfully")
        print(f"       Loss value: {loss.item():.6f}")
        
        # Check loss is finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
        
        # Get diagnostics
        diag = loss_fn.get_diagnostics(mean, log_var, graph.y)
        print(f"       NLL: {diag['nll']:.6f}")
        print(f"       Regularization: {diag['reg']:.6f}")
        print(f"       Mean log_var: {diag['mean_log_var']:.4f}")
        print(f"       Mean sigma: {diag['mean_sigma']:.4f}")
        print(f"       Frac underconfident: {diag['frac_underconfident']:.4f}")
        
        # Check diagnostics are finite
        for key, val in diag.items():
            assert np.isfinite(val), f"Diagnostic '{key}' is not finite: {val}"
        
        print("[PASS] All loss and diagnostic checks passed")
        return loss, diag
        
    except Exception as e:
        print(f"[FAIL] Loss computation failed: {e}")
        raise


def smoke_test_mc_dropout_inference(model, graph):
    """Test 4: MC Dropout inference with S=5 samples"""
    print("\n" + "="*80)
    print("TEST 4: MC Dropout Inference (S=5)")
    print("="*80)
    
    try:
        result = heteroscedastic_mc_dropout_single_graph(
            model=model,
            graph=graph,
            num_samples=5,
            device='cpu'
        )
        
        print(f"[PASS] MC Dropout inference successful")
        print(f"       Predictions shape: {result['mean'].shape} (expected: [{graph.num_nodes},])")
        print(f"       Predictions range: [{result['mean'].min():.4f}, {result['mean'].max():.4f}]")
        
        # Check uncertainty decomposition
        sigma_a = result['sigma_aleatoric']
        sigma_e = result['sigma_epistemic']
        sigma_t = result['sigma_total']
        
        print(f"\n       Aleatoric uncertainty (data noise):")
        print(f"         Mean: {sigma_a.mean():.4f}, Std: {sigma_a.std():.4f}")
        print(f"         Range: [{sigma_a.min():.4f}, {sigma_a.max():.4f}]")
        
        print(f"\n       Epistemic uncertainty (model uncertainty):")
        print(f"         Mean: {sigma_e.mean():.4f}, Std: {sigma_e.std():.4f}")
        print(f"         Range: [{sigma_e.min():.4f}, {sigma_e.max():.4f}]")
        
        print(f"\n       Total uncertainty:")
        print(f"         Mean: {sigma_t.mean():.4f}, Std: {sigma_t.std():.4f}")
        print(f"         Range: [{sigma_t.min():.4f}, {sigma_t.max():.4f}]")
        
        # Numerical checks
        assert np.all(np.isfinite(result['mean'])), "Predictions contain NaN/Inf"
        assert np.all(np.isfinite(sigma_a)), "Aleatoric uncertainty contains NaN/Inf"
        assert np.all(np.isfinite(sigma_e)), "Epistemic uncertainty contains NaN/Inf"
        assert np.all(np.isfinite(sigma_t)), "Total uncertainty contains NaN/Inf"
        
        # Non-negativity checks
        assert np.all(sigma_a >= 0), "Aleatoric uncertainty has negative values"
        assert np.all(sigma_e >= 0), "Epistemic uncertainty has negative values"
        assert np.all(sigma_t >= 0), "Total uncertainty has negative values"
        
        # Decomposition check: sigma_total^2 = sigma_a^2 + sigma_e^2
        expected_total = np.sqrt(sigma_a**2 + sigma_e**2)
        decomp_error = np.abs(sigma_t - expected_total).max()
        print(f"\n       Decomposition check (max error): {decomp_error:.6e}")
        assert decomp_error < 1e-5, f"Decomposition error too large: {decomp_error}"
        
        # Check that uncertainties are not trivial (all zeros)
        assert sigma_a.mean() > 0, "Aleatoric uncertainty is zero everywhere"
        assert sigma_t.mean() > 0, "Total uncertainty is zero everywhere"
        
        print("\n[PASS] All MC Dropout and uncertainty decomposition checks passed")
        return result
        
    except Exception as e:
        print(f"[FAIL] MC Dropout inference failed: {e}")
        raise


def main():
    """Run all smoke tests"""
    print("="*80)
    print("HETEROSCEDASTIC EXTENSION SMOKE TESTS")
    print("="*80)
    print("\nPurpose: Verify code correctness before training")
    print("Tests: Model instantiation, forward pass, loss, MC Dropout inference")
    
    # Load test data
    print("\n" + "="*80)
    print("Loading Test Data")
    print("="*80)
    
    test_dl_path = Path('data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/data_created_during_training/test_dl.pt')
    print(f"Path: {test_dl_path}")
    
    if not test_dl_path.exists():
        print(f"[ERROR] Test dataloader not found at {test_dl_path}")
        return 1
    
    test_dl = torch.load(test_dl_path, weights_only=False)
    graph = test_dl[0]  # First test graph
    
    print(f"[PASS] Loaded test dataloader")
    print(f"       Total graphs: {len(test_dl)}")
    print(f"       Selected graph 0: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"       Target range: [{graph.y.min().item():.4f}, {graph.y.max().item():.4f}]")
    
    # Run smoke tests
    try:
        model = smoke_test_model_instantiation()
        mean, log_var = smoke_test_forward_pass(model, graph)
        loss, diag = smoke_test_loss_computation(model, graph)
        result = smoke_test_mc_dropout_inference(model, graph)
        
        # Final summary
        print("\n" + "="*80)
        print("*** ALL SMOKE TESTS PASSED ***")
        print("="*80)
        print("\nSummary of outputs:")
        print(f"  Forward pass:")
        print(f"    Mean: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
        print(f"    Log_var: [{log_var.min().item():.4f}, {log_var.max().item():.4f}]")
        print(f"\n  Loss computation:")
        print(f"    Loss: {loss.item():.6f}")
        print(f"    Mean sigma: {diag['mean_sigma']:.4f}")
        print(f"\n  MC Dropout inference:")
        print(f"    Aleatoric (mean): {result['sigma_aleatoric'].mean():.4f}")
        print(f"    Epistemic (mean): {result['sigma_epistemic'].mean():.4f}")
        print(f"    Total (mean): {result['sigma_total'].mean():.4f}")
        
        print("\n[READY] Phase 1A complete. Code is ready for Phase 1B (training).")
        print("[ACTION] Please review outputs and approve before training.")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("*** SMOKE TESTS FAILED ***")
        print("="*80)
        print(f"\nError: {e}")
        print("\n[BLOCKED] Fix issues before proceeding to training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
