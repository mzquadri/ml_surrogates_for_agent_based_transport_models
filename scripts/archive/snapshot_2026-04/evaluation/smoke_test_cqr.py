"""
Smoke Tests for CQR (Conformalized Quantile Regression) Extension
==================================================================
Tests the quantile model, pinball loss, backward pass, and the
exact CQR calibration formula before committing to full training.

Tests:
  1. Model instantiation -- parameter count sanity check
  2. Forward pass       -- output shapes [N, 1] each, no NaN/Inf
  3. Pinball loss       -- scalar, finite, positive, correct structure
  4. Backward pass      -- no NaN/Inf gradients
  5. CQR formula        -- exact apply_inverse() index matches reference

All tests must pass before train_cqr.py is launched.

Author: Mohd Zamin Quadri (CQR UQ extension)
Reference: Romano et al. (2019) NeurIPS, arXiv:1905.03222
           yromano/cqr (github.com/yromano/cqr)
"""

import sys
import math
import torch
import numpy as np
from pathlib import Path

# Add parent directories to path
code_dir   = Path(__file__).parent.parent.parent
scripts_dir = code_dir / 'scripts'
sys.path.insert(0, str(scripts_dir))

from gnn.models.point_net_transf_gat_quantile import PointNetTransfGATQuantile
from gnn.losses.quantile_loss import PinballLoss


# ---------------------------------------------------------------------------
# Helper: exact CQR conformal quantile (mirrors apply_inverse() in nc.py)
# ---------------------------------------------------------------------------

def cqr_conformal_q(scores, alpha):
    """
    Exact conformal quantile following QuantileRegErrFunc.apply_inverse()
    from yromano/cqr/nonconformist/nc.py.

    Formula:
      index = int(ceil((1 - alpha) * (n + 1))) - 1
      index = clamp(index, 0, n-1)
      return scores_sorted[index]

    Args:
        scores (np.ndarray): 1D array of nonconformity scores (length n).
        alpha  (float):      Miscoverage level (e.g. 0.10 for 90% coverage).

    Returns:
        float: Conformal quantile Q_hat.
    """
    n = len(scores)
    scores_sorted = np.sort(scores)
    index = int(math.ceil((1.0 - alpha) * (n + 1))) - 1
    index = min(max(index, 0), n - 1)
    return float(scores_sorted[index])


# ---------------------------------------------------------------------------
# Test 1: Model instantiation
# ---------------------------------------------------------------------------

def smoke_test_model_instantiation():
    """Test 1: Instantiate PointNetTransfGATQuantile with T8 hyperparams."""
    print("\n" + "=" * 80)
    print("TEST 1: Model Instantiation")
    print("=" * 80)

    try:
        model = PointNetTransfGATQuantile(
            in_channels=5,
            out_channels=2,
            dropout=0.2,
            use_dropout=True,
            predict_mode_stats=False,
            log_to_wandb=False
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[PASS] Model instantiated successfully")
        print(f"       Parameters: {n_params:,}")

        # Quantile model has same architecture as T8 except GATConv(64,2) vs (64,1).
        # Actual T8 count is ~1.4M (measured). Quantile adds ~130 extra params.
        # Reasonable range: 500K -- 5M
        assert n_params > 500_000,   f"Param count too low: {n_params:,}"
        assert n_params < 5_000_000, f"Param count too high: {n_params:,}"

        print(f"[PASS] Parameter count in expected range (500K -- 5M)")
        return model

    except Exception as e:
        print(f"[FAIL] Model instantiation failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Test 2: Forward pass
# ---------------------------------------------------------------------------

def smoke_test_forward_pass(model, graph):
    """Test 2: Forward pass on a single graph -- shapes and numerics."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass")
    print("=" * 80)

    print(f"Input graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Features:    {graph.x.shape}")

    try:
        model.eval()
        with torch.no_grad():
            q_lo, q_hi = model(graph)

        print(f"[PASS] Forward pass completed")
        print(f"       q_lo shape: {q_lo.shape}  (expected: [{graph.num_nodes}, 1])")
        print(f"       q_hi shape: {q_hi.shape}  (expected: [{graph.num_nodes}, 1])")
        print(f"       q_lo range: [{q_lo.min().item():.4f}, {q_lo.max().item():.4f}]")
        print(f"       q_hi range: [{q_hi.min().item():.4f}, {q_hi.max().item():.4f}]")

        # Shape checks
        assert q_lo.shape == (graph.num_nodes, 1), f"q_lo shape mismatch: {q_lo.shape}"
        assert q_hi.shape == (graph.num_nodes, 1), f"q_hi shape mismatch: {q_hi.shape}"

        # Numerical checks
        assert not torch.isnan(q_lo).any(), "q_lo contains NaN"
        assert not torch.isinf(q_lo).any(), "q_lo contains Inf"
        assert not torch.isnan(q_hi).any(), "q_hi contains NaN"
        assert not torch.isinf(q_hi).any(), "q_hi contains Inf"

        print("[PASS] Shape and numerical checks passed")
        return q_lo, q_hi

    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Test 3: Pinball loss
# ---------------------------------------------------------------------------

def smoke_test_pinball_loss(model, graph):
    """Test 3: Pinball loss -- scalar, finite, positive, correct structure."""
    print("\n" + "=" * 80)
    print("TEST 3: Pinball Loss Computation")
    print("=" * 80)

    try:
        model.train()
        q_lo, q_hi = model(graph)

        loss_fn = PinballLoss(alpha=0.10)  # tau_lo=0.05, tau_hi=0.95
        loss = loss_fn(q_lo, q_hi, graph.y)

        print(f"[PASS] Loss computed: {loss.item():.6f}")
        print(f"       tau_lo={loss_fn.tau_lo}, tau_hi={loss_fn.tau_hi}")
        print(f"       q_lo range: [{q_lo.min().item():.4f}, {q_lo.max().item():.4f}]")
        print(f"       q_hi range: [{q_hi.min().item():.4f}, {q_hi.max().item():.4f}]")
        print(f"       y    range: [{graph.y.min().item():.4f}, {graph.y.max().item():.4f}]")

        # Checks
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() >= 0.0, f"Pinball loss must be non-negative: {loss.item()}"

        # Verify structure: loss_lo + loss_hi individually
        y    = graph.y.squeeze()
        q_lo_s = q_lo.squeeze()
        q_hi_s = q_hi.squeeze()

        errors_lo = y - q_lo_s
        errors_hi = y - q_hi_s
        loss_lo = torch.max((0.05 - 1.0) * errors_lo, 0.05 * errors_lo)
        loss_hi = torch.max((0.95 - 1.0) * errors_hi, 0.95 * errors_hi)
        expected = torch.mean(loss_lo + loss_hi)

        diff = abs(loss.item() - expected.item())
        assert diff < 1e-5, f"Loss value mismatch: got {loss.item():.6f}, expected {expected.item():.6f}"

        print(f"[PASS] Loss structure verified (matches manual computation, diff={diff:.2e})")
        print("[PASS] All pinball loss checks passed")
        return loss

    except Exception as e:
        print(f"[FAIL] Pinball loss test failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Test 4: Backward pass
# ---------------------------------------------------------------------------

def smoke_test_backward_pass(model, graph):
    """Test 4: Backward pass -- no NaN/Inf gradients."""
    print("\n" + "=" * 80)
    print("TEST 4: Backward Pass")
    print("=" * 80)

    try:
        model.train()
        # Zero out any existing gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        q_lo, q_hi = model(graph)
        loss_fn = PinballLoss(alpha=0.10)
        loss = loss_fn(q_lo, q_hi, graph.y)
        loss.backward()

        # Check gradients
        nan_params  = []
        inf_params  = []
        none_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    none_params.append(name)
                elif torch.isnan(param.grad).any():
                    nan_params.append(name)
                elif torch.isinf(param.grad).any():
                    inf_params.append(name)

        if none_params:
            print(f"  [NOTE] {len(none_params)} parameters have no gradient (may be unused nodes):")
            for n in none_params[:5]:
                print(f"         {n}")

        if nan_params:
            print(f"  [FAIL] {len(nan_params)} parameters have NaN gradients:")
            for n in nan_params[:5]:
                print(f"         {n}")
            raise AssertionError(f"NaN gradients found in {len(nan_params)} parameters")

        if inf_params:
            print(f"  [FAIL] {len(inf_params)} parameters have Inf gradients:")
            for n in inf_params[:5]:
                print(f"         {n}")
            raise AssertionError(f"Inf gradients found in {len(inf_params)} parameters")

        # Compute gradient norm for diagnostics
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        print(f"[PASS] Backward pass completed")
        print(f"       Gradient norm: {total_norm:.4f}")
        print("[PASS] No NaN or Inf gradients detected")
        return True

    except Exception as e:
        print(f"[FAIL] Backward pass failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Test 5: CQR calibration formula
# ---------------------------------------------------------------------------

def smoke_test_cqr_formula():
    """
    Test 5: Verify exact CQR conformal quantile formula.

    Cross-check against expected index from QuantileRegErrFunc.apply_inverse()
    in yromano/cqr/nonconformist/nc.py:
      index = int(ceil((1 - alpha) * (n + 1))) - 1

    Test cases:
      n=100, alpha=0.10 -> index=90  -> 91st sorted score
      n=100, alpha=0.05 -> index=95  -> 96th sorted score
      n=  5, alpha=0.10 -> index=4   -> 5th sorted score (clamped)
      n=  5, alpha=0.50 -> index=2   -> 3rd sorted score
    """
    print("\n" + "=" * 80)
    print("TEST 5: CQR Conformal Quantile Formula")
    print("=" * 80)

    test_cases = [
        # (n, alpha, expected_index, description)
        (100, 0.10,  90, "n=100, alpha=0.10: main case (90% coverage)"),
        (100, 0.05,  95, "n=100, alpha=0.05: 95% coverage"),
        (  5, 0.10,   4, "n=5,   alpha=0.10: small calibration set (clamped to n-1)"),
        (  5, 0.50,   2, "n=5,   alpha=0.50: 50% coverage"),
    ]

    all_passed = True

    for (n, alpha, expected_index, desc) in test_cases:
        # Scores: 0, 1, 2, ..., n-1  (sorted ascending by construction)
        scores = np.arange(n, dtype=float)

        # Compute index the same way as apply_inverse()
        computed_index = int(math.ceil((1.0 - alpha) * (n + 1))) - 1
        computed_index = min(max(computed_index, 0), n - 1)

        q_hat = cqr_conformal_q(scores, alpha)
        expected_value = float(expected_index)

        index_ok = (computed_index == expected_index)
        value_ok = abs(q_hat - expected_value) < 1e-9

        status = "[PASS]" if (index_ok and value_ok) else "[FAIL]"
        if not (index_ok and value_ok):
            all_passed = False

        print(f"  {status} {desc}")
        print(f"         index: computed={computed_index}, expected={expected_index}")
        print(f"         Q_hat: computed={q_hat:.1f}, expected={expected_value:.1f}")

        if not index_ok:
            raise AssertionError(
                f"Index mismatch for {desc}: got {computed_index}, expected {expected_index}"
            )
        if not value_ok:
            raise AssertionError(
                f"Q_hat mismatch for {desc}: got {q_hat}, expected {expected_value}"
            )

    # Additional check: nonconformity score formula
    # E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    # For y=5, q_lo=3, q_hi=7: E = max(3-5, 5-7) = max(-2, -2) = -2
    # For y=5, q_lo=6, q_hi=7: E = max(6-5, 5-7) = max(1, -2)  =  1
    # For y=5, q_lo=3, q_hi=4: E = max(3-5, 5-4) = max(-2, 1)  =  1

    y    = np.array([5.0, 5.0, 5.0])
    q_lo = np.array([3.0, 6.0, 3.0])
    q_hi = np.array([7.0, 7.0, 4.0])
    E    = np.maximum(q_lo - y, y - q_hi)
    expected_E = np.array([-2.0, 1.0, 1.0])

    assert np.allclose(E, expected_E), f"Nonconformity score mismatch: {E} vs {expected_E}"
    print("\n  [PASS] Nonconformity score formula verified")
    print("         E = max(q_lo - y, y - q_hi)")
    print(f"         Test: {E.tolist()} == {expected_E.tolist()}")

    # CQR interval construction
    # lo = q_lo - Q_hat,  hi = q_hi + Q_hat
    Q_hat = 2.0
    lo = q_lo - Q_hat
    hi = q_hi + Q_hat
    expected_lo = np.array([1.0, 4.0, 1.0])
    expected_hi = np.array([9.0, 9.0, 6.0])

    assert np.allclose(lo, expected_lo), f"Interval lo mismatch: {lo}"
    assert np.allclose(hi, expected_hi), f"Interval hi mismatch: {hi}"
    print("\n  [PASS] Interval construction verified")
    print("         lo = q_lo - Q_hat,  hi = q_hi + Q_hat")

    print("\n[PASS] All CQR formula checks passed")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all 5 smoke tests."""
    print("=" * 80)
    print("CQR SMOKE TESTS (5 tests)")
    print("=" * 80)
    print("Purpose: verify code correctness before training")
    print("Reference: Romano et al. (2019), yromano/cqr")

    # Locate T8 test dataloader (read-only, never modified)
    code_dir    = Path(__file__).parent.parent.parent
    test_dl_path = code_dir / 'data' / 'TR-C_Benchmarks' \
                  / 'point_net_transf_gat_8th_trial_lower_dropout' \
                  / 'data_created_during_training' / 'test_dl.pt'

    print("\n" + "=" * 80)
    print("Loading T8 Test Data (read-only)")
    print("=" * 80)
    print(f"Path: {test_dl_path}")

    if not test_dl_path.exists():
        print(f"[ERROR] T8 test dataloader not found at {test_dl_path}")
        return 1

    test_dl = torch.load(str(test_dl_path), weights_only=False)
    graph   = test_dl[0]   # First test graph -- single graph for smoke tests

    print(f"[PASS] Loaded test dataloader ({len(test_dl)} graphs)")
    print(f"       Graph 0: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"       x shape: {graph.x.shape}, pos shape: {graph.pos.shape}")
    print(f"       y range: [{graph.y.min().item():.4f}, {graph.y.max().item():.4f}]")

    # Run tests
    try:
        model = smoke_test_model_instantiation()
        q_lo, q_hi = smoke_test_forward_pass(model, graph)
        loss = smoke_test_pinball_loss(model, graph)
        smoke_test_backward_pass(model, graph)
        smoke_test_cqr_formula()

        print("\n" + "=" * 80)
        print("*** ALL 5 SMOKE TESTS PASSED ***")
        print("=" * 80)
        print("\nSummary:")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  q_lo range: [{q_lo.min().item():.4f}, {q_lo.max().item():.4f}]")
        print(f"  q_hi range: [{q_hi.min().item():.4f}, {q_hi.max().item():.4f}]")
        print(f"  Pinball loss (random init): {loss.item():.6f}")
        print("\n[READY] Code verified. Proceed to train_cqr.py")
        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("*** SMOKE TESTS FAILED ***")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n[BLOCKED] Fix issues before launching training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
