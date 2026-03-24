"""
Phase 0: Verify that weight remapping fix produces correct predictions.
Compares fixed model output against saved deterministic predictions.
Usage: conda run -n thesis-env python scripts/verify_ensemble_fix.py
"""

import os, sys, time

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TQDM_DISABLE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch
import numpy as np
from gnn.models.point_net_transf_gat import PointNetTransfGAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "TR-C_Benchmarks")
T8_FOLDER = os.path.join(DATA_ROOT, "point_net_transf_gat_8th_trial_lower_dropout")


def load_model_fixed(model_folder, dropout, device):
    """Load model with GATConv weight remapping fix."""
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=dropout,
        use_dropout=True,
        predict_mode_stats=False,
    )
    model_path = os.path.join(model_folder, "trained_model", "model.pth")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    remapped = {}
    for k, v in state_dict.items():
        if ".lin.weight" in k:
            remapped[k.replace(".lin.weight", ".lin_src.weight")] = v
            remapped[k.replace(".lin.weight", ".lin_dst.weight")] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=True)
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 0: VERIFY WEIGHT REMAPPING FIX")
    print("=" * 60)
    print("Device:", DEVICE)
    t0 = time.time()

    # Load T8 model with fix
    print("\n[1] Loading T8 model with weight remapping fix...")
    model = load_model_fixed(T8_FOLDER, dropout=0.2, device=DEVICE)
    print("    Model loaded successfully (strict=True passed)")

    # Load test data
    print("[2] Loading test data...")
    test_dl = torch.load(
        os.path.join(T8_FOLDER, "data_created_during_training", "test_dl.pt"),
        weights_only=False,
    )
    print("    %d test graphs loaded" % len(test_dl))

    # Run deterministic prediction on graph 0
    print("[3] Running deterministic prediction on graph 0...")
    model.eval()
    data = test_dl[0].to(DEVICE)
    with torch.no_grad():
        out = model(data)
        if isinstance(out, tuple):
            out = out[0]
    pred_new = out.squeeze().cpu().numpy()
    target = data.y.squeeze().cpu().numpy()
    print(
        "    Output shape:",
        pred_new.shape,
        " Range: [%.2f, %.2f]" % (pred_new.min(), pred_new.max()),
    )

    # Load saved deterministic predictions
    print("[4] Loading saved deterministic predictions...")
    saved_det = np.load(
        os.path.join(T8_FOLDER, "uq_results", "deterministic_full_100graphs.npz")
    )
    pred_saved = saved_det["predictions"][:31635]  # First graph
    target_saved = saved_det["targets"][:31635]
    print(
        "    Saved shape:",
        pred_saved.shape,
        " Range: [%.2f, %.2f]" % (pred_saved.min(), pred_saved.max()),
    )

    # Compare targets (should be identical)
    target_match = np.allclose(target, target_saved, atol=1e-5)
    print("\n[5] VERIFICATION RESULTS:")
    print("    Targets match: %s" % target_match)

    # Compare predictions
    if pred_new.shape == pred_saved.shape:
        max_diff = np.max(np.abs(pred_new - pred_saved))
        mean_diff = np.mean(np.abs(pred_new - pred_saved))
        corr = np.corrcoef(pred_new, pred_saved)[0, 1]
        print("    Max |diff|:  %.6f" % max_diff)
        print("    Mean |diff|: %.6f" % mean_diff)
        print("    Correlation: %.8f" % corr)

        if max_diff < 0.01:
            print(
                "    >> EXACT MATCH: old PyG and new PyG+remapping produce identical predictions"
            )
        elif corr > 0.9999:
            print(
                "    >> NEAR MATCH: tiny numerical differences, but functionally identical"
            )
        elif corr > 0.99:
            print(
                "    >> CLOSE MATCH: small differences exist but predictions are highly correlated"
            )
        else:
            print(
                "    >> WARNING: significant differences between old and new predictions!"
            )
    else:
        print(
            "    >> ERROR: shape mismatch! new=%s saved=%s"
            % (pred_new.shape, pred_saved.shape)
        )

    # R2 check on graph 0
    ss_res = np.sum((target - pred_new) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2_new = 1 - ss_res / ss_tot

    ss_res_s = np.sum((target_saved - pred_saved) ** 2)
    r2_saved = 1 - ss_res_s / ss_tot
    print("\n    R2 (new fix):   %.4f" % r2_new)
    print("    R2 (saved old): %.4f" % r2_saved)

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(pred_new)) or np.any(np.isinf(pred_new))
    print("    NaN/Inf in new predictions: %s" % has_nan)

    # Also verify MC Dropout works (quick 3-pass test)
    print("\n[6] Quick MC Dropout test (S=3) on graph 0...")
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    mc_preds = []
    with torch.no_grad():
        for _ in range(3):
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
            mc_preds.append(out.squeeze().cpu().numpy())
    mc_preds = np.array(mc_preds)
    mc_mean = mc_preds.mean(axis=0)
    mc_std = mc_preds.std(axis=0)
    print("    MC mean range: [%.2f, %.2f]" % (mc_mean.min(), mc_mean.max()))
    print(
        "    MC std  range: [%.4f, %.4f], mean=%.4f"
        % (mc_std.min(), mc_std.max(), mc_std.mean())
    )

    # Compare MC results with saved MC Dropout
    saved_mc = np.load(
        os.path.join(T8_FOLDER, "uq_results", "checkpoints_mc30", "graph_0000.npz")
    )
    saved_mc_mean = saved_mc["predictions"]
    saved_mc_std = saved_mc["uncertainties"]
    print(
        "    Saved MC std range: [%.4f, %.4f], mean=%.4f"
        % (saved_mc_std.min(), saved_mc_std.max(), saved_mc_std.mean())
    )
    print(
        "    MC std magnitude similar: %s"
        % (0.5 < mc_std.mean() / saved_mc_std.mean() < 2.0)
    )

    elapsed = time.time() - t0
    print("\nTotal time: %.1fs" % elapsed)
    print("VERIFICATION COMPLETE")
