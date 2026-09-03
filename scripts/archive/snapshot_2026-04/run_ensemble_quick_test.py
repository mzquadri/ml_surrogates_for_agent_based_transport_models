"""
Quick test: 3 graphs, 2 runs to verify the weight remapping fix works.
Expected: R2 ~0.57 (not ~0).
Usage: conda run -n thesis-env python scripts/run_ensemble_quick_test.py
"""

import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUTF8"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch
import numpy as np
from scipy.stats import spearmanr
from gnn.models.point_net_transf_gat import PointNetTransfGAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "TR-C_Benchmarks")


def load_model_fixed():
    folder = os.path.join(DATA_ROOT, "point_net_transf_gat_8th_trial_lower_dropout")
    model_path = os.path.join(folder, "trained_model", "model.pth")
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=0.2,
        use_dropout=True,
        predict_mode_stats=False,
    )
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    remapped = {}
    for k, v in state_dict.items():
        if ".lin.weight" in k:
            remapped[k.replace(".lin.weight", ".lin_src.weight")] = v
            remapped[k.replace(".lin.weight", ".lin_dst.weight")] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=True)
    model = model.to(DEVICE)
    return model


if __name__ == "__main__":
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = load_model_fixed()
    print("Model loaded with weight remapping (strict=True) - SUCCESS")

    # Load test data
    t8_folder = os.path.join(DATA_ROOT, "point_net_transf_gat_8th_trial_lower_dropout")
    test_dl = torch.load(
        os.path.join(t8_folder, "data_created_during_training", "test_dl.pt"),
        weights_only=False,
    )
    print("Test data loaded:", len(test_dl), "graphs")

    # Quick deterministic test on 3 graphs
    model.eval()
    preds_all, targets_all = [], []
    N = 3
    with torch.no_grad():
        for gi in range(N):
            data = test_dl[gi].to(DEVICE)
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
            preds_all.append(out.squeeze().cpu().numpy())
            targets_all.append(data.y.squeeze().cpu().numpy())
            print("  Graph", gi + 1, "- nodes:", len(data.y))

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(preds - targets))

    print()
    print("QUICK TEST RESULTS (3 graphs, deterministic):")
    print("  R2  =", round(r2, 4))
    print("  MAE =", round(mae, 4))
    print()
    if r2 > 0.4:
        print("FIX VERIFIED: R2 > 0.4 confirms weight remapping works!")
    else:
        print("WARNING: R2 still low - fix may not be working correctly")
    print("DONE")
