"""
Experiment A only: 3 graphs, 5 runs, S=30 (matching standalone MC Dropout config).
Experiment B: 3 graphs, deterministic (already correct in minimal run).
Usage: conda run -n thesis-env python scripts/run_ensemble_final.py
"""

import os
import sys
import time
import json

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
import gc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "TR-C_Benchmarks")

MODEL_FOLDERS = {
    2: "point_net_transf_gat_2nd_try",
    5: "point_net_transf_gat_5th_try",
    6: "point_net_transf_gat_6th_trial_lower_lr",
    7: "point_net_transf_gat_7th_trial_80_10_10_split",
    8: "point_net_transf_gat_8th_trial_lower_dropout",
}
DROPOUT_MAP = {2: 0.3, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.2}
MODEL_WEIGHTS_R2 = {2: 0.5117, 5: 0.5553, 6: 0.5223, 7: 0.5471, 8: 0.5957}


def load_model_fixed(model_num):
    folder = os.path.join(DATA_ROOT, MODEL_FOLDERS[model_num])
    model_path = os.path.join(folder, "trained_model", "model.pth")
    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=DROPOUT_MAP[model_num],
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


def mc_dropout_predict(model, data, S=30):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    preds = []
    with torch.no_grad():
        for _ in range(S):
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze().cpu().numpy())
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


if __name__ == "__main__":
    print("Device:", DEVICE)
    t0 = time.time()

    # Load test data
    t8_folder = os.path.join(DATA_ROOT, "point_net_transf_gat_8th_trial_lower_dropout")
    print("Loading test data...")
    test_dl = torch.load(
        os.path.join(t8_folder, "data_created_during_training", "test_dl.pt"),
        weights_only=False,
    )
    print("Loaded %d graphs (%.1fs)" % (len(test_dl), time.time() - t0))

    N_GRAPHS = 3
    N_RUNS = 5
    S = 30

    # ========== EXPERIMENT A ==========
    print()
    print("EXPERIMENT A: MC Dropout vs Ensemble Variance (T8)")
    print("  graphs=%d, runs=%d, S=%d" % (N_GRAPHS, N_RUNS, S))

    model = load_model_fixed(8)
    print("  T8 loaded.")

    all_run_preds = []
    all_run_uncs = []
    targets = None

    for run in range(N_RUNS):
        t_run = time.time()
        torch.manual_seed(42 + run * 100)
        np.random.seed(42 + run * 100)
        run_preds, run_uncs, run_targets = [], [], []

        for gi in range(N_GRAPHS):
            data = test_dl[gi].to(DEVICE)
            mean_pred, unc = mc_dropout_predict(model, data, S)
            run_preds.append(mean_pred)
            run_uncs.append(unc)
            if run == 0:
                run_targets.append(data.y.squeeze().cpu().numpy())

        all_run_preds.append(np.concatenate(run_preds))
        all_run_uncs.append(np.concatenate(run_uncs))
        if run == 0:
            targets = np.concatenate(run_targets)
        print(
            "  Run %d/%d (%.1fs) [total %.1fs]"
            % (run + 1, N_RUNS, time.time() - t_run, time.time() - t0)
        )
        sys.stdout.flush()

    ensemble_preds = np.array(all_run_preds)
    mc_uncs = np.array(all_run_uncs)
    avg_mc_unc = mc_uncs.mean(axis=0)
    ens_variance = ensemble_preds.std(axis=0)
    combined_unc = np.sqrt(avg_mc_unc**2 + ens_variance**2)
    ens_mean_pred = ensemble_preds.mean(axis=0)

    abs_err = np.abs(ens_mean_pred - targets)
    ss_res = np.sum((targets - ens_mean_pred) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2_a = 1 - ss_res / ss_tot

    rho_mc = float(spearmanr(avg_mc_unc, abs_err)[0])
    rho_ens = float(spearmanr(ens_variance, abs_err)[0])
    rho_comb = float(spearmanr(combined_unc, abs_err)[0])

    print()
    print("  EXP A RESULTS (FINAL):")
    print(
        "    R2=%.4f MAE=%.4f RMSE=%.4f"
        % (r2_a, np.mean(abs_err), np.sqrt(np.mean((ens_mean_pred - targets) ** 2)))
    )
    print("    MC rho=%.4f  Ens rho=%.4f  Comb rho=%.4f" % (rho_mc, rho_ens, rho_comb))
    print("    MC unc mean=%.4f std=%.4f" % (avg_mc_unc.mean(), avg_mc_unc.std()))
    print("    Ens unc mean=%.4f std=%.4f" % (ens_variance.mean(), ens_variance.std()))
    print()
    sys.stdout.flush()

    res_a = {
        "config": {
            "model": 8,
            "n_runs": N_RUNS,
            "S": S,
            "n_graphs": N_GRAPHS,
            "n_nodes": int(len(targets)),
            "weight_remapping": True,
        },
        "prediction": {
            "r2": float(r2_a),
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean((ens_mean_pred - targets) ** 2))),
        },
        "mc_dropout": {
            "spearman_rho": rho_mc,
            "unc_mean": float(avg_mc_unc.mean()),
            "unc_std": float(avg_mc_unc.std()),
        },
        "ensemble_variance": {
            "spearman_rho": rho_ens,
            "unc_mean": float(ens_variance.mean()),
            "unc_std": float(ens_variance.std()),
        },
        "combined": {
            "spearman_rho": rho_comb,
            "unc_mean": float(combined_unc.mean()),
            "unc_std": float(combined_unc.std()),
        },
    }

    del model, all_run_preds, all_run_uncs, ensemble_preds, mc_uncs
    gc.collect()

    # ========== EXPERIMENT B ==========
    print("EXPERIMENT B: Multi-Model Ensemble")
    print("  graphs=%d" % N_GRAPHS)

    model_nums = [2, 5, 6, 7, 8]
    all_preds_b = {}
    targets_b = np.concatenate(
        [test_dl[gi].y.squeeze().cpu().numpy() for gi in range(N_GRAPHS)]
    )

    for m_num in model_nums:
        t_m = time.time()
        model = load_model_fixed(m_num)
        model.eval()
        preds_m = []
        with torch.no_grad():
            for gi in range(N_GRAPHS):
                data = test_dl[gi].to(DEVICE)
                out = model(data)
                if isinstance(out, tuple):
                    out = out[0]
                preds_m.append(out.squeeze().cpu().numpy())
        all_preds_b[m_num] = np.concatenate(preds_m)
        del model
        gc.collect()
        print("  T%d done (%.1fs)" % (m_num, time.time() - t_m))
        sys.stdout.flush()

    pred_stack = np.stack([all_preds_b[m] for m in model_nums], axis=0)
    weights = np.array([MODEL_WEIGHTS_R2[m] for m in model_nums])
    weights = weights / weights.sum()
    weighted_pred = np.average(pred_stack, axis=0, weights=weights)
    weighted_var = np.average(
        (pred_stack - weighted_pred) ** 2, axis=0, weights=weights
    )
    ens_unc = np.sqrt(weighted_var)

    abs_err_b = np.abs(weighted_pred - targets_b)
    ss_res_b = np.sum((targets_b - weighted_pred) ** 2)
    ss_tot_b = np.sum((targets_b - targets_b.mean()) ** 2)
    r2_b = 1 - ss_res_b / ss_tot_b
    rho_b = float(spearmanr(ens_unc, abs_err_b)[0])

    indiv = {}
    for m in model_nums:
        p = all_preds_b[m]
        ss_r = np.sum((targets_b - p) ** 2)
        indiv[m] = {
            "r2": float(1 - ss_r / ss_tot_b),
            "mae": float(np.mean(np.abs(p - targets_b))),
        }

    print()
    print("  EXP B RESULTS (FINAL):")
    for m in model_nums:
        print("    T%d: R2=%.4f MAE=%.4f" % (m, indiv[m]["r2"], indiv[m]["mae"]))
    print(
        "    Ensemble: R2=%.4f MAE=%.4f RMSE=%.4f rho=%.4f"
        % (
            r2_b,
            np.mean(abs_err_b),
            np.sqrt(np.mean((weighted_pred - targets_b) ** 2)),
            rho_b,
        )
    )
    print("    Ens unc mean=%.4f std=%.4f" % (ens_unc.mean(), ens_unc.std()))
    print()

    res_b = {
        "config": {
            "models": model_nums,
            "n_graphs": N_GRAPHS,
            "n_nodes": int(len(targets_b)),
            "weight_remapping": True,
            "weighted": True,
        },
        "individual": {str(m): indiv[m] for m in model_nums},
        "ensemble": {
            "r2": float(r2_b),
            "mae": float(np.mean(abs_err_b)),
            "rmse": float(np.sqrt(np.mean((weighted_pred - targets_b) ** 2))),
            "spearman_rho": rho_b,
            "unc_mean": float(ens_unc.mean()),
            "unc_std": float(ens_unc.std()),
        },
    }

    # Save
    out_dir = os.path.join(
        DATA_ROOT,
        "point_net_transf_gat_8th_trial_lower_dropout",
        "uq_results",
        "ensemble_experiments",
    )
    os.makedirs(out_dir, exist_ok=True)

    combined = {"experiment_a": res_a, "experiment_b": res_b}
    out_path = os.path.join(out_dir, "ensemble_fixed_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    elapsed = time.time() - t0
    print("=" * 60)
    print("Total: %.1fs (%.1f min)" % (elapsed, elapsed / 60))
    print("Saved:", out_path)
    print("DONE")
