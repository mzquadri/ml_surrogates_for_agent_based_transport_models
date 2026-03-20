"""
Deterministic Inference Runner (Dropout OFF)
============================================
Runs deterministic inference on full test set.
Supports all model numbers (2-8).

Usage:
    python scripts/evaluation/run_deterministic_full.py --model 6 --cpu
"""
import os, sys, argparse, time, json, glob
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from gnn.models.point_net_transf_gat import PointNetTransfGAT

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

def get_dropout(model_num):
    """Model 8 uses dropout=0.2, others use 0.3."""
    return 0.2 if model_num == 8 else 0.3

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.asarray(x).squeeze().astype(np.float32)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res/ss_tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True, choices=[2,3,4,5,6,7,8], help="Model number")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-graphs", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    model_num = args.model
    MODEL_FOLDER = MODEL_FOLDERS[model_num]
    dropout = get_dropout(model_num)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_folder = os.path.join(REPO_ROOT, "data/TR-C_Benchmarks", MODEL_FOLDER)
    out_dir = os.path.join(model_folder, "uq_results")
    ckpt_dir = os.path.join(out_dir, "deterministic_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Model {model_num}: {MODEL_FOLDER}")
    print(f"Dropout: {dropout}")
    print(f"Device: {device}")

    test_dl_path = os.path.join(model_folder, "data_created_during_training", "test_dl.pt")
    test_set_dl = torch.load(test_dl_path, weights_only=False)
    n_total = len(test_set_dl)
    n = min(n_total, args.max_graphs) if args.max_graphs else n_total
    print(f"Test graphs: {n}/{n_total}")

    model = PointNetTransfGAT(
        in_channels=5, out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128,256,512],
        dropout=dropout, use_dropout=True, predict_mode_stats=False
    )
    model_path = os.path.join(model_folder, "trained_model", "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
    model = model.to(device)
    model.eval()  # deterministic (dropout OFF)

    completed = set()
    if not args.no_resume:
        for f in os.listdir(ckpt_dir):
            if f.startswith("graph_") and f.endswith(".npz"):
                completed.add(int(f.split("_")[1].split(".")[0]))
        completed = {i for i in completed if i < n}

    start = time.time()
    for i in tqdm(range(n), desc="Deterministic"):
        if i in completed:
            continue
        g = test_set_dl[i].to(device)
        with torch.inference_mode():
            yhat = model(g)
        pred = to_numpy(yhat)
        y = to_numpy(g.y)

        np.savez_compressed(os.path.join(ckpt_dir, f"graph_{i:04d}.npz"),
                            predictions=pred, targets=y)
        completed.add(i)

    # aggregate
    files = sorted(glob.glob(os.path.join(ckpt_dir, "graph_*.npz")))
    preds, ys = [], []
    for fp in files:
        d = np.load(fp)
        preds.append(d["predictions"])
        ys.append(d["targets"])
    yhat = np.concatenate(preds).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)

    r2 = r2_score(y, yhat)
    mae = np.mean(np.abs(y-yhat))
    rmse = np.sqrt(np.mean((y-yhat)**2))
    total_time = time.time() - start

    print("="*70)
    print(f"Deterministic results (Model {model_num}, graphs={len(files)})")
    print("="*70)
    print("R2:", float(r2))
    print("MAE:", float(mae))
    print("RMSE:", float(rmse))
    print("time_min:", total_time/60)

    out_npz = os.path.join(out_dir, f"deterministic_full_{len(files)}graphs.npz")
    np.savez_compressed(out_npz, predictions=yhat.astype(np.float32), targets=y.astype(np.float32))
    out_json = os.path.join(out_dir, f"deterministic_metrics_model{model_num}_{len(files)}graphs.json")
    with open(out_json, "w") as f:
        json.dump({"model": model_num, "r2": float(r2), "mae": float(mae), "rmse": float(rmse),
                   "graphs": int(len(files)), "nodes": int(len(y)),
                   "time_min": float(total_time/60)}, f, indent=2)
    print("Saved:", out_npz)
    print("Saved:", out_json)
    print("Saved:", out_npz)
    print("Saved:", out_json)

if __name__ == "__main__":
    main()
