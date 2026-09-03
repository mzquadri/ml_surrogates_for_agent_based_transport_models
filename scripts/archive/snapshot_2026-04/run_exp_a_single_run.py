"""
Phase 2: Experiment A -- Single MC Dropout run on T8 (100 graphs, S=30).
Processes one run at a time with per-graph checkpointing for resume.

Usage:
  conda run -n thesis-env python scripts/run_exp_a_single_run.py --run-index 0
  conda run -n thesis-env python scripts/run_exp_a_single_run.py --run-index 1
  ... (repeat for 0-4)
"""

import os, sys, time, json, argparse

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
OUT_DIR = os.path.join(T8_FOLDER, "uq_results", "ensemble_experiments")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "exp_a_checkpoints")

S = 30  # MC Dropout samples per forward pass


def load_model_fixed():
    """Load T8 model with GATConv weight remapping fix + strict=True."""
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
    model_path = os.path.join(T8_FOLDER, "trained_model", "model.pth")
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


def mc_dropout_predict(model, data, s=30):
    """MC Dropout inference: s forward passes with dropout ON."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    preds = []
    with torch.no_grad():
        for _ in range(s):
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze().cpu().numpy())
    preds = np.array(preds)  # (s, n_nodes)
    return preds.mean(axis=0), preds.std(axis=0)


def get_checkpoint_path(run_index):
    """Path for per-graph checkpoint directory for this run."""
    return os.path.join(CHECKPOINT_DIR, "run_%d" % run_index)


def get_completed_graphs(run_index):
    """Check which graphs are already completed for this run."""
    ckpt_dir = get_checkpoint_path(run_index)
    if not os.path.exists(ckpt_dir):
        return set()
    completed = set()
    for f in os.listdir(ckpt_dir):
        if f.startswith("graph_") and f.endswith(".npz"):
            idx = int(f.replace("graph_", "").replace(".npz", ""))
            completed.add(idx)
    return completed


def save_graph_checkpoint(run_index, graph_index, predictions, uncertainties, targets):
    """Save per-graph checkpoint."""
    ckpt_dir = get_checkpoint_path(run_index)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "graph_%04d.npz" % graph_index)
    np.savez_compressed(
        path,
        predictions=predictions.astype(np.float32),
        uncertainties=uncertainties.astype(np.float32),
        targets=targets.astype(np.float32),
    )


def run_single(run_index, test_dl):
    """Run a single MC Dropout run on all graphs with checkpointing."""
    seed = 42 + run_index * 100
    n_graphs = len(test_dl)

    print("Run %d: seed=%d, S=%d, graphs=%d" % (run_index, seed, S, n_graphs))

    # Check for existing checkpoints (resume support)
    completed = get_completed_graphs(run_index)
    if len(completed) == n_graphs:
        print(
            "  Run %d already fully completed (%d graphs). Skipping."
            % (run_index, n_graphs)
        )
        return True
    if completed:
        print("  Resuming: %d/%d graphs already done" % (len(completed), n_graphs))

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model
    model = load_model_fixed()
    print("  T8 model loaded (strict=True)")

    # Process each graph
    t_run = time.time()
    for gi in range(n_graphs):
        if gi in completed:
            continue

        t_g = time.time()
        data = test_dl[gi].to(DEVICE)
        mean_pred, unc = mc_dropout_predict(model, data, S)
        targets = data.y.squeeze().cpu().numpy()

        save_graph_checkpoint(run_index, gi, mean_pred, unc, targets)

        elapsed_g = time.time() - t_g
        elapsed_total = time.time() - t_run
        remaining = (
            elapsed_total
            / (gi - min(completed, default={-1}).pop() + 1 if not completed else gi + 1)
            * (n_graphs - gi - 1)
            if gi > 0
            else 0
        )

        if (gi + 1) % 10 == 0 or gi == n_graphs - 1:
            print(
                "  Graph %d/%d (%.1fs/graph, total %.1fs, ~%.0fs remaining)"
                % (gi + 1, n_graphs, elapsed_g, elapsed_total, remaining)
            )
            sys.stdout.flush()

    print(
        "  Run %d complete. Total: %.1fs (%.1f min)"
        % (run_index, time.time() - t_run, (time.time() - t_run) / 60)
    )
    return True


def assemble_run(run_index, n_graphs):
    """Assemble per-graph checkpoints into a single NPZ for this run."""
    ckpt_dir = get_checkpoint_path(run_index)
    all_preds, all_uncs, all_targets = [], [], []

    for gi in range(n_graphs):
        path = os.path.join(ckpt_dir, "graph_%04d.npz" % gi)
        data = np.load(path)
        all_preds.append(data["predictions"])
        all_uncs.append(data["uncertainties"])
        if run_index == 0:  # Only need targets once
            all_targets.append(data["targets"])

    predictions = np.concatenate(all_preds)
    uncertainties = np.concatenate(all_uncs)

    out_path = os.path.join(OUT_DIR, "exp_a_run_%d.npz" % run_index)
    save_dict = {
        "predictions": predictions.astype(np.float32),
        "uncertainties": uncertainties.astype(np.float32),
    }
    if run_index == 0:
        save_dict["targets"] = np.concatenate(all_targets).astype(np.float32)

    np.savez_compressed(out_path, **save_dict)
    print("  Assembled: %s (%d nodes)" % (out_path, len(predictions)))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment A: single MC Dropout run")
    parser.add_argument("--run-index", type=int, required=True, help="Run index (0-4)")
    args = parser.parse_args()

    assert 0 <= args.run_index <= 4, "run-index must be 0-4"

    print("=" * 60)
    print("EXPERIMENT A: MC Dropout Run %d/5 (T8, S=%d)" % (args.run_index, S))
    print("  Device:", DEVICE)
    print("=" * 60)
    t0 = time.time()

    # Load test data
    print("\nLoading test data...")
    test_dl = torch.load(
        os.path.join(T8_FOLDER, "data_created_during_training", "test_dl.pt"),
        weights_only=False,
    )
    n_graphs = len(test_dl)
    print("  %d test graphs loaded (%.1fs)" % (n_graphs, time.time() - t0))

    # Run
    print()
    run_single(args.run_index, test_dl)

    # Assemble into single NPZ
    print("\nAssembling run %d..." % args.run_index)
    assemble_run(args.run_index, n_graphs)

    elapsed = time.time() - t0
    print("\nTotal time: %.1fs (%.1f min)" % (elapsed, elapsed / 60))
    print("DONE")
