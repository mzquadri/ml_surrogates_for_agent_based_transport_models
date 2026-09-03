"""
Verify all 8 trained model checkpoint files (.pth) for the thesis project.
For each file: load, report keys, parameter count, file size, and status.
For Trial 7 and Trial 8: print full layer names and tensor shapes.
"""

import os
import sys

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS = os.path.join(BASE, "data", "TR-C_Benchmarks")

TRIALS = {
    1: os.path.join(
        BENCHMARKS, "pointnet_transf_gat_1st_bs32_5feat_seed42", "model.pth"
    ),
    2: os.path.join(
        BENCHMARKS, "point_net_transf_gat_2nd_try", "trained_model", "model.pth"
    ),
    3: os.path.join(
        BENCHMARKS,
        "point_net_transf_gat_3rd_trial_weighted_loss",
        "trained_model",
        "model.pth",
    ),
    4: os.path.join(
        BENCHMARKS,
        "point_net_transf_gat_4th_trial_weighted_loss",
        "trained_model",
        "model.pth",
    ),
    5: os.path.join(
        BENCHMARKS, "point_net_transf_gat_5th_try", "trained_model", "model.pth"
    ),
    6: os.path.join(
        BENCHMARKS,
        "point_net_transf_gat_6th_trial_lower_lr",
        "trained_model",
        "model.pth",
    ),
    7: os.path.join(
        BENCHMARKS,
        "point_net_transf_gat_7th_trial_80_10_10_split",
        "trained_model",
        "model.pth",
    ),
    8: os.path.join(
        BENCHMARKS,
        "point_net_transf_gat_8th_trial_lower_dropout",
        "trained_model",
        "model.pth",
    ),
}

# ── try importing torch, fall back to pickle ──────────────────────────────────
try:
    import torch

    HAS_TORCH = True
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    HAS_TORCH = False
    import pickle

    print("WARNING: torch not available, falling back to pickle for basic load check.")

print(f"Python version: {sys.version}")
print("=" * 90)


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TB"


def count_params(state_dict: dict) -> int:
    """Count total number of scalar parameters in a state_dict."""
    total = 0
    for v in state_dict.values():
        if HAS_TORCH:
            total += v.numel()
        else:
            total += 1  # can't count without torch
    return total


def load_checkpoint(path: str):
    """Load a .pth file and return (obj, error_string | None)."""
    if HAS_TORCH:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            return obj, None
        except Exception as e1:
            # try weights_only=True as fallback
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True)
                return obj, None
            except Exception as e2:
                return None, f"torch.load failed: {e1} / weights_only also failed: {e2}"
    else:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj, None
        except Exception as e:
            return None, f"pickle.load failed: {e}"


def describe_checkpoint(obj) -> dict:
    """Return a dict describing the loaded checkpoint object."""
    info = {"type": type(obj).__name__}

    # If it's a dict, check for common checkpoint keys
    if isinstance(obj, dict):
        info["top_keys"] = list(obj.keys())
        # Determine where the state_dict lives
        if "model_state_dict" in obj:
            sd = obj["model_state_dict"]
            info["state_dict_location"] = "obj['model_state_dict']"
        elif "state_dict" in obj:
            sd = obj["state_dict"]
            info["state_dict_location"] = "obj['state_dict']"
        elif all(
            isinstance(v, (torch.Tensor if HAS_TORCH else type(None)))
            for v in obj.values()
        ):
            sd = obj
            info["state_dict_location"] = "obj (root is state_dict)"
        else:
            sd = None
            info["state_dict_location"] = "NOT FOUND"

        if sd is not None and isinstance(sd, dict):
            info["num_layers"] = len(sd)
            info["param_count"] = count_params(sd)
            info["layer_names"] = list(sd.keys())
            if HAS_TORCH:
                info["layer_shapes"] = {k: tuple(v.shape) for k, v in sd.items()}
    else:
        info["note"] = "Not a dict — unexpected format"

    return info


# ── main loop ──────────────────────────────────────────────────────────────────
all_ok = True
for trial_num in sorted(TRIALS.keys()):
    path = TRIALS[trial_num]
    tag = f"Trial {trial_num}"
    highlight = trial_num in (7, 8)

    print(f"\n{'#' * 90}")
    print(f"## {tag}{'  *** PRIORITY ***' if highlight else ''}")
    print(f"   Path: {os.path.relpath(path, BASE)}")

    # File existence & size
    if not os.path.isfile(path):
        print(f"   STATUS: FILE NOT FOUND")
        all_ok = False
        continue

    fsize = os.path.getsize(path)
    print(f"   File size: {human_size(fsize)} ({fsize:,} bytes)")

    # Load
    obj, err = load_checkpoint(path)
    if err:
        print(f"   STATUS: LOAD FAILED — {err}")
        all_ok = False
        continue

    print(f"   STATUS: LOADED SUCCESSFULLY")

    # Describe
    info = describe_checkpoint(obj)
    print(f"   Object type: {info['type']}")

    if "top_keys" in info:
        top = info["top_keys"]
        print(f"   Top-level keys ({len(top)}): {top}")

    if "state_dict_location" in info:
        print(f"   State dict at: {info['state_dict_location']}")

    if "num_layers" in info:
        print(f"   Number of parameter tensors: {info['num_layers']}")

    if "param_count" in info:
        print(f"   Total parameters: {info['param_count']:,}")

    # For T7 and T8: full layer details
    if highlight and "layer_names" in info and "layer_shapes" in info:
        print(f"\n   --- Full layer details for {tag} ---")
        shapes = info["layer_shapes"]
        for i, name in enumerate(info["layer_names"], 1):
            shape = shapes[name]
            numel = 1
            for s in shape:
                numel *= s
            print(f"   {i:3d}. {name:60s} shape={str(shape):20s} params={numel:>10,}")

print("\n" + "=" * 90)
if all_ok:
    print("ALL 8 CHECKPOINTS VERIFIED SUCCESSFULLY.")
else:
    print("SOME CHECKPOINTS HAD ISSUES — see details above.")
print("=" * 90)
