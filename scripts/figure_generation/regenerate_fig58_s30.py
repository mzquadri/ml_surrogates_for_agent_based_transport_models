"""
Phase 1b: Regenerate selective prediction figure (Figure 5.8) from S=30 NPZ data.

Previous version used trial8_uq_ablation_results.csv (generated with S=50).
This version uses mc_dropout_full_100graphs_mc30.npz (genuine S=30 results).

Output:
  - docs/verified/figures/t8_selective_prediction_curve.pdf
  - docs/verified/figures/t8_selective_prediction_curve.png
  - thesis/latex_tum_official/figures/t8_selective_prediction_curve.pdf  (copy)
  - docs/verified/phase3_results/selective_prediction_s30.json  (verified numbers)
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
T8_UQ = (
    REPO
    / "data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results"
)
MC_NPZ = T8_UQ / "mc_dropout_full_100graphs_mc30.npz"
DET_NPZ = T8_UQ / "deterministic_full_100graphs.npz"
OUT_FIG = REPO / "docs/verified/figures"
THESIS_FIG = REPO / "thesis/latex_tum_official/figures"
RESULTS_DIR = REPO / "docs/verified/phase3_results"

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__)),
)
from thesis_style import *

FOOTNOTE = (
    "Results based on 100 test graphs (3,163,500 nodes), Trial 8, "
    "MC Dropout S\u2009=\u200930 forward passes."
)


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load NPZ data ─────────────────────────────────────────────────────────
    print("Loading MC Dropout S=30 NPZ ...")
    mc = np.load(MC_NPZ)
    mc_mean = mc["predictions"]  # MC mean predictions (S=30)
    mc_std = mc["uncertainties"]  # MC std (S=30)
    targets = mc["targets"]
    n_total = len(targets)
    print(f"  MC NPZ loaded: {n_total:,} nodes")

    print("Loading deterministic NPZ ...")
    det = np.load(DET_NPZ)
    det_pred = det["predictions"]
    det_targets = det["targets"]
    assert len(det_pred) == n_total, (
        f"Size mismatch: det={len(det_pred)} vs mc={n_total}"
    )
    assert np.allclose(targets, det_targets, atol=1e-5), (
        "Target mismatch between NPZ files"
    )
    print(f"  Deterministic NPZ loaded: {len(det_pred):,} nodes")

    # ── Sort by uncertainty (descending) ──────────────────────────────────────
    sort_idx = np.argsort(-mc_std)  # descending by sigma
    mc_mean_sorted = mc_mean[sort_idx]
    mc_std_sorted = mc_std[sort_idx]
    targets_sorted = targets[sort_idx]

    # ── Selective prediction at various retention levels ──────────────────────
    retention_levels = [
        1.00,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.60,
        0.50,
        0.40,
        0.30,
        0.25,
        0.10,
    ]
    rows = []

    for r in retention_levels:
        k = int(np.floor(r * n_total))
        if k == 0:
            continue
        # Keep the BOTTOM k rows = the least uncertain nodes
        sub_tgt = targets_sorted[n_total - k :]
        sub_pred = mc_mean_sorted[n_total - k :]
        mae = float(np.mean(np.abs(sub_tgt - sub_pred)))
        rmse = float(np.sqrt(np.mean((sub_tgt - sub_pred) ** 2)))
        rows.append(
            {
                "retained_pct": int(round(r * 100)),
                "n_nodes": k,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
            }
        )
        print(f"  Retain {int(r * 100):3d}%  n={k:>9,}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ── Baselines ─────────────────────────────────────────────────────────────
    mae_base_mc = float(np.mean(np.abs(targets - mc_mean)))
    rmse_base_mc = float(np.sqrt(np.mean((targets - mc_mean) ** 2)))
    mae_base_det = float(np.mean(np.abs(targets - det_pred)))
    rmse_base_det = float(np.sqrt(np.mean((targets - det_pred) ** 2)))
    print(f"\n  Baseline MC   (100%): MAE={mae_base_mc:.4f}  RMSE={rmse_base_mc:.4f}")
    print(f"  Baseline det  (100%): MAE={mae_base_det:.4f}  RMSE={rmse_base_det:.4f}")

    # ── Build arrays for plotting ─────────────────────────────────────────────
    x = np.array([r["retained_pct"] for r in rows])
    mae_vals = np.array([r["MAE"] for r in rows])
    rmse_vals = np.array([r["RMSE"] for r in rows])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Trial 8: Selective Prediction: Uncertainty-Based Abstention\n"
        "(MC Dropout, 30 samples, GATConv final layer)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        y=0.97,
    )
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")

    for ax, metric_vals, color, base_val, ylabel, metric_name in zip(
        axes,
        [mae_vals, rmse_vals],
        [P_BLUE, P_CORAL],
        [mae_base_mc, rmse_base_mc],
        ["MAE (veh/h)", "RMSE (veh/h)"],
        ["MAE", "RMSE"],
    ):
        ax.set_facecolor(BG)

        # Main curve
        ax.plot(
            x,
            metric_vals,
            "-o",
            color=color,
            lw=2.2,
            ms=6,
            zorder=4,
            label=f"MC \u03c3 abstention",
        )

        # Baseline MC (100% retained)
        ax.axhline(
            base_val,
            color=color,
            lw=1.2,
            linestyle="--",
            alpha=0.55,
            label=f"MC baseline 100% ({base_val:.2f})",
        )

        # Baseline deterministic
        base_d = mae_base_det if metric_name == "MAE" else rmse_base_det
        ax.axhline(
            base_d,
            color=P_MGRAY,
            lw=1.0,
            linestyle=":",
            label=f"Deterministic baseline ({base_d:.2f})",
        )

        # Annotate key retention points
        for pct in [90, 50, 25]:
            idx = np.where(x == pct)[0]
            if len(idx) > 0:
                val = metric_vals[idx[0]]
                reduction = (1 - val / base_val) * 100
                ax.annotate(
                    f"{pct}%\n{val:.2f}\n(\u2212{reduction:.1f}%)",
                    xy=(pct, val),
                    xytext=(
                        pct - 12,
                        val - (metric_vals.max() - metric_vals.min()) * 0.15,
                    ),
                    fontsize=7.0,
                    color=color,
                    ha="center",
                    arrowprops=dict(arrowstyle="-", color=P_LGRAY, lw=0.8),
                )

        ax.set_xlabel("Retained predictions (%)", fontsize=10, color=P_SLATE)
        ax.set_ylabel(ylabel, fontsize=10, color=P_SLATE)
        ax.set_xlim(5, 107)
        ax.set_xticks([10, 25, 50, 70, 80, 90, 95, 100])
        ax.tick_params(colors=P_MGRAY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(P_LGRAY)
        ax.legend(fontsize=7.5, framealpha=0.85, loc="upper left")
        ax.grid(True, color=P_LGRAY, lw=0.5, alpha=0.7)

    fig.text(
        0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14, wspace=0.30)

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"t8_selective_prediction_curve.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
        print(f"  Saved: {out}")
    plt.close(fig)

    # ── Copy PDF to thesis figures directory ──────────────────────────────────
    src = OUT_FIG / "t8_selective_prediction_curve.pdf"
    dst = THESIS_FIG / "t8_selective_prediction_curve.pdf"
    shutil.copy2(src, dst)
    print(f"  Copied to: {dst}")

    # ── Save verified results JSON ────────────────────────────────────────────
    # Extract key retention points for JSON
    def get_at(pct, col):
        for r in rows:
            if r["retained_pct"] == pct:
                return r[col]
        return None

    results = {
        "description": "Selective prediction (risk-coverage) from S=30 MC Dropout NPZ",
        "data_source": str(MC_NPZ),
        "deterministic_source": str(DET_NPZ),
        "mc_samples": 30,
        "n_total_nodes": n_total,
        "n_graphs": 100,
        "baseline_mc_mae": round(mae_base_mc, 4),
        "baseline_mc_rmse": round(rmse_base_mc, 4),
        "baseline_det_mae": round(mae_base_det, 4),
        "baseline_det_rmse": round(rmse_base_det, 4),
        "retention_table": rows,
        "key_reductions": {
            "retain_90pct": {
                "mae": get_at(90, "MAE"),
                "rmse": get_at(90, "RMSE"),
                "mae_reduction_pct": round(
                    (1 - get_at(90, "MAE") / mae_base_mc) * 100, 1
                ),
                "rmse_reduction_pct": round(
                    (1 - get_at(90, "RMSE") / rmse_base_mc) * 100, 1
                ),
            },
            "retain_50pct": {
                "mae": get_at(50, "MAE"),
                "rmse": get_at(50, "RMSE"),
                "mae_reduction_pct": round(
                    (1 - get_at(50, "MAE") / mae_base_mc) * 100, 1
                ),
                "rmse_reduction_pct": round(
                    (1 - get_at(50, "RMSE") / rmse_base_mc) * 100, 1
                ),
            },
            "retain_25pct": {
                "mae": get_at(25, "MAE"),
                "rmse": get_at(25, "RMSE"),
                "mae_reduction_pct": round(
                    (1 - get_at(25, "MAE") / mae_base_mc) * 100, 1
                ),
                "rmse_reduction_pct": round(
                    (1 - get_at(25, "RMSE") / rmse_base_mc) * 100, 1
                ),
            },
        },
    }

    json_out = RESULTS_DIR / "selective_prediction_s30.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_out}")

    print("\nDone. Figure regenerated from S=30 NPZ data.")


if __name__ == "__main__":
    main()
