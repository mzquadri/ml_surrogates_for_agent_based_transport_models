"""
Regenerate Fig 5.8: t8_selective_prediction_curve
Selective prediction (risk-coverage) curve — MAE and RMSE panels.
Values from verified JSON: selective_prediction_s30.json
"""

import os, sys, json
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "thesis",
        "latex_tum_official",
        "figures",
    ),
)
from thesis_style import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR))

# ── Load verified JSON ──────────────────────────────────────────────────
JSON_PATH = os.path.join(
    REPO, "docs", "verified", "phase3_results", "selective_prediction_s30.json"
)
with open(JSON_PATH, "r") as f:
    data = json.load(f)

rows = data["retention_table"]
baseline_mc_mae = data["baseline_mc_mae"]
baseline_mc_rmse = data["baseline_mc_rmse"]
baseline_det_mae = data["baseline_det_mae"]
baseline_det_rmse = data["baseline_det_rmse"]
key_red = data["key_reductions"]

x = np.array([r["retained_pct"] for r in rows])
mae_vals = np.array([r["MAE"] for r in rows])
rmse_vals = np.array([r["RMSE"] for r in rows])

# Key milestone indices
key_pcts = [90, 50, 25]
key_indices = {pct: np.where(x == pct)[0][0] for pct in key_pcts}

# ── Figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), Trial 8 test set."
)

# Monochromatic dark-blue scheme
LINE_COLOR = P_BLUE_DK  # dark blue for both curve lines
MARKER_COLOR = P_BLUE  # medium blue for milestone markers
DOT_COLOR = P_BLUE_LT  # light blue for small data-point dots
BASELINE_MC = P_BLUE_DK  # dark blue dashed for MC baseline
BASELINE_DET = P_MGRAY  # gray dotted for deterministic baseline
ANNOT_COLOR = P_BLUE_DK  # dark blue annotation text

panels = [
    (
        ax1,
        mae_vals,
        baseline_mc_mae,
        baseline_det_mae,
        "MAE (veh/h)",
        "MAE",
        "(a)",
        {
            "90": key_red["retain_90pct"]["mae_reduction_pct"],
            "50": key_red["retain_50pct"]["mae_reduction_pct"],
            "25": key_red["retain_25pct"]["mae_reduction_pct"],
        },
    ),
    (
        ax2,
        rmse_vals,
        baseline_mc_rmse,
        baseline_det_rmse,
        "RMSE (veh/h)",
        "RMSE",
        "(b)",
        {
            "90": key_red["retain_90pct"]["rmse_reduction_pct"],
            "50": key_red["retain_50pct"]["rmse_reduction_pct"],
            "25": key_red["retain_25pct"]["rmse_reduction_pct"],
        },
    ),
]

for ax, vals, base_mc, base_det, ylabel, mname, plabel, reductions in panels:
    panel_label(ax, plabel, x=-0.06, y=1.04)

    # Main curve — dark blue line
    ax.plot(
        x,
        vals,
        "-",
        color=LINE_COLOR,
        lw=2.2,
        zorder=3,
        label="MC $\\sigma$ abstention",
    )

    # Small dots on all data points (subtle, light blue)
    ax.plot(x, vals, "o", color=DOT_COLOR, ms=4, alpha=0.5, zorder=3)

    # Highlight key milestone points — medium blue, larger markers
    for pct in key_pcts:
        idx = key_indices[pct]
        ax.plot(
            x[idx],
            vals[idx],
            "o",
            color=MARKER_COLOR,
            ms=9,
            zorder=5,
            markeredgecolor=WHITE,
            markeredgewidth=1.5,
        )

    # MC baseline (100%) — dark blue dashed
    ax.axhline(
        base_mc,
        color=BASELINE_MC,
        lw=1.2,
        ls="--",
        alpha=0.45,
        label=f"MC baseline 100% ({base_mc:.2f})",
        zorder=2,
    )

    # Deterministic baseline — gray dotted
    ax.axhline(
        base_det,
        color=BASELINE_DET,
        lw=1.0,
        ls=":",
        label=f"Deterministic baseline ({base_det:.2f})",
        zorder=2,
    )

    # Annotations — placed carefully to avoid overlap
    annot_configs = {
        90: {"xytext_offset": (-18, -14), "va": "top"},
        50: {"xytext_offset": (-18, -14), "va": "top"},
        25: {"xytext_offset": (-15, -14), "va": "top"},
    }

    for pct in key_pcts:
        idx = key_indices[pct]
        val = vals[idx]
        red = reductions[str(pct)]
        ox, oy = annot_configs[pct]["xytext_offset"]

        ax.annotate(
            f"{pct}%: {val:.2f}  ($-${red:.1f}%)",
            xy=(x[idx], val),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="semibold",
            color=ANNOT_COLOR,
            ha="center",
            va=annot_configs[pct]["va"],
            arrowprops=dict(arrowstyle="-", color=P_LGRAY, lw=0.8),
        )

    ax.set_xlabel("Retained predictions (%)", color=P_DGRAY)
    ax.set_ylabel(ylabel, color=P_DGRAY)
    ax.set_xlim(5, 105)
    ax.set_xticks([10, 25, 50, 70, 80, 90, 95, 100])
    ax.tick_params(colors=P_SLATE)
    ax.grid(True, alpha=0.3, zorder=0)

    # Legend — lower right so it doesn't cover baselines
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")

# Suptitle
fig.suptitle(
    "T8 Selective Prediction: Uncertainty-Based Abstention (MC Dropout, S=30)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

# Footnote
fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=1.8, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(THESIS_FIG, f"t8_selective_prediction_curve.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: t8_selective_prediction_curve.pdf + .png")

# ── Print verified values ───────────────────────────────────────────────
print("\n=== Verified values used ===")
print(f"Baseline MC:  MAE={baseline_mc_mae}, RMSE={baseline_mc_rmse}")
print(f"Baseline Det: MAE={baseline_det_mae}, RMSE={baseline_det_rmse}")
for pct in [90, 50, 25]:
    k = f"retain_{pct}pct"
    r = key_red[k]
    print(
        f"  Retain {pct}%: MAE={r['mae']}, -{r['mae_reduction_pct']}% | RMSE={r['rmse']}, -{r['rmse_reduction_pct']}%"
    )
