"""Fig 5.15 — T7 Selective Prediction Curve (redesign v3).
Exact same style as approved Fig 5.8 (T8 version).
Full cross-check against verified markdown + JSON.
"""

import sys, os, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
T7_DIR = os.path.join(
    REPO,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_7th_trial_80_10_10_split",
    "uq_results",
)
MC_NPZ = os.path.join(T7_DIR, "mc_dropout_full_100graphs_mc30.npz")
DET_NPZ = os.path.join(T7_DIR, "deterministic_full_100graphs.npz")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading NPZ files ...")
mc = np.load(MC_NPZ)
det = np.load(DET_NPZ)
yhat_mc = mc["predictions"].astype(np.float64)
sigma = mc["uncertainties"].astype(np.float64)
y = mc["targets"].astype(np.float64)
yhat_det = det["predictions"].astype(np.float64)
n = len(y)
print(f"  n_nodes = {n:,}")

# ── Baselines ─────────────────────────────────────────────────────────────────
baseline_mc_mae = float(np.mean(np.abs(y - yhat_mc)))
baseline_mc_rmse = float(np.sqrt(np.mean((y - yhat_mc) ** 2)))
baseline_det_mae = float(np.mean(np.abs(y - yhat_det)))
baseline_det_rmse = float(np.sqrt(np.mean((y - yhat_det) ** 2)))

print(f"  MC  MAE={baseline_mc_mae:.4f}  RMSE={baseline_mc_rmse:.4f}")
print(f"  Det MAE={baseline_det_mae:.4f}  RMSE={baseline_det_rmse:.4f}")

# ── Selective prediction curve ────────────────────────────────────────────────
order = np.argsort(sigma)[::-1]
y_s = y[order]
yhat_s = yhat_mc[order]

RETENTION = [
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
for r in RETENTION:
    k = int(np.floor(r * n))
    if k == 0:
        continue
    sub_y = y_s[n - k :]
    sub_yhat = yhat_s[n - k :]
    m = float(np.mean(np.abs(sub_y - sub_yhat)))
    rm = float(np.sqrt(np.mean((sub_y - sub_yhat) ** 2)))
    pct = int(round(r * 100))
    rows.append({"retained_pct": pct, "MAE": round(m, 4), "RMSE": round(rm, 4)})

# ── FULL CROSS-CHECK ──────────────────────────────────────────────────────────
MD_REF = {
    100: (4.0737, 7.6202),
    95: (3.5597, 6.4348),
    90: (3.3156, 6.0181),
    85: (3.1526, 5.7530),
    80: (3.0313, 5.5703),
    75: (2.9309, 5.4218),
    70: (2.8423, 5.2910),
    60: (2.6777, 5.0391),
    50: (2.5134, 4.7997),
    40: (2.3331, 4.5399),
    30: (2.1204, 4.2421),
    25: (1.9913, 4.0711),
    10: (1.2314, 3.0780),
}
print(f"\n=== CROSS-CHECK (all 13 points) ===")
all_pass = True
for r in rows:
    ref_mae, ref_rmse = MD_REF[r["retained_pct"]]
    ok = abs(r["MAE"] - ref_mae) < 0.001 and abs(r["RMSE"] - ref_rmse) < 0.001
    if not ok:
        all_pass = False
    print(
        f"  {r['retained_pct']:>3d}%  MAE={r['MAE']:.4f}/{ref_mae:.4f}  "
        f"RMSE={r['RMSE']:.4f}/{ref_rmse:.4f}  {'OK' if ok else 'FAIL'}"
    )

REF_JSON = os.path.join(
    REPO, "docs", "verified", "phase3_results", "t7_error_detection.json"
)
with open(REF_JSON) as f:
    ref = json.load(f)
for pct_key, pct_val in [("retain_50pct", 50), ("retain_90pct", 90)]:
    ref_val = ref["selective_prediction"][pct_key]["mae"]
    comp = [r for r in rows if r["retained_pct"] == pct_val][0]["MAE"]
    ok = abs(comp - ref_val) < 0.001
    if not ok:
        all_pass = False
    print(f"  JSON {pct_val}%: {comp:.4f}/{ref_val:.4f} {'OK' if ok else 'FAIL'}")
ok = abs(baseline_mc_mae - ref["mae_veh_h"]) < 0.001
if not ok:
    all_pass = False
print(
    f"  JSON base: {baseline_mc_mae:.4f}/{ref['mae_veh_h']:.4f} {'OK' if ok else 'FAIL'}"
)
print(f"\n  >>> {'ALL CHECKS PASSED' if all_pass else 'SOME FAILED'} <<<")
if not all_pass:
    sys.exit(1)

# ── Compute reductions for milestones ─────────────────────────────────────────
x_arr = np.array([r["retained_pct"] for r in rows])
mae_arr = np.array([r["MAE"] for r in rows])
rmse_arr = np.array([r["RMSE"] for r in rows])

key_pcts = [90, 50, 25]
key_indices = {pct: int(np.where(x_arr == pct)[0][0]) for pct in key_pcts}

reductions_mae = {}
reductions_rmse = {}
for pct in key_pcts:
    idx = key_indices[pct]
    reductions_mae[str(pct)] = round(
        abs((mae_arr[idx] - baseline_mc_mae) / baseline_mc_mae * 100), 1
    )
    reductions_rmse[str(pct)] = round(
        abs((rmse_arr[idx] - baseline_mc_rmse) / baseline_mc_rmse * 100), 1
    )

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — exact same template as approved Fig 5.8
# ══════════════════════════════════════════════════════════════════════════════
LINE_COLOR = P_BLUE_DK
MARKER_COLOR = P_BLUE
DOT_COLOR = P_BLUE_LT
BASELINE_MC = P_BLUE_DK
BASELINE_DET = P_MGRAY
ANNOT_COLOR = P_BLUE_DK

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

panels = [
    (
        ax1,
        mae_arr,
        baseline_mc_mae,
        baseline_det_mae,
        "MAE (veh/h)",
        "MAE",
        "(a)",
        reductions_mae,
    ),
    (
        ax2,
        rmse_arr,
        baseline_mc_rmse,
        baseline_det_rmse,
        "RMSE (veh/h)",
        "RMSE",
        "(b)",
        reductions_rmse,
    ),
]

for ax, vals, base_mc, base_det, ylabel, mname, plabel, reductions in panels:
    panel_label(ax, plabel, x=-0.06, y=1.04)

    # Main curve — dark blue line
    ax.plot(
        x_arr,
        vals,
        "-",
        color=LINE_COLOR,
        lw=2.2,
        zorder=3,
        label="MC $\\sigma$ abstention",
    )

    # Small dots on all data points (subtle, light blue)
    ax.plot(x_arr, vals, "o", color=DOT_COLOR, ms=4, alpha=0.5, zorder=3)

    # Highlight key milestone points — medium blue, larger, white edge
    for pct in key_pcts:
        idx = key_indices[pct]
        ax.plot(
            x_arr[idx],
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

    # Annotations — BELOW the data points (same as Fig 5.8)
    # Annotations ABOVE the points, shifted up well clear of curve
    annot_configs = {
        90: {"xytext_offset": (0, 18), "va": "bottom"},
        50: {"xytext_offset": (0, 18), "va": "bottom"},
        25: {"xytext_offset": (0, 18), "va": "bottom"},
    }

    for pct in key_pcts:
        idx = key_indices[pct]
        val = vals[idx]
        red = reductions[str(pct)]
        ox, oy = annot_configs[pct]["xytext_offset"]

        ax.annotate(
            f"{pct}%: {val:.2f}  ($-${red:.1f}%)",
            xy=(x_arr[idx], val),
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
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")

fig.suptitle(
    "T7 Selective Prediction: Uncertainty-Based Abstention (MC Dropout, S=30)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

fig.tight_layout(pad=1.8, rect=[0, 0.02, 1, 0.94])

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(out_dir, f"t7_selective_prediction_curve.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {out}")
plt.close(fig)
print("Done.")
