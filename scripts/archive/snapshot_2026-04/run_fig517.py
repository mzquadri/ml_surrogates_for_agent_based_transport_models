"""
Regenerate Fig 5.17: t7_interval_width_comparison
Grouped bar chart: mean prediction interval width by method and coverage level.
Values computed from T7 NPZ files (20/80 cal/test split).
Verified against docs/verified/UQ_CALIBRATION_AUDIT_T7.md.
"""

import os, sys
import numpy as np
from scipy import stats

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

REPO = os.path.dirname(os.path.abspath(__file__))
NPZ_MC = os.path.join(
    REPO,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_7th_trial_80_10_10_split",
    "uq_results",
    "mc_dropout_full_100graphs_mc30.npz",
)

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), Trial 7 test set."
)
EPS = 1e-6

# ── Load NPZ data ───────────────────────────────────────────────────────
print("Loading T7 MC NPZ ...")
mc = np.load(NPZ_MC)
yhat_all = mc["predictions"].astype(np.float64)
sig_all = mc["uncertainties"].astype(np.float64)
y_all = mc["targets"].astype(np.float64)
n = len(y_all)
print(f"  Total nodes: {n:,}")

# 20/80 calibration/test split (same as T8 methodology)
split = n // 5
cal_yhat = yhat_all[:split]
cal_sig = sig_all[:split]
cal_y = y_all[:split]

test_yhat = yhat_all[split:]
test_sig = sig_all[split:]
test_y = y_all[split:]
print(f"  Cal: {len(cal_y):,}   Test: {len(test_y):,}")


def conformal_q(residuals, alpha):
    nn = residuals.shape[0]
    q_level = np.ceil((nn + 1) * (1 - alpha)) / nn
    q_level = min(q_level, 1.0)
    return np.quantile(residuals, q_level, method="higher")


r_cal_abs = np.abs(cal_y - cal_yhat)
r_cal_scaled = r_cal_abs / (cal_sig + EPS)

NOMINAL_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
results = []

print("\nComputing interval widths ...")
for nominal in NOMINAL_LEVELS:
    alpha = 1.0 - nominal
    z = float(stats.norm.ppf((1.0 + nominal) / 2.0))
    raw_wid = float(2.0 * z * np.mean(test_sig))

    q_global = float(conformal_q(r_cal_abs, alpha))
    wid_global = float(2.0 * q_global)

    q_adapt = float(conformal_q(r_cal_scaled, alpha))
    wid_adapt = float(2.0 * q_adapt * np.mean(test_sig + EPS))

    results.append(
        {
            "nominal": nominal,
            "raw_mc_width": round(raw_wid, 2),
            "global_width": round(wid_global, 2),
            "adapt_width": round(wid_adapt, 2),
        }
    )
    print(
        f"  {int(nominal * 100):2d}%: raw={raw_wid:.2f}  global={wid_global:.2f}  adapt={wid_adapt:.2f}"
    )

# ── Cross-checks vs audit ──────────────────────────────────────────────
print("\n-- Cross-checks vs UQ_CALIBRATION_AUDIT_T7.md --")
AUDIT = {
    50: {"raw": 1.63, "global": 3.60, "adapt": 4.96},
    70: {"raw": 2.51, "global": 7.84, "adapt": 9.17},
    80: {"raw": 3.10, "global": 12.13, "adapt": 14.28},
    90: {"raw": 3.98, "global": 20.78, "adapt": 25.40},
    95: {"raw": 4.75, "global": 31.30, "adapt": 39.22},
}

checks_passed = 0
checks_total = 0
for r in results:
    nom_pct = int(r["nominal"] * 100)
    a = AUDIT[nom_pct]
    for key_script, key_audit in [
        ("raw_mc_width", "raw"),
        ("global_width", "global"),
        ("adapt_width", "adapt"),
    ]:
        checks_total += 1
        diff = abs(r[key_script] - a[key_audit])
        ok = diff < 0.1
        if ok:
            checks_passed += 1
        status = "OK" if ok else "FAIL"
        print(
            f"  [{status}] {nom_pct}% {key_audit}: script={r[key_script]:.2f}  audit={a[key_audit]:.2f}  diff={diff:.3f}"
        )

print(f"\nCross-checks: {checks_passed}/{checks_total} PASSED")
assert checks_passed == checks_total, "SOME CROSS-CHECKS FAILED"

# ── Figure (matching Fig 5.10 template exactly) ────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(BG)

n_lev = len(results)
x_pos = np.arange(n_lev)
bar_w = 0.24

bars_raw = [r["raw_mc_width"] for r in results]
bars_glob = [r["global_width"] for r in results]
bars_adapt = [r["adapt_width"] for r in results]

b1 = ax.bar(
    x_pos - bar_w,
    bars_raw,
    bar_w,
    label="Raw MC Gaussian",
    color=P_CORAL,
    alpha=0.88,
    zorder=3,
)
b2 = ax.bar(
    x_pos,
    bars_glob,
    bar_w,
    label="Global conformal",
    color=P_BLUE,
    alpha=0.88,
    zorder=3,
)
b3 = ax.bar(
    x_pos + bar_w,
    bars_adapt,
    bar_w,
    label="Adaptive conformal",
    color=P_GREEN,
    alpha=0.88,
    zorder=3,
)

# Value labels on top of each bar
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.3,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=P_DGRAY,
        )

ax.set_xticks(x_pos)
ax.set_xticklabels(["50%", "70%", "80%", "90%", "95%"])
ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
ax.set_ylabel("Mean interval width (veh/h)", color=P_DGRAY)
ax.tick_params(colors=P_SLATE, labelsize=10)
ax.grid(True, axis="y", alpha=0.3, zorder=0)
ax.legend(fontsize=9, framealpha=0.9, loc="upper left")

# Add headroom for tallest bar labels
ax.set_ylim(0, max(bars_adapt) * 1.12)

fig.suptitle(
    "T7 Mean Prediction Interval Width by Method and Coverage Level (S=30)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
save_fig(fig, "t7_interval_width_comparison", out_dir=THESIS_FIG)
plt.close(fig)
print("\nSaved: t7_interval_width_comparison.pdf + .png")
