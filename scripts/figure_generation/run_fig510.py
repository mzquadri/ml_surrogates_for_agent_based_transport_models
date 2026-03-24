"""
Regenerate Fig 5.10: t8_interval_width_comparison
Grouped bar chart: mean prediction interval width by method and coverage level.
Values computed from trial8_uq_ablation_results.csv (20/80 cal/test split).
Verified against docs/verified/UQ_CALIBRATION_AUDIT_T8.md.
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CSV_PATH = os.path.join(
    REPO,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_8th_trial_lower_dropout",
    "trial8_uq_ablation_results.csv",
)

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), Trial 8 test set."
)
EPS = 1e-6

# ── Load and split data ─────────────────────────────────────────────────
print("Loading CSV ...")
df = pd.read_csv(CSV_PATH)
n = len(df)
split = n // 5  # 20% calibration
cal = df.iloc[:split]
test = df.iloc[split:]
print(f"  Cal: {len(cal):,}   Test: {len(test):,}")

yhat_c = cal["pred_mc_mean"].values.astype(np.float64)
sig_c = cal["pred_mc_std"].values.astype(np.float64)
y_c = cal["target"].values.astype(np.float64)

yhat_t = test["pred_mc_mean"].values.astype(np.float64)
sig_t = test["pred_mc_std"].values.astype(np.float64)
y_t = test["target"].values.astype(np.float64)


def conformal_q(residuals, alpha):
    nn = residuals.shape[0]
    q_level = np.ceil((nn + 1) * (1 - alpha)) / nn
    q_level = min(q_level, 1.0)
    return np.quantile(residuals, q_level, method="higher")


r_cal_abs = np.abs(y_c - yhat_c)
r_cal_scaled = r_cal_abs / (sig_c + EPS)
r_test_abs = np.abs(y_t - yhat_t)

NOMINAL_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
results = []

print("\nComputing interval widths ...")
for nominal in NOMINAL_LEVELS:
    alpha = 1.0 - nominal
    z = float(stats.norm.ppf((1.0 + nominal) / 2.0))
    raw_wid = float(2.0 * z * np.mean(sig_t))

    q_global = float(conformal_q(r_cal_abs, alpha))
    wid_global = float(2.0 * q_global)

    q_adapt = float(conformal_q(r_cal_scaled, alpha))
    wid_adapt = float(2.0 * q_adapt * np.mean(sig_t + EPS))

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

# ── Cross-checks ────────────────────────────────────────────────────────
print("\n-- Cross-checks vs audit --")
r90 = results[3]
r95 = results[4]
assert abs(r90["global_width"] - 19.99) < 0.1, (
    f"FAIL: global_w@90%={r90['global_width']}"
)
assert abs(r95["global_width"] - 29.54) < 0.2, (
    f"FAIL: global_w@95%={r95['global_width']}"
)
print(f"  [OK] global_width@90% = {r90['global_width']}")
print(f"  [OK] global_width@95% = {r95['global_width']}")

# ── Figure ──────────────────────────────────────────────────────────────
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

# Add some headroom for the tallest bar labels
ax.set_ylim(0, max(bars_adapt) * 1.12)

fig.suptitle(
    "T8 Mean Prediction Interval Width by Method and Coverage Level (S=30)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=8, color=P_MGRAY, style="italic")

fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────────
THESIS_FIG = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(THESIS_FIG, f"t8_interval_width_comparison.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved: t8_interval_width_comparison.pdf + .png")
