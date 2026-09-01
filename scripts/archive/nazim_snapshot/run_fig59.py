"""
Regenerate Fig 5.9: t8_calibration_curve
Two panels: (a) Calibration curve — nominal vs achieved coverage
            (b) Empirical k vs Gaussian z — sigma multiplier comparison
Values computed from trial8_uq_ablation_results.csv (20/80 cal/test split).
Verified against docs/verified/UQ_CALIBRATION_AUDIT_T8.md.
"""

import os, sys
import numpy as np
import pandas as pd
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
    """Standard conformal quantile: ceil((n+1)*(1-alpha))/n"""
    nn = residuals.shape[0]
    q_level = np.ceil((nn + 1) * (1 - alpha)) / nn
    q_level = min(q_level, 1.0)
    return np.quantile(residuals, q_level, method="higher")


# Calibration residuals
r_cal_abs = np.abs(y_c - yhat_c)
r_cal_scaled = r_cal_abs / (sig_c + EPS)

# Test residuals
r_test_abs = np.abs(y_t - yhat_t)
r_test_scaled = r_test_abs / (sig_t + EPS)

NOMINAL_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
results = []

print("\nComputing calibration audit ...")
for nominal in NOMINAL_LEVELS:
    alpha = 1.0 - nominal

    # Raw MC Gaussian
    z = float(stats.norm.ppf((1.0 + nominal) / 2.0))
    raw_cov = float(np.mean(r_test_abs <= z * sig_t))

    # Empirical k
    k_emp = float(np.quantile(r_test_scaled, nominal))

    # Global conformal
    q_global = float(conformal_q(r_cal_abs, alpha))
    cov_global = float(np.mean(r_test_abs <= q_global))

    # Adaptive conformal
    q_adapt = float(conformal_q(r_cal_scaled, alpha))
    cov_adapt = float(np.mean(r_test_abs <= q_adapt * (sig_t + EPS)))

    results.append(
        {
            "nominal": nominal,
            "z_gaussian": round(z, 4),
            "raw_mc_coverage": round(raw_cov, 4),
            "k_empirical": round(k_emp, 4),
            "global_coverage": round(cov_global, 4),
            "adapt_coverage": round(cov_adapt, 4),
        }
    )
    print(
        f"  {int(nominal * 100):2d}%: raw={raw_cov:.4f}  global={cov_global:.4f}  "
        f"adapt={cov_adapt:.4f}  k={k_emp:.4f}  z={z:.4f}"
    )

# ── Extract arrays ──────────────────────────────────────────────────────
x_nom = [r["nominal"] for r in results]
raw_covs = [r["raw_mc_coverage"] for r in results]
global_covs = [r["global_coverage"] for r in results]
adapt_covs = [r["adapt_coverage"] for r in results]
k_vals = [r["k_empirical"] for r in results]
z_vals = [r["z_gaussian"] for r in results]

# ── Cross-checks against audit ──────────────────────────────────────────
print("\n-- Cross-checks --")
r90 = results[3]  # 90%
r95 = results[4]  # 95%
assert abs(r90["global_coverage"] - 0.9017) < 0.005, (
    f"FAIL: global@90%={r90['global_coverage']}"
)
assert abs(r95["k_empirical"] - 11.3407) < 0.10, f"FAIL: k95={r95['k_empirical']}"
print(f"  [OK] global_cov@90% = {r90['global_coverage']:.4f}")
print(f"  [OK] k95 = {r95['k_empirical']:.4f}")
print(f"  [OK] z95 = {r95['z_gaussian']:.4f}")

# ── Figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

xtick_locs = [0.50, 0.70, 0.80, 0.90, 0.95]
xtick_labels = ["50%", "70%", "80%", "90%", "95%"]

# ── Panel (a): Calibration Curve ────────────────────────────────────────
ax = ax1
panel_label(ax, "(a)", x=-0.08, y=1.04)

# Perfect calibration diagonal
ax.plot(
    [0.45, 1.00],
    [0.45, 1.00],
    "--",
    color=P_LGRAY,
    lw=1.5,
    label="Perfect calibration",
    zorder=2,
)

# Raw MC Gaussian — coral (bad)
ax.plot(
    x_nom,
    raw_covs,
    "-o",
    color=P_CORAL,
    lw=2.2,
    ms=7,
    label="Raw MC Gaussian",
    zorder=4,
)

# Global conformal — blue (good)
ax.plot(
    x_nom,
    global_covs,
    "-s",
    color=P_BLUE,
    lw=2.2,
    ms=7,
    label="Global conformal",
    zorder=4,
)

# Adaptive conformal — green (good)
ax.plot(
    x_nom,
    adapt_covs,
    "-^",
    color=P_GREEN,
    lw=2.2,
    ms=7,
    label="Adaptive conformal",
    zorder=4,
)

# Annotate the worst miscalibration point: 95% nominal, raw MC
raw95 = raw_covs[-1]
ax.annotate(
    f"{raw95 * 100:.1f}% achieved\nat 95% nominal",
    xy=(0.95, raw95),
    xytext=(0.78, raw95 + 0.06),
    fontsize=8,
    color=P_CORAL,
    fontweight="semibold",
    arrowprops=dict(arrowstyle="->", color=P_CORAL, lw=1.0),
    ha="center",
)

ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
ax.set_ylabel("Achieved coverage", color=P_DGRAY)
ax.set_title("Calibration Curve", fontsize=12, color=P_DGRAY, pad=8)
ax.set_xlim(0.44, 1.01)
ax.set_ylim(0.18, 1.03)
ax.set_xticks(xtick_locs)
ax.set_xticklabels(xtick_labels)
ax.set_yticks([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
ax.set_yticklabels(
    ["20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "95%", "100%"]
)
ax.tick_params(colors=P_SLATE, labelsize=9)
ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
ax.grid(True, alpha=0.3, zorder=0)

# ── Panel (b): Empirical k vs Gaussian z ────────────────────────────────
ax = ax2
panel_label(ax, "(b)", x=-0.08, y=1.04)

# Gaussian z — coral (assumed)
ax.plot(
    x_nom,
    z_vals,
    "-o",
    color=P_CORAL,
    lw=2.2,
    ms=7,
    label="Gaussian z (assumed)",
    zorder=4,
)

# Empirical k — amber (actual needed)
ax.plot(
    x_nom,
    k_vals,
    "-s",
    color=P_AMBER,
    lw=2.2,
    ms=7,
    label="Empirical k (actual needed)",
    zorder=4,
)

# k95 annotation — placed to the left to avoid collision
k95 = k_vals[-1]
z95 = z_vals[-1]
ax.annotate(
    f"k$_{{95}}$ = {k95:.2f}",
    xy=(0.95, k95),
    xytext=(0.80, k95 - 2.5),
    fontsize=9,
    fontweight="semibold",
    color=P_AMBER,
    arrowprops=dict(arrowstyle="->", color=P_AMBER, lw=1.0),
    ha="center",
)

# z95 annotation — placed above
ax.annotate(
    f"z$_{{95}}$ = {z95:.3f}",
    xy=(0.95, z95),
    xytext=(0.80, z95 + 0.5),
    fontsize=9,
    fontweight="semibold",
    color=P_CORAL,
    arrowprops=dict(arrowstyle="->", color=P_CORAL, lw=1.0),
    ha="center",
)

# Ratio annotation between the two points — shifted left
mid_y = (k95 + z95) / 2
ax.annotate(
    f"{k95 / z95:.1f}x gap",
    xy=(0.92, mid_y),
    fontsize=8.5,
    color=P_SLATE,
    fontstyle="italic",
    ha="center",
    va="center",
)

ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
ax.set_ylabel("Sigma multiplier", color=P_DGRAY)
ax.set_title("Empirical k vs. Gaussian z", fontsize=12, color=P_DGRAY, pad=8)
ax.set_xlim(0.44, 1.01)
ax.set_xticks(xtick_locs)
ax.set_xticklabels(xtick_labels)
ax.tick_params(colors=P_SLATE, labelsize=9)
ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
ax.grid(True, alpha=0.3, zorder=0)

# Suptitle — clean, no unnecessary details
fig.suptitle(
    "T8 Calibration Audit: Nominal vs. Achieved Coverage (MC Dropout, S=30)",
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
    out = os.path.join(THESIS_FIG, f"t8_calibration_curve.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved: t8_calibration_curve.pdf + .png")
