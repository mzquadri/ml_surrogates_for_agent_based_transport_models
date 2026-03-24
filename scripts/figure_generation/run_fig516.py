"""Fig 5.16 — T7 Calibration Curve (redesign).
Exact same style as approved Fig 5.9 (T8).
Values cross-checked against UQ_CALIBRATION_AUDIT_T7.md.
"""

import sys, os, numpy as np
from scipy import stats

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
EPS = 1e-6

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading NPZ files ...")
mc = np.load(MC_NPZ)
yhat = mc["predictions"].astype(np.float64)
sig = mc["uncertainties"].astype(np.float64)
y = mc["targets"].astype(np.float64)
n = len(y)
print(f"  n_nodes = {n:,}")

# 20/80 split (same as T8 Part 3)
split = n // 5
y_c, yhat_c, sig_c = y[:split], yhat[:split], sig[:split]
y_t, yhat_t, sig_t = y[split:], yhat[split:], sig[split:]
print(f"  Cal: {len(y_c):,}   Test: {len(y_t):,}")


def conformal_q(residuals, alpha):
    nn = residuals.shape[0]
    q_level = np.ceil((nn + 1) * (1 - alpha)) / nn
    q_level = min(q_level, 1.0)
    return float(np.quantile(residuals, q_level, method="higher"))


# Residuals
r_cal_abs = np.abs(y_c - yhat_c)
r_cal_scaled = r_cal_abs / (sig_c + EPS)
r_test_abs = np.abs(y_t - yhat_t)
r_test_scaled = r_test_abs / (sig_t + EPS)

NOMINAL_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
results = []

print("\nComputing calibration audit ...")
for nominal in NOMINAL_LEVELS:
    alpha = 1.0 - nominal
    z = float(stats.norm.ppf((1.0 + nominal) / 2.0))
    raw_cov = float(np.mean(r_test_abs <= z * sig_t))
    k_emp = float(np.quantile(r_test_scaled, nominal))
    q_global = float(conformal_q(r_cal_abs, alpha))
    cov_global = float(np.mean(r_test_abs <= q_global))
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

# ── Cross-check against UQ_CALIBRATION_AUDIT_T7.md ───────────────────────────
MD_REF = {
    50: {"raw": 0.1795, "global": 0.5038, "adapt": 0.5015, "k": 2.0408},
    70: {"raw": 0.2682, "global": 0.7039, "adapt": 0.7019, "k": 3.7587},
    80: {"raw": 0.3275, "global": 0.8039, "adapt": 0.8017, "k": 5.8464},
    90: {"raw": 0.4145, "global": 0.9018, "adapt": 0.9005, "k": 10.4534},
    95: {"raw": 0.4838, "global": 0.9511, "adapt": 0.9503, "k": 16.1445},
}

print(f"\n=== CROSS-CHECK vs UQ_CALIBRATION_AUDIT_T7.md ===")
all_pass = True
for r in results:
    pct = int(r["nominal"] * 100)
    ref = MD_REF[pct]
    checks = [
        ("raw", r["raw_mc_coverage"], ref["raw"]),
        ("global", r["global_coverage"], ref["global"]),
        ("adapt", r["adapt_coverage"], ref["adapt"]),
        ("k", r["k_empirical"], ref["k"]),
    ]
    for name, comp, exp in checks:
        ok = abs(comp - exp) < 0.005
        if not ok:
            all_pass = False
        status = "OK" if ok else "FAIL"
        print(f"  {pct}% {name:>6s}: {comp:.4f} vs {exp:.4f} {status}")

print(f"\n  >>> {'ALL CHECKS PASSED' if all_pass else 'SOME FAILED'} <<<")
if not all_pass:
    sys.exit(1)

# ── Extract arrays ────────────────────────────────────────────────────────────
x_nom = [r["nominal"] for r in results]
raw_covs = [r["raw_mc_coverage"] for r in results]
global_covs = [r["global_coverage"] for r in results]
adapt_covs = [r["adapt_coverage"] for r in results]
k_vals = [r["k_empirical"] for r in results]
z_vals = [r["z_gaussian"] for r in results]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — exact same template as approved Fig 5.9 (T8)
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

xtick_locs = [0.50, 0.70, 0.80, 0.90, 0.95]
xtick_labels = ["50%", "70%", "80%", "90%", "95%"]

# ── Panel (a): Calibration Curve ──────────────────────────────────────────────
ax = ax1
panel_label(ax, "(a)", x=-0.08, y=1.04)

ax.plot(
    [0.45, 1.00],
    [0.45, 1.00],
    "--",
    color=P_LGRAY,
    lw=1.5,
    label="Perfect calibration",
    zorder=2,
)
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

# Annotate worst miscalibration: 95% nominal, raw MC
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
ax.set_ylim(0.10, 1.03)
ax.set_xticks(xtick_locs)
ax.set_xticklabels(xtick_labels)
ax.set_yticks([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
ax.set_yticklabels(
    ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "95%", "100%"]
)
ax.tick_params(colors=P_SLATE, labelsize=9)
ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
ax.grid(True, alpha=0.3, zorder=0)

# ── Panel (b): Empirical k vs Gaussian z ──────────────────────────────────────
ax = ax2
panel_label(ax, "(b)", x=-0.08, y=1.04)

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

# k95 annotation
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

# z95 annotation
ax.annotate(
    f"z$_{{95}}$ = {z95:.3f}",
    xy=(0.95, z95),
    xytext=(0.80, z95 + 0.8),
    fontsize=9,
    fontweight="semibold",
    color=P_CORAL,
    arrowprops=dict(arrowstyle="->", color=P_CORAL, lw=1.0),
    ha="center",
)

# Ratio annotation
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

fig.suptitle(
    "T7 Calibration Audit: Nominal vs. Achieved Coverage (MC Dropout, S=30)",
    fontsize=13,
    fontweight="bold",
    color=P_DGRAY,
    y=0.99,
)

fig.tight_layout(pad=1.8, rect=[0, 0.02, 1, 0.94])

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(REPO, "thesis", "latex_tum_official", "figures")
for ext in ("pdf", "png"):
    out = os.path.join(out_dir, f"t7_calibration_curve.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {out}")
plt.close(fig)
print("Done.")
