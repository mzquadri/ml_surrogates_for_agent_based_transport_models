"""Generate F9–F12: calibration figures."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GRAY, TUM_GREEN, IDEAL_GREEN, FAIL_RED

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

# ============================================================
# F9: Reliability diagram (raw MC vs after temperature scaling)
# ============================================================
# Compute reliability directly from NPZ (both raw and after T=2.887)
mc = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
y = mc["targets"]
yhat = mc["predictions"]
sigma = mc["uncertainties"]
err = np.abs(y - yhat)
T_opt = 2.887

nominal = np.linspace(0.01, 0.99, 40)
def observed_coverage(sigma_eff):
    return np.array([(err <= norm.ppf(0.5 + p/2) * sigma_eff).mean() for p in nominal])

obs_raw = observed_coverage(sigma)
obs_ts = observed_coverage(sigma * T_opt)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color=IDEAL_GREEN, linewidth=1.5, label="Ideal calibration")
ax.plot(nominal, obs_raw, "-o", color=TUM_BLUE, linewidth=2, markersize=5, label="Raw MC Dropout ($T=1$, ECE=0.356)")
ax.plot(nominal, obs_ts, "-s", color=TUM_ORANGE, linewidth=2, markersize=5, label=f"After temperature scaling ($T={T_opt}$, ECE=0.034)")
# Shade miscalibration region for raw MC
ax.fill_between(nominal, nominal, obs_raw, alpha=0.12, color=TUM_BLUE, label="Miscalibration gap (raw)")
ax.set_xlabel("Nominal coverage $p$")
ax.set_ylabel("Observed coverage")
ax.set_title("Reliability diagram: T8 MC Dropout before and after temperature scaling")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=9.5)
ax.set_aspect("equal")
fig.tight_layout()
save_both(fig, OUT, "fig09_reliability_diagram")
print("F9 saved")

# ============================================================
# F10: PIT histograms before and after temperature scaling
# ============================================================
def pit_values(sigma_eff):
    z = (y - yhat) / np.maximum(sigma_eff, 1e-10)
    return norm.cdf(z)

# Subsample for plotting speed (KS stat is computed on subsample per thesis footnote)
idx = np.random.RandomState(42).choice(len(y), 500000, replace=False)
pit_raw = pit_values(sigma)[idx]
pit_ts = pit_values(sigma * T_opt)[idx]

# Thesis/verified KS values
ks_raw = 0.245
ks_ts = 0.104

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, pit, label, ks, col in [
    (axes[0], pit_raw, f"Raw $\\sigma$ ($T = 1$)", ks_raw, TUM_BLUE),
    (axes[1], pit_ts, f"After $T = {T_opt}$", ks_ts, TUM_ORANGE),
]:
    ax.hist(pit, bins=20, range=(0, 1), color=col, edgecolor="black", linewidth=0.5, alpha=0.85, density=True)
    ax.axhline(1.0, color=IDEAL_GREEN, linestyle="--", linewidth=1.5, label="Uniform (ideal)")
    ax.set_xlabel("PIT value $F(y)$")
    ax.set_title(f"{label}\nKS = {ks:.3f},  mean(PIT) = {pit.mean():.3f}")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_xlim(0, 1)

axes[0].set_ylabel("Density")
fig.suptitle("PIT histograms before and after temperature scaling (T8, 500K-node subsample)")
fig.tight_layout()
save_both(fig, OUT, "fig10_pit_before_after_ts")
print("F10 saved")

# ============================================================
# F11: k95 comparison bar chart
# ============================================================
methods = ["Ideal\nGaussian", "T9 hetero.\n(frozen)", "T8 + TS\n($T=2.887$)", "T8\nMC Dropout", "Deep\nEnsemble"]
k95_vals = [1.96, 2.84, 4.04, 11.66, 15.18]
# Color code: green=ideal, progressively worse
colors = [IDEAL_GREEN, TUM_GREEN, TUM_LIGHTBLUE, TUM_BLUE, TUM_ORANGE]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(methods, k95_vals, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, k95_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
# Ideal reference line
ax.axhline(1.96, color=IDEAL_GREEN, linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(4.3, 2.1, r"$k_{95}^{ideal} = 1.96$", color=IDEAL_GREEN, fontsize=9, va="bottom", ha="right")

ax.set_ylabel(r"Empirical $k_{95}$ (lower is better)")
ax.set_title(r"Calibration scaling factor $k_{95}$ across UQ methods")
ax.set_ylim(0, 17.5)
fig.tight_layout()
save_both(fig, OUT, "fig11_k95_comparison")
print("F11 saved")

# ============================================================
# F12: Temperature scaling 4-panel diagnostic
# ============================================================
ts_json = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/temperature_scaling_results.json"))

# ECE grid scan (recompute so we have a smooth curve)
def ece_at_T(T):
    sig_eff = sigma[idx] * T
    # Kuleshov ECE: bin by sigma percentile, measure 1-sigma coverage per bin
    # Simpler: average |observed - nominal| over nominal grid
    grid = np.linspace(0.05, 0.95, 20)
    diffs = []
    for p in grid:
        z_alpha = norm.ppf(0.5 + p/2)
        cov = (err[idx] <= z_alpha * sig_eff).mean()
        diffs.append(abs(cov - p))
    return np.mean(diffs)

T_grid = np.linspace(0.5, 5.0, 40)
ece_grid = [ece_at_T(T) for T in T_grid]
T_best = T_grid[int(np.argmin(ece_grid))]

fig = plt.figure(figsize=(12, 8.5))

# Panel (a): ECE vs T curve
ax = plt.subplot(2, 2, 1)
ax.plot(T_grid, ece_grid, "-", color=TUM_BLUE, linewidth=2)
ax.axvline(T_opt, color=TUM_ORANGE, linestyle="--", linewidth=1.5, label=f"$T^* = {T_opt}$ (optimal)")
ax.axvline(1.0, color=TUM_GRAY, linestyle=":", linewidth=1, label="$T = 1$ (raw)")
ax.set_xlabel("Temperature $T$")
ax.set_ylabel("Expected Calibration Error")
ax.set_title("(a) ECE optimisation")
ax.legend(fontsize=9)

# Panel (b): Sigma coverage bars before/after
cov_before = [ts_json["before_calibration"]["coverage_1sig"], ts_json["before_calibration"]["coverage_2sig"], ts_json["before_calibration"]["coverage_3sig"]]
cov_after = [ts_json["after_calibration"]["coverage_1sig"], ts_json["after_calibration"]["coverage_2sig"], ts_json["after_calibration"]["coverage_3sig"]]
cov_target = [0.6827, 0.9545, 0.9973]
levels = [r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"]
x = np.arange(len(levels))
w = 0.27
ax = plt.subplot(2, 2, 2)
ax.bar(x - w, cov_before, w, color=TUM_BLUE, edgecolor="black", linewidth=0.5, label="Before ($T=1$)")
ax.bar(x, cov_after, w, color=TUM_ORANGE, edgecolor="black", linewidth=0.5, label=f"After ($T={T_opt}$)")
ax.bar(x + w, cov_target, w, color=IDEAL_GREEN, edgecolor="black", linewidth=0.5, label="Gaussian target")
for i, (b, a, t) in enumerate(zip(cov_before, cov_after, cov_target)):
    ax.text(i - w, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(i + w, t + 0.01, f"{t:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(levels)
ax.set_ylabel("Observed coverage")
ax.set_title("(b) Gaussian interval coverage")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0, 1.08)

# Panel (c): Reliability before
ax = plt.subplot(2, 2, 3)
ax.plot([0, 1], [0, 1], "--", color=IDEAL_GREEN, linewidth=1.3)
ax.plot(nominal, obs_raw, "-o", color=TUM_BLUE, linewidth=2, markersize=4)
ax.fill_between(nominal, nominal, obs_raw, alpha=0.2, color=TUM_BLUE)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
ax.set_title(f"(c) Reliability before\nECE = {ts_json['before_calibration']['ece']:.3f}")
ax.set_aspect("equal")

# Panel (d): Reliability after
ax = plt.subplot(2, 2, 4)
ax.plot([0, 1], [0, 1], "--", color=IDEAL_GREEN, linewidth=1.3)
ax.plot(nominal, obs_ts, "-s", color=TUM_ORANGE, linewidth=2, markersize=4)
ax.fill_between(nominal, nominal, obs_ts, alpha=0.2, color=TUM_ORANGE)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
ax.set_title(f"(d) Reliability after (T={T_opt})\nECE = {ts_json['after_calibration']['ece']:.3f} ($-$90.5\\%)")
ax.set_aspect("equal")

fig.suptitle("Temperature scaling diagnostic for T8 MC Dropout")
fig.tight_layout()
save_both(fig, OUT, "fig12_temperature_scaling_4panel")
print("F12 saved")
