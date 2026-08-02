"""Regenerate fig12_temperature_scaling_4panel.pdf with all 4 panels properly.

Panels:
  (a) ECE vs T scan with optimal T marked
  (b) Coverage at 1sig / 2sig / 3sig: before, after, Gaussian target
  (c) Reliability diagram before scaling
  (d) Reliability diagram after scaling
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from plot_style import set_style, save_both

# Use the lighter unified palette (matches fig13, fig15, fig07, etc.)
TUM_BLUE   = "#5B9BD5"   # light blue
TUM_ORANGE = "#ED7D31"   # light orange
TUM_GRAY   = "#A5A5A5"   # neutral grey
IDEAL_GREEN = "#70AD47"  # light green

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

mc = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
ts_json = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/temperature_scaling_results.json"))

err = np.abs(mc["targets"] - mc["predictions"])
sigma = mc["uncertainties"]
T_opt = round(ts_json["optimal_temperature"], 3)

# Subsample for the ECE scan and reliability diagrams (full set = 3.16M, slow for many T values)
rng = np.random.RandomState(42)
idx = rng.choice(len(err), size=300000, replace=False)


def coverage_at_T(T, p):
    """Empirical coverage of |err| <= z_{p/2+0.5} * (sigma * T)."""
    z = norm.ppf(0.5 + p / 2.0)
    return (err[idx] <= z * sigma[idx] * T).mean()


def ece_at_T(T):
    grid = np.linspace(0.05, 0.95, 20)
    return np.mean([abs(coverage_at_T(T, p) - p) for p in grid])


print("Computing ECE vs T scan ...")
T_grid = np.linspace(0.5, 5.0, 40)
ece_grid = np.array([ece_at_T(T) for T in T_grid])

print("Computing reliability curves ...")
nominal = np.linspace(0.05, 0.95, 19)
obs_raw = np.array([coverage_at_T(1.0, p) for p in nominal])
obs_ts  = np.array([coverage_at_T(T_opt, p) for p in nominal])

# ----------------------------------------------------------------
fig = plt.figure(figsize=(12, 8.5))

# Panel (a): ECE vs T
ax = plt.subplot(2, 2, 1)
ax.plot(T_grid, ece_grid, "-", color=TUM_BLUE, linewidth=2)
ax.axvline(T_opt, color=TUM_ORANGE, linestyle="--", linewidth=1.5,
           label=f"$T^\\star = {T_opt}$ (optimal)")
ax.axvline(1.0, color=TUM_GRAY, linestyle=":", linewidth=1, label="$T = 1$ (raw)")
ax.set_xlabel("Scaling factor $T$")
ax.set_ylabel("Expected Calibration Error")
ax.set_title("(a) ECE optimisation")
ax.legend(fontsize=9)

# Panel (b): coverage at 1sig / 2sig / 3sig
cov_before = [ts_json["before_calibration"]["coverage_1sig"],
              ts_json["before_calibration"]["coverage_2sig"],
              ts_json["before_calibration"]["coverage_3sig"]]
cov_after  = [ts_json["after_calibration"]["coverage_1sig"],
              ts_json["after_calibration"]["coverage_2sig"],
              ts_json["after_calibration"]["coverage_3sig"]]
cov_target = [0.6827, 0.9545, 0.9973]
levels = [r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"]
x = np.arange(len(levels)); w = 0.27

ax = plt.subplot(2, 2, 2)
ax.bar(x - w, cov_before, w, color=TUM_BLUE,  edgecolor="black", linewidth=0.5, label="Before ($T=1$)")
ax.bar(x,     cov_after,  w, color=TUM_ORANGE, edgecolor="black", linewidth=0.5, label=f"After ($T={T_opt}$)")
ax.bar(x + w, cov_target, w, color=IDEAL_GREEN, edgecolor="black", linewidth=0.5, label="Gaussian target")
for i, (b, a, t) in enumerate(zip(cov_before, cov_after, cov_target)):
    ax.text(i - w, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.text(i,     a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(i + w, t + 0.01, f"{t:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(levels)
ax.set_ylabel("Observed coverage")
ax.set_title("(b) Gaussian interval coverage")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0, 1.10)

# Panel (c): reliability before
ax = plt.subplot(2, 2, 3)
ax.plot([0, 1], [0, 1], "--", color=IDEAL_GREEN, linewidth=1.3, label="perfect calibration")
ax.plot(nominal, obs_raw, "-o", color=TUM_BLUE, linewidth=2, markersize=4)
ax.fill_between(nominal, nominal, obs_raw, alpha=0.2, color=TUM_BLUE)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
ax.set_title(f"(c) Reliability before\nECE = {ts_json['before_calibration']['ece']:.3f}")
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="upper left")

# Panel (d): reliability after
ax = plt.subplot(2, 2, 4)
ax.plot([0, 1], [0, 1], "--", color=IDEAL_GREEN, linewidth=1.3, label="perfect calibration")
ax.plot(nominal, obs_ts, "-s", color=TUM_ORANGE, linewidth=2, markersize=4)
ax.fill_between(nominal, nominal, obs_ts, alpha=0.2, color=TUM_ORANGE)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
ax.set_title(f"(d) Reliability after ($T={T_opt}$)\nECE = {ts_json['after_calibration']['ece']:.3f} ($-90.5\\%$)")
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("Regression $\\sigma$-scaling diagnostic for T8 MC Dropout", fontsize=13)
fig.tight_layout()
save_both(fig, OUT, "fig12_temperature_scaling_4panel")
print("Saved fig12_temperature_scaling_4panel.pdf and .png with 4 panels.")
