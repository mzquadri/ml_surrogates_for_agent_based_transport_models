"""Regenerate fig12 as single-panel ECE optimisation curve only.

User asked to simplify the 4-panel sigma-scaling diagnostic — Table 5.4 already
has the coverage / ECE before+after / reliability numbers, so only the visual
that adds genuine insight (ECE vs T sweep) is kept.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from plot_style import set_style, save_both

TUM_BLUE   = "#5B9BD5"
TUM_ORANGE = "#ED7D31"
TUM_GRAY   = "#A5A5A5"

set_style()
OUT  = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

mc = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
ts_json = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/temperature_scaling_results.json"))

err = np.abs(mc["targets"] - mc["predictions"])
sigma = mc["uncertainties"]
T_opt = round(ts_json["optimal_temperature"], 3)

rng = np.random.RandomState(42)
idx = rng.choice(len(err), size=300000, replace=False)


def coverage_at_T(T, p):
    z = norm.ppf(0.5 + p / 2.0)
    return (err[idx] <= z * sigma[idx] * T).mean()


def ece_at_T(T):
    grid = np.linspace(0.05, 0.95, 20)
    return np.mean([abs(coverage_at_T(T, p) - p) for p in grid])


print("Computing ECE vs T sweep ...")
T_grid = np.linspace(0.5, 5.0, 40)
ece_grid = np.array([ece_at_T(T) for T in T_grid])

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(T_grid, ece_grid, "-", color=TUM_BLUE, linewidth=2.2)
ax.axvline(T_opt, color=TUM_ORANGE, linestyle="--", linewidth=1.6,
           label=f"$T^\\star = {T_opt}$ (optimal)")
ax.axvline(1.0, color=TUM_GRAY, linestyle=":", linewidth=1.2,
           label="$T = 1$ (raw)")
ax.set_xlabel("Scaling factor $T$")
ax.set_ylabel("Expected Calibration Error")
ax.set_title(r"ECE optimisation for T8 MC Dropout regression $\sigma$-scaling",
             fontsize=12)
ax.legend(fontsize=10)
fig.tight_layout()
save_both(fig, OUT, "fig12_sigma_scaling_ece")
print(f"Saved fig12_sigma_scaling_ece.pdf and .png. Optimal T* = {T_opt}, ECE_min = {ece_grid.min():.4f}")
