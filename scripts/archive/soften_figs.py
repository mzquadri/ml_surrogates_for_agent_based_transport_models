"""Regenerate the quantitative figures with lighter colors (alpha 0.6) and fewer overlapping text labels.
Only the heaviest, most-text-crowded figures are re-created; schematics are left as-is."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GRAY, TUM_GREEN, IDEAL_GREEN, PASS_GREEN, FAIL_RED, TUM_DARK

set_style()
# Softer variants
BLUE_S   = "#3B7FB5"   # softer blue
ORANGE_S = "#E89759"   # softer orange
GREEN_S  = "#7EA96B"   # softer green
RED_S    = "#C86E6E"   # softer red
LIGHTBLUE_S = "#8DB8D6"

OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"
ALPHA = 0.6

# -- F1: trial bars (less label clutter)
trials = ["T1","T2","T3","T4","T5","T6","T7","T8"]
r2  = [0.7860, 0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
mae = [2.972, 4.328, 5.990, 6.080, 4.242, 4.324, 4.060, 3.957]
rmse= [5.396, 8.151, 10.270, 10.151, 7.778, 8.061, 7.534, 7.118]
colors = [TUM_GRAY if t=="T1" else (ORANGE_S if t=="T8" else BLUE_S) for t in trials]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, vals, ylabel, title in [
    (axes[0], r2,   "$R^2$",           "Coefficient of determination"),
    (axes[1], mae,  "MAE (veh/h)",     "Mean absolute error"),
    (axes[2], rmse, "RMSE (veh/h)",    "Root mean square error"),
]:
    ax.bar(trials, vals, color=colors, edgecolor="black", linewidth=0.4, alpha=ALPHA)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0, max(vals) * 1.10)
fig.tight_layout()
save_both(fig, OUT, "fig01_trial_bars_r2_mae_rmse"); print("fig01 refreshed")

# -- F2: trial progression with smaller annotations
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(trials, r2, color=colors, edgecolor="black", linewidth=0.4, alpha=ALPHA)
ax.axhline(0.55, color=TUM_GRAY, linestyle=":", linewidth=0.8)
ax.text(7.4, 0.56, "$R^2 = 0.55$ (T9 gate)", fontsize=7.5, color=TUM_GRAY, va="bottom", ha="right")
for i,v in enumerate(r2):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_ylabel("Test $R^2$"); ax.set_ylim(0, 0.88)
ax.set_title("Training trial progression")
fig.tight_layout()
save_both(fig, OUT, "fig02_trial_progression_annotated"); print("fig02 refreshed")

# -- F3: UQ ranking horizontal bar
methods = ["T8 MC Dropout","MC avg (5 seeds)","Combined MC + seed var","T9 hetero. $\\sigma_{tot}$",
           "T7 MC Dropout","Seed ens. variance","Multi-model ensemble","Deep Ensemble"]
rhos = [0.4820, 0.4908, 0.4909, 0.4797, 0.4437, 0.4370, 0.4333, 0.3997]
order = np.argsort(rhos)[::-1]
methods_s = [methods[i] for i in order]; rhos_s = [rhos[i] for i in order]
bar_colors = [ORANGE_S if "T8 MC Dropout" in m else BLUE_S for m in methods_s]
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.barh(np.arange(len(methods_s)), rhos_s, color=bar_colors, edgecolor="black", linewidth=0.4, alpha=ALPHA)
for i, v in enumerate(rhos_s):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_yticks(np.arange(len(methods_s))); ax.set_yticklabels(methods_s); ax.invert_yaxis()
ax.set_xlabel(r"Spearman $\rho$"); ax.set_xlim(0, 0.55)
ax.set_title("UQ method ranking by uncertainty--error correlation")
fig.tight_layout()
save_both(fig, OUT, "fig03_uq_ranking_spearman"); print("fig03 refreshed")

# -- F5: S-convergence (lighter band)
sc = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/s_convergence_with_rho.json"))
graphs = sc["convergence_graphs"]
s_vals = [r["S"] for r in graphs[0]["s_results"]]
rhos_pg = np.array([[g["s_results"][i]["spearman_rho"] for g in graphs] for i in range(len(s_vals))])
sigs_pg = np.array([[g["s_results"][i]["mean_sigma"] for g in graphs] for i in range(len(s_vals))])
rho_m = rhos_pg.mean(1); rho_s = rhos_pg.std(1)
sig_m = sigs_pg.mean(1); sig_s = sigs_pg.std(1)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.fill_between(s_vals, rho_m - rho_s, rho_m + rho_s, alpha=0.18, color=BLUE_S)
ax.plot(s_vals, rho_m, "-o", color=BLUE_S, linewidth=1.5, markersize=4.5, alpha=0.85)
ax.plot(30, rho_m[s_vals.index(30)], "o", markerfacecolor="none", markeredgecolor=ORANGE_S, markersize=12, markeredgewidth=1.5)
ax.set_xlabel("$S$"); ax.set_ylabel(r"Spearman $\rho$"); ax.set_title("(a)")
ax.set_xticks(s_vals); ax.set_ylim(0.40, 0.50)
ax = axes[1]
ax.fill_between(s_vals, sig_m - sig_s, sig_m + sig_s, alpha=0.18, color=ORANGE_S)
ax.plot(s_vals, sig_m, "-s", color=ORANGE_S, linewidth=1.5, markersize=4.5, alpha=0.85)
ax.plot(30, sig_m[s_vals.index(30)], "s", markerfacecolor="none", markeredgecolor=BLUE_S, markersize=12, markeredgewidth=1.5)
ax.set_xlabel("$S$"); ax.set_ylabel(r"Mean $\bar\sigma$ (veh/h)"); ax.set_title("(b)")
ax.set_xticks(s_vals)
fig.suptitle("MC Dropout $S$-convergence (T8, 10 test graphs)", fontsize=11)
fig.tight_layout()
save_both(fig, OUT, "fig05_s_convergence"); print("fig05 refreshed")

# -- F11: k95 comparison
methods_k = ["Ideal\nGaussian","T9 hetero.","T8+TS","T8 MC","Deep Ens."]
k95_vals  = [1.96, 2.84, 4.04, 11.66, 15.18]
colors_k  = [IDEAL_GREEN, GREEN_S, LIGHTBLUE_S, BLUE_S, ORANGE_S]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(methods_k, k95_vals, color=colors_k, edgecolor="black", linewidth=0.4, alpha=ALPHA)
for i, v in enumerate(k95_vals):
    ax.text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(1.96, color=IDEAL_GREEN, linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_ylabel(r"Empirical $k_{95}$"); ax.set_ylim(0, 17)
ax.set_title(r"Calibration scaling $k_{95}$ across UQ methods")
fig.tight_layout()
save_both(fig, OUT, "fig11_k95_comparison"); print("fig11 refreshed")

# -- F13: conformal coverage (lighter + less annotation)
levels = [50, 70, 80, 90, 95]
raw_cov  = [23.37, 33.70, 40.09, 48.55, 54.85]
std_cov  = [50.0, 70.0, 80.0, 90.02, 95.01]
adapt_cov= [50.2, 70.2, 80.2, 89.87, 95.0]
x = np.arange(len(levels)); w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.bar(x - w, raw_cov,   w, color=BLUE_S,   edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Raw MC Dropout")
ax.bar(x,     std_cov,   w, color=ORANGE_S, edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Standard conformal")
ax.bar(x + w, adapt_cov, w, color=GREEN_S,  edgecolor="black", linewidth=0.4, alpha=ALPHA, label="Adaptive conformal")
for i, lvl in enumerate(levels):
    ax.plot([i - 1.5*w, i + 1.5*w], [lvl, lvl], "--", color=IDEAL_GREEN, linewidth=0.8, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\\%" for l in levels])
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Achieved coverage (\\%)")
ax.set_ylim(0, 105); ax.set_title("Nominal vs achieved coverage")
ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout()
save_both(fig, OUT, "fig13_conformal_coverage_nominal_vs_achieved"); print("fig13 refreshed")

# -- F26: deep ensemble member R2
seeds = [42, 137, 256, 389, 512]
member_r2 = [0.6403, 0.6458, 0.6497, 0.6469, 0.6490]
ens_r2 = 0.6841
labels = [f"Seed {s}" for s in seeds] + ["Ensemble"]
vals = member_r2 + [ens_r2]
cols = [LIGHTBLUE_S]*5 + [ORANGE_S]
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.4, alpha=ALPHA)
for i, v in enumerate(vals):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0.5957, color=TUM_GRAY, linestyle="--", linewidth=0.9, alpha=0.6)
ax.text(5.3, 0.593, "T8 single", fontsize=8, color=TUM_GRAY, ha="right")
ax.set_ylabel("Test $R^2$"); ax.set_ylim(0.55, 0.72)
ax.set_title("Deep ensemble: mean exceeds every individual member")
fig.tight_layout()
save_both(fig, OUT, "fig26_deep_ensemble_member_r2"); print("fig26 refreshed")

# -- F28: stratified UQ
mc = np.load(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz")
y = mc["targets"]; yhat = mc["predictions"]; sigma = mc["uncertainties"]
err = np.abs(y - yhat); abs_y = np.abs(y)
ranks = np.argsort(np.argsort(abs_y)); n = len(abs_y)
bounds = [0, n//4, n//2, 3*n//4, n]
ql = ["Q1","Q2","Q3","Q4"]; mae_q=[]; rho_q=[]
for k in range(4):
    m = (ranks >= bounds[k]) & (ranks < bounds[k+1])
    mae_q.append(float(err[m].mean()))
    rho, _ = spearmanr(sigma[m], err[m]); rho_q.append(float(rho))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
pal = [LIGHTBLUE_S, BLUE_S, ORANGE_S, RED_S]
axes[0].bar(ql, mae_q, color=pal, edgecolor="black", linewidth=0.4, alpha=ALPHA)
for i,v in enumerate(mae_q): axes[0].text(i, v+max(mae_q)*0.02, f"{v:.2f}", ha="center", fontsize=9)
axes[0].set_ylabel("MAE (veh/h)"); axes[0].set_title("(a) MAE by $|\\Delta v|$ quartile")
axes[1].bar(ql, rho_q, color=pal, edgecolor="black", linewidth=0.4, alpha=ALPHA)
for i,v in enumerate(rho_q): axes[1].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
axes[1].axhline(0.4820, color=TUM_GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
axes[1].set_ylabel(r"Spearman $\rho$"); axes[1].set_ylim(0, max(rho_q)*1.2)
axes[1].set_title("(b) Uncertainty ranking by quartile")
fig.tight_layout()
save_both(fig, OUT, "fig28_stratified_uq_quartiles"); print("fig28 refreshed")

print("All softened figures saved.")
