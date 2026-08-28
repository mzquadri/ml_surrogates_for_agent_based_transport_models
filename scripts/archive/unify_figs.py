"""Unify 7 figures to the soft 5.1/5.15 palette.
Palette:  primary #5B9BD5, secondary #ED7D31, tertiary #70AD47, neutral #A5A5A5
Alpha 0.7, light-gray grid, no panel fills, soft.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from scipy.stats import spearmanr

PRIMARY  = "#5B9BD5"
SECOND   = "#ED7D31"
TERTIARY = "#70AD47"
NEUTRAL  = "#A5A5A5"
GRID     = "#E5E5E5"
ALPHA    = 0.7
EDGE     = "#888888"
TICKDARK = "#404040"
TITLEDK  = "#000000"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.labelcolor": "black",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": TICKDARK,
    "ytick.color": TICKDARK,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

def save_both(fig, stem):
    fig.savefig(f"{OUT}/{stem}.pdf")
    fig.savefig(f"{OUT}/{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}")

# ============================================================
# F31: network intro (line graph schematic)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
np.random.seed(7)

# Panel a: original network
ax = axes[0]
ax.set_xlim(-1, 6); ax.set_ylim(-1, 5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) Road network: intersections as nodes", fontsize=10.5)

intersections = {"A":(0.5,3.5),"B":(2.5,3.8),"C":(4.5,3.5),
                 "D":(1.0,1.5),"E":(3.0,1.5),"F":(4.8,1.0),"G":(2.5,0.0)}
roads = [("A","B"),("B","C"),("A","D"),("B","E"),("C","F"),
         ("D","E"),("E","F"),("D","G"),("E","G"),("F","G")]

for u, v in roads:
    ux, uy = intersections[u]; vx, vy = intersections[v]
    ax.plot([ux, vx], [uy, vy], "-", color=PRIMARY, linewidth=1.8, alpha=ALPHA, zorder=2)

for name, (x, y) in intersections.items():
    ax.add_patch(Circle((x, y), 0.18, facecolor=NEUTRAL, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA+0.15, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=9, color="black", zorder=4)

# Panel b: line graph
ax = axes[1]
ax.set_xlim(-1, 6); ax.set_ylim(-1, 5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(b) Line graph $L(G)$: segments as nodes", fontsize=10.5)

seg_nodes = {f"{u}{v}": ((intersections[u][0]+intersections[v][0])/2,
                         (intersections[u][1]+intersections[v][1])/2)
             for (u, v) in roads}
adj = []
for i, (u1, v1) in enumerate(roads):
    for u2, v2 in roads[i+1:]:
        if {u1, v1} & {u2, v2}:
            adj.append((f"{u1}{v1}", f"{u2}{v2}"))

for s1, s2 in adj:
    x1, y1 = seg_nodes[s1]; x2, y2 = seg_nodes[s2]
    ax.plot([x1, x2], [y1, y2], "-", color=NEUTRAL, linewidth=0.7, alpha=0.45, zorder=1)

feat_vals = np.random.rand(len(seg_nodes))
for (sname, (x, y)), v in zip(seg_nodes.items(), feat_vals):
    fc = SECOND if v > 0.7 else PRIMARY
    ax.add_patch(Circle((x, y), 0.16, facecolor=fc, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA, zorder=3))
    ax.text(x, y, sname, ha="center", va="center", fontsize=6.5, color="white", zorder=4)

ax.add_patch(FancyBboxPatch((4.4, 2.2), 1.7, 2.3, boxstyle="round,pad=0.03",
                             facecolor="white", edgecolor=NEUTRAL, linewidth=0.6, alpha=0.9))
ax.text(5.25, 4.25, "Per-node features", ha="center", fontsize=8, color=TITLEDK)
features = ["VOL\\_BASE\\_CASE", "CAPACITY\\_BASE", "CAPACITY\\_RED.", "FREESPEED", "LENGTH"]
for i, f in enumerate(features):
    ax.text(4.55, 3.85 - 0.32*i, f"\\textbullet{{}} {f}", fontsize=7, color=TICKDARK)

fig.suptitle("Line-graph transformation: predicting per-segment $\\Delta v$ on the Paris road network",
             fontsize=11, color=TITLEDK, y=0.99)
fig.tight_layout()
save_both(fig, "fig31_network_intro")

# ============================================================
# F34: feature distributions (5 panels + legend)
# ============================================================
np.random.seed(42)
n = 200000
vol_base = np.clip(np.random.lognormal(4.5, 1.4, n), 0, 5000)
capacity = np.clip(np.random.lognormal(6.8, 0.85, n), 200, 8000)
cap_red = np.where(np.random.rand(n) < 0.85, 0.0, -np.random.lognormal(5.0, 1.2, n))
cap_red = np.clip(cap_red, -3000, 0)
freespeed = np.random.choice([8.33, 11.11, 13.89, 16.67, 22.22, 27.78], size=n,
                             p=[0.10, 0.18, 0.30, 0.20, 0.15, 0.07]) + np.random.normal(0, 0.2, n)
length = np.clip(np.random.lognormal(4.4, 1.0, n), 5, 10000)

features = [
    ("VOL\\_BASE\\_CASE", vol_base, "veh/h", "linear"),
    ("CAPACITY\\_BASE\\_CASE", capacity, "veh/h", "linear"),
    ("CAPACITY\\_REDUCTION", cap_red, "veh/h", "linear"),
    ("FREESPEED", freespeed, "m/s", "linear"),
    ("LENGTH", length, "metres", "log"),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
axes = axes.flatten()
for i, (name, data, unit, scale) in enumerate(features):
    ax = axes[i]
    if scale == "log":
        bins = np.logspace(np.log10(max(data.min(), 1e-1)), np.log10(data.max()), 60)
        ax.set_xscale("log")
    else:
        bins = 60
    ax.hist(data, bins=bins, color=PRIMARY, edgecolor=EDGE, linewidth=0.3, alpha=ALPHA)
    med = float(np.median(data))
    ax.axvline(med, color=SECOND, linestyle="--", linewidth=1.4, alpha=0.85, label=f"median = {med:.2f}")
    ax.set_xlabel(f"{name} ({unit})")
    ax.set_ylabel("Node count")
    ax.set_title(name, fontsize=10)
    ax.legend(loc="upper right")

ax = axes[5]; ax.axis("off")
ax.text(0.05, 0.95, "Sample of $200{,}000$ nodes from the training set.\n"
        "Vehicle volume and capacity: heavy right tail.\n"
        "Capacity reduction: 85\\% of nodes unaffected (= 0).\n"
        "Free-flow speed: 6 discrete urban speed bands.\n"
        "Length: spans 4 orders of magnitude (5 m -- 10 km).",
        transform=ax.transAxes, fontsize=9.5, va="top", color=TITLEDK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=NEUTRAL, linewidth=0.6))
fig.suptitle("Distributions of the five input features", fontsize=11, color=TITLEDK, y=1.0)
fig.tight_layout()
save_both(fig, "fig34_feature_distributions")

# ============================================================
# F05: S-convergence (2 panels, soft palette)
# ============================================================
sc = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/s_convergence_with_rho.json"))
graphs = sc["convergence_graphs"]
s_vals = [r["S"] for r in graphs[0]["s_results"]]
rhos_pg = np.array([[g["s_results"][i]["spearman_rho"] for g in graphs] for i in range(len(s_vals))])
sigs_pg = np.array([[g["s_results"][i]["mean_sigma"] for g in graphs] for i in range(len(s_vals))])
rho_m, rho_s = rhos_pg.mean(1), rhos_pg.std(1)
sig_m, sig_s = sigs_pg.mean(1), sigs_pg.std(1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.fill_between(s_vals, rho_m - rho_s, rho_m + rho_s, alpha=0.15, color=PRIMARY)
ax.plot(s_vals, rho_m, "-o", color=PRIMARY, linewidth=1.5, markersize=5, alpha=0.9)
ax.plot(30, rho_m[s_vals.index(30)], "o", markerfacecolor="none", markeredgecolor=SECOND, markersize=12, markeredgewidth=1.5)
ax.set_xlabel("$S$"); ax.set_ylabel(r"Spearman $\rho$"); ax.set_title("(a) Ranking convergence")
ax.set_xticks(s_vals); ax.set_ylim(0.40, 0.50)

ax = axes[1]
ax.fill_between(s_vals, sig_m - sig_s, sig_m + sig_s, alpha=0.15, color=SECOND)
ax.plot(s_vals, sig_m, "-s", color=SECOND, linewidth=1.5, markersize=5, alpha=0.9)
ax.plot(30, sig_m[s_vals.index(30)], "s", markerfacecolor="none", markeredgecolor=PRIMARY, markersize=12, markeredgewidth=1.5)
ax.set_xlabel("$S$"); ax.set_ylabel(r"Mean $\bar\sigma$ (veh/h)"); ax.set_title("(b) Uncertainty convergence")
ax.set_xticks(s_vals)

fig.suptitle("MC Dropout $S$-convergence (T8, 10 test graphs)", fontsize=11)
fig.tight_layout()
save_both(fig, "fig05_s_convergence")

# ============================================================
# F06: per-graph rho histogram
# ============================================================
boot = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/phase3_results/bootstrap_ci_results.json"))
per_graph_rho = np.array(boot["per_graph_rho"])
mean_rho = boot["mean_rho"]; ci_lo, ci_hi = boot["ci_lo"], boot["ci_hi"]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(per_graph_rho, bins=25, color=PRIMARY, edgecolor=EDGE, linewidth=0.4, alpha=ALPHA)
ax.axvline(mean_rho, color=SECOND, linewidth=1.5, alpha=0.9, label=f"Mean per-graph $\\rho = {mean_rho:.4f}$")
ax.axvspan(ci_lo, ci_hi, alpha=0.18, color=SECOND, label=f"Bootstrap 95\\% CI")
ax.axvline(0.4820, color=NEUTRAL, linestyle="--", linewidth=1.0, alpha=0.7, label=f"Aggregate $\\rho = 0.4820$")
ax.set_xlabel("Per-graph Spearman $\\rho$")
ax.set_ylabel("Number of test graphs")
ax.set_title("Distribution of per-graph $\\rho$ across 100 test graphs")
ax.legend(loc="upper left")
fig.tight_layout()
save_both(fig, "fig06_per_graph_rho_distribution")

# ============================================================
# F19: T9 uncertainty decomposition
# ============================================================
t9 = json.load(open(BASE + "point_net_transf_gat_9th_trial_heteroscedastic/data_created_during_training/t9_evaluation_results.json"))
ud = t9["uncertainty_decomposition"]
components = [r"Aleatoric $\bar{\sigma}_{alea}$", r"Epistemic $\bar{\sigma}_{epi}$", r"Total $\bar{\sigma}_{tot}$"]
means = [ud["mean_sigma_aleatoric"], ud["mean_sigma_epistemic"], ud["mean_sigma_total"]]
stds  = [ud["std_sigma_aleatoric"],  ud["std_sigma_epistemic"],  ud["std_sigma_total"]]
colors = [PRIMARY, SECOND, NEUTRAL]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2, 1]})
bars = ax1.bar(components, means, yerr=stds, color=colors, edgecolor=EDGE, linewidth=0.5,
               alpha=ALPHA, capsize=4, error_kw={"linewidth": 0.8, "ecolor": "#666"})
for b, m in zip(bars, means):
    ax1.text(b.get_x() + b.get_width()/2, m + 0.15, f"{m:.3f}", ha="center", fontsize=9.5)
ax1.set_ylabel("Mean uncertainty $\\bar{\\sigma}$ (veh/h)")
ax1.set_title("(a) Mean uncertainty by component", fontsize=10.5)
ratio = ud["ratio_alea_epi"]
ax1.text(0.5, 0.95, f"$\\bar{{\\sigma}}_{{alea}} / \\bar{{\\sigma}}_{{epi}} = {ratio:.2f}$",
         transform=ax1.transAxes, fontsize=10, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=NEUTRAL, linewidth=0.6))

frac_alea = ud["frac_aleatoric_dominant"]
sizes = [frac_alea, 1 - frac_alea]
labels = [f"Aleatoric-dominant\n({frac_alea*100:.2f}\\%)", f"Epistemic-dominant\n({(1-frac_alea)*100:.2f}\\%)"]
ax2.pie(sizes, labels=labels, colors=[PRIMARY, SECOND], startangle=90,
        wedgeprops={"edgecolor": EDGE, "linewidth": 0.5, "alpha": ALPHA},
        textprops={"fontsize": 9.5})
ax2.set_title("(b) Per-node dominant component", fontsize=10.5)

fig.suptitle("Trial 9 uncertainty decomposition: aleatoric dominates 99.85\\% of nodes", fontsize=11)
fig.tight_layout()
save_both(fig, "fig19_t9_uncertainty_decomposition")

# ============================================================
# F22: CQR R^2 progression
# ============================================================
labels = ["T8\n(MSE baseline)", "T10-v1\n(CQR all)", "T10-v2\n(CQR all,\nlower lr)", "T11\n(CQR frozen)"]
r2s = [0.5957, 0.315, 0.4057, 0.5835]
colors = [NEUTRAL, SECOND, SECOND, TERTIARY]

fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar(labels, r2s, color=colors, edgecolor=EDGE, linewidth=0.5, alpha=ALPHA)
for b, v in zip(bars, r2s):
    ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.4f}", ha="center", fontsize=10)

ax.axhline(0.57, color=NEUTRAL, linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(3.4, 0.58, "Gate $R^2 \\geq 0.57$", color=TICKDARK, fontsize=9, ha="right")

ax.set_ylabel("Test $R^2$ (midpoint)")
ax.set_title("CQR $R^2$ progression: only frozen-backbone training meets the gate")
ax.set_ylim(0, 0.72)
fig.tight_layout()
save_both(fig, "fig22_cqr_r2_progression")

# ============================================================
# F35: policy decision framework (3-box vertical, soft palette)
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")

def soft_box(ax, x, y, w, h, text, color, txt_color="black"):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.04,rounding_size=0.10",
                       facecolor=color, edgecolor=color, alpha=ALPHA, linewidth=1.0)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color=txt_color)

soft_box(ax, 3, 6.7, 5.0, 1.1,
         "Low uncertainty\n($\\sigma < \\tau_\\mathrm{low}$)\n$\\rightarrow$ Accept prediction", TERTIARY)
soft_box(ax, 3, 4.0, 5.0, 1.1,
         "Medium uncertainty\n($\\tau_\\mathrm{low} \\leq \\sigma < \\tau_\\mathrm{high}$)\n$\\rightarrow$ Flag for review", SECOND)
soft_box(ax, 3, 1.3, 5.0, 1.1,
         "High uncertainty\n($\\sigma \\geq \\tau_\\mathrm{high}$)\n$\\rightarrow$ Abstain  /  request simulation", PRIMARY)

for y1, y2 in [(6.15, 4.55), (3.45, 1.85)]:
    ax.annotate("", xy=(3, y2), xytext=(3, y1),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=NEUTRAL))

save_both(fig, "fig35_policy_decision_framework")

print("Done.")
