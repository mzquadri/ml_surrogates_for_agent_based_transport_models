"""Generate F22–F25: T10 vs T11 CQR figures."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GRAY, TUM_GREEN, IDEAL_GREEN, FAIL_RED, PASS_GREEN

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

t10 = json.load(open(BASE + "point_net_transf_gat_10th_trial_cqr/cqr_results/cqr_metrics.json"))
t11 = json.load(open(BASE + "point_net_transf_gat_11th_trial_cqr_frozen/cqr_results/cqr_metrics.json"))
conf_std = json.load(open(BASE + "point_net_transf_gat_8th_trial_lower_dropout/uq_results/conformal_standard.json"))

T8_R2 = 0.5957
T10_v1_R2 = 0.315  # from UQ_SUMMARY (earlier T10 run)
T10_v2_R2 = t10["test_metrics"]["r2_midpoint"]  # 0.4057
T11_R2 = t11["test_metrics"]["r2_midpoint"]      # 0.5835
GATE_R2 = 0.57

# ============================================================
# F22: CQR R² progression (T8 → T10v1 → T10v2 → T11) with gate
# ============================================================
labels = ["T8\n(MSE baseline)", "T10-v1\n(CQR all,\nlr=5e-4)", "T10-v2\n(CQR all,\nlr=5e-5)", "T11\n(CQR frozen,\n134 params)"]
r2s = [T8_R2, T10_v1_R2, T10_v2_R2, T11_R2]
status = ["BASELINE", "FAIL", "FAIL", "PASS"]
colors = [TUM_GRAY, FAIL_RED, FAIL_RED, PASS_GREEN]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, r2s, color=colors, edgecolor="black", linewidth=0.6)
for b, v, s in zip(bars, r2s, status):
    ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.4f}",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(b.get_x() + b.get_width()/2, v/2, s, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=("black" if s != "BASELINE" else TUM_GRAY), alpha=0.85))

ax.axhline(GATE_R2, color=FAIL_RED, linestyle="--", linewidth=1.4, label=f"$R^2 \\geq {GATE_R2}$ gate")
ax.text(3.4, GATE_R2 + 0.008, f"Gate", color=FAIL_RED, fontsize=9, ha="right")

ax.set_ylabel("Test $R^2$ (midpoint)")
ax.set_title("CQR R$^2$ progression: only frozen-backbone training (T11) preserves accuracy")
ax.set_ylim(0, 0.72)
ax.legend(loc="upper left", fontsize=9.5)
fig.tight_layout()
save_both(fig, OUT, "fig22_cqr_r2_progression")
print("F22 saved")

# ============================================================
# F23: CQR coverage bars (PICP90 + PICP95) for T10v2, T11
# ============================================================
methods = ["T10-v2\n(unfrozen)", "T11\n(frozen)"]
picp90 = [t10["test_metrics"]["PICP_90_pct"], t11["test_metrics"]["PICP_90_pct"]]
picp95 = [t10["test_metrics"]["PICP_95_pct"], t11["test_metrics"]["PICP_95_pct"]]
gate90, gate95 = 88, 93

x = np.arange(len(methods))
w = 0.35
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - w/2, picp90, w, color=TUM_BLUE, edgecolor="black", linewidth=0.5, label="PICP$_{90}$")
b2 = ax.bar(x + w/2, picp95, w, color=TUM_ORANGE, edgecolor="black", linewidth=0.5, label="PICP$_{95}$")

# Gate lines per group
for i in range(len(methods)):
    ax.plot([i - w, i], [gate90, gate90], "--", color=FAIL_RED, linewidth=1.2)
    ax.plot([i, i + w], [gate95, gate95], "--", color=FAIL_RED, linewidth=1.2)

# Annotations with PASS/FAIL
def status_label(v, gate):
    return ("PASS", PASS_GREEN) if v >= gate else ("FAIL", FAIL_RED)

for b, v in zip(b1, picp90):
    s, c = status_label(v, gate90)
    ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.2f}\\%\n[{s}]",
            ha="center", fontsize=8.5, color=c, fontweight="bold")
for b, v in zip(b2, picp95):
    s, c = status_label(v, gate95)
    ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.2f}\\%\n[{s}]",
            ha="center", fontsize=8.5, color=c, fontweight="bold")

ax.text(-0.5, gate90 - 1.2, f"Gate90: $\\geq {gate90}\\%$", color=FAIL_RED, fontsize=8.5, ha="left")
ax.text(-0.5, gate95 - 1.2, f"Gate95: $\\geq {gate95}\\%$", color=FAIL_RED, fontsize=8.5, ha="left")

ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("Empirical coverage (\\%)")
ax.set_title("CQR coverage gates: T11 frozen-backbone passes both, T10 fails PICP$_{95}$")
ax.set_ylim(85, 99)
ax.legend(loc="lower right", fontsize=9.5)
fig.tight_layout()
save_both(fig, OUT, "fig23_cqr_coverage_bars")
print("F23 saved")

# ============================================================
# F24: CQR interval widths vs split conformal
# ============================================================
methods = ["T10-v2\n(CQR unfrozen)", "T11\n(CQR frozen)", "T8 + std\nconformal"]
w90s = [t10["test_metrics"]["width_90"], t11["test_metrics"]["width_90"], conf_std["absolute_width_90"]]
w95s = [t10["test_metrics"]["width_95"], t11["test_metrics"]["width_95"], conf_std["absolute_width_95"]]

x = np.arange(len(methods))
bw = 0.35
fig, ax = plt.subplots(figsize=(9.5, 5.2))
b1 = ax.bar(x - bw/2, w90s, bw, color=TUM_BLUE, edgecolor="black", linewidth=0.5, label="90\\% interval width")
b2 = ax.bar(x + bw/2, w95s, bw, color=TUM_ORANGE, edgecolor="black", linewidth=0.5, label="95\\% interval width")

for b, v in zip(b1, w90s):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
for b, v in zip(b2, w95s):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("Mean interval width (veh/h)")
ax.set_title("CQR interval widths: T11 narrower at 90\\% than split conformal, similar at 95\\%")
ax.set_ylim(0, max(w95s) * 1.15)
ax.legend(loc="upper right", fontsize=9.5)
fig.tight_layout()
save_both(fig, OUT, "fig24_cqr_interval_widths")
print("F24 saved")

# ============================================================
# F25: CQR gate summary matrix (rows=trials, cols=gates)
# ============================================================
gates_t9 = [
    ("$R^2 \\geq 0.55$", t9_status_r2 := False),
    ("PICP$_{90} \\geq 85\\%$", True),
    ("PICP$_{95} \\geq 90\\%$", True),
]
# T9 uses different gates (3) — show overall status
# T10/T11 use 6 gates
gate_labels_long = [
    "$R^2 \\geq 0.57$",
    "PICP$_{90} \\geq 88\\%$",
    "PICP$_{95} \\geq 93\\%$",
    "Width$_{90}$ < Width$_{95}$",
    "$\\hat{Q}_{90} > 0$",
    "$\\hat{Q}_{95} > 0$",
]

t10_gates = [
    t10["gate_results"]["r2_ge_0.57"],
    t10["gate_results"]["picp90_ge_88"],
    t10["gate_results"]["picp95_ge_93"],
    t10["gate_results"]["width90_lt_width95"],
    t10["gate_results"]["q_hat_90_finite_pos"],
    t10["gate_results"]["q_hat_95_finite_pos"],
]
t11_gates = [
    t11["gate_results"]["r2_ge_0.57"],
    t11["gate_results"]["picp90_ge_88"],
    t11["gate_results"]["picp95_ge_93"],
    t11["gate_results"]["width90_lt_width95"],
    t11["gate_results"]["q_hat_90_finite_pos"],
    t11["gate_results"]["q_hat_95_finite_pos"],
]
# T9 has only 3 gates - map them where possible
t9_gates_disp = [False, True, True, None, None, None]  # only first 3 apply

# Build matrix
trials = ["T9 (hetero.)", "T10 (CQR unfrozen)", "T11 (CQR frozen)"]
matrix = np.array([t9_gates_disp, t10_gates, t11_gates], dtype=object)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xticks(np.arange(len(gate_labels_long)))
ax.set_yticks(np.arange(len(trials)))
ax.set_xticklabels(gate_labels_long, rotation=15, ha="right")
ax.set_yticklabels(trials)
ax.set_xlim(-0.5, len(gate_labels_long) - 0.5)
ax.set_ylim(-0.5, len(trials) - 0.5)

for i, row in enumerate(matrix):
    for j, val in enumerate(row):
        if val is None:
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, facecolor="#EEEEEE", edgecolor="black"))
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#888")
        elif val:
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, facecolor=PASS_GREEN, edgecolor="black", alpha=0.85))
            ax.text(j, i, "PASS", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
        else:
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, facecolor=FAIL_RED, edgecolor="black", alpha=0.85))
            ax.text(j, i, "FAIL", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")

# Right-side overall status column
ax_right = ax.twinx()
overall_status = [
    "PARTIAL POSITIVE\n(2/3 pass; R² fail)",
    "NEGATIVE\n(3/6 fail)",
    "POSITIVE\n(6/6 pass)",
]
overall_colors = [TUM_ORANGE, FAIL_RED, PASS_GREEN]
for i, (s, c) in enumerate(zip(overall_status, overall_colors)):
    ax.text(len(gate_labels_long) + 0.1, i, s, ha="left", va="center", fontsize=9.5, color=c, fontweight="bold")
ax_right.set_yticks([])
ax.set_xlim(-0.5, len(gate_labels_long) + 1.8)

ax.invert_yaxis()
ax.tick_params(axis="x", which="both", length=0)
ax.tick_params(axis="y", which="both", length=0)
ax.set_title("Go/no-go gate matrix for uncertainty-aware training extensions (T9, T10, T11)")
fig.tight_layout()
save_both(fig, OUT, "fig25_cqr_gate_summary")
print("F25 saved")
