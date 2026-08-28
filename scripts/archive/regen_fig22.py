"""Regenerate fig22 (CQR R^2 progression) with the canonical 3-trial set
(T8, T10, T11) using the standard TUM palette to match the rest of the thesis
figures (gen_batch1 / gen_batch2 conventions: TUM_GRAY for the architectural
reference, TUM_ORANGE for the failed variant, TUM_BLUE for the primary/passing
variant)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from plot_style import set_style, save_both, TUM_BLUE, TUM_ORANGE, TUM_GRAY

set_style()
OUT = "../../../document/figures/new"
BASE = "../../data/TR-C_Benchmarks/"

t10 = json.load(open(BASE + "point_net_transf_gat_10th_trial_cqr/cqr_results/cqr_metrics.json"))
t11 = json.load(open(BASE + "point_net_transf_gat_11th_trial_cqr_frozen/cqr_results/cqr_metrics.json"))

T8_R2 = 0.5957
T10_R2 = t10["test_metrics"]["r2_midpoint"]   # 0.4057
T11_R2 = t11["test_metrics"]["r2_midpoint"]   # 0.5835
GATE_R2 = 0.57

labels = [
    "T8\n(MSE baseline)",
    "T10\n(CQR, unfrozen,\nlr $= 5{\\times}10^{-4}$)",
    "T11\n(CQR, frozen,\nsame lr, 134 params)",
]
r2s = [T8_R2, T10_R2, T11_R2]
status = ["BASELINE", "FAIL", "PASS"]
# Light TUM accent palette to match the soft tones used elsewhere in the
# thesis: pale gray for the baseline, soft apricot for the failed CQR variant,
# and TUM Accent Light Blue for the passing variant.
TUM_LIGHTGRAY = "#DAD7CB"          # TUMAccentGray from main.tex settings
TUM_PALE_ORANGE = "#F4B183"         # softened TUM_ORANGE
TUM_ACCENTLIGHTBLUE = "#98C6EA"     # TUMAccentLightBlue from main.tex settings
colors = [TUM_LIGHTGRAY, TUM_PALE_ORANGE, TUM_ACCENTLIGHTBLUE]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
bars = ax.bar(labels, r2s, color=colors, edgecolor="#555555", linewidth=0.6)
for b, v, s in zip(bars, r2s, status):
    ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.4f}",
            ha="center", fontsize=10, fontweight="bold", color="#222222")
    ax.text(b.get_x() + b.get_width()/2, v/2, s, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#888888", alpha=0.85))

ax.axhline(GATE_R2, color="#888888", linestyle="--", linewidth=1.0,
           label=f"$R^2 \\geq {GATE_R2}$ gate")
ax.text(2.45, GATE_R2 + 0.008, "Gate", color="#666666", fontsize=9, ha="right")

ax.set_ylabel("Test $R^2$ (midpoint)")
ax.set_title("CQR R$^2$ progression: only frozen-backbone training (T11) preserves accuracy")
ax.set_ylim(0, 0.72)
ax.legend(loc="upper left", fontsize=9.5)
fig.tight_layout()
save_both(fig, OUT, "fig22_cqr_r2_progression")
print(f"F22 regenerated (TUM palette): T8 = {T8_R2:.4f}, T10 = {T10_R2:.4f}, T11 = {T11_R2:.4f}")
