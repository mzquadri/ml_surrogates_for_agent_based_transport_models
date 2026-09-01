"""Generate fig36_winkler_comparison: horizontal bar chart of Winkler 90 scores."""
import matplotlib as mpl, matplotlib.pyplot as plt

NEUTRAL = "#A5A5A5"; PRIMARY = "#5B9BD5"; TERTIARY = "#70AD47"
GRID = "#E5E5E5"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11, "axes.titlesize": 11, "axes.titleweight": "normal",
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

OUT = "C:/Users/zamin/Downloads/ml_surrogates_thesis_final/ml_surrogates_thesis_final/document/figures/new"

labels = ["Raw MC Dropout", "Standard Conformal", "Adaptive Conformal"]
values = [49.68, 35.78, 32.32]
colors = [NEUTRAL, PRIMARY, TERTIARY]

fig, ax = plt.subplots(figsize=(7, 3))
y_pos = list(range(len(labels)))
bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.6)

for bar, v in zip(bars, values):
    ax.text(v + 0.6, bar.get_y() + bar.get_height()/2, f"{v:.2f}",
            va="center", ha="left", fontsize=10, color="black")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Winkler interval score at 90 percent (lower is better)")
ax.set_title("Interval quality across calibration methods")
ax.set_xlim(0, max(values) * 1.15)
ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(f"{OUT}/fig36_winkler_comparison.pdf")
fig.savefig(f"{OUT}/fig36_winkler_comparison.png", dpi=300)
plt.close(fig)
print("saved fig36_winkler_comparison")
