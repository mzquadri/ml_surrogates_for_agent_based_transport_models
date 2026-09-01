import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mc
import matplotlib.lines as mlines
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
N = 8
TRIALS = list(range(1, N + 1))
R2 = [0.7860, 0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
MAE = [2.9716, 4.3277, 5.9897, 6.0795, 4.2421, 4.3242, 4.0601, 3.9573]
RMSE = [5.3955, 8.1505, 10.2701, 10.1508, 7.7779, 8.0609, 7.5343, 7.1183]
PEARSON = [0.8875, 0.7185, 0.6391, 0.6336, 0.7468, 0.7262, 0.7409, 0.7726]
SPEARMAN = [0.5452, 0.2011, 0.2807, 0.2723, 0.2276, 0.2006, 0.2267, 0.2929]
ELENA_R2 = 0.78

HPARAM_HEADERS = ["Trial", "Batch", "LR", "Dropout", "W.Loss", "Arch.", "Split"]
HPARAM_ROWS = [
    ["T1", "32", "0.001", "0.30", "No", "Linear", "80/15/5"],
    ["T2", "16", "0.0005", "0.30", "No", "GATConv", "80/15/5"],
    ["T3", "16", "0.0005", "0.00", "Yes", "GATConv", "80/15/5"],
    ["T4", "16", "0.0005", "0.30", "Yes", "GATConv", "80/15/5"],
    ["T5", "8", "0.0005", "0.30", "No", "GATConv", "80/15/5"],
    ["T6", "8", "0.0003", "0.30", "No", "GATConv", "80/15/5"],
    ["T7", "8", "0.0006", "0.30", "No", "GATConv", "80/10/10"],
    ["T8", "8", "0.0005", "0.20", "No", "GATConv", "80/10/10"],
]

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
CREAM = "#F5F2EC"
REF_RED = "#A31515"

# ColorBrewer-inspired academic palette — proven for publication figures
C_R2 = "#2166AC"  # strong cobalt blue
C_MAE = "#B2182B"  # strong crimson
C_RMSE = "#1A7034"  # strong forest green
C_PEARSON = "#6A3D9A"  # strong purple
C_SPEAR = "#9B6B00"  # strong amber


def lighten(hex_color, factor=0.42):
    """Blend hex_color toward white (factor: 0=original, 1=white)."""
    r, g, b = mc.to_rgb(hex_color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10,
    }
)

# ── Figure & GridSpec ─────────────────────────────────────────────────────────
# Compact: width=34, height=14 → each row ≈ 5 in, gap ≈ 2 in
fig = plt.figure(figsize=(34, 13))
fig.patch.set_facecolor(BG)

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(
    0.5,
    0.985,
    "Model Training Trials  —  PointNet + Transformer + GAT",
    ha="center",
    va="top",
    fontsize=22,
    fontweight="bold",
    color="#0D0D0D",
)
fig.text(
    0.5,
    0.966,
    "8 trials  |  Paris traffic network  |  1,000 disruption scenarios"
    "  |  31,635 road segments per graph",
    ha="center",
    va="top",
    fontsize=11,
    color="#555555",
    style="italic",
)
fig.add_artist(
    mlines.Line2D(
        [0.06, 0.94],
        [0.956, 0.956],
        transform=fig.transFigure,
        color="#C8C8C8",
        linewidth=0.9,
    )
)

gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    height_ratios=[1, 1],
    hspace=0.35,
    wspace=0.36,
    top=0.862,  # 1.3 in clear gap below subtitle — no overlap
    bottom=0.068,
    left=0.062,
    right=0.976,
)

x = np.arange(N)
XLABELS = [f"T{t}" for t in TRIALS]


# ── Bar chart helper ──────────────────────────────────────────────────────────
def make_chart(
    row, col, values, color, panel, title, ylabel, direction="up", val_fmt="{:.3f}"
):
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor(BG)

    # Full box frame — journal figure style
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.8)
        spine.set_visible(True)

    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, color="#BBBBBB", direction="in")

    best_idx = int(np.argmax(values) if direction == "up" else np.argmin(values))

    # Best bar = full color; others = lighter
    bar_colors = [color if i == best_idx else lighten(color, 0.42) for i in range(N)]

    bars = ax.bar(
        x,
        values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.60,
        zorder=3,
    )

    max_v = max(values)
    ax.set_ylim(0, max_v * 1.26)

    # Value labels
    for i, (b, v) in enumerate(zip(bars, values)):
        is_best = i == best_idx
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max_v * 0.010,
            val_fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0D0D0D" if is_best else "#555555",
            fontweight="bold" if is_best else "normal",
        )

    # "best" badge — just above the best bar's value label
    bx = bars[best_idx].get_x() + bars[best_idx].get_width() / 2
    ax.text(
        bx,
        values[best_idx] + max_v * 0.068,
        "best",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=color,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor=BG,
            edgecolor=color,
            linewidth=1.3,
            alpha=0.97,
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=10.5, color="#333333")
    ax.set_ylabel(ylabel, fontsize=10.5, color="#444444", labelpad=7)

    # Title with panel letter inline — avoids any overlap with figure subtitle
    ax.set_title(
        f"{panel}  {title}",
        fontsize=13.5,
        fontweight="bold",
        color="#111111",
        loc="left",
        pad=7,
    )

    # Direction label — italic, top-right corner inside axes
    dir_label = "↑  higher is better" if direction == "up" else "↓  lower is better"
    ax.text(
        0.985,
        0.970,
        dir_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#999999",
        style="italic",
    )

    return ax, bars


# ── Row 0 — R², MAE, RMSE ────────────────────────────────────────────────────
ax_r2, _ = make_chart(
    0,
    0,
    R2,
    C_R2,
    panel="(a)",
    title="R²  Score",
    ylabel="R²",
    direction="up",
    val_fmt="{:.3f}",
)
# Elena reference line
ax_r2.axhline(ELENA_R2, color=REF_RED, linewidth=1.7, linestyle="--", zorder=6)
ax_r2.text(
    7.45,
    ELENA_R2 + max(R2) * 0.028,
    f"Elena et al.  R² = {ELENA_R2}",
    ha="right",
    va="bottom",
    fontsize=9,
    color=REF_RED,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.26",
        facecolor="#FFF5F5",
        edgecolor=REF_RED,
        linewidth=1.1,
        alpha=0.96,
    ),
)

make_chart(
    0,
    1,
    MAE,
    C_MAE,
    panel="(b)",
    title="MAE  [veh/hr]",
    ylabel="Mean Absolute Error  [veh/hr]",
    direction="down",
    val_fmt="{:.2f}",
)

make_chart(
    0,
    2,
    RMSE,
    C_RMSE,
    panel="(c)",
    title="RMSE  [veh/hr]",
    ylabel="Root Mean Squared Error  [veh/hr]",
    direction="down",
    val_fmt="{:.2f}",
)

# ── Row 1 — Pearson, Spearman, Table ─────────────────────────────────────────
make_chart(
    1,
    0,
    PEARSON,
    C_PEARSON,
    panel="(d)",
    title="Pearson  r",
    ylabel="Pearson  r",
    direction="up",
    val_fmt="{:.3f}",
)

make_chart(
    1,
    1,
    SPEARMAN,
    C_SPEAR,
    panel="(e)",
    title="Spearman  ρ",
    ylabel="Spearman  ρ",
    direction="up",
    val_fmt="{:.3f}",
)

# ── (f) Hyperparameter table ──────────────────────────────────────────────────
ax_ht = fig.add_subplot(gs[1, 2])
ax_ht.set_facecolor(BG)
ax_ht.set_xticks([])
ax_ht.set_yticks([])
for sp in ax_ht.spines.values():
    sp.set_visible(False)

ax_ht.set_title(
    "(f)  Hyperparameter Configuration",
    fontsize=13.5,
    fontweight="bold",
    color="#111111",
    loc="left",
    pad=7,
)

tbl = ax_ht.table(
    cellText=HPARAM_ROWS,
    colLabels=HPARAM_HEADERS,
    cellLoc="center",
    bbox=[0.0, 0.05, 1.0, 0.90],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)

N_COLS = len(HPARAM_HEADERS)

# Header
for j in range(N_COLS):
    cell = tbl[0, j]
    cell.set_facecolor("#1C2B3E")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#FFFFFF")
    cell.set_linewidth(1.0)

# Data rows
for i, row_data in enumerate(HPARAM_ROWS):
    row_bg = CREAM if i % 2 == 0 else BG
    for j, val in enumerate(row_data):
        cell = tbl[i + 1, j]
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.7)
        if j == 4 and val == "Yes":
            cell.set_facecolor("#FDECEA")
            cell.set_text_props(color="#8B0000", fontweight="bold")
        elif j == 5 and val == "Linear":
            cell.set_facecolor("#FEF5E0")
            cell.set_text_props(color="#7A5000", fontweight="bold")
        elif j == 6 and val == "80/10/10":
            cell.set_facecolor("#E8F4EC")
            cell.set_text_props(color="#1A7034", fontweight="bold")
        elif j == 0:
            cell.set_facecolor("#EDF2FA")
            cell.set_text_props(color="#2166AC", fontweight="bold")
        else:
            cell.set_facecolor(row_bg)
            cell.set_text_props(color="#333333")

ax_ht.text(
    0.5,
    0.01,
    "Yellow = Linear arch.   |   Red = Weighted loss   |   Green = 80/10/10 split",
    transform=ax_ht.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.5,
    color="#888888",
    style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = r"C:\Users\zamin\Downloads\Nazim\trials_dashboard.png"
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved:", OUT)
