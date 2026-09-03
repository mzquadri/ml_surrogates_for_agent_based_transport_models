import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

PT_PATH = (
    r"C:\Users\zamin\Downloads\Nazim\Thesis\Nazim"
    r"\ml_surrogates_thesis_final\code\data\train_data"
    r"\dist_not_connected_10k_1pct\datalist_batch_1.pt"
)
graphs = torch.load(PT_PATH, weights_only=False, map_location="cpu")
g = graphs[0]
x_np = g.x.numpy()  # [31635, 6]

BLUE = "#4878A8"
RED = "#D66B6B"
GREEN = "#5DA573"
GOLD = "#D4A843"
GREY = "#888888"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    }
)
OUT = r"C:\Users\zamin\Downloads\Nazim"

# ── 6 separate big figures, one per feature ───────────────────────────────────
feat_info = [
    # (col, name, color, x_label, unit_note, example_text, mean_explanation)
    (
        0,
        "VOL_BASE_CASE",
        BLUE,
        "Vehicles per hour  (veh/hr)",
        "Range: 0 to 1,596  |  Most roads: very low traffic",
        "Example:\n  Road A:  val = 10  →  only 10 cars/hr pass here  (quiet alley)\n"
        "  Road B:  val = 500 →  500 cars/hr pass here  (busy boulevard)\n"
        "  Road C:  val = 0   →  no cars at all  (pedestrian street)",
        "Mean = 50.9 means:\n  On average, 51 cars per hour pass through\n"
        "  one road segment in Paris under normal conditions.",
    ),
    (
        1,
        "CAPACITY_BASE",
        GREEN,
        "Vehicles per hour  (veh/hr)",
        "Range: 0 to 14,400  |  Higher = road can handle more traffic",
        "Example:\n  Narrow alley:      capacity = 200 veh/hr  (1 lane, slow)\n"
        "  Normal street:     capacity = 1,000 veh/hr\n"
        "  Large motorway:    capacity = 14,400 veh/hr  (many fast lanes)",
        "Mean = 1,029 means:\n  An average Paris road can hold ~1,029 cars per hour\n"
        "  before becoming congested.",
    ),
    (
        2,
        "CAPACITY_REDUCTION",
        RED,
        "Vehicles per hour removed  (veh/hr)",
        "Range: -4,800 to 0  |  Always zero or negative (capacity only goes down)",
        "Example:\n  val =    0  →  no change, road is fully open\n"
        "  val = -500  →  500 veh/hr removed  (e.g., one lane closed for repairs)\n"
        "  val = -4800 →  road almost fully closed  (major intervention)",
        "Mean = -56.9 means:\n  In this scenario, the average road lost only ~57 veh/hr of capacity.\n"
        "  Most roads (near 0) are untouched; a few roads have big reductions.",
    ),
    (
        3,
        "FREESPEED",
        GOLD,
        "Speed  (metres per second,  m/s)",
        "Range: 0 to 33.3 m/s  |  33.3 m/s = 120 km/h  |  8.3 m/s = 30 km/h",
        "Example:\n  val = 4.2 m/s  →  15 km/h  (pedestrian zone / very slow street)\n"
        "  val = 8.3 m/s  →  30 km/h  (typical Paris inner street)\n"
        "  val = 33.3 m/s →  120 km/h  (motorway / peripherique)",
        "Mean = 8.2 m/s  =  ~29 km/h means:\n  The average speed limit in Paris road network is ~29 km/h.\n"
        "  This matches Paris's widespread 30 km/h speed limit policy.",
    ),
    (
        4,
        "HIGHWAY_TYPE",
        GREY,
        "Road type code  (categorical)",
        "Values: -1 to 9  |  Each number = a different type of road",
        "Example codes:\n  -1 = unknown / unclassified\n"
        "   0 = motorway / peripherique  (highest speed)\n"
        "   3 = primary road  (main boulevard)\n"
        "   5 = residential street  (most common)\n"
        "   9 = footway / service road  (slowest)",
        "This column is EXCLUDED from model training.\n"
        "It is kept for reference only  (road type does not change\n"
        "between scenarios — only capacity and volume change).",
    ),
    (
        5,
        "LENGTH",
        BLUE,
        "Road segment length  (metres)",
        "Range: 4.2 m to 2,568 m  |  Most roads: short segments",
        "Example:\n  val =   10 m  →  tiny connector road / alley\n"
        "  val =   91 m  →  average Paris street segment\n"
        "  val = 2,568 m →  long motorway segment (e.g., part of peripherique)",
        "Mean = 91.6 m means:\n  The average road segment in Paris is about 91 metres long.\n"
        "  That is roughly the length of a typical city block.",
    ),
]

for col, name, clr, x_lbl, range_note, example_txt, mean_txt in feat_info:
    vals = x_np[:, col]
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Feature  {col}  of  x :     {name}",
        fontsize=16,
        fontweight="bold",
        color=clr,
        y=1.01,
    )

    # ── Left: histogram ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if name == "HIGHWAY_TYPE":
        unique, counts = np.unique(vals, return_counts=True)
        bars = ax.bar(
            unique, counts, color=clr, alpha=0.75, edgecolor="white", width=0.7
        )
        ax.set_xticks(unique)
        # label each bar with count
        for b, c in zip(bars, counts):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 50,
                f"{c:,}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )
    else:
        ax.hist(vals, bins=80, color=clr, alpha=0.75, edgecolor="white", linewidth=0.3)
        mean_val = np.mean(vals)
        ax.axvline(mean_val, color="#CC0000", linewidth=2.2, linestyle="--", zorder=5)
        # mean arrow + label
        y_max = ax.get_ylim()[1]
        ax.annotate(
            f"MEAN = {mean_val:.1f}",
            xy=(mean_val, y_max * 0.75),
            xytext=(mean_val + (vals.max() - vals.min()) * 0.08, y_max * 0.85),
            fontsize=10,
            color="#CC0000",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.5),
        )

    # ── AXIS LABELS — very explicit ──────────────────────────────────────────
    ax.set_xlabel(
        f"X AXIS  →  {x_lbl}\n(each tick = a value that a road can have for  {name})",
        fontsize=10,
        labelpad=10,
        color="#333333",
    )
    ax.set_ylabel(
        "Y AXIS  →  Number of roads\n"
        "(out of 31,635 total roads in Paris)\n"
        "that have that value",
        fontsize=10,
        labelpad=10,
        color="#333333",
    )

    # range note below title
    ax.set_title(range_note, fontsize=9, color="#666666", style="italic", pad=6)

    # ── Right: explanation panel ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_facecolor(BG)

    # example box
    ex_box = mpatches.FancyBboxPatch(
        (0.02, 0.48),
        0.96,
        0.48,
        transform=ax2.transAxes,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        edgecolor=clr,
        facecolor=CREAM,
        clip_on=False,
    )
    ax2.add_patch(ex_box)
    ax2.text(
        0.5,
        0.96,
        "Real-world examples",
        ha="center",
        va="top",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="bold",
        color=clr,
    )
    ax2.text(
        0.5,
        0.84,
        example_txt,
        ha="center",
        va="top",
        transform=ax2.transAxes,
        fontsize=9.2,
        color="#222222",
        family="monospace",
    )

    # mean explanation box
    mean_box = mpatches.FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.42,
        transform=ax2.transAxes,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        edgecolor="#CC0000",
        facecolor="#FFF0F0",
        clip_on=False,
    )
    ax2.add_patch(mean_box)
    ax2.text(
        0.5,
        0.43,
        "What does the MEAN line mean?",
        ha="center",
        va="top",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#CC0000",
    )
    ax2.text(
        0.5,
        0.31,
        mean_txt,
        ha="center",
        va="top",
        transform=ax2.transAxes,
        fontsize=9.2,
        color="#222222",
    )

    plt.tight_layout()
    safe_name = name.lower()
    plt.savefig(
        f"{OUT}\\detail_2_{safe_name}.png", dpi=150, bbox_inches="tight", facecolor=BG
    )
    plt.close()
    print(f"Saved detail_2_{safe_name}.png")

print("All 6 feature plots done.")
