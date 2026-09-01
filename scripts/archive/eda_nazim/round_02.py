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
vals = graphs[0].x[:, 0].numpy()  # VOL_BASE_CASE

BLUE = "#4878A8"
CREAM = "#F5F0E8"
BG = "#FFFFFF"

plt.rcParams.update(
    {"font.family": "serif", "figure.facecolor": BG, "axes.facecolor": BG}
)

fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor(BG)

# ── Main histogram (left, wider) ─────────────────────────────────────────────
ax = fig.add_axes([0.07, 0.18, 0.56, 0.65])
ax.set_facecolor(BG)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.hist(vals, bins=80, color=BLUE, alpha=0.80, edgecolor="white", linewidth=0.3)

mean_val = np.mean(vals)
ax.axvline(mean_val, color="#CC0000", linewidth=2.2, linestyle="--", zorder=5)
ax.text(
    mean_val + 15,
    ax.get_ylim()[1] * 0.82,
    f"Mean = {mean_val:.1f} veh/hr",
    fontsize=10.5,
    color="#CC0000",
    fontweight="bold",
)

# X axis label
ax.set_xlabel(
    "X AXIS  --  Gaariyon ki taadaad  (vehicles per hour)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)

# Y axis label
ax.set_ylabel(
    "Y AXIS  --  Kitni sadkon par\nyeh traffic hai\n(out of 31,635)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)

ax.set_title(
    "Round 2 / 12   --   Feature 1 :   VOL_BASE_CASE\n"
    '"Normal halat mein, is sadak se kitni gaariyan guzarti hain?"',
    fontsize=13,
    fontweight="bold",
    color="#222222",
    pad=10,
)

# ── Right: explanation panel ──────────────────────────────────────────────────
ax2 = fig.add_axes([0.67, 0.05, 0.31, 0.88])
ax2.axis("off")
ax2.set_facecolor(BG)

# --- What is X axis ---
r1 = mpatches.FancyBboxPatch(
    (0.02, 0.68),
    0.96,
    0.29,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor=BLUE,
    facecolor=CREAM,
    clip_on=False,
)
ax2.add_patch(r1)
ax2.text(
    0.5,
    0.95,
    "X Axis kya hai?",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color=BLUE,
)
ax2.text(
    0.5,
    0.87,
    "Har sadak par ek ghante mein\n"
    "kitni gaariyan guzarti hain.\n\n"
    "0  =  koi gari nahi  (pedestrian road)\n"
    "50  =  50 gaariyan/ghanta\n"
    "1596  =  bahut busy road",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

# --- What is Y axis ---
r2 = mpatches.FancyBboxPatch(
    (0.02, 0.36),
    0.96,
    0.29,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor="#5DA573",
    facecolor=CREAM,
    clip_on=False,
)
ax2.add_patch(r2)
ax2.text(
    0.5,
    0.63,
    "Y Axis kya hai?",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color="#5DA573",
)
ax2.text(
    0.5,
    0.55,
    "Kitni sadkon ka traffic\nus value ke barabar hai.\n\n"
    "Bar jitni UNCHI hai --\nutni ZYAADA sadkon par\nyeh traffic level hai.",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

# --- Mean explanation ---
r3 = mpatches.FancyBboxPatch(
    (0.02, 0.03),
    0.96,
    0.30,
    transform=ax2.transAxes,
    boxstyle="round,pad=0.03",
    linewidth=1.8,
    edgecolor="#CC0000",
    facecolor="#FFF4F4",
    clip_on=False,
)
ax2.add_patch(r3)
ax2.text(
    0.5,
    0.31,
    "Laal line (Mean) kya hai?",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=11,
    fontweight="bold",
    color="#CC0000",
)
ax2.text(
    0.5,
    0.23,
    f"Mean = {mean_val:.1f} veh/hr\n\n"
    "Matlab: Paris ki ek\n"
    '"average" sadak par sirf\n'
    "~51 gaariyan/ghanta hoti hain.\n"
    "Zyaadatar sadken quiet hain!",
    ha="center",
    va="top",
    transform=ax2.transAxes,
    fontsize=10,
    color="#222222",
    linespacing=1.5,
)

plt.savefig(
    r"C:\Users\zamin\Downloads\Nazim\round_02_vol_base_case.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Saved round_02_vol_base_case.png")
