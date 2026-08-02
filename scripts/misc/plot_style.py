"""Shared TUM-style plotting module for thesis figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt

TUM_BLUE = "#005293"
TUM_ORANGE = "#E37222"
TUM_LIGHTBLUE = "#64A0C8"
TUM_GREEN = "#A2AD00"
TUM_DARK = "#333333"
TUM_GRAY = "#808080"
IDEAL_GREEN = "#4A7C59"
FAIL_RED = "#C8102E"
PASS_GREEN = "#3E8914"

PALETTE = [TUM_BLUE, TUM_ORANGE, TUM_LIGHTBLUE, TUM_GREEN, TUM_DARK, "#8B2C6F", "#5B8FA8"]


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_both(fig, outdir, stem):
    """Save figure as PDF and PNG (300dpi) with matching filenames."""
    import os
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{stem}.pdf")
    png = os.path.join(outdir, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png
