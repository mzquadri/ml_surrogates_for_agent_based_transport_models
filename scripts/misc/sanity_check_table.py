"""Black-and-white sanity-check / ablation summary table.

Renders an 8-row table with columns:
  #  |  Question  |  Trial(s) used  |  Result  |  What it proves  |  Confounders

Output: sanity_check_table.pdf and .png in the figures/new/ directory.
Pure matplotlib, no colour, no dependencies on the project plot_style module.
"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "document", "figures", "new"
))
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = ["#", "Sanity-check question", "Existing trial(s)",
           "Result", "What it proves", "Remaining confounders"]

ROWS = [
    ["1",
     "Does dropout hurt point\naccuracy?",
     "T1 (no dropout) vs T8\n(dropout 0.2); DE members\n(dropout 0.2)",
     "T1 R²=0.786;\nDE members 0.640–0.650;\nT8 0.596",
     "Suggestive: dropout has a\nmeasurable cost, but smaller\nthan T1−T8 alone implies",
     "T1 used a different test\nset (50 graphs, 80/15/5);\nnot like-for-like"],

    ["2",
     "Does freezing the backbone\nhelp CQR?",
     "T10 (unfrozen, lr 5e-4)\nvs T11 (frozen, lr 5e-4)\n— same head, loss, data, lr",
     "R² 0.406 → 0.584;\n3/6 → 6/6 gates pass",
     "YES — cleanest single-knob\nablation in the thesis",
     "Only one lr; only\nPointNetTransfGAT;\nonly MSE-pretrained backbone"],

    ["3",
     "Does ensembling improve\naccuracy?",
     "5 DE members vs ensemble\nmean",
     "Members 0.640–0.650;\nensemble mean 0.684",
     "YES — classic averaging\neffect; mean above every\nmember",
     "T8 (0.596) sits below all\nmembers; “+14.8% over T8”\npartly a T8-checkpoint artifact"],

    ["4",
     "Does raw MC Dropout give\ncalibrated intervals?",
     "T8 calibration audit\n(S=30, 3.16M nodes);\nreplicated on T7",
     "k₉₅ = 11.66 (ideal 1.96);\n95% nominal → 54.85% emp.",
     "NO — raw σ is a ranking\nsignal, not an interval\nestimator",
     "Tested only on PointNet-\nTransfGAT checkpoints; absolute\nk₉₅ may shift on re-train"],

    ["5",
     "Does conformal prediction\nrestore coverage?",
     "T8 + split conformal\n(50 calib / 50 eval);\n+ adaptive conformal",
     "PICP₉₀ = 90.02%; conditional\ndecile [59,98] → [83.7,96.4]",
     "YES for marginal coverage;\npartially for conditional",
     "Exchangeability holds at\nscenario level only;\nconditional drift remains"],

    ["6",
     "Does graph structure add\nvalue over a flat NN?",
     "T8 (GNN, R²=0.5957)\nvs MLP (R²=0.4928)",
     "ΔR² ≈ 0.10",
     "YES — graph topology\ncontributes ~0.10 R²",
     "MLP at sklearn defaults,\nnot tuned"],

    ["7",
     "Does σ rank errors\nusefully?",
     "T8 MC Dropout: aggregate\nρ; AUROC at top-10%;\nreplicated on T7",
     "ρ = 0.482 (CI [0.460,\n0.469]); AUROC = 0.755",
     "YES on aggregate;\nreplicates on T7 (ρ=0.444)",
     "Collapses within Q4\n(ρ = 0.10) — fails on the\npolicy-relevant tail"],

    ["8",
     "Does MC + DE σ quadrature\nbeat MC alone?",
     "Experiment A:\nσ = √(σ_MC² + σ_DE²)",
     "ρ = 0.491 ≈ MC alone\n(0.491)",
     "NO — signals correlated;\nno measurable gain",
     "Only quadrature tested;\nlearned weighting / max-rule\nnot explored"],
]

def render_table():
    fig_w = 16.0
    fig_h = 9.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    col_widths = [0.025, 0.16, 0.16, 0.18, 0.20, 0.20]

    table = ax.table(
        cellText=ROWS,
        colLabels=HEADERS,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 3.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        cell.set_facecolor("white")
        cell.PAD = 0.04

        if r == 0:
            cell.set_text_props(weight="bold", color="black", ha="center", va="center")
            cell.set_linewidth(1.2)
            cell.set_height(cell.get_height() * 0.7)
        else:
            cell.set_text_props(va="center", ha="left")
            if c == 0:
                cell.set_text_props(ha="center", va="center", weight="bold")

    ax.set_title("Sanity-Check / Ablation Summary  —  what existing trials prove and do not prove",
                 fontsize=12, weight="bold", pad=14, loc="left")

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "sanity_check_table.pdf")
    png_path = os.path.join(OUT_DIR, "sanity_check_table.png")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=220)
    plt.close()
    print(f"Saved:\n  {pdf_path}\n  {png_path}")

if __name__ == "__main__":
    render_table()
