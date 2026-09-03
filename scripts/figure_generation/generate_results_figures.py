#!/usr/bin/env python
"""Figures for the surrogate's accuracy and the uncertainty layer built on it.

Everything is computed from artifacts tracked in this repository, so the figures
regenerate from a plain clone with no release downloads. Numbers shown here are
the same ones `scripts/verify_headline_results.py` asserts.

Usage
-----
    python scripts/figure_generation/generate_results_figures.py

Output: docs/figures/results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from thesis_style import COLORS  # noqa: E402
from artifact_paths import resolve  # noqa: E402

OUT = REPO / "docs" / "figures" / "results"
T8 = "point_net_transf_gat_8th_trial_lower_dropout"
T7 = "point_net_transf_gat_7th_trial_80_10_10_split"
NPG = 31_635

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def load(trial):
    m = np.load(resolve(f"{trial}/uq_results/mc_dropout_full_100graphs_mc30.npz"))
    d = np.load(resolve(f"{trial}/uq_results/deterministic_full_100graphs.npz"))
    return (m["targets"].astype(np.float64), m["predictions"].astype(np.float64),
            m["uncertainties"].astype(np.float64), d["predictions"].astype(np.float64))


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png  -- {caption}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_accuracy(t, pdet):
    """What the surrogate gets right, and how it fails."""
    res = pdet - t
    r2 = 1 - (res ** 2).sum() / ((t - t.mean()) ** 2).sum()
    mae, rmse = np.abs(res).mean(), np.sqrt((res ** 2).mean())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    lim = np.percentile(np.abs(t), 99.7)
    hb = ax.hexbin(t, pdet, gridsize=110, extent=(-lim, lim, -lim, lim),
                   bins="log", cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([-lim, lim], [-lim, lim], color=COLORS["coral"], lw=1.2, ls="--")
    ax.set_xlabel("simulated change (veh/h)")
    ax.set_ylabel("predicted change (veh/h)")
    ax.set_title("Prediction against simulation", fontweight="600", color=COLORS["dgray"])
    ax.text(0.04, 0.95, f"R² = {r2:.4f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, va="top", fontsize=8.6, family="monospace",
            color=COLORS["slate"])
    plt.colorbar(hb, ax=ax, fraction=0.045, pad=0.02, label="links (log)")

    ax = axes[1]
    ax.hist(res[np.abs(res) < 40], bins=140, color=COLORS["blue"], log=True)
    ax.axvline(0, color=COLORS["coral"], lw=1.2, ls="--")
    ax.set_xlabel("residual: predicted − simulated (veh/h)")
    ax.set_ylabel("links (log)")
    ax.set_title("Residual distribution", fontweight="600", color=COLORS["dgray"])
    ax.text(0.03, 0.95, f"mean {res.mean():+.3f}\nmedian {np.median(res):+.3f}",
            transform=ax.transAxes, va="top", fontsize=8.6, family="monospace",
            color=COLORS["slate"])

    ax = axes[2]
    # Error against the size of the true change: where does the model struggle?
    edges = np.percentile(np.abs(t), np.linspace(0, 100, 21))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(np.abs(t), edges) - 1, 0, len(edges) - 2)
    centres = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(np.abs(res)[idx == b]) for b in range(len(centres))])
    p90 = np.array([np.percentile(np.abs(res)[idx == b], 90) for b in range(len(centres))])
    ax.plot(centres, med, color=COLORS["blue_dk"], lw=1.8, label="median |error|")
    ax.plot(centres, p90, color=COLORS["coral"], lw=1.4, ls="--", label="90th percentile")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("|simulated change| (veh/h)")
    ax.set_ylabel("|error| (veh/h)")
    ax.set_title("Error grows with the size of the effect",
                 fontweight="600", color=COLORS["dgray"])
    ax.legend(fontsize=8.4)

    fig.suptitle("Trial 8 surrogate accuracy on 100 held-out scenarios",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.05,
             "3,163,500 link-level predictions. The model captures the bulk of the response but systematically under-predicts the largest changes, "
             "which is\nwhere a point prediction alone is least safe to act on — the motivation for the uncertainty layer.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "01_accuracy", "how good the surrogate is and how it fails")


def fig_sigma_vs_error(t, pmc, sig):
    """Does the uncertainty track the error?"""
    err = np.abs(pmc - t)
    rho = stats.spearmanr(sig, err).statistic

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    # Clip to the bulk of the joint distribution: the long thin tails otherwise
    # leave most of the panel empty and wash the dense region out.
    xl, yl = np.percentile(sig, 98), np.percentile(err, 96)
    hb = ax.hexbin(sig, err, gridsize=90, extent=(0, xl, 0, yl), bins="log",
                   cmap="Oranges", mincnt=1, linewidths=0)
    # Binned median makes the trend legible rather than merely present.
    edges = np.linspace(0, xl, 31)
    idx = np.clip(np.digitize(sig, edges) - 1, 0, len(edges) - 2)
    cx = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(err[idx == b]) if (idx == b).any() else np.nan
                    for b in range(len(cx))])
    ax.plot(cx, med, color=COLORS["blue_dk"], lw=2.2, label="median |error| per σ bin")
    ax.legend(fontsize=8.4, loc="upper left")
    ax.set_xlim(0, xl); ax.set_ylim(0, yl)
    ax.set_xlabel("MC Dropout σ (veh/h)")
    ax.set_ylabel("|error| (veh/h)")
    ax.set_title("Uncertainty against error  (central 98% of σ)",
                 fontweight="600", color=COLORS["dgray"])
    ax.text(0.96, 0.06, f"Spearman ρ = {rho:.4f}\n(all 3,163,500 links)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
            family="monospace", color=COLORS["slate"])
    plt.colorbar(hb, ax=ax, fraction=0.045, pad=0.02, label="links (log)")

    ax = axes[1]
    # Decile view: the monotone trend is the property that matters operationally.
    q = np.percentile(sig, np.linspace(0, 100, 11))
    b = np.clip(np.digitize(sig, q) - 1, 0, 9)
    med = [np.median(err[b == i]) for i in range(10)]
    p90 = [np.percentile(err[b == i], 90) for i in range(10)]
    x = np.arange(1, 11)
    ax.bar(x, med, color=COLORS["blue"], label="median |error|")
    ax.plot(x, p90, color=COLORS["coral"], lw=1.6, marker="o", ms=4,
            label="90th percentile |error|")
    ax.set_xticks(x)
    ax.set_xlabel("σ decile  (1 = most confident)")
    ax.set_ylabel("|error| (veh/h)")
    ax.set_title("Error by uncertainty decile", fontweight="600", color=COLORS["dgray"])
    ax.legend(fontsize=8.4)
    ax.text(0.03, 0.94, f"decile 10 median error is\n{med[-1] / med[0]:.1f}× decile 1",
            transform=ax.transAxes, va="top", fontsize=8.4, color=COLORS["slate"])

    fig.suptitle("Does MC Dropout σ know where the model is wrong?",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.05,
             "σ ranks errors well: error rises monotonically across σ deciles. It is a ranking signal, not a calibrated scale — the raw magnitudes are "
             "far too small,\nwhich the next figure shows and the calibration step corrects.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "02_uncertainty_vs_error", "sigma ranks error but is not a scale")


def fig_calibration(t, pmc, sig):
    """Raw sigma badly undercovers; temperature scaling fixes the average."""
    err = np.abs(pmc - t)
    cut = 20 * NPG
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    z = stats.norm.ppf(0.5 + levels / 2)

    def cov(e, s, T=1.0):
        return np.array([(e <= T * s * zi).mean() for zi in z])

    def ece(e, s, T=1.0):
        return float(np.abs(cov(e, s, T) - levels).mean())

    coarse = np.linspace(0.5, 6.0, 551)
    Tb = min(coarse, key=lambda T: ece(err[:cut], sig[:cut], T))
    Tb = min(np.linspace(Tb - 0.02, Tb + 0.02, 81),
             key=lambda T: ece(err[:cut], sig[:cut], T))
    ev_e, ev_s = err[cut:], sig[cut:]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=COLORS["slate"], lw=1.1, ls="--", label="perfect calibration")
    ax.plot(levels, cov(ev_e, ev_s), color=COLORS["coral"], lw=2, marker="o", ms=4.5,
            label=f"raw σ  (ECE {ece(ev_e, ev_s):.3f})")
    ax.plot(levels, cov(ev_e, ev_s, Tb), color=COLORS["green"], lw=2, marker="s", ms=4.5,
            label=f"σ × T,  T = {Tb:.3f}  (ECE {ece(ev_e, ev_s, Tb):.3f})")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability diagram", fontweight="600", color=COLORS["dgray"])
    ax.legend(fontsize=8.4, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    w = 0.36
    x = np.arange(len(levels))
    ax.bar(x - w / 2, 100 * (cov(ev_e, ev_s) - levels), w, color=COLORS["coral"],
           label="raw σ")
    ax.bar(x + w / 2, 100 * (cov(ev_e, ev_s, Tb) - levels), w, color=COLORS["green"],
           label="after scaling")
    ax.axhline(0, color=COLORS["slate"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(l * 100)}" for l in levels])
    ax.set_xlabel("nominal level (%)")
    ax.set_ylabel("coverage gap (percentage points)")
    ax.set_title("How far off, at each level", fontweight="600", color=COLORS["dgray"])
    ax.legend(fontsize=8.4)

    fig.suptitle("Calibrating MC Dropout σ  (protocol graph20_80_v1)",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.06,
             "T is fitted on the first 20 test graphs and scored on the remaining 80. Raw σ covers only ~49% of errors at the nominal 90% level; one scalar "
             "removes\nmost of that gap. This protocol is not the one the thesis reports — the two must not be pooled, see CORRIGENDUM C3.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "03_calibration", "raw sigma undercovers; one scalar fixes it")


def fig_selective_and_detection(t, pmc, sig, pdet):
    """The two decisions the uncertainty actually supports."""
    err = np.abs(pmc - t)
    err_det = np.abs(pdet - t)
    order = np.argsort(sig, kind="stable")

    fracs = np.linspace(0.05, 1.0, 40)
    mae = [err[order[: int(f * err.size)]].mean() for f in fracs]
    base = err.mean()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    ax = axes[0]
    ax.plot(100 * fracs, mae, color=COLORS["blue_dk"], lw=2)
    ax.axhline(base, color=COLORS["slate"], lw=1.1, ls="--")
    ax.text(99, base, " no selection", va="bottom", ha="right", fontsize=8.2,
            color=COLORS["slate"])
    for f in (0.5, 0.9):
        m = err[order[: int(f * err.size)]].mean()
        ax.plot([100 * f], [m], "o", color=COLORS["coral"], ms=6)
        ax.annotate(f"{int(f*100)}% kept\nMAE {m:.2f}  ({100*(1-m/base):.1f}%)",
                    (100 * f, m), textcoords="offset points", xytext=(6, 10),
                    fontsize=8, color=COLORS["coral"])
    ax.set_xlabel("% of links retained (most confident first)")
    ax.set_ylabel("MAE on retained links (veh/h)")
    ax.set_title("Selective prediction", fontweight="600", color=COLORS["dgray"])

    ax = axes[1]
    from sklearn.metrics import roc_curve, roc_auc_score
    for pct, c in [(10, COLORS["coral"]), (20, COLORS["blue"])]:
        y = (err_det >= np.percentile(err_det, 100 - pct)).astype(np.int8)
        fpr, tpr, _ = roc_curve(y, sig)
        s = max(1, len(fpr) // 4000)
        ax.plot(fpr[::s], tpr[::s], color=c, lw=1.9,
                label=f"top-{pct}% errors  AUROC {roc_auc_score(y, sig):.4f}")
    ax.plot([0, 1], [0, 1], color=COLORS["slate"], lw=1, ls="--", label="random  0.500")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Error detection", fontweight="600", color=COLORS["dgray"])
    ax.legend(fontsize=8.2, loc="lower right")

    ax = axes[2]
    cut = 20 * NPG
    cal, ev = err_det[:cut], err_det[cut:]
    lv = [0.5, 0.7, 0.8, 0.9, 0.95]
    got, width = [], []
    for a in lv:
        k = int(np.ceil((cal.size + 1) * a))
        q = np.sort(cal)[k - 1]
        got.append(100 * (ev <= q).mean()); width.append(2 * q)
    x = np.arange(len(lv))
    ax.bar(x, got, color=COLORS["green"], width=0.5)
    for i, (g, w) in enumerate(zip(got, width)):
        ax.text(i, g + 1.2, f"{g:.2f}%\n±{w/2:.1f}", ha="center", fontsize=7.8,
                color=COLORS["slate"])
    ax.plot(x, [100 * a for a in lv], "o--", color=COLORS["coral"], ms=5,
            label="nominal")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(a*100)}%" for a in lv])
    ax.set_ylim(0, 112)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage (%)")
    ax.set_title("Split conformal coverage", fontweight="600", color=COLORS["dgray"])
    ax.legend(fontsize=8.2, loc="lower right")

    fig.suptitle("What the uncertainty is good for", fontsize=13,
                 fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.09,
             "Left and centre need only the ranking of σ, which is why they work despite σ being uncalibrated. Right needs a calibrated interval, which "
             "conformal\nprediction supplies by construction — at the cost of one width for every link. Interval half-widths in veh/h are annotated.\n"
             "AUROC here is computed from the tracked NPZ archive (0.7561 / 0.7378). The headline values 0.7585 / 0.7401 come from "
             "trial8_uq_ablation_results.csv,\na separate stochastic MC replay of the same model — see CORRIGENDUM C4 and C7. The conformal panel "
             "uses the graph20_80_v1 protocol.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "04_selective_and_conformal", "the decisions the uncertainty supports")


def fig_trials():
    """Where Trial 8 sits among the trials that are directly comparable."""
    rows = []
    for p in sorted((REPO / "results" / "trials").glob("*/test_evaluation_complete.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        tm, hp = d.get("test_metrics", {}), d.get("hyperparameters", {})
        if tm.get("r2") is None:
            continue
        rows.append((p.parent.name, tm["r2"], tm.get("mae"), hp.get("dropout"),
                     hp.get("use_weighted_loss")))
    if not rows:
        print("  skipped trial comparison (no test_evaluation_complete.json found)")
        return
    rows.sort(key=lambda r: r[1])
    labels = [r[0].replace("point_net_transf_gat_", "").replace("_", " ") for r in rows]
    r2 = [r[1] for r in rows]
    mae = [r[2] for r in rows]
    best = int(np.argmax(r2))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    for ax, vals, lab, better in [
        (axes[0], r2, "test R²", "higher is better"),
        (axes[1], mae, "test MAE (veh/h)", "lower is better"),
    ]:
        cols = [COLORS["blue"]] * len(vals)
        cols[best] = COLORS["coral"]
        ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8.2)
        ax.set_xlabel(f"{lab}  ({better})")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.4f}" if lab.endswith("R²") else f" {v:.3f}",
                    va="center", fontsize=7.8, color=COLORS["slate"])
    axes[0].set_title("Accuracy across retained trials", fontweight="600",
                      color=COLORS["dgray"])
    axes[1].set_title("Error across retained trials", fontweight="600",
                      color=COLORS["dgray"])

    fig.suptitle("Trial comparison — Trial 8 is the UQ baseline",
                 fontsize=13, fontweight="600", color=COLORS["dgray"])
    fig.text(0.5, -0.07,
             "Only trials with a retained test_evaluation_complete.json are shown. Trial 1 is excluded from UQ work entirely: it used a Linear output "
             "head and zero\ndropout, so MC Dropout is undefined for it. Trial 3 used a weighted loss and is not comparable on these metrics.",
             ha="center", fontsize=8.2, color=COLORS["mgray"])
    fig.tight_layout()
    save(fig, "05_trial_comparison", "where Trial 8 sits among the trials")


def main():
    print("Generating results figures from tracked artifacts\n")
    t, pmc, sig, pdet = load(T8)
    print(f"  Trial 8: {t.size:,} node predictions\n")
    fig_accuracy(t, pdet)
    fig_sigma_vs_error(t, pmc, sig)
    fig_calibration(t, pmc, sig)
    fig_selective_and_detection(t, pmc, sig, pdet)
    fig_trials()
    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
