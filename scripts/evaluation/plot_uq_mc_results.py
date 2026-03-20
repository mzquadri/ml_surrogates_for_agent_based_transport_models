import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to mc_dropout_full_*.npz")
    ap.add_argument("--outdir", default=None, help="Output folder for plots")
    ap.add_argument("--scatter_n", type=int, default=200_000, help="Subsample points for scatter")
    ap.add_argument("--bins", type=int, default=10, help="Quantile bins for uncertainty->error curve")
    args = ap.parse_args()

    data = np.load(args.npz)
    yhat = data["predictions"].astype(np.float64)
    sigma = data["uncertainties"].astype(np.float64)
    y = data["targets"].astype(np.float64)

    err = np.abs(y - yhat)

    outdir = args.outdir or os.path.join(os.path.dirname(args.npz), "uq_plots")
    os.makedirs(outdir, exist_ok=True)

    # 1) Uncertainty histogram (log scale helpful due to heavy tail)
    plt.figure()
    plt.hist(sigma, bins=200, log=True)
    plt.xlabel("MC Dropout uncertainty (sigma)")
    plt.ylabel("Count (log scale)")
    plt.title("Uncertainty distribution (MC Dropout)")
    plt.tight_layout()
    p1 = os.path.join(outdir, "uncertainty_hist.png")
    plt.savefig(p1, dpi=200)
    plt.close()

    # 2) Uncertainty vs error scatter (subsample)
    n = len(sigma)
    k = min(args.scatter_n, n)
    idx = np.random.default_rng(0).choice(n, size=k, replace=False)

    plt.figure()
    plt.scatter(sigma[idx], err[idx], s=2, alpha=0.25)
    plt.xlabel("Uncertainty (sigma)")
    plt.ylabel("Absolute error |y - yhat|")
    plt.title(f"Uncertainty vs absolute error (subsample n={k})")
    plt.tight_layout()
    p2 = os.path.join(outdir, "uncertainty_vs_error_scatter.png")
    plt.savefig(p2, dpi=200)
    plt.close()

    # 3) Binned curve: mean error vs uncertainty quantiles
    qs = np.linspace(0, 1, args.bins + 1)
    edges = np.quantile(sigma, qs)
    # avoid duplicate edges
    edges = np.unique(edges)
    bin_ids = np.digitize(sigma, edges[1:-1], right=True)

    bin_centers = []
    mean_err = []
    mean_sigma = []
    counts = []

    for b in range(bin_ids.min(), bin_ids.max() + 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_err.append(err[mask].mean())
        mean_sigma.append(sigma[mask].mean())
        counts.append(mask.sum())
        bin_centers.append(b)

    plt.figure()
    plt.plot(mean_sigma, mean_err, marker="o")
    plt.xlabel("Mean uncertainty in bin")
    plt.ylabel("Mean absolute error in bin")
    plt.title("Mean error increases with uncertainty (quantile bins)")
    plt.tight_layout()
    p3 = os.path.join(outdir, "binned_error_vs_uncertainty.png")
    plt.savefig(p3, dpi=200)
    plt.close()

    # 4) Coverage curve: fraction where |err| <= k*sigma
    ks = np.linspace(0.25, 3.0, 12)
    cover = [(err <= (k_ * sigma + 1e-12)).mean() for k_ in ks]

    plt.figure()
    plt.plot(ks, cover, marker="o")
    plt.xlabel("k")
    plt.ylabel("Coverage: P(|err| <= k*sigma)")
    plt.title("Coverage curve using MC sigma")
    plt.ylim(0, 1)
    plt.tight_layout()
    p4 = os.path.join(outdir, "coverage_curve_k_sigma.png")
    plt.savefig(p4, dpi=200)
    plt.close()

    print("Saved plots to:", outdir)
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)
    print(" -", p4)

if __name__ == "__main__":
    main()
