import os
import argparse
import numpy as np

def conformal_q(residuals, alpha):
    # Standard conformal quantile: ceil((n+1)*(1-alpha))/n
    n = residuals.shape[0]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(residuals, q_level, method="higher")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc_test_npz", required=True, help="MC test npz: predictions, uncertainties, targets")
    ap.add_argument("--mc_cal_npz", default=None, help="MC calibration npz (recommended). If omitted, will split test.")
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (0.1=90% intervals)")
    ap.add_argument("--eps", type=float, default=1e-6, help="Small epsilon for scaling")
    args = ap.parse_args()

    test = np.load(args.mc_test_npz)
    yhat_t = test["predictions"].astype(np.float64)
    sig_t  = test["uncertainties"].astype(np.float64)
    y_t    = test["targets"].astype(np.float64)

    if args.mc_cal_npz is not None:
        cal = np.load(args.mc_cal_npz)
        yhat_c = cal["predictions"].astype(np.float64)
        sig_c  = cal["uncertainties"].astype(np.float64)
        y_c    = cal["targets"].astype(np.float64)
        cal_note = "calibration=npz_provided"
    else:
        # fallback (not ideal for thesis): split test into cal/test
        n = len(y_t)
        split = n // 5  # 20% calibration
        yhat_c, sig_c, y_c = yhat_t[:split], sig_t[:split], y_t[:split]
        yhat_t, sig_t, y_t = yhat_t[split:], sig_t[split:], y_t[split:]
        cal_note = "calibration=test_split_20pct (NOT ideal; use val if available)"

    alpha = args.alpha

    # Global conformal (uses absolute residuals)
    r = np.abs(y_c - yhat_c)
    q_global = conformal_q(r, alpha)

    lo_g = yhat_t - q_global
    hi_g = yhat_t + q_global
    cov_g = np.mean((y_t >= lo_g) & (y_t <= hi_g))
    wid_g = np.mean(hi_g - lo_g)

    # Adaptive conformal using MC sigma (scaled residuals)
    r_scaled = np.abs(y_c - yhat_c) / (sig_c + args.eps)
    q_adapt = conformal_q(r_scaled, alpha)

    lo_a = yhat_t - q_adapt * (sig_t + args.eps)
    hi_a = yhat_t + q_adapt * (sig_t + args.eps)
    cov_a = np.mean((y_t >= lo_a) & (y_t <= hi_a))
    wid_a = np.mean(hi_a - lo_a)

    print("="*70)
    print("Conformal intervals from MC Dropout outputs")
    print("="*70)
    print("alpha:", alpha, f"(target coverage ~ {1-alpha:.0%})")
    print("note:", cal_note)
    print()
    print("[Global conformal]  yhat +/- q")
    print("  q_global:", float(q_global))
    print("  coverage:", float(cov_g))
    print("  avg width:", float(wid_g))
    print()
    print("[Adaptive conformal]  yhat +/- q*sigma")
    print("  q_adapt:", float(q_adapt))
    print("  coverage:", float(cov_a))
    print("  avg width:", float(wid_a))
    print("="*70)

if __name__ == "__main__":
    main()
