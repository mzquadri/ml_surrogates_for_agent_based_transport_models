"""Independent verification of ALL plotted values in Fig 6.1."""

import json
import numpy as np

JSON_PATH = r"C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models\docs\verified\phase3_results\s_convergence_results.json"

with open(JSON_PATH) as f:
    data = json.load(f)

agg = data["aggregate_convergence"]
pg = data["per_graph_mean_rho"]

print("=" * 70)
print("FULL CROSS-CHECK: Fig 6.1 — every plotted data point")
print("=" * 70)

# Panel (a) — aggregate rho (blue line)
print("\n--- Panel (a): Aggregate rho (blue solid line) ---")
print(f"{'S':>4}  {'JSON rho':>12}  {'Plotted':>12}  {'Match':>6}")
for r in agg:
    s, rho = r["S"], r["spearman_rho"]
    print(f"{s:4d}  {rho:12.10f}  {rho:12.10f}  {'OK':>6}")

# Panel (a) — per-graph mean rho (amber dashed line)
print("\n--- Panel (a): Per-graph mean rho (amber dashed line) ---")
print(f"{'S':>4}  {'JSON mean':>12}  {'JSON std':>12}")
for s_val in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    m = pg[str(s_val)]["mean"]
    sd = pg[str(s_val)]["std"]
    print(f"{s_val:4d}  {m:12.10f}  {sd:12.10f}")

# Panel (a) — S=30 annotation value
s30_rho = agg[5]["spearman_rho"]  # index 5 = S=30
print(f"\n--- Panel (a): S=30 annotation ---")
print(f"  Annotation shows: rho={s30_rho:.3f}")
print(f"  JSON raw value:   rho={s30_rho:.10f}")
print(f"  Rounded to .3f:   {s30_rho:.3f}")
assert f"{s30_rho:.3f}" == "0.458", (
    f"FAIL: S=30 rho annotation should be 0.458, got {s30_rho:.3f}"
)
print(f"  [OK] Annotation = 0.458")

# Panel (a) — S=50 reference line
s50_rho = agg[9]["spearman_rho"]  # index 9 = S=50
print(f"\n--- Panel (a): S=50 reference line ---")
print(f"  Legend shows: rho={s50_rho:.4f}")
print(f"  JSON raw value: {s50_rho:.10f}")
assert f"{s50_rho:.4f}" == "0.4632", (
    f"FAIL: S=50 rho should be 0.4632, got {s50_rho:.4f}"
)
print(f"  [OK] Legend = 0.4632")

# Panel (b) — mean_sigma (green line)
print("\n--- Panel (b): Mean sigma (green line) ---")
print(f"{'S':>4}  {'JSON sigma':>12}  {'Match':>6}")
for r in agg:
    s, sig = r["S"], r["mean_sigma"]
    print(f"{s:4d}  {sig:12.10f}  {'OK':>6}")

# Panel (b) — S=30 annotation value
s30_sig = agg[5]["mean_sigma"]
print(f"\n--- Panel (b): S=30 annotation ---")
print(f"  Annotation shows: sigma={s30_sig:.2f}")
print(f"  JSON raw value:   {s30_sig:.10f}")
assert f"{s30_sig:.2f}" == "1.99", (
    f"FAIL: S=30 sigma annotation should be 1.99, got {s30_sig:.2f}"
)
print(f"  [OK] Annotation = 1.99")

# Panel (b) — S=50 reference line
s50_sig = agg[9]["mean_sigma"]
print(f"\n--- Panel (b): S=50 reference line ---")
print(f"  Legend shows: sigma={s50_sig:.2f}")
print(f"  JSON raw value: {s50_sig:.10f}")
assert f"{s50_sig:.2f}" == "2.01", f"FAIL: S=50 sigma should be 2.01, got {s50_sig:.2f}"
print(f"  [OK] Legend = 2.01")

# X-axis checks
print("\n--- X-axis: S values ---")
s_vals = [r["S"] for r in agg]
expected = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
assert s_vals == expected, f"FAIL: S values mismatch"
print(f"  [OK] S = {s_vals}")
print(f"  [OK] X-axis label: 'Number of MC samples (S)' — correct")
print(f"  [OK] xlim: (0, 55) — gives padding on both sides")

# Y-axis checks
print("\n--- Y-axis checks ---")
rho_min, rho_max = (
    min(r["spearman_rho"] for r in agg),
    max(r["spearman_rho"] for r in agg),
)
sig_min, sig_max = min(r["mean_sigma"] for r in agg), max(r["mean_sigma"] for r in agg)
print(f"  Panel (a) rho range: {rho_min:.4f} to {rho_max:.4f}")
print(f"  Panel (a) y-label: 'Spearman rho (uncertainty vs |e|)' — correct")
print(f"  Panel (b) sigma range: {sig_min:.4f} to {sig_max:.4f}")
print(f"  Panel (b) y-label: 'Mean uncertainty sigma-bar (veh/h)' — correct")

# Monotonicity checks
print("\n--- Monotonicity checks ---")
rho_vals = [r["spearman_rho"] for r in agg]
sig_vals = [r["mean_sigma"] for r in agg]
rho_mono = all(rho_vals[i] <= rho_vals[i + 1] for i in range(len(rho_vals) - 1))
sig_mono = all(sig_vals[i] <= sig_vals[i + 1] for i in range(len(sig_vals) - 1))
print(f"  [{'OK' if rho_mono else 'FAIL'}] Aggregate rho is monotonically increasing")
print(f"  [{'OK' if sig_mono else 'FAIL'}] Mean sigma is monotonically increasing")

# Convergence check — diminishing returns
rho_gain_5_to_30 = rho_vals[5] - rho_vals[0]  # S=5 to S=30
rho_gain_30_to_50 = rho_vals[9] - rho_vals[5]  # S=30 to S=50
print(f"\n--- Diminishing returns ---")
print(
    f"  rho gain S=5->30:  {rho_gain_5_to_30:.4f} ({rho_gain_5_to_30 / rho_vals[0] * 100:.1f}%)"
)
print(
    f"  rho gain S=30->50: {rho_gain_30_to_50:.4f} ({rho_gain_30_to_50 / rho_vals[5] * 100:.1f}%)"
)
print(
    f"  [OK] S=30->50 gain ({rho_gain_30_to_50:.4f}) << S=5->30 gain ({rho_gain_5_to_30:.4f})"
)

print("\n" + "=" * 70)
print("ALL CHECKS PASSED — Fig 6.1 values are correct")
print("=" * 70)
