"""
CSV Data Verification Script for Nazim's Thesis Project
========================================================
Verifies integrity and structure of:
  1. trial8_uq_ablation_results.csv  (~200 MB, large)
  2. TRIALS_SUMMARY.csv              (small summary)
  3. all_models_summary.csv          (small summary)
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parent / "data" / "TR-C_Benchmarks")

FILES = {
    "trial8_uq_ablation_results.csv": os.path.join(
        BASE,
        "point_net_transf_gat_8th_trial_lower_dropout",
        "trial8_uq_ablation_results.csv",
    ),
    "TRIALS_SUMMARY.csv": os.path.join(
        BASE, "TRIALS_SUMMARY_REPORT", "TRIALS_SUMMARY.csv"
    ),
    "all_models_summary.csv": os.path.join(
        BASE, "ALL_MODELS_COMPARISON", "all_models_summary.csv"
    ),
}

SEP = "=" * 80
SUBSEP = "-" * 60


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def classify_columns(df):
    """Split columns into numeric vs categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, cat_cols


def print_nan_report(df):
    nan_counts = df.isna().sum()
    nan_pct = (df.isna().mean() * 100).round(2)
    report = pd.DataFrame({"NaN_count": nan_counts, "NaN_%": nan_pct})
    report = report[report["NaN_count"] > 0]
    if report.empty:
        print("  No NaN values found in any column.")
    else:
        print(report.to_string())
    print()


def verify_large_csv(name, path):
    """Full verification for the large trial8 file."""
    print(SEP)
    print(f"  FILE: {name}")
    print(f"  PATH: {path}")
    print(f"  SIZE: {file_size_mb(path):.2f} MB")
    print(SEP)

    print("\n[1] Loading file ...")
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    elapsed = time.time() - t0
    print(f"    Loaded in {elapsed:.1f}s\n")

    # ── Shape ────────────────────────────────────────────────────────────────
    print(f"[2] Shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns\n")

    # ── Column names & dtypes ────────────────────────────────────────────────
    print("[3] Columns & dtypes:")
    print(SUBSEP)
    for col in df.columns:
        print(f"    {col:<45s}  {str(df[col].dtype)}")
    print()

    # ── Memory usage ─────────────────────────────────────────────────────────
    mem = df.memory_usage(deep=True)
    total_mb = mem.sum() / (1024**2)
    print(f"[4] Memory usage: {total_mb:.2f} MB (in-memory, deep)")
    print("    Per-column (top 10 by size):")
    mem_sorted = mem.drop("Index").sort_values(ascending=False).head(10)
    for col, val in mem_sorted.items():
        print(f"      {col:<45s}  {val / (1024**2):.2f} MB")
    print()

    # ── NaN report ───────────────────────────────────────────────────────────
    print("[5] NaN / missing values per column:")
    print(SUBSEP)
    print_nan_report(df)

    # ── First & last rows ────────────────────────────────────────────────────
    print("[6] First 5 rows:")
    print(SUBSEP)
    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.max_colwidth", 40
    ):
        print(df.head(5).to_string(index=False))
    print()

    print("[7] Last 5 rows:")
    print(SUBSEP)
    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.max_colwidth", 40
    ):
        print(df.tail(5).to_string(index=False))
    print()

    # ── Categorical unique values ────────────────────────────────────────────
    numeric_cols, cat_cols = classify_columns(df)
    print(f"[8] Numeric columns  ({len(numeric_cols)}): {numeric_cols}")
    print(f"    Categorical cols ({len(cat_cols)}): {cat_cols}\n")

    if cat_cols:
        print("[9] Unique values in categorical columns:")
        print(SUBSEP)
        for col in cat_cols:
            nuniq = df[col].nunique()
            vals = df[col].unique()
            if nuniq <= 30:
                print(f"    {col} ({nuniq} unique): {sorted(vals.tolist())}")
            else:
                sample = sorted(vals[:15].tolist())
                print(f"    {col} ({nuniq} unique): {sample}  ... (showing first 15)")
        print()

    # ── Numeric summary stats ────────────────────────────────────────────────
    print("[10] Numeric summary statistics:")
    print(SUBSEP)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        "{:.6f}".format,
    ):
        print(df[numeric_cols].describe().T.to_string())
    print()

    # ── Row-count sanity check ───────────────────────────────────────────────
    nrows = df.shape[0]
    print("[11] Row-count sanity check:")
    print(SUBSEP)
    expected_node_level = 3_163_500
    print(f"    Actual rows          : {nrows:,}")
    print(f"    Expected (node-level): {expected_node_level:,}")
    if nrows == expected_node_level:
        print("    >> MATCH: row count equals expected node-level count.")
    else:
        ratio = nrows / expected_node_level
        print(f"    >> MISMATCH: ratio = {ratio:.4f}x of expected.")
        # Try to infer grouping
        if cat_cols:
            for col in cat_cols:
                nuniq = df[col].nunique()
                if nrows % nuniq == 0:
                    per_group = nrows // nuniq
                    print(
                        f"       {nrows:,} / {nuniq} unique '{col}' = {per_group:,} rows per group"
                    )
    print()

    return df


def verify_small_csv(name, path):
    """Full print for small summary CSVs."""
    print(SEP)
    print(f"  FILE: {name}")
    print(f"  PATH: {path}")
    print(f"  SIZE: {file_size_mb(path):.4f} MB")
    print(SEP)

    df = pd.read_csv(path)

    print(f"\n[1] Shape: {df.shape[0]} rows  x  {df.shape[1]} columns\n")

    print("[2] Columns & dtypes:")
    print(SUBSEP)
    for col in df.columns:
        print(f"    {col:<55s}  {str(df[col].dtype)}")
    print()

    print("[3] NaN / missing values:")
    print(SUBSEP)
    print_nan_report(df)

    print("[4] Full content:")
    print(SUBSEP)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
        "display.max_colwidth",
        60,
        "display.max_rows",
        None,
        "display.float_format",
        "{:.6f}".format,
    ):
        print(df.to_string(index=True))
    print()

    numeric_cols, cat_cols = classify_columns(df)
    if cat_cols:
        print("[5] Unique values in categorical columns:")
        print(SUBSEP)
        for col in cat_cols:
            vals = df[col].unique()
            print(f"    {col} ({len(vals)} unique): {vals.tolist()}")
        print()

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + SEP)
    print("  CSV DATA VERIFICATION  –  Nazim Thesis Project")
    print(SEP + "\n")

    # Check existence
    for name, path in FILES.items():
        exists = os.path.exists(path)
        sz = f"{file_size_mb(path):.2f} MB" if exists else "FILE NOT FOUND"
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name:<40s}  {sz}")
    print()

    # ── 1. Large file ────────────────────────────────────────────────────────
    print("\n>>> VERIFYING LARGE FILE <<<\n")
    df_trial8 = verify_large_csv(
        "trial8_uq_ablation_results.csv",
        FILES["trial8_uq_ablation_results.csv"],
    )

    # ── 2. TRIALS_SUMMARY.csv ───────────────────────────────────────────────
    print("\n>>> VERIFYING TRIALS_SUMMARY.csv <<<\n")
    df_trials = verify_small_csv("TRIALS_SUMMARY.csv", FILES["TRIALS_SUMMARY.csv"])

    # ── 3. all_models_summary.csv ────────────────────────────────────────────
    print("\n>>> VERIFYING all_models_summary.csv <<<\n")
    df_all = verify_small_csv("all_models_summary.csv", FILES["all_models_summary.csv"])

    print(SEP)
    print("  VERIFICATION COMPLETE")
    print(SEP)
