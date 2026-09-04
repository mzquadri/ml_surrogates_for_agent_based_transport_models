#!/usr/bin/env python3
"""
Put the oversized artifacts back where the code expects them.

Nineteen files exceed GitHub's 100 MB per-file limit and so cannot be tracked in git. They are attached to this repository's `thesis-data-v1` release instead, under names that
encode their destination path with `__` in place of `/`:

    TR-C_Benchmarks__point_net_transf_gat_8th_trial_lower_dropout__data_created_during_training__test_dl.pt
      -> TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/data_created_during_training/test_dl.pt

That encoding exists because release assets are a flat list. Fifteen of these files are
named `test_dl.pt`, one per trial, and would collide with each other if uploaded under
their real names.

Usage, from the repository root:

    gh release download thesis-data-v1 \n      --repo mzquadri/ml_surrogates_for_agent_based_transport_models \n      --dir /tmp/large
    python scripts/restore_large_files.py /tmp/large

Add --dry-run to see what it would do without writing anything.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def destination_for(asset_name: str) -> Path:
    """The real path an encoded asset name belongs at."""
    return REPO_ROOT / Path(*asset_name.split("__"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "download_dir", help="Directory the release assets were downloaded into"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the moves without making them"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move, keeping the downloads in place",
    )
    args = parser.parse_args()

    source = Path(args.download_dir)
    if not source.is_dir():
        print(f"Not a directory: {source}")
        return 1

    assets = sorted(p for p in source.iterdir() if p.is_file() and "__" in p.name)
    if not assets:
        print(f"No encoded assets found in {source}.")
        print("Expected filenames containing '__', e.g. TR-C_Benchmarks__<trial>__...")
        return 1

    placed = skipped = 0
    for asset in assets:
        target = destination_for(asset.name)
        if target.exists() and target.stat().st_size == asset.stat().st_size:
            print(f"  already in place  {target.relative_to(REPO_ROOT)}")
            skipped += 1
            continue
        print(f"  {'would place' if args.dry_run else 'placing'}      "
              f"{target.relative_to(REPO_ROOT)}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copy2(asset, target)
            else:
                shutil.move(str(asset), str(target))
        placed += 1

    print()
    print(f"{placed} file(s) {'to place' if args.dry_run else 'placed'}, {skipped} already present.")
    if not args.dry_run and placed:
        print("Verify the whole tree with: sha256sum -c MANIFEST.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
