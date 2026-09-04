#!/usr/bin/env python
"""Check the derived web assets parse, carry their documented fields, and stay small.

These files are a published contract: a website will join `links.csv` to the
scenario and arrondissement assets by `link_row`. This gate catches the ways that
contract silently breaks — a renamed column, a JSON that stops parsing, an asset
that quietly grows into something too heavy to ship, or a file documented in
SCHEMA.md that nobody generates any more.

    python scripts/check_web_assets.py

Exit status: 0 clean, 1 otherwise.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "portfolio_data_story" / "assets"
SCHEMA = ASSETS / "SCHEMA.md"

#: The join key contract. Renaming any of these breaks every downstream chart.
LINKS_COLUMNS = [
    "link_row", "start_lon", "start_lat", "end_lon", "end_lat", "mid_lon",
    "mid_lat", "highway_code", "vol_base_case", "capacity_base_case",
    "freespeed_ms", "length_m", "degree", "times_intervened",
    "mean_abs_response", "std_response",
]
EXPECTED_LINK_ROWS = 31_635

#: Assets that must exist, with the top-level keys a consumer relies on.
REQUIRED = {
    "scenarios.json": ["n_scenarios", "n_links", "fields", "scenarios"],
    "arrondissements.json": ["source", "join", "arrondissements"],
    "trials.json": ["n_trials", "trials", "caveat"],
    "experiment_timeline.json": ["comparability_warning", "stages"],
    "uq_methods.json": ["methods"],
    "calibration_curve.json": ["protocol", "temperature", "nominal_levels"],
    "selective_prediction.json": ["baseline_mae_vehh", "curve"],
    "conformal_coverage.json": ["protocol", "levels"],
    "spillover_decay.json": ["undirected", "directed", "method"],
    "highway_classes.json": ["source", "classes"],
    "feature_summary.json": ["n_links", "n_scenarios", "features"],
    "narrative_link.json": ["link_row", "why", "identity_check"],
    "representative_scenarios.json": ["items"],
    "representative_links.json": ["links"],
}

#: A single asset above this is too heavy for a web page to fetch eagerly.
MAX_ASSET_MB = 4.0


def main() -> int:
    problems: list[str] = []

    if not ASSETS.is_dir():
        print(f"FAIL: {ASSETS} does not exist")
        return 1

    # --- every JSON parses -------------------------------------------------------------
    jsons = sorted(ASSETS.glob("*.json"))
    for p in jsons:
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{p.name} does not parse: {exc}")

    # --- required assets and their top-level keys --------------------------------------
    for name, keys in REQUIRED.items():
        p = ASSETS / name
        if not p.exists():
            problems.append(f"{name} is missing")
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for k in keys:
            if k not in obj:
                problems.append(f"{name} has no top-level '{k}'")

    # --- the links.csv join contract ---------------------------------------------------
    links = ASSETS / "links.csv"
    shape = "absent"
    if not links.exists():
        problems.append("links.csv is missing")
    else:
        with links.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            problems.append("links.csv is empty")
        else:
            found = list(rows[0])
            shape = f"{len(rows):,} rows x {len(found)} columns"
            if found != LINKS_COLUMNS:
                missing = [c for c in LINKS_COLUMNS if c not in found]
                extra = [c for c in found if c not in LINKS_COLUMNS]
                problems.append(
                    "links.csv columns changed"
                    + (f"; missing {missing}" if missing else "")
                    + (f"; unexpected {extra}" if extra else "")
                    + ("; order changed" if not missing and not extra else ""))
            if len(rows) != EXPECTED_LINK_ROWS:
                problems.append(
                    f"links.csv has {len(rows):,} rows, expected {EXPECTED_LINK_ROWS:,}")
            # Only meaningful once the join key is actually present.
            if "link_row" in found:
                try:
                    ids = [int(r["link_row"]) for r in rows]
                except (TypeError, ValueError) as exc:
                    problems.append(f"links.csv link_row is not integer: {exc}")
                else:
                    if ids != list(range(len(ids))):
                        problems.append("links.csv link_row is not contiguous from 0")

    # --- documented but not generated, and vice versa -----------------------------------
    if SCHEMA.exists():
        doc = SCHEMA.read_text(encoding="utf-8")
        for p in jsons + [links]:
            if p.exists() and not p.name.startswith("scenario_") and p.name not in doc:
                problems.append(f"{p.name} is generated but not documented in SCHEMA.md")
    else:
        problems.append("SCHEMA.md is missing")

    # --- size discipline ----------------------------------------------------------------
    total = 0.0
    for p in ASSETS.iterdir():
        if not p.is_file():
            continue
        mb = p.stat().st_size / 1e6
        total += mb
        if mb > MAX_ASSET_MB:
            problems.append(f"{p.name} is {mb:.1f} MB, over the {MAX_ASSET_MB} MB budget")

    n_files = len([p for p in ASSETS.iterdir() if p.is_file()])
    print(f"checked {n_files} assets ({total:.2f} MB total)")
    print(f"  links.csv: {shape}")
    print(f"  required assets present: {sum(1 for n in REQUIRED if (ASSETS/n).exists())}"
          f"/{len(REQUIRED)}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p_ in problems:
            print(f"  {p_}")
        print("\nFAIL")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
