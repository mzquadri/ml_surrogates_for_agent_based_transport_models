#!/usr/bin/env python
"""Check that documentation links resolve and every SVG parses.

Two failure modes this catches, both of which have actually occurred here:

  * a document points at a file that was renamed, moved, or never committed
  * an SVG is malformed and silently fails to render on GitHub

`docs/archive/` is reported but does not fail the run. Those documents are frozen
snapshots whose links were valid in the repository layout they came from;
rewriting them would misrepresent the historical record.

    python scripts/check_docs.py            # active docs must be clean
    python scripts/check_docs.py --all      # also fail on archived docs

Exit status: 0 clean, 1 broken link or unparseable SVG in a non-archived file.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIX = ("http://", "https://", "#", "mailto:", "../../")

#: Links that are known to be unresolvable, with the reason. Each of these
#: documents carries a banner saying so. Anything NOT listed here still fails,
#: so this records accepted gaps rather than silencing the check.
ACCEPTED = {
    "docs/COMPLETE_VERIFICATION_REPORT.md":
        "embeds a verification/ figure set that was never committed; the report is "
        "kept for its text and numbers and says so at the top",
    "docs/ENSEMBLE_UQ_EXPERIMENTS_REPORT.md":
        "one image lives in the untracked data tree; see DATA.md",
}


def tracked(pattern: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", pattern], cwd=REPO,
                         capture_output=True, text=True).stdout.split("\n")
    return [REPO / p for p in out if p.strip()]


def is_archived(p: Path) -> bool:
    return "/archive/" in p.as_posix()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="also fail on links inside docs/archive/")
    args = ap.parse_args()

    broken_active, broken_archive, accepted = [], [], []
    for md in tracked("*.md"):
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        for raw in LINK.findall(text):
            url = raw.split(" ")[0].strip()
            if not url or url.startswith(SKIP_PREFIX):
                continue
            target = url.split("#")[0]
            if not target:
                continue
            if not (md.parent / target).resolve().exists():
                rel = md.relative_to(REPO).as_posix()
                rec = f"{rel} -> {url}"
                if rel in ACCEPTED:
                    accepted.append(rec)
                elif is_archived(md):
                    broken_archive.append(rec)
                else:
                    broken_active.append(rec)

    bad_svg = []
    for svg in tracked("*.svg"):
        if not svg.exists():
            continue
        try:
            ET.parse(svg)
        except ET.ParseError as exc:
            bad_svg.append(f"{svg.relative_to(REPO).as_posix()}: {exc}")

    n_md = len([p for p in tracked('*.md') if p.exists()])
    n_svg = len([p for p in tracked('*.svg') if p.exists()])
    print(f"checked {n_md} markdown files and {n_svg} SVGs")

    if broken_active:
        print(f"\nbroken links in active documents ({len(broken_active)}):")
        for r in broken_active:
            print(f"  {r}")
    else:
        print("  active documents: all links resolve")

    if accepted:
        print(f"\naccepted known gaps ({len(accepted)}), not failing the run:")
        for doc, why in ACCEPTED.items():
            n = sum(1 for r in accepted if r.startswith(doc))
            if n:
                print(f"  {doc}  ({n} links)\n      {why}")

    if broken_archive:
        print(f"\nbroken links in docs/archive/ ({len(broken_archive)}) "
              f"— frozen snapshots, {'failing' if args.all else 'not failing'} the run")
        if args.all:
            for r in broken_archive:
                print(f"  {r}")

    if bad_svg:
        print(f"\nunparseable SVGs ({len(bad_svg)}):")
        for r in bad_svg:
            print(f"  {r}")
    else:
        print(f"  SVGs: {n_svg}/{n_svg} parse")

    failed = bool(broken_active) or bool(bad_svg) or (args.all and bool(broken_archive))
    print("\nFAIL" if failed else "\nOK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
