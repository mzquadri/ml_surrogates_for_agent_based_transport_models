"""Locate evaluation artifacts regardless of which copy the user has on disk.

The analysis scripts were written against the layout the training runs wrote,
``data/TR-C_Benchmarks/<trial>/<...>``, which is the layout the companion data
repository still uses. The artifacts that are small enough to track are also
mirrored inside this repository under ``results/predictions/<trial>/<...>``.

Both layouts share the same ``<trial>/<...>`` tail, so a caller only has to name
that tail once. ``resolve`` checks every root that could hold it and returns the
first hit, which lets the documented commands run from a plain clone without the
7.8 GB data tree, and keeps working unchanged when that tree is present.

Search order:

1. ``$THESIS_DATA_ROOT`` if set, for a data tree kept outside the repository.
2. ``data/TR-C_Benchmarks``  -- the companion data repository, cloned into ``data/``.
3. ``results/predictions``   -- the mirror tracked in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

#: Roots that may hold a ``<trial>/<...>`` artifact path, in search order.
SEARCH_ROOTS: list[Path] = []
if os.environ.get("THESIS_DATA_ROOT"):
    SEARCH_ROOTS.append(Path(os.environ["THESIS_DATA_ROOT"]).expanduser().resolve())
SEARCH_ROOTS += [
    REPO / "data" / "TR-C_Benchmarks",
    REPO / "results" / "predictions",
]


class ArtifactNotFound(FileNotFoundError):
    """Raised when an artifact is in none of the known roots."""


def resolve(relative: str, *, hint: str | None = None) -> Path:
    """Return the first existing copy of ``relative`` across ``SEARCH_ROOTS``.

    ``relative`` is the ``<trial>/<...>`` tail shared by both layouts, for
    example ``"point_net_transf_gat_7th_trial_80_10_10_split/uq_results/x.npz"``.

    Raises ``ArtifactNotFound`` naming every location tried, plus ``hint`` if the
    file is only available as a release asset.
    """
    rel = Path(relative)
    for root in SEARCH_ROOTS:
        candidate = root / rel
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {root / rel}" for root in SEARCH_ROOTS)
    message = f"Could not find the artifact '{relative}'.\n\nLooked in:\n{tried}"
    if hint:
        message += f"\n\n{hint}"
    raise ArtifactNotFound(message)


#: Files too large for git that live on a release rather than in the tree.
RELEASE_HINT = (
    "This file exceeds GitHub's 100 MB limit, so it is published as a release\n"
    "asset instead of being tracked. Fetch it with:\n\n"
    "  gh release download large-files-v1 \\\n"
    "    --repo mzquadri/ml-surrogates-thesis-data \\\n"
    "    --pattern '*trial8_uq_ablation_results.csv' --dir data/TR-C_Benchmarks/\\\n"
    "point_net_transf_gat_8th_trial_lower_dropout/\n\n"
    "Or point THESIS_DATA_ROOT at a data tree you already have."
)
