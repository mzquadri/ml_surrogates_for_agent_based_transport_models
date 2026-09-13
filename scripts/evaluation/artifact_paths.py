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


def _flattened_names(root: Path, rel: Path) -> list[str]:
    """The names a release asset holding ``rel`` can carry on disk.

    Assets are published under their path from the root of the repository they
    came from, with the separators replaced by ``__``::

        results/predictions/uq_verification_run/ensemble_verified.npz
        -> results__predictions__uq_verification_run__ensemble_verified.npz

    How much of that prefix belongs to the search root differs by release,
    because the two releases were cut from different repositories. The companion
    data repository had ``TR-C_Benchmarks`` at its top level and publishes
    ``TR-C_Benchmarks__<trial>__<file>``; this repository publishes
    ``results__predictions__<trial>__<file>``. Every suffix of the root's own
    path is therefore a candidate prefix, which covers both without hard-coding
    either.
    """
    try:
        parts = root.relative_to(REPO).parts
    except ValueError:
        parts = (root.name,)
    tail = "__".join(rel.parts)
    return ["__".join(parts[index:]) + "__" + tail for index in range(len(parts))]


def resolve(relative: str, *, hint: str | None = None) -> Path:
    """Return the first existing copy of ``relative`` across ``SEARCH_ROOTS``.

    ``relative`` is the ``<trial>/<...>`` tail shared by both layouts, for
    example ``"point_net_transf_gat_7th_trial_80_10_10_split/uq_results/x.npz"``.

    A file fetched from a release is also found under its published name. The
    command in ``RELEASE_HINT`` used to leave the artifact on disk and still
    report it missing: ``gh release download`` writes each asset under the name
    it was published with and cannot rename it, so the 209 MB ablation table
    landed as ``TR-C_Benchmarks__<trial>__trial8_uq_ablation_results.csv`` while
    this function looked only for ``<trial>/trial8_uq_ablation_results.csv``.
    Two headline numbers reported SKIP for that reason alone.

    Raises ``ArtifactNotFound`` naming every location tried, plus ``hint`` if the
    file is only available as a release asset.
    """
    rel = Path(relative)
    for root in SEARCH_ROOTS:
        candidate = root / rel
        if candidate.exists():
            return candidate

    for root in SEARCH_ROOTS:
        for name in _flattened_names(root, rel):
            # Either beside where the artifact would have been, which is what
            # the documented --dir produces, or at the root of the tree.
            for directory in (root / rel.parent, root):
                candidate = directory / name
                if candidate.exists():
                    return candidate

    tried = "\n".join(f"  - {root / rel}" for root in SEARCH_ROOTS)
    published = sorted({name for root in SEARCH_ROOTS
                        for name in _flattened_names(root, rel)})
    message = (
        f"Could not find the artifact '{relative}'.\n\nLooked in:\n{tried}\n\n"
        "Also looked for these published release-asset names, in each root\n"
        "and beside the path above:\n"
        + "\n".join(f"  - {name}" for name in published)
    )
    if hint:
        message += f"\n\n{hint}"
    raise ArtifactNotFound(message)


#: Files too large for git that live on a release rather than in the tree.
#:
#: This text is printed at the moment a lookup fails, so it has to describe a
#: route that actually ends with the file where it is looked for. It used to
#: name a bare `gh release download --dir <trial>/`, which leaves the asset
#: under its published name and so fails again in exactly the same way. The
#: supported route is restore_large_files.py, which decodes the `__` separators
#: back into directories, and is what the README documents.
RELEASE_HINT = (
    "This file exceeds GitHub's 100 MB limit, so it is published as a release\n"
    "asset instead of being tracked. Fetch it with:\n\n"
    "  gh release download thesis-data-v1 \\\n"
    "    --repo mzquadri/ml_surrogates_for_agent_based_transport_models \\\n"
    "    --pattern '*trial8_uq_ablation_results.csv' --dir /tmp/large\n"
    "  python scripts/restore_large_files.py /tmp/large\n\n"
    "That decodes the published name, which encodes the destination path with\n"
    "'__' in place of '/', into the directories this function searches.\n\n"
    "Or point THESIS_DATA_ROOT at a data tree you already have."
)
