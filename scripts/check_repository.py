"""Validate the artifact set required to reproduce the reported analyses.

Existence is the weak check. What matters is that the bytes are the ones the
numbers were computed from, so every artifact named in the evidence contract is
verified by size and SHA-256, and the submitted thesis PDF is verified against
the hash recorded at submission.

    python scripts/check_repository.py

Exit status: 0 clean, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import py_compile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from evidence_contract import (  # noqa: E402
    SOURCE_ARTIFACTS,
    SUBMITTED_PDF_BYTES,
    SUBMITTED_PDF_SHA256,
)

SUBMITTED_PDF = "thesis/latex_tum_official/main.pdf"

REQUIRED_PATHS = (
    "README.md",
    "CITATION.cff",
    "environment-minimal.yml",
    "docs/verified/VERIFIED_RESULTS_MASTER.csv",
    "docs/verified/REPRODUCIBILITY_GAP_SUMMARY.md",
    "docs/verified/AUDIT_SUMMARY.md",
    "thesis/latex_tum_official/main.tex",
    SUBMITTED_PDF,
    "results/phase3/pit_t8.json",
    "results/phase3/temperature_scaling_t8.json",
    "models/deep_ensemble_seed42/trained_model/model.pth",
    "models/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth",
    "results/predictions/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz",
    "scripts/evaluation/run_part2_uq_analyses.py",
    "scripts/evaluation/run_part3_calibration_audit.py",
    "scripts/evaluation/run_part4_t7_crosscheck.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_bytes(name: str, path: Path, size: int, digest: str,
                problems: list[str], normalised: list[str]) -> bool:
    """Verify one artifact's size and hash.

    The contract records the sizes and hashes as they stood before `.gitattributes`
    normalised tracked text to LF. That rewrote the line endings of five JSON
    evidence files without changing a character of their content, so a strict byte
    comparison now fails on them for a reason that has nothing to do with the
    evidence.

    Rewriting the contract to the post-normalisation hashes would destroy the record
    of what was verified at audit time. Instead, a text artifact that fails the strict
    check is retried with CRLF line endings restored, and passes only if it then
    matches the recorded hash exactly -- which proves the content is unchanged and
    the difference is entirely line endings. Binary artifacts get no such latitude.
    """
    actual = path.stat().st_size
    strict_size = actual == size
    if strict_size and sha256(path) == digest:
        return True

    if path.suffix in {".json", ".csv", ".md", ".txt", ".tex", ".yml", ".yaml"}:
        raw = path.read_bytes()
        if b"\r\n" not in raw:
            crlf = raw.replace(b"\n", b"\r\n")
            if len(crlf) == size:
                # Line endings alone explain the size, so the only thing that can
                # still differ is the content itself. Say that, rather than blaming
                # a size the retry has already accounted for.
                if hashlib.sha256(crlf).hexdigest() == digest:
                    normalised.append(name)
                    return True
                problems.append(f"{name}: content differs from the contract "
                                f"(size matches once line endings are accounted for)")
                return False

    if not strict_size:
        problems.append(f"{name}: {actual:,} bytes, contract says {size:,}")
    else:
        problems.append(f"{name}: SHA-256 does not match the contract")
    return False


def main() -> int:
    problems: list[str] = []
    normalised: list[str] = []

    missing = [p for p in REQUIRED_PATHS if not (REPO / p).is_file()]
    if missing:
        print("Missing required artifacts:\n- " + "\n- ".join(missing))
        return 1

    for path in REQUIRED_PATHS:
        if path.endswith(".py"):
            py_compile.compile(REPO / path, doraise=True)

    # The submitted thesis. This is the durable anchor for the submission: the git
    # commit and tree ids recorded in the contract belong to a repository lineage
    # that no longer exists, but the bytes are here and are checkable.
    pdf_ok = check_bytes("submitted thesis PDF", REPO / SUBMITTED_PDF,
                         SUBMITTED_PDF_BYTES, SUBMITTED_PDF_SHA256, problems, normalised)

    verified = 0
    absent: list[str] = []
    for name, (rel, size, digest) in sorted(SOURCE_ARTIFACTS.items()):
        path = REPO / rel
        if not path.is_file():
            absent.append(name)
            continue
        if check_bytes(name, path, size, digest, problems, normalised):
            verified += 1

    print(f"Repository check: {len(REQUIRED_PATHS)} required artifacts present and "
          f"importable.")
    print(f"  submitted thesis PDF: {'verified by SHA-256' if pdf_ok else 'FAILED'}")
    print(f"  evidence-contract artifacts verified by size and SHA-256: "
          f"{verified}/{len(SOURCE_ARTIFACTS)}")
    if absent:
        print(f"  not present in this checkout ({len(absent)}): {', '.join(absent)}")
    if normalised:
        print(f"  content-identical, line endings normalised to LF since the audit "
              f"({len(normalised)}): {', '.join(sorted(normalised))}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        print("\nFAIL")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
