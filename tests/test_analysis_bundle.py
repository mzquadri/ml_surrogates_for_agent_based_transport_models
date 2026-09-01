import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.evidence_contract import (
    LOCAL_TEST_LOADER_BYTES,
    LOCAL_TEST_LOADER_PATH,
    LOCAL_TEST_LOADER_SHA256,
    SOURCE_ARTIFACTS,
    SUBMITTED_DOCUMENT_FILE_COUNT,
    SUBMITTED_DOCUMENT_GIT_TREE,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "analysis_outputs" / "thesis_intelligence.json"
SUBMITTED_PDF = ROOT / "document" / "main.pdf"


def document_tree_contract() -> tuple[int, str]:
    root = ROOT / "document"
    files = sorted(path for path in root.rglob("*") if path.is_file())
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD:document"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return len(files), tree


def test_generated_bundle_is_aggregate_and_path_safe() -> None:
    with BUNDLE_PATH.open(encoding="utf-8") as handle:
        bundle = json.load(handle)

    assert bundle["privacy"] == {
        "classification": "safe aggregate export",
        "contains_absolute_paths": False,
        "contains_pickle_payloads": False,
        "contains_row_level_records": False,
        "source_data_policy": "local processing only; confidential data junction excluded",
    }
    serialized = json.dumps(bundle)
    assert "C:\\Users\\" not in serialized
    assert "MohdZaminQuadri" not in serialized
    assert '"prediction": [' not in serialized
    assert '"target": [' not in serialized


def test_bundle_and_submitted_pdf_provenance_is_locked() -> None:
    with BUNDLE_PATH.open(encoding="utf-8") as handle:
        bundle = json.load(handle)

    assert bundle["source_provenance"] == {
        "audit_source_commit": "fdb4ef0c9c736576ae34d5e331d8b66a7a6d877a",
        "audit_source_repository": (
            "https://github.com/mzquadri/"
            "ml_surrogates_for_agent_based_transport_models"
        ),
        "submitted_artifact_commit": "4b95a3d8aca5929bb88b84bb7f7ae86c48e2f428",
        "submitted_pdf_sha256": (
            "0ac5309d060cda53d82a05cc837136fe853e7f9dcbabd2f4fb4b4282a39bc97e"
        ),
    }
    assert SUBMITTED_PDF.stat().st_size == 674_395
    assert hashlib.sha256(SUBMITTED_PDF.read_bytes()).hexdigest() == (
        bundle["source_provenance"]["submitted_pdf_sha256"]
    )
    assert document_tree_contract() == (
        SUBMITTED_DOCUMENT_FILE_COUNT,
        SUBMITTED_DOCUMENT_GIT_TREE,
    )


def test_full_array_target_zero_fraction_correction_is_locked() -> None:
    with BUNDLE_PATH.open(encoding="utf-8") as handle:
        bundle = json.load(handle)

    target = bundle["analyses"]["t8_mc"]["quality"]["targets"]
    assert target["count"] == 3_163_500
    assert target["zero_count"] == 872_540
    assert target["zero_count"] / target["count"] == pytest.approx(0.2758147621)
    assert any(
        row["topic"] == "target zero-mass claim" for row in bundle["discrepancies"]
    )


def test_manifest_contains_only_relative_paths_and_valid_hashes() -> None:
    with BUNDLE_PATH.open(encoding="utf-8") as handle:
        bundle = json.load(handle)

    for row in bundle["artifact_manifest"]:
        assert not Path(row["path"]).is_absolute()
        if row["sha256"] is not None:
            assert len(row["sha256"]) == 64
            int(row["sha256"], 16)

    source_rows = {
        row["name"]: row
        for row in bundle["artifact_manifest"]
        if row["name"] in SOURCE_ARTIFACTS
    }
    assert set(source_rows) == set(SOURCE_ARTIFACTS)
    for name, (path, size, digest) in SOURCE_ARTIFACTS.items():
        assert source_rows[name] == {
            "bytes": size,
            "exists": True,
            "name": name,
            "path": path,
            "sha256": digest,
            "trust_boundary": "tracked audited-source artifact",
        }

    local_loader = next(
        row for row in bundle["artifact_manifest"] if row["name"] == "t8_local_test_loader"
    )
    assert local_loader == {
        "bytes": LOCAL_TEST_LOADER_BYTES,
        "exists": True,
        "name": "t8_local_test_loader",
        "path": LOCAL_TEST_LOADER_PATH,
        "sha256": LOCAL_TEST_LOADER_SHA256,
        "trust_boundary": "hash-locked local audited-source pickle artifact; never export",
    }
