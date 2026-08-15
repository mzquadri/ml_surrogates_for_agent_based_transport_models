import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "analysis_outputs" / "thesis_intelligence.json"


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
