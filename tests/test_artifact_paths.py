"""Artifact resolution, including the published names release assets carry.

The reason this file exists: `scripts/verify_headline_results.py` reported two
of its thirteen headline checks as SKIP because their 209 MB source was "not
present", while the file was sitting in the tree. `gh release download` writes
an asset under the name it was published with and cannot rename it, so the
command the repository documents produced

    TR-C_Benchmarks__<trial>__trial8_uq_ablation_results.csv

and `resolve` looked only for `<trial>/trial8_uq_ablation_results.csv`. Running
the documented reproduction verbatim left both AUROC numbers unverified.

The naming rule is tested as a pure function against the real asset names, and
the filesystem behaviour is tested with short stand-ins. Windows caps a path at
260 characters unless long paths are enabled, and the real names are long
enough that a temporary directory plus the genuine 93-character asset name
exceeds it, which would make this suite fail for a reason that has nothing to
do with what it is checking.
"""

import tempfile
from pathlib import Path

import pytest

from scripts.evaluation import artifact_paths
from scripts.evaluation.artifact_paths import ArtifactNotFound, _flattened_names, resolve

REAL_TRIAL = "point_net_transf_gat_8th_trial_lower_dropout"


# -- the naming rule, against the names the releases actually publish ---------


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A repository root the roots below are relative to.

    _flattened_names reads the root's path relative to REPO to decide how much
    of the published prefix belongs to it, so REPO has to be the parent of the
    roots under test.
    """
    monkeypatch.setattr(artifact_paths, "REPO", tmp_path)
    return tmp_path


def test_the_data_release_name_is_generated_exactly(fake_repo):
    """thesis-data-v1, cut from the companion data repository."""
    names = _flattened_names(
        fake_repo / "data" / "TR-C_Benchmarks",
        Path(f"{REAL_TRIAL}/trial8_uq_ablation_results.csv"),
    )
    assert f"TR-C_Benchmarks__{REAL_TRIAL}__trial8_uq_ablation_results.csv" in names


def test_the_results_release_name_is_generated_exactly(fake_repo):
    """results-large-v1, cut from this repository, so the prefix is longer."""
    names = _flattened_names(
        fake_repo / "results" / "predictions",
        Path("uq_verification_run/ensemble_verified.npz"),
    )
    assert "results__predictions__uq_verification_run__ensemble_verified.npz" in names


def test_every_offered_prefix_is_a_suffix_of_the_root_path(fake_repo):
    names = _flattened_names(fake_repo / "results" / "predictions", Path("t/f.npz"))
    assert names == ["results__predictions__t__f.npz", "predictions__t__f.npz"]


def test_a_root_outside_the_repository_falls_back_to_its_own_name(fake_repo):
    """THESIS_DATA_ROOT can point anywhere, so relative_to would raise."""
    names = _flattened_names(Path(tempfile.gettempdir()) / "elsewhere", Path("t/f.npz"))
    assert names == ["elsewhere__t__f.npz"]


# ── resolution on disk ────────────────────────────────────────────────────────


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Both search roots, relocated under a temporary repository.

    Short stand-in names, for the path-length reason in the module docstring.
    """
    repo = tmp_path / "r"
    data = repo / "data" / "TR-C_Benchmarks"
    mirror = repo / "results" / "predictions"
    for directory in (data, mirror):
        directory.mkdir(parents=True)
    monkeypatch.setattr(artifact_paths, "REPO", repo)
    monkeypatch.setattr(artifact_paths, "SEARCH_ROOTS", [data, mirror])
    return data, mirror


ARTIFACT = "t8/ablation.csv"
PUBLISHED = "TR-C_Benchmarks__t8__ablation.csv"


def test_the_plain_layout_still_resolves(roots):
    data, _ = roots
    target = data / ARTIFACT
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")
    assert resolve(ARTIFACT) == target


def test_the_first_root_wins_when_both_hold_a_copy(roots):
    data, mirror = roots
    for root in (data, mirror):
        target = root / ARTIFACT
        target.parent.mkdir(parents=True)
        target.write_text("x", encoding="utf-8")
    assert resolve(ARTIFACT) == data / ARTIFACT


def test_an_exact_path_is_preferred_over_a_published_name(roots):
    data, _ = roots
    exact = data / ARTIFACT
    exact.parent.mkdir(parents=True)
    exact.write_text("x", encoding="utf-8")
    (data / PUBLISHED).write_text("x", encoding="utf-8")
    assert resolve(ARTIFACT) == exact


def test_a_release_asset_downloaded_as_documented_resolves(roots):
    """The shape the documented --dir command leaves behind."""
    data, _ = roots
    directory = data / "t8"
    directory.mkdir(parents=True)
    published = directory / PUBLISHED
    published.write_text("x", encoding="utf-8")
    assert resolve(ARTIFACT) == published


def test_a_release_asset_downloaded_to_the_root_resolves(roots):
    data, _ = roots
    published = data / PUBLISHED
    published.write_text("x", encoding="utf-8")
    assert resolve(ARTIFACT) == published


def test_a_missing_artifact_names_both_forms_it_looked_for(roots):
    with pytest.raises(ArtifactNotFound) as raised:
        resolve(ARTIFACT, hint="fetch it from the release")
    message = str(raised.value)
    assert str(Path("TR-C_Benchmarks") / "t8" / "ablation.csv") in message
    assert PUBLISHED in message
    assert "fetch it from the release" in message


def test_a_prefix_that_is_not_part_of_the_root_is_not_accepted(roots):
    data, _ = roots
    (data / "something__t8__ablation.csv").write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactNotFound):
        resolve(ARTIFACT)
