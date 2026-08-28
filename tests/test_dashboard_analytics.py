from pathlib import Path

import numpy as np
import pytest

from thesis_dashboard.analytics import (
    array_statistics,
    conformal_quantile,
    load_prediction_arrays,
    regression_metrics,
    risk_curve,
    sample_rows,
)


def test_regression_metrics_for_perfect_predictions() -> None:
    values = np.array([1.0, 2.0, 3.0])

    metrics = regression_metrics(values, values)

    assert metrics == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_risk_curve_accepts_least_uncertain_rows_first() -> None:
    predictions = np.array([0.0, 5.0, 2.0, 10.0])
    targets = np.zeros(4)
    uncertainties = np.array([0.1, 0.8, 0.2, 0.9])

    rows = risk_curve(predictions, uncertainties, targets, [50, 100])

    assert rows[0]["accepted"] == 2
    assert rows[0]["mae"] == 1.0
    assert rows[1]["mae"] == pytest.approx(4.25)
    assert rows[0]["reduction_pct"] == pytest.approx(76.470588)


def test_risk_curve_rejects_invalid_retention() -> None:
    values = np.array([1.0])

    with pytest.raises(ValueError, match="Retention"):
        risk_curve(values, values, values, [0])


def test_prediction_loader_filters_aligned_non_finite_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.npz"
    np.savez(
        artifact,
        predictions=np.array([1.0, np.nan, 3.0]),
        uncertainties=np.array([0.1, 0.2, 0.3]),
        targets=np.array([1.5, 2.5, np.inf]),
    )

    predictions, uncertainties, targets = load_prediction_arrays(artifact)

    np.testing.assert_array_equal(predictions, np.array([1.0]))
    np.testing.assert_array_equal(uncertainties, np.array([0.1]))
    np.testing.assert_array_equal(targets, np.array([1.5]))


def test_prediction_loader_rejects_missing_and_misaligned_arrays(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    np.savez(missing, predictions=np.array([1.0]), targets=np.array([1.0]))
    with pytest.raises(ValueError, match="Missing NPZ arrays"):
        load_prediction_arrays(missing)

    misaligned = tmp_path / "misaligned.npz"
    np.savez(
        misaligned,
        predictions=np.array([1.0, 2.0]),
        uncertainties=np.array([0.1]),
        targets=np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="equal length"):
        load_prediction_arrays(misaligned)


def test_perfect_risk_baseline_has_zero_reduction() -> None:
    values = np.array([1.0, 2.0, 3.0])
    rows = risk_curve(values, np.array([0.3, 0.1, 0.2]), values, [50, 100])

    assert rows[0]["accepted"] == 1
    assert rows[0]["reduction_pct"] == 0.0
    assert rows[1]["reduction_pct"] == 0.0


def test_array_statistics_tracks_quality_and_plausibility() -> None:
    stats = array_statistics(
        np.array([0.0, 1.0, 1.0, 100.0, np.nan]),
        plausible_min=0.0,
        plausible_max=10.0,
    )

    assert stats["count"] == 5
    assert stats["finite_count"] == 4
    assert stats["non_finite_count"] == 1
    assert stats["unique_count"] == 3
    assert stats["plausible_range_failure_count"] == 1


def test_conformal_quantile_uses_finite_sample_higher_correction() -> None:
    scores = np.arange(1.0, 11.0)

    assert conformal_quantile(scores, 0.8) == 10.0
    assert conformal_quantile(scores, 0.9) == 10.0


def test_sampling_is_deterministic_and_without_replacement() -> None:
    values = np.arange(100, dtype=float)

    first = sample_rows(values, values / 100, -values, sample_size=20, seed=42)
    second = sample_rows(values, values / 100, -values, sample_size=20, seed=42)

    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert np.unique(first["prediction"]).size == 20
