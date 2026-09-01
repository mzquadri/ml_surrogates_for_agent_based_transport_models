from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED_ARRAYS = ("predictions", "uncertainties", "targets")
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def load_numeric_npz(
    path: str | Path, required: Iterable[str]
) -> dict[str, np.ndarray]:
    """Load required numeric arrays from a trusted NPZ without pickle support."""
    required_names = tuple(required)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required_names if name not in archive]
        if missing:
            raise ValueError(f"Missing NPZ arrays: {', '.join(missing)}")
        arrays = {name: np.asarray(archive[name]).reshape(-1) for name in required_names}

    object_arrays = [name for name, values in arrays.items() if values.dtype.hasobject]
    if object_arrays:
        raise ValueError(f"Object arrays are not allowed: {', '.join(object_arrays)}")
    return arrays


def load_prediction_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a trusted NPZ prediction artifact."""
    arrays = load_numeric_npz(path, REQUIRED_ARRAYS)
    predictions = arrays["predictions"]
    uncertainties = arrays["uncertainties"]
    targets = arrays["targets"]
    validate_aligned_arrays(
        {"predictions": predictions, "uncertainties": uncertainties, "targets": targets}
    )

    finite = (
        np.isfinite(predictions) & np.isfinite(uncertainties) & np.isfinite(targets)
    )
    if not finite.all():
        predictions = predictions[finite]
        uncertainties = uncertainties[finite]
        targets = targets[finite]
    if predictions.size == 0:
        raise ValueError("Prediction artifact has no finite rows")
    return predictions, uncertainties, targets


def validate_aligned_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    """Validate non-empty, equally sized, numeric one-dimensional arrays."""
    if not arrays:
        raise ValueError("At least one array is required")
    sizes = {name: np.asarray(values).size for name, values in arrays.items()}
    if len(set(sizes.values())) != 1:
        details = ", ".join(f"{name}={size}" for name, size in sizes.items())
        raise ValueError(f"Arrays must have equal length ({details})")
    size = next(iter(sizes.values()))
    if size == 0:
        raise ValueError("Arrays must not be empty")
    for name, values in arrays.items():
        if not np.issubdtype(np.asarray(values).dtype, np.number):
            raise ValueError(f"{name} must be numeric")
    return size


def array_statistics(
    values: np.ndarray,
    *,
    plausible_min: float | None = None,
    plausible_max: float | None = None,
) -> dict[str, float | int | str | list[int] | None]:
    """Return exact descriptive statistics for one numeric variable."""
    original = np.asarray(values)
    flat = original.reshape(-1)
    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask].astype(np.float64, copy=False)
    if finite.size == 0:
        raise ValueError("Cannot summarize an array without finite values")

    quantiles = np.quantile(finite, QUANTILES)
    q1, median, q3 = quantiles[2], quantiles[3], quantiles[4]
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    skewness = (
        float(np.mean(((finite - mean) / std) ** 3)) if std > 0 else 0.0
    )
    plausible_failures = np.zeros(finite.size, dtype=bool)
    if plausible_min is not None:
        plausible_failures |= finite < plausible_min
    if plausible_max is not None:
        plausible_failures |= finite > plausible_max

    return {
        "dtype": str(original.dtype),
        "shape": list(original.shape),
        "count": int(flat.size),
        "finite_count": int(finite.size),
        "non_finite_count": int((~finite_mask).sum()),
        "missing_fraction": float((~finite_mask).mean()),
        "zero_count": int(np.count_nonzero(finite == 0)),
        "unique_count": int(np.unique(finite).size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": mean,
        "median": float(median),
        "std": std,
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(q1),
        "q75": float(q3),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "iqr": float(iqr),
        "skewness": skewness,
        "outlier_count_iqr": int(
            np.count_nonzero((finite < lower_fence) | (finite > upper_fence))
        ),
        "plausible_min": plausible_min,
        "plausible_max": plausible_max,
        "plausible_range_failure_count": int(plausible_failures.sum()),
    }


def regression_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    validate_aligned_arrays({"predictions": predictions, "targets": targets})
    predictions64 = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not (np.isfinite(predictions64).all() and np.isfinite(targets64).all()):
        raise ValueError("Regression arrays must contain only finite values")
    residuals = predictions64 - targets64
    squared_error = residuals**2
    target_variation = float(np.sum((targets64 - np.mean(targets64)) ** 2))
    r2 = 1.0 - float(np.sum(squared_error)) / target_variation if target_variation else np.nan
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "r2": float(r2),
    }


def risk_curve(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    retentions: Iterable[int],
) -> list[dict[str, float | int]]:
    """Calculate selective-prediction risk after sorting by confidence."""
    total = validate_aligned_arrays(
        {"predictions": predictions, "uncertainties": uncertainties, "targets": targets}
    )
    predictions64 = np.asarray(predictions, dtype=np.float64).reshape(-1)
    uncertainties64 = np.asarray(uncertainties, dtype=np.float64).reshape(-1)
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not (
        np.isfinite(predictions64).all()
        and np.isfinite(uncertainties64).all()
        and np.isfinite(targets64).all()
    ):
        raise ValueError("Risk-curve arrays must contain only finite values")
    if np.any(uncertainties64 < 0):
        raise ValueError("Uncertainties must be non-negative")

    errors = np.abs(predictions64 - targets64)
    ordered = np.argsort(uncertainties64, kind="stable")
    ordered_errors = errors[ordered]
    ordered_squared_errors = ordered_errors**2
    cumulative_error = np.cumsum(ordered_errors)
    cumulative_squared_error = np.cumsum(ordered_squared_errors)
    baseline_mae = cumulative_error[-1] / total
    rows: list[dict[str, float | int]] = []

    for retention in retentions:
        if not 0 < retention <= 100:
            raise ValueError("Retention must be between 1 and 100")
        accepted = max(1, int(np.floor(total * retention / 100)))
        mae = cumulative_error[accepted - 1] / accepted
        rmse = np.sqrt(cumulative_squared_error[accepted - 1] / accepted)
        reduction = 0.0 if baseline_mae == 0 else 100 * (1 - mae / baseline_mae)
        rows.append(
            {
                "retention": retention,
                "accepted": accepted,
                "reviewed": total - accepted,
                "mae": float(mae),
                "rmse": float(rmse),
                "reduction_pct": float(reduction),
                "uncertainty_threshold": float(uncertainties64[ordered[accepted - 1]]),
            }
        )
    return rows


def uncertainty_metrics(
    predictions: np.ndarray, uncertainties: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    validate_aligned_arrays(
        {"predictions": predictions, "uncertainties": uncertainties, "targets": targets}
    )
    errors = np.abs(np.asarray(predictions, dtype=np.float64) - targets)
    sigma = np.asarray(uncertainties, dtype=np.float64)
    if np.any(sigma < 0) or not (np.isfinite(errors).all() and np.isfinite(sigma).all()):
        raise ValueError("Uncertainty metrics require finite, non-negative uncertainty")
    rho = float(cast(Any, spearmanr(sigma, errors)).statistic)
    positive_sigma = np.maximum(sigma, np.finfo(np.float64).eps)
    ratios = errors / positive_sigma
    return {
        "spearman_rho": float(rho),
        "mean_uncertainty": float(np.mean(sigma)),
        "std_uncertainty": float(np.std(sigma)),
        "k90": float(np.quantile(ratios, 0.90)),
        "k95": float(np.quantile(ratios, 0.95)),
        "raw_gaussian_coverage_90": float(np.mean(errors <= 1.6448536269514722 * sigma)),
        "raw_gaussian_coverage_95": float(np.mean(errors <= 1.959963984540054 * sigma)),
    }


def gaussian_reliability(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    nominal_levels: Iterable[float],
) -> list[dict[str, float]]:
    errors = np.abs(np.asarray(predictions, dtype=np.float64) - targets)
    sigma = np.asarray(uncertainties, dtype=np.float64)
    rows = []
    for nominal in nominal_levels:
        if not 0 < nominal < 1:
            raise ValueError("Nominal coverage must be between zero and one")
        z_score = NormalDist().inv_cdf((1 + nominal) / 2)
        rows.append(
            {
                "nominal": float(nominal),
                "empirical": float(np.mean(errors <= z_score * sigma)),
                "z_score": float(z_score),
            }
        )
    return rows


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample corrected split-conformal quantile."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be non-empty and finite")
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    level = min(np.ceil((values.size + 1) * coverage) / values.size, 1.0)
    return float(np.quantile(values, level, method="higher"))


def error_detection_metrics(
    uncertainty: np.ndarray, absolute_error: np.ndarray, percentiles: Iterable[int]
) -> list[dict[str, float | int]]:
    validate_aligned_arrays({"uncertainty": uncertainty, "absolute_error": absolute_error})
    sigma = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    errors = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    rows = []
    for percentile in percentiles:
        if not 0 < percentile < 100:
            raise ValueError("Error percentile must be between zero and 100")
        cutoff = float(np.percentile(errors, percentile))
        labels = errors >= cutoff
        rows.append(
            {
                "percentile": percentile,
                "cutoff": cutoff,
                "positive_count": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "auroc": float(roc_auc_score(labels, sigma)),
                "auprc": float(average_precision_score(labels, sigma)),
            }
        )
    return rows


def binned_relationship(
    x: np.ndarray, y: np.ndarray, *, bins: int = 20
) -> list[dict[str, float | int]]:
    """Aggregate y by quantile bins of x without retaining row-level records."""
    validate_aligned_arrays({"x": x, "y": y})
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    edges = np.unique(np.quantile(x_values, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return [
            {
                "x_min": float(x_values[0]),
                "x_max": float(x_values[0]),
                "x_mean": float(x_values[0]),
                "y_mean": float(np.mean(y_values)),
                "y_median": float(np.median(y_values)),
                "count": int(y_values.size),
            }
        ]
    groups = np.clip(np.digitize(x_values, edges[1:-1], right=True), 0, edges.size - 2)
    rows = []
    for group in range(edges.size - 1):
        selected = groups == group
        if not selected.any():
            continue
        rows.append(
            {
                "x_min": float(np.min(x_values[selected])),
                "x_max": float(np.max(x_values[selected])),
                "x_mean": float(np.mean(x_values[selected])),
                "y_mean": float(np.mean(y_values[selected])),
                "y_median": float(np.median(y_values[selected])),
                "count": int(selected.sum()),
            }
        )
    return rows


def sample_rows(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    sample_size: int = 40_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    validate_aligned_arrays(
        {"predictions": predictions, "uncertainties": uncertainties, "targets": targets}
    )
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    rng = np.random.default_rng(seed)
    size = min(sample_size, predictions.size)
    indices = rng.choice(predictions.size, size=size, replace=False)
    return {
        "prediction": predictions[indices],
        "target": targets[indices],
        "uncertainty": uncertainties[indices],
        "absolute_error": np.abs(predictions[indices] - targets[indices]),
    }
