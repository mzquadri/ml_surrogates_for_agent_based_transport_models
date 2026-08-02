"""
Heteroscedastic MC Dropout Inference
=====================================
Performs MC Dropout inference with heteroscedastic model to decompose uncertainty into:
- Aleatoric (data noise): E[sigma_aleatoric^2]
- Epistemic (model uncertainty): Var[mu]
- Total: sqrt(aleatoric^2 + epistemic^2)

This script provides utilities for running heteroscedastic MC Dropout and analyzing
the uncertainty decomposition.

References:
- Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"
  https://arxiv.org/abs/1703.04977

Author: Mohd Zamin Quadri (for thesis UQ extension)
"""

import torch
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


def heteroscedastic_mc_dropout_single_graph(
    model, graph, num_samples: int = 30, device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Perform MC Dropout on a single graph with heteroscedastic model.

    Args:
        model: Trained heteroscedastic model (PointNetTransfGATHeteroscedastic)
        graph: PyG Data object with node features, edges, positions
        num_samples: Number of MC samples (default: 30, same as T8)
        device: Device to run inference on

    Returns:
        dict: Dictionary with keys:
            - 'mean': Final predictive mean (N,)
            - 'sigma_aleatoric': Aleatoric uncertainty (N,)
            - 'sigma_epistemic': Epistemic uncertainty (N,)
            - 'sigma_total': Total uncertainty (N,)
            - 'ground_truth': True values (N,)
            - 'mean_samples': All mean samples (S, N) for diagnostics
            - 'variance_samples': All variance samples (S, N) for diagnostics
    """
    model.train()  # Enable dropout for MC sampling
    graph = graph.to(device)

    means_list = []
    variances_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            mean, log_var = model(graph)

            # Convert to numpy
            means_list.append(mean.squeeze().cpu().numpy())
            variances_list.append(torch.exp(log_var).squeeze().cpu().numpy())

    # Stack samples: shape (S, N) where S = num_samples, N = num_nodes
    means = np.stack(means_list, axis=0)
    variances = np.stack(variances_list, axis=0)

    # Uncertainty decomposition
    # Mean prediction: E[mu]
    mean_pred = np.mean(means, axis=0)

    # Aleatoric uncertainty: sqrt(E[sigma^2])
    sigma_aleatoric = np.sqrt(np.mean(variances, axis=0))

    # Epistemic uncertainty: Std[mu]
    sigma_epistemic = np.std(means, axis=0)

    # Total uncertainty: sqrt(sigma_aleatoric^2 + sigma_epistemic^2)
    sigma_total = np.sqrt(sigma_aleatoric**2 + sigma_epistemic**2)

    # Ground truth
    ground_truth = graph.y.squeeze().cpu().numpy()

    return {
        "mean": mean_pred,
        "sigma_aleatoric": sigma_aleatoric,
        "sigma_epistemic": sigma_epistemic,
        "sigma_total": sigma_total,
        "ground_truth": ground_truth,
        "mean_samples": means,
        "variance_samples": variances,
    }


def heteroscedastic_mc_dropout(
    model,
    data_loader,
    num_samples: int = 30,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Perform MC Dropout on full dataset with heteroscedastic model.

    Args:
        model: Trained heteroscedastic model
        data_loader: PyG DataLoader with test graphs
        num_samples: Number of MC samples (default: 30)
        device: Device to run inference on
        verbose: Show progress bar

    Returns:
        dict: Dictionary with concatenated results across all graphs:
            - 'mean': Final predictive means (N_total,)
            - 'sigma_aleatoric': Aleatoric uncertainties (N_total,)
            - 'sigma_epistemic': Epistemic uncertainties (N_total,)
            - 'sigma_total': Total uncertainties (N_total,)
            - 'ground_truth': True values (N_total,)
    """
    model.train()  # Enable dropout
    model = model.to(device)

    all_means = []
    all_sigma_aleatoric = []
    all_sigma_epistemic = []
    all_sigma_total = []
    all_targets = []

    iterator = (
        tqdm(data_loader, desc="Heteroscedastic MC Dropout") if verbose else data_loader
    )

    for graph in iterator:
        result = heteroscedastic_mc_dropout_single_graph(
            model, graph, num_samples, device
        )

        all_means.append(result["mean"])
        all_sigma_aleatoric.append(result["sigma_aleatoric"])
        all_sigma_epistemic.append(result["sigma_epistemic"])
        all_sigma_total.append(result["sigma_total"])
        all_targets.append(result["ground_truth"])

    return {
        "mean": np.concatenate(all_means),
        "sigma_aleatoric": np.concatenate(all_sigma_aleatoric),
        "sigma_epistemic": np.concatenate(all_sigma_epistemic),
        "sigma_total": np.concatenate(all_sigma_total),
        "ground_truth": np.concatenate(all_targets),
    }


def analyze_uncertainty_decomposition(
    results: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Analyze uncertainty decomposition statistics.

    Args:
        results: Output from heteroscedastic_mc_dropout()

    Returns:
        dict: Statistics including:
            - mean_sigma_aleatoric: Average aleatoric uncertainty
            - mean_sigma_epistemic: Average epistemic uncertainty
            - mean_sigma_total: Average total uncertainty
            - ratio_aleatoric_epistemic: sigma_a / sigma_e ratio
            - frac_aleatoric_dominant: Fraction where sigma_a > sigma_e
            - correlation_error_sigma_total: Correlation between |error| and sigma_total
    """
    from scipy.stats import spearmanr

    sigma_a = results["sigma_aleatoric"]
    sigma_e = results["sigma_epistemic"]
    sigma_t = results["sigma_total"]

    errors = np.abs(results["mean"] - results["ground_truth"])

    # Correlation between total uncertainty and errors (ranking quality)
    rho, pval = spearmanr(sigma_t, errors)

    stats = {
        "mean_sigma_aleatoric": float(np.mean(sigma_a)),
        "std_sigma_aleatoric": float(np.std(sigma_a)),
        "mean_sigma_epistemic": float(np.mean(sigma_e)),
        "std_sigma_epistemic": float(np.std(sigma_e)),
        "mean_sigma_total": float(np.mean(sigma_t)),
        "std_sigma_total": float(np.std(sigma_t)),
        "ratio_aleatoric_epistemic": float(np.mean(sigma_a) / np.mean(sigma_e)),
        "frac_aleatoric_dominant": float(np.mean(sigma_a > sigma_e)),
        "frac_epistemic_dominant": float(np.mean(sigma_e > sigma_a)),
        "correlation_error_sigma_total": float(rho),
        "correlation_pval": float(pval),
    }

    return stats


def compute_calibration_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute calibration metrics using total uncertainty.

    Args:
        results: Output from heteroscedastic_mc_dropout()

    Returns:
        dict: Calibration metrics:
            - k_90: Calibration factor for 90% coverage
            - k_95: Calibration factor for 95% coverage (compare to baseline 11.65)
            - k_99: Calibration factor for 99% coverage
            - picp_gaussian_90: Coverage using 1.645 * sigma
            - picp_gaussian_95: Coverage using 1.96 * sigma
    """
    errors = np.abs(results["mean"] - results["ground_truth"])
    sigma_total = results["sigma_total"]

    # Normalized residuals: |error| / sigma
    normalized_residuals = errors / (sigma_total + 1e-8)

    # Calibration factors: What multiple of sigma is needed for X% coverage?
    k_90 = np.percentile(normalized_residuals, 90)
    k_95 = np.percentile(normalized_residuals, 95)
    k_99 = np.percentile(normalized_residuals, 99)

    # Gaussian interval coverage (should be ~90%, ~95% if well-calibrated)
    picp_90 = np.mean(errors <= 1.645 * sigma_total) * 100
    picp_95 = np.mean(errors <= 1.96 * sigma_total) * 100

    return {
        "k_90": float(k_90),
        "k_95": float(k_95),
        "k_99": float(k_99),
        "picp_gaussian_90": float(picp_90),
        "picp_gaussian_95": float(picp_95),
    }


def compare_with_baseline(
    het_results: Dict[str, np.ndarray], baseline_locked_path: str
) -> Dict[str, Dict[str, float]]:
    """
    Compare heteroscedastic results with locked baseline.

    Args:
        het_results: Output from heteroscedastic_mc_dropout()
        baseline_locked_path: Path to baseline_locked.json

    Returns:
        dict: Comparison showing improvements/degradations
    """
    import json
    from scipy.stats import spearmanr

    # Load baseline
    with open(baseline_locked_path, "r") as f:
        baseline = json.load(f)

    # Compute heteroscedastic metrics
    errors = np.abs(het_results["mean"] - het_results["ground_truth"])
    rho_het, _ = spearmanr(het_results["sigma_total"], errors)

    # Calibration
    normalized_residuals = errors / (het_results["sigma_total"] + 1e-8)
    k_95_het = np.percentile(normalized_residuals, 95)

    # R², MAE, RMSE on mean predictions
    targets = het_results["ground_truth"]
    predictions = het_results["mean"]
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_het = 1 - (ss_res / ss_tot)
    mae_het = np.mean(errors)
    rmse_het = np.sqrt(np.mean((targets - predictions) ** 2))

    # Baseline values
    baseline_metrics = baseline["metrics"]
    r2_base = baseline_metrics["mc_dropout"]["r2"]
    mae_base = baseline_metrics["mc_dropout"]["mae"]
    rho_base = baseline_metrics["mc_dropout"]["spearman_rho"]
    k_95_base = baseline_metrics["calibration"]["k_95"]

    # Comparison
    comparison = {
        "r2": {
            "baseline": r2_base,
            "heteroscedastic": r2_het,
            "diff": r2_het - r2_base,
            "pct_change": (r2_het - r2_base) / r2_base * 100,
        },
        "mae": {
            "baseline": mae_base,
            "heteroscedastic": mae_het,
            "diff": mae_het - mae_base,
            "pct_change": (mae_het - mae_base) / mae_base * 100,
        },
        "spearman_rho": {
            "baseline": rho_base,
            "heteroscedastic": rho_het,
            "diff": rho_het - rho_base,
            "pct_change": (rho_het - rho_base) / rho_base * 100,
        },
        "k_95": {
            "baseline": k_95_base,
            "heteroscedastic": k_95_het,
            "diff": k_95_het - k_95_base,
            "pct_change": (k_95_het - k_95_base) / k_95_base * 100,
        },
    }

    return comparison


# Example usage
if __name__ == "__main__":
    print("Heteroscedastic MC Dropout Inference Module")
    print("=" * 60)
    print("\nThis module provides utilities for:")
    print("1. Running MC Dropout with heteroscedastic models")
    print("2. Decomposing uncertainty into aleatoric/epistemic components")
    print("3. Computing calibration metrics")
    print("4. Comparing with baseline T8 results")
    print("\nUse these functions in your evaluation scripts.")
