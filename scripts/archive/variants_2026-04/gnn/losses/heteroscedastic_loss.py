"""
Heteroscedastic Gaussian Loss for Graph Regression
===================================================
Implements negative log-likelihood loss for heteroscedastic regression where
the model predicts both mean (mu) and variance (sigma^2) for each output.

Loss = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2) + lambda * regularization

This loss encourages the model to:
1. Predict accurate means (mu close to y)
2. Predict honest uncertainties (sigma^2 reflects actual error magnitude)
3. Avoid variance collapse (regularization prevents sigma -> 0)

References:
- Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"
  https://arxiv.org/abs/1703.04977

Author: Mohd Zamin Quadri (for heteroscedastic UQ extension)
"""

import torch
import torch.nn as nn
import numpy as np


class HeteroscedasticGaussianLoss(nn.Module):
    """
    Negative Log-Likelihood loss for heteroscedastic Gaussian regression.
    
    The model outputs [mean, log_var] for each prediction.
    This loss trains the model to predict both the mean and the aleatoric uncertainty.
    
    Args:
        var_reg_lambda (float): Regularization weight to prevent variance collapse.
                                Higher values encourage larger predicted variances.
                                Recommended range: [0.001, 0.1]
        var_reg_type (str): Type of variance regularization:
                            - 'log': Penalize extreme log_var values (default)
                            - 'l1': Penalize small variances directly
                            - 'none': No regularization (not recommended)
    
    Example:
        >>> loss_fn = HeteroscedasticGaussianLoss(var_reg_lambda=0.01, var_reg_type='log')
        >>> mean = torch.tensor([1.0, 2.0, 3.0])
        >>> log_var = torch.tensor([-1.0, -0.5, 0.0])
        >>> target = torch.tensor([1.1, 2.3, 2.8])
        >>> loss = loss_fn(mean, log_var, target)
    """
    
    def __init__(self, var_reg_lambda=0.01, var_reg_type='log'):
        super().__init__()
        self.var_reg_lambda = var_reg_lambda
        self.var_reg_type = var_reg_type
        
        if var_reg_type not in ['log', 'l1', 'none']:
            raise ValueError(f"var_reg_type must be 'log', 'l1', or 'none', got {var_reg_type}")
    
    def forward(self, mean, log_var, target):
        """
        Compute heteroscedastic Gaussian NLL loss.
        
        Args:
            mean (torch.Tensor): Predicted means, shape (N,)
            log_var (torch.Tensor): Predicted log variances, shape (N,)
            target (torch.Tensor): Ground truth targets, shape (N,)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure inputs are 1D
        mean = mean.squeeze()
        log_var = log_var.squeeze()
        target = target.squeeze()
        
        # NUMERICAL SAFETY: Clamp log_var to prevent overflow/underflow
        # log_var in [-10, 10] => variance in [4.5e-5, 2.2e4]
        # This covers reasonable uncertainty ranges while preventing exp() overflow
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        # Variance from log_var (numerically stable)
        # Using log_var instead of raw var prevents gradient explosion
        variance = torch.exp(log_var)
        
        # Negative Log-Likelihood term
        # NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        #     = 0.5 * (log_var + (y - mu)^2 / exp(log_var))
        squared_error = (target - mean) ** 2
        nll = 0.5 * (log_var + squared_error / variance)
        
        # Regularization to prevent variance collapse
        if self.var_reg_type == 'l1':
            # Penalize small variances (encourage model to be uncertain)
            reg = -torch.mean(variance)
        elif self.var_reg_type == 'log':
            # Penalize extreme log_var values (keep log_var near 0)
            reg = torch.mean(log_var ** 2)
        else:  # 'none'
            reg = 0.0
        
        # Total loss
        total_loss = torch.mean(nll) + self.var_reg_lambda * reg
        
        # NUMERICAL SAFETY: Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(
                f"Loss is NaN or Inf! "
                f"mean range: [{mean.min().item():.2f}, {mean.max().item():.2f}], "
                f"log_var range: [{log_var.min().item():.2f}, {log_var.max().item():.2f}], "
                f"target range: [{target.min().item():.2f}, {target.max().item():.2f}]"
            )
        
        return total_loss
    
    def get_diagnostics(self, mean, log_var, target):
        """
        Compute diagnostic metrics for monitoring training.
        
        Returns:
            dict: Dictionary with diagnostic values:
                - nll: Mean NLL (without regularization)
                - reg: Regularization term value
                - mean_log_var: Mean predicted log variance
                - std_log_var: Std of predicted log variances
                - mean_var: Mean predicted variance
                - frac_underconfident: Fraction of predictions where |error| > sigma
        """
        mean = mean.squeeze()
        log_var = log_var.squeeze()
        target = target.squeeze()
        
        # NUMERICAL SAFETY: Clamp log_var (same as forward pass)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        variance = torch.exp(log_var)
        squared_error = (target - mean) ** 2
        nll = 0.5 * (log_var + squared_error / variance)
        
        if self.var_reg_type == 'l1':
            reg = -torch.mean(variance)
        elif self.var_reg_type == 'log':
            reg = torch.mean(log_var ** 2)
        else:
            reg = 0.0
        
        # Calibration: fraction where |error| > sigma (should be ~32% for Gaussian)
        abs_error = torch.abs(target - mean)
        sigma = torch.sqrt(variance)
        underconfident = (abs_error > sigma).float().mean()
        
        diagnostics = {
            'nll': nll.mean().item(),
            'reg': reg.item() if isinstance(reg, torch.Tensor) else reg,
            'mean_log_var': log_var.mean().item(),
            'std_log_var': log_var.std().item(),
            'mean_var': variance.mean().item(),
            'mean_sigma': sigma.mean().item(),
            'frac_underconfident': underconfident.item(),
        }
        
        # NUMERICAL SAFETY: Check for NaN/Inf in diagnostics
        for key, val in diagnostics.items():
            if isinstance(val, float) and (not np.isfinite(val)):
                raise ValueError(f"Diagnostic '{key}' is NaN or Inf: {val}")
        
        return diagnostics


class HeteroscedasticMSELoss(nn.Module):
    """
    Simplified heteroscedastic loss that only penalizes mean predictions.
    
    Useful for debugging or as a baseline. The model still outputs [mean, log_var]
    but the loss only uses the mean term (equivalent to standard MSE on mean).
    
    Args:
        ignore_var (bool): If True, completely ignore variance output
    """
    
    def __init__(self, ignore_var=True):
        super().__init__()
        self.ignore_var = ignore_var
        self.mse = nn.MSELoss()
    
    def forward(self, mean, log_var, target):
        """Compute MSE on mean predictions only."""
        mean = mean.squeeze()
        target = target.squeeze()
        return self.mse(mean, target)
