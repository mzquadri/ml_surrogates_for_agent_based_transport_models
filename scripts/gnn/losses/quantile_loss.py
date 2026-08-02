"""
Pinball (Quantile) Loss for Conformalized Quantile Regression (CQR)
====================================================================
Implements the pinball loss for simultaneous lower and upper quantile
regression at levels tau_lo = alpha/2 and tau_hi = 1 - alpha/2.

Loss formulation follows Romano, Patterson & Candes (2019) NeurIPS,
arXiv:1905.03222, and the yromano/cqr reference implementation
(cqr/torch_models.py, AllQuantileLoss).

For each node:
  errors     = y - q           (target minus prediction)
  pinball(q, tau) = max((tau - 1) * errors, tau * errors)

Total loss = mean_nodes(pinball(q_lo, tau_lo) + pinball(q_hi, tau_hi))
           = sum over quantiles, then mean over samples.

This matches AllQuantileLoss.forward() from the reference repo exactly:
  loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

Author: Mohd Zamin Quadri (CQR UQ extension)
Reference: Romano et al. (2019), yromano/cqr
"""

import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """
    Joint pinball loss for lower and upper quantile regression.

    Trains a model to simultaneously predict:
      - q_lo: the (alpha/2) quantile  (e.g. 0.05 for 90% PI)
      - q_hi: the (1 - alpha/2) quantile (e.g. 0.95 for 90% PI)

    The loss is the sum of the two individual pinball losses, averaged
    over all nodes/samples.

    Args:
        alpha (float): Miscoverage level. Default 0.10 gives 90% intervals.
                       tau_lo = alpha/2 = 0.05
                       tau_hi = 1 - alpha/2 = 0.95

    Example:
        >>> loss_fn = PinballLoss(alpha=0.10)
        >>> q_lo = torch.tensor([-1.0, -2.0])
        >>> q_hi = torch.tensor([ 1.0,  2.0])
        >>> y    = torch.tensor([ 0.5,  1.5])
        >>> loss = loss_fn(q_lo, q_hi, y)
    """

    def __init__(self, alpha: float = 0.10):
        super().__init__()
        self.alpha   = alpha
        self.tau_lo  = alpha / 2.0          # 0.05 for alpha=0.10
        self.tau_hi  = 1.0 - alpha / 2.0    # 0.95 for alpha=0.10

    def forward(self, q_lo: torch.Tensor, q_hi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute joint pinball loss.

        Args:
            q_lo (torch.Tensor): Predicted lower quantile, shape (N,) or (N, 1)
            q_hi (torch.Tensor): Predicted upper quantile, shape (N,) or (N, 1)
            y    (torch.Tensor): Ground truth targets,    shape (N,) or (N, 1)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Squeeze all to 1D [N] for consistent computation
        q_lo = q_lo.squeeze()
        q_hi = q_hi.squeeze()
        y    = y.squeeze()

        # Pinball errors: errors = y - q  (target - prediction)
        # Matches AllQuantileLoss: errors = target - pred
        errors_lo = y - q_lo
        errors_hi = y - q_hi

        # Pinball loss per node
        # pinball(q, tau) = max((tau - 1) * errors, tau * errors)
        loss_lo = torch.max(
            (self.tau_lo - 1.0) * errors_lo,
            self.tau_lo          * errors_lo
        )
        loss_hi = torch.max(
            (self.tau_hi - 1.0) * errors_hi,
            self.tau_hi          * errors_hi
        )

        # Sum over quantiles, mean over nodes
        # Matches: torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        total_loss = torch.mean(loss_lo + loss_hi)

        return total_loss
