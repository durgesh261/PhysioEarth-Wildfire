"""Physics-Informed Loss Functions Enforcing Conservation Laws and Monotonicity."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """Enforces physical conservation laws alongside supervised empirical loss.

    1. Data Fidelity: MSE on Temperature and Fuel.
    2. Monotonic Fuel Depletion: Prevents unphysical fuel regeneration (dY/dt <= 0).
    3. Firebreak Conservation: Ensures zero combustion across natural barriers (water/rock).
    4. Thermodynamic Bounds: Enforces non-negative temperatures and [0, 1] fuel fractions.
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_monotonic: float = 2.5,
        lambda_firebreak: float = 2.0,
        lambda_bounds: float = 1.5,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_monotonic = lambda_monotonic
        self.lambda_firebreak = lambda_firebreak
        self.lambda_bounds = lambda_bounds

    def forward(
        self,
        pred_state: torch.Tensor,     # Shape (B, 2, H, W) -> [T_pred, Y_pred]
        target_state: torch.Tensor,   # Shape (B, 2, H, W) -> [T_true, Y_true]
        prev_fuel: torch.Tensor,      # Shape (B, 1, H, W) -> Y_t (previous fuel state)
        firebreak_mask: torch.Tensor, # Shape (B, 1, H, W) -> boolean mask where fuel == 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        t_pred = pred_state[:, 0:1]
        y_pred = pred_state[:, 1:2]
        t_true = target_state[:, 0:1]
        y_true = target_state[:, 1:2]

        # 1. Supervised Data Loss
        loss_t = F.mse_loss(t_pred, t_true)
        loss_y = F.mse_loss(y_pred, y_true)
        loss_data = loss_t + loss_y

        # 2. Fuel Monotonicity Violation Loss (dY/dt <= 0)
        # Fuel cannot increase: delta_Y = y_pred - prev_fuel must be <= 0
        fuel_increase = F.relu(y_pred - prev_fuel)
        loss_monotonic = torch.mean(fuel_increase ** 2)

        # 3. Firebreak Barrier Violation Loss
        # Active combustion (T > 0.1) cannot exist on natural firebreaks (rivers, lakes)
        firebreak_penetration = t_pred * firebreak_mask.float()
        loss_firebreak = torch.mean(firebreak_penetration ** 2)

        # 4. Physical Domain Bounds Violation Loss
        # T >= 0, 0 <= Y <= 1
        loss_bounds = (
            torch.mean(F.relu(-t_pred) ** 2) +
            torch.mean(F.relu(-y_pred) ** 2) +
            torch.mean(F.relu(y_pred - 1.0) ** 2)
        )

        total_loss = (
            self.lambda_data * loss_data +
            self.lambda_monotonic * loss_monotonic +
            self.lambda_firebreak * loss_firebreak +
            self.lambda_bounds * loss_bounds
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_data": float(loss_data.item()),
            "loss_monotonic": float(loss_monotonic.item()),
            "loss_firebreak": float(loss_firebreak.item()),
            "loss_bounds": float(loss_bounds.item()),
        }

        return total_loss, metrics
