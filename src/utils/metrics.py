"""Physical and Empirical Evaluation Metrics for Wildfire Dynamics."""

from typing import Dict, Tuple
import numpy as np


def compute_physical_violation_rate(
    fuel_trajectory: np.ndarray,          # Shape (T, H, W)
    temp_trajectory: np.ndarray,          # Shape (T, H, W)
    firebreak_mask: np.ndarray,           # Shape (H, W) boolean
    eps: float = 1e-3,
) -> Dict[str, float]:
    """Quantifies the rate of physical conservation law violations across a predicted rollout.

    Returns:
        Dict containing:
        - fuel_regeneration_rate: Fraction of space-time pixels where dY/dt > eps.
        - firebreak_breach_rate: Fraction of firebreak pixels ignited (T > 0.25).
        - negative_energy_rate: Fraction of pixels with unphysical T < 0.
        - total_physical_violation_rate (PVR): Combined physical inconsistency score.
    """
    T, H, W = fuel_trajectory.shape
    total_pixels = T * H * W

    # 1. Fuel Regeneration Violation: Delta Y = Y_{t} - Y_{t-1} > eps
    delta_fuel = fuel_trajectory[1:] - fuel_trajectory[:-1]
    fuel_violations = np.sum(delta_fuel > eps)
    fuel_violation_rate = float(fuel_violations / ((T - 1) * H * W))

    # 2. Firebreak Barrier Breach Violation
    firebreak_pixels = np.sum(firebreak_mask)
    if firebreak_pixels > 0:
        breached = temp_trajectory[:, firebreak_mask] > 0.25
        firebreak_breach_rate = float(np.sum(breached) / (T * firebreak_pixels))
    else:
        firebreak_breach_rate = 0.0

    # 3. Domain Bounds (Negative Temperature / Negative Fuel)
    neg_temp = np.sum(temp_trajectory < -eps)
    neg_fuel = np.sum(fuel_trajectory < -eps)
    bounds_violation_rate = float((neg_temp + neg_fuel) / (2 * total_pixels))

    # Combined Physical Violation Rate (PVR)
    pvr = float((fuel_violation_rate * 0.4 + firebreak_breach_rate * 0.4 + bounds_violation_rate * 0.2) * 100.0)

    return {
        "pvr_total_percent": pvr,
        "fuel_regeneration_rate": fuel_violation_rate,
        "firebreak_breach_rate": firebreak_breach_rate,
        "bounds_violation_rate": bounds_violation_rate,
    }


def compute_burned_area_iou(
    pred_burned: np.ndarray,  # Shape (H, W) boolean or [0, 1]
    true_burned: np.ndarray,  # Shape (H, W) boolean or [0, 1]
) -> float:
    """Computes Intersection-over-Union (IoU) of burned scar area."""
    p_mask = pred_burned > 0.35
    t_mask = true_burned > 0.35

    intersection = np.sum(p_mask & t_mask)
    union = np.sum(p_mask | t_mask)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_rollout_rmse(
    pred_field: np.ndarray,  # Shape (T, H, W)
    true_field: np.ndarray,  # Shape (T, H, W)
) -> float:
    """Computes global root mean squared error across time steps."""
    return float(np.sqrt(np.mean((pred_field - true_field) ** 2)))
