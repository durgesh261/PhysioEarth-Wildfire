"""Unit tests for physical evaluation metrics and violation rates."""

import numpy as np
import pytest
from src.utils.metrics import (
    compute_burned_area_iou,
    compute_physical_violation_rate,
    compute_rollout_rmse,
)


def test_physical_violation_rate_clean():
    """A strictly physical trajectory should have 0.0 PVR."""
    # Fuel strictly decreasing: 1.0 -> 0.8 -> 0.5
    fuel_traj = np.array([
        np.full((8, 8), 1.0),
        np.full((8, 8), 0.8),
        np.full((8, 8), 0.5),
    ])
    temp_traj = np.array([
        np.full((8, 8), 0.1),
        np.full((8, 8), 0.5),
        np.full((8, 8), 0.3),
    ])
    firebreak = np.zeros((8, 8), dtype=bool)

    pvr = compute_physical_violation_rate(fuel_traj, temp_traj, firebreak)
    assert pvr["pvr_total_percent"] == 0.0
    assert pvr["fuel_regeneration_rate"] == 0.0


def test_physical_violation_rate_unphysical():
    """Unphysical fuel regeneration or barrier breach must yield positive PVR."""
    # Fuel regenerating: 0.5 -> 0.9 (unphysical!)
    fuel_traj = np.array([
        np.full((8, 8), 0.5),
        np.full((8, 8), 0.9),
    ])
    temp_traj = np.array([
        np.full((8, 8), 0.2),
        np.full((8, 8), 0.7),
    ])
    firebreak = np.zeros((8, 8), dtype=bool)
    firebreak[2:4, 2:4] = True  # river barrier

    pvr = compute_physical_violation_rate(fuel_traj, temp_traj, firebreak)
    assert pvr["pvr_total_percent"] > 0.0
    assert pvr["fuel_regeneration_rate"] > 0.0
    assert pvr["firebreak_breach_rate"] > 0.0


def test_iou_and_rmse():
    """Verify IoU and RMSE bounds and behavior."""
    mask1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask2 = np.copy(mask1)
    assert compute_burned_area_iou(mask1, mask2) == 1.0

    mask_opposite = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert compute_burned_area_iou(mask1, mask_opposite) == 0.0

    rmse = compute_rollout_rmse(mask1, mask2)
    assert rmse == 0.0
