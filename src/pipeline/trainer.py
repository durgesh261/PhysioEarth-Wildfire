"""Benchmark Pipeline: Train & Evaluate Physics-Constrained FNO vs Baseline U-Net."""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ..models.fno_2d import FNO2d
from ..models.baseline_unet import BaselineUNet
from ..models.physics_loss import PhysicsInformedLoss
from ..physics.reaction_diffusion import solve_wildfire_pde, WildfireTrajectory
from ..physics.terrain import LandscapeState, generate_synthetic_landscape
from ..utils.metrics import (
    compute_burned_area_iou,
    compute_physical_violation_rate,
    compute_rollout_rmse,
)


@dataclass
class ModelRolloutOutput:
    """Stores predictions of an autoregressive rollout."""
    pred_temperature: np.ndarray   # Shape (T, H, W)
    pred_fuel: np.ndarray          # Shape (T, H, W)
    true_temperature: np.ndarray   # Shape (T, H, W)
    true_fuel: np.ndarray          # Shape (T, H, W)
    firebreak_mask: np.ndarray
    rmse_temp: float
    rmse_fuel: float
    burned_iou: float
    pvr_metrics: Dict[str, float]
    inference_time_ms: float


def prepare_input_tensor(
    t_state: np.ndarray,
    y_state: np.ndarray,
    landscape: LandscapeState,
    device: torch.device,
) -> torch.Tensor:
    """Stacks state and environmental channels into tensor shape (1, 7, H, W)."""
    # Channels: [T, Y, elevation, grad_x, grad_y, wind_x, wind_y]
    stacked = np.stack([
        t_state,
        y_state,
        landscape.elevation,
        landscape.grad_x,
        landscape.grad_y,
        landscape.wind_x,
        landscape.wind_y,
    ], axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
    return tensor


def train_single_model(
    model: torch.nn.Module,
    train_trajectories: List[WildfireTrajectory],
    landscapes: List[LandscapeState],
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int = 15,
    lr: float = 2e-3,
) -> List[float]:
    """Trains a model on single-step transition pairs (s_t, env) -> s_{t+1}."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    history = []

    # Build dataset pairs
    inputs_list = []
    targets_list = []
    prev_fuels_list = []
    firebreaks_list = []

    for traj, land in zip(train_trajectories, landscapes):
        fb_mask = torch.tensor(land.firebreaks, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
        for t in range(traj.time_steps - 1):
            inp = prepare_input_tensor(traj.temperature[t], traj.fuel[t], land, device)
            tgt = np.stack([traj.temperature[t + 1], traj.fuel[t + 1]], axis=0)
            tgt_tensor = torch.tensor(tgt, dtype=torch.float32, device=device).unsqueeze(0)
            prev_fuel_tensor = torch.tensor(traj.fuel[t], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            inputs_list.append(inp)
            targets_list.append(tgt_tensor)
            prev_fuels_list.append(prev_fuel_tensor)
            firebreaks_list.append(fb_mask)

    for epoch in range(epochs):
        epoch_loss = 0.0
        # Shuffle order
        indices = np.random.permutation(len(inputs_list))
        for idx in indices:
            optimizer.zero_grad()
            inp = inputs_list[idx]
            tgt = targets_list[idx]
            prev_fuel = prev_fuels_list[idx]
            fb = firebreaks_list[idx]

            pred = model(inp)

            if isinstance(criterion, PhysicsInformedLoss):
                loss, _ = criterion(pred, tgt, prev_fuel, fb)
            else:
                loss = criterion(pred, tgt)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(inputs_list)
        history.append(avg_loss)

    return history


def evaluate_autoregressive_rollout(
    model: torch.nn.Module,
    test_traj: WildfireTrajectory,
    landscape: LandscapeState,
    device: torch.device,
    rollout_steps: int = 25,
) -> ModelRolloutOutput:
    """Evaluates multi-step autoregressive rollout where predictions feed into subsequent inputs."""
    model.eval()
    H, W = landscape.elevation.shape
    pred_t = np.zeros((rollout_steps, H, W), dtype=np.float32)
    pred_y = np.zeros((rollout_steps, H, W), dtype=np.float32)

    # Initial condition from t = 0
    curr_t = test_traj.temperature[0]
    curr_y = test_traj.fuel[0]
    pred_t[0] = curr_t
    pred_y[0] = curr_y

    start_time = time.perf_counter()
    with torch.no_grad():
        for step in range(1, rollout_steps):
            inp = prepare_input_tensor(curr_t, curr_y, landscape, device)
            pred = model(inp).squeeze(0).cpu().numpy()

            curr_t = pred[0]
            curr_y = pred[1]

            pred_t[step] = curr_t
            pred_y[step] = curr_y

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # True trajectory slice
    true_t = test_traj.temperature[:rollout_steps]
    true_y = test_traj.fuel[:rollout_steps]

    rmse_t = compute_rollout_rmse(pred_t, true_t)
    rmse_y = compute_rollout_rmse(pred_y, true_y)
    iou = compute_burned_area_iou(pred_t[-1], true_t[-1])
    pvr = compute_physical_violation_rate(pred_y, pred_t, landscape.firebreaks)

    return ModelRolloutOutput(
        pred_temperature=pred_t,
        pred_fuel=pred_y,
        true_temperature=true_t,
        true_fuel=true_y,
        firebreak_mask=landscape.firebreaks,
        rmse_temp=rmse_t,
        rmse_fuel=rmse_y,
        burned_iou=iou,
        pvr_metrics=pvr,
        inference_time_ms=elapsed_ms,
    )
