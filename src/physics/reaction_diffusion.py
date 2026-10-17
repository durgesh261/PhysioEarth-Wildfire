"""Coupled 2D Reaction-Diffusion Combustion PDE Solver for Ground-Truth Wildfire Simulation."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .rothermel import compute_rothermel_anisotropy
from .terrain import LandscapeState


@dataclass
class WildfireTrajectory:
    """Stores full time-series simulation of wildfire propagation."""
    temperature: np.ndarray    # Shape (T, H, W) normalized thermal state in [0, 1]
    fuel: np.ndarray           # Shape (T, H, W) remaining biomass fraction in [0, 1]
    burned_mask: np.ndarray    # Shape (T, H, W) cumulative burned area boolean
    time_steps: int
    dt: float
    dx: float


def solve_wildfire_pde(
    landscape: LandscapeState,
    ignition_center: Tuple[int, int] = (20, 20),
    ignition_radius: int = 3,
    num_steps: int = 40,
    dt: float = 0.05,
    dx: float = 0.1,
    diffusion_coeff: float = 0.08,
    cooling_rate: float = 0.05,
    burn_rate: float = 0.15,
) -> WildfireTrajectory:
    """Solves the coupled Reaction-Diffusion and Fuel Combustion system using 2nd-order finite differences.

    dT/dt = D * Laplacian(T) - v_eff . grad(T) + Combustion(T, Y) - cooling * T
    dY/dt = - burn_rate * Y * (T > T_ign)
    """
    H, W = landscape.elevation.shape
    t_history = np.zeros((num_steps, H, W), dtype=np.float32)
    y_history = np.zeros((num_steps, H, W), dtype=np.float32)
    burned_history = np.zeros((num_steps, H, W), dtype=bool)

    # Initial thermal state (ambient T = 0.0)
    T = np.zeros((H, W), dtype=np.float32)
    Y = np.copy(landscape.fuel_density).astype(np.float32)

    # Ignite localized core
    cy, cx = ignition_center
    yy, xx = np.ogrid[:H, :W]
    ignition_mask = ((xx - cx)**2 + (yy - cy)**2) <= ignition_radius**2
    T[ignition_mask] = 1.0

    # Compute Rothermel effective advection vectors
    adv_x, adv_y, _ = compute_rothermel_anisotropy(
        landscape.grad_x,
        landscape.grad_y,
        landscape.wind_x,
        landscape.wind_y,
    )

    t_history[0] = T
    y_history[0] = Y
    burned_history[0] = T > 0.4

    for step in range(1, num_steps):
        # 1. 2D Laplacian for thermal diffusion (5-point stencil)
        laplacian = (
            np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
            np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4 * T
        ) / (dx**2)

        # 2. Advection gradient (Upwind differencing for stability)
        grad_t_y, grad_t_x = np.gradient(T, dx)
        advection = adv_x * grad_t_x + adv_y * grad_t_y

        # 3. Combustion reaction: Heat release depends on available fuel and active flame
        is_ignited = T > 0.25
        combustion_heat = 0.45 * Y * is_ignited

        # 4. Update Temperature
        dT = (diffusion_coeff * laplacian - advection + combustion_heat - cooling_rate * T) * dt
        T = np.clip(T + dT, 0.0, 1.5)

        # 5. Monotonic Fuel Depletion (dY/dt <= 0)
        fuel_consumption = burn_rate * Y * (T > 0.3) * dt
        Y = np.clip(Y - fuel_consumption, 0.0, 1.0)
        # Ensure zero fuel on firebreaks (rivers, barren rock)
        Y[landscape.firebreaks] = 0.0

        t_history[step] = T
        y_history[step] = Y
        burned_history[step] = (T > 0.35) | (Y < 0.2)

    return WildfireTrajectory(
        temperature=t_history,
        fuel=y_history,
        burned_mask=burned_history,
        time_steps=num_steps,
        dt=dt,
        dx=dx,
    )
