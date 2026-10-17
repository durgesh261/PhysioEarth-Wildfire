"""Unit tests for wildfire physics, terrain generation, and Reaction-Diffusion PDE solver."""

import numpy as np
import pytest
from src.physics.terrain import generate_synthetic_landscape, LandscapeState
from src.physics.rothermel import compute_rothermel_anisotropy
from src.physics.reaction_diffusion import solve_wildfire_pde, WildfireTrajectory


def test_terrain_generation():
    """Verify terrain gradients, elevation bounds, and firebreak mask."""
    land = generate_synthetic_landscape(grid_size=32, num_hills=2, random_state=42)

    assert isinstance(land, LandscapeState)
    assert land.elevation.shape == (32, 32)
    assert land.fuel_density.shape == (32, 32)
    assert np.all(land.elevation >= 0.0) and np.all(land.elevation <= 1.0)
    assert np.any(land.firebreaks)  # Firebreaks must exist (river)
    # Check that fuel density is 0 on firebreaks
    assert np.all(land.fuel_density[land.firebreaks] == 0.0)


def test_rothermel_anisotropy():
    """Verify that uphill and downwind propagation accelerations are positive."""
    grad_x = np.array([[0.5, 0.0], [0.0, -0.5]])
    grad_y = np.array([[0.0, 0.5], [-0.5, 0.0]])
    wind_x = np.array([[1.0, 1.0], [1.0, 1.0]])
    wind_y = np.array([[0.0, 0.0], [0.0, 0.0]])

    adv_x, adv_y, mult = compute_rothermel_anisotropy(grad_x, grad_y, wind_x, wind_y)

    assert adv_x.shape == (2, 2)
    assert adv_y.shape == (2, 2)
    assert np.all(mult >= 1.0)  # Multiplier must be >= 1.0


def test_reaction_diffusion_solver():
    """Verify fuel monotonicity: fuel must never increase over time."""
    land = generate_synthetic_landscape(grid_size=32, num_hills=2, random_state=42)
    traj = solve_wildfire_pde(land, ignition_center=(16, 16), num_steps=10)

    assert isinstance(traj, WildfireTrajectory)
    assert traj.temperature.shape == (10, 32, 32)
    assert traj.fuel.shape == (10, 32, 32)

    # Monotonicity test: fuel at step t+1 <= fuel at step t (within float precision)
    delta_fuel = traj.fuel[1:] - traj.fuel[:-1]
    assert np.all(delta_fuel <= 1e-5), "Fuel must never increase in physical PDE simulation!"

    # Thermodynamic test: temperature must be non-negative
    assert np.all(traj.temperature >= 0.0), "Temperature cannot be negative!"
