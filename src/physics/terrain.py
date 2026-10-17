"""Terrain, Fuel, and Wind Vector Field Generation for Wildfire Propagation."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class LandscapeState:
    """Stores spatial arrays representing environmental state."""
    elevation: np.ndarray        # Shape (H, W) in meters
    grad_x: np.ndarray           # dh/dx slope gradient
    grad_y: np.ndarray           # dh/dy slope gradient
    fuel_density: np.ndarray     # Biomass fuel fraction Y in [0, 1]
    wind_x: np.ndarray           # Atmospheric wind vector x-component
    wind_y: np.ndarray           # Atmospheric wind vector y-component
    firebreaks: np.ndarray       # Boolean mask where fuel == 0 (rivers, rock, lakes)


def generate_synthetic_landscape(
    grid_size: int = 64,
    num_hills: int = 4,
    base_wind: Tuple[float, float] = (1.5, 0.8),
    random_state: int = 42,
) -> LandscapeState:
    """Generates a realistic heterogeneous landscape with hills, valleys, fuel, and wind fields.

    Args:
        grid_size: Spatial grid dimension (H = W = grid_size).
        num_hills: Number of Gaussian topographical peaks.
        base_wind: Global atmospheric wind vector (w_x, w_y).
        random_state: Seed for reproducibility.

    Returns:
        LandscapeState containing normalized physical layers.
    """
    rng = np.random.default_rng(random_state)
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(x, y)

    # 1. Generate multi-peak topographic elevation
    elevation = np.zeros((grid_size, grid_size), dtype=np.float32)
    for _ in range(num_hills):
        cx, cy = rng.uniform(-2.0, 2.0, size=2)
        height = rng.uniform(0.8, 2.5)
        spread = rng.uniform(0.6, 1.4)
        elevation += height * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * spread**2))

    # Add gentle undulating slope
    elevation += 0.2 * xx + 0.15 * yy
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-6)

    # Compute numerical terrain gradients dh/dx, dh/dy
    grad_y, grad_x = np.gradient(elevation, edge_order=2)

    # 2. Generate heterogeneous fuel distribution (vegetation density)
    fuel = 0.85 + 0.15 * np.sin(1.5 * xx) * np.cos(1.5 * yy)
    fuel += rng.normal(0, 0.04, size=(grid_size, grid_size))
    fuel = np.clip(fuel, 0.0, 1.0).astype(np.float32)

    # Add natural firebreaks (river / rocky canyon across diagonal)
    river_path = np.abs(yy - 0.4 * np.sin(xx * 2.0)) < 0.18
    fuel[river_path] = 0.0
    firebreaks = fuel == 0.0

    # 3. Ambient wind field with localized topographic channeling
    # Wind speeds up over mountain ridges (Bernoulli effect)
    wind_x = np.full((grid_size, grid_size), base_wind[0], dtype=np.float32)
    wind_y = np.full((grid_size, grid_size), base_wind[1], dtype=np.float32)

    # Topographic wind deflection: wind curves around steep elevation peaks
    wind_x -= 0.3 * grad_x
    wind_y -= 0.3 * grad_y

    return LandscapeState(
        elevation=elevation,
        grad_x=grad_x.astype(np.float32),
        grad_y=grad_y.astype(np.float32),
        fuel_density=fuel,
        wind_x=wind_x,
        wind_y=wind_y,
        firebreaks=firebreaks,
    )
