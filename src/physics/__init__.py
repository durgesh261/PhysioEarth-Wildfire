"""Physics module containing terrain generation, Rothermel rate-of-spread, and Reaction-Diffusion PDE solver."""

from .terrain import LandscapeState, generate_synthetic_landscape
from .rothermel import compute_rothermel_anisotropy
from .reaction_diffusion import WildfireTrajectory, solve_wildfire_pde

__all__ = [
    "LandscapeState",
    "generate_synthetic_landscape",
    "compute_rothermel_anisotropy",
    "WildfireTrajectory",
    "solve_wildfire_pde",
]
