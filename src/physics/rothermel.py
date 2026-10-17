from typing import Tuple
import numpy as np


def compute_rothermel_anisotropy(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    wind_x: np.ndarray,
    wind_y: np.ndarray,
    c_wind: float = 0.8,
    c_slope: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes directional spread coefficients based on Rothermel's (1972) equations.

    R = R_0 * (1 + Phi_W + Phi_S)

    Args:
        grad_x: Slope gradient in x direction (dh/dx).
        grad_y: Slope gradient in y direction (dh/dy).
        wind_x: Wind vector in x direction.
        wind_y: Wind vector in y direction.
        c_wind: Coefficient scaling wind advection.
        c_slope: Coefficient scaling topographic uphill convection.

    Returns:
        effective_advection_x: Combined net directional propagation velocity in x.
        effective_advection_y: Combined net directional propagation velocity in y.
        spread_multiplier: Scalar scalar expansion factor (isotropic + anisotropic).
    """
    # 1. Slope factor Phi_S: Fire propagates exponentially faster uphill
    # Upslope vector points in the direction of increasing elevation
    slope_mag = np.sqrt(grad_x**2 + grad_y**2)
    phi_slope_x = c_slope * grad_x * np.maximum(0.0, slope_mag)
    phi_slope_y = c_slope * grad_y * np.maximum(0.0, slope_mag)

    # 2. Wind factor Phi_W: Wind pushes flame front downstream
    phi_wind_x = c_wind * wind_x
    phi_wind_y = c_wind * wind_y

    # Combined effective advection vector
    eff_adv_x = phi_wind_x + phi_slope_x
    eff_adv_y = phi_wind_y + phi_slope_y

    # Scalar propagation intensity
    spread_multiplier = 1.0 + np.sqrt(eff_adv_x**2 + eff_adv_y**2)

    return eff_adv_x.astype(np.float32), eff_adv_y.astype(np.float32), spread_multiplier.astype(np.float32)
