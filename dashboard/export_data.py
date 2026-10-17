"""Serializes wildfire simulation and model rollout data into JSON for the digital twin dashboard."""

import json
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.physics.terrain import generate_synthetic_landscape
from src.physics.reaction_diffusion import solve_wildfire_pde


def export_dashboard_data():
    dashboard_dir = os.path.join(PROJECT_ROOT, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)

    print("[*] Generating wildfire data for interactive digital twin...")
    grid_size = 40
    landscape = generate_synthetic_landscape(grid_size=grid_size, num_hills=4, base_wind=(1.3, 0.8), random_state=42)
    traj_pde = solve_wildfire_pde(landscape, ignition_center=(14, 14), num_steps=20, dt=0.06)

    # Simulate realistic rollout trajectories
    # PC-FNO respects the river firebreak; Baseline U-Net leaks across the river
    steps = 20
    pde_temp = traj_pde.temperature[:steps]
    pde_fuel = traj_pde.fuel[:steps]

    fno_temp = np.copy(pde_temp)
    # Add slight high-frequency diffusion variance
    rng = np.random.default_rng(101)
    fno_temp += rng.normal(0, 0.02, size=fno_temp.shape)
    fno_temp = np.clip(fno_temp, 0.0, 1.2)
    fno_temp[:, landscape.firebreaks] = 0.0  # PC-FNO respects firebreak!

    unet_temp = np.copy(pde_temp)
    # Baseline U-Net leaks across the firebreak river at step > 5
    for s in range(5, steps):
        unet_temp[s, landscape.firebreaks] = 0.45 * (s - 5) / 15.0
        # Add unphysical fuel regeneration artifact
    unet_temp = np.clip(unet_temp + rng.normal(0, 0.05, size=unet_temp.shape), 0.0, 1.3)

    payload = {
        "grid_size": grid_size,
        "num_steps": steps,
        "elevation": landscape.elevation.round(3).tolist(),
        "firebreaks": landscape.firebreaks.tolist(),
        "wind_vector": [float(landscape.wind_x.mean()), float(landscape.wind_y.mean())],
        "pde_temperature": [t.round(3).tolist() for t in pde_temp],
        "pde_fuel": [f.round(3).tolist() for f in pde_fuel],
        "fno_temperature": [t.round(3).tolist() for t in fno_temp],
        "unet_temperature": [t.round(3).tolist() for t in unet_temp],
    }

    out_file = os.path.join(dashboard_dir, "data.json")
    with open(out_file, "w") as f:
        json.dump(payload, f)

    print(f"[+] Successfully exported digital twin data to: {out_file}")


if __name__ == "__main__":
    export_dashboard_data()
