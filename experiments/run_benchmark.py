"""Main Research Benchmark: Train and Compare PC-FNO vs Baseline U-Net."""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.physics.terrain import generate_synthetic_landscape
from src.physics.reaction_diffusion import solve_wildfire_pde
from src.models.fno_2d import FNO2d
from src.models.baseline_unet import BaselineUNet
from src.models.physics_loss import PhysicsInformedLoss
from src.pipeline.trainer import (
    train_single_model,
    evaluate_autoregressive_rollout,
)


def run_experiment():
    print("=" * 80)
    print(" PHYSIOEARTH-WILDFIRE: PHYSICS-CONSTRAINED NEURAL OPERATOR BENCHMARK")
    print(" Resolution-Invariant Wildfire Dynamics with Topographic & Fuel Conservation")
    print("=" * 80)

    device = torch.device("cpu")
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Generate Synthetic Environmental Landscapes
    print("\n[Step 1/5] Synthesizing topographies (NASA/DEM equivalent grids)...")
    land_train1 = generate_synthetic_landscape(grid_size=48, num_hills=3, base_wind=(1.4, 0.7), random_state=101)
    land_train2 = generate_synthetic_landscape(grid_size=48, num_hills=4, base_wind=(0.8, 1.2), random_state=202)
    land_test = generate_synthetic_landscape(grid_size=48, num_hills=5, base_wind=(1.2, 1.0), random_state=999)

    # 2. Compute Ground-Truth Numerical PDE Trajectories
    print("[Step 2/5] Simulating Reaction-Diffusion ground-truth trajectories (WRF-SFIRE surrogate)...")
    t0 = time.perf_counter()
    traj_train1 = solve_wildfire_pde(land_train1, ignition_center=(16, 16), num_steps=25)
    traj_train2 = solve_wildfire_pde(land_train2, ignition_center=(28, 20), num_steps=25)
    traj_test = solve_wildfire_pde(land_test, ignition_center=(18, 18), num_steps=25)
    pde_time_ms = (time.perf_counter() - t0) * 1000.0 / 3.0
    print(f" -> Numerical PDE Solver average simulation time: {pde_time_ms:.1f} ms per run")

    # 3. Instantiate Models & Loss Functions
    print("\n[Step 3/5] Instantiating models:")
    print("  • Model A: PhysioEarth PC-FNO (Fourier Neural Operator + Physics-Informed Loss)")
    print("  • Model B: Baseline U-Net (Unconstrained standard CNN)")

    fno_model = FNO2d(in_channels=7, out_channels=2, modes1=8, modes2=8, width=24, num_layers=3).to(device)
    unet_model = BaselineUNet(in_channels=7, out_channels=2, base_features=16).to(device)

    loss_physics = PhysicsInformedLoss(lambda_data=1.0, lambda_monotonic=2.5, lambda_firebreak=2.0)
    loss_standard = nn.MSELoss()

    # 4. Train Models
    print("\n[Step 4/5] Training models on historical fire regimes (12 epochs)...")
    print(" -> Training PhysioEarth PC-FNO...")
    train_single_model(fno_model, [traj_train1, traj_train2], [land_train1, land_train2], loss_physics, device, epochs=12, lr=3e-3)
    print(" -> Training Baseline U-Net...")
    train_single_model(unet_model, [traj_train1, traj_train2], [land_train1, land_train2], loss_standard, device, epochs=12, lr=3e-3)

    # 5. Multi-Step Autoregressive Rollout Evaluation
    print("\n[Step 5/5] Evaluating multi-step autoregressive rollouts (t=0 -> t=20)...")
    eval_fno = evaluate_autoregressive_rollout(fno_model, traj_test, land_test, device, rollout_steps=20)
    eval_unet = evaluate_autoregressive_rollout(unet_model, traj_test, land_test, device, rollout_steps=20)

    # Compile Benchmark Matrix
    speedup_fno = pde_time_ms / max(1e-3, eval_fno.inference_time_ms)
    speedup_unet = pde_time_ms / max(1e-3, eval_unet.inference_time_ms)

    metrics_data = [
        {
            "Model": "Numerical PDE Solver (Ground Truth)",
            "Temp_RMSE": 0.0000,
            "Fuel_RMSE": 0.0000,
            "Burned_IoU": 1.0000,
            "PVR_Total_Pct": 0.00,
            "Fuel_Regen_Pct": 0.00,
            "Firebreak_Breach_Pct": 0.00,
            "Latency_ms": round(pde_time_ms, 2),
            "Speedup": "1.0x (Baseline)",
        },
        {
            "Model": "Baseline 2D U-Net (Unconstrained)",
            "Temp_RMSE": round(eval_unet.rmse_temp, 4),
            "Fuel_RMSE": round(eval_unet.rmse_fuel, 4),
            "Burned_IoU": round(eval_unet.burned_iou, 4),
            "PVR_Total_Pct": round(eval_unet.pvr_metrics["pvr_total_percent"], 2),
            "Fuel_Regen_Pct": round(eval_unet.pvr_metrics["fuel_regeneration_rate"] * 100, 2),
            "Firebreak_Breach_Pct": round(eval_unet.pvr_metrics["firebreak_breach_rate"] * 100, 2),
            "Latency_ms": round(eval_unet.inference_time_ms, 2),
            "Speedup": f"{speedup_unet:.1f}x",
        },
        {
            "Model": "PhysioEarth PC-FNO (Ours)",
            "Temp_RMSE": round(eval_fno.rmse_temp, 4),
            "Fuel_RMSE": round(eval_fno.rmse_fuel, 4),
            "Burned_IoU": round(eval_fno.burned_iou, 4),
            "PVR_Total_Pct": round(eval_fno.pvr_metrics["pvr_total_percent"], 2),
            "Fuel_Regen_Pct": round(eval_fno.pvr_metrics["fuel_regeneration_rate"] * 100, 2),
            "Firebreak_Breach_Pct": round(eval_fno.pvr_metrics["firebreak_breach_rate"] * 100, 2),
            "Latency_ms": round(eval_fno.inference_time_ms, 2),
            "Speedup": f"{speedup_fno:.1f}x",
        },
    ]

    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(results_dir, "benchmark_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(" BENCHMARK RESULTS COMPARISON MATRIX")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"[+] Saved metrics CSV to: {csv_path}")

    # Generate Publication Figures
    generate_figures(eval_fno, eval_unet, traj_test, land_test, df, results_dir)


def generate_figures(eval_fno, eval_unet, traj_test, land_test, df, results_dir):
    """Renders academic-grade comparative figures."""
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})

    # Figure 1: Spatial Spread Comparison (PDE vs PC-FNO vs Baseline U-Net)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True)
    time_indices = [2, 7, 13, 19]

    for col, t_idx in enumerate(time_indices):
        # Row 1: Ground Truth PDE
        im0 = axes[0, col].imshow(traj_test.temperature[t_idx], cmap="inferno", vmin=0, vmax=1.2)
        axes[0, col].set_title(f"Step t = {t_idx} (Ground Truth)")
        if col == 0:
            axes[0, col].set_ylabel("Numerical PDE\n(Ground Truth)", fontweight="bold")

        # Row 2: Baseline U-Net
        axes[1, col].imshow(eval_unet.pred_temperature[t_idx], cmap="inferno", vmin=0, vmax=1.2)
        if col == 0:
            axes[1, col].set_ylabel("Baseline U-Net\n(Unconstrained)", fontweight="bold", color="#d62728")

        # Row 3: PhysioEarth PC-FNO (Ours)
        axes[2, col].imshow(eval_fno.pred_temperature[t_idx], cmap="inferno", vmin=0, vmax=1.2)
        if col == 0:
            axes[2, col].set_ylabel("PhysioEarth PC-FNO\n(Physics-Constrained)", fontweight="bold", color="#1f77b4")

    # Overlay firebreak mask (river) in cyan contour
    for ax in axes.flat:
        ax.contour(land_test.firebreaks, levels=[0.5], colors=["#00ffff"], linewidths=1.2, linestyles="dashed")

    plt.suptitle("Multi-Step Wildfire Temperature Field Propagation Over Topography\n(Cyan Dashed Line = River Firebreak)", y=0.98, fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig1_path = os.path.join(results_dir, "figure1_wildfire_spread_comparison.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Rendered Figure 1: {fig1_path}")

    # Figure 2: Physical Violation Rate Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    models = ["Baseline U-Net", "PhysioEarth PC-FNO"]
    pvr_vals = [df.loc[1, "PVR_Total_Pct"], df.loc[2, "PVR_Total_Pct"]]
    bar_colors = ["#f43f5e", "#10b981"]

    ax1.bar(models, pvr_vals, color=bar_colors, width=0.5, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Physical Violation Rate (PVR %)")
    ax1.set_title("Total Physical Conservation Violations\n(Lower is Better)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Sub-metrics breakdown
    metrics = ["Fuel Regeneration", "Firebreak Breach"]
    x = np.arange(len(metrics))
    width = 0.35
    unet_sub = [df.loc[1, "Fuel_Regen_Pct"], df.loc[1, "Firebreak_Breach_Pct"]]
    fno_sub = [df.loc[2, "Fuel_Regen_Pct"], df.loc[2, "Firebreak_Breach_Pct"]]

    ax2.bar(x - width/2, unet_sub, width, label="Baseline U-Net", color="#f43f5e")
    ax2.bar(x + width/2, fno_sub, width, label="PhysioEarth PC-FNO", color="#10b981")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel("Violation Percentage (%)")
    ax2.set_title("Conservation Law Breakdown", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(results_dir, "figure2_pvr_physics_violations.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Rendered Figure 2: {fig2_path}")


if __name__ == "__main__":
    run_experiment()
