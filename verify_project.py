"""Automated System Verification Script for PhysioEarth-Wildfire.

Runs end-to-end checks across mathematical foundations, spectral operators,
physics conservation laws, unit tests, and dashboard data integrity.
"""

import os
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def log_step(step_num: int, title: str):
    print(f"\n[Check {step_num}/6] {title}...")


def check_dependencies_and_fft():
    log_step(1, "Verifying Environment & PyTorch Spectral FFT Backend")
    print(f" -> Python Version : {sys.version.split()[0]}")
    print(f" -> PyTorch Version: {torch.__version__}")
    
    # Test 2D Complex FFT
    x = torch.randn(2, 3, 32, 32)
    fft_res = torch.fft.rfft2(x)
    ifft_res = torch.fft.irfft2(fft_res, s=(32, 32))
    diff = torch.norm(x - ifft_res).item()
    assert diff < 1e-4, f"FFT reconstruction error too high: {diff}"
    print(f" -> PyTorch FFT 2D Reconstruction: PASS (error: {diff:.2e})")
    return True


def check_physics_engine():
    log_step(2, "Verifying Reaction-Diffusion PDE & Fuel Monotonicity")
    from src.physics.terrain import generate_synthetic_landscape
    from src.physics.rothermel import compute_rothermel_anisotropy
    from src.physics.reaction_diffusion import solve_wildfire_pde

    land = generate_synthetic_landscape(grid_size=32, num_hills=2, random_state=42)
    adv_x, adv_y, mult = compute_rothermel_anisotropy(land.grad_x, land.grad_y, land.wind_x, land.wind_y)
    assert mult.min() >= 1.0, "Spread multiplier must be >= 1.0"
    print(f" -> Rothermel Directional Acceleration: PASS (max multiplier: {mult.max():.2f}x)")

    traj = solve_wildfire_pde(land, ignition_center=(16, 16), num_steps=12, dt=0.05)
    # Check strict fuel monotonicity: dY/dt <= 0
    delta_fuel = traj.fuel[1:] - traj.fuel[:-1]
    max_regen = float(delta_fuel.max())
    assert max_regen <= 1e-5, f"Unphysical fuel regeneration detected: {max_regen}"
    print(f" -> Fuel Conservation Law (dY/dt <= 0): PASS (max delta: {max_regen:.2e})")
    return True


def check_neural_operator():
    log_step(3, "Verifying 2D Fourier Neural Operator & Resolution Invariance")
    from src.models.fno_2d import FNO2d

    fno = FNO2d(in_channels=7, out_channels=2, modes1=6, modes2=6, width=16, num_layers=2)
    
    # Test on Base Grid (32x32)
    x32 = torch.randn(1, 7, 32, 32)
    out32 = fno(x32)
    assert out32.shape == (1, 2, 32, 32), f"Unexpected output shape: {out32.shape}"
    print(f" -> Grid Resolution 1 (32x32) Forward Pass: PASS {out32.shape}")

    # Test Resolution Invariance on Higher Resolution (48x48) without retraining
    x48 = torch.randn(1, 7, 48, 48)
    out48 = fno(x48)
    assert out48.shape == (1, 2, 48, 48), f"Resolution invariance failed: {out48.shape}"
    print(f" -> Zero-Shot High-Res (48x48) Evaluation: PASS {out48.shape} (Resolution-Invariant)")
    return True


def check_physics_loss():
    log_step(4, "Verifying Physics-Informed Conservation Loss Function")
    from src.models.physics_loss import PhysicsInformedLoss

    loss_fn = PhysicsInformedLoss(lambda_monotonic=5.0)
    prev_fuel = torch.full((1, 1, 16, 16), 0.5)
    firebreak = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    firebreak[:, :, 4:8, :] = True

    # Case A: Physically Valid (Fuel consumed to 0.3)
    pred_valid = torch.cat([torch.full((1, 1, 16, 16), 0.6), torch.full((1, 1, 16, 16), 0.3)], dim=1)
    _, m_valid = loss_fn(pred_valid, pred_valid, prev_fuel, firebreak)
    assert m_valid["loss_monotonic"] == 0.0

    # Case B: Physically Invalid (Fuel regenerated to 0.9)
    pred_invalid = torch.cat([torch.full((1, 1, 16, 16), 0.6), torch.full((1, 1, 16, 16), 0.9)], dim=1)
    _, m_invalid = loss_fn(pred_invalid, pred_valid, prev_fuel, firebreak)
    assert m_invalid["loss_monotonic"] > 0.0
    print(f" -> Physical Monotonicity Penalty: PASS (penalty response: {m_invalid['loss_monotonic']:.4f})")
    return True


def check_unit_tests():
    log_step(5, "Running Pytest Test Suite Programmatically")
    import pytest
    res = pytest.main(["-q", "tests"])
    assert res == 0, f"Pytest encountered failures (exit code: {res})"
    print(" -> Pytest Automated Test Suite: PASS (100% test pass rate)")
    return True


def check_dashboard_and_artifacts():
    log_step(6, "Verifying Digital Twin Dashboard Assets & Benchmark Outputs")
    required_files = [
        "README.md",
        "dashboard/index.html",
        "dashboard/styles.css",
        "dashboard/app.js",
        "dashboard/data.json",
        "dashboard/paper.html",
        "dashboard/docs.html",
        "experiments/results/benchmark_metrics.csv",
        "experiments/results/figure1_wildfire_spread_comparison.png",
        "experiments/results/figure2_pvr_physics_violations.png",
    ]

    missing = []
    for rel_path in required_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)

    if missing:
        raise FileNotFoundError(f"Missing required artifact files: {missing}")

    print(f" -> Verified {len(required_files)}/{len(required_files)} system files and visual artifacts: PASS")
    return True


def main():
    print("=" * 80)
    print(" PHYSIOEARTH-WILDFIRE: SYSTEM INTEGRITY & RESEARCH VERIFICATION")
    print("=" * 80)
    t_start = time.perf_counter()

    try:
        check_dependencies_and_fft()
        check_physics_engine()
        check_neural_operator()
        check_physics_loss()
        check_unit_tests()
        check_dashboard_and_artifacts()
    except Exception as e:
        print(f"\n[FAIL] Verification stopped due to error: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f" [SUCCESS] ALL 6 SYSTEM VERIFICATION CHECKPOINTS PASSED ({elapsed:.2f}s)")
    print("=" * 80)
    print(" -> Public Repository : https://github.com/durgesh261/PhysioEarth-Wildfire")
    print(" -> Live Digital Twin : https://durgesh261.github.io/PhysioEarth-Wildfire/")
    print(" -> Research Paper    : https://durgesh261.github.io/PhysioEarth-Wildfire/paper.html")
    print(" -> Documentation     : https://durgesh261.github.io/PhysioEarth-Wildfire/docs.html")
    print("=" * 80)


if __name__ == "__main__":
    main()
