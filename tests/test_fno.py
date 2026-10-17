"""Unit tests for 2D Fourier Neural Operator (FNO) and Resolution Invariance."""

import pytest
import torch
from src.models.fno_2d import FNO2d, SpectralConv2d
from src.models.physics_loss import PhysicsInformedLoss


def test_spectral_conv2d_forward():
    """Verify spectral convolution shape on complex frequency representations."""
    conv = SpectralConv2d(in_channels=4, out_channels=4, modes1=6, modes2=6)
    x = torch.randn(2, 4, 24, 24)
    out = conv(x)

    assert out.shape == (2, 4, 24, 24)
    assert not torch.isnan(out).any()


def test_fno_resolution_invariance():
    """Key scientific property: An FNO trained on one grid must evaluate on arbitrary resolutions."""
    fno = FNO2d(in_channels=7, out_channels=2, modes1=6, modes2=6, width=16, num_layers=2)

    # Resolution 1: 32x32
    x_32 = torch.randn(1, 7, 32, 32)
    out_32 = fno(x_32)
    assert out_32.shape == (1, 2, 32, 32)

    # Resolution 2: 48x48 (Higher-resolution evaluation without retraining)
    x_48 = torch.randn(1, 7, 48, 48)
    out_48 = fno(x_48)
    assert out_48.shape == (1, 2, 48, 48)


def test_physics_informed_loss():
    """Verify that fuel regeneration triggers high penalty loss."""
    loss_fn = PhysicsInformedLoss(lambda_monotonic=5.0)

    prev_fuel = torch.full((1, 1, 16, 16), 0.5)
    firebreak = torch.zeros((1, 1, 16, 16), dtype=torch.bool)

    # Valid physical transition: fuel decreases to 0.4
    pred_valid = torch.cat([torch.full((1, 1, 16, 16), 0.6), torch.full((1, 1, 16, 16), 0.4)], dim=1)
    target = pred_valid.clone()
    _, metrics_valid = loss_fn(pred_valid, target, prev_fuel, firebreak)
    assert metrics_valid["loss_monotonic"] == 0.0

    # Unphysical transition: fuel increases to 0.8 (phantom biomass)
    pred_invalid = torch.cat([torch.full((1, 1, 16, 16), 0.6), torch.full((1, 1, 16, 16), 0.8)], dim=1)
    _, metrics_invalid = loss_fn(pred_invalid, target, prev_fuel, firebreak)
    assert metrics_invalid["loss_monotonic"] > 0.0
