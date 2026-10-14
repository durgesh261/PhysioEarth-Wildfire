"""Unit tests for models, recursive autophagy engine, and baseline metrics."""

import numpy as np
import pytest
from src.models.recursion_engine import generate_synthetic_manifold, simulate_recursive_autophagy
from src.models.baseline_metrics import compute_frechet_distance, compute_vendi_score, compute_coverage_precision


def test_synthetic_manifold_generation():
    """Verify synthetic manifold shape and properties."""
    pts_2d = generate_synthetic_manifold(n_samples=500, ambient_dim=2, random_state=42)
    assert pts_2d.shape == (500, 2)

    pts_high_dim = generate_synthetic_manifold(n_samples=300, ambient_dim=10, random_state=42)
    assert pts_high_dim.shape == (300, 10)


def test_recursive_autophagy_simulation():
    """Verify recursive generation creates correct number of trajectories and preserves shapes."""
    init_pts = generate_synthetic_manifold(n_samples=400, ambient_dim=2, random_state=42)
    res = simulate_recursive_autophagy(
        initial_manifold=init_pts,
        n_generations=3,
        n_components=4,
        sample_size=350,
        random_state=42,
    )

    assert len(res.generations) == 4  # Gen 0 + 3 recursive generations
    assert res.generations[0].shape == (400, 2)
    assert res.generations[1].shape == (350, 2)
    assert res.generations[3].shape == (350, 2)


def test_baseline_metrics():
    """Verify classical metrics (Fréchet, Vendi, Precision/Coverage)."""
    ref = np.random.randn(200, 4)
    eval_same = np.copy(ref)
    eval_shifted = ref + 3.0

    # Fréchet distance on identical data should be ~0
    fd_ident = compute_frechet_distance(ref, eval_same)
    assert np.isclose(fd_ident, 0.0, atol=1e-4)

    # Shifted data should yield positive Fréchet distance
    fd_shift = compute_frechet_distance(ref, eval_shifted)
    assert fd_shift > 0.0

    # Vendi score should be >= 1.0
    vs = compute_vendi_score(ref)
    assert vs >= 1.0

    # Precision & Coverage on identical data should be 1.0
    prec, cov = compute_coverage_precision(ref, eval_same, k=5)
    assert np.isclose(prec, 1.0, atol=1e-2)
    assert np.isclose(cov, 1.0, atol=1e-2)
