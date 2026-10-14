"""Unit tests for Vietoris-Rips filtration and persistence extraction."""

import numpy as np
import pytest
from src.topology.filtration import compute_vietoris_rips, PersistenceResult


def test_vietoris_rips_circle():
    """A clean circle should produce exactly one high-persistence H1 feature."""
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])

    res = compute_vietoris_rips(pts, max_dim=1, subsample_n=None)

    assert isinstance(res, PersistenceResult)
    assert len(res.h0_intervals) > 0
    assert len(res.h1_intervals) > 0

    # The most persistent H1 feature should have a substantial lifetime (> 0.5)
    max_h1_life = np.max(res.h1_lifetimes)
    assert max_h1_life > 0.4
    assert res.num_points == 100
    assert res.dim == 2


def test_vietoris_rips_subsampling():
    """Verify deterministic subsampling when N > subsample_n."""
    pts = np.random.randn(500, 3)
    res = compute_vietoris_rips(pts, max_dim=1, subsample_n=150, random_state=42)

    assert res.num_points == 150
    assert res.dim == 3
    assert len(res.h0_lifetimes) == 149  # 150 points - 1 infinite component
