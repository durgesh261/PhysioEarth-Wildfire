"""Unit tests for topological metrics (Persistent Entropy, Bottleneck/Wasserstein, Betti)."""

import numpy as np
import pytest
from src.topology.metrics import (
    compute_persistent_entropy,
    compute_bottleneck_distance,
    compute_wasserstein_distance,
    compute_betti_numbers,
)


def test_persistent_entropy_properties():
    """Verify entropy bounds and edge cases."""
    # Empty lifetimes should return 0.0
    assert compute_persistent_entropy(np.array([])) == 0.0

    # Single feature should return 0.0 when normalized
    assert compute_persistent_entropy(np.array([1.5]), normalize=True) == 0.0

    # Uniform lifetimes should achieve maximum normalized entropy (1.0)
    uniform_lifetimes = np.array([2.0, 2.0, 2.0, 2.0])
    ent = compute_persistent_entropy(uniform_lifetimes, normalize=True)
    assert np.isclose(ent, 1.0, atol=1e-5)


def test_bottleneck_and_wasserstein_distance():
    """Verify metric properties: identity of indiscernibles and non-negativity."""
    dgm1 = np.array([[0.1, 0.9], [0.2, 0.6], [0.0, 0.4]])
    dgm2 = np.copy(dgm1)

    # Identical diagrams should have 0 distance
    w_inf = compute_bottleneck_distance(dgm1, dgm2)
    w_2 = compute_wasserstein_distance(dgm1, dgm2, p=2)
    assert np.isclose(w_inf, 0.0, atol=1e-5)
    assert np.isclose(w_2, 0.0, atol=1e-5)

    # Perturbed diagram should yield positive distance
    dgm_perturbed = np.array([[0.1, 1.2], [0.2, 0.6], [0.0, 0.4]])
    w_inf_pert = compute_bottleneck_distance(dgm1, dgm_perturbed)
    assert w_inf_pert > 0.0


def test_betti_numbers():
    """Verify Betti curve counts alive features correctly."""
    dgm = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.3]])
    eps = np.array([0.05, 0.25, 0.5, 0.7, 0.9])
    betti = compute_betti_numbers(dgm, eps)

    # At 0.05: 0 features alive
    assert betti[0] == 0
    # At 0.25: [0.2, 0.8] and [0.1, 0.3] are alive (count = 2)
    assert betti[1] == 2
    # At 0.5: [0.2, 0.8] and [0.4, 0.6] are alive (count = 2)
    assert betti[2] == 2
    # At 0.7: only [0.2, 0.8] alive (count = 1)
    assert betti[3] == 1
    # At 0.9: 0 features alive
    assert betti[4] == 0
