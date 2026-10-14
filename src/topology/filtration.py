"""Vietoris-Rips Simplicial Filtration & Persistent Homology Computation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import ripser


@dataclass
class PersistenceResult:
    """Dataclass holding persistence diagrams and topological lifetime summaries."""
    dgms: List[np.ndarray]  # dgms[0] = H0 intervals [birth, death], dgms[1] = H1 intervals
    h0_intervals: np.ndarray
    h1_intervals: np.ndarray
    h0_lifetimes: np.ndarray
    h1_lifetimes: np.ndarray
    num_points: int
    dim: int
    max_edge_length: float


def compute_vietoris_rips(
    point_cloud: np.ndarray,
    max_dim: int = 1,
    subsample_n: Optional[int] = 1000,
    thresh: float = np.inf,
    random_state: int = 42,
) -> PersistenceResult:
    """Computes Vietoris-Rips persistent homology on a point cloud.

    Args:
        point_cloud: Array of shape (N, D), representing coordinates in embedding/data space.
        max_dim: Maximum homology dimension to compute (0: components, 1: loops/cavities).
        subsample_n: If specified and N > subsample_n, performs deterministic uniform subsampling
                     to ensure computational tractability.
        thresh: Maximum filtration edge length threshold.
        random_state: Random seed for reproducible subsampling.

    Returns:
        PersistenceResult containing parsed diagrams and lifetimes.
    """
    pts = np.asarray(point_cloud, dtype=np.float64)
    n_points, dim = pts.shape

    if subsample_n is not None and n_points > subsample_n:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(n_points, size=subsample_n, replace=False)
        pts = pts[sample_indices]
        n_points = subsample_n

    # Run Ripser C++ backend
    result = ripser.ripser(pts, maxdim=max_dim, thresh=thresh)
    dgms = result["dgms"]

    h0 = dgms[0]
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    # Filter out infinite death for lifespan calculations (H0 usually has one infinite feature)
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h1_finite = h1[np.isfinite(h1[:, 1])]

    h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([])
    h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0] if len(h1_finite) > 0 else np.array([])

    return PersistenceResult(
        dgms=dgms,
        h0_intervals=h0,
        h1_intervals=h1,
        h0_lifetimes=h0_lifetimes,
        h1_lifetimes=h1_lifetimes,
        num_points=n_points,
        dim=dim,
        max_edge_length=float(np.max(h0_finite[:, 1])) if len(h0_finite) > 0 else 1.0,
    )
