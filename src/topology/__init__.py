"""Topology module for computing simplicial filtrations, persistent homology, and topological distances."""

from .filtration import compute_vietoris_rips, PersistenceResult
from .metrics import (
    compute_persistent_entropy,
    compute_bottleneck_distance,
    compute_wasserstein_distance,
    compute_total_persistence,
    compute_betti_numbers,
    summarize_topological_profile,
)
from .visualizer import (
    plot_persistence_diagrams,
    plot_persistence_barcodes,
    plot_manifold_evolution,
    plot_metric_sensitivity_comparison,
)

__all__ = [
    "compute_vietoris_rips",
    "PersistenceResult",
    "compute_persistent_entropy",
    "compute_bottleneck_distance",
    "compute_wasserstein_distance",
    "compute_total_persistence",
    "compute_betti_numbers",
    "summarize_topological_profile",
    "plot_persistence_diagrams",
    "plot_persistence_barcodes",
    "plot_manifold_evolution",
    "plot_metric_sensitivity_comparison",
]
