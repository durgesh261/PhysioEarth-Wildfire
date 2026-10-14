"""Models and baseline metrics module for generative simulation and classical evaluation."""

from .baseline_metrics import (
    compute_frechet_distance,
    compute_vendi_score,
    compute_total_variance,
    compute_coverage_precision,
    summarize_baseline_metrics,
)
from .recursion_engine import (
    generate_synthetic_manifold,
    simulate_recursive_autophagy,
    RecursiveAutophagyResult,
)

__all__ = [
    "compute_frechet_distance",
    "compute_vendi_score",
    "compute_total_variance",
    "compute_coverage_precision",
    "summarize_baseline_metrics",
    "generate_synthetic_manifold",
    "simulate_recursive_autophagy",
    "RecursiveAutophagyResult",
]
