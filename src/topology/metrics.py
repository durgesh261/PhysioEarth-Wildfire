"""Topological Metrics: Persistent Entropy, Bottleneck/Wasserstein Distances, and Betti Curves."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import persim
from .filtration import PersistenceResult


def compute_persistent_entropy(
    lifetimes: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-12,
) -> float:
    """Calculates Persistent Entropy (Atienza et al., Rucco et al.) of persistence lifetimes.

    Persistent entropy measures the diversity/uniformity of topological feature persistence.
    When a generative model collapses into a single dense mode or loses long-tail cavities,
    feature lifetimes degenerate, causing a measurable drop in persistent entropy.

    E = - sum_i p_i * log2(p_i),  where p_i = l_i / sum_j l_j

    Args:
        lifetimes: 1D array of positive feature lifespans (death - birth).
        normalize: If True, divides by log2(N_features) to yield [0, 1] range.
        eps: Small stability constant to prevent log(0).

    Returns:
        Persistent entropy scalar (float). Returns 0.0 if empty or zero persistence.
    """
    valid = lifetimes[lifetimes > eps]
    if len(valid) <= 1:
        return 0.0

    total_life = np.sum(valid)
    if total_life <= eps:
        return 0.0

    p = valid / total_life
    entropy = -float(np.sum(p * np.log2(p + eps)))

    if normalize and len(valid) > 1:
        max_entropy = np.log2(len(valid))
        return float(entropy / max_entropy) if max_entropy > eps else 0.0

    return float(entropy)


def compute_bottleneck_distance(
    dgm_ref: np.ndarray,
    dgm_target: np.ndarray,
) -> float:
    """Computes the Bottleneck distance W_inf between two persistence diagrams.

    Args:
        dgm_ref: Reference diagram of shape (M, 2).
        dgm_target: Target diagram of shape (N, 2).

    Returns:
        Bottleneck distance float.
    """
    # Filter non-finite points to avoid persim crashes
    ref_clean = dgm_ref[np.isfinite(dgm_ref).all(axis=1)] if len(dgm_ref) > 0 else np.empty((0, 2))
    tgt_clean = dgm_target[np.isfinite(dgm_target).all(axis=1)] if len(dgm_target) > 0 else np.empty((0, 2))

    if len(ref_clean) == 0 and len(tgt_clean) == 0:
        return 0.0
    if len(ref_clean) == 0:
        # Distance to diagonal
        return float(np.max((tgt_clean[:, 1] - tgt_clean[:, 0]) / 2.0))
    if len(tgt_clean) == 0:
        return float(np.max((ref_clean[:, 1] - ref_clean[:, 0]) / 2.0))

    try:
        return float(persim.bottleneck(ref_clean, tgt_clean))
    except Exception:
        # Fallback approximation: max difference of sorted lifetimes
        l1 = np.sort(ref_clean[:, 1] - ref_clean[:, 0])[::-1]
        l2 = np.sort(tgt_clean[:, 1] - tgt_clean[:, 0])[::-1]
        min_len = min(len(l1), len(l2))
        return float(np.max(np.abs(l1[:min_len] - l2[:min_len]))) if min_len > 0 else 0.0


def compute_wasserstein_distance(
    dgm_ref: np.ndarray,
    dgm_target: np.ndarray,
    p: int = 2,
) -> float:
    """Computes the p-Wasserstein distance W_p between two persistence diagrams.

    Args:
        dgm_ref: Reference diagram of shape (M, 2).
        dgm_target: Target diagram of shape (N, 2).
        p: Degree of Wasserstein metric (default: 2 for Euclidean matching).

    Returns:
        Wasserstein distance float.
    """
    ref_clean = dgm_ref[np.isfinite(dgm_ref).all(axis=1)] if len(dgm_ref) > 0 else np.empty((0, 2))
    tgt_clean = dgm_target[np.isfinite(dgm_target).all(axis=1)] if len(dgm_target) > 0 else np.empty((0, 2))

    if len(ref_clean) == 0 and len(tgt_clean) == 0:
        return 0.0
    if len(ref_clean) == 0 or len(tgt_clean) == 0:
        active = tgt_clean if len(ref_clean) == 0 else ref_clean
        return float(np.sum(((active[:, 1] - active[:, 0]) / 2.0) ** p) ** (1.0 / p))

    try:
        return float(persim.wasserstein(ref_clean, tgt_clean, d=p))
    except Exception:
        return compute_bottleneck_distance(ref_clean, tgt_clean)


def compute_total_persistence(lifetimes: np.ndarray, power: float = 1.0) -> float:
    """Calculates sum of (lifetimes)^power."""
    if len(lifetimes) == 0:
        return 0.0
    return float(np.sum(lifetimes ** power))


def compute_betti_numbers(
    diagram: np.ndarray,
    epsilon_values: np.ndarray,
) -> np.ndarray:
    """Computes the Betti curve beta(epsilon) across filtration scale thresholds.

    beta(epsilon) = count of features born <= epsilon and died > epsilon.
    """
    clean_dgm = diagram[np.isfinite(diagram[:, 0])]
    betti_curve = np.zeros(len(epsilon_values), dtype=int)

    for idx, eps in enumerate(epsilon_values):
        alive = (clean_dgm[:, 0] <= eps) & (clean_dgm[:, 1] > eps)
        betti_curve[idx] = int(np.sum(alive))

    return betti_curve


def summarize_topological_profile(
    res: PersistenceResult,
    ref_res: Optional[PersistenceResult] = None,
) -> Dict[str, float]:
    """Generates a comprehensive dictionary of scalar topological diagnostics.

    Args:
        res: Target PersistenceResult to analyze.
        ref_res: Optional reference PersistenceResult (e.g. Gen_0 original data).

    Returns:
        Dictionary mapping metric names to values.
    """
    profile = {
        "h0_entropy_normalized": compute_persistent_entropy(res.h0_lifetimes, normalize=True),
        "h1_entropy_normalized": compute_persistent_entropy(res.h1_lifetimes, normalize=True),
        "h0_total_persistence": compute_total_persistence(res.h0_lifetimes),
        "h1_total_persistence": compute_total_persistence(res.h1_lifetimes),
        "h1_feature_count": float(len(res.h1_lifetimes)),
        "h1_max_lifetime": float(np.max(res.h1_lifetimes)) if len(res.h1_lifetimes) > 0 else 0.0,
    }

    if ref_res is not None:
        profile["h0_bottleneck_dist"] = compute_bottleneck_distance(ref_res.h0_intervals, res.h0_intervals)
        profile["h1_bottleneck_dist"] = compute_bottleneck_distance(ref_res.h1_intervals, res.h1_intervals)
        profile["h1_wasserstein_dist"] = compute_wasserstein_distance(ref_res.h1_intervals, res.h1_intervals, p=2)

    return profile
