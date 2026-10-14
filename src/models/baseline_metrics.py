"""Classical Statistical and Diversity Baselines for Generative Distribution Evaluation."""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel


def compute_frechet_distance(
    pts_ref: np.ndarray,
    pts_eval: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Computes the Fréchet Distance (FID equivalent in latent feature space).

    FD = ||mu_ref - mu_eval||^2 + Tr(Sigma_ref + Sigma_eval - 2 * (Sigma_ref * Sigma_eval)^(1/2))

    Args:
        pts_ref: Reference data array of shape (N, D).
        pts_eval: Evaluation data array of shape (M, D).
        eps: Small diagonal regularization for positive semi-definiteness.

    Returns:
        Fréchet distance float (lower is closer to reference).
    """
    mu1 = np.mean(pts_ref, axis=0)
    mu2 = np.mean(pts_eval, axis=0)

    sigma1 = np.cov(pts_ref, rowvar=False) + eps * np.eye(pts_ref.shape[1])
    sigma2 = np.cov(pts_eval, rowvar=False) + eps * np.eye(pts_eval.shape[1])

    diff = mu1 - mu2
    mean_diff_sq = float(np.dot(diff, diff))

    # Product of covariance matrices
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # Numerical imaginary artifact cleanup
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = float(np.trace(covmean))
    tr_sum = float(np.trace(sigma1) + np.trace(sigma2))

    fd = mean_diff_sq + tr_sum - 2.0 * tr_covmean
    return max(0.0, float(fd))


def compute_vendi_score(
    pts: np.ndarray,
    gamma: Optional[float] = None,
    eps: float = 1e-12,
) -> float:
    """Calculates the Vendi Score (Friedman & Dieng, 2023) measuring point diversity.

    Vendi Score = exp( - sum_i lambda_i * log(lambda_i) ),
    where lambda_i are the eigenvalues of the normalized similarity matrix K / N.

    Returns:
        Effective number of distinct modes/clusters (scalar >= 1.0).
    """
    n = len(pts)
    if n <= 1:
        return 1.0

    k_mat = rbf_kernel(pts, pts, gamma=gamma)
    k_normalized = k_mat / n

    eigenvals = linalg.eigvalsh(k_normalized)
    eigenvals = eigenvals[eigenvals > eps]

    # Normalize to probability simplex
    eigenvals = eigenvals / np.sum(eigenvals)

    entropy = -np.sum(eigenvals * np.log(eigenvals + eps))
    return float(np.exp(entropy))


def compute_total_variance(pts: np.ndarray) -> float:
    """Calculates total variance Tr(Cov(X))."""
    cov = np.cov(pts, rowvar=False)
    return float(np.trace(cov)) if cov.ndim == 2 else float(cov)


def compute_coverage_precision(
    pts_ref: np.ndarray,
    pts_eval: np.ndarray,
    k: int = 5,
) -> Tuple[float, float]:
    """Computes k-NN based Precision and Coverage metrics (Sajjadi et al., 2018).

    - Precision: Fraction of generated points falling within the reference manifold.
    - Coverage (Recall): Fraction of reference points covered by the generated manifold.
    """
    from sklearn.neighbors import NearestNeighbors

    # Fit kNN on reference
    nn_ref = NearestNeighbors(n_neighbors=k + 1).fit(pts_ref)
    ref_distances, _ = nn_ref.kneighbors(pts_ref)
    ref_radii = ref_distances[:, -1]  # distance to k-th neighbor

    # Check which eval points fall within at least one reference ball
    eval_to_ref_dist, _ = nn_ref.kneighbors(pts_eval, n_neighbors=1)
    # A generated point is in-distribution if its distance to nearest ref point <= ref_radii of that point
    nn_nearest_idx = nn_ref.kneighbors(pts_eval, n_neighbors=1, return_distance=False).flatten()
    precision = float(np.mean(eval_to_ref_dist.flatten() <= ref_radii[nn_nearest_idx]))

    # Fit kNN on eval to compute coverage of reference
    nn_eval = NearestNeighbors(n_neighbors=k + 1).fit(pts_eval)
    eval_distances, _ = nn_eval.kneighbors(pts_eval)
    eval_radii = eval_distances[:, -1]

    ref_to_eval_dist, _ = nn_eval.kneighbors(pts_ref, n_neighbors=1)
    eval_nearest_idx = nn_eval.kneighbors(pts_ref, n_neighbors=1, return_distance=False).flatten()
    coverage = float(np.mean(ref_to_eval_dist.flatten() <= eval_radii[eval_nearest_idx]))

    return precision, coverage


def summarize_baseline_metrics(
    pts: np.ndarray,
    pts_ref: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Summarizes classical baseline statistical metrics."""
    metrics = {
        "vendi_score": compute_vendi_score(pts),
        "total_variance": compute_total_variance(pts),
    }

    if pts_ref is not None:
        metrics["frechet_distance"] = compute_frechet_distance(pts_ref, pts)
        prec, cov = compute_coverage_precision(pts_ref, pts, k=5)
        metrics["precision"] = prec
        metrics["coverage"] = cov

    return metrics
