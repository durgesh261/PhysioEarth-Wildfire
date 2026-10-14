"""Recursive Autophagy Engine: Simulating Synthetic Data Collapse and Manifold Degeneracy."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class RecursiveAutophagyResult:
    """Stores the trajectory of point clouds across recursive model generations."""
    generations: List[np.ndarray]  # generations[g] is point cloud at generation g
    generation_labels: List[str]
    n_generations: int
    ambient_dim: int
    n_samples: int


def generate_synthetic_manifold(
    n_samples: int = 1500,
    ambient_dim: int = 2,
    noise_std: float = 0.04,
    random_state: int = 42,
) -> np.ndarray:
    """Generates a ground-truth topological manifold containing distinct clusters and loops (H1).

    Constructs a manifold consisting of:
    1. An outer circular ring (generating persistent H1 homology loops).
    2. An inner satellite cluster representing localized sub-populations.
    3. An auxiliary disconnected cluster (rich H0 components).

    Optionally embeds the 2D manifold into a higher-dimensional ambient space R^ambient_dim
    via an orthogonal random projection matrix.
    """
    rng = np.random.default_rng(random_state)

    # 1. Outer Ring (Circle with non-trivial 1D homology loop)
    n_ring = int(n_samples * 0.55)
    theta = rng.uniform(0, 2 * np.pi, size=n_ring)
    radius = 1.0 + rng.normal(0, noise_std, size=n_ring)
    ring_x = radius * np.cos(theta)
    ring_y = radius * np.sin(theta)
    ring_pts = np.column_stack([ring_x, ring_y])

    # 2. Central Cluster (inside the loop)
    n_center = int(n_samples * 0.20)
    center_pts = rng.normal(loc=[0.0, 0.0], scale=noise_std * 2.5, size=(n_center, 2))

    # 3. Satellite Cluster (disconnected component, distance 2.0 away)
    n_sat = n_samples - n_ring - n_center
    sat_pts = rng.normal(loc=[2.2, 0.0], scale=noise_std * 2.0, size=(n_sat, 2))

    base_2d = np.vstack([ring_pts, center_pts, sat_pts])

    if ambient_dim <= 2:
        return base_2d

    # Orthogonal high-dimensional embedding
    q_matrix, _ = np.linalg.qr(rng.standard_normal((ambient_dim, 2)))
    embedded = np.dot(base_2d, q_matrix.T)
    # Add mild ambient isotropic noise
    ambient_noise = rng.normal(0, noise_std * 0.2, size=(n_samples, ambient_dim))
    return embedded + ambient_noise


def simulate_recursive_autophagy(
    initial_manifold: np.ndarray,
    n_generations: int = 5,
    n_components: int = 8,
    sample_size: int = 1200,
    variance_decay_rate: float = 0.05,
    random_state: int = 42,
) -> RecursiveAutophagyResult:
    """Simulates recursive model training on synthetic data (Model Autophagy).

    At each generation g in {1 ... n_generations}:
    1. A generative model (GMM) is trained on data from Generation g-1.
    2. Finite-sample estimators contract covariance matrices (variance underestimation).
    3. Low-probability topological bridges and loop boundaries collapse into local centroids.
    4. New synthetic sample is drawn to train Generation g+1.

    Args:
        initial_manifold: Ground truth Gen_0 dataset of shape (N, D).
        n_generations: Number of recursive generations to simulate.
        n_components: Number of Gaussian mixture components in the generative learner.
        sample_size: Number of synthetic points sampled per generation.
        variance_decay_rate: Factor modeling recursive variance shrinkage (Shumailov et al.).
        random_state: Random seed for reproducibility.

    Returns:
        RecursiveAutophagyResult containing the point clouds at all generations.
    """
    rng = np.random.default_rng(random_state)
    dim = initial_manifold.shape[1]

    trajectory = [initial_manifold]
    labels = ["Gen 0 (Ground Truth)"]

    current_data = initial_manifold

    for g in range(1, n_generations + 1):
        g_seed = random_state + g * 101

        # Fit generative distribution on previous generation
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            max_iter=150,
            random_state=g_seed,
            reg_covar=1e-4,
        )
        gmm.fit(current_data)

        # Recursive degradation: Covariance contraction on minority components
        # (models progressively abandon low-density structural tails)
        shrunk_covariances = gmm.covariances_ * ((1.0 - variance_decay_rate) ** g)
        gmm.covariances_ = shrunk_covariances

        # Re-sample synthetic generation
        synthetic_samples, _ = gmm.sample(n_samples=sample_size)

        trajectory.append(synthetic_samples)
        labels.append(f"Gen {g} (Recursive {g})")
        current_data = synthetic_samples

    return RecursiveAutophagyResult(
        generations=trajectory,
        generation_labels=labels,
        n_generations=n_generations,
        ambient_dim=dim,
        n_samples=sample_size,
    )
