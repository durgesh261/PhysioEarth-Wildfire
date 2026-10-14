"""Publication-Quality Visualizers for Topological Data Analysis and Manifold Evolution."""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import persim


def set_academic_style():
    """Sets clean, publication-grade aesthetics for matplotlib figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_persistence_diagrams(
    dgms_per_generation: List[List[np.ndarray]],
    generation_labels: List[str],
    save_path: Optional[str] = None,
):
    """Plots a multi-panel grid of Persistence Diagrams across generations."""
    set_academic_style()
    n_gens = len(dgms_per_generation)
    fig, axes = plt.subplots(1, n_gens, figsize=(4.2 * n_gens, 4.0), sharex=True, sharey=True)

    if n_gens == 1:
        axes = [axes]

    for idx, (dgms, label) in enumerate(zip(dgms_per_generation, generation_labels)):
        ax = axes[idx]
        h0 = dgms[0]
        h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

        # Filter finite for clean plotting
        h0_finite = h0[np.isfinite(h0).all(axis=1)] if len(h0) > 0 else np.empty((0, 2))
        h1_finite = h1[np.isfinite(h1).all(axis=1)] if len(h1) > 0 else np.empty((0, 2))

        all_finite = np.vstack([h0_finite, h1_finite]) if len(h1_finite) > 0 else h0_finite
        max_val = float(np.max(all_finite[:, 1])) * 1.05 if len(all_finite) > 0 else 1.0

        # Plot diagonal
        ax.plot([0, max_val], [0, max_val], color="gray", linestyle="--", alpha=0.7, label="Diagonal")

        if len(h0_finite) > 0:
            ax.scatter(h0_finite[:, 0], h0_finite[:, 1], s=20, color="#1f77b4", alpha=0.6, label="H0 (Components)")
        if len(h1_finite) > 0:
            ax.scatter(h1_finite[:, 0], h1_finite[:, 1], s=35, color="#d62728", marker="^", alpha=0.85, label="H1 (Loops)")

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Birth")
        if idx == 0:
            ax.set_ylabel("Death")
            ax.legend(loc="lower right")

    plt.suptitle("Evolution of Persistence Diagrams During Model Autophagy", y=1.02, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_persistence_barcodes(
    dgms: List[np.ndarray],
    title: str = "Persistence Barcode",
    save_path: Optional[str] = None,
):
    """Plots homological barcodes showing lifetime intervals for H0 and H1."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    h0 = dgms[0]
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    y_idx = 0
    # Plot H0 bars
    for b, d in h0:
        death = d if np.isfinite(d) else 2.0
        ax.hlines(y_idx, b, death, color="#1f77b4", linewidth=1.2, alpha=0.7)
        y_idx += 1

    h0_count = y_idx
    # Plot H1 bars
    for b, d in h1:
        if np.isfinite(d):
            ax.hlines(y_idx, b, d, color="#d62728", linewidth=2.0, alpha=0.9)
            y_idx += 1

    ax.set_xlabel("Filtration Scale (epsilon)")
    ax.set_ylabel("Topological Feature Index")
    ax.set_title(title, fontweight="bold")

    # Add custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="#1f77b4", lw=2),
        Line2D([0], [0], color="#d62728", lw=2),
    ]
    ax.legend(custom_lines, ["H0 (Connected components)", "H1 (Topological loops)"], loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_manifold_evolution(
    point_clouds: List[np.ndarray],
    generation_labels: List[str],
    save_path: Optional[str] = None,
):
    """Visualizes geometric degeneration and mode collapse in 2D space."""
    set_academic_style()
    n_gens = len(point_clouds)
    fig, axes = plt.subplots(1, n_gens, figsize=(4.0 * n_gens, 3.8), sharex=True, sharey=True)

    if n_gens == 1:
        axes = [axes]

    for idx, (pts, label) in enumerate(zip(point_clouds, generation_labels)):
        ax = axes[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.5, color="#2ca02c" if idx == 0 else "#9467bd")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Dimension 1")
        if idx == 0:
            ax.set_ylabel("Dimension 2")

    plt.suptitle("Geometric Manifold Degeneracy Across Recursive Generations", y=1.02, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_metric_sensitivity_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Key Research Figure: Compares normalized Persistent Entropy vs classical Fréchet Distance."""
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    gens = metrics_df["generation"]

    # Left Panel: Topological Metrics (Sensitivity to structure)
    if "h1_entropy_normalized" in metrics_df.columns:
        ax1.plot(gens, metrics_df["h1_entropy_normalized"], marker="o", color="#d62728", lw=2.5, label="H1 Persistent Entropy (Normalized)")
    if "h0_entropy_normalized" in metrics_df.columns:
        ax1.plot(gens, metrics_df["h0_entropy_normalized"], marker="s", color="#1f77b4", lw=2.0, linestyle="--", label="H0 Persistent Entropy")
    if "vendi_score_normalized" in metrics_df.columns:
        ax1.plot(gens, metrics_df["vendi_score_normalized"], marker="^", color="#ff7f0e", lw=1.8, linestyle=":", label="Vendi Diversity Score (Normalized)")

    ax1.set_xlabel("Recursive Generation (g)")
    ax1.set_ylabel("Structural Score (Higher = Richer Topology)")
    ax1.set_title("Topological Metrics: Early Collapse Detection", fontweight="bold")
    ax1.legend(loc="lower left")

    # Right Panel: Classical Baselines (FID-equivalent & Variance)
    if "frechet_distance" in metrics_df.columns:
        ax2.plot(gens, metrics_df["frechet_distance"], marker="D", color="#9467bd", lw=2.5, label="Fréchet Distance (FID equivalent)")
    if "h1_bottleneck_dist" in metrics_df.columns:
        ax2.plot(gens, metrics_df["h1_bottleneck_dist"], marker="x", color="#8c564b", lw=2.0, linestyle="--", label="H1 Bottleneck Distance W_inf")

    ax2.set_xlabel("Recursive Generation (g)")
    ax2.set_ylabel("Distance to Ground Truth Gen_0")
    ax2.set_title("Divergence from True Manifold", fontweight="bold")
    ax2.legend(loc="upper left")

    plt.suptitle("Quantitative Sensitivity Comparison: TDA vs Classical Statistical Baselines", y=1.02, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
