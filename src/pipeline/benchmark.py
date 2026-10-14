"""End-to-End Comparative Benchmark: TDA vs. Classical Generative Metrics."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..models.baseline_metrics import summarize_baseline_metrics
from ..models.recursion_engine import RecursiveAutophagyResult, simulate_recursive_autophagy, generate_synthetic_manifold
from ..topology.filtration import compute_vietoris_rips, PersistenceResult
from ..topology.metrics import summarize_topological_profile


@dataclass
class BenchmarkOutput:
    """Stores full benchmark outputs, metric trajectories, and persistence results."""
    metrics_df: pd.DataFrame
    persistence_results: List[PersistenceResult]
    generation_labels: List[str]
    early_detection_summary: Dict[str, int]


def run_full_topological_benchmark(
    autophagy_res: Optional[RecursiveAutophagyResult] = None,
    n_generations: int = 5,
    subsample_n: int = 1000,
    random_state: int = 42,
) -> BenchmarkOutput:
    """Executes full comparative benchmarking across all generations.

    1. Computes Vietoris-Rips persistent homology (H0, H1) for each generation.
    2. Computes Persistent Entropy, Bottleneck/Wasserstein distances to Gen_0.
    3. Computes Fréchet Distance, Vendi Score, Total Variance, Coverage.
    4. Compares sensitivity and identifies earliest detection generation.
    """
    if autophagy_res is None:
        init_manifold = generate_synthetic_manifold(
            n_samples=1500,
            ambient_dim=2,
            random_state=random_state,
        )
        autophagy_res = simulate_recursive_autophagy(
            initial_manifold=init_manifold,
            n_generations=n_generations,
            random_state=random_state,
        )

    generations = autophagy_res.generations
    labels = autophagy_res.generation_labels

    # 1. Compute Persistence for all generations
    persistence_results: List[PersistenceResult] = []
    for g_idx, pts in enumerate(generations):
        res = compute_vietoris_rips(
            point_cloud=pts,
            max_dim=1,
            subsample_n=subsample_n,
            random_state=random_state + g_idx,
        )
        persistence_results.append(res)

    ref_persistence = persistence_results[0]
    ref_points = generations[0]

    # 2. Extract metrics
    records = []
    for g_idx, (pts, res, label) in enumerate(zip(generations, persistence_results, labels)):
        row = {
            "generation": g_idx,
            "label": label,
            "n_points": len(pts),
        }

        # Topological diagnostics
        topo_dict = summarize_topological_profile(
            res=res,
            ref_res=ref_persistence if g_idx > 0 else None,
        )
        row.update(topo_dict)

        # Classical baseline diagnostics
        base_dict = summarize_baseline_metrics(
            pts=pts,
            pts_ref=ref_points if g_idx > 0 else None,
        )
        row.update(base_dict)

        records.append(row)

    df = pd.DataFrame(records)

    # Normalize Vendi score relative to Gen 0 for direct sensitivity plotting
    gen0_vendi = df.loc[df["generation"] == 0, "vendi_score"].values[0]
    df["vendi_score_normalized"] = df["vendi_score"] / gen0_vendi

    # 3. Compute Earliest Collapse Alarm Generation (when metric shifts by >= 20% from baseline)
    alarm_summary = {}

    # Topological H1 Persistent Entropy drop
    h1_ent_init = df.loc[0, "h1_entropy_normalized"]
    drop_h1 = df[df["h1_entropy_normalized"] <= h1_ent_init * 0.80]
    alarm_summary["H1_Persistent_Entropy_Alarm_Gen"] = int(drop_h1["generation"].iloc[0]) if len(drop_h1) > 0 else -1

    # Fréchet Distance spike (FD > 0.15 threshold)
    fd_spike = df[df["frechet_distance"] >= 0.15]
    alarm_summary["Frechet_Distance_Alarm_Gen"] = int(fd_spike["generation"].iloc[0]) if len(fd_spike) > 0 else -1

    # Vendi Score drop
    vendi_drop = df[df["vendi_score_normalized"] <= 0.80]
    alarm_summary["Vendi_Score_Alarm_Gen"] = int(vendi_drop["generation"].iloc[0]) if len(vendi_drop) > 0 else -1

    return BenchmarkOutput(
        metrics_df=df,
        persistence_results=persistence_results,
        generation_labels=labels,
        early_detection_summary=alarm_summary,
    )
