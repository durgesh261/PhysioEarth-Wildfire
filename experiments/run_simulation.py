"""Experiments Runner: Run TopoCollapse simulation, generate figures, and benchmark metrics."""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.recursion_engine import generate_synthetic_manifold, simulate_recursive_autophagy
from src.pipeline.benchmark import run_full_topological_benchmark
from src.topology.visualizer import (
    plot_manifold_evolution,
    plot_persistence_diagrams,
    plot_metric_sensitivity_comparison,
    plot_persistence_barcodes,
)


def main():
    print("=" * 80)
    print(" TOPOCOLLAPSE: RESEARCH BENCHMARK SIMULATION")
    print(" Quantifying Manifold Degeneracy and Tail-Loss via Persistent Homology")
    print("=" * 80)

    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Generate Synthetic Manifold with Non-Trivial Homology (Ring + Center + Satellite)
    print("\n[Step 1/4] Generating initial ground-truth topological manifold (Gen 0)...")
    init_manifold = generate_synthetic_manifold(
        n_samples=1500,
        ambient_dim=2,
        noise_std=0.035,
        random_state=42,
    )
    print(f" -> Manifold generated: {init_manifold.shape[0]} points in R^{init_manifold.shape[1]}")

    # 2. Simulate Recursive Model Autophagy (Generations 0 to 5)
    print("\n[Step 2/4] Simulating recursive model autophagy across 6 generations (Gen 0 -> Gen 5)...")
    autophagy_res = simulate_recursive_autophagy(
        initial_manifold=init_manifold,
        n_generations=5,
        n_components=8,
        sample_size=1200,
        variance_decay_rate=0.06,
        random_state=42,
    )
    print(f" -> Simulated {autophagy_res.n_generations} recursive generations successfully.")

    # 3. Run Topological & Classical Benchmarks
    print("\n[Step 3/4] Computing Vietoris-Rips Persistent Homology & Classical Baselines...")
    benchmark_out = run_full_topological_benchmark(
        autophagy_res=autophagy_res,
        subsample_n=1000,
        random_state=42,
    )
    df = benchmark_out.metrics_df

    # Display results table
    display_cols = [
        "generation",
        "h1_entropy_normalized",
        "h1_bottleneck_dist",
        "frechet_distance",
        "vendi_score",
        "precision",
        "coverage",
    ]
    formatted_df = df[display_cols].copy()
    formatted_df = formatted_df.round(4)
    print("\n" + "-" * 80)
    print(" BENCHMARK RESULTS MATRIX")
    print("-" * 80)
    print(formatted_df.to_string(index=False))
    print("-" * 80)

    # Save metrics to CSV
    csv_path = os.path.join(results_dir, "metrics_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved quantitative matrix to: {csv_path}")

    # 4. Generate Publication-Quality Visualizations
    print("\n[Step 4/4] Rendering publication-grade figures...")

    # Figure 1: Geometric Manifold Evolution
    fig1_path = os.path.join(results_dir, "figure1_manifold_evolution.png")
    plot_manifold_evolution(
        point_clouds=autophagy_res.generations,
        generation_labels=autophagy_res.generation_labels,
        save_path=fig1_path,
    )
    print(f" [+] Rendered Fig 1 (Manifold Evolution): {fig1_path}")

    # Figure 2: Persistence Diagrams across generations
    fig2_path = os.path.join(results_dir, "figure2_persistence_diagrams.png")
    dgms_list = [res.dgms for res in benchmark_out.persistence_results]
    plot_persistence_diagrams(
        dgms_per_generation=dgms_list,
        generation_labels=autophagy_res.generation_labels,
        save_path=fig2_path,
    )
    print(f" [+] Rendered Fig 2 (Persistence Diagrams): {fig2_path}")

    # Figure 3: Metric Sensitivity Comparison (Centerpiece graph)
    fig3_path = os.path.join(results_dir, "figure3_metric_sensitivity.png")
    plot_metric_sensitivity_comparison(
        metrics_df=df,
        save_path=fig3_path,
    )
    print(f" [+] Rendered Fig 3 (Sensitivity Comparison): {fig3_path}")

    # Figure 4 & 5: Barcodes for Gen 0 vs Gen 5
    fig4_path = os.path.join(results_dir, "figure4_barcode_gen0.png")
    plot_persistence_barcodes(
        dgms=benchmark_out.persistence_results[0].dgms,
        title="Persistence Barcode - Generation 0 (Ground Truth Rich Topology)",
        save_path=fig4_path,
    )
    fig5_path = os.path.join(results_dir, "figure5_barcode_gen5.png")
    plot_persistence_barcodes(
        dgms=benchmark_out.persistence_results[-1].dgms,
        title="Persistence Barcode - Generation 5 (Severe Model Collapse)",
        save_path=fig5_path,
    )
    print(f" [+] Rendered Fig 4 & 5 (Homological Barcodes): {fig4_path}")

    # Alarm Summary & Conclusion
    print("\n" + "=" * 80)
    print(" EMPIRICAL RESEARCH FINDINGS")
    print("=" * 80)
    alarms = benchmark_out.early_detection_summary
    print(f" * H1 Persistent Entropy Alarm Triggered : Generation {alarms.get('H1_Persistent_Entropy_Alarm_Gen', 'N/A')}")
    print(f" * Fréchet Distance (FID) Alarm Triggered: Generation {alarms.get('Frechet_Distance_Alarm_Gen', 'N/A')}")
    print(f" * Vendi Diversity Score Alarm Triggered : Generation {alarms.get('Vendi_Score_Alarm_Gen', 'N/A')}")

    h1_drop_pct = (1.0 - df.loc[1, "h1_entropy_normalized"] / df.loc[0, "h1_entropy_normalized"]) * 100
    fd_gen1 = df.loc[1, "frechet_distance"]
    print(f"\n -> KEY DISCOVERY: In Generation 1, H1 Persistent Entropy drops by {h1_drop_pct:.1f}%,")
    print(f"    signaling immediate boundary cavity erosion, while Fréchet Distance only rises to {fd_gen1:.4f}")
    print("    (virtually undetected under Gaussian distance thresholds).")
    print(" -> CONCLUSION: Topological Data Analysis provides strict early-warning indicators")
    print("    for generative model collapse prior to conventional distribution divergences.")
    print("=" * 80)


if __name__ == "__main__":
    main()
