"""Serializes simulation data into JSON for the interactive web dashboard."""

import json
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.recursion_engine import generate_synthetic_manifold, simulate_recursive_autophagy
from src.pipeline.benchmark import run_full_topological_benchmark


def export_dashboard_data():
    dashboard_dir = os.path.join(PROJECT_ROOT, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)

    print("[*] Generating simulation data for web dashboard...")
    init_manifold = generate_synthetic_manifold(n_samples=1200, ambient_dim=2, noise_std=0.035, random_state=42)
    autophagy_res = simulate_recursive_autophagy(
        initial_manifold=init_manifold,
        n_generations=5,
        n_components=8,
        sample_size=1000,
        variance_decay_rate=0.06,
        random_state=42,
    )
    benchmark_out = run_full_topological_benchmark(autophagy_res=autophagy_res, subsample_n=800, random_state=42)
    df = benchmark_out.metrics_df

    generations_data = []
    for g_idx, (pts, res, label) in enumerate(zip(autophagy_res.generations, benchmark_out.persistence_results, autophagy_res.generation_labels)):
        # Subsample points for smooth 60fps web rendering (500 pts per gen)
        sub_pts = pts[::2][:500].tolist()

        # Extract finite H0 and H1 points
        h0 = res.h0_intervals
        h0_finite = h0[np.isfinite(h0).all(axis=1)][:150].tolist() if len(h0) > 0 else []

        h1 = res.h1_intervals
        h1_finite = h1[np.isfinite(h1).all(axis=1)].tolist() if len(h1) > 0 else []

        row = df.loc[df["generation"] == g_idx].to_dict(orient="records")[0]

        # Clean NaN values for valid JSON
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None

        generations_data.append({
            "generation": g_idx,
            "label": label,
            "points": sub_pts,
            "h0_diagram": h0_finite,
            "h1_diagram": h1_finite,
            "metrics": row,
        })

    out_file = os.path.join(dashboard_dir, "data.json")
    with open(out_file, "w") as f:
        json.dump(generations_data, f, indent=2)

    print(f"[+] Successfully exported dashboard data to: {out_file}")


if __name__ == "__main__":
    export_dashboard_data()
