"""Pipeline module for running comparative benchmarks between TDA and classical metrics."""

from .benchmark import run_full_topological_benchmark, BenchmarkOutput

__all__ = ["run_full_topological_benchmark", "BenchmarkOutput"]
