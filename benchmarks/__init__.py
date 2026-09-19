"""
Benchmarks package for KV cache compression evaluation.
"""
from .dataset_generator import build_synthetic_prompt, generate_benchmark_dataset

__all__ = ["build_synthetic_prompt", "generate_benchmark_dataset"]
