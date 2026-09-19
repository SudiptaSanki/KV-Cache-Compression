"""
Single-Command End-to-End Benchmark Reproducer.

Executes:
1. Synthetic long-context prompt dataset generation (if not already generated).
2. Multi-policy benchmark execution across budgets (100%, 50%, 25%, 12.5%) with 3 repeats.
3. Metric aggregation, relative quality scoring, and plot generation.
4. Summary verification.
"""

import os
import sys
import time
import argparse

from kv_cache.utils import get_device, get_model_dtype
from benchmarks.dataset_generator import generate_benchmark_dataset
from benchmarks.run_benchmark import run_experiments
from benchmarks.evaluate import evaluate_and_report

def main():
    parser = argparse.ArgumentParser(description="End-to-End KV Cache Compression Study")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--prompts-file", type=str, default="benchmarks/prompts.jsonl")
    parser.add_argument("--results-csv", type=str, default="results/results.csv")
    parser.add_argument("--plots-dir", type=str, default="results/plots")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--limit-prompts", type=int, default=12) # 12 prompts * (1 + 9) * 3 = 360 runs
    parser.add_argument("--regenerate-dataset", action="store_true")
    args = parser.parse_args()

    print("="*80)
    print(" INSTRUCTION-PRESERVING KV-CACHE COMPRESSION BENCHMARK REPRODUCER")
    print("="*80)

    device = get_device()
    dtype = get_model_dtype(device)
    print(f"[*] Detected Compute Device: {device} (dtype: {dtype})")

    # Step 1: Ensure dataset exists
    if args.regenerate_dataset or not os.path.exists(args.prompts_file):
        print("\n[*] Step 1: Generating synthetic long-context prompt dataset...")
        os.makedirs(os.path.dirname(args.prompts_file), exist_ok=True)
        generate_benchmark_dataset(
            output_path=args.prompts_file,
            tokenizer_name=args.model,
            target_lengths=[512, 1024, 2048],
            variations_per_length=8,
            seed=42
        )
    else:
        print(f"\n[*] Step 1: Using existing prompts file: {args.prompts_file}")

    # Step 2: Run benchmark
    print(f"\n[*] Step 2: Running benchmark across policies and cache budgets (Repeats: {args.repeats})...")
    t0 = time.time()
    df_results = run_experiments(
        prompts_file=args.prompts_file,
        model_name=args.model,
        budgets=[1.0, 0.50, 0.25, 0.125],
        repeats=args.repeats,
        max_new_tokens=args.max_new_tokens,
        output_csv=args.results_csv,
        limit_prompts=args.limit_prompts
    )
    benchmark_elapsed = time.time() - t0
    print(f"[*] Benchmark completed in {benchmark_elapsed:.2f} seconds.")

    # Step 3: Evaluate and generate plots
    print("\n[*] Step 3: Computing metrics and generating publication plots...")
    summary = evaluate_and_report(
        csv_path=args.results_csv,
        plots_dir=args.plots_dir
    )

    print("\n" + "="*80)
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f" Results CSV: {os.path.abspath(args.results_csv)}")
    print(f" Plots Dir:   {os.path.abspath(args.plots_dir)}")
    print("="*80)

if __name__ == "__main__":
    main()
