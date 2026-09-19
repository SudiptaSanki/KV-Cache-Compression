"""
Evaluation and Visualization Suite for KV Cache Compression Benchmark.

Computes core metrics:
1. Instruction retention rate
2. Relative quality score vs Full-KV baseline
3. Memory and cache footprint reduction
4. Decode latency and throughput (tokens/sec)

Generates 4 publication-ready figures:
- retention_vs_cache.png (The Key Plot)
- memory_vs_retention.png
- throughput_vs_retention.png
- latency_vs_context.png
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean scientific plotting style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

POLICY_PALETTE = {
    "Full_KV": "#2b5c8f",            # Deep Blue
    "Recent_Window": "#d95f02",       # Red-Orange
    "Attention_Guided": "#7570b3",   # Muted Purple
    "Protected_Attention": "#1b9e77" # Vibrant Green
}

POLICY_LABELS = {
    "Full_KV": "Full KV (Baseline A)",
    "Recent_Window": "Recent Window (Baseline B)",
    "Attention_Guided": "Attention-Guided (Method C)",
    "Protected_Attention": "Protected-Token Attention (Extension D)"
}

def load_and_preprocess_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["instruction_retained"] = df["instruction_retained"].astype(bool)
    df["cache_retention_pct"] = df["cache_retention_ratio"] * 100.0
    return df

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes aggregated summary metrics per policy and budget ratio.
    Also computes relative quality normalized to Full KV baseline.
    """
    summary = df.groupby(["policy", "budget_ratio"]).agg(
        n_trials=("instruction_retained", "count"),
        instruction_retention_mean=("instruction_retained", "mean"),
        instruction_retention_std=("instruction_retained", "std"),
        cache_retention_pct_mean=("cache_retention_pct", "mean"),
        retained_positions_mean=("retained_positions", "mean"),
        kv_cache_mb_mean=("kv_cache_bytes", lambda x: np.mean(x) / (1024 * 1024)),
        peak_memory_mb_mean=("peak_memory_mb", "mean"),
        prefill_latency_mean=("prefill_latency_sec", "mean"),
        decode_latency_mean=("decode_latency_sec", "mean"),
        tokens_per_second_mean=("tokens_per_second", "mean"),
    ).reset_index()

    # Determine baseline quality
    full_kv_row = summary[summary["policy"] == "Full_KV"]
    full_kv_score = full_kv_row["instruction_retention_mean"].values[0] if len(full_kv_row) > 0 else 1.0

    summary["relative_quality"] = summary["instruction_retention_mean"] / max(full_kv_score, 1e-6)
    return summary

def generate_plots(df: pd.DataFrame, summary: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. The Key Plot: Instruction Retention vs. KV Cache Retained (%)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for policy, label in POLICY_LABELS.items():
        pol_data = summary[summary["policy"] == policy].sort_values("cache_retention_pct_mean")
        if pol_data.empty:
            continue
        color = POLICY_PALETTE.get(policy, "#333333")
        marker = "o" if policy != "Full_KV" else "s"
        linestyle = "--" if "Window" in policy else "-"
        ax.plot(
            pol_data["cache_retention_pct_mean"],
            pol_data["instruction_retention_mean"] * 100.0,
            marker=marker,
            markersize=8,
            linewidth=2.2,
            label=label,
            color=color,
            linestyle=linestyle
        )

    ax.set_title("Key Finding: Instruction Retention vs. Retained KV Cache", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("KV Cache Retained (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Instruction Retention Rate (%)", fontsize=11, labelpad=8)
    ax.set_ylim(-5, 105)
    ax.set_xlim(5, 105)
    ax.axhline(100, color="gray", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="lower right", fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "retention_vs_cache.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------
    # 2. KV Retention (%) vs. Peak Memory (MB)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for policy, label in POLICY_LABELS.items():
        pol_data = summary[summary["policy"] == policy].sort_values("cache_retention_pct_mean")
        if pol_data.empty:
            continue
        color = POLICY_PALETTE.get(policy, "#333333")
        ax.plot(
            pol_data["cache_retention_pct_mean"],
            pol_data["peak_memory_mb_mean"],
            marker="o",
            markersize=8,
            linewidth=2.2,
            label=label,
            color=color
        )

    ax.set_title("Memory Scaling: Retained KV Cache vs. Peak Memory Footprint", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("KV Cache Retained (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Peak Memory (MB)", fontsize=11, labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper left", fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "memory_vs_retention.png")
    fig.savefig(p2)
    plt.close(fig)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------
    # 3. KV Retention (%) vs. Decode Throughput (tokens/s)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for policy, label in POLICY_LABELS.items():
        pol_data = summary[summary["policy"] == policy].sort_values("cache_retention_pct_mean")
        if pol_data.empty:
            continue
        color = POLICY_PALETTE.get(policy, "#333333")
        ax.plot(
            pol_data["cache_retention_pct_mean"],
            pol_data["tokens_per_second_mean"],
            marker="^",
            markersize=8,
            linewidth=2.2,
            label=label,
            color=color
        )

    ax.set_title("Inference Speed: KV Cache Budget vs. Decode Throughput", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("KV Cache Retained (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Decode Throughput (Tokens / Second)", fontsize=11, labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="best", fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "throughput_vs_retention.png")
    fig.savefig(p3)
    plt.close(fig)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------
    # 4. Context Length vs. Decode Latency across Policies
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ctx_summary = df.groupby(["policy", "target_length"])["decode_latency_sec"].mean().reset_index()
    for policy, label in POLICY_LABELS.items():
        pol_data = ctx_summary[ctx_summary["policy"] == policy].sort_values("target_length")
        if pol_data.empty:
            continue
        color = POLICY_PALETTE.get(policy, "#333333")
        ax.plot(
            pol_data["target_length"],
            pol_data["decode_latency_sec"],
            marker="s",
            markersize=8,
            linewidth=2.2,
            label=label,
            color=color
        )

    ax.set_title("Context Length vs. Decode Latency by Policy", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Target Context Length (Tokens)", fontsize=11, labelpad=8)
    ax.set_ylabel("Decode Latency (Seconds for 50 Tokens)", fontsize=11, labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper left", fontsize=10)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "latency_vs_context.png")
    fig.savefig(p4)
    plt.close(fig)
    print(f"Saved: {p4}")

def evaluate_and_report(csv_path: str, plots_dir: str = "results/plots"):
    df = load_and_preprocess_results(csv_path)
    summary = compute_summary_metrics(df)

    print("\n" + "="*85)
    print("                      BENCHMARK EVALUATION SUMMARY TABLE")
    print("="*85)
    print(summary[[
        "policy",
        "budget_ratio",
        "cache_retention_pct_mean",
        "instruction_retention_mean",
        "relative_quality",
        "kv_cache_mb_mean",
        "tokens_per_second_mean"
    ]].to_string(index=False))
    print("="*85 + "\n")

    generate_plots(df, summary, plots_dir)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KV Cache Compression Results")
    parser.add_argument("--csv", type=str, default="results/results.csv")
    parser.add_argument("--plots-dir", type=str, default="results/plots")
    args = parser.parse_args()

    evaluate_and_report(args.csv, args.plots_dir)
