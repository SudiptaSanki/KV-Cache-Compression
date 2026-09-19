"""
Main Experiment Runner for KV Cache Compression Benchmark.

Executes controlled trials across:
- Policies: Full KV, Recent Window, Attention-Guided, Protected Attention-Guided
- Cache budgets: 100%, 50%, 25%, 12.5%
- Repeats: 3 repeats per configuration with controlled seeds
- Records latency, memory, cache size, and instruction retention into results/results.csv
"""

import os
import json
import argparse
import time
from typing import Optional, List, Dict, Any
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache.utils import get_device, get_model_dtype
from kv_cache.baseline import run_full_kv_baseline, run_recent_window_baseline
from kv_cache.attention_select import run_attention_guided_selection
from kv_cache.protected_kv import run_protected_attention_selection

def check_instruction_retention(generated_text: str, target_keyword: str) -> bool:
    """
    Evaluates whether the critical instruction keyword is retained in the generated output.
    Uses case-insensitive substring/token matching.
    """
    gen_lower = generated_text.lower()
    target_lower = target_keyword.lower()
    return target_lower in gen_lower

def run_experiments(
    prompts_file: str,
    model_name: str = "EleutherAI/pythia-70m",
    budgets: list = [1.0, 0.50, 0.25, 0.125],
    repeats: int = 3,
    max_new_tokens: int = 50,
    observation_window: int = 32,
    sink_tokens: int = 4,
    output_csv: str = "results/results.csv",
    limit_prompts: Optional[int] = None
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    device = get_device()
    dtype = get_model_dtype(device)
    print(f"[Benchmark] Initializing on device: {device} with dtype: {dtype}")

    print(f"[Benchmark] Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="eager"
    ).to(device)
    model.eval()

    # Load prompts
    prompts_data = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts_data.append(json.loads(line))

    if limit_prompts is not None and limit_prompts > 0:
        prompts_data = prompts_data[:limit_prompts]

    print(f"[Benchmark] Loaded {len(prompts_data)} prompts from {prompts_file}")

    all_records = []
    total_configs = len(prompts_data) * (1 + 3 * (len(budgets) - 1)) * repeats
    print(f"[Benchmark] Starting execution of {total_configs} total trial runs...")

    pbar = tqdm(total=total_configs, desc="Running Benchmark")

    for prompt_entry in prompts_data:
        p_id = prompt_entry["id"]
        domain = prompt_entry.get("domain", "Unknown")
        prompt_text = prompt_entry["prompt"]
        target_keyword = prompt_entry["target_keyword"]
        token_span = prompt_entry.get("token_span", (0, 0))
        target_length = prompt_entry.get("target_length", len(prompt_text))

        for repeat_idx in range(repeats):
            seed = 42 + repeat_idx
            torch.manual_seed(seed)

            # 1. Full KV Baseline (Always evaluated at budget 1.0)
            res_full = run_full_kv_baseline(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                device=device,
                max_new_tokens=max_new_tokens
            )
            retained = check_instruction_retention(res_full["generated_text"], target_keyword)
            res_full.update({
                "prompt_id": p_id,
                "domain": domain,
                "target_length": target_length,
                "target_keyword": target_keyword,
                "instruction_retained": retained,
                "repeat": repeat_idx + 1,
                "seed": seed,
                "model_name": model_name
            })
            all_records.append(res_full)
            pbar.update(1)

            # 2. Compressed Policies at lower budgets (50%, 25%, 12.5%)
            for budget in budgets:
                if budget >= 1.0:
                    continue  # Full KV already recorded above

                # Policy B: Recent Window
                res_win = run_recent_window_baseline(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    device=device,
                    budget_ratio=budget,
                    sink_tokens=sink_tokens,
                    max_new_tokens=max_new_tokens
                )
                retained_win = check_instruction_retention(res_win["generated_text"], target_keyword)
                res_win.update({
                    "prompt_id": p_id,
                    "domain": domain,
                    "target_length": target_length,
                    "target_keyword": target_keyword,
                    "instruction_retained": retained_win,
                    "repeat": repeat_idx + 1,
                    "seed": seed,
                    "model_name": model_name
                })
                all_records.append(res_win)
                pbar.update(1)

                # Policy C: Attention-Guided Selection
                res_attn = run_attention_guided_selection(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    device=device,
                    budget_ratio=budget,
                    observation_window=observation_window,
                    sink_tokens=sink_tokens,
                    max_new_tokens=max_new_tokens
                )
                retained_attn = check_instruction_retention(res_attn["generated_text"], target_keyword)
                res_attn.update({
                    "prompt_id": p_id,
                    "domain": domain,
                    "target_length": target_length,
                    "target_keyword": target_keyword,
                    "instruction_retained": retained_attn,
                    "repeat": repeat_idx + 1,
                    "seed": seed,
                    "model_name": model_name
                })
                all_records.append(res_attn)
                pbar.update(1)

                # Policy D: Protected Attention Selection (Extension)
                res_prot = run_protected_attention_selection(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    protected_token_spans=[token_span],
                    device=device,
                    budget_ratio=budget,
                    observation_window=observation_window,
                    sink_tokens=sink_tokens,
                    max_new_tokens=max_new_tokens
                )
                retained_prot = check_instruction_retention(res_prot["generated_text"], target_keyword)
                res_prot.update({
                    "prompt_id": p_id,
                    "domain": domain,
                    "target_length": target_length,
                    "target_keyword": target_keyword,
                    "instruction_retained": retained_prot,
                    "repeat": repeat_idx + 1,
                    "seed": seed,
                    "model_name": model_name
                })
                all_records.append(res_prot)
                pbar.update(1)

            # Periodically write out results
            df_partial = pd.DataFrame(all_records)
            df_partial.to_csv(output_csv, index=False)

    pbar.close()
    df_final = pd.DataFrame(all_records)
    df_final.to_csv(output_csv, index=False)
    print(f"\n[Benchmark] Successfully finished! Saved {len(df_final)} trials to {output_csv}")
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Long-Context KV-Cache Compression Benchmark")
    parser.add_argument("--prompts-file", type=str, default="benchmarks/prompts.jsonl")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.50, 0.25, 0.125])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="results/results.csv")
    args = parser.parse_args()

    run_experiments(
        prompts_file=args.prompts_file,
        model_name=args.model,
        budgets=args.budgets,
        repeats=args.repeats,
        max_new_tokens=args.max_new_tokens,
        output_csv=args.output_csv,
        limit_prompts=args.limit_prompts
    )
