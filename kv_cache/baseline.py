"""
Baselines for KV-Cache Compression:
1. Full KV Baseline (Policy A): Retains 100% of KV cache positions.
2. Recent-Window Baseline (Policy B): Sliding window with attention sinks (StreamingLLM style).
"""

import time
import torch
from typing import Dict, Any, Optional
from .utils import (
    MemoryTracker,
    calculate_kv_cache_bytes,
    get_kv_cache_len,
    prune_dynamic_cache,
    autoregressive_decode
)

def run_full_kv_baseline(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100
) -> Dict[str, Any]:
    """
    Executes Baseline A: Full KV cache generation without any compression.
    """
    mem_tracker = MemoryTracker(device)
    mem_tracker.start()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    # Prefill phase
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    prefill_time = time.perf_counter() - t_prefill_start

    past_key_values = prefill_out.past_key_values
    original_cache_len = get_kv_cache_len(past_key_values)
    original_cache_bytes = calculate_kv_cache_bytes(past_key_values)

    # Decode phase
    last_token = input_ids[:, -1:]
    generated_tokens, decode_time, tokens_per_sec = autoregressive_decode(
        model=model,
        past_key_values=past_key_values,
        last_token_id=last_token,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        start_position=seq_len
    )

    peak_mem_mb = mem_tracker.stop()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "policy": "Full_KV",
        "budget_ratio": 1.0,
        "original_seq_len": seq_len,
        "retained_positions": original_cache_len,
        "cache_retention_ratio": 1.0,
        "kv_cache_bytes": original_cache_bytes,
        "prefill_latency_sec": prefill_time,
        "decode_latency_sec": decode_time,
        "tokens_per_second": tokens_per_sec,
        "peak_memory_mb": peak_mem_mb,
        "generated_tokens": len(generated_tokens),
        "generated_text": generated_text
    }

def run_recent_window_baseline(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    budget_ratio: float,
    sink_tokens: int = 4,
    max_new_tokens: int = 100
) -> Dict[str, Any]:
    """
    Executes Baseline B: Recent-window eviction.
    Retains sink_tokens initial tokens (attention sink) + the most recent tokens
    up to the allocated cache budget.
    """
    mem_tracker = MemoryTracker(device)
    mem_tracker.start()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    # Prefill phase
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    prefill_time = time.perf_counter() - t_prefill_start

    past_key_values = prefill_out.past_key_values
    original_cache_len = get_kv_cache_len(past_key_values)
    original_cache_bytes = calculate_kv_cache_bytes(past_key_values)

    budget_tokens = max(1, int(round(seq_len * budget_ratio)))

    if budget_tokens < seq_len:
        # Determine retained indices: sink + recent
        actual_sink = min(sink_tokens, budget_tokens)
        recent_count = budget_tokens - actual_sink
        
        sink_indices = list(range(actual_sink))
        recent_indices = list(range(seq_len - recent_count, seq_len)) if recent_count > 0 else []
        retained_indices = sorted(set(sink_indices + recent_indices))
        
        past_key_values = prune_dynamic_cache(past_key_values, retained_indices)

    retained_positions = get_kv_cache_len(past_key_values)
    compressed_cache_bytes = calculate_kv_cache_bytes(past_key_values)

    # Decode phase
    last_token = input_ids[:, -1:]
    generated_tokens, decode_time, tokens_per_sec = autoregressive_decode(
        model=model,
        past_key_values=past_key_values,
        last_token_id=last_token,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        start_position=seq_len
    )

    peak_mem_mb = mem_tracker.stop()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "policy": "Recent_Window",
        "budget_ratio": budget_ratio,
        "original_seq_len": seq_len,
        "retained_positions": retained_positions,
        "cache_retention_ratio": retained_positions / max(1, seq_len),
        "kv_cache_bytes": compressed_cache_bytes,
        "prefill_latency_sec": prefill_time,
        "decode_latency_sec": decode_time,
        "tokens_per_second": tokens_per_sec,
        "peak_memory_mb": peak_mem_mb,
        "generated_tokens": len(generated_tokens),
        "generated_text": generated_text
    }
