"""
Extension D: Instruction-Protected Attention-Guided KV Cache Compression.

This module implements the controlled extension described in the IIT Bombay build specification:
It explicitly protects instruction-critical positions in the KV cache, reserving a portion
of the cache budget for them, and allocates the remaining budget to historical positions
ranked by attention-derived importance scores.
"""

import time
import torch
from typing import Dict, Any, List, Set, Optional, Tuple
from .utils import (
    MemoryTracker,
    calculate_kv_cache_bytes,
    get_kv_cache_len,
    prune_dynamic_cache,
    autoregressive_decode
)
from .attention_select import compute_attention_importance_scores

def select_protected_attention_indices(
    scores: torch.Tensor,
    seq_len: int,
    budget_tokens: int,
    protected_indices: Set[int],
    sink_tokens: int = 4,
    observation_window: int = 32
) -> List[int]:
    """
    Selects KV positions to retain while strictly guaranteeing retention of protected instruction tokens.
    
    Order of priority:
      1. Protected instruction tokens (user-specified task instructions)
      2. Attention sinks (first `sink_tokens`)
      3. Observation window (last `observation_window` tokens)
      4. Top-K historical tokens ranked by attention score outside the above
    """
    if budget_tokens >= seq_len:
        return list(range(seq_len))

    obs_w = min(observation_window, seq_len)
    sink_w = min(sink_tokens, budget_tokens)

    # Base mandatory sets
    sink_set = set(range(sink_w))
    obs_set = set(range(seq_len - obs_w, seq_len))
    
    # Priority 1: Protected positions within prompt bounds
    valid_protected = {idx for idx in protected_indices if 0 <= idx < seq_len}
    
    # Mandatory core: protected tokens + sinks + observation window
    base_reserved = valid_protected | sink_set | obs_set

    if len(base_reserved) >= budget_tokens:
        # If the combination exceeds budget, ensure all protected tokens are kept,
        # then fill up to budget using recent observation tokens and sink
        retained = set(valid_protected)
        # Add sink tokens if budget allows
        for s in range(sink_w):
            if len(retained) >= budget_tokens:
                break
            retained.add(s)
        # Add trailing observation tokens backwards from seq_len - 1
        for o in range(seq_len - 1, -1, -1):
            if len(retained) >= budget_tokens:
                break
            retained.add(o)
        return sorted(retained)

    # Remaining budget available for attention-guided historical selection
    remaining_budget = budget_tokens - len(base_reserved)

    # Candidate positions: all positions not already in base_reserved
    candidate_positions = [i for i in range(seq_len) if i not in base_reserved]
    
    if candidate_positions and remaining_budget > 0:
        candidate_scores = scores[candidate_positions]
        k = min(remaining_budget, len(candidate_positions))
        _, topk_relative_indices = torch.topk(candidate_scores, k=k)
        selected_candidates = [candidate_positions[idx] for idx in topk_relative_indices.tolist()]
        retained = sorted(base_reserved | set(selected_candidates))
    else:
        retained = sorted(base_reserved)

    return retained

def run_protected_attention_selection(
    model,
    tokenizer,
    prompt: str,
    protected_token_spans: List[Tuple[int, int]],
    device: torch.device,
    budget_ratio: float,
    observation_window: int = 32,
    sink_tokens: int = 4,
    max_new_tokens: int = 100
) -> Dict[str, Any]:
    """
    Executes Extension D: Instruction-Protected Attention-Guided Compression.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: Model tokenizer
        prompt: Full prompt string
        protected_token_spans: List of (start_token_idx, end_token_idx) tuples for critical instructions
        device: Target compute device
        budget_ratio: Target fraction of KV cache to retain
        observation_window: Size of observation query window
        sink_tokens: Number of initial sink tokens
        max_new_tokens: Number of generation tokens
    """
    mem_tracker = MemoryTracker(device)
    mem_tracker.start()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    # Expand protected token spans into index set
    protected_indices: Set[int] = set()
    for start, end in protected_token_spans:
        for idx in range(start, min(end, seq_len)):
            protected_indices.add(idx)

    # Prefill phase with attention extraction
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(**inputs, output_attentions=True, use_cache=True)
    prefill_time = time.perf_counter() - t_prefill_start

    past_key_values = prefill_out.past_key_values
    original_cache_len = get_kv_cache_len(past_key_values)
    original_cache_bytes = calculate_kv_cache_bytes(past_key_values)

    budget_tokens = max(1, int(round(seq_len * budget_ratio)))

    if budget_tokens < seq_len:
        # Compute attention scores from observation window
        scores = compute_attention_importance_scores(
            attentions=prefill_out.attentions,
            seq_len=seq_len,
            observation_window=observation_window
        )

        retained_indices = select_protected_attention_indices(
            scores=scores,
            seq_len=seq_len,
            budget_tokens=budget_tokens,
            protected_indices=protected_indices,
            sink_tokens=sink_tokens,
            observation_window=observation_window
        )

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
        "policy": "Protected_Attention",
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
        "generated_text": generated_text,
        "protected_tokens_count": len(protected_indices)
    }
