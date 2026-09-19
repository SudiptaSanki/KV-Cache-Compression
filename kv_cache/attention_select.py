"""
Method C: Simplified SnapKV-Style Attention-Guided KV Cache Selection.
Uses observed attention weights from the prompt's trailing observation window
to compute importance scores for historical KV positions, retaining the top-K.
"""

import time
import torch
from typing import Dict, Any, List, Optional
from .utils import (
    MemoryTracker,
    calculate_kv_cache_bytes,
    get_kv_cache_len,
    prune_dynamic_cache,
    autoregressive_decode
)

def compute_attention_importance_scores(
    attentions: tuple,
    seq_len: int,
    observation_window: int = 32
) -> torch.Tensor:
    """
    Computes an importance score for each key position across layers.
    Args:
        attentions: Tuple of attention tensors from prefill [num_layers],
                    each shaped [batch_size, num_heads, seq_len, seq_len].
        seq_len: Total prompt sequence length.
        observation_window: Number of trailing prompt tokens to use as queries.
    Returns:
        1D tensor of shape [seq_len] with mean attention importance per position.
    """
    obs_w = min(observation_window, seq_len)
    num_layers = len(attentions)
    
    # Accumulate importance across layers
    total_scores = torch.zeros(seq_len, device=attentions[0].device, dtype=torch.float32)
    
    for layer_attn in attentions:
        # layer_attn: [batch_size=1, num_heads, seq_len, seq_len]
        # Slice queries to observation window: [num_heads, obs_w, seq_len]
        obs_attn = layer_attn[0, :, -obs_w:, :]
        # Mean across heads and observation query tokens: [seq_len]
        layer_score = obs_attn.mean(dim=(0, 1)).to(torch.float32)
        total_scores += layer_score
        
    return total_scores / num_layers

def select_attention_guided_indices(
    scores: torch.Tensor,
    seq_len: int,
    budget_tokens: int,
    sink_tokens: int = 4,
    observation_window: int = 32
) -> List[int]:
    """
    Selects KV positions to retain:
      - First `sink_tokens` (attention sink)
      - Last `obs_w` tokens (observation window)
      - Top-K historical tokens outside the above based on attention scores
    """
    if budget_tokens >= seq_len:
        return list(range(seq_len))

    obs_w = min(observation_window, seq_len)
    sink_w = min(sink_tokens, budget_tokens)

    # Base mandatory sets
    sink_set = set(range(sink_w))
    obs_set = set(range(seq_len - obs_w, seq_len))
    mandatory_set = sink_set | obs_set

    if len(mandatory_set) >= budget_tokens:
        # If mandatory set exceeds budget, prioritize the most recent tokens + sink
        remaining = budget_tokens - sink_w
        if remaining > 0:
            retained = sorted(sink_set | set(range(seq_len - remaining, seq_len)))
        else:
            retained = sorted(list(sink_set)[:budget_tokens])
        return retained

    # Remaining budget for historical positions
    remaining_budget = budget_tokens - len(mandatory_set)

    # Candidate historical positions
    candidate_positions = [i for i in range(seq_len) if i not in mandatory_set]
    if candidate_positions and remaining_budget > 0:
        candidate_scores = scores[candidate_positions]
        k = min(remaining_budget, len(candidate_positions))
        _, topk_relative_indices = torch.topk(candidate_scores, k=k)
        selected_candidates = [candidate_positions[idx] for idx in topk_relative_indices.tolist()]
        retained = sorted(mandatory_set | set(selected_candidates))
    else:
        retained = sorted(mandatory_set)

    return retained

def run_attention_guided_selection(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    budget_ratio: float,
    observation_window: int = 32,
    sink_tokens: int = 4,
    max_new_tokens: int = 100
) -> Dict[str, Any]:
    """
    Executes Method C: SnapKV-style attention-guided compression.
    """
    mem_tracker = MemoryTracker(device)
    mem_tracker.start()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

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

        retained_indices = select_attention_guided_indices(
            scores=scores,
            seq_len=seq_len,
            budget_tokens=budget_tokens,
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
        "policy": "Attention_Guided",
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
