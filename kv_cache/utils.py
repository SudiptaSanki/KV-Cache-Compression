"""
Utility functions for KV-cache compression, memory/latency profiling,
and cache manipulation.
"""

import time
import os
import torch
import psutil
from typing import List, Tuple, Union, Optional

def get_device() -> torch.device:
    """
    Detects and validates the compute device.
    Verifies that CUDA actually executes kernels (e.g. guard against unsupported arch like sm_120 on older wheels).
    Falls back gracefully to CPU.
    """
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            _ = test_tensor + 1
            return torch.device("cuda")
        except Exception:
            # Device exists but lacks compatible compiled kernels
            return torch.device("cpu")
    return torch.device("cpu")

def get_model_dtype(device: torch.device) -> torch.dtype:
    """
    Returns float32 on CPU to prevent NaNs in small causal models,
    and float16 on GPU for efficiency.
    """
    if device.type == "cuda":
        return torch.float16
    return torch.float32

class MemoryTracker:
    """
    Profiles peak memory usage during prefill and decode.
    Tracks GPU VRAM (via CUDA memory stats) if available, or process RSS RAM via psutil.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = 0.0
        self.peak_memory = 0.0

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = self.baseline_memory

    def stop(self) -> float:
        if self.device.type == "cuda":
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            current = self.process.memory_info().rss / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, current)
        return self.peak_memory

def calculate_kv_cache_bytes(past_key_values) -> int:
    """
    Calculates the exact total memory in bytes occupied by the stored KV cache.
    Works with transformers DynamicCache.
    """
    total_bytes = 0
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                total_bytes += layer.keys.element_size() * layer.keys.nelement()
            if hasattr(layer, "values") and layer.values is not None:
                total_bytes += layer.values.element_size() * layer.values.nelement()
    elif isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                k, v = layer[0], layer[1]
                total_bytes += k.element_size() * k.nelement() + v.element_size() * v.nelement()
    return total_bytes

def get_kv_cache_len(past_key_values) -> int:
    """
    Returns the sequence length of the stored KV cache in layer 0.
    """
    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
        l0 = past_key_values.layers[0]
        if hasattr(l0, "keys") and l0.keys is not None:
            return l0.keys.shape[2]
    elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        return past_key_values[0][0].shape[2]
    return 0

def prune_dynamic_cache(past_key_values, indices: Union[torch.Tensor, List[int], List[torch.Tensor]]):
    """
    Prunes a transformers DynamicCache to retain only the specified sequence indices.
    Supports either:
      - A single 1D index tensor/list applied across all layers
      - A list of 1D index tensors, one per layer
    """
    if hasattr(past_key_values, "layers"):
        for i, layer in enumerate(past_key_values.layers):
            layer_idx = indices[i] if isinstance(indices, list) and isinstance(indices[0], (torch.Tensor, list)) else indices
            if not isinstance(layer_idx, torch.Tensor):
                layer_idx = torch.tensor(layer_idx, dtype=torch.long, device=layer.keys.device)
            else:
                layer_idx = layer_idx.to(device=layer.keys.device, dtype=torch.long)
            
            # Sort indices to preserve monotonic causal temporal ordering
            sorted_idx, _ = torch.sort(layer_idx)
            layer.keys = layer.keys[:, :, sorted_idx, :]
            layer.values = layer.values[:, :, sorted_idx, :]
    return past_key_values

def autoregressive_decode(
    model,
    past_key_values,
    last_token_id: torch.Tensor,
    max_new_tokens: int = 100,
    eos_token_id: Optional[int] = None,
    start_position: Optional[int] = None
) -> Tuple[List[int], float, float]:
    """
    Runs autoregressive decoding for up to max_new_tokens.
    If start_position is provided, passes explicit position_ids to preserve
    correct RoPE relative distances with pruned KV caches.
    Returns:
      - generated_tokens: List of generated token IDs
      - decode_time_sec: Total time spent in decoding
      - tokens_per_second: Decode throughput
    """
    curr = last_token_id
    generated_tokens = []
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            kwargs = {"input_ids": curr, "past_key_values": past_key_values, "use_cache": True}
            if start_position is not None:
                kwargs["position_ids"] = torch.tensor(
                    [[start_position + step]],
                    dtype=torch.long,
                    device=curr.device
                )
            out = model(**kwargs)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            curr = next_token
            
    decode_time_sec = max(time.perf_counter() - t0, 1e-6)
    tokens_per_second = len(generated_tokens) / decode_time_sec
    return generated_tokens, decode_time_sec, tokens_per_second
