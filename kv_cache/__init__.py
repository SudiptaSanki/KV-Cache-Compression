"""
KV Cache Compression Research Package
Provides implementations of:
- Full KV baseline
- Recent Window baseline
- SnapKV-style Attention-Guided KV selection
- Instruction-Protected Attention-Guided selection (Extension)
"""

from .baseline import run_full_kv_baseline, run_recent_window_baseline
from .attention_select import run_attention_guided_selection, compute_attention_importance_scores
from .protected_kv import run_protected_attention_selection
from .utils import get_device, get_model_dtype, MemoryTracker, calculate_kv_cache_bytes, get_kv_cache_len

__all__ = [
    "run_full_kv_baseline",
    "run_recent_window_baseline",
    "run_attention_guided_selection",
    "compute_attention_importance_scores",
    "run_protected_attention_selection",
    "get_device",
    "get_model_dtype",
    "MemoryTracker",
    "calculate_kv_cache_bytes",
    "get_kv_cache_len"
]
