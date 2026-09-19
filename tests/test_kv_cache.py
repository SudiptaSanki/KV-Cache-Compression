"""
Unit tests for KV Cache compression algorithms and utility functions.
"""

import unittest
import torch
from kv_cache.attention_select import (
    compute_attention_importance_scores,
    select_attention_guided_indices
)
from kv_cache.protected_kv import select_protected_attention_indices
from kv_cache.utils import get_device, get_model_dtype

class TestKVCacheCompression(unittest.TestCase):

    def test_attention_scoring_shape(self):
        # 6 layers, 1 batch, 8 heads, 30 seq_len, 30 seq_len
        num_layers = 6
        num_heads = 8
        seq_len = 30
        mock_attentions = tuple(
            torch.rand(1, num_heads, seq_len, seq_len) for _ in range(num_layers)
        )
        scores = compute_attention_importance_scores(mock_attentions, seq_len, observation_window=10)
        self.assertEqual(scores.shape, (seq_len,))
        self.assertFalse(torch.isnan(scores).any())

    def test_attention_guided_selection_budget(self):
        seq_len = 100
        budget = 25
        scores = torch.rand(seq_len)
        indices = select_attention_guided_indices(
            scores=scores,
            seq_len=seq_len,
            budget_tokens=budget,
            sink_tokens=4,
            observation_window=8
        )
        self.assertEqual(len(indices), budget)
        # Sinks (0, 1, 2, 3) must be present
        for i in range(4):
            self.assertIn(i, indices)
        # Observation window (92..99) must be present
        for i in range(92, 100):
            self.assertIn(i, indices)

    def test_protected_attention_selection_preserves_tokens(self):
        seq_len = 200
        budget = 30
        protected_set = {10, 11, 12, 13, 14, 15}  # e.g., critical instruction
        scores = torch.rand(seq_len)
        # Make protected positions have low score to ensure protection is what keeps them
        scores[list(protected_set)] = -100.0

        indices = select_protected_attention_indices(
            scores=scores,
            seq_len=seq_len,
            budget_tokens=budget,
            protected_indices=protected_set,
            sink_tokens=4,
            observation_window=8
        )
        self.assertEqual(len(indices), budget)
        # Every single protected token MUST be retained
        for p in protected_set:
            self.assertIn(p, indices, f"Protected token {p} was evicted!")

    def test_device_fallback(self):
        device = get_device()
        self.assertIn(device.type, ["cuda", "cpu"])
        dtype = get_model_dtype(device)
        self.assertIn(dtype, [torch.float16, torch.bfloat16, torch.float32])

if __name__ == "__main__":
    unittest.main()
