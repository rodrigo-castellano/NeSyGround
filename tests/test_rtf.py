"""Tests for BCGrounder with RTF resolution — basic functionality checks."""

import torch
import pytest
from grounder import KB, BCGrounder

DEVICE = torch.device("cpu")


class TestRTFSmoke:
    """Smoke tests: RTF resolution doesn't crash with valid inputs."""

    def _make_grounder(self):
        facts = torch.tensor([
            [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
            [1, 1, 3], [1, 2, 4], [1, 3, 5],
        ], dtype=torch.long)
        heads = torch.tensor([[2, 6, 7]], dtype=torch.long)
        bodies = torch.tensor([[[1, 6, 7]]], dtype=torch.long)
        rule_lens = torch.tensor([1], dtype=torch.long)
        kb = KB(facts, heads, bodies, rule_lens,
                constant_no=5, predicate_no=3,
                padding_idx=99, device=DEVICE)
        return BCGrounder(
            kb, resolution='rtf', filter='prune',
            max_goals=4, depth=2, max_total_groundings=64,
            max_derived_per_state=20,
        )

    def test_rtf_runs_without_error(self):
        grounder = self._make_grounder()
        queries = torch.tensor([[2, 1, 6]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.body.shape[0] == 1

    def test_rtf_batch(self):
        grounder = self._make_grounder()
        queries = torch.tensor([[2, 1, 6], [2, 2, 6]], dtype=torch.long)
        query_mask = torch.tensor([True, True])
        result = grounder(queries, query_mask)
        assert result.body.shape[0] == 2

    def test_rtf_masked_query(self):
        grounder = self._make_grounder()
        queries = torch.tensor([[2, 1, 6]], dtype=torch.long)
        query_mask = torch.tensor([False])
        result = grounder(queries, query_mask)
        assert result.count[0].item() == 0
