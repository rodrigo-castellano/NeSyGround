"""Tests for soundness filters: prune and provset."""

import torch
import pytest
from grounder import KB, BCGrounder

DEVICE = torch.device("cpu")


class TestPruneFilter:
    """Test PruneIncompleteProofs filter."""

    def _make_grounder(self, filter_mode: str):
        facts = torch.tensor([
            [1, 1, 2],  # base(a, b)
            [1, 2, 3],  # base(b, c)
        ], dtype=torch.long)
        heads = torch.tensor([[2, 4, 5]], dtype=torch.long)
        bodies = torch.tensor([[[1, 4, 5]]], dtype=torch.long)
        rule_lens = torch.tensor([1], dtype=torch.long)
        kb = KB(facts, heads, bodies, rule_lens,
                constant_no=3, predicate_no=3,
                padding_idx=99, device=DEVICE)
        return BCGrounder(
            kb, resolution='sld', filter=filter_mode,
            max_goals=4, depth=2, max_total_groundings=64,
            max_derived_per_state=20,
        )

    def test_prune_keeps_provable(self):
        grounder = self._make_grounder('prune')
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() >= 1

    def test_prune_filters_unprovable(self):
        grounder = self._make_grounder('prune')
        queries = torch.tensor([[2, 3, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() == 0

    def test_no_filter_returns_more(self):
        """Filter='none' should return >= filter='prune' results."""
        grounder_none = self._make_grounder('none')
        grounder_prune = self._make_grounder('prune')
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result_none = grounder_none(queries, query_mask)
        result_prune = grounder_prune(queries, query_mask)
        none_count = result_none.count[0].item()
        assert none_count >= result_prune.count[0].item()


class TestHashAtoms:
    """Test the shared hash utility."""

    def test_hash_consistency(self):
        from grounder.filters._hash import hash_atoms
        from grounder.fact_index import pack_triples_64
        atoms = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        base = 100
        h1 = hash_atoms(atoms, base)
        h2 = pack_triples_64(atoms, base)
        assert (h1 == h2).all()
