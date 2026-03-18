"""Tests for grounder.fact_index: ArgKeyFactIndex, InvertedFactIndex, BlockSparseFactIndex."""

import torch
import pytest
from grounder.data.fact_index import (
    ArgKeyFactIndex,
    InvertedFactIndex,
    BlockSparseFactIndex,
    fact_contains,
    pack_triples_64,
)

DEVICE = torch.device("cpu")
PAD = 99
C_NO = 5


def _make_facts():
    """Small KB: parent(alice=1,bob=2), parent(bob=2,charlie=3), friend(alice=1,charlie=3)."""
    return torch.tensor([
        [1, 1, 2],  # parent(alice, bob)
        [1, 2, 3],  # parent(bob, charlie)
        [2, 1, 3],  # friend(alice, charlie)
    ], dtype=torch.long)


class TestPackTriples:
    def test_unique_hashes(self):
        facts = _make_facts()
        base = 100
        h = pack_triples_64(facts, base)
        assert h.shape == (3,)
        assert h.unique().shape[0] == 3  # all distinct


class TestFactContains:
    def test_membership(self):
        facts = _make_facts()
        base = 100
        hashes = pack_triples_64(facts, base).sort().values
        # Queries: first two are facts, third is not
        queries = torch.tensor([
            [1, 1, 2],
            [2, 1, 3],
            [1, 3, 1],  # not a fact
        ])
        result = fact_contains(queries, hashes, base)
        assert result.tolist() == [True, True, False]


class TestArgKeyFactIndex:
    def test_targeted_lookup(self):
        facts = _make_facts()
        idx = ArgKeyFactIndex(facts, constant_no=C_NO, padding_idx=PAD, device=DEVICE, pack_base=100)
        # Query: parent(alice, V0) -> should find parent(alice, bob)
        queries = torch.tensor([[1, 1, 6]])  # pred=1, arg0=1 (ground), arg1=6 (var)
        item_idx, valid = idx.targeted_lookup(queries, max_results=4)
        assert valid.any()
        # The returned fact should be parent(alice, bob) = [1, 1, 2]
        valid_items = item_idx[0, valid[0]]
        found_facts = idx.facts_idx[valid_items]
        assert any((found_facts == torch.tensor([1, 1, 2])).all(dim=1))

    def test_no_match(self):
        facts = _make_facts()
        idx = ArgKeyFactIndex(facts, constant_no=C_NO, padding_idx=PAD, device=DEVICE, pack_base=100)
        # Query with predicate 5 (doesn't exist) — targeted_lookup may return
        # clamped indices but unification will fail on predicate mismatch.
        # Check that no unification succeeds:
        from grounder.resolution.primitives import unify_one_to_one
        queries = torch.tensor([[5, 1, 6]])
        item_idx, valid = idx.targeted_lookup(queries, max_results=4)
        if valid.any():
            safe_idx = item_idx.clamp(0, idx.facts_idx.shape[0] - 1)
            fact_atoms = idx.facts_idx[safe_idx[0, valid[0]]]
            q_exp = queries.expand(fact_atoms.shape[0], -1)
            ok, _ = unify_one_to_one(q_exp, fact_atoms, C_NO, PAD)
            assert not ok.any(), "No fact should unify with predicate 5"

    def test_both_vars(self):
        facts = _make_facts()
        idx = ArgKeyFactIndex(facts, constant_no=C_NO, padding_idx=PAD, device=DEVICE, pack_base=100)
        # Both args are vars: parent(V0, V1)
        queries = torch.tensor([[1, 6, 7]])
        item_idx, valid = idx.targeted_lookup(queries, max_results=4)
        # Should find all parent facts
        n_valid = valid[0].sum().item()
        assert n_valid == 2  # parent(alice,bob) + parent(bob,charlie)

    def test_exists(self):
        facts = _make_facts()
        idx = ArgKeyFactIndex(facts, constant_no=C_NO, padding_idx=PAD, device=DEVICE, pack_base=100)
        atoms = torch.tensor([
            [1, 1, 2],  # parent(alice, bob) — exists
            [1, 2, 3],  # parent(bob, charlie) — exists
            [2, 1, 3],  # friend(alice, charlie) — exists
            [1, 3, 1],  # parent(charlie, alice) — doesn't exist
        ])
        result = idx.exists(atoms)
        assert result.tolist() == [True, True, True, False]


class TestInvertedFactIndex:
    def test_enumerate(self):
        facts = _make_facts()
        idx = InvertedFactIndex(
            facts, constant_no=C_NO, predicate_no=2,
            padding_idx=PAD, device=DEVICE, max_facts_per_query=4,
        )
        # Enumerate parent facts with arg0=1
        pred = torch.tensor([1])
        bound = torch.tensor([1])
        direction = torch.tensor([0])  # arg0 is bound
        cands, mask = idx.enumerate(pred, bound, direction)
        assert mask.any()
        # Should return obj=2 (from parent(alice=1, bob=2))
        valid_cands = cands[0, mask[0]]
        assert 2 in valid_cands.tolist()


class TestBlockSparseFactIndex:
    def test_enumerate(self):
        facts = _make_facts()
        idx = BlockSparseFactIndex(
            facts, constant_no=C_NO, predicate_no=2,
            padding_idx=PAD, device=DEVICE, max_facts_per_query=4,
            max_memory_mb=256,
        )
        pred = torch.tensor([1])
        bound = torch.tensor([2])
        direction = torch.tensor([0])
        cands, mask = idx.enumerate(pred, bound, direction)
        assert mask.any()
        valid_cands = cands[0, mask[0]]
        assert 3 in valid_cands.tolist()
