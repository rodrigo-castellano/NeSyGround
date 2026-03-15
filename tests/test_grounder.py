"""Tests for grounder.grounder: BCGrounder end-to-end (SLD resolution, no filter)."""

import torch
import pytest
from grounder import BCGrounder

DEVICE = torch.device("cpu")


class TestGrandparentChain:
    """Grandparent rule: gp(X,Z) :- parent(X,Y), parent(Y,Z).

    KB:
        parent(alice=1, bob=2)
        parent(bob=2, charlie=3)

    Query: gp(alice, ?Z) → should find grounding with Z=charlie.
    """

    @pytest.fixture
    def grounder(self):
        facts = torch.tensor([[1, 1, 2], [1, 2, 3]], dtype=torch.long)
        heads = torch.tensor([[2, 4, 5]], dtype=torch.long)  # gp(V0, V1)
        bodies = torch.tensor([[[1, 4, 6], [1, 6, 5]]], dtype=torch.long)  # parent(V0,V2), parent(V2,V1)
        rule_lens = torch.tensor([2], dtype=torch.long)
        return BCGrounder(
            facts_idx=facts,
            rules_heads_idx=heads,
            rules_bodies_idx=bodies,
            rule_lens=rule_lens,
            constant_no=3,
            padding_idx=99,
            device=DEVICE,
            predicate_no=3,
            resolution='sld',
            filter='none',
            max_goals=5,
            depth=3,
            max_total_groundings=8,
            fact_index_type='arg_key',
        )

    def test_finds_grounding(self, grounder):
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)  # gp(alice, V0)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() >= 1, "Should find at least one grounding"

    def test_grounding_body_is_correct(self, grounder):
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        valid_idx = result.mask[0].nonzero(as_tuple=True)[0]
        assert len(valid_idx) > 0
        body = result.body[0, valid_idx[0]]
        # Body should be parent(alice,bob), parent(bob,charlie)
        body_list = body.tolist()
        assert [1, 1, 2] in body_list, f"Expected parent(alice,bob) in body, got {body_list}"
        assert [1, 2, 3] in body_list, f"Expected parent(bob,charlie) in body, got {body_list}"

    def test_no_grounding_for_invalid_query(self, grounder):
        queries = torch.tensor([[2, 3, 4]], dtype=torch.long)  # gp(charlie, V0) - no matching chain
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() == 0

    def test_masked_query_not_processed(self, grounder):
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([False])  # masked out
        result = grounder(queries, query_mask)
        assert result.count[0].item() == 0


class TestSingleBodyRule:
    """Single-body rule: derived(X,Y) :- base(X,Y).

    Depth 2 should suffice: d0 matches rule, d1 resolves the single body atom.
    """

    @pytest.fixture
    def grounder(self):
        facts = torch.tensor([
            [1, 1, 2],  # base(a, b)
            [1, 2, 3],  # base(b, c)
            [1, 3, 1],  # base(c, a)
        ], dtype=torch.long)
        heads = torch.tensor([[2, 4, 5]], dtype=torch.long)  # derived(V0, V1)
        bodies = torch.tensor([[[1, 4, 5]]], dtype=torch.long)  # base(V0, V1)
        rule_lens = torch.tensor([1], dtype=torch.long)
        return BCGrounder(
            facts_idx=facts,
            rules_heads_idx=heads,
            rules_bodies_idx=bodies,
            rule_lens=rule_lens,
            constant_no=3,
            padding_idx=99,
            device=DEVICE,
            predicate_no=3,
            resolution='sld',
            filter='none',
            max_goals=4,
            depth=2,
            max_total_groundings=8,
            fact_index_type='arg_key',
        )

    def test_finds_grounding_depth2(self, grounder):
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)  # derived(a, V0)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() >= 1

    def test_body_matches_fact(self, grounder):
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        valid_idx = result.mask[0].nonzero(as_tuple=True)[0]
        body = result.body[0, valid_idx[0]]
        assert [1, 1, 2] in body.tolist(), "Body should contain base(a, b)"


class TestBatchQueries:
    """Test that batched queries work correctly."""

    def test_batch_of_two(self):
        facts = torch.tensor([
            [1, 1, 2],
            [1, 2, 3],
        ], dtype=torch.long)
        heads = torch.tensor([[2, 4, 5]], dtype=torch.long)
        bodies = torch.tensor([[[1, 4, 5]]], dtype=torch.long)
        rule_lens = torch.tensor([1], dtype=torch.long)
        grounder = BCGrounder(
            facts_idx=facts,
            rules_heads_idx=heads,
            rules_bodies_idx=bodies,
            rule_lens=rule_lens,
            constant_no=3,
            padding_idx=99,
            device=DEVICE,
            predicate_no=3,
            resolution='sld',
            filter='none',
            max_goals=4,
            depth=2,
            max_total_groundings=8,
            fact_index_type='arg_key',
        )
        # Two queries: derived(a, V0) and derived(b, V0)
        queries = torch.tensor([[2, 1, 4], [2, 2, 4]], dtype=torch.long)
        query_mask = torch.tensor([True, True])
        result = grounder(queries, query_mask)
        # Both should find groundings
        assert result.count[0].item() >= 1, "Query 0 should find grounding"
        assert result.count[1].item() >= 1, "Query 1 should find grounding"


class TestMultipleRules:
    """KB with multiple rules sharing a predicate."""

    def test_two_rules_same_head(self):
        facts = torch.tensor([
            [1, 1, 2],  # r1(a, b)
            [2, 1, 3],  # r2(a, c)
        ], dtype=torch.long)
        # Two rules: h(X,Y) :- r1(X,Y) and h(X,Y) :- r2(X,Y)
        heads = torch.tensor([[3, 4, 5], [3, 4, 5]], dtype=torch.long)
        bodies = torch.tensor([
            [[1, 4, 5]],  # r1(V0, V1)
            [[2, 4, 5]],  # r2(V0, V1)
        ], dtype=torch.long)
        rule_lens = torch.tensor([1, 1], dtype=torch.long)
        grounder = BCGrounder(
            facts_idx=facts,
            rules_heads_idx=heads,
            rules_bodies_idx=bodies,
            rule_lens=rule_lens,
            constant_no=3,
            padding_idx=99,
            device=DEVICE,
            predicate_no=4,
            resolution='sld',
            filter='none',
            max_goals=4,
            depth=2,
            max_total_groundings=8,
            fact_index_type='arg_key',
        )
        queries = torch.tensor([[3, 1, 4]], dtype=torch.long)  # h(a, V0)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        # Should find 2 groundings: via r1 and via r2
        assert result.count[0].item() == 2, \
            f"Expected 2 groundings, got {result.count[0].item()}"


class TestNoRules:
    """Edge case: no rules in KB."""

    def test_no_rules(self):
        facts = torch.tensor([[1, 1, 2]], dtype=torch.long)
        heads = torch.empty(0, 3, dtype=torch.long)
        bodies = torch.empty(0, 1, 3, dtype=torch.long)
        rule_lens = torch.empty(0, dtype=torch.long)
        grounder = BCGrounder(
            facts_idx=facts,
            rules_heads_idx=heads,
            rules_bodies_idx=bodies,
            rule_lens=rule_lens,
            constant_no=3,
            padding_idx=99,
            device=DEVICE,
            predicate_no=2,
            resolution='sld',
            filter='none',
            max_goals=4,
            depth=2,
            max_total_groundings=8,
            fact_index_type='arg_key',
        )
        queries = torch.tensor([[1, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        # No rules → no groundings
        assert result.count[0].item() == 0
