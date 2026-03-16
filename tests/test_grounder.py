"""Tests for BCGrounder end-to-end (SLD resolution)."""

import torch
import pytest
from grounder import KB, BCGrounder

DEVICE = torch.device("cpu")

# Padding facts to ensure K_f >= K_f_min_budget (10) in init_mgu.
_PAD_FACTS = torch.tensor(
    [[1, 10, i] for i in range(11, 23)], dtype=torch.long
)


def _make_grounder(facts, heads, bodies, rule_lens, *,
                   predicate_no, **bc_kwargs):
    """Helper: build KB + BCGrounder with test defaults."""
    kb = KB(facts, heads, bodies, rule_lens,
            constant_no=23, predicate_no=predicate_no,
            padding_idx=99, device=DEVICE,
            fact_index_type='arg_key')
    defaults = dict(resolution='sld', filter='fp_batch',
                    max_goals=4, depth=2, max_total_groundings=16)
    defaults.update(bc_kwargs)
    return BCGrounder(kb, **defaults)


class TestGrandparentChain:
    """gp(X,Z) :- parent(X,Y), parent(Y,Z)."""

    @pytest.fixture
    def grounder(self):
        facts = torch.cat([
            torch.tensor([[1, 1, 2], [1, 2, 3]], dtype=torch.long),
            _PAD_FACTS,
        ])
        heads = torch.tensor([[2, 24, 25]], dtype=torch.long)
        bodies = torch.tensor([[[1, 24, 26], [1, 26, 25]]], dtype=torch.long)
        rule_lens = torch.tensor([2], dtype=torch.long)
        return _make_grounder(facts, heads, bodies, rule_lens,
                              predicate_no=3, max_goals=5, depth=3,
                              max_total_groundings=8)

    def test_finds_grounding(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() >= 1

    def test_grounding_body_is_correct(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        valid_idx = result.mask[0].nonzero(as_tuple=True)[0]
        assert len(valid_idx) > 0
        body = result.body[0, valid_idx[0]]
        body_list = body.tolist()
        assert [1, 1, 2] in body_list
        assert [1, 2, 3] in body_list

    def test_no_grounding_for_invalid_query(self, grounder):
        queries = torch.tensor([[2, 3, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() == 0

    def test_masked_query_not_processed(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([False])
        result = grounder(queries, query_mask)
        assert result.count[0].item() == 0


class TestSingleBodyRule:
    """derived(X,Y) :- base(X,Y)."""

    @pytest.fixture
    def grounder(self):
        facts = torch.cat([
            torch.tensor([[1, 1, 2], [1, 2, 3], [1, 3, 1]], dtype=torch.long),
            _PAD_FACTS,
        ])
        heads = torch.tensor([[2, 24, 25]], dtype=torch.long)
        bodies = torch.tensor([[[1, 24, 25]]], dtype=torch.long)
        rule_lens = torch.tensor([1], dtype=torch.long)
        return _make_grounder(facts, heads, bodies, rule_lens,
                              predicate_no=3)

    def test_finds_grounding_depth2(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() >= 1

    def test_body_matches_fact(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        valid_idx = result.mask[0].nonzero(as_tuple=True)[0]
        body = result.body[0, valid_idx[0]]
        assert [1, 1, 2] in body.tolist()


class TestBatchQueries:
    def test_batch_of_two(self):
        facts = torch.cat([
            torch.tensor([[1, 1, 2], [1, 2, 3]], dtype=torch.long),
            _PAD_FACTS,
        ])
        heads = torch.tensor([[2, 24, 25]], dtype=torch.long)
        bodies = torch.tensor([[[1, 24, 25]]], dtype=torch.long)
        rule_lens = torch.tensor([1], dtype=torch.long)
        grounder = _make_grounder(facts, heads, bodies, rule_lens,
                                  predicate_no=3)
        queries = torch.tensor([[2, 1, 24], [2, 2, 24]], dtype=torch.long)
        query_mask = torch.tensor([True, True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() >= 1
        assert result.count[1].item() >= 1


class TestMultipleRules:
    def test_two_rules_same_head(self):
        facts = torch.cat([
            torch.tensor([[1, 1, 2], [2, 1, 3]], dtype=torch.long),
            _PAD_FACTS,
        ])
        heads = torch.tensor([[3, 24, 25], [3, 24, 25]], dtype=torch.long)
        bodies = torch.tensor([[[1, 24, 25]], [[2, 24, 25]]], dtype=torch.long)
        rule_lens = torch.tensor([1, 1], dtype=torch.long)
        grounder = _make_grounder(facts, heads, bodies, rule_lens,
                                  predicate_no=4)
        queries = torch.tensor([[3, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.count[0].item() == 2


class TestNoRules:
    """Empty KB must raise ValueError."""

    def test_no_rules_raises(self):
        facts = torch.cat([
            torch.tensor([[1, 1, 2]], dtype=torch.long),
            _PAD_FACTS,
        ])
        heads = torch.empty(0, 3, dtype=torch.long)
        bodies = torch.empty(0, 1, 3, dtype=torch.long)
        rule_lens = torch.empty(0, dtype=torch.long)
        with pytest.raises(ValueError, match="rules_heads_idx is empty"):
            KB(facts, heads, bodies, rule_lens,
               constant_no=23, predicate_no=2,
               padding_idx=99, device=DEVICE)
