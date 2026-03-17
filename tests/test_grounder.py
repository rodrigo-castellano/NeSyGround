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
        output = grounder(queries, query_mask)
        assert output.evidence.count[0].item() >= 1

    def test_grounding_body_is_correct(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        output = grounder(queries, query_mask)
        valid_idx = output.evidence.mask[0].nonzero(as_tuple=True)[0]
        assert len(valid_idx) > 0
        body = output.evidence.body[0, valid_idx[0]]
        body_list = body.tolist()
        assert [1, 1, 2] in body_list
        assert [1, 2, 3] in body_list

    def test_no_grounding_for_invalid_query(self, grounder):
        queries = torch.tensor([[2, 3, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        output = grounder(queries, query_mask)
        assert output.evidence.count[0].item() == 0

    def test_masked_query_not_processed(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([False])
        output = grounder(queries, query_mask)
        assert output.evidence.count[0].item() == 0


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
        output = grounder(queries, query_mask)
        assert output.evidence.count[0].item() >= 1

    def test_body_matches_fact(self, grounder):
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)
        query_mask = torch.tensor([True])
        output = grounder(queries, query_mask)
        valid_idx = output.evidence.mask[0].nonzero(as_tuple=True)[0]
        body = output.evidence.body[0, valid_idx[0]]
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
        output = grounder(queries, query_mask)
        assert output.evidence.count[0].item() >= 1
        assert output.evidence.count[1].item() >= 1


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
        output = grounder(queries, query_mask)
        assert output.evidence.count[0].item() == 2


class TestDepth2BodyAccumulation:
    """Depth-2 proof: body atoms from BOTH rule applications must be recorded.

    Setup:
      Rule 1: Q(X,Y) :- mid(X,Y)           [pred 3 :- pred 2]
      Rule 2: mid(X,Y) :- base(X,Z), base(Z,Y)  [pred 2 :- pred 1, pred 1]
      Facts:  base(1,2), base(2,3)          [pred 1]

    Query Q(1,3):
      depth 0 — match Rule 1 → body = [mid(1,3)]
      depth 1 — match Rule 2 on mid(1,3) → body = [base(1,2), base(2,3)]
      depth 2 — base facts resolved → PROVED

    grounding_body should contain atoms from BOTH rules:
      [mid(1,3), base(1,2), base(2,3)]   (3 atoms, body_capacity = depth * M = 3 * 2 = 6)
    """

    @pytest.fixture
    def grounder(self):
        facts = torch.cat([
            torch.tensor([[1, 1, 2], [1, 2, 3]], dtype=torch.long),
            _PAD_FACTS,
        ])
        # Rule 1: Q(X,Y) :- mid(X,Y)
        # Rule 2: mid(X,Y) :- base(X,Z), base(Z,Y)
        heads = torch.tensor([
            [3, 24, 25],   # Q(X,Y)
            [2, 24, 25],   # mid(X,Y)
        ], dtype=torch.long)
        bodies = torch.tensor([
            [[2, 24, 25], [99, 99, 99]],   # mid(X,Y), pad
            [[1, 24, 26], [1, 26, 25]],     # base(X,Z), base(Z,Y)
        ], dtype=torch.long)
        rule_lens = torch.tensor([1, 2], dtype=torch.long)
        # depth=3 to give enough room; max_goals=5 for G >= 3
        # filter='none': accumulated body includes intermediate atoms (e.g.
        # mid(1,3)) which are not base facts. fp_batch cannot verify them
        # because it only sees query-level heads. The body accumulation
        # feature stores proof evidence for the reasoner — soundness is
        # guaranteed by the proof itself (all goals resolved to facts).
        return _make_grounder(facts, heads, bodies, rule_lens,
                              predicate_no=4, max_goals=5, depth=3,
                              max_total_groundings=16,
                              filter='none')

    def test_finds_grounding(self, grounder):
        """Q(1,3) should be provable via the 2-rule chain."""
        queries = torch.tensor([[3, 1, 3]], dtype=torch.long)
        query_mask = torch.tensor([True])
        output = grounder(queries, query_mask)
        assert output.evidence.count[0].item() >= 1

    def test_body_accumulates_across_depths(self, grounder):
        """grounding_body must contain body atoms from BOTH rule applications.

        Rule 1 contributes: mid(1,3) = [2, 1, 3]
        Rule 2 contributes: base(1,2) = [1, 1, 2], base(2,3) = [1, 2, 3]
        """
        queries = torch.tensor([[3, 1, 3]], dtype=torch.long)
        query_mask = torch.tensor([True])
        output = grounder(queries, query_mask)

        valid_idx = output.evidence.mask[0].nonzero(as_tuple=True)[0]
        assert len(valid_idx) > 0, "Expected at least one valid grounding"

        body = output.evidence.body[0, valid_idx[0]]
        body_list = body.tolist()

        # Body atoms from Rule 1 (depth 0 application)
        assert [2, 1, 3] in body_list, (
            f"mid(1,3) missing from grounding body: {body_list}")

        # Body atoms from Rule 2 (depth 1 application)
        assert [1, 1, 2] in body_list, (
            f"base(1,2) missing from grounding body: {body_list}")
        assert [1, 2, 3] in body_list, (
            f"base(2,3) missing from grounding body: {body_list}")


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
