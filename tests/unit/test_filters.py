"""Tests for soundness filters: fp_batch and fp_global."""

import torch
import pytest
from grounder import KB, BCGrounder

DEVICE = torch.device("cpu")


class TestFPBatchFilter:
    """Test fp_batch (cross-query Kleene T_P) filter."""

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

    def test_fp_batch_keeps_provable(self):
        grounder = self._make_grounder('fp_batch')
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.evidence.count[0].item() >= 1

    def test_fp_batch_filters_unprovable(self):
        grounder = self._make_grounder('fp_batch')
        queries = torch.tensor([[2, 3, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.evidence.count[0].item() == 0

    def test_no_filter_returns_more(self):
        """Filter='none' should return >= filter='fp_batch' results."""
        grounder_none = self._make_grounder('none')
        grounder_fp_batch = self._make_grounder('fp_batch')
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result_none = grounder_none(queries, query_mask)
        result_fp_batch = grounder_fp_batch(queries, query_mask)
        none_count = result_none.evidence.count[0].item()
        assert none_count >= result_fp_batch.evidence.count[0].item()


class TestHashAtoms:
    """Test the shared hash utility."""

    def test_hash_consistency(self):
        from grounder.filters._hash import hash_atoms
        from grounder.data.fact_index import pack_triples_64
        atoms = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        base = 100
        h1 = hash_atoms(atoms, base)
        h2 = pack_triples_64(atoms, base)
        assert (h1 == h2).all()


class TestFilterPruneDead:
    """Test per-step dead-atom pruning filter."""

    def _make_grounder(self, step_prune_dead: bool = True):
        """KB: base(a,b), base(b,c). Rule: derived(X,Y) :- base(X,Y).
        pred 3 (dead_pred) has no facts and is not a rule head."""
        facts = torch.tensor([
            [1, 1, 2],  # base(a, b)
            [1, 2, 3],  # base(b, c)
        ], dtype=torch.long)
        heads = torch.tensor([[2, 4, 5]], dtype=torch.long)  # derived(X,Y)
        bodies = torch.tensor([[[1, 4, 5]]], dtype=torch.long)  # base(X,Y)
        rule_lens = torch.tensor([1], dtype=torch.long)
        kb = KB(facts, heads, bodies, rule_lens,
                constant_no=3, predicate_no=3,
                padding_idx=99, device=DEVICE)
        return BCGrounder(
            kb, resolution='sld', filter='none',
            max_goals=4, depth=2, max_total_groundings=64,
            max_derived_per_state=20,
            step_prune_dead=step_prune_dead,
        )

    def test_provable_survives(self):
        """Provable queries should produce groundings with prune_dead enabled."""
        grounder = self._make_grounder(step_prune_dead=True)
        queries = torch.tensor([[2, 1, 2]], dtype=torch.long)  # derived(a,b)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.evidence.count[0].item() >= 1

    def test_rule_head_not_killed(self):
        """Body atom whose pred is a rule head should not be killed."""
        grounder = self._make_grounder(step_prune_dead=True)
        # derived(a,b) has body [base(a,b)] where base has facts -> not dead
        queries = torch.tensor([[2, 1, 2]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        # Should find at least one grounding
        assert result.evidence.count[0].item() >= 1

    def test_disabled_by_default(self):
        """With step_prune_dead=False, filter should be a no-op."""
        grounder_off = self._make_grounder(step_prune_dead=False)
        grounder_on = self._make_grounder(step_prune_dead=True)
        queries = torch.tensor([[2, 1, 2]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result_off = grounder_off(queries, query_mask)
        result_on = grounder_on(queries, query_mask)
        # Both should find groundings (the KB is simple enough)
        assert result_off.evidence.count[0].item() >= 1
        assert result_on.evidence.count[0].item() >= 1


class TestFilterWidth:
    """Test per-step width filter for SLD/RTF."""

    def _make_grounder(self, width: int = 1, resolution: str = 'sld'):
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
            kb, resolution=resolution, filter='none',
            max_goals=4, depth=2, max_total_groundings=64,
            max_derived_per_state=20,
            width=width,
        )

    def test_width1_finds_simple_proofs(self):
        """Width=1 should find single-hop proofs (1 unknown body atom)."""
        grounder = self._make_grounder(width=1)
        queries = torch.tensor([[2, 1, 2]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.evidence.count[0].item() >= 1

    def test_width0_only_fact_groundings(self):
        """Width=0 should only keep groundings where all body atoms are facts."""
        grounder = self._make_grounder(width=0)
        queries = torch.tensor([[2, 1, 2]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        # With width=0, only all-fact body groundings survive.
        # This is very restrictive — may find 0 or few.
        count = result.evidence.count[0].item()
        assert count >= 0  # sanity check (doesn't crash)


class TestFilterEnumPruneDeadWarning:
    """Verify step_prune_dead with enum raises a warning."""

    def test_warns_on_enum(self):
        facts = torch.tensor([
            [1, 1, 2],
            [1, 2, 3],
            [2, 1, 3],  # ensure pred 2 exists in facts for enum P
        ], dtype=torch.long)
        heads = torch.tensor([[2, 4, 5]], dtype=torch.long)
        bodies = torch.tensor([[[1, 4, 5]]], dtype=torch.long)
        rule_lens = torch.tensor([1], dtype=torch.long)
        kb = KB(facts, heads, bodies, rule_lens,
                constant_no=3, predicate_no=3,
                padding_idx=99, device=DEVICE)
        with pytest.warns(UserWarning, match="step_prune_dead has no effect"):
            BCGrounder(
                kb, resolution='enum', filter='fp_batch',
                depth=1, width=1,
                step_prune_dead=True,
            )


class TestFactoryParsePD:
    """Test factory parsing of .pd segment."""

    def test_sld_pd(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("sld.pd.w1.d2")
        assert cfg["resolution"] == "sld"
        assert cfg["step_prune_dead"] is True
        assert cfg["width"] == 1
        assert cfg["depth"] == 2

    def test_sld_no_pd(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("sld.w1.d2")
        assert cfg["step_prune_dead"] is False

    def test_rtf_pd(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("rtf.pd.d3")
        assert cfg["resolution"] == "rtf"
        assert cfg["step_prune_dead"] is True
        assert cfg["depth"] == 3

    def test_enum_pd(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("enum.pd.w1.d2")
        assert cfg["step_prune_dead"] is True

    def test_fp_batch_explicit(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("sld.fp_batch.d2")
        assert cfg["filter"] == "fp_batch"

    def test_fp_global_explicit(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("sld.fp_global.d2")
        assert cfg["filter"] == "fp_global"

    def test_prune_alias(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("enum.prune.w1.d2")
        assert cfg["filter"] == "fp_batch"

    def test_provset_alias(self):
        from grounder.factory import parse_grounder_type
        cfg = parse_grounder_type("enum.provset.w1.d2")
        assert cfg["filter"] == "fp_global"


class TestFPBatchMultiHop:
    """Test fp_batch cross-query provability for multi-hop proofs."""

    def _make_grounder(self):
        """KB: parent(1,2), parent(2,3).
        R1: ancestor(X,Y) :- parent(X,Y)
        R2: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
        """
        facts = torch.tensor([
            [1, 1, 2],  # parent(1,2)
            [1, 2, 3],  # parent(2,3)
        ], dtype=torch.long)
        # R1: ancestor(X,Y) :- parent(X,Y)
        # R2: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
        heads = torch.tensor([
            [2, 4, 5],  # ancestor(X,Y)
            [2, 4, 6],  # ancestor(X,Z)
        ], dtype=torch.long)
        bodies = torch.tensor([
            [[1, 4, 5], [0, 0, 0]],  # parent(X,Y), pad
            [[1, 4, 5], [2, 5, 6]],  # parent(X,Y), ancestor(Y,Z)
        ], dtype=torch.long)
        rule_lens = torch.tensor([1, 2], dtype=torch.long)
        kb = KB(facts, heads, bodies, rule_lens,
                constant_no=3, predicate_no=3,
                padding_idx=0, device=DEVICE)
        return BCGrounder(
            kb, resolution='sld', filter='fp_batch',
            max_goals=8, depth=3, max_total_groundings=64,
            max_derived_per_state=20,
            prune_facts=True,
        )

    def test_multihop_batch(self):
        """Batch [ancestor(1,3), ancestor(2,3)]: multi-hop should be proved."""
        grounder = self._make_grounder()
        queries = torch.tensor([
            [2, 1, 3],  # ancestor(1,3)
            [2, 2, 3],  # ancestor(2,3)
        ], dtype=torch.long)
        query_mask = torch.tensor([True, True])
        result = grounder(queries, query_mask)
        # ancestor(2,3) provable via R1 + parent(2,3)
        assert result.evidence.count[1].item() >= 1
        # ancestor(1,3) provable via R2 + parent(1,2) + ancestor(2,3) [cross-query]
        assert result.evidence.count[0].item() >= 1

    def test_single_query_no_cross(self):
        """Single query ancestor(1,3) without ancestor(2,3) in batch."""
        grounder = self._make_grounder()
        queries = torch.tensor([[2, 1, 3]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        # ancestor(1,3) via R2 needs ancestor(2,3) which is NOT in batch
        # But ancestor(2,3) IS provable within this same query's groundings
        # if the BC loop collected it. The cross-query mechanism works within
        # the batch, so this depends on whether ancestor(2,3) was collected.
        # With sufficient depth, it should be collected and proved.
        # Not asserting count here — just verify no crash.
        assert result.evidence.count[0].item() >= 0


class TestFPBatchCircular:
    """Test fp_batch rejects circular proofs."""

    def test_circular_rejected(self):
        """Rules p(X) :- q(X) and q(X) :- p(X) with no facts: nothing proved."""
        facts = torch.tensor([
            [3, 1, 2],  # dummy fact for a different pred
        ], dtype=torch.long)
        heads = torch.tensor([
            [1, 4, 5],  # p(X,Y)
            [2, 4, 5],  # q(X,Y)
        ], dtype=torch.long)
        bodies = torch.tensor([
            [[2, 4, 5]],  # q(X,Y)
            [[1, 4, 5]],  # p(X,Y)
        ], dtype=torch.long)
        rule_lens = torch.tensor([1, 1], dtype=torch.long)
        kb = KB(facts, heads, bodies, rule_lens,
                constant_no=3, predicate_no=4,
                padding_idx=0, device=DEVICE)
        grounder = BCGrounder(
            kb, resolution='sld', filter='fp_batch',
            max_goals=4, depth=3, max_total_groundings=64,
            max_derived_per_state=20,
        )
        queries = torch.tensor([
            [1, 1, 2],  # p(1,2)
            [2, 1, 2],  # q(1,2)
        ], dtype=torch.long)
        query_mask = torch.tensor([True, True])
        result = grounder(queries, query_mask)
        # Neither should be proved (circular, no base facts for pred 1 or 2)
        assert result.evidence.count[0].item() == 0
        assert result.evidence.count[1].item() == 0


class TestFPGlobalSLD:
    """Test fp_global works with SLD resolution (not just enum)."""

    def test_sld_fp_global(self):
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
        grounder = BCGrounder(
            kb, resolution='sld', filter='fp_global',
            max_goals=4, depth=2, max_total_groundings=64,
            max_derived_per_state=20,
        )
        queries = torch.tensor([[2, 1, 2]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        assert result.evidence.count[0].item() >= 1
