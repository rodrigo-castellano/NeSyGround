"""Tests for the rule-evidence entry point ``BCGrounder.run_bc``.

``run_bc`` is the entry point for SBR/DCR/R2N pool-iter consumers. It
returns a :class:`grounder.types.RuleGroundings` whose ``atom_table``
covers every query atom and whose ``query_pool_idx`` points each query
to its slot in the pool — even when no grounding produced that atom as
a head.

Companion to :func:`grounder.groundings.populate_query_pool_idx`, which
is the underlying helper.
"""
from __future__ import annotations

import pytest
import torch

from grounder import KB, BCGrounder
from grounder.groundings import atom_hash, populate_query_pool_idx
from grounder.types import RuleGroundings


DEVICE = torch.device("cpu")
PAD = 99

# Padding facts to ensure K_f >= 10 in init_mgu (matches test_grounder.py).
_PAD_FACTS = torch.tensor(
    [[1, 10, i] for i in range(11, 23)], dtype=torch.long
)


def _make_grandparent_grounder():
    """gp(X,Z) :- parent(X,Y), parent(Y,Z) on a small KB."""
    facts = torch.cat([
        torch.tensor([[1, 1, 2], [1, 2, 3]], dtype=torch.long),
        _PAD_FACTS,
    ])
    heads = torch.tensor([[2, 24, 25]], dtype=torch.long)   # gp(X, Z)
    bodies = torch.tensor(
        [[[1, 24, 26], [1, 26, 25]]], dtype=torch.long)     # parent(X,Y), parent(Y,Z)
    rule_lens = torch.tensor([2], dtype=torch.long)
    kb = KB(facts, heads, bodies, rule_lens,
            constant_no=23, predicate_no=3,
            padding_idx=PAD, device=DEVICE,
            fact_index_type='arg_key')
    return BCGrounder(
        kb, resolution='sld', filter='fp_batch',
        max_goals=5, depth=3, max_total_groundings=8,
        prune_facts=True,
        # collect_rule_groundings doesn't matter — run_bc forces it on.
    )


# ─────────────────────────────────────────────────────────────────────
# populate_query_pool_idx — unit tests on the helper
# ─────────────────────────────────────────────────────────────────────

class TestPopulateQueryPoolIdx:

    def test_query_already_in_pool_is_found_at_existing_slot(self):
        # Pool contains atom (2, 1, 3). Query at the same atom — should
        # land at slot 0 (the existing atom_table slot), not be appended.
        atom_table = torch.tensor([[2, 1, 3], [1, 1, 2]], dtype=torch.long)
        rg = RuleGroundings(
            atom_table=atom_table, A_in={}, A_out={},
            num_atoms=2, num_rules=1)
        queries = torch.tensor([[2, 1, 3]], dtype=torch.long)
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        assert out.atom_table.shape[0] == 2  # not extended
        assert out.query_pool_idx.shape == (1,)
        assert out.atom_table[out.query_pool_idx[0]].tolist() == [2, 1, 3]

    def test_novel_query_extends_pool(self):
        atom_table = torch.tensor([[1, 1, 2]], dtype=torch.long)
        rg = RuleGroundings(
            atom_table=atom_table, A_in={}, A_out={},
            num_atoms=1, num_rules=1)
        queries = torch.tensor([[2, 5, 7]], dtype=torch.long)   # novel
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        assert out.atom_table.shape[0] == 2  # extended by 1
        # Existing slot 0 unchanged so A_in/A_out remain valid.
        assert out.atom_table[0].tolist() == [1, 1, 2]
        # Query landed at the new slot.
        assert out.atom_table[out.query_pool_idx[0]].tolist() == [2, 5, 7]

    def test_mixed_existing_and_novel_queries(self):
        atom_table = torch.tensor(
            [[1, 1, 2], [2, 3, 4]], dtype=torch.long)
        rg = RuleGroundings(
            atom_table=atom_table, A_in={}, A_out={},
            num_atoms=2, num_rules=1)
        queries = torch.tensor(
            [[2, 3, 4], [5, 7, 8], [1, 1, 2]], dtype=torch.long)
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        # Pool grew by exactly one (the novel (5, 7, 8)).
        assert out.atom_table.shape[0] == 3
        # Each query maps to its correct slot.
        gathered = out.atom_table[out.query_pool_idx]
        assert torch.equal(gathered, queries)

    def test_duplicate_novel_queries_dedup_to_one_slot(self):
        atom_table = torch.tensor([[1, 1, 2]], dtype=torch.long)
        rg = RuleGroundings(
            atom_table=atom_table, A_in={}, A_out={},
            num_atoms=1, num_rules=1)
        queries = torch.tensor(
            [[5, 7, 8], [5, 7, 8]], dtype=torch.long)        # same atom twice
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        # Pool extended by ONE, not two — duplicates dedupe.
        assert out.atom_table.shape[0] == 2
        # Both queries point to the same slot.
        assert out.query_pool_idx[0].item() == out.query_pool_idx[1].item()

    def test_empty_pool_built_from_queries_only(self):
        atom_table = torch.zeros(0, 3, dtype=torch.long)
        rg = RuleGroundings(
            atom_table=atom_table, A_in={}, A_out={},
            num_atoms=0, num_rules=1)
        queries = torch.tensor([[2, 5, 7], [3, 8, 9]], dtype=torch.long)
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        # Pool now contains the padding sentinel + the unique queries.
        # gathered atoms must match.
        gathered = out.atom_table[out.query_pool_idx]
        assert torch.equal(gathered, queries)

    def test_rejects_bad_query_shape(self):
        rg = RuleGroundings(
            atom_table=torch.zeros(0, 3, dtype=torch.long),
            A_in={}, A_out={}, num_atoms=0, num_rules=1)
        with pytest.raises(ValueError, match=r"\[B, 3\]"):
            populate_query_pool_idx(
                rg, torch.zeros(2, dtype=torch.long), padding_idx=PAD)


# ─────────────────────────────────────────────────────────────────────
# BCGrounder.run_bc — end-to-end
# ─────────────────────────────────────────────────────────────────────

class TestRunBC:

    def test_returns_rule_groundings_with_query_pool_idx(self):
        grounder = _make_grandparent_grounder()
        queries = torch.tensor([[2, 1, 24]], dtype=torch.long)   # gp(1, ?)
        query_mask = torch.tensor([True])
        rg = grounder.run_bc(queries, query_mask)

        assert isinstance(rg, RuleGroundings)
        assert rg.query_pool_idx is not None
        assert rg.query_pool_idx.shape == (1,)
        # Query atom must be findable in the augmented pool.
        gathered = rg.atom_table[rg.query_pool_idx]
        assert torch.equal(gathered, queries)

    def test_query_atom_in_pool_when_grounded(self):
        # Query gp(1, 3) — provable: parent(1,2) + parent(2,3) → gp(1,3).
        # The grounder produces gp(1, 3) as a head atom, so it's
        # already in atom_table; query_pool_idx points there.
        grounder = _make_grandparent_grounder()
        queries = torch.tensor([[2, 1, 3]], dtype=torch.long)
        query_mask = torch.tensor([True])
        rg = grounder.run_bc(queries, query_mask)

        # The atom_table must contain (2, 1, 3) somewhere.
        target_h = atom_hash(queries)
        pool_h = atom_hash(rg.atom_table)
        assert (pool_h == target_h.item()).any()
        # And query_pool_idx points at it.
        assert rg.atom_table[rg.query_pool_idx[0]].tolist() == [2, 1, 3]

    def test_unprovable_query_still_has_pool_slot(self):
        # gp(3, 1) is unprovable in this KB (no parent chain from 3 to 1).
        # The grounder produces NO rule groundings for it, so the head
        # atom isn't in atom_table from the per-tree pipeline. run_bc
        # must still extend the pool so the query has a slot.
        grounder = _make_grandparent_grounder()
        queries = torch.tensor([[2, 3, 1]], dtype=torch.long)
        query_mask = torch.tensor([True])
        rg = grounder.run_bc(queries, query_mask)

        assert rg.query_pool_idx is not None
        assert rg.query_pool_idx.shape == (1,)
        # The query atom is now in the pool exactly once.
        assert rg.atom_table[rg.query_pool_idx[0]].tolist() == [2, 3, 1]

    def test_a_in_a_out_indices_stable_under_pool_extension(self):
        # When ``populate_query_pool_idx`` extends an existing
        # ``RuleGroundings`` with novel query atoms, the existing
        # A_in/A_out tensors must still be valid: their indices
        # point into slots 0..N_orig-1 of the original pool, and
        # those slots remain stable under append-only extension.
        # Build the input ``RuleGroundings`` via the canonical
        # ``evidence_to_rule_groundings`` path so we get real firings
        # to validate the invariant against.
        from grounder.groundings import evidence_to_rule_groundings

        grounder = _make_grandparent_grounder()
        # Just the provable query — produces one firing of the rule.
        queries_grounded = torch.tensor([[2, 1, 3]], dtype=torch.long)
        out = grounder(queries_grounded, torch.tensor([True]))
        rg_pre = evidence_to_rule_groundings(
            out.evidence, padding_idx=PAD, num_rules=1)
        assert any(A_in_r.numel() > 0 for A_in_r in rg_pre.A_in.values()), (
            "evidence_to_rule_groundings produced no firings — test setup is wrong")

        # Snapshot pre-extension A_in/A_out and atom_table.
        pre_atom_table = rg_pre.atom_table.clone()
        pre_A_in = {r: t.clone() for r, t in rg_pre.A_in.items()}
        pre_A_out = {r: t.clone() for r, t in rg_pre.A_out.items()}

        # Now extend with a novel (unprovable) query atom.
        novel_query = torch.tensor([[2, 99, 99]], dtype=torch.long)
        rg_post = populate_query_pool_idx(rg_pre, novel_query, padding_idx=PAD)

        # The new atom_table is a strict prefix-extension of the old.
        assert rg_post.atom_table.shape[0] == pre_atom_table.shape[0] + 1
        assert torch.equal(
            rg_post.atom_table[: pre_atom_table.shape[0]], pre_atom_table)

        # A_in/A_out unchanged (same indices into the prefix).
        for r, A_in_r in rg_post.A_in.items():
            assert torch.equal(A_in_r, pre_A_in[r])
            # And every index stays in range — including for the
            # extended pool size.
            if A_in_r.numel():
                assert int(A_in_r.max().item()) < rg_post.atom_table.shape[0]
        for r, A_out_r in rg_post.A_out.items():
            assert torch.equal(A_out_r, pre_A_out[r])

    def test_collect_flag_state_restored_after_call(self):
        # run_bc forces collect_rule_groundings=True for the duration of
        # the call. Verify the original setting is restored afterwards.
        grounder = _make_grandparent_grounder()
        original = grounder._collect_rule_groundings
        queries = torch.tensor([[2, 1, 3]], dtype=torch.long)
        query_mask = torch.tensor([True])
        _ = grounder.run_bc(queries, query_mask)
        assert grounder._collect_rule_groundings == original
