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
from grounder.factory import make_bcwd
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
        rg = RuleGroundings.empty(num_rules=1)
        rg.atom_table = atom_table
        rg.num_atoms = 2
        queries = torch.tensor([[2, 1, 3]], dtype=torch.long)
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        assert out.atom_table.shape[0] == 2  # not extended
        assert out.query_pool_idx.shape == (1,)
        assert out.atom_table[out.query_pool_idx[0]].tolist() == [2, 1, 3]

    def test_novel_query_extends_pool(self):
        atom_table = torch.tensor([[1, 1, 2]], dtype=torch.long)
        rg = RuleGroundings.empty(num_rules=1)
        rg.atom_table = atom_table
        rg.num_atoms = 1
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
        rg = RuleGroundings.empty(num_rules=1)
        rg.atom_table = atom_table
        rg.num_atoms = 2
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
        rg = RuleGroundings.empty(num_rules=1)
        rg.atom_table = atom_table
        rg.num_atoms = 1
        queries = torch.tensor(
            [[5, 7, 8], [5, 7, 8]], dtype=torch.long)        # same atom twice
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        # Pool extended by ONE, not two — duplicates dedupe.
        assert out.atom_table.shape[0] == 2
        # Both queries point to the same slot.
        assert out.query_pool_idx[0].item() == out.query_pool_idx[1].item()

    def test_empty_pool_built_from_queries_only(self):
        rg = RuleGroundings.empty(num_rules=1)
        queries = torch.tensor([[2, 5, 7], [3, 8, 9]], dtype=torch.long)
        out = populate_query_pool_idx(rg, queries, padding_idx=PAD)
        # Pool now contains the padding sentinel + the unique queries.
        # gathered atoms must match.
        gathered = out.atom_table[out.query_pool_idx]
        assert torch.equal(gathered, queries)

    def test_rejects_bad_query_shape(self):
        rg = RuleGroundings.empty(num_rules=1)
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


# ─────────────────────────────────────────────────────────────────────
# pad_outputs — padded vs unpadded equivalence
# ─────────────────────────────────────────────────────────────────────

def _make_enum_grandparent_grounder(w: int = 1, d: int = 2):
    """gp(X,Z) :- parent(X,Y), parent(Y,Z) under the enum (BC w/d) path.

    The padded-output feature is only active for the enum resolution
    path (the one that backs the BC_{w,d} family) — that's the path
    whose flat-output shape oscillates per batch on countries_s3+BC13
    / family+BC{12,13} and that this patch is designed to stabilise.
    SLD/RTF don't expose the per-rule ``G_r`` cap that drives padding.
    """
    facts = torch.cat([
        torch.tensor([[1, 1, 2], [1, 2, 3]], dtype=torch.long),
        _PAD_FACTS,
    ])
    heads = torch.tensor([[2, 24, 25]], dtype=torch.long)
    bodies = torch.tensor(
        [[[1, 24, 26], [1, 26, 25]]], dtype=torch.long)
    rule_lens = torch.tensor([2], dtype=torch.long)
    kb = KB(facts, heads, bodies, rule_lens,
            constant_no=23, predicate_no=3,
            padding_idx=PAD, device=DEVICE,
            fact_index_type='block_sparse')
    return make_bcwd(kb, w=w, d=d, u=0)


class TestPadOutputsEquivalence:
    """The padded path must produce a ``RuleGroundings`` whose real
    (non-padded) firings exactly match the unpadded output. Padding rows
    are masked off via ``firings_valid`` and must point at sentinel
    pool slot 0 so downstream gathers are safe.
    """

    @pytest.mark.parametrize("queries_in", [
        torch.tensor([[2, 1, 3]], dtype=torch.long),                 # provable
        torch.tensor([[2, 3, 1]], dtype=torch.long),                 # unprovable
        torch.tensor([[2, 1, 3], [2, 3, 1], [2, 1, 24]],
                     dtype=torch.long),                              # batch
    ])
    def test_padded_matches_unpadded_on_first_K_r(self, queries_in):
        grounder = _make_enum_grandparent_grounder()
        query_mask = torch.ones(queries_in.size(0), dtype=torch.bool)

        rg_unpad = grounder.run_bc(queries_in, query_mask, pad_outputs=False)
        rg_pad = grounder.run_bc(queries_in, query_mask, pad_outputs=True)

        # Unpadded has no firings_valid; padded has it iff any rule has firings.
        assert getattr(rg_unpad, "firings_valid", None) is None
        if not rg_unpad.A_in:
            # Unprovable batch — nothing to pad. Padding is a no-op.
            assert rg_pad.firings_valid is None
            return
        assert rg_pad.firings_valid is not None

        # Same rules present in both.
        assert set(rg_pad.A_in.keys()) == set(rg_unpad.A_in.keys())

        # Per-rule pad target — must be >= every rule's actual K_r AND
        # be a power of two (so compile sees only ~log2 graph variants).
        max_K_r = max(int(t.shape[0]) for t in rg_unpad.A_in.values())
        G_pad_expected = 1 if max_K_r <= 1 else (1 << (max_K_r - 1).bit_length())
        for r in rg_unpad.A_in.keys():
            A_in_unpad = rg_unpad.A_in[r]
            A_in_pad = rg_pad.A_in[r]
            A_out_unpad = rg_unpad.A_out[r]
            A_out_pad = rg_pad.A_out[r]
            K_r = A_in_unpad.shape[0]

            # Padded shape uses the across-rules max K_r rounded up to
            # next pow2 — fixed for all rules in this batch.
            assert A_in_pad.shape[0] == G_pad_expected
            assert A_in_pad.shape[1] == A_in_unpad.shape[1]
            assert A_out_pad.shape[0] == G_pad_expected
            assert A_out_pad.shape[1] == 1
            # Crucially: padding never *truncates* — every real firing
            # has a slot in the padded tensor.
            assert G_pad_expected >= K_r

            # First K_r rows: byte-identical to the unpadded output.
            assert torch.equal(A_in_pad[:K_r], A_in_unpad)
            assert torch.equal(A_out_pad[:K_r], A_out_unpad)

            # Padding rows: all point at sentinel slot 0.
            assert (A_in_pad[K_r:] == 0).all().item()
            assert (A_out_pad[K_r:] == 0).all().item()

            # firings_valid mask: True for the first K_r, False after.
            valid = rg_pad.firings_valid[r]
            assert valid.shape == (G_pad_expected,)
            assert valid[:K_r].all().item()
            assert (~valid[K_r:]).all().item()

        # atom_table — padded is a prefix-extension of unpadded (or equal).
        N_unpad = rg_unpad.atom_table.shape[0]
        assert rg_pad.atom_table.shape[0] >= N_unpad
        assert torch.equal(
            rg_pad.atom_table[:N_unpad], rg_unpad.atom_table)

        # query_pool_idx points into a valid (and identical-prefix) pool.
        assert torch.equal(rg_pad.query_pool_idx, rg_unpad.query_pool_idx)
