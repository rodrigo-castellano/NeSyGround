"""Equivalence tests for the flat-RuleGroundings refactor.

For each grounder path that produces a ``RuleGroundings``, compare the
flat-tensor fields against the reconstructed-from-shim dict form to
verify both views agree byte-for-byte. Also checks that
``evidence_to_rule_groundings`` produces the SAME per-rule (A_in, A_out)
contents as the legacy dict implementation (reconstructed in this file
for direct comparison).
"""
from __future__ import annotations

import torch

from grounder.groundings import (
    atom_hash,
    evidence_to_rule_groundings,
    evidence_unique_app_keys,
)
from grounder.types import ProofEvidence, RuleGroundings, _PerRuleDictView


def _legacy_dict_from_evidence(evidence, padding_idx, num_rules):
    """Re-implementation of the OLD per-rule-dict construction path.

    Faithful copy of the pre-flatten ``evidence_to_rule_groundings``
    logic with ``A_in`` / ``A_out`` as Python dicts. Used as the
    ground-truth source for the equivalence check.
    """
    keys = evidence_unique_app_keys(evidence, padding_idx)
    if keys.shape[0] == 0:
        return {
            "atom_table": torch.zeros(0, 3, dtype=torch.long),
            "A_in": {}, "A_out": {},
            "num_atoms": 0,
            "num_rules": int(num_rules or 0),
        }

    M = (keys.shape[1] - 4) // 3
    N = keys.shape[0]
    ridx = keys[:, 0]
    head = keys[:, 1:4]
    body = keys[:, 4:].reshape(N, M, 3)

    body_active = body[..., 0] != padding_idx
    head_h = atom_hash(head)
    body_h = atom_hash(body)
    flat_active_body_h = body_h[body_active]
    flat_active_body = body[body_active]
    pad_atom = torch.tensor(
        [padding_idx, padding_idx, padding_idx],
        dtype=torch.long, device=keys.device).unsqueeze(0)
    pad_h = atom_hash(pad_atom)

    all_h = torch.cat([head_h, flat_active_body_h, pad_h])
    uniq_h, inv = torch.unique(all_h, return_inverse=True)

    n_head = head_h.shape[0]
    n_body = flat_active_body_h.shape[0]
    repr_atoms = torch.zeros(
        uniq_h.shape[0], 3, dtype=torch.long, device=keys.device)
    repr_atoms[inv[:n_head]] = head
    repr_atoms[inv[n_head:n_head + n_body]] = flat_active_body
    repr_atoms[inv[n_head + n_body]] = pad_atom.squeeze(0)
    pad_id = int(inv[n_head + n_body].item())

    head_id = inv[:n_head]
    body_id = torch.full((N, M), pad_id,
                         dtype=torch.long, device=keys.device)
    body_id[body_active] = inv[n_head:n_head + n_body]

    if num_rules is None:
        num_rules_ = int(ridx.max().item()) + 1 if ridx.numel() else 0
    else:
        num_rules_ = int(num_rules)

    A_in = {}
    A_out = {}
    if num_rules_ > 0:
        rule_range = torch.arange(num_rules_, device=ridx.device)
        sel_all = ridx.unsqueeze(0) == rule_range.unsqueeze(1)
        fired = sel_all.any(dim=1).tolist()
        for r in range(num_rules_):
            if not fired[r]:
                continue
            sel = sel_all[r]
            A_in[r] = body_id[sel]
            A_out[r] = head_id[sel].unsqueeze(-1)

    return {
        "atom_table": repr_atoms, "A_in": A_in, "A_out": A_out,
        "num_atoms": int(repr_atoms.shape[0]),
        "num_rules": num_rules_,
    }


def _atom_h_set(t: torch.Tensor) -> set:
    return set(atom_hash(t).tolist())


def _build_test_evidence():
    """Synthesize a structured ProofEvidence with multiple rules + duplicates.

    Layout matches the post-PAG_paper-cycle BC output: B=2, C=3, D=1,
    M=2. Three queries, two rules (rule 0: M=2 body, rule 1: M=1
    body). Some rows are masked off.
    """
    B, C, D, M = 2, 3, 1, 2
    # Rule 0: head (p_h, h, t) <- body[0]=(p_b0, h, x), body[1]=(p_b1, x, t)
    # Rule 1: head (p_h, h, t) <- body[0]=(p_b1, h, t) (M_r=1, body[1]=PAD)
    PAD = 99
    body = torch.full((B, C, D, M, 3), PAD, dtype=torch.long)
    head = torch.full((B, C, D, 3), PAD, dtype=torch.long)
    rule_idx = torch.full((B, C, D), -1, dtype=torch.long)
    body_count = torch.zeros(B, C, D, dtype=torch.long)
    mask = torch.zeros(B, C, dtype=torch.bool)

    # (B=0, C=0): rule 0, head (5, 1, 2), body (3, 1, 9), (4, 9, 2)
    head[0, 0, 0] = torch.tensor([5, 1, 2])
    body[0, 0, 0, 0] = torch.tensor([3, 1, 9])
    body[0, 0, 0, 1] = torch.tensor([4, 9, 2])
    rule_idx[0, 0, 0] = 0
    body_count[0, 0, 0] = 2
    mask[0, 0] = True

    # (B=0, C=1): rule 1, head (5, 1, 2), body (6, 1, 2), pad
    head[0, 1, 0] = torch.tensor([5, 1, 2])
    body[0, 1, 0, 0] = torch.tensor([6, 1, 2])
    rule_idx[0, 1, 0] = 1
    body_count[0, 1, 0] = 1
    mask[0, 1] = True

    # (B=1, C=0): rule 0, head (5, 3, 7), body (3, 3, 8), (4, 8, 7)
    head[1, 0, 0] = torch.tensor([5, 3, 7])
    body[1, 0, 0, 0] = torch.tensor([3, 3, 8])
    body[1, 0, 0, 1] = torch.tensor([4, 8, 7])
    rule_idx[1, 0, 0] = 0
    body_count[1, 0, 0] = 2
    mask[1, 0] = True

    # (B=1, C=1): rule 0, same head/body as (B=0, C=0) — should DEDUP.
    head[1, 1, 0] = torch.tensor([5, 1, 2])
    body[1, 1, 0, 0] = torch.tensor([3, 1, 9])
    body[1, 1, 0, 1] = torch.tensor([4, 9, 2])
    rule_idx[1, 1, 0] = 0
    body_count[1, 1, 0] = 2
    mask[1, 1] = True

    return ProofEvidence(
        body=body, mask=mask, count=torch.tensor([2, 3]),
        rule_idx=rule_idx, body_count=body_count,
        D=D, M=M, head=head,
    ), PAD


class TestFlattenEquivalence:

    def test_a_in_a_out_match_legacy_construction(self):
        ev, PAD = _build_test_evidence()
        num_rules = 2

        # New (flat) path.
        rg = evidence_to_rule_groundings(ev, PAD, num_rules=num_rules)

        # Legacy dict path (reconstructed in this test).
        legacy = _legacy_dict_from_evidence(ev, PAD, num_rules)

        # atom_table contents must match as a SET of atoms (rows may be
        # in different order, since dedup ordering depends on unique-hash
        # output which is the same in both impls — but verify anyway).
        assert _atom_h_set(rg.atom_table) == _atom_h_set(legacy["atom_table"])
        assert rg.num_atoms == legacy["num_atoms"]
        assert rg.num_rules == legacy["num_rules"]

        # Per-rule A_in / A_out via the shim. The shim trims to per-rule
        # M_r; the legacy form also stored per-rule [G_r, M] (with M = global
        # M). Both must give identical atom_id sets per rule.
        for r in range(num_rules):
            shim_in = rg.A_in[r] if r in rg.A_in else None
            shim_out = rg.A_out[r] if r in rg.A_out else None
            legacy_in = legacy["A_in"].get(r)
            legacy_out = legacy["A_out"].get(r)

            if legacy_in is None:
                assert shim_in is None or shim_in.numel() == 0
                continue
            assert shim_in is not None
            # Same number of firings per rule.
            assert shim_in.shape[0] == legacy_in.shape[0]

            # Convert to atom triples for comparison (M may differ
            # because shim trims to per-rule M_r).
            shim_bodies = rg.atom_table[shim_in]                  # [G_r, M_r, 3]
            legacy_bodies = legacy["atom_table"][legacy_in]       # [G_r, M, 3]
            # Compare as sorted multiset of body-atom tuples per firing.
            shim_sets = sorted(
                tuple(sorted(tuple(a.tolist()) for a in row))
                for row in shim_bodies)
            legacy_active_atom = legacy_bodies[..., 0] != PAD
            legacy_sets = sorted(
                tuple(sorted(
                    tuple(a.tolist())
                    for a, ok in zip(row, valid) if ok))
                for row, valid in zip(legacy_bodies, legacy_active_atom))
            assert shim_sets == legacy_sets, f"rule {r} body mismatch"

            # Heads: same multiset.
            shim_heads = rg.atom_table[shim_out.squeeze(-1)]
            legacy_heads = legacy["atom_table"][legacy_out.squeeze(-1)]
            assert sorted(tuple(a.tolist()) for a in shim_heads) == \
                sorted(tuple(a.tolist()) for a in legacy_heads), \
                f"rule {r} heads mismatch"

    def test_flat_fields_consistent_with_shim(self):
        """``rg.A_in[r]`` via shim slice must equal flat[offset:offset+K_r]."""
        ev, PAD = _build_test_evidence()
        num_rules = 2
        rg = evidence_to_rule_groundings(ev, PAD, num_rules=num_rules)

        for r in range(num_rules):
            offsets = rg.rule_offsets
            s = int(offsets[r].item())
            e = int(offsets[r + 1].item())
            slice_K_r = e - s
            assert rg.rule_idx[s:e].eq(r).all()
            # firing_valid all True on the unpadded path.
            assert rg.firing_valid[s:e].all()
            # body_pool_idx slice has shape [K_r, M_max].
            assert rg.body_pool_idx[s:e].shape == (slice_K_r, rg.M_max)
            assert rg.head_pool_idx[s:e].shape == (slice_K_r,)
            # The shim trims to per-rule M_r; the trimmed prefix must
            # match body_pool_idx[s:e, :M_r].
            if r in rg.A_in:
                shim = rg.A_in[r]
                M_r = shim.shape[1]
                if M_r > 0:
                    assert torch.equal(
                        shim, rg.body_pool_idx[s:e, :M_r])

    def test_empty_rg_round_trip(self):
        rg = RuleGroundings.empty(num_rules=5)
        # Iterating an empty shim yields no rules.
        assert list(rg.A_in.keys()) == []
        assert list(rg.A_out.keys()) == []
        assert len(rg.A_in) == 0
        assert rg.firing_valid.numel() == 0
        # Offsets are [0, 0, 0, 0, 0, 0].
        assert rg.rule_offsets.tolist() == [0] * 6

    def test_closure_fast_rg_shape_and_layout(self):
        """``closure_to_rule_groundings_fast`` produces a static-shape
        RuleGroundings whose layout matches the documented contract:
        atom_table = [queries | flat body slots | pad], queries occupy
        ``query_pool_idx``, firings sorted by rule_idx with invalid
        firings sentinel-bucketed past ``rule_offsets[num_rules]``.
        """
        from grounder.groundings import closure_to_rule_groundings_fast
        device = torch.device("cpu")
        PAD = 99
        B, K, M = 3, 1, 2
        num_rules = 4
        # Queries:    [(5,1,2), (5,3,7), (5,8,9)]
        queries = torch.tensor(
            [[5, 1, 2], [5, 3, 7], [5, 8, 9]], dtype=torch.long)
        # Witness slot 0 fires rule 1 with body (3,1,9),(4,9,2). Valid.
        # Witness slot 1 fires rule 0 with body (3,3,8),pad. Valid.
        # Witness slot 2 doesn't fire (invalid).
        witness_rule = torch.tensor([[1], [0], [-1]], dtype=torch.long)
        witness_body = torch.full((B, K, M, 3), PAD, dtype=torch.long)
        witness_body[0, 0, 0] = torch.tensor([3, 1, 9])
        witness_body[0, 0, 1] = torch.tensor([4, 9, 2])
        witness_body[1, 0, 0] = torch.tensor([3, 3, 8])
        witness_count = torch.tensor([[2], [1], [0]], dtype=torch.long)
        valid_rule = torch.tensor([[True], [True], [False]])

        rg = closure_to_rule_groundings_fast(
            witness_rule=witness_rule, witness_body=witness_body,
            witness_count=witness_count, valid_rule=valid_rule,
            queries=queries, num_rules=num_rules, M_max=M,
            padding_idx=PAD,
        )

        # atom_table = B + B*K*M + 1 = 3 + 3*1*2 + 1 = 10 rows.
        assert rg.atom_table.shape == (10, 3)
        # First B = 3 rows are the queries.
        assert torch.equal(rg.atom_table[:B], queries)
        # query_pool_idx = arange(B).
        assert torch.equal(
            rg.query_pool_idx, torch.arange(B, dtype=torch.long))
        # Sorted firings: rule 0 first (1 firing), rule 1 second
        # (1 firing), then num_rules sentinel bucket (1 firing for
        # the invalid slot).
        assert rg.rule_idx.shape == (B * K,)
        # rule_offsets[r+1] - rule_offsets[r] for valid rules; the
        # invalid firing sits past rule_offsets[num_rules] so per-rule
        # iteration sees only valid rules.
        assert rg.rule_offsets.tolist() == [0, 1, 2, 2, 2]
        # Per-rule slices reconstruct the witnesses.
        s, e = int(rg.rule_offsets[0]), int(rg.rule_offsets[1])
        # Rule 0: 1 firing originating from slot 1 (witness body
        # (3, 3, 8); body count 1).
        assert rg.firing_valid[s:e].all()
        assert rg.head_pool_idx[s:e].tolist() == [1]
        # body atom slot s, m=0 should index into atom_table at slot
        # B + 1*M + 0 = 5; atom_table[5] = (3, 3, 8).
        bp = rg.body_pool_idx[s, 0].item()
        assert rg.atom_table[bp].tolist() == [3, 3, 8]
        assert rg.body_atom_valid[s, 0].item()
        # m=1 is padding for rule 0's witness (body count 1).
        assert not rg.body_atom_valid[s, 1].item()
        # Rule 1: 1 firing originating from slot 0.
        s, e = int(rg.rule_offsets[1]), int(rg.rule_offsets[2])
        assert rg.head_pool_idx[s:e].tolist() == [0]
        # body atom slot at m=0 should be at offset B + 0*M + 0 = 3;
        # atom_table[3] = (3, 1, 9).
        bp = rg.body_pool_idx[s, 0].item()
        assert rg.atom_table[bp].tolist() == [3, 1, 9]
        bp = rg.body_pool_idx[s, 1].item()
        assert rg.atom_table[bp].tolist() == [4, 9, 2]
        assert rg.body_atom_valid[s].all()
        # Invalid firing past rule_offsets[num_rules]: firing_valid False.
        assert not rg.firing_valid[2].item()

    def test_padded_rg_keeps_firing_valid(self):
        """After pad_rule_groundings, ``rg.firings_valid`` is non-None and
        masks padding rows; rg.firing_valid (flat) also reflects this."""
        from grounder.groundings import pad_rule_groundings
        ev, PAD = _build_test_evidence()
        num_rules = 2
        rg = evidence_to_rule_groundings(ev, PAD, num_rules=num_rules)
        # Pad each rule to 4 firings (most have 2; padding 2 extra).
        rg_pad = pad_rule_groundings(
            rg, pad_per_rule_to=4, pad_atom_table_to=16, pad_idx_for_atoms=0)
        # Total firings = num_rules * 4.
        assert rg_pad.rule_idx.numel() == num_rules * 4
        # firings_valid view is now populated.
        assert rg_pad.firings_valid is not None
        for r in range(num_rules):
            mask = rg_pad.firings_valid[r]
            assert mask.dtype == torch.bool
            assert mask.shape == (4,)
            # First K_r entries True; trailing entries False.
            real_K_r = int((rg.rule_offsets[r + 1] - rg.rule_offsets[r]).item())
            real_K_r = min(real_K_r, 4)
            assert mask[:real_K_r].all()
            if real_K_r < 4:
                assert not mask[real_K_r:].any()
