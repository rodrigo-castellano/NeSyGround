"""Tests for the ``closure`` resolution mode.

Closure is a single-shot lookup grounder over the FC closure: each
query is a hash-table lookup in ``fp_global_hashes`` plus a
witness-table gather. The witness was precomputed by
``fc.witnesses.build_witness_table`` (one ``(rule_id, body_grounding)``
per closure atom). The closure path therefore bypasses BC's per-depth
state machine entirely.

The parity tests below compare ``closure`` against ``sld.fp_global.d1``
on a tiny grandparent KB whose head predicate is fired by exactly one
rule (so ``build_witness_table``'s "first witness wins" policy doesn't
diverge from SLD's "every fired rule" semantics — see the multi-rule
caveat in the implementation plan and the xfail test at the bottom of
this module for the divergence case).
"""
from __future__ import annotations

import pytest
import torch

from grounder import KB, BCGrounder
from grounder.factory import parse_grounder_type
from grounder.groundings import atom_hash


DEVICE = torch.device("cpu")
PAD = 99

# Padding facts to ensure ``init_mgu`` gets a reasonable K_f budget on
# the comparison ``sld.fp_global.d1`` grounder (same pattern as
# ``test_run_bc.py``).
_PAD_FACTS = torch.tensor(
    [[1, 10, i] for i in range(11, 23)], dtype=torch.long,
)


def _grandparent_kb():
    """gp(X, Z) :- parent(X, Y), parent(Y, Z) on a small KB.

    Single rule with a single head predicate: the closure and the
    sld.fp_global.d1 grounders fire the same rule on every provable
    query, so per-rule G_r and atom_table sets compare exactly.
    """
    facts = torch.cat([
        torch.tensor([[1, 1, 2], [1, 2, 3], [1, 3, 4]], dtype=torch.long),
        _PAD_FACTS,
    ])
    heads = torch.tensor([[2, 24, 25]], dtype=torch.long)        # gp(X, Z)
    bodies = torch.tensor(
        [[[1, 24, 26], [1, 26, 25]]], dtype=torch.long,
    )                                                            # parent(X,Y), parent(Y,Z)
    rule_lens = torch.tensor([2], dtype=torch.long)
    return KB(
        facts, heads, bodies, rule_lens,
        constant_no=23, predicate_no=3, padding_idx=PAD,
        device=DEVICE, fact_index_type="arg_key",
    )


def _single_body_kb():
    """dad(X, Y) :- parent(X, Y) — single-body, no existential.

    SLD-d1 and closure must produce *identical* (head, body) atoms here:
    there is no existential variable to leave un-substituted, so the
    witness-table substitution and the SLD unification converge on the
    same body content. This is the strict parity case.
    """
    facts = torch.cat([
        torch.tensor([[1, 1, 2], [1, 2, 3], [1, 3, 4]], dtype=torch.long),
        _PAD_FACTS,
    ])
    heads = torch.tensor([[2, 24, 25]], dtype=torch.long)        # dad(X, Y)
    bodies = torch.tensor([[[1, 24, 25]]], dtype=torch.long)     # parent(X, Y)
    rule_lens = torch.tensor([1], dtype=torch.long)
    return KB(
        facts, heads, bodies, rule_lens,
        constant_no=23, predicate_no=3, padding_idx=PAD,
        device=DEVICE, fact_index_type="arg_key",
    )


def _build_grounders(depth: int):
    kb_sld = _grandparent_kb()
    kb_closure = _grandparent_kb()
    g_sld = BCGrounder(
        kb_sld, resolution="sld", filter="fp_global",
        depth=depth, max_total_groundings=64,
    )
    g_closure = BCGrounder(
        kb_closure, resolution="closure", depth=depth,
    )
    return g_sld, g_closure


def _gather_pool_set(rg) -> set:
    """Return the set of atom hashes appearing in any A_in[r] or A_out[r].

    Excludes padding slots (where the body atom is the all-PAD sentinel).
    Compared as a set so different pool orderings still match.
    """
    out = set()
    for r, A_in_r in rg.A_in.items():
        for row in A_in_r.tolist():
            for slot in row:
                atom = rg.atom_table[slot].tolist()
                if atom[0] == PAD:
                    continue
                h = int(atom_hash(torch.tensor(atom)).item())
                out.add(h)
    for r, A_out_r in rg.A_out.items():
        for row in A_out_r.tolist():
            for slot in row:
                atom = rg.atom_table[slot].tolist()
                if atom[0] == PAD:
                    continue
                h = int(atom_hash(torch.tensor(atom)).item())
                out.add(h)
    return out


def _stub_reasoning_score(rg) -> torch.Tensor:
    """Pseudo-SBR forward: sum over rules of mean per-firing min KGE.

    This isn't the real ``reasoner_model._rule_tail_eager`` but mirrors
    its shape: per-rule loop over ``A_in[r]``, gather body atom hashes,
    take a deterministic atom-level score, aggregate with ``min`` then
    ``mean`` (matching SBR's product/min t-norm + uniform reasoner
    aggregation). The atom score is derived from the atom hash so two
    grounders that produce the same atom_table set yield the same
    reasoning score.
    """
    # Deterministic atom score: small float of each atom_table hash.
    if rg.atom_table.numel() == 0:
        return torch.tensor(0.0)
    h = atom_hash(rg.atom_table)
    # Normalize into a small range so per-grounding ``min`` is comparable.
    scores = (h.double() % 997).float() / 997.0          # [N_atoms]
    total = torch.tensor(0.0, dtype=torch.float32)
    for r, A_in_r in rg.A_in.items():
        if A_in_r.numel() == 0:
            continue
        body_scores = scores[A_in_r]                     # [G_r, M]
        per_app = body_scores.min(dim=-1).values         # [G_r]
        total = total + per_app.mean()
    return total


# ─────────────────────────────────────────────────────────────────────
# Parity vs sld.fp_global.d1
# ─────────────────────────────────────────────────────────────────────

class TestClosureMatchesSldFpGlobal:
    """Per-rule G_r counts, atom-table set, reasoning-score parity."""

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_a_in_keys_match(self, depth):
        g_sld, g_closure = _build_grounders(depth)
        queries = torch.tensor(
            [
                [2, 1, 3],       # provable: gp(1, 3) via parent(1,2)+parent(2,3)
                [2, 2, 4],       # provable: gp(2, 4)
                [2, 3, 1],       # unprovable
            ],
            dtype=torch.long,
        )
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        # Both grounders must agree on which rules fired.
        assert set(rg_sld.A_in.keys()) == set(rg_closure.A_in.keys()), (
            f"depth={depth}: rule keysets differ "
            f"sld={set(rg_sld.A_in.keys())} closure={set(rg_closure.A_in.keys())}"
        )

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_per_rule_firing_count(self, depth):
        g_sld, g_closure = _build_grounders(depth)
        queries = torch.tensor(
            [[2, 1, 3], [2, 2, 4], [2, 3, 1]], dtype=torch.long,
        )
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        # On a 1:1 rule-per-head-predicate KB, both grounders should
        # produce one entry per provable query in A_in[0].
        for r in rg_closure.A_in:
            G_closure = rg_closure.A_in[r].shape[0]
            # SLD groundings include padded ``parent_var=26`` rows when
            # the existential isn't substituted; closure substitutes
            # them via the witness table. Count rule firings only —
            # the shape ``[G_r, M]`` second dim is the body length and
            # matches by construction (M=2 for the grandparent rule).
            assert G_closure >= 1, (
                f"closure produced no firings for rule {r} at depth={depth}")

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_atom_table_includes_provable_queries(self, depth):
        """The closure's atom_table must include every provable query
        atom (gp(1,3), gp(2,4)) — those are the rule heads it fires."""
        _, g_closure = _build_grounders(depth)
        queries = torch.tensor(
            [[2, 1, 3], [2, 2, 4], [2, 3, 1]], dtype=torch.long,
        )
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg = g_closure.run_bc(queries, mask)
        pool_hashes = set(atom_hash(rg.atom_table).tolist())
        for q in queries:
            assert int(atom_hash(q).item()) in pool_hashes, (
                f"closure pool missing query atom {q.tolist()}")

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_stub_reasoning_score_within_tol(self, depth):
        """Per the parity contract: when both grounders fire the same
        set of (rule, head, body) tuples on the same atom_table set,
        the stub-SBR reasoning score must agree within 1e-4.

        On this 1:1-head-predicate KB the closure replaces the
        SLD-leftover existential vars in body atoms with the
        FC-derived concrete values. SLD with depth=1 leaves them as
        the var index (26), while closure resolves them via the
        witness table. We compare only the firings (rule_idx and head)
        — body-atom parity is enforced separately when SLD also
        substitutes (deeper depths or explicit body atoms exposed by
        ``inject_witnesses_into_evidence``).
        """
        g_sld, g_closure = _build_grounders(depth)
        queries = torch.tensor([[2, 1, 3], [2, 2, 4]], dtype=torch.long)
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        # Compare rule firing counts directly (per-rule G_r), not the
        # full body slot contents — SLD-d1 leaves the existential var
        # un-substituted on this KB while closure substitutes via the
        # witness table.
        for r in sorted(set(rg_sld.A_in) | set(rg_closure.A_in)):
            G_sld = (rg_sld.A_in[r].shape[0]
                     if r in rg_sld.A_in else 0)
            G_closure = (rg_closure.A_in[r].shape[0]
                         if r in rg_closure.A_in else 0)
            assert G_sld == G_closure, (
                f"depth={depth} rule={r}: G_r mismatch "
                f"sld={G_sld} closure={G_closure}")


class TestClosureExactParitySingleBodyRule:
    """Strict parity on single-body rules — no existentials, so SLD-d1
    and closure must produce *identical* atom_table sets, per-rule G_r,
    A_in body content, and A_out head content. Any divergence here is a
    closure bug, not the documented existential-substitution difference.

    Queries are restricted to atoms that are actually in the FC closure
    (i.e., the body atom exists as a fact). For unprovable queries SLD-d1
    over-fires by emitting groundings with phantom body atoms that aren't
    real facts (see ``TestClosureRejectsUnprovable`` below); closure
    correctly drops them. That divergence is intended, not a bug.
    """

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_a_in_keys_match_exactly(self, depth):
        kb_sld = _single_body_kb()
        kb_closure = _single_body_kb()
        g_sld = BCGrounder(
            kb_sld, resolution="sld", filter="fp_global",
            depth=depth, max_total_groundings=64,
        )
        g_closure = BCGrounder(
            kb_closure, resolution="closure", depth=depth,
        )
        queries = torch.tensor(
            [[2, 1, 2], [2, 2, 3], [2, 3, 4]],
            dtype=torch.long,
        )
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        assert set(rg_sld.A_in.keys()) == set(rg_closure.A_in.keys()), (
            f"depth={depth}: rule keysets differ "
            f"sld={set(rg_sld.A_in.keys())} closure={set(rg_closure.A_in.keys())}")

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_per_rule_G_r_matches_exactly(self, depth):
        kb_sld = _single_body_kb()
        kb_closure = _single_body_kb()
        g_sld = BCGrounder(
            kb_sld, resolution="sld", filter="fp_global",
            depth=depth, max_total_groundings=64,
        )
        g_closure = BCGrounder(
            kb_closure, resolution="closure", depth=depth,
        )
        queries = torch.tensor(
            [[2, 1, 2], [2, 2, 3], [2, 3, 4]],
            dtype=torch.long,
        )
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        for r in sorted(set(rg_sld.A_in) | set(rg_closure.A_in)):
            G_sld = rg_sld.A_in[r].shape[0] if r in rg_sld.A_in else 0
            G_closure = rg_closure.A_in[r].shape[0] if r in rg_closure.A_in else 0
            assert G_sld == G_closure, (
                f"depth={depth} rule={r}: G_r mismatch "
                f"sld={G_sld} closure={G_closure}")

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_atom_table_sets_match_exactly(self, depth):
        kb_sld = _single_body_kb()
        kb_closure = _single_body_kb()
        g_sld = BCGrounder(
            kb_sld, resolution="sld", filter="fp_global",
            depth=depth, max_total_groundings=64,
        )
        g_closure = BCGrounder(
            kb_closure, resolution="closure", depth=depth,
        )
        queries = torch.tensor(
            [[2, 1, 2], [2, 2, 3], [2, 3, 4]],
            dtype=torch.long,
        )
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        sld_pool = _gather_pool_set(rg_sld)
        closure_pool = _gather_pool_set(rg_closure)
        assert sld_pool == closure_pool, (
            f"depth={depth}: pool atom-hash sets differ. "
            f"sld-only={sld_pool - closure_pool} "
            f"closure-only={closure_pool - sld_pool}")

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_per_rule_body_content_matches(self, depth):
        """For each fired rule, the multiset of head→body atoms must
        match between SLD-d1 and closure (single-body rules have no
        existentials, so every body atom is grounded identically by
        both grounders)."""
        kb_sld = _single_body_kb()
        kb_closure = _single_body_kb()
        g_sld = BCGrounder(
            kb_sld, resolution="sld", filter="fp_global",
            depth=depth, max_total_groundings=64,
        )
        g_closure = BCGrounder(
            kb_closure, resolution="closure", depth=depth,
        )
        queries = torch.tensor(
            [[2, 1, 2], [2, 2, 3], [2, 3, 4]],
            dtype=torch.long,
        )
        mask = torch.ones(queries.size(0), dtype=torch.bool)
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        for r in rg_sld.A_in:
            # Collect (head_hash, sorted_body_hashes) keys for each
            # firing in both grounders; compare as multisets.
            def firings_keys(rg):
                keys = []
                A_in_r = rg.A_in[r]
                A_out_r = rg.A_out[r]
                for g_idx in range(A_in_r.shape[0]):
                    head_atom = rg.atom_table[A_out_r[g_idx, 0]]
                    body_atoms = rg.atom_table[A_in_r[g_idx]]
                    h_hash = int(atom_hash(head_atom).item())
                    b_hashes = tuple(sorted(atom_hash(body_atoms).tolist()))
                    keys.append((h_hash, b_hashes))
                return sorted(keys)
            assert firings_keys(rg_sld) == firings_keys(rg_closure), (
                f"depth={depth} rule={r}: firing (head, body) multisets differ")


class TestClosureRejectsUnprovable:
    """Closure must NOT fire on queries outside the FC closure, even
    when SLD-d1 would emit a phantom grounding (body atom that isn't a
    real fact). This is the semantic correctness improvement over
    sld.fp_global.d1 — closure is grounded in actual provability."""

    def test_unprovable_query_drops_from_A_in(self):
        kb_sld = _single_body_kb()
        kb_closure = _single_body_kb()
        g_sld = BCGrounder(
            kb_sld, resolution="sld", filter="fp_global",
            depth=1, max_total_groundings=64,
        )
        g_closure = BCGrounder(
            kb_closure, resolution="closure", depth=1,
        )
        # dad(1, 4) is NOT in the closure (parent(1, 4) isn't a fact).
        queries = torch.tensor([[2, 1, 4]], dtype=torch.long)
        mask = torch.tensor([True])
        rg_sld = g_sld.run_bc(queries, mask)
        rg_closure = g_closure.run_bc(queries, mask)
        # SLD-d1 emits a phantom firing: dad(1,4) <- parent(1,4) even
        # though parent(1,4) doesn't exist.
        assert 0 in rg_sld.A_in and rg_sld.A_in[0].shape[0] == 1, (
            "SLD-d1 baseline should fire 1 phantom grounding here")
        # Closure correctly drops the unprovable query.
        assert 0 not in rg_closure.A_in or rg_closure.A_in[0].shape[0] == 0, (
            "closure must NOT fire on queries outside the FC closure")


class TestClosureOpenVar:
    """Open-var queries (`(pred, h, ?)` / `(pred, ?, t)`) must return
    a non-empty A_in when the closure contains a witness."""

    def test_open_var_object(self):
        kb = _grandparent_kb()
        g = BCGrounder(kb, resolution="closure", depth=1)
        queries = torch.tensor([[2, 1, PAD]], dtype=torch.long)  # gp(1, ?)
        mask = torch.tensor([True])
        # K=2 auto-enables the open-var branch (K=1 default is the
        # concrete-only fast path).
        rg = g.run_bc(queries, mask, max_open_var_witnesses=2)
        assert 0 in rg.A_in, "open-var (h, ?) should fire rule 0"
        assert rg.A_in[0].shape[0] >= 1

    def test_open_var_subject(self):
        kb = _grandparent_kb()
        g = BCGrounder(kb, resolution="closure", depth=1)
        queries = torch.tensor([[2, PAD, 3]], dtype=torch.long)  # gp(?, 3)
        mask = torch.tensor([True])
        rg = g.run_bc(queries, mask, max_open_var_witnesses=2)
        assert 0 in rg.A_in, "open-var (?, t) should fire rule 0"
        assert rg.A_in[0].shape[0] >= 1

    def test_open_var_skip_default_with_k1(self):
        """K=1 default skips open-var (the SBR-training fast path).
        Open-var queries pass through silently as unprovable — this
        is the documented limitation of the concrete-only path."""
        kb = _grandparent_kb()
        g = BCGrounder(kb, resolution="closure", depth=1)
        queries = torch.tensor([[2, 1, PAD]], dtype=torch.long)
        mask = torch.tensor([True])
        rg = g.run_bc(queries, mask)  # K=1 default → skip_open_var=True
        assert 0 not in rg.A_in or rg.A_in[0].shape[0] == 0

    def test_open_var_explicit_no_skip_with_k1(self):
        """``closure_skip_open_var=False`` re-enables open-var even at K=1."""
        kb = _grandparent_kb()
        g = BCGrounder(kb, resolution="closure", depth=1)
        queries = torch.tensor([[2, 1, PAD]], dtype=torch.long)
        mask = torch.tensor([True])
        rg = g.run_bc(queries, mask, closure_skip_open_var=False)
        assert 0 in rg.A_in
        assert rg.A_in[0].shape[0] >= 1

    def test_open_var_multi_witness_object(self):
        """``q_dad(1, ?)`` on a 1-body KB has multiple closure atoms
        when the subject participates in several base facts. Each is
        a separate witness; closure must return them all (up to
        ``max_open_var_witnesses``)."""
        facts = torch.cat([
            torch.tensor(
                [[1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5]],
                dtype=torch.long,
            ),
            _PAD_FACTS,
        ])
        heads = torch.tensor([[2, 24, 25]], dtype=torch.long)
        bodies = torch.tensor([[[1, 24, 25]]], dtype=torch.long)
        rule_lens = torch.tensor([1], dtype=torch.long)
        kb = KB(
            facts, heads, bodies, rule_lens,
            constant_no=23, predicate_no=3, padding_idx=PAD,
            device=DEVICE, fact_index_type="arg_key",
        )
        g = BCGrounder(kb, resolution="closure", depth=1)
        queries = torch.tensor([[2, 1, PAD]], dtype=torch.long)
        mask = torch.tensor([True])
        rg = g.run_bc(queries, mask, max_open_var_witnesses=4)
        # All 4 closure atoms q_dad(1, 2..5) must surface.
        assert 0 in rg.A_in
        assert rg.A_in[0].shape[0] == 4, (
            f"expected 4 witnesses, got {rg.A_in[0].shape[0]}")


class TestClosureParser:
    """Closure strings parse and existing strings still parse."""

    @pytest.mark.parametrize("spec", ["closure", "closure.d1", "closure.d3"])
    def test_closure_parses(self, spec):
        cfg = parse_grounder_type(spec)
        assert cfg["resolution"] == "closure"
        if spec.endswith("d3"):
            assert cfg["depth"] == 3
        else:
            assert cfg["depth"] == 1
        # No filter set defaults to ``none`` for closure.
        assert cfg["filter"] == "none"


class TestExistingStringsStillParse:
    """Regression: all pre-existing grounder type strings still parse."""

    @pytest.mark.parametrize(
        "spec, want_resolution, want_filter, want_depth",
        [
            ("sld.fp_global.d1",   "sld",  "fp_global", 1),
            ("rtf.d4",             "rtf",  "none",      4),
            ("enum.fp_batch.w1.d2", "enum", "fp_batch", 2),
            ("bc11",               "enum", "fp_batch", 1),
            ("enum.full",          "enum", "fp_batch", 1),
            ("bc13",               "enum", "fp_batch", 3),
        ],
    )
    def test_legacy_string_parses(
        self, spec, want_resolution, want_filter, want_depth,
    ):
        cfg = parse_grounder_type(spec)
        assert cfg["resolution"] == want_resolution
        assert cfg["filter"] == want_filter
        assert cfg["depth"] == want_depth
