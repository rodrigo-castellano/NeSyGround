"""Leapfrog Triejoin (worst-case-optimal join) for forward chaining.

Veldhuizen 2014 / Generic-Join. A variable-elimination join whose output is
AGM-bounded: where 3+ atoms share a variable (triangle / transitive shapes) the
intermediate never goes Cartesian, so peak memory tracks the *output* not the
per-stage ``|partial| × fan-out`` product the staged ragged join materialises.
This targets the FC transitive-closure memory blow-up (e.g. wn18rr).

ONE join drives both phases via a per-atom source selector ``anchor_j``:

  * step 0 (full join)       — ``anchor_j=None``: every atom reads base ∪ provable.
  * step t>0 (semi-naive)    — ``anchor_j=k``: ordered atom ``k`` reads only the
    step-(t-1) delta (Δ), every other atom reads base ∪ provable (I). ``run`` calls
    this once per body atom as the Δ-anchor; the union over anchors is the
    semi-naive ΔT_r, and ``_filter_new`` drops already-known heads.

Mixed into ``FCDynamic`` (it reads engine state: ``self._pred_facts``,
``self._base_ps_off``, ``self._fact_hashes``, …). Selected by ``join_algo='leapfrog'``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from grounder.data.rule_index import RulePattern
from grounder.forward.staged.joins import (
    _pred_pairs_from_ps, _ps_expand, _po_expand,
    _ps_expand_combined, _po_expand_combined,
)


class LeapfrogMixin:
    """Worst-case-optimal join for FCDynamic (full join + semi-naive anchored term)."""

    def _apply_rule_lftj(
        self, cr: RulePattern, ordered_bps: list,
        prov_ps_off: Tensor, prov_ps_vals: Tensor,
        prov_po_off: Tensor, prov_po_vals: Tensor,
        provable_hashes: Tensor,
        *,
        anchor_j: Optional[int] = None,
        delta_ps_off: Optional[Tensor] = None, delta_ps_vals: Optional[Tensor] = None,
        delta_po_off: Optional[Tensor] = None, delta_po_vals: Optional[Tensor] = None,
        delta_hashes: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Leapfrog Triejoin over the rule body.

        Variable-elimination order: most-shared variable first (Generic-Join
        heuristic). For each variable v: seed its candidate set from a tight atom
        (one whose other arg is bound / a constant, else the smallest domain),
        then apply every other atom binding v as a fact-membership filter — the
        survivors are the AGM-bounded join.

        Per-atom source: the ordered atom ``anchor_j`` reads only the delta Δ (as
        iterator + membership); every other atom reads base ∪ provable (I).
        ``anchor_j=None`` ⇒ full join (all atoms read base ∪ provable). Emits the
        head in compact predicate space (decompacted in ``run()``).
        """
        E, E2 = self.E, self.E * self.E
        cno = self._constant_no
        body = ordered_bps[: cr.num_body]
        if not body:
            return None

        # Variable → list of (atom_idx, role) where it appears.
        var_atoms: Dict[int, List[Tuple[int, int]]] = {}
        for j, bp in enumerate(body):
            for role, a in ((0, bp["arg0_var"]), (1, bp["arg1_var"])):
                if a > cno:        # variable, not constant
                    var_atoms.setdefault(a, []).append((j, role))
        if not var_atoms:
            return None

        # Greedy elimination order: most-shared first, then arbitrary.
        var_order = sorted(var_atoms.keys(),
                           key=lambda v: -len(var_atoms[v]))

        def domain(j: int, pred_k: int, role: int) -> Tensor:
            """Sorted unique values at ``role`` of ``pred_k`` from atom ``j``'s
            source — Δ when ``j`` is the anchor, else base ∪ provable."""
            if j == anchor_j:
                s_d, o_d = _pred_pairs_from_ps(
                    pred_k, delta_ps_off, delta_ps_vals, E)
                return torch.unique(s_d if role == 0 else o_d)
            s_b, o_b = self._pred_facts.get(pred_k, self._empty)
            s_p, o_p = _pred_pairs_from_ps(
                pred_k, prov_ps_off, prov_ps_vals, E)
            if role == 0:
                return torch.unique(
                    torch.cat([s_b, s_p]) if s_p.numel() else s_b)
            return torch.unique(
                torch.cat([o_b, o_p]) if o_p.numel() else o_b)

        def ps_lookup(j: int, pred_k: int, kv: Tensor):
            if j == anchor_j:
                return _ps_expand(pred_k, kv, delta_ps_off, delta_ps_vals, E)
            return _ps_expand_combined(
                pred_k, kv,
                self._base_ps_off, self._base_ps_vals,
                prov_ps_off, prov_ps_vals, E)

        def po_lookup(j: int, pred_k: int, kv: Tensor):
            if j == anchor_j:
                return _po_expand(pred_k, kv, delta_po_off, delta_po_vals, E)
            return _po_expand_combined(
                pred_k, kv,
                self._base_po_off, self._base_po_vals,
                prov_po_off, prov_po_vals, E)

        def in_kb(j: int, qh: Tensor) -> Tensor:
            """Is atom ``j``'s tuple in its source? facts ∪ Δ for the anchor,
            facts ∪ provable otherwise."""
            n_q = qh.shape[0]
            in_f = torch.zeros(n_q, dtype=torch.bool, device=qh.device)
            nf = self._num_facts
            if nf > 0:
                pos = torch.searchsorted(self._fact_hashes, qh)
                v = pos < nf
                cl = pos.clamp(max=max(nf - 1, 0))
                in_f = v & (self._fact_hashes[cl] == qh)
            other = delta_hashes if j == anchor_j else provable_hashes
            if other is not None and other.numel() > 0:
                n_o = other.shape[0]
                pos_o = torch.searchsorted(other, qh)
                v_o = pos_o < n_o
                cl_o = pos_o.clamp(max=max(n_o - 1, 0))
                return in_f | (v_o & (other[cl_o] == qh))
            return in_f

        partial: Dict[int, Tensor] = {}
        for v in var_order:
            atoms_with_v = var_atoms[v]

            # Find a "seed" atom — one whose other argument is bound
            # by ``partial`` or is a rule constant. That gives a tight
            # iterator for v.
            seed = None
            for j, role in atoms_with_v:
                bp = body[j]
                other_role = 1 - role
                other_var = (bp["arg0_var"] if other_role == 0
                             else bp["arg1_var"])
                if other_var in partial or other_var <= cno:
                    seed = (j, role, other_var, other_role)
                    break

            if seed is None:
                # No atom has a bound other-arg ⇒ first variable
                # of the elimination. Seed from the smallest
                # per-role domain.
                best = min(atoms_with_v,
                           key=lambda jr: domain(
                               jr[0], body[jr[0]]["pred_idx"], jr[1]).numel())
                j, role = best
                v_dom = domain(j, body[j]["pred_idx"], role)
                if v_dom.numel() == 0:
                    return None
                partial[v] = v_dom
                # Don't process other atoms with v as filters yet —
                # they'll need bindings of OTHER vars first.
                continue

            j, role, other_var, other_role = seed
            bp = body[j]
            if other_var <= cno:
                # ``other_var`` is a constant — single-value seed.
                const_vals = torch.tensor(
                    [int(other_var)], dtype=torch.long,
                    device=self.device_str)
                source_arr = const_vals
                map_back = False
            else:
                source_arr = partial[other_var]
                map_back = True

            if other_role == 0:
                ri, v_vals = ps_lookup(j, bp["pred_idx"], source_arr)
            else:
                ri, v_vals = po_lookup(j, bp["pred_idx"], source_arr)
            if ri.numel() == 0:
                return None

            # Expand partial: row k goes to ri[k] in input partial,
            # with v = v_vals[k].
            if map_back:
                partial = {t: arr[ri] for t, arr in partial.items()}
            partial[v] = v_vals

            # Apply remaining atoms binding v as filters.
            for j2, role2 in atoms_with_v:
                if j2 == j:
                    continue
                bp2 = body[j2]
                other_role2 = 1 - role2
                other_var2 = (bp2["arg0_var"] if other_role2 == 0
                              else bp2["arg1_var"])
                if other_var2 in partial:
                    v_arr = partial[v]
                    o_arr = partial[other_var2]
                    sv, ov = (v_arr, o_arr) if role2 == 0 else (o_arr, v_arr)
                elif other_var2 <= cno:
                    v_arr = partial[v]
                    const_t = torch.full_like(
                        v_arr, int(other_var2))
                    sv, ov = (v_arr, const_t) if role2 == 0 else (const_t, v_arr)
                else:
                    # Other-arg is an unbound variable — defer until
                    # it's processed. (No filter possible yet.)
                    continue

                qh = bp2["pred_idx"] * E2 + sv * E + ov
                ok = in_kb(j2, qh)
                if not ok.any():
                    return None
                partial = {t: arr[ok] for t, arr in partial.items()}

        # All body variables bound; emit head in compact predicate
        # space (decompacted in run()).
        hx = partial.get(cr.head_var0)
        hy = partial.get(cr.head_var1)
        if hx is None or hy is None:
            return None
        head_pred_compact = self._pred_to_compact.get(
            int(cr.head_pred_idx), 0)
        return head_pred_compact * E2 + hx * E + hy


__all__ = ["LeapfrogMixin"]
