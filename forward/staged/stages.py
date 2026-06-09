"""Stage machinery for the staged ragged join (mixed into ``FCDynamic``).

``_StagesMixin`` owns the per-rule join evaluation: the full join at step 0
(``_apply_rule``) and one semi-naive anchored term at step t>0
(``_apply_rule_anchored``), plus the shared stage loops (``_run_stages`` /
``_run_stages_anchored``) and their ``staged`` vs ``chunked`` dispatchers. The
methods reference engine state (``self._pred_facts``, ``self._base_ps_off``,
``self._constant_no``, …) set up by ``FCDynamic.__init__``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from grounder.data.rule_index import RulePattern
from grounder.forward.staged.joins import (
    _pred_pairs_from_ps, _ps_expand, _po_expand,
    _ps_expand_combined, _po_expand_combined,
)
from grounder.forward.staged.plan import _compute_frontiers


class _StagesMixin:
    """Per-rule join stages for FCDynamic (full join + semi-naive anchored term)."""

    def _filter_by_consts(
        self, partial: Dict[int, Tensor], bp: dict,
    ) -> Optional[Dict[int, Tensor]]:
        """Filter partial rows so that constant args in ``bp`` are satisfied.

        Body args with value <= constant_no are entity ids (constants), not
        variables. If such a constant was stored into ``partial`` under its
        own id (from the first-atom seed or a later stage bind), this keeps
        only rows where the stored value equals the constant id.

        Returns None if the filter empties the partial (caller should abort
        this rule application).
        """
        cno = self._constant_no
        if not partial:
            return partial
        for av in (bp["arg0_var"], bp["arg1_var"]):
            if av <= cno and av in partial:
                mask = partial[av] == av
                if not bool(mask.any()):
                    return None
                partial = {v: t[mask] for v, t in partial.items()}
        return partial

    # ── Full join (step 0) ────────────────────────────────────────────

    def _apply_rule(
        self, cr: RulePattern, ordered_bps: list,
        prov_ps_off: Tensor, prov_ps_vals: Tensor,
        prov_po_off: Tensor, prov_po_vals: Tensor,
        provable_hashes: Tensor,
    ) -> Optional[Tensor]:
        """Full staged ragged join: all stages use base ∪ provable."""
        if self.join_algo == "leapfrog":
            return self._apply_rule_lftj(
                cr, ordered_bps,
                prov_ps_off, prov_ps_vals, prov_po_off, prov_po_vals, provable_hashes)
        E, E2 = self.E, self.E * self.E
        m = cr.num_body

        bp0 = ordered_bps[0]
        pred0 = bp0["pred_idx"]
        s_b, o_b = self._pred_facts.get(pred0, self._empty)
        s_p, o_p = _pred_pairs_from_ps(pred0, prov_ps_off, prov_ps_vals, E)
        s0 = torch.cat([s_b, s_p]) if s_p.numel() > 0 else s_b
        o0 = torch.cat([o_b, o_p]) if s_p.numel() > 0 else o_b
        if s0.numel() == 0:
            return None
        if bp0["arg0_var"] == bp0["arg1_var"]:
            keep = s0 == o0
            s0, o0 = s0[keep], o0[keep]
            if s0.numel() == 0:
                return None

        partial: Dict[int, Tensor] = {bp0["arg0_var"]: s0, bp0["arg1_var"]: o0}
        # Filter by any constant args in bp0 before projecting to the frontier
        # (projection may drop constant-keyed entries and lose the constraint).
        filtered = self._filter_by_consts(partial, bp0)
        if filtered is None:
            return None
        partial = filtered
        frontiers = _compute_frontiers(cr, ordered_bps)
        partial = {v: t for v, t in partial.items() if v in frontiers[0]}

        def ps_look(pred_k, kv):
            return _ps_expand_combined(
                pred_k, kv,
                self._base_ps_off, self._base_ps_vals,
                prov_ps_off, prov_ps_vals, E)

        def po_look(pred_k, kv):
            return _po_expand_combined(
                pred_k, kv,
                self._base_po_off, self._base_po_vals,
                prov_po_off, prov_po_vals, E)

        return self._run_stages_dispatch(
            cr, partial, frontiers, ps_look, po_look,
            provable_hashes, E, E2, ordered_bps)

    # ── Semi-naive anchored term (step t > 0) ─────────────────────────

    def _apply_rule_anchored(
        self, cr: RulePattern,
        anchor_k: int, join_order: List[int], ordered_bps: list,
        delta_ps_off: Tensor, delta_ps_vals: Tensor,
        delta_po_off: Tensor, delta_po_vals: Tensor, delta_hashes: Tensor,
        prov_ps_off: Tensor, prov_ps_vals: Tensor,
        prov_po_off: Tensor, prov_po_vals: Tensor,
        provable_hashes: Tensor,
    ) -> Optional[Tensor]:
        """One term of the m-term semi-naive formula."""
        E, E2 = self.E, self.E * self.E
        m = cr.num_body
        new_anchor_k = join_order.index(anchor_k)

        if self.join_algo == "leapfrog":
            return self._apply_rule_lftj(
                cr, ordered_bps,
                prov_ps_off, prov_ps_vals, prov_po_off, prov_po_vals, provable_hashes,
                anchor_j=new_anchor_k,
                delta_ps_off=delta_ps_off, delta_ps_vals=delta_ps_vals,
                delta_po_off=delta_po_off, delta_po_vals=delta_po_vals,
                delta_hashes=delta_hashes)

        bp0 = ordered_bps[0]
        pred0 = bp0["pred_idx"]

        if new_anchor_k == 0:
            s0, o0 = _pred_pairs_from_ps(pred0, delta_ps_off, delta_ps_vals, E)
        else:
            s_b, o_b = self._pred_facts.get(pred0, self._empty)
            s_p, o_p = _pred_pairs_from_ps(pred0, prov_ps_off, prov_ps_vals, E)
            if s_p.numel() > 0:
                s0, o0 = torch.cat([s_b, s_p]), torch.cat([o_b, o_p])
            else:
                s0, o0 = s_b, o_b

        if s0.numel() == 0:
            return None
        if bp0["arg0_var"] == bp0["arg1_var"]:
            keep = s0 == o0
            s0, o0 = s0[keep], o0[keep]
            if s0.numel() == 0:
                return None

        partial: Dict[int, Tensor] = {bp0["arg0_var"]: s0, bp0["arg1_var"]: o0}
        # Filter by any constant args in bp0 before projecting to the frontier.
        filtered = self._filter_by_consts(partial, bp0)
        if filtered is None:
            return None
        partial = filtered
        frontiers = _compute_frontiers(cr, ordered_bps)
        partial = {v: t for v, t in partial.items() if v in frontiers[0]}

        def ps_look(k_stage, pred_k, kv):
            if k_stage == new_anchor_k:
                return _ps_expand(pred_k, kv, delta_ps_off, delta_ps_vals, E)
            return _ps_expand_combined(
                pred_k, kv,
                self._base_ps_off, self._base_ps_vals,
                prov_ps_off, prov_ps_vals, E)

        def po_look(k_stage, pred_k, kv):
            if k_stage == new_anchor_k:
                return _po_expand(pred_k, kv, delta_po_off, delta_po_vals, E)
            return _po_expand_combined(
                pred_k, kv,
                self._base_po_off, self._base_po_vals,
                prov_po_off, prov_po_vals, E)

        def case_a_found(k_stage, qh, N_cur):
            nf = self._num_facts
            in_f = torch.zeros(N_cur, dtype=torch.bool, device=qh.device)
            if nf > 0:
                pos_f = torch.searchsorted(self._fact_hashes, qh)
                vf = pos_f < nf
                cf = torch.clamp(pos_f, 0, max(nf - 1, 0))
                in_f = vf & (self._fact_hashes[cf] == qh)
            if k_stage == new_anchor_k:
                if delta_hashes.numel() > 0:
                    n_d = delta_hashes.shape[0]
                    pos_d = torch.searchsorted(delta_hashes, qh)
                    vd = pos_d < n_d
                    cd = torch.clamp(pos_d, 0, max(n_d - 1, 0))
                    return in_f | (vd & (delta_hashes[cd] == qh))
                return in_f
            else:
                if provable_hashes.numel() > 0:
                    n_ph = provable_hashes.shape[0]
                    pos_p = torch.searchsorted(provable_hashes, qh)
                    vp = pos_p < n_ph
                    cp = torch.clamp(pos_p, 0, max(n_ph - 1, 0))
                    return in_f | (vp & (provable_hashes[cp] == qh))
                return in_f

        # Run stages 1..m-1, dispatched by join_algo. Chunking the
        # post-stage-0 partial keeps ``_apply_rule_anchored``'s peak
        # memory bounded on big closures (wn18rr step ≥ 1).
        return self._run_stages_anchored_dispatch(
            cr, partial, frontiers, ordered_bps, m,
            ps_look, po_look, case_a_found, E, E2)

    def _run_stages_anchored_dispatch(
        self, cr, partial, frontiers, ordered_bps, m,
        ps_look, po_look, case_a_found, E, E2,
    ) -> Optional[Tensor]:
        """Dispatch entry for the anchored stage loop.

        ``staged`` runs once over the full partial; ``chunked`` slices
        it. Same closure either way; chunked bounds peak memory.
        """
        if not partial:
            return None
        n = next(iter(partial.values())).shape[0]
        if n == 0:
            return None
        # ``chunked`` slices the post-stage-0 partial so peak memory stays
        # bounded on big closures (wn18rr step ≥ 1).
        if self.join_algo == "chunked":
            chunk = self.join_chunk_size or 100_000
            if n > chunk:
                head_chunks: List[Tensor] = []
                for start in range(0, n, chunk):
                    end = min(start + chunk, n)
                    sliced = {v: t[start:end] for v, t in partial.items()}
                    h = self._run_stages_anchored(
                        cr, sliced, frontiers, ordered_bps, m,
                        ps_look, po_look, case_a_found, E, E2)
                    if h is not None and h.numel() > 0:
                        head_chunks.append(h)
                if not head_chunks:
                    return None
                return torch.cat(head_chunks)
        return self._run_stages_anchored(
            cr, partial, frontiers, ordered_bps, m,
            ps_look, po_look, case_a_found, E, E2)

    def _run_stages_anchored(
        self, cr, partial, frontiers, ordered_bps, m,
        ps_look, po_look, case_a_found, E, E2,
    ) -> Optional[Tensor]:
        """Anchored stage loop (stages 1..m-1) for one chunk of the
        partial bindings tensor. Same logic as the original inline
        loop in ``_apply_rule_anchored`` — extracted so the chunked
        dispatcher can call it per slice.
        """
        for k in range(1, m):
            bpk = ordered_bps[k]
            pred_k = bpk["pred_idx"]
            a0v, a1v = bpk["arg0_var"], bpk["arg1_var"]
            a0_bound = a0v in partial
            a1_bound = a1v in partial

            if not partial or next(iter(partial.values())).shape[0] == 0:
                return None

            if a0_bound and a1_bound:
                sv = partial[a0v]
                ov = partial[a1v]
                qh = pred_k * E2 + sv * E + ov
                found = case_a_found(k, qh, sv.shape[0])
                if not found.any():
                    return None
                partial = {v: t[found] for v, t in partial.items()}
            elif a0_bound:
                ri, ov_v = ps_look(k, pred_k, partial[a0v])
                if ri.numel() == 0:
                    return None
                partial = {v: t[ri] for v, t in partial.items()}
                partial[a1v] = ov_v
            elif a1_bound:
                ri, sv_v = po_look(k, pred_k, partial[a1v])
                if ri.numel() == 0:
                    return None
                partial = {v: t[ri] for v, t in partial.items()}
                partial[a0v] = sv_v
            else:
                return None

            # Filter by any constant args in bpk before projection.
            filtered = self._filter_by_consts(partial, bpk)
            if filtered is None:
                return None
            partial = filtered

            if k < m - 1:
                partial = {v: t for v, t in partial.items()
                           if v in frontiers[k]}

        hx = partial.get(cr.head_var0)
        hy = partial.get(cr.head_var1)
        if hx is None or hy is None:
            return None
        # Emit in COMPACT predicate space — internal hashes (provable,
        # delta, fact) all live there. ``run()`` decompacts at the end
        # via ``_decompact_hashes``.
        head_pred_compact = self._pred_to_compact.get(
            int(cr.head_pred_idx), 0)
        return head_pred_compact * E2 + hx * E + hy

    # ── Shared stage-loop for full join ───────────────────────────────

    def _run_stages_dispatch(
        self, cr, partial, frontiers, ps_look, po_look,
        provable_hashes, E, E2, ordered_bps=None,
    ) -> Optional[Tensor]:
        """Stage-loop entry — dispatches by ``self.join_algo``.

        ``staged`` runs the entire partial through the stage loop in
        one go (the original behaviour). ``chunked`` slices the
        post-stage-0 partial into ``join_chunk_size`` rows and runs
        stages 1..m-1 per slice — same closure, bounded peak memory.
        """
        if not partial:
            return None
        n = next(iter(partial.values())).shape[0]
        if n == 0:
            return None
        if self.join_algo == "chunked":
            chunk = self.join_chunk_size or 100_000
            if n > chunk:
                head_chunks: List[Tensor] = []
                for start in range(0, n, chunk):
                    end = min(start + chunk, n)
                    sliced = {v: t[start:end] for v, t in partial.items()}
                    h = self._run_stages(
                        cr, sliced, frontiers, ps_look, po_look,
                        provable_hashes, E, E2, ordered_bps)
                    if h is not None and h.numel() > 0:
                        head_chunks.append(h)
                if not head_chunks:
                    return None
                return torch.cat(head_chunks)
        return self._run_stages(
            cr, partial, frontiers, ps_look, po_look,
            provable_hashes, E, E2, ordered_bps)

    def _run_stages(
        self, cr, partial, frontiers, ps_look, po_look,
        provable_hashes, E, E2, ordered_bps=None,
    ) -> Optional[Tensor]:
        """Stages 1..m-1 using caller-provided ps_look / po_look."""
        m = cr.num_body
        bps = ordered_bps if ordered_bps is not None else cr.body_patterns

        for k in range(1, m):
            bpk = bps[k]
            pred_k = bpk["pred_idx"]
            a0v, a1v = bpk["arg0_var"], bpk["arg1_var"]
            a0_bound = a0v in partial
            a1_bound = a1v in partial

            if not partial or next(iter(partial.values())).shape[0] == 0:
                return None

            if a0_bound and a1_bound:
                sv = partial[a0v]
                ov = partial[a1v]
                qh = pred_k * E2 + sv * E + ov
                nf = self._num_facts
                in_f = torch.zeros(sv.shape[0], dtype=torch.bool,
                                   device=sv.device)
                if nf > 0:
                    pos_f = torch.searchsorted(self._fact_hashes, qh)
                    vf = pos_f < nf
                    cf = torch.clamp(pos_f, 0, max(nf - 1, 0))
                    in_f = vf & (self._fact_hashes[cf] == qh)
                if provable_hashes.numel() > 0:
                    n_ph = provable_hashes.shape[0]
                    pos_p = torch.searchsorted(provable_hashes, qh)
                    vp = pos_p < n_ph
                    cp = torch.clamp(pos_p, 0, max(n_ph - 1, 0))
                    found = in_f | (vp & (provable_hashes[cp] == qh))
                else:
                    found = in_f
                if not found.any():
                    return None
                partial = {v: t[found] for v, t in partial.items()}
            elif a0_bound:
                ri, ov_v = ps_look(pred_k, partial[a0v])
                if ri.numel() == 0:
                    return None
                partial = {v: t[ri] for v, t in partial.items()}
                partial[a1v] = ov_v
            elif a1_bound:
                ri, sv_v = po_look(pred_k, partial[a1v])
                if ri.numel() == 0:
                    return None
                partial = {v: t[ri] for v, t in partial.items()}
                partial[a0v] = sv_v
            else:
                return None

            # Filter by any constant args in bpk before projection.
            filtered = self._filter_by_consts(partial, bpk)
            if filtered is None:
                return None
            partial = filtered

            if k < m - 1:
                partial = {v: t for v, t in partial.items()
                           if v in frontiers[k]}

        hx = partial.get(cr.head_var0)
        hy = partial.get(cr.head_var1)
        if hx is None or hy is None:
            return None
        # Emit in COMPACT predicate space — internal hashes (provable,
        # delta, fact) all live there. ``run()`` decompacts at the end
        # via ``_decompact_hashes``.
        head_pred_compact = self._pred_to_compact.get(
            int(cr.head_pred_idx), 0)
        return head_pred_compact * E2 + hx * E + hy


__all__ = ["_StagesMixin"]
