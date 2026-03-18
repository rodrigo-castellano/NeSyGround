"""Forward chaining — CPU semi-naive FC for provable set computation.

Computes the set of all atoms provable from base facts using the rules.
Uses truly semi-naive formula at step t > 0:
    ΔT_r(I, Δ) = ∪_{k=0}^{m-1} { h(θ) | b_k(θ) ∈ Δ_{t-1},
                                           ∀j≠k: b_j(θ) ∈ I_{t-1} }

Takes raw tensors (no ns_lib domain objects).

Key export:
    run_forward_chaining(compiled_rules, facts_idx, num_entities,
                         num_predicates, depth, device) → (sorted_hashes, n_provable)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data.rule_index import RulePattern


# ══════════════════════════════════════════════════════════════════════
# Sorted merge
# ══════════════════════════════════════════════════════════════════════

def _sorted_merge(a: Tensor, b: Tensor) -> Tensor:
    """Merge two sorted unique 1-D long tensors into a sorted unique tensor."""
    if a.numel() == 0:
        return b
    if b.numel() == 0:
        return a
    return torch.cat([a, b.to(a.device)]).unique()


# ══════════════════════════════════════════════════════════════════════
# Atom index: PS and PO offset arrays from a hash tensor
# ══════════════════════════════════════════════════════════════════════

def _build_atom_index(
    hashes: Tensor, E: int, P: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Hash encoding: pred * E^2 + subj * E + obj.
    Returns ps_offsets [P*E+1], ps_vals, po_offsets [P*E+1], po_vals.
    """
    dev = hashes.device
    E2 = E * E
    N = hashes.numel()
    empty_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
    empty_val = torch.zeros(0, dtype=torch.long, device=dev)
    if N == 0:
        return empty_off, empty_val, empty_off.clone(), empty_val.clone()

    preds = hashes // E2
    rem = hashes % E2
    subjs = rem // E
    objs = rem % E
    ones = torch.ones(N, dtype=torch.long, device=dev)

    ps_keys = preds * E + subjs
    ps_sort = torch.argsort(ps_keys, stable=True)
    ps_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
    ps_off.scatter_add_(0, ps_keys[ps_sort] + 1, ones)
    ps_off = torch.cumsum(ps_off, 0)

    po_keys = preds * E + objs
    po_sort = torch.argsort(po_keys, stable=True)
    po_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
    po_off.scatter_add_(0, po_keys[po_sort] + 1, ones)
    po_off = torch.cumsum(po_off, 0)

    return ps_off, objs[ps_sort], po_off, subjs[po_sort]


# ══════════════════════════════════════════════════════════════════════
# PS/PO lookup helpers
# ══════════════════════════════════════════════════════════════════════

def _pred_pairs_from_ps(
    pred_idx: int, ps_off: Tensor, ps_vals: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """All (subj, obj) pairs for pred_idx from a PS index."""
    dev = ps_off.device
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if ps_vals.numel() == 0:
        return empty, empty
    base = pred_idx * E
    counts = ps_off[base + 1: base + E + 1] - ps_off[base: base + E]
    total = int(counts.sum().item())
    if total == 0:
        return empty, empty
    subjs = torch.repeat_interleave(
        torch.arange(E, dtype=torch.long, device=dev), counts)
    objs = ps_vals[int(ps_off[base].item()): int(ps_off[base + E].item())]
    return subjs, objs


def _ps_expand(
    pred_idx: int, key_vals: Tensor,
    ps_off: Tensor, ps_vals: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """PS lookup: for each subject key, enumerate all objects."""
    dev = ps_off.device
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if ps_vals.numel() == 0 or key_vals.numel() == 0:
        return empty, empty
    N = key_vals.shape[0]
    keys = (pred_idx * E + key_vals).clamp(0, max(ps_off.shape[0] - 2, 0))
    starts = ps_off[keys]
    counts = (ps_off[keys + 1] - starts).clamp(min=0)
    total = int(counts.sum().item())
    if total == 0:
        return empty, empty
    row_ids = torch.repeat_interleave(
        torch.arange(N, dtype=torch.long, device=dev), counts)
    cumcnt = counts.cumsum(0)
    k_idx = torch.arange(total, dtype=torch.long, device=dev) - \
            torch.repeat_interleave(cumcnt - counts, counts)
    val_abs = (starts[row_ids] + k_idx).clamp(0, ps_vals.numel() - 1)
    return row_ids, ps_vals[val_abs]


def _po_expand(
    pred_idx: int, key_vals: Tensor,
    po_off: Tensor, po_vals: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """PO lookup: for each object key, enumerate all subjects."""
    dev = po_off.device
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if po_vals.numel() == 0 or key_vals.numel() == 0:
        return empty, empty
    N = key_vals.shape[0]
    keys = (pred_idx * E + key_vals).clamp(0, max(po_off.shape[0] - 2, 0))
    starts = po_off[keys]
    counts = (po_off[keys + 1] - starts).clamp(min=0)
    total = int(counts.sum().item())
    if total == 0:
        return empty, empty
    row_ids = torch.repeat_interleave(
        torch.arange(N, dtype=torch.long, device=dev), counts)
    cumcnt = counts.cumsum(0)
    k_idx = torch.arange(total, dtype=torch.long, device=dev) - \
            torch.repeat_interleave(cumcnt - counts, counts)
    val_abs = (starts[row_ids] + k_idx).clamp(0, po_vals.numel() - 1)
    return row_ids, po_vals[val_abs]


def _ps_expand_combined(
    pred_idx: int, key_vals: Tensor,
    base_ps_off: Tensor, base_ps_vals: Tensor,
    prov_ps_off: Tensor, prov_ps_vals: Tensor,
    E: int,
) -> Tuple[Tensor, Tensor]:
    """PS lookup in base ∪ provable."""
    dev = base_ps_off.device
    ri_b, ov_b = _ps_expand(pred_idx, key_vals, base_ps_off, base_ps_vals, E)
    ri_p, ov_p = _ps_expand(pred_idx, key_vals, prov_ps_off, prov_ps_vals, E)
    if ri_b.numel() == 0 and ri_p.numel() == 0:
        return (torch.zeros(0, dtype=torch.long, device=dev),
                torch.zeros(0, dtype=torch.long, device=dev))
    if ri_b.numel() == 0:
        return ri_p, ov_p
    if ri_p.numel() == 0:
        return ri_b, ov_b
    return torch.cat([ri_b, ri_p]), torch.cat([ov_b, ov_p])


def _po_expand_combined(
    pred_idx: int, key_vals: Tensor,
    base_po_off: Tensor, base_po_vals: Tensor,
    prov_po_off: Tensor, prov_po_vals: Tensor,
    E: int,
) -> Tuple[Tensor, Tensor]:
    """PO lookup in base ∪ provable."""
    dev = base_po_off.device
    ri_b, sv_b = _po_expand(pred_idx, key_vals, base_po_off, base_po_vals, E)
    ri_p, sv_p = _po_expand(pred_idx, key_vals, prov_po_off, prov_po_vals, E)
    if ri_b.numel() == 0 and ri_p.numel() == 0:
        return (torch.zeros(0, dtype=torch.long, device=dev),
                torch.zeros(0, dtype=torch.long, device=dev))
    if ri_b.numel() == 0:
        return ri_p, sv_p
    if ri_p.numel() == 0:
        return ri_b, sv_b
    return torch.cat([ri_b, ri_p]), torch.cat([sv_b, sv_p])


# ══════════════════════════════════════════════════════════════════════
# Frontier variable sets for staged ragged join
# ══════════════════════════════════════════════════════════════════════

def _compute_join_order(bps, m: int) -> List[int]:
    """Greedy join ordering: start from atom 0, pick next that shares a var."""
    if m <= 1:
        return list(range(m))
    var_sets = [frozenset({bps[k]["arg0_var"], bps[k]["arg1_var"]})
                for k in range(m)]
    order = [0]
    seen = set(var_sets[0])
    remaining = list(range(1, m))
    while remaining:
        for i, idx in enumerate(remaining):
            if var_sets[idx] & seen:
                order.append(idx)
                seen |= var_sets[idx]
                remaining.pop(i)
                break
        else:
            order.extend(remaining)
            break
    return order


def _compute_frontiers(cr: RulePattern, ordered_bps=None) -> List[set]:
    """F_k = head_vars ∪ (seen_vars_0..k ∩ future_vars_{k+1..m-1})."""
    m = cr.num_body
    head_vars = {cr.head_var0, cr.head_var1}
    bps = ordered_bps if ordered_bps is not None else cr.body_patterns[:m]
    future_vars: List[set] = [set() for _ in range(m)]
    for k in range(m - 1):
        for j in range(k + 1, m):
            bp = bps[j]
            future_vars[k].add(bp["arg0_var"])
            future_vars[k].add(bp["arg1_var"])
    frontiers: List[set] = []
    seen_vars: set = set()
    for k in range(m):
        bp = bps[k]
        seen_vars.add(bp["arg0_var"])
        seen_vars.add(bp["arg1_var"])
        frontiers.append(head_vars | (seen_vars & future_vars[k]))
    return frontiers


# ══════════════════════════════════════════════════════════════════════
# FCDynamic — CPU, Python loops, staged ragged join
# ══════════════════════════════════════════════════════════════════════

class FCDynamic(nn.Module):
    """CPU forward chaining — staged ragged join, truly semi-naive.

    Handles all connected rule types including non-chain (fork) rules.

    Args:
        compiled_rules: List of RulePattern from grounder/compilation.py.
        facts_idx: [F, 3] raw fact triples (pred, subj, obj).
        num_entities: Total number of entities.
        num_predicates: Total number of predicates.
        device: Target device (typically 'cpu').
    """

    def __init__(
        self,
        compiled_rules: List[RulePattern],
        facts_idx: Tensor,
        num_entities: int,
        num_predicates: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.compiled_rules = compiled_rules
        dev = str(device)
        self.device_str = dev
        E = num_entities
        P = num_predicates
        self.E = E
        self.P = P

        facts = facts_idx.to(dev)
        fact_preds = facts[:, 0]
        fact_subjs = facts[:, 1]
        fact_objs = facts[:, 2]
        num_facts = facts.shape[0]
        self._num_facts = num_facts

        # Build sorted fact hashes for membership tests
        E2 = E * E
        if num_facts > 0:
            fh = fact_preds * E2 + fact_subjs * E + fact_objs
            self._fact_hashes = fh.sort().values
        else:
            self._fact_hashes = torch.zeros(0, dtype=torch.long, device=dev)

        # Per-predicate fact lists
        _empty = (torch.zeros(0, dtype=torch.long, device=dev),
                  torch.zeros(0, dtype=torch.long, device=dev))
        pred_facts: Dict[int, Tuple[Tensor, Tensor]] = {}
        for cr in compiled_rules:
            for bp in cr.body_patterns[: cr.num_body]:
                p = bp["pred_idx"]
                if p not in pred_facts:
                    mask = fact_preds == p
                    pred_facts[p] = (fact_subjs[mask].clone(),
                                     fact_objs[mask].clone())
        self._pred_facts = pred_facts
        self._empty = _empty

        # Build base PS/PO offset arrays from facts
        if num_facts > 0:
            base_hashes = fact_preds * E2 + fact_subjs * E + fact_objs
        else:
            base_hashes = torch.zeros(0, dtype=torch.long, device=dev)
        (self._base_ps_off, self._base_ps_vals,
         self._base_po_off, self._base_po_vals) = _build_atom_index(
            base_hashes, E, P)

        # Pre-compute greedy join order and ordered body patterns
        self._join_orders: List[List[int]] = []
        self._ordered_bps: List[list] = []
        for cr in compiled_rules:
            order = _compute_join_order(cr.body_patterns, cr.num_body)
            self._join_orders.append(order)
            self._ordered_bps.append([cr.body_patterns[i] for i in order])

    def _filter_new(self, all_new: Tensor, provable_hashes: Tensor) -> Tensor:
        if provable_hashes.numel() == 0:
            return all_new
        n_ph = provable_hashes.shape[0]
        pos = torch.searchsorted(provable_hashes, all_new)
        valid = pos < n_ph
        clamped = torch.clamp(pos, 0, max(n_ph - 1, 0))
        already = valid & (provable_hashes[clamped] == all_new)
        return all_new[~already]

    def _accumulate(
        self, new_hashes_list: List[Tensor], provable_hashes: Tensor,
    ) -> Tensor:
        if not new_hashes_list:
            return torch.zeros(0, dtype=torch.long, device=self.device_str)
        return self._filter_new(
            torch.unique(torch.cat(new_hashes_list)), provable_hashes)

    # ── Full join (step 0) ────────────────────────────────────────────

    def _apply_rule(
        self, cr: RulePattern, ordered_bps: list,
        prov_ps_off: Tensor, prov_ps_vals: Tensor,
        prov_po_off: Tensor, prov_po_vals: Tensor,
        provable_hashes: Tensor,
    ) -> Optional[Tensor]:
        """Full staged ragged join: all stages use base ∪ provable."""
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

        return self._run_stages(
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

        # Stages 1..m-1 in join order
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

            if k < m - 1:
                partial = {v: t for v, t in partial.items()
                           if v in frontiers[k]}

        hx = partial.get(cr.head_var0)
        hy = partial.get(cr.head_var1)
        if hx is None or hy is None:
            return None
        return cr.head_pred_idx * E2 + hx * E + hy

    # ── Internal: shared stage-loop for full join ─────────────────────

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

            if k < m - 1:
                partial = {v: t for v, t in partial.items()
                           if v in frontiers[k]}

        hx = partial.get(cr.head_var0)
        hy = partial.get(cr.head_var1)
        if hx is None or hy is None:
            return None
        return cr.head_pred_idx * E2 + hx * E + hy

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self, depth: int) -> Tuple[Tensor, int]:
        t0 = time.time()
        E, P = self.E, self.P
        dev = self.device_str

        provable_hashes = torch.zeros(0, dtype=torch.long, device=dev)
        prov_ps_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
        prov_ps_vals = torch.zeros(0, dtype=torch.long, device=dev)
        prov_po_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
        prov_po_vals = torch.zeros(0, dtype=torch.long, device=dev)

        delta_hashes = torch.zeros(0, dtype=torch.long, device=dev)
        delta_ps_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
        delta_ps_vals = torch.zeros(0, dtype=torch.long, device=dev)
        delta_po_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
        delta_po_vals = torch.zeros(0, dtype=torch.long, device=dev)

        for step in range(depth):
            new_list: List[Tensor] = []

            for cr_idx, cr in enumerate(self.compiled_rules):
                ordered_bps = self._ordered_bps[cr_idx]
                join_order = self._join_orders[cr_idx]
                if step == 0:
                    hh = self._apply_rule(
                        cr, ordered_bps,
                        prov_ps_off, prov_ps_vals,
                        prov_po_off, prov_po_vals, provable_hashes)
                    if hh is not None:
                        new_list.append(hh)
                else:
                    for anchor_k in range(cr.num_body):
                        hh = self._apply_rule_anchored(
                            cr, anchor_k, join_order, ordered_bps,
                            delta_ps_off, delta_ps_vals,
                            delta_po_off, delta_po_vals, delta_hashes,
                            prov_ps_off, prov_ps_vals,
                            prov_po_off, prov_po_vals, provable_hashes)
                        if hh is not None:
                            new_list.append(hh)

            added = self._accumulate(new_list, provable_hashes)
            if added.numel() == 0:
                break

            provable_hashes = _sorted_merge(provable_hashes, added)
            print(f"    FC step {step}: +{added.numel()} atoms "
                  f"(total {provable_hashes.numel()})")

            delta_hashes = added
            delta_ps_off, delta_ps_vals, delta_po_off, delta_po_vals = \
                _build_atom_index(delta_hashes, E, P)
            prov_ps_off, prov_ps_vals, prov_po_off, prov_po_vals = \
                _build_atom_index(provable_hashes, E, P)

        n_provable = provable_hashes.numel()
        elapsed = time.time() - t0
        print(f"  FC complete: {n_provable} provable atoms ({elapsed:.2f}s)")
        if n_provable > 0:
            return provable_hashes.to(dev), n_provable
        return torch.zeros(1, dtype=torch.long, device=dev), 0


# ══════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════

def run_forward_chaining(
    compiled_rules: List[RulePattern],
    facts_idx: Tensor,
    num_entities: int,
    num_predicates: int,
    depth: int = 10,
    device: str = "cpu",
) -> Tuple[Tensor, int]:
    """Run forward chaining and return (sorted_hashes, n_provable).

    Args:
        compiled_rules: List of RulePattern from grounder/compilation.py.
        facts_idx: [F, 3] raw fact triples.
        num_entities: Total entity count.
        num_predicates: Total predicate count.
        depth: Max FC iterations.
        device: Target device.

    Returns:
        sorted_hashes: 1-D sorted tensor of provable atom hashes.
        n_provable: Number of provable atoms (0 if none).
    """
    fc = FCDynamic(compiled_rules, facts_idx, num_entities, num_predicates,
                   device)
    return fc.run(depth)
