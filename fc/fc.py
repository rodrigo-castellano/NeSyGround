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
        *,
        join_algo: str = "staged",
        join_chunk_size: int = 0,
    ) -> None:
        """
        ``join_algo``:
          * ``'staged'`` (default) — naive staged ragged join, the
            original implementation. Per-stage intermediate is
            ``|partial| × fan-out`` which can blow memory on
            high-fan-out KBs.
          * ``'chunked'`` — same staged algorithm, but process the
            partial-bindings tensor in slices of size
            ``join_chunk_size``. Bounds peak memory per stage at the
            cost of more Python iterations. Same closure as
            ``'staged'`` (verified by smoke tests).
          * ``'leapfrog'`` — Leapfrog Triejoin (Veldhuizen 2014).
            Variable-elimination style: per variable, intersect
            sorted iterators across the atoms that bind it. Output
            is AGM-bounded — for chain queries
            ``|out| ≤ |result|``, never the Cartesian. ~10× smaller
            peak intermediate on transitive 2/3-body rules.
            Same closure as ``'staged'`` (verified by smoke tests).

        ``join_chunk_size`` (only used by ``'chunked'``): rows per
        chunk in the partial-bindings slicer. ``0`` means
        "auto" — pick a chunk that fits ~1 GiB at the worst-case
        stage fan-out. Typical: 100k–1M.
        """
        super().__init__()
        if join_algo not in ("staged", "chunked", "leapfrog"):
            raise ValueError(
                f"join_algo must be 'staged', 'chunked', or "
                f"'leapfrog'; got {join_algo!r}")
        self.join_algo = join_algo
        self.join_chunk_size = int(join_chunk_size)
        self.compiled_rules = compiled_rules
        dev = str(device)
        self.device_str = dev
        E = num_entities
        # Compact the predicate space to only the predicates that
        # actually have facts or appear in rule bodies. Without this,
        # ``num_predicates`` reflects ``padding_idx + 1`` which can be
        # ~num_entities (e.g. 40573 for wn18rr) — and the per-stage
        # ``[P * E + 1]`` offset array blows past 12 GiB. Compacting
        # to actual-used predicates collapses ``P`` to ~11 for
        # wn18rr, giving a ~3700× memory reduction at no
        # correctness cost. ``head_pred_idx`` stays in the original
        # space so the output ``head_hashes`` are decoded correctly
        # by callers using the original predicate ids.
        used_preds = set()
        if facts_idx.numel() > 0:
            used_preds.update(facts_idx[:, 0].tolist())
        for cr in compiled_rules:
            for bp in cr.body_patterns[: cr.num_body]:
                used_preds.add(int(bp["pred_idx"]))
            # head_pred_idx kept original — only used for output hash
            # multiplication, doesn't index the offset arrays.
        # Always reserve compact 0 for "no predicate" (sentinel /
        # padding). Original predicates remap above that.
        sorted_preds = [-1] + sorted(p for p in used_preds if p >= 0)
        self._pred_to_compact: Dict[int, int] = {
            p: i for i, p in enumerate(sorted_preds) if p >= 0}
        # Predicates that don't appear get mapped to 0 (sentinel) —
        # they have no facts so the offset slice is empty anyway.
        P = len(sorted_preds)
        self.E = E
        self.P = P
        # Original predicate count for any external callers / debug.
        self.P_orig = num_predicates
        # constant_no: any body-arg value <= constant_no is a constant
        # (entity id), not a logical variable. Used by _filter_by_consts.
        self._constant_no = (int(compiled_rules[0].constant_no)
                             if compiled_rules else 0)

        facts = facts_idx.to(dev)
        # Translate fact predicates to the compact space. Internal
        # tensors (_fact_hashes, base_ps/po, _pred_facts) all live in
        # the compact space so the offset arrays stay [P_compact*E+1]
        # instead of [P_orig*E+1].
        if facts.numel() > 0 and self._pred_to_compact:
            fact_preds_orig = facts[:, 0]
            # Build a translation lookup tensor [P_orig+1] mapping
            # original pred id → compact id (0 for unmapped).
            max_orig = int(fact_preds_orig.max().item())
            tt_size = max(max_orig + 1, num_predicates)
            translate = torch.zeros(tt_size, dtype=torch.long, device=dev)
            for orig, compact in self._pred_to_compact.items():
                if 0 <= orig < tt_size:
                    translate[orig] = compact
            self._pred_translate = translate
            fact_preds = translate[fact_preds_orig]
        else:
            self._pred_translate = torch.zeros(
                num_predicates, dtype=torch.long, device=dev)
            fact_preds = facts[:, 0] if facts.numel() > 0 else \
                torch.zeros(0, dtype=torch.long, device=dev)
        fact_subjs = facts[:, 1] if facts.numel() > 0 else \
            torch.zeros(0, dtype=torch.long, device=dev)
        fact_objs = facts[:, 2] if facts.numel() > 0 else \
            torch.zeros(0, dtype=torch.long, device=dev)
        num_facts = facts.shape[0]
        self._num_facts = num_facts

        # Build sorted fact hashes for membership tests (compact space)
        E2 = E * E
        if num_facts > 0:
            fh = fact_preds * E2 + fact_subjs * E + fact_objs
            self._fact_hashes = fh.sort().values
        else:
            self._fact_hashes = torch.zeros(0, dtype=torch.long, device=dev)

        # Per-predicate fact lists, keyed by COMPACT predicate id.
        _empty = (torch.zeros(0, dtype=torch.long, device=dev),
                  torch.zeros(0, dtype=torch.long, device=dev))
        pred_facts: Dict[int, Tuple[Tensor, Tensor]] = {}
        for cr in compiled_rules:
            for bp in cr.body_patterns[: cr.num_body]:
                p_orig = int(bp["pred_idx"])
                p = self._pred_to_compact.get(p_orig, 0)
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

        # Pre-compute greedy join order and ordered body patterns.
        # Translate ``pred_idx`` to the compact predicate space so the
        # internal lookups (offset arrays, fact hashes) stay
        # bounded by ``P_compact`` instead of ``num_predicates``.
        self._join_orders: List[List[int]] = []
        self._ordered_bps: List[list] = []
        self._head_pred_compact: List[int] = []
        for cr in compiled_rules:
            order = _compute_join_order(cr.body_patterns, cr.num_body)
            self._join_orders.append(order)
            bps_orig = [cr.body_patterns[i] for i in order]
            bps_compact = []
            for bp in bps_orig:
                bp_c = dict(bp)
                bp_c["pred_idx"] = self._pred_to_compact.get(
                    int(bp["pred_idx"]), 0)
                bps_compact.append(bp_c)
            self._ordered_bps.append(bps_compact)
            self._head_pred_compact.append(
                self._pred_to_compact.get(int(cr.head_pred_idx), 0))

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

    def _apply_rule_lftj(
        self, cr: RulePattern, ordered_bps: list,
        prov_ps_off: Tensor, prov_ps_vals: Tensor,
        prov_po_off: Tensor, prov_po_vals: Tensor,
        provable_hashes: Tensor,
    ) -> Optional[Tensor]:
        """Leapfrog Triejoin for full join (step 0).

        Variable elimination order: most-shared variable first
        (Generic-Join heuristic). For each variable v:

          1. Find an atom A binding v whose other argument is already
             constrained by ``partial`` (or by a constant in the
             rule); use ``ps_expand`` / ``po_expand`` against A's
             facts to get v's candidate set per binding row.
          2. Apply every other atom binding v as a fact-membership
             filter. Surviving rows have *all* atoms binding v
             satisfied — this is the AGM-bound output.

        For binary-atom 2-body chain rules this collapses to the
        same answer as staged joins; the WCO advantage materialises
        when 3+ atoms share a variable (triangle-style queries).
        Same closure as staged on every dataset (verified by smoke
        test).
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

        def domain(pred_k: int, role: int) -> Tensor:
            """Sorted unique values at ``role`` of ``pred_k`` across
            base facts ∪ derived (provable) facts."""
            s_b, o_b = self._pred_facts.get(pred_k, self._empty)
            s_p, o_p = _pred_pairs_from_ps(
                pred_k, prov_ps_off, prov_ps_vals, E)
            if role == 0:
                return torch.unique(
                    torch.cat([s_b, s_p]) if s_p.numel() else s_b)
            return torch.unique(
                torch.cat([o_b, o_p]) if o_p.numel() else o_b)

        def ps_lookup(pred_k: int, kv: Tensor):
            return _ps_expand_combined(
                pred_k, kv,
                self._base_ps_off, self._base_ps_vals,
                prov_ps_off, prov_ps_vals, E)

        def po_lookup(pred_k: int, kv: Tensor):
            return _po_expand_combined(
                pred_k, kv,
                self._base_po_off, self._base_po_vals,
                prov_po_off, prov_po_vals, E)

        def in_kb(qh: Tensor) -> Tensor:
            """Is ``qh`` in facts ∪ provable_hashes?"""
            n_q = qh.shape[0]
            in_f = torch.zeros(n_q, dtype=torch.bool, device=qh.device)
            nf = self._num_facts
            if nf > 0:
                pos = torch.searchsorted(self._fact_hashes, qh)
                v = pos < nf
                cl = pos.clamp(max=max(nf - 1, 0))
                in_f = v & (self._fact_hashes[cl] == qh)
            if provable_hashes.numel() > 0:
                n_ph = provable_hashes.shape[0]
                pos_p = torch.searchsorted(provable_hashes, qh)
                v_p = pos_p < n_ph
                cl_p = pos_p.clamp(max=max(n_ph - 1, 0))
                return in_f | (v_p & (provable_hashes[cl_p] == qh))
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
                               body[jr[0]]["pred_idx"], jr[1]).numel())
                j, role = best
                v_dom = domain(body[j]["pred_idx"], role)
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
                ri, v_vals = ps_lookup(bp["pred_idx"], source_arr)
            else:
                ri, v_vals = po_lookup(bp["pred_idx"], source_arr)
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
                ok = in_kb(qh)
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

    def _apply_rule(
        self, cr: RulePattern, ordered_bps: list,
        prov_ps_off: Tensor, prov_ps_vals: Tensor,
        prov_po_off: Tensor, prov_po_vals: Tensor,
        provable_hashes: Tensor,
    ) -> Optional[Tensor]:
        """Full staged ragged join: all stages use base ∪ provable.

        Dispatches to ``_apply_rule_lftj`` when ``join_algo='leapfrog'``.
        """
        if self.join_algo == "leapfrog":
            return self._apply_rule_lftj(
                cr, ordered_bps,
                prov_ps_off, prov_ps_vals,
                prov_po_off, prov_po_vals, provable_hashes)
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
        # ``leapfrog`` mode also chunks the anchored / staged
        # post-stage-0 partial — without this the per-iteration
        # cumulative state on wn18rr's 33M-atom delta saturates the
        # GPU even though LFTJ is used at step 0.
        if self.join_algo in ("chunked", "leapfrog"):
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

    # ── Internal: shared stage-loop for full join ─────────────────────

    def _run_stages_dispatch(
        self, cr, partial, frontiers, ps_look, po_look,
        provable_hashes, E, E2, ordered_bps=None,
    ) -> Optional[Tensor]:
        """Stage-loop entry — dispatches by ``self.join_algo``.

        ``staged`` runs the entire partial through the stage loop in
        one go (the original behaviour). ``chunked`` slices the
        post-stage-0 partial into ``join_chunk_size`` rows and runs
        stages 1..m-1 per slice — same closure, bounded peak memory.
        ``leapfrog`` ignores ``partial`` and re-builds the join via
        Leapfrog Triejoin (variable-elimination, AGM-bound output).
        """
        if not partial:
            return None
        n = next(iter(partial.values())).shape[0]
        if n == 0:
            return None
        if self.join_algo == "leapfrog":
            return self._run_stages_leapfrog(
                cr, partial, frontiers, ps_look, po_look,
                provable_hashes, E, E2, ordered_bps)
        # ``leapfrog`` mode also chunks the anchored / staged
        # post-stage-0 partial — without this the per-iteration
        # cumulative state on wn18rr's 33M-atom delta saturates the
        # GPU even though LFTJ is used at step 0.
        if self.join_algo in ("chunked", "leapfrog"):
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

    def _run_stages_leapfrog(
        self, cr, partial, frontiers, ps_look, po_look,
        provable_hashes, E, E2, ordered_bps=None,
    ) -> Optional[Tensor]:
        """Leapfrog Triejoin / Generic Join (placeholder).

        The full WCO algorithm requires per-predicate trie iterators
        (sorted by ``arg0`` then ``arg1``) and a leapfrog intersect
        across the iterators bound to each variable. This is the
        substantial part of the change — see _build_leapfrog_state
        and _leapfrog_iter below.

        Until that's wired up, fall through to the staged path so
        the closure remains correct.
        """
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

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self, depth: int) -> Tuple[Tensor, int]:
        t0 = time.time()
        E, P = self.E, self.P
        E2 = E * E
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
            # ``provable_hashes`` is in compact predicate space —
            # decompact to original space before returning so callers
            # (e.g. BCGrounder.fp_global_hashes membership tests)
            # see the same predicate ids the rest of the codebase
            # uses.
            decompact = self._decompact_hashes(provable_hashes, E, E2)
            decompact_sorted = decompact.sort().values
            return decompact_sorted.to(dev), n_provable
        return torch.zeros(1, dtype=torch.long, device=dev), 0

    def _decompact_hashes(self, hashes: Tensor, E: int, E2: int) -> Tensor:
        """Translate ``compact_pred * E^2 + s * E + o`` →
        ``orig_pred * E^2 + s * E + o`` for every entry."""
        if hashes.numel() == 0:
            return hashes
        compact_pred = hashes // E2
        rest = hashes - compact_pred * E2
        # Inverse map: compact_id → original predicate id.
        # ``self._pred_to_compact`` is original→compact; build the
        # inverse once and cache.
        if not hasattr(self, "_compact_to_pred"):
            inv = torch.zeros(self.P, dtype=torch.long,
                              device=hashes.device)
            for orig, compact in self._pred_to_compact.items():
                if 0 <= compact < self.P:
                    inv[compact] = orig
            self._compact_to_pred = inv
        orig_pred = self._compact_to_pred[compact_pred.clamp(
            max=self.P - 1)]
        return orig_pred * E2 + rest


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
    *,
    method: str = "spmm",
    join_algo: str = "staged",
    join_chunk_size: int = 0,
) -> Tuple[Tensor, int]:
    """Run forward chaining and return (sorted_hashes, n_provable).

    Args:
        compiled_rules: List of RulePattern from grounder/compilation.py.
        facts_idx: [F, 3] raw fact triples.
        num_entities: Total entity count.
        num_predicates: Total predicate count.
        depth: Max FC iterations.
        device: Target device.
        method: Forward-chaining strategy.
            * ``'spmm'`` (default) — semi-naive sparse-matrix-multiplication
              FC. Per-predicate ``[E, E]`` sparse CSR matrices; rules
              compile to MATMUL / ELEM_AND / CASE_A / EXIST_AND ops on
              those matrices with delta-tracked semi-naive iteration.
              Scales to 100M+ atoms on commodity hardware (validated
              on fb15k237). Falls back to ``'staged'`` automatically
              when the rule set has any 3-body rule (SpMM doesn't
              support those).
            * ``'staged'`` — staged ragged join (original FCDynamic).
              Slower per atom and bounded by Python dispatch overhead;
              use for rule sets with 3+ body atoms or when SpMM
              classifies a rule as ``UNSUPPORTED``.
        join_algo: ``staged``-only knob. Per-rule join strategy:
            ``'staged'`` (default), ``'chunked'``, or ``'leapfrog'``.
        join_chunk_size: Rows per chunk in the ``staged`` chunked
            path. ``0`` = the FCDynamic default (100k rows).

    Returns:
        sorted_hashes: 1-D sorted tensor of provable atom hashes.
        n_provable: Number of provable atoms (0 if none).
    """
    if method == "spmm":
        # SpMM doesn't support num_body >= 3 — auto-fallback so callers
        # don't have to know which rule sets are 3-body.
        from grounder.fc.spmm import run_forward_chaining_spmm
        if not any(getattr(cr, "num_body", 0) >= 3 for cr in compiled_rules):
            return run_forward_chaining_spmm(
                compiled_rules, facts_idx, num_entities, num_predicates,
                depth=depth, device=device)
        # Fall through to staged for 3-body rule sets.

    fc = FCDynamic(compiled_rules, facts_idx, num_entities, num_predicates,
                   device,
                   join_algo=join_algo,
                   join_chunk_size=join_chunk_size)
    return fc.run(depth)
