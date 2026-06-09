"""FCDynamic — CPU semi-naive forward chaining via the staged ragged join.

Computes the set of all atoms provable from base facts using the rules. Truly
semi-naive at step t > 0:
    ΔT_r(I, Δ) = ∪_{k=0}^{m-1} { h(θ) | b_k(θ) ∈ Δ_{t-1}, ∀j≠k: b_j(θ) ∈ I_{t-1} }

Setup (predicate compaction + fact indexing + base PS/PO offsets), the
semi-naive ``run`` loop, and the compact→original decode live here; the per-rule
join stages live in ``stages._StagesMixin``. Takes raw tensors (no ns_lib types).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data.rule_index import RulePattern
from grounder.forward.staged.joins import _build_atom_index
from grounder.forward.staged.leapfrog import LeapfrogMixin
from grounder.forward.staged.stages import _StagesMixin


def _sorted_merge(a: Tensor, b: Tensor) -> Tensor:
    """Merge two sorted unique 1-D long tensors into a sorted unique tensor."""
    if a.numel() == 0:
        return b
    if b.numel() == 0:
        return a
    return torch.cat([a, b.to(a.device)]).unique()


class FCDynamic(LeapfrogMixin, _StagesMixin, nn.Module):
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
          * ``'leapfrog'`` — variable-elimination join (``staged/leapfrog.py``).
            Same closure as ``'staged'``; currently NO perf/memory advantage
            (expand-then-filter; the worst-case-optimal intersection core is
            not yet implemented). Opt-in.

        ``join_chunk_size`` (only used by ``'chunked'``): rows per
        chunk in the partial-bindings slicer. ``0`` means
        "auto" — pick a chunk that fits ~1 GiB at the worst-case
        stage fan-out. Typical: 100k–1M.
        """
        super().__init__()
        if join_algo not in ("staged", "chunked", "leapfrog"):
            raise ValueError(
                f"join_algo must be 'staged', 'chunked', or 'leapfrog'; "
                f"got {join_algo!r}")
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
            # head_pred_idx MUST be in the compact map: output hashes
            # are emitted in compact space (see ``_apply_rule`` ->
            # ``head_pred_compact * E2 + ...``) and decompacted at
            # ``run()`` return. Without this, heads that don't appear
            # in any fact or body collide with the compact-0 sentinel
            # and get filtered out.
            used_preds.add(int(cr.head_pred_idx))
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
        from grounder.forward.staged.plan import _compute_join_order
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
            # (e.g. closure-membership tests via ``check_in_fp_global``)
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


__all__ = ["FCDynamic"]
