"""RulePattern — the SOLE owner of per-rule variable binding analysis.

For one rule it computes: which args are head variables vs free variables, each
body argument's binding *source*, and a dependency order in which every body atom
has at least one already-bound argument (so free variables can be enumerated one
at a time). ``pbc.py`` only TENSORIZES these results — it never re-derives them.

A "binding" labels where an argument's value comes from:
  0  = head variable 0          (BINDING_HEAD_VAR0)
  1  = head variable 1          (BINDING_HEAD_VAR1)
  2+k= free variable k          (BINDING_FREE_VAR_OFFSET + k)

BIT-EXACT RECIPES (fingerprint-enforced — the dedup key depends on them):
  * variable test: ``id > constant_no``  (== Encoding.is_var; constant_no = E-1)
  * free vars = sorted(body vars − head vars); free-var index = position there
  * greedy reorder: repeatedly take the first remaining atom with a known arg;
    anchor variants force a chosen atom first then greedy-fill the rest
  * variants keep the ORIGINAL (pre-reorder) body order in ``_orig_*`` so all
    anchor variants of one rule share a canonical body order under the
    ``(orig_rule_idx, head, sorted_body)`` dedup
"""
from __future__ import annotations

from typing import List, Tuple

from torch import Tensor

BINDING_HEAD_VAR0 = 0
BINDING_HEAD_VAR1 = 1
BINDING_FREE_VAR_OFFSET = 2


def atom_enum_meta(body_pattern: dict, known: set) -> dict:
    """Enumeration meta for one body atom given the currently-known bindings.

    Mutates ``known`` to add the single free variable this atom introduces (when
    exactly one arg is still unknown). An atom with neither arg known (only
    reachable as a forced anchor) yields inert ``introduces_fv=-1``.
    """
    b0, b1 = body_pattern["arg0_binding"], body_pattern["arg1_binding"]
    pred = body_pattern["pred_idx"]
    if b0 in known and b1 in known:
        return {"introduces_fv": -1, "enum_bound_src": 0, "enum_direction": 0, "enum_pred": pred}
    if b0 in known:
        known.add(b1)
        return {"introduces_fv": b1 - BINDING_FREE_VAR_OFFSET,
                "enum_bound_src": b0, "enum_direction": 0, "enum_pred": pred}
    if b1 in known:
        known.add(b0)
        return {"introduces_fv": b0 - BINDING_FREE_VAR_OFFSET,
                "enum_bound_src": b1, "enum_direction": 1, "enum_pred": pred}
    return {"introduces_fv": -1, "enum_bound_src": 0, "enum_direction": 0, "enum_pred": pred}


def greedy_body_order(body_patterns: List[dict], remaining: List[int],
                      known: set, order: List[int], meta: List[dict]) -> None:
    """Greedy dependency reorder of the remaining body atoms (in place).

    Repeatedly append the first atom with >=1 known binding (record its meta,
    grow ``known``); when none is reachable, append the rest with inert meta.
    """
    while remaining:
        found = False
        for idx in remaining:
            bp = body_patterns[idx]
            if bp["arg0_binding"] in known or bp["arg1_binding"] in known:
                order.append(idx)
                meta.append(atom_enum_meta(bp, known))
                remaining.remove(idx)
                found = True
                break
        if not found:
            for idx in remaining:
                order.append(idx)
                meta.append(atom_enum_meta(body_patterns[idx], known))
            break


class RulePattern:
    """Per-rule binding pattern + dependency-ordered enumeration metadata."""

    def __init__(self, rule_idx: int, head: Tensor, body: Tensor,
                 body_len: int, constant_no: int) -> None:
        self.rule_idx = rule_idx
        self.constant_no = constant_no
        self.head_pred_idx: int = head[0].item()
        self.head_var0: int = head[1].item()
        self.head_var1: int = head[2].item()
        self.num_body: int = body_len
        self.body_pred_indices: List[int] = [body[j, 0].item() for j in range(body_len)]

        head_vars = {v for v in (self.head_var0, self.head_var1) if v > constant_no}
        body_vars = {body[j, k].item() for j in range(body_len) for k in (1, 2)
                     if body[j, k].item() > constant_no}
        self.free_vars_list = sorted(body_vars - head_vars)
        self.num_free = len(self.free_vars_list)
        self._fv_idx = {v: i for i, v in enumerate(self.free_vars_list)}

        self.body_patterns = [
            {"pred_idx": body[j, 0].item(),
             "arg0_binding": self._binding(body[j, 1].item()),
             "arg1_binding": self._binding(body[j, 2].item()),
             "arg0_var": body[j, 1].item(),
             "arg1_var": body[j, 2].item()}
            for j in range(body_len)]

        # ORIGINAL body order, saved before reorder (anchor variants build off it).
        self._orig_body_patterns = list(self.body_patterns)
        self._orig_body_pred_indices = list(self.body_pred_indices)
        self._reorder_body()

    def _binding(self, val: int) -> int:
        if val == self.head_var0:
            return BINDING_HEAD_VAR0
        if val == self.head_var1:
            return BINDING_HEAD_VAR1
        if val in self._fv_idx:
            return BINDING_FREE_VAR_OFFSET + self._fv_idx[val]
        return BINDING_HEAD_VAR0   # constants bind to head-var-0 slot (inert)

    def _reorder_body(self) -> None:
        known = {BINDING_HEAD_VAR0, BINDING_HEAD_VAR1}
        order: List[int] = []
        meta: List[dict] = []
        greedy_body_order(self.body_patterns, list(range(self.num_body)), known, order, meta)
        self.enum_meta = meta
        self.body_patterns = [self.body_patterns[i] for i in order]
        self.body_pred_indices = [self.body_pred_indices[i] for i in order]

    def reorder_with_anchor(self, forced_first: int) -> Tuple[List, List, List]:
        """Greedy reorder that forces a specific body atom first (off ORIGINAL
        order). Returns ``(body_patterns, body_pred_indices, enum_meta)``;
        does not mutate ``self``."""
        known = {BINDING_HEAD_VAR0, BINDING_HEAD_VAR1}
        order: List[int] = [forced_first]
        meta: List[dict] = [atom_enum_meta(self._orig_body_patterns[forced_first], known)]
        remaining = [i for i in range(self.num_body) if i != forced_first]
        greedy_body_order(self._orig_body_patterns, remaining, known, order, meta)
        return ([self._orig_body_patterns[i] for i in order],
                [self._orig_body_pred_indices[i] for i in order],
                meta)


class PatternVariant:
    """Anchor variant of a RulePattern (duck-typed, propagates canonical order)."""

    __slots__ = ("rule_idx", "head_pred_idx", "num_body", "num_free",
                 "body_patterns", "body_pred_indices", "enum_meta",
                 "_orig_body_patterns", "_orig_body_pred_indices")

    def __init__(self, base: RulePattern, body_patterns: List,
                 body_pred_indices: List, enum_meta: List) -> None:
        self.rule_idx = base.rule_idx
        self.head_pred_idx = base.head_pred_idx
        self.num_body = base.num_body
        self.num_free = base.num_free
        self.body_patterns = body_patterns
        self.body_pred_indices = body_pred_indices
        self.enum_meta = enum_meta
        # Canonical (pre-anchor-reorder) order shared across variants so
        # arg_source_dep / body_preds_dep agree -> the terminal dedup collapses
        # all anchor variants of one logical rule application.
        self._orig_body_patterns = base._orig_body_patterns
        self._orig_body_pred_indices = base._orig_body_pred_indices


def compile_rules(rule_heads: Tensor, rule_bodies: Tensor,
                  rule_lens: Tensor, constant_no: int) -> List[RulePattern]:
    """Raw rule tensors → ``List[RulePattern]`` (for FC / nesy callers)."""
    return [RulePattern(i, rule_heads[i], rule_bodies[i],
                        int(rule_lens[i].item()), constant_no)
            for i in range(rule_heads.size(0))]


__all__ = [
    "RulePattern", "PatternVariant", "compile_rules",
    "atom_enum_meta", "greedy_body_order",
    "BINDING_HEAD_VAR0", "BINDING_HEAD_VAR1", "BINDING_FREE_VAR_OFFSET",
]
