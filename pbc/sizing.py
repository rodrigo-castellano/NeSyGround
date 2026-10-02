"""PBC's worst case per goal: how many goals and groundings one goal can lead to at each step, from the rules and facts.

A multi-type branching model over the grounder's own enumeration (torch-ns ``docs/grounding_limits.md`` describes the
approach). A goal ``p(u, v)`` at a step is grounded by each rule variant of predicate ``p``: its head binds ``X, Y`` to
``u, v``; the variant's body atoms are taken in its order, an atom with one bound argument **looked up** (its free
variable ranges over the facts' values), one with both bound **checked**. A variant's candidates on the goal are the
product of its lookups' sizes. Its checked atoms whose predicate some rule concludes are the next step's goals (at
width >= 1 a kept grounding's unknown atoms; at width 0 none).

Worst sizes are typed by where an entity comes from: ``ANY`` (a goal's argument: any entity), or ``(q, a)`` (an entity
that a lookup reached: one appearing as argument ``a`` of predicate ``q``). A lookup of predicate ``q`` with argument
``a`` bound, from an entity of type ``t``, has at most ``max over the entities x of t of |{facts q(.., x at a, ..)}|``
values, and reaches type ``(q, other argument)``. Counts propagate linearly over the goal types (predicate and its
arguments' types), from one goal of predicate ``p`` with both arguments ``ANY``; no deduplication, so they bound what a
step holds. Exact integers.

At a width-0 step (the last step of fp_batch's BC_{w,d}) a rule's kept groundings are its all-fact ones, the same set
whichever variant enumerates them: the engine enumerates one variant per rule there (``last_variant``), and the model
counts that variant's candidates. At a width-w step every candidate may be kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

ANY = "any"


@dataclass(frozen=True)
class Variant:
    """One rule variant as the sizing sees it: ``lookups`` in order, each ``(pred, bound source, side, new free
    variable)`` (side 1: the subject bound, 2: the object), and ``checked`` atoms ``(pred, source 0, source 1)``; a
    source is 0 / 1 (the head's arguments) or ``2 + k`` (free variable ``k``)."""
    rule: int
    head: int
    lookups: Tuple[Tuple[int, int, int, int], ...]
    checked: Tuple[Tuple[int, int, int], ...]


def variants(patterns) -> List[Variant]:
    """Every anchor variant of every rule (``RulePattern`` s), in the engine's order: rule-major, then anchor."""
    out = []
    for p in patterns:
        for j in range(p.num_body):
            bps, preds, meta = p.reorder_with_anchor(j)
            lookups, checked = [], []
            for bp, m in zip(bps, meta):
                if m["introduces_fv"] >= 0:
                    side = 1 if m["enum_direction"] == 0 else 2
                    lookups.append((m["enum_pred"], m["enum_bound_src"], side, 2 + m["introduces_fv"]))
                else:
                    checked.append((bp["pred_idx"], bp["arg0_binding"], bp["arg1_binding"]))
            out.append(Variant(p.rule_idx, p.head_pred_idx, tuple(lookups), tuple(checked)))
    return out


class Lookups:
    """The worst lookup sizes of a fact set: ``worst(q, side, t)`` for a lookup of predicate ``q`` with argument ``side``
    bound to an entity of type ``t``."""

    def __init__(self, facts) -> None:
        P, base = facts.P, facts.base
        # count[a][q, x]: the facts of q with x as argument a (1: subject, 2: object)
        self.count = {a: facts.offsets[a - 1].diff().view(P, base) for a in (1, 2)}
        self._memo: Dict[Tuple[int, int, object], int] = {}

    def worst(self, q: int, side: int, t) -> int:
        k = (q, side, t)
        if k not in self._memo:
            c = self.count[side][q]
            if t == ANY:
                self._memo[k] = int(c.max())
            else:
                tp, ta = t
                self._memo[k] = int(torch.where(self.count[ta][tp] > 0, c, 0).max())
        return self._memo[k]


@dataclass
class Sizing:
    """Per root predicate ``p`` and step ``k``: the goals entering the step, its candidates (walked) and its kept
    groundings (stored), in the worst case, from one goal of predicate ``p``."""
    widths: Tuple[int, ...]
    goals: Dict[int, List[int]]
    candidates: Dict[int, List[int]]
    kept: Dict[int, List[int]]

    def worst(self, field: str) -> List[int]:
        """Per step, the largest over the root predicates."""
        t = getattr(self, field)
        return [max(v[k] for v in t.values()) for k in range(len(self.widths))] if t else [0] * len(self.widths)

    def bytes_per_goal(self, goal_bytes: int, kept_bytes: int) -> int:
        """The most bytes one goal needs: ``goal_bytes`` per goal and ``kept_bytes`` per kept grounding, summed over
        the steps, the largest over the root predicates."""
        return max((sum(goal_bytes * g + kept_bytes * k for g, k in zip(self.goals[p], self.kept[p]))
                    for p in self.goals), default=0)


def size(kb, widths: Sequence[int], last_variant: Dict[int, int], roots: Sequence[int] | None = None) -> Sizing:
    """The worst case per goal over ``len(widths)`` steps (``widths[k]``: step ``k``'s width; ``last_variant[r]``: the
    one variant of rule ``r`` a width-0 step enumerates, an index into ``variants``), for goals of each predicate in
    ``roots`` (default: every predicate some rule concludes)."""
    vs = variants(kb.patterns())
    by_head: Dict[int, List[int]] = {}
    for i, v in enumerate(vs):
        by_head.setdefault(v.head, []).append(i)
    heads = set(by_head)
    look = Lookups(kb.facts)

    def ground(i: int, t0, t1) -> Tuple[int, List[tuple]]:
        """Variant ``i`` on one goal with argument types ``t0, t1``: (its candidates, their checked atoms' types)."""
        v = vs[i]
        types, n = {0: t0, 1: t1}, 1
        for q, src, side, new in v.lookups:
            n *= look.worst(q, side, types[src])
            types[new] = (q, 3 - side)
        return n, [(q, types.get(a, ANY), types.get(b, ANY)) for q, a, b in v.checked]

    goals_out, cand_out, kept_out = {}, {}, {}
    for p in (roots if roots is not None else sorted(heads)):
        frontier: Dict[tuple, int] = {(p, ANY, ANY): 1}
        goals, cand, kept = [], [], []
        for k, w in enumerate(widths):
            goals.append(sum(frontier.values()))
            c = kk = 0
            nxt: Dict[tuple, int] = {}
            for (q, t0, t1), m in frontier.items():
                for i in by_head.get(q, ()):
                    if w == 0 and last_variant.get(vs[i].rule, i) != i:
                        continue
                    n, checked = ground(i, t0, t1)
                    c += m * n
                    kk += m * n
                    if w > 0 and k < len(widths) - 1:
                        for g in checked:
                            if g[0] in heads:
                                nxt[g] = nxt.get(g, 0) + m * n
            cand.append(c)
            kept.append(kk)
            frontier = nxt
        goals_out[p], cand_out[p], kept_out[p] = goals, cand, kept
    return Sizing(tuple(widths), goals_out, cand_out, kept_out)


def cheapest_variants(kb) -> Dict[int, int]:
    """Per rule, the variant with the smallest worst candidate count on a goal with both arguments ``ANY`` (the first
    such): what a width-0 step enumerates."""
    vs, look, best = variants(kb.patterns()), Lookups(kb.facts), {}
    for i, v in enumerate(vs):
        types, n = {0: ANY, 1: ANY}, 1
        for q, src, side, new in v.lookups:
            n *= look.worst(q, side, types[src])
            types[new] = (q, 3 - side)
        if v.rule not in best or n < best[v.rule][0]:
            best[v.rule] = (n, i)
    return {r: i for r, (_, i) in best.items()}


__all__ = ["ANY", "Variant", "variants", "Lookups", "Sizing", "size", "cheapest_variants"]
