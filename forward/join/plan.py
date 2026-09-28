"""Join planning for the staged ragged join — body-atom order + frontier vars.

``_compute_join_order`` picks a greedy connected order over a rule's body atoms;
``_compute_frontiers`` computes, per stage, the variable set that must be kept
alive (head vars plus any seen var still needed by a future atom) so the stage
loop can project away dead bindings.
"""
from __future__ import annotations

from typing import List

from grounder.data.rule_index import RulePattern


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


__all__ = ["_compute_join_order", "_compute_frontiers"]
