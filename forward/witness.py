"""One grounding per derived atom: the proof forward chaining certifies, as a rule and its body (``witnesses``).

An atom's witness is the first rule of its predicate (the rules' order) whose body grounds in the facts and the derived
atoms, with the first body grounding a depth-first search finds: the body atoms in their dependency order
(``RulePattern.body_patterns``), each free variable walking its values in ascending order, so the binding is the
lexicographically least. An atom no rule body grounds keeps rule -1 and itself as its body.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from grounder.kb import RulePattern

Atom = Tuple[int, int, int]


def _body(patterns: List[dict], env: Dict[int, int], c_no: int, by_s, by_o, by_p) -> Optional[List[Atom]]:
    """The first grounding of ``patterns`` under the bindings ``env`` (a variable is an id above ``c_no``)."""
    if not patterns:
        return []
    bp, rest = patterns[0], patterns[1:]
    p, v0, v1 = bp["pred_idx"], bp["arg0_var"], bp["arg1_var"]
    a0 = v0 if v0 <= c_no else env.get(v0)
    a1 = v1 if v1 <= c_no else env.get(v1)

    def bind(s: int, o: int) -> Optional[List[Atom]]:
        inner = dict(env)
        if v0 > c_no:
            inner[v0] = s
        if v1 > c_no:
            inner[v1] = o
        tail = _body(rest, inner, c_no, by_s, by_o, by_p)
        return None if tail is None else [(p, s, o)] + tail

    if a0 is not None and a1 is not None:
        return bind(a0, a1) if a1 in by_s.get((p, a0), ()) else None
    if a0 is not None:
        pairs = ((a0, o) for o in by_s.get((p, a0), ()))
    elif a1 is not None:
        pairs = ((s, a1) for s in by_o.get((p, a1), ()))
    else:
        pairs = ((s, o) for s, objs in by_p.get(p, ()) for o in objs)
    for s, o in pairs:
        found = bind(s, o)
        if found is not None:
            return found
    return None


@torch.no_grad()
def witnesses(patterns: List[RulePattern], facts: Tensor, atoms: Tensor, *, constant_no: int, M: int,
              pad: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Each atom's witness among ``atoms [N, 3]`` (derived from ``facts [F, 3]`` by ``patterns``): ``(rule [N]`` (-1:
    none), ``body [N, M, 3]`` (``pad`` past its atoms), ``count [N])``. The body atoms are looked up among the facts
    and ``atoms`` together."""
    every = torch.unique(torch.cat([facts, atoms]).long().cpu(), dim=0).tolist()      # sorted (p, s, o)
    by_s: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    by_o: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for p, s, o in every:
        by_s[(p, s)].append(o)
        by_o[(p, o)].append(s)
    by_p: Dict[int, List[Tuple[int, List[int]]]] = defaultdict(list)                  # per predicate, (s, objects)
    for (p, s), objs in by_s.items():
        by_p[p].append((s, objs))
    by_head: Dict[int, List[Tuple[int, RulePattern]]] = defaultdict(list)
    for r, rp in enumerate(patterns):
        by_head[rp.head_pred_idx].append((r, rp))

    N = atoms.shape[0]
    rule = torch.full((N,), -1, dtype=torch.long)
    body = torch.full((N, M, 3), pad, dtype=torch.long)
    count = torch.zeros(N, dtype=torch.long)
    for i, (p, s, o) in enumerate(atoms.long().cpu().tolist()):
        found, r_found = None, -1
        for r, rp in by_head.get(p, ()):
            env: Dict[int, int] = {}
            if rp.head_var0 > constant_no:
                env[rp.head_var0] = s
            elif rp.head_var0 != s:
                continue
            if rp.head_var1 > constant_no:
                env[rp.head_var1] = o
            elif rp.head_var1 != o:
                continue
            found = _body(list(rp.body_patterns), env, constant_no, by_s, by_o, by_p)
            if found is not None:
                r_found = r
                break
        if found is None:
            found = [(p, s, o)]
        found = found[:M]
        rule[i] = r_found
        count[i] = len(found)
        if found:
            body[i, :len(found)] = torch.tensor(found)
    dev = atoms.device
    return rule.to(dev), body.to(dev), count.to(dev)


__all__ = ["witnesses"]
