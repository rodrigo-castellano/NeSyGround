"""PBC's rule tables: every anchor variant of every rule, as the tensors the step reads.

A rule ``h :- b_1, ..., b_m`` has one variant per body atom forced first (its anchor), the rest in dependency order: a
goal's groundings where the first atom is unknown but a later one a fact are found by the variant anchored on that
later atom. Per variant ``i`` (rule-major, then anchor):

    rule [R']                 the rule it varies (its id in the KB)
    n_body [R'], has_free [R']
    fv_pred, fv_src, fv_dir, fv_valid [R', V]   free variable k (in enumeration order): the lookup that binds it — its
                              predicate, the column of its bound argument (0 / 1: the head's arguments, 2 + j: free
                              variable j), direction (0: the subject bound, 1: the object), whether the variant has it
    arg_src [R', M, 2]        each body argument's column, in the rule's given body order (so every variant of a rule
                              writes its groundings' body columns alike)
    body_pred [R', M]         the body's predicates, in the given order
    by_pred [P, K], by_pred_mask [P, K]   the variants of head predicate p (n_by_pred [P]: how many)
    heads [P]                 the predicates some rule concludes
    fv_used                   per free variable, whether some variant has it
    enumerated [R', M]        the body atoms a lookup draws from the facts (never tested: facts by construction)

and per rule (for the binding filter of the groundings): ``bind_head [R]``, ``bind_body [R, M]`` (the predicates),
``bind_on [R, 2 + 2M]`` (the argument slots in use), ``bind_first [R, 2 + 2M]`` (the first slot holding the same
variable: a repeated variable binds one constant).
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from grounder.kb import FREE, PatternVariant


class Tables:
    def __init__(self, kb) -> None:
        dev, pad = kb.device, kb.pad
        patterns = kb.patterns()
        variants: List[PatternVariant] = [PatternVariant(p, *p.reorder_with_anchor(j))
                                          for p in patterns for j in range(p.num_body)]
        R, M = max(len(variants), 1), max((v.num_body for v in variants), default=1)
        V = max(max((v.num_free for v in variants), default=0), 1)
        self.M, self.V = M, V
        rule = torch.zeros(R, dtype=torch.long)
        head = torch.zeros(R, dtype=torch.long)
        n_body = torch.zeros(R, dtype=torch.long)
        has_free = torch.zeros(R, dtype=torch.bool)
        fv_pred, fv_src, fv_dir = (torch.zeros(R, V, dtype=torch.long) for _ in range(3))
        fv_valid = torch.zeros(R, V, dtype=torch.bool)
        arg_src = torch.zeros(R, M, 2, dtype=torch.long)
        body_pred = torch.zeros(R, M, dtype=torch.long)
        for i, v in enumerate(variants):
            rule[i], head[i], n_body[i], has_free[i] = v.rule_idx, v.head_pred_idx, v.num_body, v.num_free > 0
            dep = {}                                    # free variable -> its enumeration position
            for m in v.enum_meta:
                if m["introduces_fv"] >= 0:
                    dep[m["introduces_fv"]] = len(dep)
            for m in v.enum_meta:
                if m["introduces_fv"] >= 0 and dep[m["introduces_fv"]] < V:
                    k = dep[m["introduces_fv"]]
                    src = m["enum_bound_src"]
                    fv_pred[i, k], fv_dir[i, k], fv_valid[i, k] = m["enum_pred"], m["enum_direction"], True
                    fv_src[i, k] = FREE + dep[src - FREE] if src >= FREE else src
            for j in range(min(len(v._orig_body_patterns), M)):
                bp = v._orig_body_patterns[j]
                body_pred[i, j] = v._orig_body_pred_indices[j]
                for a, name in enumerate(("arg0_binding", "arg1_binding")):
                    val = bp[name]
                    if val >= FREE and val - FREE in dep:
                        val = FREE + dep[val - FREE]
                    arg_src[i, j, a] = val
        P = kb.P
        by_pred, offsets = _csr(head[:len(variants)], P)
        K = max(int(offsets.diff().max()), 1)
        pos = torch.arange(K)
        starts = offsets[:-1].unsqueeze(1)
        self.by_pred = by_pred[(starts + pos).clamp(0, max(len(variants) - 1, 0))]
        self.by_pred_mask = pos < offsets.diff().unsqueeze(1)
        self.n_by_pred = offsets.diff()
        heads = torch.zeros(P, dtype=torch.bool)
        heads[torch.tensor([p.head_pred_idx for p in patterns], dtype=torch.long)] = True
        self.rule, self.n_body, self.has_free = rule, n_body, has_free
        self.fv_pred, self.fv_src, self.fv_dir, self.fv_valid = fv_pred, fv_src, fv_dir, fv_valid
        self.arg_src, self.body_pred, self.heads = arg_src, body_pred, heads
        self.fv_used = tuple(bool(fv_valid[:, k].any()) for k in range(V))
        # [R', M]: the body atoms a free variable's lookup draws from the facts (facts by construction)
        arg, known = arg_src.clamp(max=1 + V), torch.zeros(R, M, dtype=torch.bool)
        for k in range(V):
            by_subject = fv_dir[:, k] == 0
            pair = torch.stack([torch.where(by_subject, fv_src[:, k], 2 + k),
                                torch.where(by_subject, 2 + k, fv_src[:, k])], -1)          # [R', 2]
            known |= (fv_valid[:, k].unsqueeze(1) & (body_pred == fv_pred[:, k].unsqueeze(1))
                      & (arg == pair.unsqueeze(1)).all(-1))
        self.enumerated = known & (torch.arange(M) < n_body.unsqueeze(1))
        self.variants = variants
        self._binding(kb, M, pad)
        for name, t in vars(self).items():
            if isinstance(t, Tensor):
                setattr(self, name, t.to(dev))

    def _binding(self, kb, M: int, pad: int) -> None:
        """The per-rule binding filter's tables (over the KB's rules)."""
        heads, bodies, lens = kb.rules.heads.cpu(), kb.rules.bodies.cpu(), kb.rules.lens.cpu()
        R, n = heads.shape[0], 2 + 2 * M
        self.bind_head = heads[:, 0].clone()
        self.bind_body = torch.full((R, M), pad, dtype=torch.long)
        self.bind_on = torch.zeros(R, n, dtype=torch.bool)
        self.bind_first = torch.arange(n).repeat(R, 1)
        for r in range(R):
            L = int(lens[r])
            var = [int(heads[r, 1]), int(heads[r, 2])] + [-(s + 1) for s in range(2, n)]
            self.bind_on[r, :2] = True
            for m in range(min(L, M)):
                self.bind_body[r, m] = int(bodies[r, m, 0])
                var[2 + 2 * m], var[3 + 2 * m] = int(bodies[r, m, 1]), int(bodies[r, m, 2])
                self.bind_on[r, 2 + 2 * m] = self.bind_on[r, 3 + 2 * m] = True
            first = {}
            for s in range(n):
                if self.bind_on[r, s]:
                    self.bind_first[r, s] = first.setdefault(var[s], s)


def _csr(keys: Tensor, size: int):
    order = torch.argsort(keys, stable=True)
    offsets = torch.zeros(size + 1, dtype=torch.long)
    offsets[1:] = torch.bincount(keys, minlength=size).cumsum(0)
    return order, offsets


__all__ = ["Tables"]
