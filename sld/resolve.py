"""SLD resolution of one goal per state, for atoms of any width ``W`` (``[pred, a1, ..., a_n]``).

    unify          pairwise most general unifier of two atoms
    substitute     apply ``(from, to)`` bindings, in order
    resolve_facts  the goal against the facts: the facts sharing its first constant argument (a lookup), unified
    resolve_rules  the goal against the rule heads of its predicate, each rule's variables renamed apart first

A constant is an id ``< E``, a variable an id ``>= E`` other than ``pad``. Fixed shapes, no host syncs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def is_const(ids: Tensor, E: int) -> Tensor:
    return ids < E


def is_var(ids: Tensor, E: int, pad: int) -> Tensor:
    return (ids >= E) & (ids != pad)


@torch.no_grad()
def unify(a: Tensor, b: Tensor, E: int, pad: int) -> Tuple[Tensor, Tensor]:
    """The most general unifier of atoms ``a`` and ``b`` ``[..., W]``: ``(ok [...], subs [..., W - 1, 2])``, a
    ``(from, to)`` binding per argument (``pad``: none). Fails on a predicate mismatch, two different constants, or one
    variable bound to two different terms."""
    pad_t = torch.tensor(pad, dtype=a.dtype, device=a.device)
    n = a.shape[-1] - 1
    qa, ta = a[..., 1:], b[..., 1:]
    ok = (a[..., 0] == b[..., 0]) & ~(is_const(qa, E) & is_const(ta, E) & (qa != ta)).any(-1)
    bind_q = is_var(qa, E, pad) & is_const(ta, E)                     # a's variable to b's constant
    bind_t = is_var(ta, E, pad) & (qa != pad)                          # b's variable to a's term
    frm = torch.where(bind_q, qa, torch.where(bind_t, ta, pad_t))
    to = torch.where(bind_q, ta, torch.where(bind_t, qa, pad_t))
    subs = torch.stack([frm, to], -1)                                  # [..., n, 2]
    for i in range(n):
        for j in range(i + 1, n):
            same = (subs[..., i, 0] == subs[..., j, 0]) & (subs[..., i, 0] != pad)
            ok = ok & ~(same & (subs[..., i, 1] != subs[..., j, 1]))
    return ok, torch.where((~ok)[..., None, None], pad_t, subs)


@torch.no_grad()
def substitute(atoms: Tensor, subs: Tensor, pad: int) -> Tensor:
    """``atoms [N, L, W]`` with the bindings ``subs [N, S, 2]`` applied in order (binding ``s`` sees binding ``s - 1``'s
    result, so chains of variables resolve)."""
    if atoms.numel() == 0:
        return atoms
    N = atoms.shape[0]
    out = atoms[:, :, 1:]
    none = torch.tensor(-1, dtype=out.dtype, device=out.device)       # no argument is -1
    for s in range(subs.shape[1]):
        frm = torch.where(subs[:, s, 0] != pad, subs[:, s, 0], none).view(N, 1, 1)
        out = torch.where(out == frm, subs[:, s, 1].view(N, 1, 1), out)
    return torch.cat([atoms[:, :, :1], out], 2)


def fact_lookup(kb, goals: Tensor, K: int) -> Tuple[Tensor, Tensor]:
    """``(fact [N, K], valid [N, K])``: the facts sharing each goal's first constant argument (all the facts of its
    predicate when it has none), the first ``K``, in the facts' order."""
    facts, E, pad = kb.facts, kb.E, kb.pad
    pos = torch.arange(K, device=goals.device)
    pred = goals[:, 0]
    in_range = (pred >= 0) & (pred < facts.P)
    p = torch.where(in_range, pred, 0)
    start = facts.pred_offsets[p]
    count = facts.pred_offsets[p + 1] - start
    ok = in_range & (pred != pad)

    def rows(s: Tensor) -> Tensor:
        return (s.unsqueeze(1) + pos).clamp(max=len(facts) - 1)
    fact = rows(start)                                    # the predicate's facts: rows in order
    for j in reversed(range(1, goals.shape[1])):          # the first constant argument wins
        const = (goals[:, j] < E) & (goals[:, j] != pad)
        s, c = facts.lookup(pred, goals[:, j], j)
        fact = torch.where(const.unsqueeze(1), facts.order[j - 1][rows(s)], fact)
        count, ok = torch.where(const, c, count), ok | const
    return fact, (pos < count.clamp(max=K).unsqueeze(1)) & ok.unsqueeze(1)


def resolve_facts(kb, goals: Tensor, remaining: Tensor, active: Tensor, K: int,
                  excluded: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Each goal ``[N, W]`` against the facts: ``(children [N, K, L, W], ok [N, K], subs [N, K, W - 1, 2])``, a child
    being the ``remaining [N, L, W]`` goals under the fact's bindings. ``excluded [N, 1, W]``: an atom never read as a
    fact (the root query)."""
    N, W = goals.shape
    L = remaining.shape[1]
    fact, valid = fact_lookup(kb, goals, K)
    atoms = kb.facts.atoms[fact.reshape(-1)].view(N, K, W)
    ok, subs = unify(goals.unsqueeze(1).expand(-1, K, -1), atoms, kb.E, kb.pad)
    ok = ok & valid & active.unsqueeze(1)
    if excluded is not None:
        ok = ok & ~(atoms == excluded[:, 0, :].unsqueeze(1)).all(-1)
    rem = remaining.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, L, W)
    children = substitute(rem, subs.reshape(N * K, W - 1, 2), kb.pad).view(N, K, L, W)
    return torch.where(ok.view(N, K, 1, 1), children, kb.pad), ok, subs


def resolve_rules(kb, goals: Tensor, remaining: Tensor, active: Tensor, K: int,
                  var_base: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Each goal ``[N, W]`` against the heads of the (at most ``K``) rules of its predicate, each rule's variables
    renamed apart to ``var_base [N] + (variable - E)``: ``(children [N, K, L, W], ok [N, K], rule [N, K],
    subs [N, K, W - 1, 2])``, a child being the rule's body then the ``remaining [N, L, W]`` goals, under the head's
    bindings. A child whose goals would not fit ``L`` atoms is dropped."""
    N, W = goals.shape
    L = remaining.shape[1]
    E, pad, rules = kb.E, kb.pad, kb.rules
    rule, valid = rules.lookup(goals[:, 0], K)                                     # [N, K]
    M = rules.M
    heads, bodies, lens = rules.heads[rule], rules.bodies[rule], rules.lens[rule]  # [N, K, W], [N, K, M, W], [N, K]
    base = var_base.view(N, 1, 1)
    heads = torch.cat([heads[..., :1], torch.where(heads[..., 1:] >= E, heads[..., 1:] - E + base, heads[..., 1:])],
                      -1)
    base = base.unsqueeze(-1)
    bodies = torch.cat([bodies[..., :1], torch.where(bodies[..., 1:] >= E, bodies[..., 1:] - E + base,
                                                     bodies[..., 1:])], -1)
    ok, subs = unify(goals.unsqueeze(1).expand(-1, K, -1), heads, E, pad)
    ok = ok & valid & active.unsqueeze(1)
    flat = subs.reshape(N * K, W - 1, 2)
    body = substitute(bodies.reshape(N * K, M, W), flat, pad).view(N, K, M, W)
    n_rem = L - M
    if n_rem < L:                                   # the goals past n_rem would not fit behind the body
        ok = ok & ~(remaining[:, n_rem:, 0] != pad).any(-1).unsqueeze(1)
    rem = remaining[:, :n_rem].unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, n_rem, W)
    rem = substitute(rem, flat, pad).view(N, K, n_rem, W)
    body = torch.where((torch.arange(M, device=goals.device) >= lens.unsqueeze(-1)).unsqueeze(-1), pad, body)
    return torch.cat([body, rem], 2), ok, rule, subs


__all__ = ["unify", "substitute", "fact_lookup", "resolve_facts", "resolve_rules", "is_const", "is_var"]
