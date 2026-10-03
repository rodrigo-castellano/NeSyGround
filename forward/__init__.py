"""Forward chaining: the atoms the rules derive from the facts, semi-naive, to a fixpoint or a depth.

    closure = Forward(kb, depth=10).closure()
    groundings = Forward(kb).ground(queries)          # [S, Q, 3]: each derived query's witness (forward.witness)

Two engines, one closure: ``spmm`` (per-predicate [E, E] sparse matrices; each step fires every rule on the atoms new at
the last step, or on all of them when those are most) when every rule has a matrix form, else the staged ``join``
(any rule shape, constants in bodies).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.forward.router import run_forward_chaining
from grounder.forward.witness import witnesses
from grounder.ops import canonical, decode, key
from grounder.types import Closure, Groundings


class Forward(nn.Module):
    """Forward chaining over ``kb`` for at most ``depth`` steps (``method="spmm"``: the matrix engine when it applies;
    ``"join"``: always the staged join)."""

    def __init__(self, kb, *, depth: int = 10, method: str = "spmm") -> None:
        super().__init__()
        if method not in ("spmm", "join"):
            raise ValueError(f"method: 'spmm' or 'join', got {method!r}")
        self.kb, self.depth, self.method = kb, depth, method
        self._table = None

    @torch.no_grad()
    def closure(self) -> Closure:
        kb = self.kb
        hashes, n = run_forward_chaining(kb.patterns(), kb.facts.atoms, kb.base, kb.P, self.depth, str(kb.device),
                                         method=self.method)
        return Closure(decode(hashes[:n], kb.base))

    def table(self):
        """The closure's keys (sorted) and each derived atom's witness ``(rule, body, count)``; built once."""
        if self._table is None:
            kb = self.kb
            atoms = self.closure().atoms
            M = max(int(kb.rules.lens.max()) if len(kb.rules) else 1, 1)
            self._table = (key(atoms, kb.base),) + witnesses(kb.patterns(), kb.facts.atoms, atoms,
                                                             constant_no=kb.E - 1, M=M, pad=kb.pad)
        return self._table

    @torch.no_grad()
    def ground(self, queries: Tensor, mask: Optional[Tensor] = None, **_) -> Groundings:
        """The groundings of the pools ``queries [S, Q, 3]`` (or ``[Q, 3]``; ``mask [S, Q]``: the real queries): a
        query forward chaining derives, with a rule body that grounds, has one — its witness (``forward.witness``);
        any other query (a fact only, not derived, a padding or open atom) none."""
        if queries.dim() == 2:
            queries = queries.unsqueeze(0)
            mask = None if mask is None else mask.unsqueeze(0)
        kb, (keys, rule, body, _) = self.kb, self.table()
        S, Q = queries.shape[:2]
        qs = queries.long().reshape(-1, 3).to(keys.device)
        qpool = torch.arange(S, device=qs.device).repeat_interleave(Q)
        live = torch.ones(S * Q, dtype=torch.bool, device=qs.device) if mask is None else mask.reshape(-1).to(qs.device)
        live &= (qs[:, 1:] < kb.E).all(1) & (qs[:, 0] < kb.P) & (qs >= 0).all(1)
        k = key(qs.clamp(min=0), kb.base)
        at = torch.searchsorted(keys, k).clamp(max=max(keys.numel() - 1, 0))
        fire = live & (keys[at] == k) & (rule[at] >= 0) if keys.numel() else torch.zeros_like(live)
        at = at[fire]
        return canonical(rule[at], qs[fire], body[at], qpool[fire], qs, qpool, S, len(kb.rules), kb.base, kb.pad)


__all__ = ["Forward", "run_forward_chaining"]
