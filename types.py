"""The grounders' outputs."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class Proofs(NamedTuple):
    """SLD's completed proofs: per query, at most ``P`` slots, each a proof's trail — per depth the rule applied (-1: a
    fact step), the goal it resolved (``head``), its body under every later binding, the body's length — and ``mask``,
    the slots that hold one."""
    body: Tensor        # [Q, P, D, M, W]
    rule: Tensor        # [Q, P, D]
    head: Tensor        # [Q, P, D, W]
    count: Tensor       # [Q, P, D]
    mask: Tensor        # [Q, P]

    @staticmethod
    def empty(B: int, P: int, D: int, M: int, W: int, pad: int, device) -> "Proofs":
        return Proofs(torch.zeros(B, P, D, M, W, dtype=torch.long, device=device),
                      torch.full((B, P, D), -1, dtype=torch.long, device=device),
                      torch.full((B, P, D, W), pad, dtype=torch.long, device=device),
                      torch.zeros(B, P, D, dtype=torch.long, device=device),
                      torch.zeros(B, P, dtype=torch.bool, device=device))


class Groundings(NamedTuple):
    """The groundings of ``S`` pools of queries: each pool's atoms (its groundings' atoms, sorted by key, then its
    queries that are none of them, sorted) and its groundings (sorted by rule, then head and body rows), as rows of
    ``atoms``. The groundings of pool ``s`` and rule ``r`` are rows ``offsets[s, r] : offsets[s, r + 1]``; pool ``s``'s
    atoms, ``atom_offsets[s] : atom_offsets[s + 1]``; its queries' rows, ``query[s]``."""
    atoms: Tensor         # [A, 3]
    rule: Tensor          # [N]
    head: Tensor          # [N]
    body: Tensor          # [N, M] (padding slots: a padding atom's row)
    body_mask: Tensor     # [N, M]
    offsets: Tensor       # [S, R + 1]
    atom_offsets: Tensor  # [S + 1]
    query: Tensor         # [S, Q]

    def pool(self, s: int) -> "Groundings":
        """Pool ``s`` alone (rows local to it)."""
        a0, a1 = int(self.atom_offsets[s]), int(self.atom_offsets[s + 1])
        n0, n1 = int(self.offsets[s, 0]), int(self.offsets[s, -1])
        return Groundings(self.atoms[a0:a1], self.rule[n0:n1], self.head[n0:n1] - a0, self.body[n0:n1] - a0,
                          self.body_mask[n0:n1], (self.offsets[s] - n0).unsqueeze(0),
                          self.atom_offsets.new_tensor([0, a1 - a0]), (self.query[s] - a0).unsqueeze(0))


class Closure(NamedTuple):
    """Forward chaining's derived atoms: every atom some rule derives within the depth (a fact too, when a rule
    derives it), sorted."""
    atoms: Tensor       # [N, 3]


__all__ = ["Groundings", "Proofs", "Closure"]
