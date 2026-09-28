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


class Closure(NamedTuple):
    """Forward chaining's derived atoms: every atom some rule derives within the depth (a fact too, when a rule
    derives it), sorted."""
    atoms: Tensor       # [N, 3]


__all__ = ["Proofs", "Closure"]
