"""Forward chaining: the atoms the rules derive from the facts, semi-naive, to a fixpoint or a depth.

    closure = Forward(kb, depth=10).closure()

Two engines, one closure: ``spmm`` (per-predicate [E, E] sparse matrices; each step fires every rule on the atoms new at
the last step, or on all of them when those are most) when every rule has a matrix form, else the staged ``join``
(any rule shape, constants in bodies).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from grounder.forward.router import run_forward_chaining
from grounder.ops import decode
from grounder.types import Closure


class Forward(nn.Module):
    """Forward chaining over ``kb`` for at most ``depth`` steps (``method="spmm"``: the matrix engine when it applies;
    ``"join"``: always the staged join)."""

    def __init__(self, kb, *, depth: int = 10, method: str = "spmm") -> None:
        super().__init__()
        if method not in ("spmm", "join"):
            raise ValueError(f"method: 'spmm' or 'join', got {method!r}")
        self.kb, self.depth, self.method = kb, depth, method

    @torch.no_grad()
    def closure(self) -> Closure:
        kb = self.kb
        hashes, n = run_forward_chaining(kb.patterns(), kb.facts.atoms, kb.base, kb.P, self.depth, str(kb.device),
                                         method=self.method)
        return Closure(decode(hashes[:n], kb.base))


__all__ = ["Forward", "run_forward_chaining"]
