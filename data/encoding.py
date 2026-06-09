"""Encoding — the single source of truth for entity/variable id classification.

Contract (reconciles the loader's and probfol grounder_adapter's conventions):
``E`` is the FIRST variable id (== ``constant_no + 1``). An id is a CONSTANT iff
``id < E``, a VARIABLE iff ``id >= E and id != pad``; ``pad`` is inert (neither).

Both id conventions agree on this boundary:
  - 1-based loader: entities ``1..constant_no``  → var_start = constant_no+1 = E
  - probfol 0-based: entities ``0..constant_no`` (constant_no = num_entities-1)
                     → var_start = constant_no+1 = num_entities = E
so ``Encoding`` is always built with ``E = constant_no + 1`` regardless of base.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class Encoding:
    """Frozen id convention for one KB. ``E`` = first variable id; ``pad`` inert."""

    E: int          # var-start: first variable id (= constant_no + 1)
    pad: int

    @classmethod
    def from_constant_no(cls, constant_no: int, pad: int) -> "Encoding":
        """Build from the loader/KB ``constant_no`` (highest constant id, any base)."""
        return cls(E=constant_no + 1, pad=pad)

    def is_const(self, ids: Tensor) -> Tensor:        # [...] -> [...] bool
        return ids < self.E

    def is_var(self, ids: Tensor) -> Tensor:          # [...] -> [...] bool
        return (ids >= self.E) & (ids != self.pad)


__all__ = ["Encoding"]
