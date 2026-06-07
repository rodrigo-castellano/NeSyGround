"""Canonical STATIC shape symbols for one grounding run.

``Shapes`` is computed once in the factory, carried on ``RunPlan``, never
recomputed by a phase. Data-dependent counts (``S_out``, ``T``) are NOT owned
here — they live on the per-step NamedTuples.

Symbols: ``D`` depth, ``B`` batch, ``N=B*S``, ``G`` goals/state, ``M`` body
atoms, ``A=D*M``, ``S`` states, ``C`` collected budget, ``K_f`` facts/pred,
``K_r`` rules/pred, ``G_r`` groundings/rule, ``K_v`` cands/free-var, ``V`` free
vars, ``E`` entities, ``pad`` padding index. (PBC width ``w`` lives in PBCConfig,
not here.)
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Shapes:
    """Immutable static-shape registry. Every field is a Python int fixed at
    construction; ``N == B*S`` and ``A == D*M`` are enforced."""

    B: int          # batch size — queries per forward (per chunk)
    S: int          # states per proof step (default 256)
    G: int          # goals per state (= M + (M-1)*D)
    M: int          # body atoms per rule (max over the KB's rules)
    D: int          # depth — number of proof steps
    A: int          # accumulated-body capacity (= D*M)
    C: int          # collected-groundings budget per query
    K_f: int        # fact children per state (from the fact index)
    K_r: int        # rules per predicate (from the rule index)
    G_r: int        # groundings per rule per query (the PBC G_r cap)
    K_v: int        # candidates per free variable (= min(K_f, G_r))
    V: int          # free variables per rule (from the rules)
    E: int          # number of entities (the all-entity candidate space)
    N: int          # flattened state count (= B*S)
    pad: int        # padding index — sentinel id (neither entity nor variable)

    def __post_init__(self) -> None:
        if self.N != self.B * self.S:
            raise ValueError(f"Shapes.N ({self.N}) must equal B*S ({self.B*self.S})")
        if self.A != self.D * self.M:
            raise ValueError(f"Shapes.A ({self.A}) must equal D*M ({self.D*self.M})")

    def n(self) -> int:
        """Flattened state count ``N = B*S``."""
        return self.B * self.S

    def a(self) -> int:
        """Accumulated-body capacity ``A = D*M``."""
        return self.D * self.M

    def with_batch(self, B: int) -> "Shapes":
        """Immutable copy rebatched to ``B`` (recomputes ``N``)."""
        return replace(self, B=B, N=B * self.S)


__all__ = ["Shapes"]
