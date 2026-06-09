"""Canonical STATIC shape symbols for one grounding run.

``Shapes`` is computed once in the factory, carried on ``RunPlan``, never
recomputed by a phase. Data-dependent counts (``G_out``, ``T``) are NOT owned
here — they live on the per-step NamedTuples.

Symbols: ``D`` depth, ``B`` batch, ``N=B*G``, ``L`` atoms/goal, ``M`` body
atoms, ``A=D*M``, ``G`` goals/step, ``Y_q`` collected budget, ``K_f`` facts/pred,
``K_r`` rules/pred, ``Y_r`` groundings/rule, ``K_v`` cands/free-var, ``V`` free
vars, ``E`` entities, ``pad`` padding index. (PBC ``width`` lives on the ``PBC``
config, not here.)
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Shapes:
    """Immutable static-shape registry. Every field is a Python int fixed at
    construction; ``N == B*G`` and ``A == D*M`` are enforced."""

    B: int          # batch size — queries per forward (per chunk)
    G: int          # goals per step — the frontier width (= states)
    L: int          # atoms per goal — the goal length (= M + (M-1)*D)
    M: int          # body atoms per rule (max over the KB's rules)
    D: int          # depth — number of proof steps
    A: int          # accumulated-body capacity (= D*M)
    Y_q: int        # groundings per query — the collected budget
    K_f: int        # fact children per atom (from the fact index)
    K_r: int        # rules per predicate (from the rule index)
    Y_r: int        # groundings per rule per query (the PBC Y_r cap)
    K_v: int        # candidates per free variable (= min(K_f, Y_r))
    V: int          # free variables per rule (from the rules)
    E: int          # number of entities (the all-entity candidate space)
    N: int          # flattened goal count (= B*G)
    pad: int        # padding index — sentinel id (neither entity nor variable)

    def __post_init__(self) -> None:
        if self.N != self.B * self.G:
            raise ValueError(f"Shapes.N ({self.N}) must equal B*G ({self.B*self.G})")
        if self.A != self.D * self.M:
            raise ValueError(f"Shapes.A ({self.A}) must equal D*M ({self.D*self.M})")

    def with_batch(self, B: int) -> "Shapes":
        """Immutable copy rebatched to ``B`` (recomputes ``N``)."""
        return replace(self, B=B, N=B * self.G)


__all__ = ["Shapes"]
