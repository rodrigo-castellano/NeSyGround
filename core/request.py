"""The unified run request — ``GroundRequest`` + the typed ``OutputSpec``/``Tier``.

ADDITIVE (Phase A): nothing in the engine consumes these yet. The engine still
reads the BC-only ``state.OutputSpec`` (3 bools); Step 2 threads ``tiers`` through
plan/loop/finalize and reconciles the two.

``Tier`` is the BACKWARD tier set only; FC does not use it (FC asks for a closure
depth via ``GroundRequest.closure_depth``, produces a ``Closure``). ``OutputSpec``
generalizes the 3 bools to a typed ``FrozenSet[Tier]`` named ``tiers`` (NOT
``cells`` — that collides with ``CapabilityRow.cells``), keeping
``groundings``/``firings``/``trees`` as back-compat @property for one window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import FrozenSet, Optional

from torch import Tensor


class Tier(StrEnum):
    """The BACKWARD output tiers (no overloaded PROOF_STATE, no FC PROVABLE_SET)."""

    PROOF_STATE = "proof_state"   # always-on base (DpRL); ProofState
    FIRINGS = "firings"           # RuleGroundings (torch-ns run_bc)
    TREES = "trees"               # completed-tree firings (probfol)


@dataclass(frozen=True)
class OutputSpec:
    """Per-call BC request as a typed tier set. ``PROOF_STATE`` is the always-on
    base; ``FIRINGS``/``TREES`` add accumulators (require provenance)."""

    tiers: FrozenSet[Tier] = frozenset({Tier.PROOF_STATE})

    def needs_provenance(self) -> bool:
        return bool(self.tiers & {Tier.FIRINGS, Tier.TREES})

    def wants(self, t: Tier) -> bool:
        return t in self.tiers

    # back-compat @property for ONE migration window (drop with the bools in Phase C)
    @property
    def groundings(self) -> bool:
        return Tier.PROOF_STATE in self.tiers

    @property
    def firings(self) -> bool:
        return Tier.FIRINGS in self.tiers

    @property
    def trees(self) -> bool:
        return Tier.TREES in self.tiers


@dataclass(frozen=True)
class GroundRequest:
    """The single runtime request — unifies the two run shapes. ``queries=None``
    is data-directed (FC); BC carries queries + an ``OutputSpec``. FC reads only
    ``closure_depth`` (its single knob)."""

    queries: Optional[Tensor] = None          # [B,3]; None => data-directed (FC)
    query_mask: Optional[Tensor] = None       # [B]
    output_spec: OutputSpec = field(default_factory=OutputSpec)  # BC tiers
    excluded_queries: Optional[Tensor] = None
    closure_depth: Optional[int] = None       # FC-only depth override (None for BC)

    def requires_queries(self) -> bool:
        return self.queries is not None


__all__ = ["Tier", "OutputSpec", "GroundRequest"]
