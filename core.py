"""The cross-family seam — request and result contract.

``GroundRequest`` is the single run request (BC reads queries/query_mask/output_spec;
FC reads closure_depth). ``OutputSpec``/``Tier`` are the typed BACKWARD tier set.
``GroundResult`` marks any result (``kind`` + ``as_rule_groundings``); the tag/bridge
are attached to ``BackwardResult`` here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, FrozenSet, Optional, Protocol, runtime_checkable

from torch import Tensor

from grounder.types import Closure
from grounder.base.types import CompletedTreeFirings, GoalState, RuleGroundings


# ── Tier / OutputSpec / GroundRequest ──
class Tier(StrEnum):
    """The BACKWARD output tiers."""

    PROOF_STATE = "proof_state"   # default base (DpRL); GoalState — omit to skip final-frontier packing
    FIRINGS = "firings"           # RuleGroundings (torch-ns run_bc)
    TREES = "trees"               # completed-tree firings (probfol)


@dataclass(frozen=True)
class OutputSpec:
    """Per-call BC request as a typed tier set. PROOF_STATE is the DEFAULT base;
    a FIRINGS-only spec legally omits it — the engine then skips the final
    step's pack/postprocess and the GoalState build entirely (the firings are
    captured at resolve time, before pack, so the output is unaffected)."""

    tiers: FrozenSet[Tier] = frozenset({Tier.PROOF_STATE})

    def needs_provenance(self) -> bool:
        return bool(self.tiers & {Tier.FIRINGS, Tier.TREES})

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
    """The single runtime request. ``queries=None`` is data-directed (FC); BC carries
    queries + an ``OutputSpec``. FC reads only ``closure_depth``."""

    queries: Optional[Tensor] = None          # [B,3]; None => data-directed (FC)
    query_mask: Optional[Tensor] = None       # [B]
    output_spec: OutputSpec = field(default_factory=OutputSpec)  # BC tiers
    excluded_queries: Optional[Tensor] = None
    closure_depth: Optional[int] = None       # FC-only depth override (None for BC)


# ── GroundResult contract + the BackwardResult envelope ──
@runtime_checkable
class GroundResult(Protocol):
    """Marker for any grounder result. ``kind`` ∈ {"backward","closure"}."""

    kind: str

    def as_rule_groundings(self) -> Optional[RuleGroundings]: ...


@dataclass(frozen=True)
class BackwardResult:
    """The backward proof bundle — each field present iff its OutputSpec tier was
    requested. Satisfies ``GroundResult`` (``kind`` + ``as_rule_groundings``)."""

    goal_state: Optional[GoalState] = None
    completed_tree_firings: Optional[CompletedTreeFirings] = None
    rule_groundings: Optional[RuleGroundings] = None
    kind: ClassVar[str] = "backward"

    def as_rule_groundings(self) -> Optional[RuleGroundings]:
        return self.rule_groundings


__all__ = [
    "Tier", "OutputSpec", "GroundRequest", "GroundResult", "BackwardResult", "Closure",
]
