"""The cross-family seam — request, result contract, the Grounder protocol, Pipeline.

``GroundRequest`` is the single run request (BC reads queries/query_mask/output_spec;
FC reads closure_depth). ``OutputSpec``/``Tier`` are the typed BACKWARD tier set.
``GroundResult`` marks any result (``kind`` + ``as_rule_groundings``); the tag/bridge
are attached to ``BackwardResult`` here. ``Grounder`` is the minimal family protocol
(ground/capability_row/producible_tiers/rebound). ``Pipeline`` composes program
transforms over a base grounder (Transform∘Grounder is itself a Grounder).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    TYPE_CHECKING, ClassVar, FrozenSet, Optional, Protocol, Sequence, runtime_checkable,
)

from torch import Tensor

from grounder.forward.api import Closure
from grounder.base.types import CompletedTreeFirings, GoalState, RuleGroundings

if TYPE_CHECKING:
    from grounder.data.kb import KB
    from grounder.execution.capability import CapabilityRow
    from grounder.transform.api import ProgramTransform


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


# ── Grounder protocol (both families implement it) ──
@runtime_checkable
class Grounder(Protocol):
    kb: "KB"

    def ground(self, request: GroundRequest) -> GroundResult: ...

    def capability_row(self) -> "CapabilityRow": ...   # exec axis (layout x compile)

    def producible_tiers(self) -> FrozenSet[Tier]: ...  # which tiers this family can fill

    def rebound(self, kb: "KB") -> "Grounder": ...      # re-snapshot over a rewritten KB


# ── Pipeline (Transform∘Grounder is itself a Grounder) ──
class Pipeline:
    """A grounder that pre-stages program transforms (order-significant)."""

    def __init__(self, transforms: "Sequence[ProgramTransform]", base: Grounder) -> None:
        self.transforms = tuple(transforms)
        self.base = base

    @property
    def kb(self):
        return self.base.kb

    def ground(self, req: GroundRequest) -> GroundResult:
        kb = self.base.kb
        for t in self.transforms:
            kb, req = t.apply(kb, req)
        return self.base.rebound(kb).ground(req)

    def capability_row(self):
        return self.base.capability_row()

    def producible_tiers(self):
        return self.base.producible_tiers()

    def rebound(self, kb) -> "Pipeline":
        return Pipeline(self.transforms, self.base.rebound(kb))


__all__ = [
    "Tier", "OutputSpec", "GroundRequest", "GroundResult", "Grounder", "Pipeline",
    "BackwardResult", "Closure",
]
