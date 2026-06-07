"""The minimal common abstraction across both grounder families.

A ``Grounder`` is the single runtime verb (``ground``) plus the execution-axis
declaration (``capability_row``), the tier-capability declaration
(``producible_tiers``), and the transform re-snapshot hook (``rebound``).
``nn.Module``-ness is an impl detail of concrete grounders, NOT part of the
protocol. ADDITIVE (Phase A): no family implements ``ground``/``rebound`` yet —
those land in Step 2.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, Protocol, runtime_checkable

from grounder.core.request import GroundRequest, Tier
from grounder.core.result import GroundResult

if TYPE_CHECKING:
    from grounder.data.kb import KB
    from grounder.execution.capability import CapabilityRow


@runtime_checkable
class Grounder(Protocol):
    kb: "KB"

    def ground(self, request: GroundRequest) -> GroundResult: ...

    def capability_row(self) -> "CapabilityRow": ...   # exec axis (layout x compile)

    def producible_tiers(self) -> FrozenSet[Tier]: ...  # which tiers this family can fill

    def rebound(self, kb: "KB") -> "Grounder": ...      # re-snapshot over a rewritten KB


__all__ = ["Grounder"]
