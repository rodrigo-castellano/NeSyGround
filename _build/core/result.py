"""The grounder result contract — a marker Protocol over a 2-member tagged union.

``GroundResult`` is the structural marker (``kind`` + ``as_rule_groundings``).
The two members are ``BackwardResult`` (the proof bundle) and ``Closure`` (the FC
result). ``as_rule_groundings()`` is the one optional bridge: BC returns its
firings; FC returns ``None`` (FC has no provenance today).

ADDITIVE (Phase A): ``BackwardResult`` is an ALIAS of today's ``GrounderOutput``
(the Phase-C dataclass rename is NOT done here). The ``kind`` tag +
``as_rule_groundings`` are attached additively to ``GrounderOutput`` — a new class
attribute + method, NOT a dataclass field — so its fields and every construction
site stay byte-identical.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from grounder._build.forward.api import Closure
from grounder._build.types import GrounderOutput, RuleGroundings


@runtime_checkable
class GroundResult(Protocol):
    """Marker for any grounder result. ``kind`` ∈ {"backward","closure"}."""

    kind: str

    def as_rule_groundings(self) -> Optional[RuleGroundings]: ...


# ── BackwardResult: a one-window alias of today's GrounderOutput (tagged) ──
def _backward_as_rule_groundings(self: GrounderOutput) -> Optional[RuleGroundings]:
    return self.rule_groundings


# attach the tag + bridge additively (no new dataclass field -> byte-identical)
GrounderOutput.kind = "backward"
GrounderOutput.as_rule_groundings = _backward_as_rule_groundings

BackwardResult = GrounderOutput   # alias; the dataclass rename is Phase C


__all__ = ["GroundResult", "BackwardResult", "Closure"]
