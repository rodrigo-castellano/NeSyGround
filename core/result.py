"""The grounder result contract — a marker Protocol over a 2-member tagged union.

``GroundResult`` is the structural marker (``kind`` + ``as_rule_groundings``).
The two members are ``BackwardResult`` (the proof bundle) and ``Closure`` (the FC
result). ``as_rule_groundings()`` is the one optional bridge: BC returns its
firings; FC returns ``None`` (FC has no provenance today).

``BackwardResult`` is now the canonical dataclass (renamed from ``GrounderOutput``
in ``types.py``; ``GrounderOutput`` is the one-window alias). The ``kind`` tag +
``as_rule_groundings`` are attached additively here — a class attribute + method,
NOT a dataclass field — so its fields and every construction site stay byte-identical.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from grounder.forward.api import Closure
from grounder.types import BackwardResult, RuleGroundings


@runtime_checkable
class GroundResult(Protocol):
    """Marker for any grounder result. ``kind`` ∈ {"backward","closure"}."""

    kind: str

    def as_rule_groundings(self) -> Optional[RuleGroundings]: ...


# ── BackwardResult (canonical dataclass): attach the tag + bridge ──
def _backward_as_rule_groundings(self: BackwardResult) -> Optional[RuleGroundings]:
    return self.rule_groundings


# attach the tag + bridge additively (no new dataclass field -> byte-identical)
BackwardResult.kind = "backward"
BackwardResult.as_rule_groundings = _backward_as_rule_groundings


__all__ = ["GroundResult", "BackwardResult", "Closure"]
