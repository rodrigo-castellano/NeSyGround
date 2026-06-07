"""SldRuleIndex — the rule index for sld/rtf resolution.

sld/rtf resolve a goal against rule *heads* by predicate, then unify (MGU); they
need only the base sorted-storage + segment lookup, no binding analysis. This is
the index the KB builds (resolution-agnostic); ``pbc`` builds its own on top.
The named subclass keeps the resolution→index mapping explicit (sld/rtf →
SldRuleIndex, pbc → PbcRuleIndex) without adding behavior.
"""
from __future__ import annotations

from grounder._build.data.rule_index.base import RuleIndex


class SldRuleIndex(RuleIndex):
    """Segment-lookup rule index for sld/rtf (no binding analysis)."""


__all__ = ["SldRuleIndex"]
