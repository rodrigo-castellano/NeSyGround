"""NeSy hook protocols — injection points in the grounding pipeline.

Four injection points:

ResolutionFactHook
    After fact resolution — scores/filters ground fact candidates.
    Injected inside resolve_sld/rtf after mgu_resolve_facts.

ResolutionRuleHook
    After rule resolution — scores/filters rule candidates.
    Injected inside resolve_sld/rtf after mgu_resolve_rules.

StepHook
    After each STEP — filters/reranks proof states between iterations.
    Injected between step() calls in the canonical loop.

GroundingHook
    After grounding — scores/ranks/filters the final output.
    Injected in forward() after ground().
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from torch import Tensor


@runtime_checkable
class ResolutionFactHook(Protocol):
    """Applied after mgu_resolve_facts — scores/filters ground fact candidates."""

    def filter_facts(
        self,
        fact_goals: Tensor,      # [B, S, K_f, G, 3]
        fact_success: Tensor,    # [B, S, K_f]
        queries: Tensor,         # [B, S, 3] the query atoms that produced these facts
    ) -> Tensor:                 # [B, S, K_f] modified success mask
        """Score/filter fact candidates, return modified success mask."""
        ...


@runtime_checkable
class ResolutionRuleHook(Protocol):
    """Applied after mgu_resolve_rules — scores/filters rule candidates."""

    def filter_rules(
        self,
        rule_goals: Tensor,      # [B, S, K_r, G, 3]
        rule_success: Tensor,    # [B, S, K_r]
        queries: Tensor,         # [B, S, 3] the query atoms
    ) -> Tensor:                 # [B, S, K_r] modified success mask
        """Score/filter rule candidates, return modified success mask."""
        ...


@runtime_checkable
class StepHook(Protocol):
    """After each STEP — filters/reranks proof states."""

    def on_step(
        self,
        body: Tensor,       # [B, tG, G_body, 3]  (accumulated body atoms)
        mask: Tensor,       # [B, tG]
        rule_idx: Tensor,   # [B, tG]
        d: int,             # current depth
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process states after one proof step.

        Returns: (body, mask, rule_idx) — possibly filtered/reranked.
        """
        ...


@runtime_checkable
class GroundingHook(Protocol):
    """After grounding — scores/ranks/filters final output."""

    def apply(
        self,
        body: Tensor,       # [B, tG, G_body, 3]  (accumulated body atoms)
        mask: Tensor,       # [B, tG]
        rule_idx: Tensor,   # [B, tG]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process final groundings.

        Returns: (body, mask, rule_idx) — possibly resized/reordered.
        """
        ...
