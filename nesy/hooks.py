"""NeSy hook protocols — injection points in the grounding pipeline.

Three injection points:

ResolutionHook
    During RESOLVE — scores/filters entity candidates or rules.
    Injected inside resolve_sld/rtf/enum.

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
class ResolutionHook(Protocol):
    """During RESOLVE — scores entity candidates or filters rules."""

    def score_candidates(
        self, candidates: Tensor, context: Tensor,
    ) -> Tensor:
        """Score entity candidates.

        Args:
            candidates: [N, K] candidate entity indices.
            context:    [N, 3] query atoms providing context.

        Returns:
            scores: [N, K] candidate scores.
        """
        ...


@runtime_checkable
class StepHook(Protocol):
    """After each STEP — filters/reranks proof states."""

    def on_step(
        self,
        body: Tensor,       # [B, tG, M, 3]
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
        body: Tensor,       # [B, tG, M, 3]
        mask: Tensor,       # [B, tG]
        rule_idx: Tensor,   # [B, tG]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process final groundings.

        Returns: (body, mask, rule_idx) — possibly resized/reordered.
        """
        ...
