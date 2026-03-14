"""NeSy hook protocols — applied during/after resolution or as provability replacement.

Hook protocols define the interface for plugging neural components into the
grounding pipeline. They are orthogonal to FOL grounding logic.

Three protocol types:
    ResolutionHook:     scores/filters entity candidates during resolution.
    PostResolutionHook: scores/ranks final groundings after resolution.
    ProvabilityHook:    replaces hard provability with soft scores.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class ResolutionHook(Protocol):
    """Applied during resolution — scores/filters entity candidates."""

    def score_candidates(self, candidates: Tensor, context: Tensor) -> Tensor:
        """Score entity candidates.

        Args:
            candidates: [N, K] candidate entity indices.
            context:    [N, 3] query atoms providing context.

        Returns:
            scores: [N, K] candidate scores.
        """
        ...


@runtime_checkable
class PostResolutionHook(Protocol):
    """Applied after resolution — scores/ranks final groundings."""

    def score_groundings(
        self,
        body: Tensor,      # [B, tG, M, 3]
        mask: Tensor,       # [B, tG]
        ridx: Tensor,       # [B, tG]
    ) -> Tensor:
        """Score groundings for ranking.

        Args:
            body: [B, tG, M, 3] grounded body atoms.
            mask: [B, tG] validity mask.
            ridx: [B, tG] rule indices.

        Returns:
            scores: [B, tG] grounding scores.
        """
        ...


@runtime_checkable
class ProvabilityHook(Protocol):
    """Replaces hard provability with soft scores."""

    def provability_score(self, atoms: Tensor) -> Tensor:
        """Compute soft provability scores.

        Args:
            atoms: [..., 3] atom triples.

        Returns:
            scores: [...] soft provability in [0, 1].
        """
        ...
