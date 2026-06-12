"""NeSy hook protocols — injection points in the grounding pipeline.

Four injection points:

ResolutionFactHook
    After fact resolution — scores/filters ground fact candidates.
    Injected inside resolve_sld/rtf after resolve_facts.

ResolutionRuleHook
    After rule resolution — scores/filters rule candidates.
    Injected inside resolve_sld/rtf after resolve_rules.

StepHook
    After each STEP — filters/reranks proof states between iterations.
    Injected between step() calls in the canonical loop.

GroundingHook
    After grounding — scores/ranks/filters the final output.
    Injected after ground().
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from torch import Tensor


@runtime_checkable
class ResolutionFactHook(Protocol):
    """Applied after resolve_facts — scores/filters ground fact candidates."""

    def filter_facts(
        self,
        fact_goals: Tensor,      # [B, G, K_f, L, 3]
        fact_success: Tensor,    # [B, G, K_f]
        queries: Tensor,         # [B, G, 3] the query atoms that produced these facts
    ) -> Tensor:                 # [B, G, K_f] modified success mask
        """Score/filter fact candidates, return modified success mask."""
        ...


@runtime_checkable
class ResolutionRuleHook(Protocol):
    """Applied after resolve_rules — scores/filters rule candidates."""

    def filter_rules(
        self,
        rule_goals: Tensor,      # [B, G, K_r, L, 3]
        rule_success: Tensor,    # [B, G, K_r]
        queries: Tensor,         # [B, G, 3] the query atoms
    ) -> Tensor:                 # [B, G, K_r] modified success mask
        """Score/filter rule candidates, return modified success mask."""
        ...


@runtime_checkable
class GuidedScorer(Protocol):
    """Atom plausibility prior for KGE-guided grounding (``PBC.guided_topk``).

    Called by the guided join path on GROUND NON-FACT atoms only — facts score
    exactly 1.0 via the fact index and variable atoms are neutral, both handled
    grounder-side (see ``resolution.pbc.guided.GuidedBeam``). Must be
    deterministic and is invoked under ``no_grad``.
    """

    def score_atoms(self, atoms: Tensor) -> Tensor:   # [N, 3] (p, a0, a1) → [N]
        """Probability in (0, 1] per ground atom."""
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


__all__ = [
    "ResolutionFactHook",
    "ResolutionRuleHook",
    "GuidedScorer",
    "StepHook",
    "GroundingHook",
]
