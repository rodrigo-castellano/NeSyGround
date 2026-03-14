"""Result types for the grounder package."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from torch import Tensor


@dataclass
class ResolveResult:
    """Output of unify_with_facts / unify_with_rules."""
    states: Tensor          # [B, K, G, 3] derived states
    success: Tensor         # [B, K] validity mask
    subs: Tensor            # [B, K, 2, 2] substitutions
    body_lens: Optional[Tensor] = None  # [B, K] rule body lengths (rules only)


@dataclass
class PackResult:
    """Output of pack_combined."""
    derived: Tensor         # [B, K, M, 3] compacted states
    counts: Tensor          # [B] valid count per batch


@dataclass
class StepResult:
    """Output of one backward-chaining proof step."""
    proof_goals: Tensor              # [B, S, G, 3]
    next_var_indices: Tensor         # [B]
    state_valid: Tensor              # [B, S]
    grounding_body: Optional[Tensor] = None  # [B, S, M, 3]
    top_ridx: Optional[Tensor] = None       # [B, S]
    fact_counts: Optional[Tensor] = None    # [B]


@dataclass
class ForwardResult:
    """Output of multi-depth forward (TS grounding collection).

    Alias: GroundingResult (preferred for new code).
    """
    collected_body: Tensor   # [B, tG, M, 3]
    collected_mask: Tensor   # [B, tG]
    collected_count: Tensor  # [B]
    collected_ridx: Tensor   # [B, tG]


# Alias for new code — ForwardResult is kept for backward compat.
GroundingResult = ForwardResult
