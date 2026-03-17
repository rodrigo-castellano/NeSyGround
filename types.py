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
    grounding_body: Optional[Tensor] = None  # [B, S, G_body, 3] (G_body = depth * M)
    top_ridx: Optional[Tensor] = None       # [B, S]
    fact_counts: Optional[Tensor] = None    # [B]


@dataclass
class GroundingResult:
    """Output of grounding — ground rule instantiations for a batch of queries.

    The body tensor accumulates body atoms from ALL rule applications across
    depths. Each depth step adds up to M body atoms (one rule application),
    so body contains up to depth * M atoms per grounding.
    """
    body: Tensor       # [B, tG, G_body, 3] where G_body = depth * M
    mask: Tensor       # [B, tG]
    count: Tensor      # [B]
    rule_idx: Tensor   # [B, tG]
    body_count: Optional[Tensor] = None  # [B, tG] valid body atoms per grounding
