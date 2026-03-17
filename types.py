"""Result types for the grounder package.

Output API:
  ProofState    — where the proof search is (for RL action selection)
  ProofEvidence — how we got here (accumulated body atoms for scoring)
  GrounderOutput — unified return from forward()

Internal pipeline types (NamedTuples for torch.compile safety):
  ResolvedChildren — output of RESOLVE phase
  PackedStates     — output of PACK phase
  SyncParams       — metadata for SYNC phase
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════
# Output API
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ProofState:
    """Where we are: snapshot of the proof search after the last step.

    Used by RL for action selection (read proof_goals to pick next resolution).
    """
    proof_goals: Tensor     # [B, S, G, 3]  — remaining goals per branch
    state_valid: Tensor     # [B, S]         — which branches are alive
    top_ridx:    Tensor     # [B, S]         — first rule applied per branch


@dataclass
class ProofEvidence:
    """How we got here: accumulated proof trace from completed groundings.

    Used by reasoners for scoring (score body atoms, aggregate per grounding).
    body_count is a compressed mask: atoms 0..body_count-1 are real,
    body_count..G_body-1 are padding (sentinel inside, count-mask outside).
    """
    body:       Tensor      # [B, tG, G_body, 3] — all body atoms across depths
    mask:       Tensor      # [B, tG]             — valid groundings
    count:      Tensor      # [B]                 — groundings per query
    rule_idx:   Tensor      # [B, tG]             — which rule per grounding
    body_count: Tensor      # [B, tG]             — valid atoms per grounding


@dataclass
class GrounderOutput:
    """Unified return from forward(). Consumers pick what they need.

    RL (depth=1): reads output.state.proof_goals
    Reasoning (depth=D): reads output.evidence.body
    """
    state:    ProofState
    evidence: ProofEvidence


# Backward compat alias — prefer ProofEvidence in new code.
GroundingResult = ProofEvidence


# ═══════════════════════════════════════════════════════════════════════
# Internal pipeline types (NamedTuples — torch.compile safe, iterable)
# ═══════════════════════════════════════════════════════════════════════


class ResolvedChildren(NamedTuple):
    """Output of RESOLVE phase — fact and rule children from unification."""
    fact_goals:   Tensor   # [B, S, K_f, G, 3]
    fact_gbody:   Tensor   # [B, S, K_f, M, 3]
    fact_success: Tensor   # [B, S, K_f]
    rule_goals:   Tensor   # [B, S, K_r, G, 3]
    rule_gbody:   Tensor   # [B, S, K_r, M, 3]
    rule_success: Tensor   # [B, S, K_r]
    sub_rule_idx: Tensor   # [B, S, K_r]
    fact_subs:    Tensor   # [B, S, K_f, 2, 2]
    rule_subs:    Tensor   # [B, S, K_r, 2, 2]


class PackedStates(NamedTuple):
    """Output of PACK phase — compacted proof states + sync metadata."""
    grounding_body: Tensor  # [B, S_out, M, 3]
    proof_goals:    Tensor  # [B, S_out, G, 3]
    top_ridx:       Tensor  # [B, S_out]
    state_valid:    Tensor  # [B, S_out]
    body_count:     Tensor  # [B, S_out]
    parent_map:     Tensor  # [B, S_out]
    winning_subs:   Tensor  # [B, S_out, 2, 2]
    has_new_body:   Tensor  # [B, S_out]


class SyncParams(NamedTuple):
    """Metadata for SYNC phase — how to update accumulated_body."""
    parent_map:    Tensor  # [B, S_out]
    winning_subs:  Tensor  # [B, S_out, 2, 2]
    has_new_body:  Tensor  # [B, S_out]
    parent_bcount: Tensor  # [B, S_out]
