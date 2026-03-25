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
from typing import Dict, NamedTuple, Optional

import torch

from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════
# Output API
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ProofState:
    """Where we are: snapshot of the proof search after the last step.

    Used by RL for action selection (read proof_goals to pick next resolution).
    next_var_indices is populated when standardization is configured — it
    tracks the free variable counter for multi-step resolution.
    """
    proof_goals: Tensor              # [B, S, G, 3]  — remaining goals per branch
    state_valid: Tensor              # [B, S]         — which branches are alive
    top_ridx:    Tensor              # [B, S]         — first rule applied per branch
    next_var_indices: Optional[Tensor] = None  # [B] — free variable counter


@dataclass
class ProofEvidence:
    """How we got here: accumulated proof trace from completed groundings.

    Structured layout (D > 0):
        body       [B, C, D, M, 3] — body atoms per (grounding, depth, position)
        rule_idx   [B, C, D]       — which rule was applied at each depth
        body_count [B, C, D]       — valid body atoms per depth

    Legacy flat layout (D == 0):
        body       [B, C, G_body, 3] — flat accumulated body
        rule_idx   [B, C]            — top-level rule only
        body_count [B, C]            — total valid atoms
    """
    body:       Tensor      # [B, C, D, M, 3] or [B, C, G_body, 3]
    mask:       Tensor      # [B, C]
    count:      Tensor      # [B]
    rule_idx:   Tensor      # [B, C, D] or [B, C]
    body_count: Tensor      # [B, C, D] or [B, C]
    D: int = 0              # depth (0 = legacy flat layout)
    M: int = 0              # body atoms per rule (0 = unknown)
    head: Optional[Tensor] = None  # [B, C, D, 3] head atom at each depth

    @property
    def body_flat(self) -> Tensor:
        """[B, C, D*M, 3] flat view for SBR/legacy consumers."""
        if self.body.dim() == 5:
            B, C, D, M, _ = self.body.shape
            return self.body.reshape(B, C, D * M, 3)
        return self.body

    @property
    def rule_idx_top(self) -> Tensor:
        """[B, C] top-level rule index (depth 0) for legacy consumers."""
        if self.rule_idx.dim() == 3:
            return self.rule_idx[:, :, 0]
        return self.rule_idx

    @property
    def body_count_total(self) -> Tensor:
        """[B, C] total valid atoms for legacy consumers."""
        if self.body_count.dim() == 3:
            return self.body_count.sum(dim=-1)
        return self.body_count

    @property
    def body_atom_mask_flat(self) -> Tensor:
        """[B, C, D*M] per-atom validity mask for the flat body view.

        Unlike ``atom_idx < body_count_total``, this respects per-depth
        alignment: atoms at depth d occupy positions d*M..d*M+M-1, and
        only the first body_count[d] of those are valid.
        """
        if self.body_count.dim() == 3:
            B, C, D = self.body_count.shape
            M = self.body.shape[3] if self.body.dim() == 5 else 1
            # [B, C, D, M] mask: atom index m < body_count[d]
            m_idx = torch.arange(M, device=self.body_count.device)
            per_depth = m_idx < self.body_count.unsqueeze(-1)  # [B, C, D, M]
            return per_depth.reshape(B, C, D * M)
        # Legacy: atom_idx < body_count
        G = self.body.shape[2]
        idx = torch.arange(G, device=self.body_count.device)
        return idx < self.body_count.unsqueeze(-1)


@dataclass
class RuleGroundings:
    """Per-rule (A_in, A_out) grounding tensors — keras-compatible format.

    Each rule application is a separate entry. Atoms are shared via a global
    index table. Compatible with gather/scatter reasoning (SBR, R2N, DCR).

    atom_table[i] = [pred, subj, obj] for atom i.
    A_in[r][g, m]  = atom index of m-th body atom in grounding g of rule r.
    A_out[r][g, 0] = atom index of head atom in grounding g of rule r.
    """
    atom_table: Tensor                # [num_atoms, 3]
    A_in: Dict[int, Tensor]           # rule_idx → [G_r, M_r]
    A_out: Dict[int, Tensor]          # rule_idx → [G_r, 1]
    num_atoms: int
    num_rules: int


@dataclass
class GrounderOutput:
    """Unified return from forward(). Consumers pick what they need.

    RL:              reads output.state (ProofState)
    Explainability:  reads output.evidence (ProofEvidence)
    NeSy reasoning:  reads output.rule_groundings (RuleGroundings)
    """
    state:    ProofState
    evidence: Optional[ProofEvidence] = None
    rule_groundings: Optional[RuleGroundings] = None


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


class FlatResolvedChildren(NamedTuple):
    """Flat resolve output — only valid children, no S×K dense tensor.

    Used by the flat enum path. Pack receives this and scatter-compacts
    into dense [B, S_max, G, 3] for downstream phases.
    """
    flat_goals:     Tensor   # [T, G, 3]  — body atoms + remaining goals
    flat_gbody:     Tensor   # [T, A, 3]  — parent grounding body (for evidence)
    flat_rule_idx:  Tensor   # [T]         — rule index per child
    flat_b_idx:     Tensor   # [T]         — batch index
    flat_s_idx:     Tensor   # [T]         — parent state index
    flat_subs:      Tensor   # [T, 2, 2]  — substitutions (padding for enum)
    B: int                   # batch size
    S: int                   # input states per batch


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
    current_ridx:   Tensor  # [B, S_out] — rule applied at THIS depth step


class SyncParams(NamedTuple):
    """Metadata for SYNC phase — how to update accumulated_body."""
    parent_map:    Tensor  # [B, S_out]
    winning_subs:  Tensor  # [B, S_out, 2, 2]
    has_new_body:  Tensor  # [B, S_out]
    parent_bcount: Tensor  # [B, S_out] or [B, S_out, D]
    current_ridx:  Tensor  # [B, S_out] — rule applied at THIS depth step
