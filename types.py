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

    # ── Conversions to other grounding metrics ──────────────────────
    # Implementations live in ``grounder.groundings`` to avoid cyclic
    # imports at type-definition time.

    def unique_app_count(self, padding_idx: int) -> int:
        """Count distinct ``(rule, head, sorted_body)`` tuples in this
        evidence — keras-comparable rule_grounding metric.

        Equivalent to building :meth:`to_rule_groundings` and summing
        ``A_in[r].shape[0]``, but cheaper because it skips the per-rule
        split and atom-table construction.
        """
        from grounder.groundings import evidence_unique_app_count
        return evidence_unique_app_count(self, padding_idx)

    def to_rule_groundings(
        self, padding_idx: int, num_rules: Optional[int] = None,
    ) -> "RuleGroundings":
        """Build a :class:`RuleGroundings` dataclass from this evidence.

        Equivalent to keras's ``rule2groundings`` — each rule
        application is one entry, body atoms point into a global atom
        table. Pass ``num_rules`` to keep ``A_in`` keyspace stable
        across runs (otherwise inferred as ``max(rule_idx)+1``).
        """
        from grounder.groundings import evidence_to_rule_groundings
        return evidence_to_rule_groundings(self, padding_idx, num_rules)


@dataclass
class RuleGroundings:
    """Per-rule (A_in, A_out) grounding tensors — keras-compatible format.

    Each rule application is a separate entry. Atoms are shared via a global
    index table. Compatible with gather/scatter reasoning (SBR, R2N, DCR).

    atom_table[i] = [pred, subj, obj] for atom i.
    A_in[r][g, m]  = atom index of m-th body atom in grounding g of rule r.
    A_out[r][g, 0] = atom index of head atom in grounding g of rule r.

    ``query_pool_idx`` is populated by :func:`run_bc` (the
    rule-evidence entry point used by SBR/DCR/R2N's pool-iter loop):
    each query atom gets a slot in ``atom_table``, even when no
    grounding produced it as a head. Pool-iter consumers gather
    ``pool[query_pool_idx]`` to read the final query score.
    Left ``None`` by the per-tree path (``evidence_to_rule_groundings``
    has no notion of queries).
    """
    atom_table: Tensor                # [num_atoms, 3]
    A_in: Dict[int, Tensor]           # rule_idx → [G_r, M_r]
    A_out: Dict[int, Tensor]          # rule_idx → [G_r, 1]
    num_atoms: int
    num_rules: int
    query_pool_idx: Optional[Tensor] = None  # [B] indices into atom_table
    # Optional per-rule validity mask, ``[G_r]`` bool per rule. Populated
    # only when the grounder was configured with fixed-shape output
    # padding (``pad_outputs=True``): the first K_r entries are real
    # firings; the trailing G_r - K_r entries are padding rows whose
    # ``A_in``/``A_out`` slots point at the atom_table padding sentinel
    # and whose ``firings_valid[r][K_r:]`` is False. Consumers that build
    # FiringsTensors from this dict must forward these flags into
    # ``FiringsTensors.firing_valid`` so the rule loop masks padding out.
    firings_valid: Optional[Dict[int, Tensor]] = None


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
