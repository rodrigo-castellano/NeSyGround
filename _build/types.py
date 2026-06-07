"""Output dataclasses (consumer-facing, frozen) + compile-safe NamedTuple seams.

``Layout`` is carried as a VALUE field on the resolved-children tuples (pack
dispatches on the value, never ``isinstance``). ``subs_noop`` lives only where it
is determined: post-pack ``SyncParams`` and constant-True on
``FlatResolvedChildren`` — never on dense ``ResolvedChildren`` (sld/rtf subs are
real there). No ``D==0`` legacy flat layout; ``CompletedTreeFirings`` carries
``Shapes`` so every consumer preserves D/M.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple, Optional

import torch
from torch import Tensor

from grounder._build.shapes import Shapes


class Layout(StrEnum):
    DENSE = "dense"
    FLAT = "flat"
    SPARSE = "sparse"


# ── Output views (frozen, consumer-facing) ──
@dataclass(frozen=True)
class ProofState:
    """Final proof-search state — the 'groundings' output tier (DpRL)."""

    proof_goals: Tensor          # [B, S, G, 3]
    state_valid: Tensor          # [B, S]
    top_rule_idx: Tensor         # [B, S]
    next_var: Optional[Tensor] = None   # [B]


@dataclass(frozen=True)
class CompletedTreeFirings:
    """Fired rule apps inside completed proof trees — the ``completed_tree_firings``
    output (probfol, torch-ns resolve). The ``FiringSet`` is PRIMARY for
    ``RuleGroundings``; this completed-tree view undercounts ~3x and is the
    chunk-merge fallback. Carries ``Shapes`` so consumers preserve D/M."""

    body: Tensor                 # [B, C, D, M, 3]
    grounding_valid: Tensor      # [B, C]  (which of the C groundings are real)
    count: Tensor                # [B]
    rule_idx: Tensor             # [B, C, D]
    body_count: Tensor           # [B, C, D]
    head: Tensor                 # [B, C, D, 3]
    shapes: Shapes

    @property
    def body_flat(self) -> Tensor:          # [B, C, D*M, 3]
        B, C, D, M, _ = self.body.shape
        return self.body.reshape(B, C, D * M, 3)

    @property
    def body_atom_mask_flat(self) -> Tensor:  # [B, C, D*M]
        B, C, D = self.body_count.shape
        M = self.body.shape[3]
        slot = torch.arange(M, device=self.body.device).view(1, 1, 1, M)
        mask = slot < self.body_count.unsqueeze(-1)        # [B,C,D,M]
        return mask.reshape(B, C, D * M)

    @property
    def tree_top_rule_idx(self) -> Tensor:  # [B, C]  (depth-0 rule of each tree)
        return self.rule_idx[:, :, 0]

    @property
    def top_rule_idx(self) -> Tensor:       # one-window alias of tree_top_rule_idx
        return self.tree_top_rule_idx

    @property
    def body_count_total(self) -> Tensor:   # [B, C]
        return self.body_count.sum(dim=-1)


# one-window back-compat alias (consumers import ProofEvidence)
ProofEvidence = CompletedTreeFirings


@dataclass(frozen=True)
class RuleGroundings:
    """Flat CSR-style rule firings, keras-compatible — the ``rule_groundings``
    output (torch-ns run_bc). fp_batch DROPS rows and rebuilds ``rule_offsets``
    (pruning removes rows, never masks)."""

    atom_table: Tensor           # [num_atoms, 3]
    body_pool_idx: Tensor        # [N_firings, M_max] long
    body_atom_valid: Tensor      # [N_firings, M_max] bool
    head_pool_idx: Tensor        # [N_firings] long
    rule_idx: Tensor             # [N_firings] long, sorted ascending
    rule_offsets: Tensor         # [num_rules + 1] long, cumsum
    num_atoms: int
    num_rules: int
    M_max: int
    query_pool_idx: Optional[Tensor] = None   # [B]
    firing_valid: Optional[Tensor] = None     # [N_firings] bool; tkk fast path

    def __post_init__(self) -> None:
        # tkk's to_firings_tensors fast path reads firing_valid; default all-True
        # of length N_firings (pruning drops rows, never masks → always True here).
        if self.firing_valid is None:
            object.__setattr__(self, "firing_valid", torch.ones(
                self.head_pool_idx.shape[0], dtype=torch.bool,
                device=self.head_pool_idx.device))

    @staticmethod
    def empty(num_rules: int, M: int, device, dtype=torch.long) -> "RuleGroundings":
        z = torch.zeros
        return RuleGroundings(
            atom_table=torch.empty(0, 3, dtype=dtype, device=device),
            body_pool_idx=torch.empty(0, M, dtype=dtype, device=device),
            body_atom_valid=torch.empty(0, M, dtype=torch.bool, device=device),
            head_pool_idx=torch.empty(0, dtype=dtype, device=device),
            rule_idx=torch.empty(0, dtype=dtype, device=device),
            rule_offsets=z(num_rules + 1, dtype=dtype, device=device),
            num_atoms=0, num_rules=num_rules, M_max=M,
        )

    def rule_slice(self, r: int) -> tuple[int, int]:
        """(start, end) firing range for rule ``r`` via ``rule_offsets``."""
        return int(self.rule_offsets[r]), int(self.rule_offsets[r + 1])


@dataclass(frozen=True)
class BackwardResult:
    """The backward proof bundle — each field present iff its OutputSpec tier was
    requested. ``kind`` + ``as_rule_groundings`` are attached in ``core/result.py``."""

    state: ProofState
    completed_tree_firings: Optional[CompletedTreeFirings] = None
    rule_groundings: Optional[RuleGroundings] = None

    @property
    def evidence(self) -> Optional[CompletedTreeFirings]:
        """One-window alias of ``completed_tree_firings`` (consumer ``.evidence``)."""
        return self.completed_tree_firings


# one-window back-compat alias (consumers import GrounderOutput)
GrounderOutput = BackwardResult


# ── Internal pipeline NamedTuples (torch.compile-safe; engine step seams) ──
class ResolvedChildren(NamedTuple):
    layout: Layout              # value tag; pack reads children.layout
    fact_goals: Tensor          # [B, S, K_f, G, 3]
    fact_grounding_body: Tensor # [B, S, K_f, M, 3]
    fact_success: Tensor        # [B, S, K_f]
    rule_goals: Tensor          # [B, S, K_r, G, 3]
    rule_grounding_body: Tensor # [B, S, K_r, M, 3]
    rule_success: Tensor        # [B, S, K_r]
    sub_rule_idx: Tensor        # [B, S, K_r]
    fact_subs: Tensor           # [B, S, K_f, 2, 2]
    rule_subs: Tensor           # [B, S, K_r, 2, 2]
    # NO subs_noop: for sld/rtf it is a POST-pack property.


class FlatResolvedChildren(NamedTuple):
    layout: Layout              # == Layout.FLAT
    flat_goals: Tensor          # [T, G, 3]
    flat_grounding_body: Tensor # [T, A, 3]
    flat_rule_idx: Tensor       # [T]
    flat_batch_idx: Tensor      # [T]
    flat_state_idx: Tensor      # [T]
    flat_subs: Tensor           # [T, 2, 2]
    flat_is_fact: Tensor        # [T] bool  True for fact rows, False for rule rows
    flat_top_ridx: Tensor       # [T] long  parent's top_rule_idx (first/inherited split)
    B: int
    S: int
    subs_noop: bool             # constant True for enum-flat (identity subs)


class PackedStates(NamedTuple):
    grounding_body: Tensor      # [B, S_out, M, 3]  (S_out padded to Shapes.S on dense)
    proof_goals: Tensor         # [B, S_out, G, 3]
    top_rule_idx: Tensor        # [B, S_out]
    state_valid: Tensor         # [B, S_out]
    body_count: Tensor          # [B, S_out]
    has_new_body: Tensor        # [B, S_out]
    current_rule_idx: Tensor    # [B, S_out]
    selected_goal: Tensor       # [B, S_out, 3]


class SyncParams(NamedTuple):
    parent_map: Tensor          # [B, S_out]
    winning_subs: Tensor        # [B, S_out, 2, 2]
    has_new_body: Tensor        # [B, S_out]
    parent_body_count: Tensor   # [B, S_out, D]
    current_rule_idx: Tensor    # [B, S_out]
    current_head: Tensor        # [B, S_out, 3]
    subs_noop: bool             # determined HERE (post-pack)


__all__ = [
    "Layout", "ProofState", "CompletedTreeFirings", "ProofEvidence",
    "RuleGroundings", "BackwardResult", "GrounderOutput",
    "ResolvedChildren", "FlatResolvedChildren", "PackedStates", "SyncParams",
]
