"""Per-call grounding state — ``Frontier`` (working) + run-scoped accumulators.

Accumulators exist ONLY for the requested ``OutputSpec`` tiers (wanting just
groundings pays for nothing more). Scopes nest run ⊃ chunk ⊃ step:
  - ``Frontier``                   — working frontier (chunk-scoped, threaded step→step; PURE)
  - ``FiringSet`` / ``ProofTrees`` — append-only accumulators (run-scoped)
Tiers: groundings→GoalState, firings→RuleGroundings, trees→CompletedTreeFirings.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from torch import Tensor

from grounder.core import OutputSpec  # tiers drive which accumulators are active


# ── Working frontier (chunk-scoped, threaded step→step; PURE — no accumulators) ──
@dataclass(frozen=True)
class Frontier:
    """The proof-search working tape: one row per active partial goal.
    Immutable — each ``step`` returns a new Frontier. Provenance fields
    (accumulated_*/per_depth/selected_atom) are only meaningful when
    ``OutputSpec.needs_provenance()`` (acc_D=acc_M=1 stub otherwise)."""

    goal_atoms: Tensor            # [B, G, L, 3]    the atoms of each goal
    grounding_body: Tensor        # [B, G, M, 3]    per-goal working body atoms
    goal_valid: Tensor            # [B, G]          active-goal mask
    top_rule_idx: Tensor          # [B, G]          top (depth-0) rule index per goal
    accumulated_body: Tensor      # [B, G, D, M, 3] per-depth accumulated body atoms (provenance)
    rule_idx_per_depth: Tensor        # [B, G, D]       per-depth rule index (provenance)
    head_per_depth: Tensor        # [B, G, D, 3]    per-depth head atom (provenance)
    body_count: Tensor            # [B, G, D]       per-depth #body atoms bound
    next_var: Optional[Tensor] = None      # [B]      var-renaming counter (standardization only)
    selected_atom: Optional[Tensor] = None  # [B, G, 3] atom selected this step (provenance)
    initial_next_var: Optional[Tensor] = None  # [B]    passed-in next_var (terminal standardize base)
    initial_goals: Optional[Tensor] = None     # [B, M_in, 3] passed-in goals (terminal standardize parents)

    def replace(self, **kw) -> "Frontier":
        return replace(self, **kw)


# ── Accumulator: fired rule applications (firings tier; run-scoped) ──
@dataclass(frozen=True)
class FiringSet:
    """Append-only set of fired rule applications across all steps+chunks (the
    'considered' set — PRIMARY for RuleGroundings; completed-tree evidence alone
    undercounts ~3x). Tuple-of-pieces; concatenated at finalize."""

    rule_idx: Tuple[Tensor, ...] = ()    # per emission [T]    (UNcollapsed rule/variant id)
    head: Tuple[Tensor, ...] = ()        #              [T, 3] (the selected-goal head)
    body: Tuple[Tensor, ...] = ()        #              [T, M, 3] (NOT pre-substituted)
    query_idx: Tuple[Tensor, ...] = ()   #              [T]    GLOBAL query index

    @staticmethod
    def empty() -> "FiringSet":
        return FiringSet()

    @staticmethod
    def from_emission(rule_idx: Tensor, head: Tensor, body: Tensor,
                      query_idx: Tensor) -> "FiringSet":
        return FiringSet(rule_idx=(rule_idx,), head=(head,), body=(body,),
                         query_idx=(query_idx,))

    def extend(self, other: "FiringSet") -> "FiringSet":
        return FiringSet(self.rule_idx + other.rule_idx, self.head + other.head,
                         self.body + other.body, self.query_idx + other.query_idx)


# ── Accumulator: completed proof trees (trees tier; run-scoped) ──
@dataclass(frozen=True)
class ProofTrees:
    """Append-only completed proof-tree pieces (one per chunk), concatenated along
    the query axis at finalize -> CompletedTreeFirings. Subject to the per-query Y_q
    budget during collection."""

    body: Tuple[Tensor, ...] = ()        # per chunk [B_i, Y_q, D, M, 3]
    rule_idx: Tuple[Tensor, ...] = ()    #           [B_i, Y_q, D]
    head: Tuple[Tensor, ...] = ()        #           [B_i, Y_q, D, 3]
    body_count: Tuple[Tensor, ...] = ()  #           [B_i, Y_q, D]
    count: Tuple[Tensor, ...] = ()       #           [B_i]

    @staticmethod
    def empty() -> "ProofTrees":
        return ProofTrees()

    @staticmethod
    def from_chunk(body: Tensor, rule_idx: Tensor, head: Tensor,
                   body_count: Tensor, count: Tensor) -> "ProofTrees":
        return ProofTrees((body,), (rule_idx,), (head,), (body_count,), (count,))

    def extend(self, other: "ProofTrees") -> "ProofTrees":
        return ProofTrees(self.body + other.body, self.rule_idx + other.rule_idx,
                          self.head + other.head, self.body_count + other.body_count,
                          self.count + other.count)


# ── Run-scoped state: only the accumulators OutputSpec requested + the offset ──
@dataclass(frozen=True)
class RunState:
    """Per-call state that spans chunks. Holds ONLY the requested accumulators.
    Threaded through the chunk loop; finalized once at merge."""

    spec: OutputSpec
    chunk_query_offset: int = 0
    firings: Optional[FiringSet] = None
    trees: Optional[ProofTrees] = None

    @staticmethod
    def init(spec: OutputSpec) -> "RunState":
        return RunState(
            spec=spec, chunk_query_offset=0,
            firings=FiringSet.empty() if spec.firings else None,
            trees=ProofTrees.empty() if spec.trees else None,
        )

    def with_chunk(self, *, trees: Optional[ProofTrees] = None, n_chunk: int) -> "RunState":
        """Fold one chunk's trees piece in and advance the global query offset.
        Firings accumulate per-step in ``capture_step`` (already lifted to global), not here."""
        t = self.trees.extend(trees) if (self.trees is not None and trees is not None) else self.trees
        return replace(self, trees=t, chunk_query_offset=self.chunk_query_offset + n_chunk)


__all__ = ["OutputSpec", "Frontier", "FiringSet", "ProofTrees", "RunState"]
