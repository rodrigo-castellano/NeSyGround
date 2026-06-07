"""Per-call grounding state — organized by SCOPE x ROLE, conditional on OutputSpec.

Theory (first principles): state = the minimum needed to produce the REQUESTED
output. The output is a per-call choice (``OutputSpec``); accumulators exist ONLY
for the requested tiers, so a caller that wants just groundings pays for nothing
more.

Scopes nest ``run ⊃ chunk ⊃ step``. Roles:
  - immutable context           -> ``RunPlan`` (not here)
  - working frontier            -> ``Frontier``  (chunk-scoped, threaded step→step)
  - append-only accumulators    -> ``FiringSet`` / ``ProofTrees`` (run-scoped)
The compile-only static scatter buffer is an EXECUTION-layer local, NOT here.

Output tiers (verified against consumers):
  - groundings : final proof state              -> ProofState      (DpRL)
  - firings    : fired rule applications        -> RuleGroundings  (torch-ns run_bc)
  - trees      : completed proof trees          -> CompletedTreeFirings (probfol, torch-ns resolve)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from torch import Tensor

# ── What to produce (drives which accumulators are active) ──
# Step 2: the engine now carries the TYPED tier set (core.OutputSpec.tiers); the
# bool surface (groundings/firings/trees + needs_provenance) is kept as @property
# so plan/loop/finalize/buffers read exactly as before (byte-identical).
from grounder._build.core.request import OutputSpec


# ── Working frontier (chunk-scoped, threaded step→step; PURE — no accumulators) ──
@dataclass(frozen=True)
class Frontier:
    """The proof-search working tape: one row per active partial proof state.
    Immutable — each ``step`` returns a new Frontier. Provenance fields
    (accumulated_*/per_depth/selected_goal) are only meaningful when
    ``OutputSpec.needs_provenance()`` (acc_D=acc_M=1 stub otherwise)."""

    proof_goals: Tensor           # [B, S, G, 3]    remaining goals to prove
    grounding_body: Tensor        # [B, S, M, 3]    per-state working body atoms
    state_valid: Tensor           # [B, S]          active-state mask
    top_rule_idx: Tensor          # [B, S]          top (depth-0) rule index per state
    accumulated_body: Tensor      # [B, S, D, M, 3] per-depth accumulated body atoms (provenance)
    ridx_per_depth: Tensor        # [B, S, D]       per-depth rule index (provenance)
    head_per_depth: Tensor        # [B, S, D, 3]    per-depth head atom (provenance)
    body_count: Tensor            # [B, S, D]       per-depth #body atoms bound
    next_var: Optional[Tensor] = None      # [B]      var-renaming counter (standardization only)
    selected_goal: Optional[Tensor] = None  # [B, S, 3] goal selected this step (provenance)

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
    the query axis at finalize -> CompletedTreeFirings. Subject to the per-query C
    budget during collection."""

    body: Tuple[Tensor, ...] = ()        # per chunk [B_i, C, D, M, 3]
    rule_idx: Tuple[Tensor, ...] = ()    #           [B_i, C, D]
    head: Tuple[Tensor, ...] = ()        #           [B_i, C, D, 3]
    body_count: Tuple[Tensor, ...] = ()  #           [B_i, C, D]
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

    def with_chunk(self, *, firings: Optional[FiringSet] = None,
                   trees: Optional[ProofTrees] = None, n_chunk: int) -> "RunState":
        """Fold one chunk's pieces in (caller has already lifted firing query_idx
        to global) and advance the offset. Returns a new RunState."""
        f = self.firings.extend(firings) if (self.firings is not None and firings is not None) else self.firings
        t = self.trees.extend(trees) if (self.trees is not None and trees is not None) else self.trees
        return replace(self, firings=f, trees=t,
                       chunk_query_offset=self.chunk_query_offset + n_chunk)


__all__ = ["OutputSpec", "Frontier", "FiringSet", "ProofTrees", "RunState"]
