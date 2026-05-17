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

from dataclasses import dataclass, field
from typing import Dict, Iterator, NamedTuple, Optional, Tuple

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


class _PerRuleDictView:
    """Lazy dict-like view over a flat ``RuleGroundings`` for legacy code.

    Returns ``rg.body_pool_idx[offsets[r]:offsets[r+1]]`` for ``[r]``,
    ``rg.head_pool_idx[offsets[r]:offsets[r+1]].unsqueeze(-1)``, etc.,
    depending on the ``which`` flag. Adds **one host sync** per slice
    (the offsets are a tensor scalar). Hot-path consumers (SBR /
    DCR / R2N rule-loop) must NOT use this — they read the flat
    tensors directly.

    Trims trailing pad slots in ``body_pool_idx`` to a per-rule
    ``M_r = max(active body length within rule r, 0)`` so that the
    legacy ``A_in[r]`` shape matches the pre-flatten semantics
    (variable per-rule M). Falls back to ``M_max`` when no firings
    are active in the rule.
    """
    __slots__ = ("_rg", "_which")

    def __init__(self, rg: "RuleGroundings", which: str) -> None:
        self._rg = rg
        self._which = which

    def _rule_slice(self, r: int) -> Tuple[int, int]:
        offsets = self._rg.rule_offsets
        s = int(offsets[r].item())
        e = int(offsets[r + 1].item())
        return s, e

    def _per_rule_M(self, r: int, s: int, e: int) -> int:
        if e <= s:
            return 0
        valid = self._rg.body_atom_valid[s:e]                 # [G_r, M_max]
        if valid.numel() == 0:
            return 0
        m_per_row = valid.long().sum(dim=-1)                  # [G_r]
        return int(m_per_row.max().item())

    def __getitem__(self, r: int) -> Tensor:
        s, e = self._rule_slice(r)
        if self._which == "A_in":
            body = self._rg.body_pool_idx[s:e]                # [G_r, M_max]
            M_r = self._per_rule_M(r, s, e)
            return body[:, :M_r]
        if self._which == "A_out":
            return self._rg.head_pool_idx[s:e].unsqueeze(-1)
        if self._which == "firings_valid":
            return self._rg.firing_valid[s:e]
        raise KeyError(self._which)

    def __contains__(self, r: int) -> bool:
        if r < 0 or r + 1 >= self._rg.rule_offsets.numel():
            return False
        s, e = self._rule_slice(r)
        return e > s

    def get(self, r: int, default=None):
        if r in self:
            return self[r]
        return default

    def keys(self) -> Iterator[int]:
        n = int(self._rg.rule_offsets.numel()) - 1
        if n <= 0:
            return iter(())
        # Compute non-empty rules once via tensor diff to avoid per-rule
        # syncs in the common ``for r in rg.A_in:`` pattern.
        offsets = self._rg.rule_offsets
        sizes = (offsets[1:] - offsets[:-1]).tolist()         # 1 sync
        return iter(i for i, s in enumerate(sizes) if s > 0)

    def items(self):
        for r in self.keys():
            yield r, self[r]

    def values(self):
        for r in self.keys():
            yield self[r]

    def __iter__(self):
        return self.keys()

    def __len__(self) -> int:
        # Number of non-empty rules.
        n = int(self._rg.rule_offsets.numel()) - 1
        if n <= 0:
            return 0
        offsets = self._rg.rule_offsets
        sizes = (offsets[1:] - offsets[:-1]).tolist()         # 1 sync
        return sum(1 for s in sizes if s > 0)

    def __bool__(self) -> bool:
        return len(self) > 0


@dataclass
class RuleGroundings:
    """Per-rule (A_in, A_out) grounding tensors — keras-compatible format.

    **Flat-tensor layout (canonical).** Each rule application is one
    "firing". Firings from all rules are concatenated in sorted
    ``rule_idx`` order; ``rule_offsets[r]:rule_offsets[r+1]`` slices
    out rule ``r``'s firings. Body atoms are padded to ``M_max`` with
    pad-slot 0 (the consumer-side sentinel); ``body_atom_valid`` and
    ``firing_valid`` carry the per-slot / per-firing masks.

    Compatible with gather/scatter reasoning (SBR, R2N, DCR). The
    legacy ``A_in`` / ``A_out`` / ``firings_valid`` dict interfaces
    are provided as lazy properties for backward compatibility.

    atom_table[i]              = [pred, subj, obj] for atom i.
    body_pool_idx[f, m]        = atom index of m-th body atom of firing f.
    head_pool_idx[f]           = atom index of head of firing f.
    rule_idx[f]                = which rule firing f belongs to.
    rule_offsets[r]..[r+1]     = slice covering firings of rule r.

    ``query_pool_idx`` is populated by :func:`run_bc` (the
    rule-evidence entry point used by SBR/DCR/R2N's pool-iter loop):
    each query atom gets a slot in ``atom_table``, even when no
    grounding produced it as a head. Pool-iter consumers gather
    ``pool[query_pool_idx]`` to read the final query score.
    Left ``None`` by the per-tree path (``evidence_to_rule_groundings``
    has no notion of queries).

    ``firing_valid`` is False for padding firings produced by
    ``pad_rule_groundings`` (``pad_outputs=True`` codepath); it's
    True everywhere on the unpadded path.
    """
    atom_table: Tensor                # [num_atoms, 3]
    # Flat firings (concatenated over rules in sorted rule_idx order).
    body_pool_idx: Tensor             # [N_firings, M_max] long
    body_atom_valid: Tensor           # [N_firings, M_max] bool
    head_pool_idx: Tensor             # [N_firings] long
    rule_idx: Tensor                  # [N_firings] long, sorted ascending
    rule_offsets: Tensor              # [num_rules + 1] long, cumsum
    firing_valid: Tensor              # [N_firings] bool
    num_atoms: int = 0
    num_rules: int = 0
    M_max: int = 0
    query_pool_idx: Optional[Tensor] = None  # [B] indices into atom_table
    _has_real_firings_valid: bool = False    # True when padded outputs

    # ── Legacy dict-style accessors (read-only views) ───────────────
    @property
    def A_in(self) -> _PerRuleDictView:
        return _PerRuleDictView(self, "A_in")

    @property
    def A_out(self) -> _PerRuleDictView:
        return _PerRuleDictView(self, "A_out")

    @property
    def firings_valid(self) -> Optional[_PerRuleDictView]:
        if not self._has_real_firings_valid:
            return None
        return _PerRuleDictView(self, "firings_valid")

    @classmethod
    def empty(
        cls,
        *,
        num_rules: int = 0,
        M_max: int = 0,
        device: Optional[torch.device] = None,
    ) -> "RuleGroundings":
        """Build an empty (zero-firings) ``RuleGroundings``.

        The returned object has an empty atom table, zero firings, and
        a ``rule_offsets`` of length ``num_rules + 1`` filled with
        zeros. Used as the default return for trees with no proofs.
        """
        if device is None:
            device = torch.device("cpu")
        return cls(
            atom_table=torch.zeros(0, 3, dtype=torch.long, device=device),
            body_pool_idx=torch.zeros(
                0, max(M_max, 0), dtype=torch.long, device=device),
            body_atom_valid=torch.zeros(
                0, max(M_max, 0), dtype=torch.bool, device=device),
            head_pool_idx=torch.zeros(0, dtype=torch.long, device=device),
            rule_idx=torch.zeros(0, dtype=torch.long, device=device),
            rule_offsets=torch.zeros(
                max(num_rules, 0) + 1, dtype=torch.long, device=device),
            firing_valid=torch.zeros(0, dtype=torch.bool, device=device),
            num_atoms=0,
            num_rules=int(max(num_rules, 0)),
            M_max=int(max(M_max, 0)),
        )


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
