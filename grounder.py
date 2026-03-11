"""Main class hierarchy for the grounder package.

Grounder(nn.Module)            — base: owns KB state (facts, rules, indices)
  └─ BCGrounder(Grounder)      — backward chaining: step() + multi-depth forward()
     ├─ PrologGrounder          — K = K_f + K_r, independent fact + rule resolution
     └─ RTFGrounder             — K = K_f * K_r, two-level Rule-Then-Fact

Tensor conventions:
    B = batch, S = states per query, G = max goals, M = max body atoms
    K = max derived children per parent state
    K_f = max fact matches, K_r = max rule matches

    proof_goals:    [B, S, G, 3]  — active proof-goal states
    grounding_body: [B, S, M, 3]  — grounding body (tracks resolved atoms)
    state_valid:    [B, S]         — bool mask for active states
    top_ridx:       [B, S]         — top-level rule index per state
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.fact_index import (
    ArgKeyFactIndex,
    BlockSparseFactIndex,
    InvertedFactIndex,
    fact_contains,
    pack_triples_64,
)
from grounder.packing import compact_atoms, pack_fact_rule
from grounder.postprocessing import (
    collect_groundings,
    prune_ground_facts_3d,
)
from grounder.primitives import apply_substitutions, unify_one_to_one
from grounder.rule_index import RuleIndex
from grounder.types import ForwardResult, StepResult


# ============================================================================
# Grounder — base class owning KB state
# ============================================================================

class Grounder(nn.Module):
    """Base class owning knowledge-base state: facts, rules, indices.

    Args:
        facts_idx:        [F, 3] fact triples (pred, arg0, arg1)
        rules_heads_idx:  [R, 3] rule head atoms
        rules_bodies_idx: [R, Bmax, 3] rule body atoms (padded)
        rule_lens:        [R] number of body atoms per rule
        constant_no:      highest constant index (variables start at constant_no + 1)
        padding_idx:      padding value
        device:           target device
        predicate_no:     total number of predicates (exclusive upper bound)
        pack_base:        multiplier for hash packing (auto if None)
        fact_index_type:  'arg_key' | 'inverted' | 'block_sparse'
        shuffle_facts:    shuffle facts per predicate before building indices
        shuffle_seed:     random seed for fact shuffling
        num_entities:     total entities (required for inverted/block_sparse)
        max_facts_per_query: K_f for inverted/block_sparse indices
        max_memory_mb:    memory budget for block_sparse dense blocks
    """

    def __init__(
        self,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        *,
        predicate_no: Optional[int] = None,
        pack_base: Optional[int] = None,
        fact_index_type: str = "arg_key",
        shuffle_facts: bool = False,
        shuffle_seed: int = 42,
        num_entities: Optional[int] = None,
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
    ) -> None:
        super().__init__()
        self.constant_no = int(constant_no)
        self.padding_idx = int(padding_idx)
        self._device = device

        # Move tensors to device
        facts_idx = facts_idx.to(device=device, dtype=torch.long)
        rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)
        rules_bodies_idx = rules_bodies_idx.to(device=device, dtype=torch.long)
        rule_lens = rule_lens.to(device=device, dtype=torch.long)

        # Max body atoms per rule
        self.M = int(rule_lens.max().item()) if rule_lens.numel() > 0 else 1

        # Pack base for hash computation
        if pack_base is not None:
            self.pack_base = int(pack_base)
        else:
            self.pack_base = max(int(constant_no), int(padding_idx)) + 2

        # --- Build fact index ---
        if fact_index_type == "arg_key":
            self.fact_index = ArgKeyFactIndex(
                facts_idx, constant_no, padding_idx, device,
                pack_base=self.pack_base,
            )
        elif fact_index_type == "inverted":
            assert num_entities is not None and predicate_no is not None
            self.fact_index = InvertedFactIndex(
                facts_idx, constant_no, padding_idx, device,
                num_entities, predicate_no + 1, max_facts_per_query,
            )
        elif fact_index_type == "block_sparse":
            assert num_entities is not None and predicate_no is not None
            self.fact_index = BlockSparseFactIndex(
                facts_idx, constant_no, padding_idx, device,
                num_entities, predicate_no + 1, max_facts_per_query, max_memory_mb,
            )
        else:
            raise ValueError(f"Unknown fact_index_type: {fact_index_type}")

        # Canonical references (from the fact index, which sorts facts)
        self.register_buffer("facts_idx", self.fact_index.facts_idx)
        self.register_buffer("fact_hashes", self.fact_index.fact_hashes)

        # --- Build rule index ---
        self.rule_index = RuleIndex(
            rules_heads_idx, rules_bodies_idx, rule_lens, device,
            predicate_no=predicate_no, padding_idx=padding_idx,
        )

        self.num_rules = rules_heads_idx.shape[0]
        self.K_r = self.rule_index.max_rule_pairs

        # K_f: auto-computed from fact index
        if isinstance(self.fact_index, ArgKeyFactIndex):
            self.K_f = self.fact_index.max_fact_pairs
        else:
            self.K_f = max_facts_per_query

        # Optional: shuffle facts per predicate
        if shuffle_facts and self.facts_idx.numel() > 0:
            self._shuffle_facts_per_predicate(predicate_no, seed=shuffle_seed)

    @property
    def num_facts(self) -> int:
        return self.facts_idx.shape[0]

    def _shuffle_facts_per_predicate(
        self, predicate_no: Optional[int], seed: int = 42,
    ) -> None:
        """Randomly shuffle facts within each predicate segment."""
        device = self.facts_idx.device
        gen = torch.Generator(device=device).manual_seed(seed)
        preds = self.facts_idx[:, 0]
        num_preds = max(
            int(preds.max().item()) + 1,
            (predicate_no + 1) if predicate_no else 1,
        )
        order = torch.argsort(preds, stable=True)
        facts_sorted = self.facts_idx[order]
        counts = torch.bincount(preds.long(), minlength=num_preds)
        starts = torch.zeros(num_preds + 1, dtype=torch.long, device=device)
        starts[1:] = counts.cumsum(0)

        shuffled = facts_sorted.clone()
        for p in range(num_preds):
            s, e = starts[p].item(), starts[p + 1].item()
            if e - s > 1:
                shuffled[s:e] = facts_sorted[
                    s + torch.randperm(e - s, device=device, generator=gen)
                ]
        self.facts_idx.copy_(shuffled)


# ============================================================================
# BCGrounder — backward chaining with step() + multi-depth forward()
# ============================================================================

class BCGrounder(Grounder):
    """Backward-chaining grounder with single-step and multi-depth pipelines.

    The 5-stage pipeline per step:
        1. SELECT:  extract query atom, remaining goals, active mask
        2. RESOLVE FACTS: unify query with facts
        3. RESOLVE RULES: unify query with rule heads, add body atoms
        4. PACK:    flatten S×K children → compact to S_out
        5. POSTPROCESS: prune ground facts, compact atoms, collect groundings

    Args:
        max_goals:               G dimension (max atoms per proof-goal state)
        depth:                   number of proof steps (fixed at construction)
        max_states:              S cap (max concurrent states). Default: K.
        K_MAX:                   hard cap on K
        max_derived_per_state:   explicit K override (None = auto)
        implicit_atom_selection: resolve ALL atom positions (unrolled, compile-safe)
        standardization_mode:    'none' | 'offset' | 'canonical'
        runtime_var_end_index:   upper bound for variable IDs
        compile_mode:            None | 'reduce-overhead' | 'max-autotune'
        max_total_groundings:    output budget for grounding collection (tG)
    """

    def __init__(
        self,
        *args,
        max_goals: int,
        depth: int = 1,
        max_states: Optional[int] = None,
        K_MAX: int = 550,
        max_derived_per_state: Optional[int] = None,
        implicit_atom_selection: bool = False,
        standardization_mode: str = "none",
        runtime_var_end_index: Optional[int] = None,
        compile_mode: Optional[str] = None,
        max_total_groundings: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_goals = max_goals  # G
        self.depth = depth
        self.implicit_atom_selection = implicit_atom_selection
        self.standardization_mode = standardization_mode
        self.runtime_var_end_index = runtime_var_end_index
        self.compile_mode = compile_mode

        # K computation (subclass overrides _compute_K_uncapped)
        K_uncapped = self._compute_K_uncapped()
        self.K = min(K_uncapped, K_MAX)
        if max_derived_per_state is not None:
            self.K = int(max_derived_per_state)

        # S computation
        if max_states is not None:
            self.S = max_states
        else:
            self.S = self.K

        # Max vars per rule (template: constant_no+1=V0, +2=V1, ...; 2 head args + free)
        if self.rule_index.rule_lens_sorted.numel() > 0:
            self.max_vars_per_rule = int(self.rule_index.rule_lens_sorted.max().item()) + 2
        else:
            self.max_vars_per_rule = 3

        # Grounding collection budget
        self.effective_total_G = max_total_groundings

        # Build compiled step functions
        self._build_compiled_fns()

    def _compute_K_uncapped(self) -> int:
        """Compute K before capping. Override in subclasses."""
        return self.K_f + self.K_r

    def _build_compiled_fns(self) -> None:
        """Build torch.compile wrapper for step function."""
        if (
            self.compile_mode
            and self.depth > 1
            and self._device.type == "cuda"
        ):
            self._fn_step = torch.compile(
                self._step_impl, fullgraph=True, mode=self.compile_mode)
        else:
            self._fn_step = self._step_impl

    # ------------------------------------------------------------------
    # Stage 1: SELECT
    # ------------------------------------------------------------------

    def _select(
        self, proof_goals: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract query atom (position 0) and detect active states.

        Args:
            proof_goals: [B, S, G, 3]

        Returns:
            queries:     [B, S, 3]
            remaining:   [B, S, G, 3] (position 0 masked to padding)
            active_mask: [B, S] bool
        """
        active_mask = proof_goals[:, :, 0, 0] != self.padding_idx  # [B, S]
        queries = proof_goals[:, :, 0, :]  # [B, S, 3]
        remaining = proof_goals.clone()
        remaining[:, :, 0, :] = self.padding_idx
        return queries, remaining, active_mask

    # ------------------------------------------------------------------
    # Stages 2+3: RESOLVE (abstract — subclass implements)
    # ------------------------------------------------------------------

    def _resolve_facts(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Resolve query atoms against facts.

        Returns:
            fact_goals:   [B, S, K_f, G, 3] remaining goals after fact subs
            fact_gbody:   [B, S, K_f, M, 3] grounding body after fact subs
            fact_success: [B, S, K_f] validity mask
        """
        raise NotImplementedError

    def _resolve_rules(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Resolve query atoms against rules.

        Returns:
            rule_goals:   [B, S, K_r, G, 3] new proof goals (body + remaining)
            rule_gbody:   [B, S, K_r, M, 3] grounding body after rule subs
            rule_success: [B, S, K_r] validity mask
            sub_rule_idx: [B, S, K_r] matched rule indices (original, unsorted)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Stage 4: PACK
    # ------------------------------------------------------------------

    def _pack_step(
        self,
        fact_goals: Tensor,         # [B, S, K_f, G, 3]
        fact_gbody: Tensor,         # [B, S, K_f, M, 3]
        fact_success: Tensor,       # [B, S, K_f]
        rule_goals: Tensor,         # [B, S, K_r, G, 3]
        rule_gbody: Tensor,         # [B, S, K_r, M, 3]
        rule_success: Tensor,       # [B, S, K_r]
        top_ridx: Tensor,           # [B, S]
        sub_rule_idx: Tensor,       # [B, S, K_r]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Flatten S×K children and compact to [B, S_out, ...].

        Auto-detects whether grounding body / rule index need initialization
        (first depth) or propagation (subsequent depths) based on sentinel
        values: top_ridx == -1 means uninitialized.

        Returns:
            new_gbody: [B, S_out, M, 3]
            new_goals: [B, S_out, G, 3]
            new_ridx:  [B, S_out]
            new_valid: [B, S_out]
        """
        B = fact_goals.shape[0]
        S_in = fact_goals.shape[1]
        K_f = fact_goals.shape[2]
        K_r = rule_goals.shape[2]
        M = fact_gbody.shape[3]
        G = fact_goals.shape[3]

        n_f = S_in * K_f
        n_r = S_in * K_r

        # Flatten fact results: [B, S, K_f, ...] → [B, S*K_f, ...]
        flat_f_gbody = fact_gbody.reshape(B, n_f, M, 3)
        flat_f_goals = fact_goals.reshape(B, n_f, G, 3)
        flat_f_valid = fact_success.reshape(B, n_f)
        flat_f_ridx = top_ridx.unsqueeze(2).expand(B, S_in, K_f).reshape(B, n_f)

        # Flatten rule results: [B, S, K_r, ...] → [B, S*K_r, ...]
        # Auto-detect init vs propagate: top_ridx == -1 means uninitialized (first depth)
        ridx_uninit = (top_ridx == -1)  # [B, S]
        ridx_uninit_exp = ridx_uninit.unsqueeze(2).expand(B, S_in, K_r)  # [B, S, K_r]

        # gbody: init from rule body (first M goals) or propagate existing
        rule_body_as_gbody = rule_goals[:, :, :, :M, :].reshape(B, n_r, M, 3)
        propagated_gbody = rule_gbody.reshape(B, n_r, M, 3)
        flat_r_gbody = torch.where(
            ridx_uninit_exp.reshape(B, n_r, 1, 1).expand_as(rule_body_as_gbody),
            rule_body_as_gbody,
            propagated_gbody,
        )

        # ridx: init from matched rule or propagate parent's
        flat_r_ridx = torch.where(
            ridx_uninit_exp.reshape(B, n_r),
            sub_rule_idx.reshape(B, n_r),
            top_ridx.unsqueeze(2).expand(B, S_in, K_r).reshape(B, n_r),
        )

        flat_r_goals = rule_goals.reshape(B, n_r, G, 3)
        flat_r_valid = rule_success.reshape(B, n_r)

        return pack_fact_rule(
            flat_f_gbody, flat_f_goals, flat_f_valid, flat_f_ridx,
            flat_r_gbody, flat_r_goals, flat_r_valid, flat_r_ridx,
            self.S, self.padding_idx,
        )

    # ------------------------------------------------------------------
    # Stage 5: POST-PROCESS
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        grounding_body: Tensor,     # [B, S, M, 3]
        proof_goals: Tensor,        # [B, S, G, 3]
        state_valid: Tensor,        # [B, S]
        top_ridx: Tensor,           # [B, S]
        collected_body: Tensor,     # [B, tG, M, 3]
        collected_mask: Tensor,     # [B, tG]
        collected_ridx: Tensor,     # [B, tG]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Prune ground facts, compact atoms, collect groundings.

        Returns:
            proof_goals:     [B, S, G, 3] (pruned + compacted)
            collected_body:  [B, tG, M, 3]
            collected_mask:  [B, tG]
            collected_ridx:  [B, tG]
            state_valid:     [B, S] (terminal states deactivated)
        """
        # Prune known ground facts from proof goals
        proof_goals = prune_ground_facts_3d(
            proof_goals, state_valid,
            self.fact_hashes, self.pack_base,
            self.constant_no, self.padding_idx,
        )
        # Left-align remaining atoms
        proof_goals = compact_atoms(proof_goals, self.padding_idx)
        # Collect completed groundings
        cb, cm, cr, sv = collect_groundings(
            grounding_body, proof_goals, state_valid, top_ridx,
            collected_body, collected_mask, collected_ridx,
            self.constant_no, self.padding_idx, self.effective_total_G,
        )
        return proof_goals, cb, cm, cr, sv

    # ------------------------------------------------------------------
    # Filter hook (no-op default, override in subclasses)
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        fact_goals: Tensor,
        fact_gbody: Tensor,
        fact_success: Tensor,
        rule_goals: Tensor,
        rule_gbody: Tensor,
        rule_success: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply strategy/domain-specific filters. No-op by default."""
        return fact_goals, fact_gbody, fact_success, rule_goals, rule_gbody, rule_success

    # ------------------------------------------------------------------
    # Step (single proof step)
    # ------------------------------------------------------------------

    def _step_impl(
        self,
        grounding_body: Tensor,     # [B, S, M, 3]
        proof_goals: Tensor,        # [B, S, G, 3]
        top_ridx: Tensor,           # [B, S]
        state_valid: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Single backward-chaining proof step: SELECT → RESOLVE → PACK.

        Behaviour adapts automatically based on tensor content:
        - When grounding_body is all-padding (first depth), fact resolution is
          suppressed and rule body atoms initialize the grounding body.
        - When grounding_body has been initialized (subsequent depths), both fact
          and rule resolution run, and the existing grounding body is propagated.

        Returns:
            new_gbody:     [B, S_out, M, 3]
            new_goals:     [B, S_out, G, 3]
            new_ridx:      [B, S_out]
            new_valid:     [B, S_out]
            new_next_var:  [B]
        """
        B = proof_goals.shape[0]
        S = proof_goals.shape[1]
        pad = self.padding_idx

        # Stage 1: SELECT
        queries, remaining, active_mask = self._select(proof_goals)

        # Stage 2: RESOLVE FACTS
        fact_goals, fact_gbody, fact_success = self._resolve_facts(
            queries, remaining, grounding_body, state_valid, active_mask)

        # Auto-skip facts when gbody is uninitialized (all padding = first depth).
        # Fact matches at depth 0 have no rule body and would waste output slots.
        gbody_uninit = (grounding_body[:, :, 0, 0] == pad)  # [B, S]
        fact_success = fact_success & ~gbody_uninit.unsqueeze(-1)

        # Stage 3: RESOLVE RULES
        rule_goals, rule_gbody, rule_success, sub_rule_idx = self._resolve_rules(
            queries, remaining, grounding_body, state_valid, active_mask,
            next_var_indices)

        # Apply filters (no-op by default)
        (fact_goals, fact_gbody, fact_success,
         rule_goals, rule_gbody, rule_success) = self._apply_filters(
            fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody, rule_success)

        # Stage 4: PACK
        new_gbody, new_goals, new_ridx, new_valid = self._pack_step(
            fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody, rule_success,
            top_ridx, sub_rule_idx)

        # Update next_var_indices
        V = self.max_vars_per_rule
        K_r = self.K_r
        new_next_var = next_var_indices + S * K_r * V

        return new_gbody, new_goals, new_ridx, new_valid, new_next_var

    # ------------------------------------------------------------------
    # Forward (multi-depth proof loop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        queries: Tensor,        # [B, 3]
        query_mask: Tensor,     # [B] bool
    ) -> ForwardResult:
        """Multi-depth backward-chaining proof.

        Args:
            queries:    [B, 3] query atoms to prove
            query_mask: [B] bool mask for valid queries

        Returns:
            ForwardResult with collected_body, collected_mask, collected_count,
            collected_ridx.
        """
        B = queries.size(0)
        M = self.M
        G = self.max_goals
        S = self.S
        tG = self.effective_total_G
        dev = queries.device
        pad = self.padding_idx
        E = self.constant_no + 1

        if self.num_rules == 0:
            return ForwardResult(
                collected_body=queries.new_zeros(B, tG, M, 3),
                collected_mask=torch.zeros(B, tG, dtype=torch.bool, device=dev),
                collected_count=queries.new_zeros(B),
                collected_ridx=queries.new_zeros(B, tG),
            )

        # --- Initialize output buffers ---
        collected_body = queries.new_zeros(B, tG, M, 3)
        collected_mask = torch.zeros(B, tG, dtype=torch.bool, device=dev)
        collected_ridx = queries.new_zeros(B, tG)

        # --- Initialize proof state: 1 state per query ---
        # gbody = all padding (sentinel: not yet initialized)
        # top_ridx = -1 (sentinel: not yet assigned a rule)
        grounding_body = torch.full(
            (B, 1, M, 3), pad, dtype=torch.long, device=dev)
        proof_goals = torch.full(
            (B, 1, G, 3), pad, dtype=torch.long, device=dev)
        proof_goals[:, 0, 0, :] = queries
        top_ridx = torch.full((B, 1), -1, dtype=torch.long, device=dev)
        state_valid = query_mask.unsqueeze(1)  # [B, 1]
        next_var_indices = torch.full((B,), E, dtype=torch.long, device=dev)

        # --- Proof loop: depth steps ---
        for _d in range(self.depth):
            # Clone CUDA graph outputs before passing to next compiled region
            if _d > 0:
                grounding_body = grounding_body.clone()
                proof_goals = proof_goals.clone()
                top_ridx = top_ridx.clone()
                state_valid = state_valid.clone()
                next_var_indices = next_var_indices.clone()

            (grounding_body, proof_goals, top_ridx,
             state_valid, next_var_indices) = self._fn_step(
                grounding_body, proof_goals, top_ridx,
                state_valid, next_var_indices)
            next_var_indices = next_var_indices.clone()

            # Postprocess: prune ground facts + compact + collect groundings
            proof_goals, collected_body, collected_mask, collected_ridx, state_valid = \
                self._postprocess(
                    grounding_body, proof_goals, state_valid, top_ridx,
                    collected_body, collected_mask, collected_ridx)

        collected_count = collected_mask.sum(dim=1)

        # Clamp body atom args to valid entity range
        body_args = collected_body[:, :, :, 1:3]
        collected_body = torch.cat([
            collected_body[:, :, :, 0:1],
            body_args.clamp(0, E - 1),
        ], dim=3)

        return ForwardResult(
            collected_body=collected_body,
            collected_mask=collected_mask,
            collected_count=collected_count,
            collected_ridx=collected_ridx,
        )


# ============================================================================
# PrologGrounder — single-level Prolog: facts + rules independently
# ============================================================================

class PrologGrounder(BCGrounder):
    """Single-level Prolog resolution: K = K_f + K_r.

    Resolves query atoms independently against facts and rules via MGU.
    Facts and rules produce separate child sets that are concatenated.
    """

    def _compute_K_uncapped(self) -> int:
        return self.K_f + self.K_r

    # --- Fact resolution ---

    def _resolve_facts(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if isinstance(self.fact_index, ArgKeyFactIndex):
            return self._resolve_facts_argkey(
                queries, remaining, grounding_body, state_valid, active_mask)
        return self._resolve_facts_enumerate(
            queries, remaining, grounding_body, state_valid, active_mask)

    def _resolve_facts_argkey(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """ArgKey-based fact resolution: targeted_lookup → unify → apply subs.

        Returns:
            fact_goals:   [B, S, K_f, G, 3]
            fact_gbody:   [B, S, K_f, M, 3]
            fact_success: [B, S, K_f]
        """
        B, S, _ = queries.shape
        G = remaining.shape[2]
        M_g = grounding_body.shape[2]
        dev = queries.device
        pad = self.padding_idx
        K_f = self.K_f

        N = B * S
        flat_q = queries.reshape(N, 3)
        flat_active = (active_mask & state_valid).reshape(N)

        # Targeted fact lookup
        fact_item_idx, fact_valid = self.fact_index.targeted_lookup(flat_q, K_f)
        # fact_item_idx: [N, K_f], fact_valid: [N, K_f]

        F = self.facts_idx.shape[0]
        if F == 0:
            return (
                torch.full((B, S, K_f, G, 3), pad, dtype=torch.long, device=dev),
                grounding_body.unsqueeze(2).expand(B, S, K_f, M_g, 3).clone(),
                torch.zeros(B, S, K_f, dtype=torch.bool, device=dev),
            )

        safe_idx = fact_item_idx.clamp(0, max(F - 1, 0))
        fact_atoms = self.facts_idx[safe_idx.view(-1)].view(N, K_f, 3)

        # Unify queries with facts
        q_exp = flat_q.unsqueeze(1).expand(-1, K_f, -1)  # [N, K_f, 3]
        ok_flat, subs_flat = unify_one_to_one(
            q_exp.reshape(-1, 3), fact_atoms.reshape(-1, 3),
            self.constant_no, pad)
        ok = ok_flat.view(N, K_f)
        subs = subs_flat.view(N, K_f, 2, 2)

        success = ok & fact_valid & flat_active.unsqueeze(1)

        # Apply substitutions to remaining + grounding_body jointly
        flat_rem = remaining.reshape(N, G, 3)
        flat_gbody = grounding_body.reshape(N, M_g, 3)
        combined = torch.cat([
            flat_rem.unsqueeze(1).expand(-1, K_f, -1, -1).reshape(N * K_f, G, 3),
            flat_gbody.unsqueeze(1).expand(-1, K_f, -1, -1).reshape(N * K_f, M_g, 3),
        ], dim=1)  # [N*K_f, G+M_g, 3]
        subs_flat_for_apply = subs.reshape(N * K_f, 2, 2)
        combined = apply_substitutions(combined, subs_flat_for_apply, pad)

        fact_goals = combined[:, :G, :].view(B, S, K_f, G, 3)
        fact_gbody = combined[:, G:, :].view(B, S, K_f, M_g, 3)

        # Mask out invalid entries
        pad_t = torch.tensor(pad, dtype=torch.long, device=dev)
        fact_goals = torch.where(
            success.view(B, S, K_f, 1, 1), fact_goals, pad_t)
        fact_gbody = torch.where(
            success.view(B, S, K_f, 1, 1), fact_gbody,
            torch.tensor(0, dtype=torch.long, device=dev))

        fact_success = success.view(B, S, K_f)
        return fact_goals, fact_gbody, fact_success

    def _resolve_facts_enumerate(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Enumerate-based fact resolution (Inverted/BlockSparse).

        Returns:
            fact_goals:   [B, S, K_f, G, 3]
            fact_gbody:   [B, S, K_f, M, 3]
            fact_success: [B, S, K_f]
        """
        B, S, _ = queries.shape
        G = remaining.shape[2]
        M_g = grounding_body.shape[2]
        dev = queries.device
        pad = self.padding_idx
        c_no = self.constant_no
        E = c_no + 1

        pred = queries[:, :, 0]
        arg0 = queries[:, :, 1]
        arg1 = queries[:, :, 2]

        arg0_ground = (arg0 <= c_no)
        arg1_ground = (arg1 <= c_no)
        has_ground = arg0_ground | arg1_ground
        both_ground = arg0_ground & arg1_ground

        use_arg0 = arg0_ground
        direction = torch.where(
            use_arg0, torch.zeros_like(arg0), torch.ones_like(arg0))
        bound_arg = torch.where(use_arg0, arg0, arg1)
        free_var = torch.where(use_arg0, arg1, arg0)

        # Enumerate candidates
        N = B * S
        safe_pred = pred.clamp(0).reshape(-1)
        safe_bound = bound_arg.clamp(0, E - 1).reshape(-1)
        cands, cand_mask = self.fact_index.enumerate(
            safe_pred, safe_bound, direction.reshape(-1))
        K_f = cands.shape[1]
        cands = cands.view(B, S, K_f)
        cand_mask = cand_mask.view(B, S, K_f)

        # Build substitutions: free_var → candidate
        N_f = B * S * K_f
        free_var_exp = free_var.unsqueeze(2).expand(B, S, K_f)
        subs = torch.full((N_f, 2, 2), pad, dtype=torch.long, device=dev)
        subs[:, 0, 0] = free_var_exp.reshape(-1)
        subs[:, 0, 1] = cands.reshape(-1)

        # Both-ground filter
        other_arg = torch.where(use_arg0, arg1, arg0)
        both_filter = torch.where(
            both_ground.unsqueeze(2).expand(B, S, K_f),
            cands == other_arg.unsqueeze(2).expand(B, S, K_f),
            torch.ones(B, S, K_f, dtype=torch.bool, device=dev),
        )

        fact_success = (
            cand_mask & has_ground.unsqueeze(-1)
            & state_valid.unsqueeze(-1) & active_mask.unsqueeze(-1)
            & both_filter
        )

        # Apply subs to remaining + grounding_body jointly
        rem_exp = remaining.unsqueeze(2).expand(B, S, K_f, G, 3).reshape(N_f, G, 3)
        gbody_exp = grounding_body.unsqueeze(2).expand(B, S, K_f, M_g, 3).reshape(N_f, M_g, 3)
        combined = torch.cat([rem_exp, gbody_exp], dim=1)
        combined = apply_substitutions(combined, subs, pad)

        fact_goals = combined[:, :G, :].view(B, S, K_f, G, 3)
        fact_gbody = combined[:, G:, :].view(B, S, K_f, M_g, 3)

        return fact_goals, fact_gbody, fact_success

    # --- Rule resolution ---

    def _resolve_rules(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Resolve queries against rules via MGU.

        Returns:
            rule_goals:   [B, S, K_r, G, 3]
            rule_gbody:   [B, S, K_r, M_g, 3]
            rule_success: [B, S, K_r]
            sub_rule_idx: [B, S, K_r]
        """
        B, S, _ = queries.shape
        G = remaining.shape[2]
        M_g = grounding_body.shape[2]
        dev = queries.device
        pad = self.padding_idx
        c_no = self.constant_no
        E = c_no + 1
        K_r = self.K_r
        V = self.max_vars_per_rule
        Bmax = self.rule_index.rules_bodies_sorted.shape[1] if self.num_rules > 0 else 1

        if self.num_rules == 0:
            return (
                torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
                torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            )

        query_preds = queries[:, :, 0]

        # Table-based rule lookup
        N = B * S
        sub_rule_idx_flat, sub_rule_mask_flat = self.rule_index.lookup_by_table(
            query_preds.reshape(-1))
        sub_rule_idx = sub_rule_idx_flat.view(B, S, K_r)
        sub_rule_mask = sub_rule_mask_flat.view(B, S, K_r)

        # Gather rule data
        flat_ridx = sub_rule_idx.reshape(-1)  # [B*S*K_r]
        sub_heads = self.rule_index.rules_heads_sorted[flat_ridx]  # [N_r, 3]
        sub_bodies = self.rule_index.rules_bodies_sorted[flat_ridx]  # [N_r, Bmax, 3]
        sub_lens = self.rule_index.rule_lens_sorted[flat_ridx]  # [N_r]

        N_r = B * S * K_r

        # ---- Standardization Apart ----
        # Each (batch, state, rule) triple gets a unique variable namespace
        nv_exp = next_var_indices.view(B, 1, 1).expand(B, S, K_r)  # [B, S, K_r]
        state_offsets = torch.arange(S * K_r, device=dev).view(1, S, K_r) * V
        rule_var_base = (nv_exp + state_offsets).reshape(N_r)  # [N_r]

        # Rename head variables
        template_start = E
        std_heads = sub_heads.clone()
        is_var_h = (std_heads[:, 1:] >= template_start)
        h_offset = rule_var_base.unsqueeze(1).expand(N_r, 2)
        std_heads_args = torch.where(
            is_var_h,
            std_heads[:, 1:] - template_start + h_offset,
            std_heads[:, 1:],
        )
        std_heads = torch.cat([std_heads[:, 0:1], std_heads_args], dim=1)

        # Rename body variables
        std_bodies = sub_bodies.clone()
        is_var_b = (std_bodies[:, :, 1:] >= template_start)
        b_offset = rule_var_base.view(N_r, 1, 1).expand(N_r, Bmax, 2)
        std_bodies_args = torch.where(
            is_var_b,
            std_bodies[:, :, 1:] - template_start + b_offset,
            std_bodies[:, :, 1:],
        )
        std_bodies = torch.cat([std_bodies[:, :, 0:1], std_bodies_args], dim=2)

        # ---- Unification ----
        flat_queries = queries.unsqueeze(2).expand(B, S, K_r, 3).reshape(N_r, 3)
        ok_flat, subs_flat = unify_one_to_one(flat_queries, std_heads, c_no, pad)
        rule_success = ok_flat.view(B, S, K_r)
        rule_subs = subs_flat.view(B, S, K_r, 2, 2)

        rule_success = (
            rule_success & sub_rule_mask
            & state_valid.unsqueeze(-1) & active_mask.unsqueeze(-1)
        )

        # ---- Apply subs to [body, remaining, grounding_body] ----
        subs_flat_apply = rule_subs.reshape(N_r, 2, 2)
        rem_exp = remaining.unsqueeze(2).expand(B, S, K_r, G, 3).reshape(N_r, G, 3)
        gbody_exp = grounding_body.unsqueeze(2).expand(
            B, S, K_r, M_g, 3).reshape(N_r, M_g, 3)
        combined = torch.cat([std_bodies, rem_exp, gbody_exp], dim=1)
        combined = apply_substitutions(combined, subs_flat_apply, pad)

        rule_body_subst = combined[:, :Bmax, :].view(B, S, K_r, Bmax, 3)
        rule_remaining = combined[:, Bmax:Bmax + G, :].view(B, S, K_r, G, 3)
        rule_gbody_out = combined[:, Bmax + G:, :].view(B, S, K_r, M_g, 3)

        # Mask body atoms beyond rule length
        sub_lens_v = sub_lens.view(B, S, K_r)
        atom_idx = torch.arange(Bmax, device=dev).view(1, 1, 1, Bmax)
        inactive = atom_idx >= sub_lens_v.unsqueeze(-1)
        rule_body_subst = torch.where(
            inactive.unsqueeze(-1).expand(B, S, K_r, Bmax, 3),
            torch.tensor(pad, dtype=torch.long, device=dev),
            rule_body_subst,
        )

        # Build new goals = body + remaining
        rule_goals = torch.full(
            (B, S, K_r, G, 3), pad, dtype=torch.long, device=dev)
        rule_goals[:, :, :, :Bmax, :] = rule_body_subst
        n_rem = min(G - Bmax, G)
        if n_rem > 0:
            rule_goals[:, :, :, Bmax:Bmax + n_rem, :] = rule_remaining[:, :, :, :n_rem, :]

        return rule_goals, rule_gbody_out, rule_success, sub_rule_idx


# ============================================================================
# RTFGrounder — two-level Rule-Then-Fact (skeleton)
# ============================================================================

class RTFGrounder(BCGrounder):
    """Two-level Rule-Then-Fact: K = K_f * K_r.

    First resolves queries against rule heads, then resolves body atoms
    against facts. Supports body_order_agnostic and rtf_cascade options.
    """

    def __init__(
        self, *args,
        body_order_agnostic: bool = False,
        rtf_cascade: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.body_order_agnostic = body_order_agnostic
        self.rtf_cascade = rtf_cascade

    def _compute_K_uncapped(self) -> int:
        return self.K_f * self.K_r

    def _resolve_facts(self, queries, remaining, grounding_body,
                       state_valid, active_mask):
        raise NotImplementedError("RTFGrounder._resolve_facts not yet implemented")

    def _resolve_rules(self, queries, remaining, grounding_body,
                       state_valid, active_mask, next_var_indices):
        raise NotImplementedError("RTFGrounder._resolve_rules not yet implemented")
