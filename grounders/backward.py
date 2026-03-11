"""BCGrounder — backward chaining with step() + multi-depth forward().

The 5-stage pipeline per step:
    1. SELECT:  extract query atom, remaining goals, active mask
    2. RESOLVE FACTS: unify query with facts
    3. RESOLVE RULES: unify query with rule heads, add body atoms
    4. PACK:    flatten S×K children → compact to S_out
    5. POSTPROCESS: prune ground facts, compact atoms, collect groundings
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.grounders.base import Grounder
from grounder.packing import compact_atoms, pack_combined, pack_fact_rule
from grounder.postprocessing import collect_groundings, prune_ground_facts_3d
from grounder.types import ForwardResult


class BCGrounder(Grounder):
    """Backward-chaining grounder with single-step and multi-depth pipelines.

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
        track_grounding_body:    whether to suppress facts when gbody is uninitialized
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
        track_grounding_body: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_goals = max_goals  # G
        self.depth = depth
        self.implicit_atom_selection = implicit_atom_selection
        self.standardization_mode = standardization_mode
        self.runtime_var_end_index = runtime_var_end_index
        self.compile_mode = compile_mode
        self.track_grounding_body = track_grounding_body

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
        excluded_queries: Optional[Tensor] = None,
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

        When ``track_grounding_body=False`` (RL mode), uses simplified packing
        that skips gbody/ridx processing for better throughput.

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

        # Fast path: skip gbody/ridx when not tracking grounding body
        if not self.track_grounding_body:
            # Normalize widths: rule_goals may be wider (Bmax+G) in RL mode
            G_f = fact_goals.shape[3]
            G_r = rule_goals.shape[3]
            G_pack = max(G_f, G_r)
            if G_f < G_pack:
                fact_goals = torch.nn.functional.pad(
                    fact_goals, (0, 0, 0, G_pack - G_f), value=self.padding_idx)
            elif G_r < G_pack:
                rule_goals = torch.nn.functional.pad(
                    rule_goals, (0, 0, 0, G_pack - G_r), value=self.padding_idx)
            flat_goals = torch.cat([
                fact_goals.reshape(B, n_f, G_pack, 3),
                rule_goals.reshape(B, n_r, G_pack, 3),
            ], dim=1)  # [B, n_f+n_r, G_pack, 3]
            flat_valid = torch.cat([
                fact_success.reshape(B, n_f),
                rule_success.reshape(B, n_r),
            ], dim=1)  # [B, n_f+n_r]
            new_goals, counts = pack_combined(
                flat_goals, flat_valid, self.S, G_pack, self.padding_idx)
            new_gbody = torch.zeros(B, self.S, M, 3, dtype=torch.long, device=fact_goals.device)
            new_ridx = torch.zeros(B, self.S, dtype=torch.long, device=fact_goals.device)
            arange_k = torch.arange(self.S, device=fact_goals.device).unsqueeze(0)
            new_valid = arange_k < counts.unsqueeze(1)
            return new_gbody, new_goals, new_ridx, new_valid

        # Full path: track grounding body and rule indices
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
        excluded_queries: Optional[Tensor] = None,
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
            queries, remaining, grounding_body, state_valid, active_mask,
            excluded_queries)

        # Auto-skip facts when gbody is uninitialized (all padding = first depth).
        # Fact matches at depth 0 have no rule body and would waste output slots.
        if self.track_grounding_body:
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

        # Update next_var_indices: advance by S * V (one namespace per state).
        # Rules within a state share a namespace (independent children).
        V = self.max_vars_per_rule
        new_next_var = next_var_indices + S * V

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
