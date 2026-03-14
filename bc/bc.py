"""BCGrounder — unified backward chaining with configurable resolution.

The 5-stage pipeline per step:
    1. SELECT:  extract query atom, remaining goals, active mask
    2. RESOLVE FACTS: unify query with facts (strategy-dependent)
    3. RESOLVE RULES: unify query with rule heads (strategy-dependent)
    4. PACK:    flatten S×K children → compact to S_out
    5. POSTPROCESS: prune ground facts, compact atoms, collect groundings

Resolution strategies:
    'sld': SLD — facts + rules independently (K = K_f + K_r)
    'rtf': RTF — rule head first, then body-fact (K = K_f * K_r)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.base import Grounder
from grounder.bc.common import compact_atoms, pack_combined, pack_fact_rule
from grounder.bc.common import collect_groundings, prune_ground_facts
from grounder.fact_index import ArgKeyFactIndex
from grounder.primitives import apply_substitutions, unify_one_to_one
from grounder.resolution.rtf import resolve_rules_with_facts
from grounder.types import ForwardResult


class BCGrounder(Grounder):
    """Backward-chaining grounder with configurable resolution strategy.

    Args:
        max_goals:               G dimension (max atoms per proof-goal state)
        depth:                   number of proof steps (fixed at construction)
        resolution:              'sld' (K=K_f+K_r) or 'rtf' (K=K_f*K_r)
        body_order_agnostic:     RTF: try all body atom positions (default False)
        rtf_cascade:             RTF: multi-step cascade resolution (default False)
        max_states:              S cap (max concurrent states). Default: K.
        K_MAX:                   hard cap on K
        max_derived_per_state:   explicit K override (None = auto)
        implicit_atom_selection: resolve ALL atom positions (unrolled, compile-safe)
        standardization_mode:    'none' | 'offset' | 'canonical'
        runtime_var_end_index:   upper bound for variable IDs
        compile_mode:            None | 'reduce-overhead' | 'max-autotune'
        max_total_groundings:    output budget for grounding collection (tG)
        max_groundings_per_rule: per-rule budget; caps tG to R_eff * this value
        track_grounding_body:    whether to suppress facts when gbody is uninitialized
    """

    def __init__(
        self,
        *args,
        max_goals: int,
        depth: int = 1,
        resolution: str = "sld",
        body_order_agnostic: bool = False,
        rtf_cascade: bool = False,
        max_states: Optional[int] = None,
        K_MAX: int = 550,
        max_derived_per_state: Optional[int] = None,
        implicit_atom_selection: bool = False,
        standardization_mode: str = "none",
        runtime_var_end_index: Optional[int] = None,
        compile_mode: Optional[str] = None,
        max_total_groundings: int = 64,
        max_groundings_per_rule: Optional[int] = None,
        track_grounding_body: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Resolution strategy — must be set before _compute_K_uncapped()
        self.resolution = resolution
        self.body_order_agnostic = body_order_agnostic
        self.rtf_cascade = rtf_cascade

        self.max_goals = max_goals  # G
        self.depth = depth
        self.implicit_atom_selection = implicit_atom_selection
        self.standardization_mode = standardization_mode
        self.runtime_var_end_index = runtime_var_end_index
        self.compile_mode = compile_mode
        self.track_grounding_body = track_grounding_body

        # RTF-specific: max fact pairs for body-level resolution
        if self.resolution == "rtf":
            self._max_fact_pairs_body = self.K_f

        # K computation (subclass may override _compute_K_uncapped)
        K_uncapped = self._compute_K_uncapped()
        self.K = min(K_uncapped, K_MAX)
        if max_derived_per_state is not None:
            self.K = int(max_derived_per_state)

        # Cap K_f to K to avoid OOM on intermediate tensors [B, S, K_f, ...]
        # Extra fact candidates beyond K are discarded during PACK anyway.
        if self.K_f > self.K:
            self.K_f = self.K

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

        # Grounding collection budget — cap by R_eff * per-rule budget
        # (matches BCPrologStatic sizing: avoids oversized output tensors)
        if max_groundings_per_rule is not None:
            R_eff = self.rule_index.R_eff
            self.effective_total_G = min(
                max_total_groundings,
                R_eff * max(max_groundings_per_rule, 1),
            )
        else:
            self.effective_total_G = max_total_groundings

        # Build compiled step functions
        self._build_compiled_fns()

    def _compute_K_uncapped(self) -> int:
        """Compute K before capping. Dispatches on resolution strategy."""
        if self.resolution == "rtf":
            return self.K_f * self.K_r
        return self.K_f + self.K_r  # SLD default

    def _build_compiled_fns(self) -> None:
        """Build torch.compile wrapper for fused step+postprocess."""
        if (
            self.compile_mode
            and self.depth > 1
            and self._device.type == "cuda"
        ):
            self._fn_step = torch.compile(
                self._step_and_postprocess, fullgraph=True,
                mode=self.compile_mode)
            # CUDA graphs need clones between replays to avoid buffer overwrite
            self._clone_between_steps = (self.compile_mode == 'reduce-overhead')
            # Signal to outer model that this grounder manages its own compilation
            # (outer model should NOT wrap this in a monolithic torch.compile)
            self._multi_step = True
        else:
            self._fn_step = self._step_and_postprocess
            self._clone_between_steps = False
            self._multi_step = False

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

        # Zero padded atoms so downstream lookups never see out-of-range
        # indices (padding_idx lives above the entity/predicate space).
        queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)

        remaining = proof_goals.clone()
        remaining[:, :, 0, :] = self.padding_idx
        return queries, remaining, active_mask

    # ------------------------------------------------------------------
    # Shared rule-head unification (used by SLD + RTF)
    # ------------------------------------------------------------------

    def _resolve_rule_heads(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Level-1 rule head unification: lookup + standardize + unify + apply subs.

        Returns:
            rule_body_subst: [B, S, K_r, Bmax, 3]  substituted body atoms
            rule_remaining:  [B, S, K_r, G, 3]      remaining with subs applied
            rule_gbody:      [B, S, K_r, M_g, 3]    grounding body with subs applied
            rule_success:    [B, S, K_r]             success mask
            sub_rule_idx:    [B, S, K_r]             original rule indices
            sub_lens:        [B, S, K_r]             rule body lengths
            Bmax:            int                     max body width
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

        query_preds = queries[:, :, 0]

        # Segment-based rule lookup (returns positions into sorted arrays)
        N = B * S
        sorted_pos_flat, sub_rule_mask_flat, _ = self.rule_index.lookup_by_segments(
            query_preds.reshape(-1), K_r)
        sub_rule_mask = sub_rule_mask_flat.view(B, S, K_r)

        # Clamp for safe indexing (invalid positions masked by sub_rule_mask)
        R = self.rule_index.rules_heads_sorted.shape[0]
        safe_pos = sorted_pos_flat.clamp(0, max(R - 1, 0))

        # Get original rule indices for sub_rule_idx (used by _pack_step)
        sub_rule_idx = self.rule_index.rules_idx_sorted[safe_pos].view(B, S, K_r)

        # Gather rule data using sorted positions
        flat_sorted_pos = safe_pos.reshape(-1)  # [B*S*K_r]
        sub_heads = self.rule_index.rules_heads_sorted[flat_sorted_pos]  # [N_r, 3]
        sub_bodies = self.rule_index.rules_bodies_sorted[flat_sorted_pos]  # [N_r, Bmax, 3]
        sub_lens = self.rule_index.rule_lens_sorted[flat_sorted_pos]  # [N_r]

        N_r = B * S * K_r

        # ---- Standardization Apart ----
        # Each (batch, state) pair gets a unique variable namespace.
        nv_exp = next_var_indices.view(B, 1, 1).expand(B, S, K_r)  # [B, S, K_r]
        state_offsets = torch.arange(S, device=dev).view(1, S, 1).expand(1, S, K_r) * V
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

        # ---- Apply subs to [body, remaining] (+ grounding_body if tracked) ----
        subs_flat_apply = rule_subs.reshape(N_r, 2, 2)
        rem_exp = remaining.unsqueeze(2).expand(B, S, K_r, G, 3).reshape(N_r, G, 3)

        if self.track_grounding_body:
            gbody_exp = grounding_body.unsqueeze(2).expand(
                B, S, K_r, M_g, 3).reshape(N_r, M_g, 3)
            combined = torch.cat([std_bodies, rem_exp, gbody_exp], dim=1)
            combined = apply_substitutions(combined, subs_flat_apply, pad)
            rule_body_subst = combined[:, :Bmax, :].view(B, S, K_r, Bmax, 3)
            rule_remaining = combined[:, Bmax:Bmax + G, :].view(B, S, K_r, G, 3)
            rule_gbody_out = combined[:, Bmax + G:, :].view(B, S, K_r, M_g, 3)
        else:
            combined = torch.cat([std_bodies, rem_exp], dim=1)
            combined = apply_substitutions(combined, subs_flat_apply, pad)
            rule_body_subst = combined[:, :Bmax, :].view(B, S, K_r, Bmax, 3)
            rule_remaining = combined[:, Bmax:, :].view(B, S, K_r, G, 3)
            rule_gbody_out = torch.zeros(B, S, K_r, M_g, 3, dtype=torch.long, device=dev)

        # Mask body atoms beyond rule length
        sub_lens_v = sub_lens.view(B, S, K_r)
        atom_idx = torch.arange(Bmax, device=dev).view(1, 1, 1, Bmax)
        inactive = atom_idx >= sub_lens_v.unsqueeze(-1)
        rule_body_subst = torch.where(
            inactive.unsqueeze(-1).expand(B, S, K_r, Bmax, 3),
            torch.tensor(pad, dtype=torch.long, device=dev),
            rule_body_subst,
        )

        return rule_body_subst, rule_remaining, rule_gbody_out, rule_success, sub_rule_idx, sub_lens_v, Bmax

    # ------------------------------------------------------------------
    # Stages 2+3: RESOLVE — dispatches on self.resolution
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
        """Resolve query atoms against facts. Dispatches on resolution strategy.

        Returns:
            fact_goals:   [B, S, K_f, G, 3]
            fact_gbody:   [B, S, K_f, M, 3]
            fact_success: [B, S, K_f]
        """
        if self.resolution == "rtf":
            return self._resolve_facts_rtf(
                queries, remaining, grounding_body, state_valid, active_mask,
                excluded_queries)
        return self._resolve_facts_sld(
            queries, remaining, grounding_body, state_valid, active_mask,
            excluded_queries)

    def _resolve_rules(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Resolve query atoms against rules. Dispatches on resolution strategy.

        Returns:
            rule_goals:   [B, S, K_r, G, 3]
            rule_gbody:   [B, S, K_r, M, 3]
            rule_success: [B, S, K_r]
            sub_rule_idx: [B, S, K_r]
        """
        if self.resolution == "rtf":
            return self._resolve_rules_rtf(
                queries, remaining, grounding_body, state_valid, active_mask,
                next_var_indices)
        return self._resolve_rules_sld(
            queries, remaining, grounding_body, state_valid, active_mask,
            next_var_indices)

    # ------------------------------------------------------------------
    # SLD fact resolution (from PrologGrounder)
    # ------------------------------------------------------------------

    def _resolve_facts_sld(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if isinstance(self.fact_index, ArgKeyFactIndex):
            return self._resolve_facts_argkey(
                queries, remaining, grounding_body, state_valid, active_mask,
                excluded_queries)
        return self._resolve_facts_enumerate(
            queries, remaining, grounding_body, state_valid, active_mask,
            excluded_queries)

    def _resolve_facts_argkey(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        excluded_queries: Optional[Tensor] = None,
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

        # Store last fact item idx for subclasses (used by RL engines)
        self._last_fact_item_idx = fact_item_idx.view(B, S, K_f)

        # Unify queries with facts
        q_exp = flat_q.unsqueeze(1).expand(-1, K_f, -1)  # [N, K_f, 3]
        ok_flat, subs_flat = unify_one_to_one(
            q_exp.reshape(-1, 3), fact_atoms.reshape(-1, 3),
            self.constant_no, pad)
        ok = ok_flat.view(N, K_f)
        subs = subs_flat.view(N, K_f, 2, 2)

        success = ok & fact_valid & flat_active.unsqueeze(1)

        # Cycle prevention: exclude queries matching excluded_queries
        if excluded_queries is not None and self.facts_idx.numel() > 0:
            # excluded_queries: [B, 1, 3] — broadcast over S states
            excl_flat = excluded_queries[:, 0, :].unsqueeze(1).expand(B, S, 3).reshape(N, 1, 3)
            fact_atoms_r = self.facts_idx[safe_idx.view(-1)].view(N, K_f, 3)
            match_excl = (fact_atoms_r == excl_flat).all(dim=-1)  # [N, K_f]
            success = success & ~match_excl

        # Apply substitutions to remaining (+ grounding_body if tracked)
        subs_flat_for_apply = subs.reshape(N * K_f, 2, 2)
        flat_rem = remaining.reshape(N, G, 3)
        rem_exp = flat_rem.unsqueeze(1).expand(-1, K_f, -1, -1).reshape(N * K_f, G, 3)

        if self.track_grounding_body:
            flat_gbody = grounding_body.reshape(N, M_g, 3)
            combined = torch.cat([
                rem_exp,
                flat_gbody.unsqueeze(1).expand(-1, K_f, -1, -1).reshape(N * K_f, M_g, 3),
            ], dim=1)  # [N*K_f, G+M_g, 3]
            combined = apply_substitutions(combined, subs_flat_for_apply, pad)
            fact_goals = combined[:, :G, :].view(B, S, K_f, G, 3)
            fact_gbody = combined[:, G:, :].view(B, S, K_f, M_g, 3)
        else:
            fact_goals = apply_substitutions(rem_exp, subs_flat_for_apply, pad).view(B, S, K_f, G, 3)
            fact_gbody = torch.zeros(B, S, K_f, M_g, 3, dtype=torch.long, device=dev)

        # Mask out invalid entries
        pad_t = torch.tensor(pad, dtype=torch.long, device=dev)
        fact_goals = torch.where(
            success.view(B, S, K_f, 1, 1), fact_goals, pad_t)
        if self.track_grounding_body:
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
        excluded_queries: Optional[Tensor] = None,
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

        pred = queries[:, :, 0]
        arg0 = queries[:, :, 1]
        arg1 = queries[:, :, 2]

        arg0_ground = (arg0 <= c_no)
        arg1_ground = (arg1 <= c_no)
        has_ground = arg0_ground | arg1_ground
        both_ground = arg0_ground & arg1_ground

        # Active: state_valid AND at least one ground arg
        is_active = state_valid & active_mask

        use_arg0 = arg0_ground
        direction = torch.where(
            use_arg0, torch.zeros_like(arg0), torch.ones_like(arg0))
        # Safe defaults for inactive/unground states (0 is always valid entity index)
        bound_arg = torch.where(
            has_ground & is_active,
            torch.where(use_arg0, arg0, arg1),
            torch.zeros_like(arg0))
        safe_pred = torch.where(is_active, pred, torch.zeros_like(pred))
        free_var = torch.where(use_arg0, arg1, arg0)

        # Enumerate candidates
        cands, cand_mask = self.fact_index.enumerate(
            safe_pred.reshape(-1), bound_arg.reshape(-1), direction.reshape(-1))
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

        # Apply subs to remaining (+ grounding_body if tracked)
        rem_exp = remaining.unsqueeze(2).expand(B, S, K_f, G, 3).reshape(N_f, G, 3)

        if self.track_grounding_body:
            gbody_exp = grounding_body.unsqueeze(2).expand(B, S, K_f, M_g, 3).reshape(N_f, M_g, 3)
            combined = torch.cat([rem_exp, gbody_exp], dim=1)
            combined = apply_substitutions(combined, subs, pad)
            fact_goals = combined[:, :G, :].view(B, S, K_f, G, 3)
            fact_gbody = combined[:, G:, :].view(B, S, K_f, M_g, 3)
        else:
            fact_goals = apply_substitutions(rem_exp, subs, pad).view(B, S, K_f, G, 3)
            fact_gbody = torch.zeros(B, S, K_f, M_g, 3, dtype=torch.long, device=dev)

        return fact_goals, fact_gbody, fact_success

    # ------------------------------------------------------------------
    # SLD rule resolution (from PrologGrounder)
    # ------------------------------------------------------------------

    def _resolve_rules_sld(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """SLD rule resolution: head unification + body assembly.

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
        K_r = self.K_r

        if self.num_rules == 0:
            return (
                torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
                torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            )

        # Shared rule-head unification (lookup + standardize + unify + apply subs)
        rule_body_subst, rule_remaining, rule_gbody_out, rule_success, sub_rule_idx, _, Bmax = \
            self._resolve_rule_heads(
                queries, remaining, grounding_body,
                state_valid, active_mask, next_var_indices)

        # Build new goals = body + remaining
        if not self.track_grounding_body:
            # RL mode: wider output (Bmax+G) to preserve all remaining atoms.
            G_out = Bmax + G
            rule_goals = torch.full(
                (B, S, K_r, G_out, 3), pad, dtype=torch.long, device=dev)
            rule_goals[:, :, :, :Bmax, :] = rule_body_subst
            rule_goals[:, :, :, Bmax:Bmax + G, :] = rule_remaining
        else:
            # TS mode: fit into G width (body occupies first Bmax slots)
            rule_goals = torch.full(
                (B, S, K_r, G, 3), pad, dtype=torch.long, device=dev)
            rule_goals[:, :, :, :Bmax, :] = rule_body_subst
            n_rem = min(G - Bmax, G)
            if n_rem > 0:
                rule_goals[:, :, :, Bmax:Bmax + n_rem, :] = rule_remaining[:, :, :, :n_rem, :]

        return rule_goals, rule_gbody_out, rule_success, sub_rule_idx

    # ------------------------------------------------------------------
    # RTF fact resolution (empty — all work done in _resolve_rules_rtf)
    # ------------------------------------------------------------------

    def _resolve_facts_rtf(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, S, _ = queries.shape
        G = remaining.shape[2]
        M_g = grounding_body.shape[2]
        dev = queries.device
        pad = self.padding_idx
        return (
            torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
        )

    # ------------------------------------------------------------------
    # RTF rule resolution (two-level: rules then body-fact)
    # ------------------------------------------------------------------

    def _resolve_rules_rtf(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Two-level resolution: rule head unification then body-fact resolution.

        Returns:
            rule_goals:   [B, S, K_rtf, G, 3]
            rule_gbody:   [B, S, K_rtf, M_g, 3]
            rule_success: [B, S, K_rtf]
            sub_rule_idx: [B, S, K_rtf]
        """
        B, S, _ = queries.shape
        G = remaining.shape[2]
        M_g = grounding_body.shape[2]
        dev = queries.device
        pad = self.padding_idx
        K_r = self.K_r

        if self.num_rules == 0:
            return (
                torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
                torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            )

        # ================================================================
        # Level 1: Shared rule head unification
        # ================================================================
        rule_body_subst, rule_remaining, rule_gbody_l1, rule_success_l1, \
            sub_rule_idx_l1, _, Bmax = self._resolve_rule_heads(
                queries, remaining, grounding_body,
                state_valid, active_mask, next_var_indices)

        # Build intermediate states: [body | remaining] shape [B, S, K_r, Bmax+G, 3]
        M_inter = Bmax + G
        rule_states_l1 = torch.full(
            (B, S, K_r, M_inter, 3), pad, dtype=torch.long, device=dev)
        rule_states_l1[:, :, :, :Bmax, :] = rule_body_subst
        n_rem = min(G, M_inter - Bmax)
        if n_rem > 0:
            rule_states_l1[:, :, :, Bmax:Bmax + n_rem, :] = rule_remaining[:, :, :, :n_rem, :]

        # ================================================================
        # Level 2: Body-fact resolution
        # ================================================================
        # Flatten S into batch for body-fact helpers: [B*S, K_r, M_inter, 3]
        rule_states_flat = rule_states_l1.reshape(B * S, K_r, M_inter, 3)
        rule_success_flat = rule_success_l1.reshape(B * S, K_r)

        resolved_flat, resolved_ok_flat = resolve_rules_with_facts(
            self, rule_states_flat, rule_success_flat, excluded_queries=None)

        K_rtf = resolved_ok_flat.shape[1]
        G_out = resolved_flat.shape[2]

        # Reshape back to [B, S, K_rtf, ...]
        resolved = resolved_flat.view(B, S, K_rtf, G_out, 3)
        resolved_ok = resolved_ok_flat.view(B, S, K_rtf)

        # Pad/trim G dimension to match expected G
        if G_out < G:
            resolved = torch.nn.functional.pad(
                resolved, (0, 0, 0, G - G_out), value=pad)
        elif G_out > G and self.track_grounding_body:
            # Only trim in TS mode; RL mode keeps wider output so
            # get_derived_states_compiled can truncate to M properly.
            resolved = resolved[:, :, :, :G, :]

        # Build rule_gbody: propagate from level-1
        # Each level-1 rule match spawns K_f children → expand gbody
        K_f_body = self._max_fact_pairs_body
        rule_gbody_out = rule_gbody_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_body, M_g, 3).reshape(B, S, K_r * K_f_body, M_g, 3)
        # Trim/pad to match K_rtf
        if rule_gbody_out.shape[2] < K_rtf:
            rule_gbody_out = torch.nn.functional.pad(
                rule_gbody_out,
                (0, 0, 0, 0, 0, K_rtf - rule_gbody_out.shape[2]),
                value=0)
        elif rule_gbody_out.shape[2] > K_rtf:
            rule_gbody_out = rule_gbody_out[:, :, :K_rtf]

        # Build sub_rule_idx: expand level-1 rule indices
        sub_rule_idx_out = sub_rule_idx_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_body).reshape(B, S, K_r * K_f_body)
        if sub_rule_idx_out.shape[2] < K_rtf:
            sub_rule_idx_out = torch.nn.functional.pad(
                sub_rule_idx_out, (0, K_rtf - sub_rule_idx_out.shape[2]), value=0)
        elif sub_rule_idx_out.shape[2] > K_rtf:
            sub_rule_idx_out = sub_rule_idx_out[:, :, :K_rtf]

        return resolved, rule_gbody_out, resolved_ok, sub_rule_idx_out

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
        proof_goals, _, _ = prune_ground_facts(
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
    # Fused step + postprocess (single compiled unit per depth)
    # ------------------------------------------------------------------

    def _step_and_postprocess(
        self,
        grounding_body: Tensor,     # [B, S, M, 3]
        proof_goals: Tensor,        # [B, S, G, 3]
        top_ridx: Tensor,           # [B, S]
        state_valid: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
        collected_body: Tensor,     # [B, tG, M, 3]
        collected_mask: Tensor,     # [B, tG]
        collected_ridx: Tensor,     # [B, tG]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused step + postprocess: single compiled unit per depth.

        Returns:
            grounding_body:  [B, S, M, 3]
            proof_goals:     [B, S, G, 3]
            top_ridx:        [B, S]
            state_valid:     [B, S]
            next_var_indices:[B]
            collected_body:  [B, tG, M, 3]
            collected_mask:  [B, tG]
            collected_ridx:  [B, tG]
        """
        # Step: SELECT -> RESOLVE -> PACK
        grounding_body, proof_goals, top_ridx, state_valid, next_var_indices = \
            self._step_impl(
                grounding_body, proof_goals, top_ridx,
                state_valid, next_var_indices)

        # Postprocess: prune ground facts + compact + collect groundings
        proof_goals, collected_body, collected_mask, collected_ridx, state_valid = \
            self._postprocess(
                grounding_body, proof_goals, state_valid, top_ridx,
                collected_body, collected_mask, collected_ridx)

        return (grounding_body, proof_goals, top_ridx, state_valid,
                next_var_indices, collected_body, collected_mask, collected_ridx)

    # ------------------------------------------------------------------
    # Forward (multi-depth proof loop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        queries: Tensor,        # [B, 3]
        query_mask: Tensor,     # [B] bool
        on_depth_complete=None,  # Optional[Callable[[int, Tensor], None]]
    ) -> ForwardResult:
        """Multi-depth backward-chaining proof.

        Args:
            queries:    [B, 3] query atoms to prove
            query_mask: [B] bool mask for valid queries
            on_depth_complete: Optional callback called after each depth with
                (depth, newly_proved) where newly_proved is [B] bool mask of
                queries that were first proved at this depth.

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

        # --- Track which queries have been proved for on_depth_complete ---
        prev_proved = torch.zeros(B, dtype=torch.bool, device=dev)

        # --- Proof loop: depth steps ---
        for _d in range(self.depth):
            # Clone between CUDA graph replays to prevent buffer overwrite.
            # Each step's CUDA graph writes to fixed output addresses;
            # cloning ensures the next replay doesn't corrupt live inputs.
            if _d > 0 and self._clone_between_steps:
                grounding_body = grounding_body.clone()
                proof_goals = proof_goals.clone()
                top_ridx = top_ridx.clone()
                state_valid = state_valid.clone()
                next_var_indices = next_var_indices.clone()
                collected_body = collected_body.clone()
                collected_mask = collected_mask.clone()
                collected_ridx = collected_ridx.clone()

            # Save state_valid before step+postprocess deactivates proved states
            sv_before = state_valid if on_depth_complete is None else state_valid.clone()

            (grounding_body, proof_goals, top_ridx, state_valid,
             next_var_indices, collected_body, collected_mask,
             collected_ridx) = self._fn_step(
                grounding_body, proof_goals, top_ridx,
                state_valid, next_var_indices,
                collected_body, collected_mask, collected_ridx)

            # Invoke depth callback if provided
            if on_depth_complete is not None:
                # Detect proofs from goal state: a query is proved when any
                # of its states has all goals resolved (all padding) and is
                # still valid (before collect_groundings deactivated it).
                all_done = (proof_goals[:, :, :, 0] == pad).all(dim=2)  # [B, S]
                has_proof = (all_done & sv_before).any(dim=1)  # [B]
                newly_proved = has_proof & ~prev_proved
                on_depth_complete(_d + 1, newly_proved)
                prev_proved = prev_proved | newly_proved

        collected_count = collected_mask.sum(dim=1)

        return ForwardResult(
            collected_body=collected_body,
            collected_mask=collected_mask,
            collected_count=collected_count,
            collected_ridx=collected_ridx,
        )


# ======================================================================
# Thin subclasses — backward-compatible aliases
# ======================================================================


class PrologGrounder(BCGrounder):
    """Single-level Prolog resolution: K = K_f + K_r.

    Thin subclass that sets resolution='sld'. Overrides resolve methods
    to call SLD variants directly (avoids dispatch in compiled graph).
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("resolution", "sld")
        super().__init__(*args, **kwargs)

    def _compute_K_uncapped(self) -> int:
        return self.K_f + self.K_r

    def _resolve_facts(
        self,
        queries: Tensor,
        remaining: Tensor,
        grounding_body: Tensor,
        state_valid: Tensor,
        active_mask: Tensor,
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self._resolve_facts_sld(
            queries, remaining, grounding_body, state_valid, active_mask,
            excluded_queries)

    def _resolve_rules(
        self,
        queries: Tensor,
        remaining: Tensor,
        grounding_body: Tensor,
        state_valid: Tensor,
        active_mask: Tensor,
        next_var_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._resolve_rules_sld(
            queries, remaining, grounding_body, state_valid, active_mask,
            next_var_indices)


class RTFGrounder(BCGrounder):
    """Two-level Rule-Then-Fact: K = K_f * K_r.

    Thin subclass that sets resolution='rtf'. Overrides resolve methods
    to call RTF variants directly (avoids dispatch in compiled graph).
    """

    def __init__(
        self, *args,
        body_order_agnostic: bool = False,
        rtf_cascade: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("resolution", "rtf")
        super().__init__(
            *args,
            body_order_agnostic=body_order_agnostic,
            rtf_cascade=rtf_cascade,
            **kwargs,
        )

    def _compute_K_uncapped(self) -> int:
        return self.K_f * self.K_r

    def _resolve_facts(
        self,
        queries: Tensor,
        remaining: Tensor,
        grounding_body: Tensor,
        state_valid: Tensor,
        active_mask: Tensor,
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self._resolve_facts_rtf(
            queries, remaining, grounding_body, state_valid, active_mask,
            excluded_queries)

    def _resolve_rules(
        self,
        queries: Tensor,
        remaining: Tensor,
        grounding_body: Tensor,
        state_valid: Tensor,
        active_mask: Tensor,
        next_var_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._resolve_rules_rtf(
            queries, remaining, grounding_body, state_valid, active_mask,
            next_var_indices)
