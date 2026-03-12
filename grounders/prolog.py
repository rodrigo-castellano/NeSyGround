"""PrologGrounder — single-level Prolog: facts + rules independently.

K = K_f + K_r. Resolves query atoms independently against facts and rules
via MGU. Facts and rules produce separate child sets that are concatenated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.fact_index import ArgKeyFactIndex
from grounder.grounders.backward import BCGrounder
from grounder.primitives import apply_substitutions, unify_one_to_one


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

        # Store last fact item idx for subclasses
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
