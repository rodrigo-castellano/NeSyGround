"""MGU resolution primitives — resolve_facts and resolve_rules.

Shared by SLD and RTF strategies. All operations are CUDA-graph-safe
(fixed shapes, no .item(), no data-dependent branching).

Two public functions:
  resolve_facts  — targeted lookup or enumerate → unify → substitute
  resolve_rules  — segment lookup → standardize apart → unify head → substitute body
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.fact_index import ArgKeyFactIndex
from grounder.primitives import apply_substitutions, unify_one_to_one


def resolve_facts(
    goals: Tensor,              # [B, S, 3]
    remaining: Tensor,          # [B, S, G, 3]
    fact_index,                 # FactIndex (ArgKey, Inverted, or BlockSparse)
    facts_idx: Tensor,          # [F, 3]
    constant_no: int,
    padding_idx: int,
    K_f: int,
    state_valid: Tensor,        # [B, S]
    active_mask: Tensor,        # [B, S]
    grounding_body: Optional[Tensor] = None,  # [B, S, M_g, 3]
    excluded_queries: Optional[Tensor] = None,  # [B, 1, 3]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Resolve goal atoms against facts via MGU.

    Dispatches on fact_index type:
      - ArgKeyFactIndex: targeted_lookup → unify_one_to_one → substitute
      - Inverted/BlockSparse: enumerate → build substitutions → substitute

    Args:
        goals:              [B, S, 3] query atoms to resolve.
        remaining:          [B, S, G, 3] remaining goal atoms.
        fact_index:         FactIndex instance with targeted_lookup or enumerate.
        facts_idx:          [F, 3] all facts.
        constant_no:        highest constant index.
        padding_idx:        padding value.
        K_f:                max fact matches per goal.
        state_valid:        [B, S] which states are valid.
        active_mask:        [B, S] which states have active goals.
        grounding_body:     [B, S, M_g, 3] optional tracking tensor.
        excluded_queries:   [B, 1, 3] atoms to exclude (cycle prevention).

    Returns:
        children:  [B, S, K_f, G, 3] resolved successor states.
        gbody:     [B, S, K_f, M_g, 3] grounding body with subs applied
                   (zeros if grounding_body is None).
        success:   [B, S, K_f] validity mask.
    """
    if isinstance(fact_index, ArgKeyFactIndex):
        return _resolve_facts_argkey(
            goals, remaining, fact_index, facts_idx,
            constant_no, padding_idx, K_f,
            state_valid, active_mask, grounding_body, excluded_queries)
    return _resolve_facts_enumerate(
        goals, remaining, fact_index,
        constant_no, padding_idx,
        state_valid, active_mask, grounding_body, excluded_queries)


def resolve_rules(
    goals: Tensor,              # [B, S, 3]
    remaining: Tensor,          # [B, S, G, 3]
    rule_index,                 # RuleIndex
    constant_no: int,
    padding_idx: int,
    K_r: int,
    max_vars_per_rule: int,
    num_rules: int,
    state_valid: Tensor,        # [B, S]
    active_mask: Tensor,        # [B, S]
    next_var_indices: Tensor,   # [B]
    grounding_body: Optional[Tensor] = None,  # [B, S, M_g, 3]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    """Resolve goal atoms against rules via MGU head unification.

    Lookup matching rules → standardize apart → unify head → apply subs
    to body and remaining atoms.

    Args:
        goals:              [B, S, 3] query atoms to resolve.
        remaining:          [B, S, G, 3] remaining goal atoms.
        rule_index:         RuleIndex instance.
        constant_no:        highest constant index.
        padding_idx:        padding value.
        K_r:                max rule matches per goal.
        max_vars_per_rule:  max template variables per rule.
        num_rules:          total number of rules.
        state_valid:        [B, S] which states are valid.
        active_mask:        [B, S] which states have active goals.
        next_var_indices:   [B] next available variable index per batch.
        grounding_body:     [B, S, M_g, 3] optional tracking tensor.

    Returns:
        body_subst:    [B, S, K_r, Bmax, 3] substituted rule body atoms.
        rule_remaining:[B, S, K_r, G, 3] remaining goals with subs applied.
        gbody:         [B, S, K_r, M_g, 3] grounding body with subs
                       (zeros if grounding_body is None).
        success:       [B, S, K_r] validity mask.
        rule_idx:      [B, S, K_r] original rule indices.
        sub_lens:      [B, S, K_r] body lengths per matched rule.
        Bmax:          int — max body size across rules.
    """
    B, S, _ = goals.shape
    G = remaining.shape[2]
    M_g = grounding_body.shape[2] if grounding_body is not None else 0
    dev = goals.device
    pad = padding_idx
    c_no = constant_no
    E = c_no + 1
    V = max_vars_per_rule
    Bmax = rule_index.rules_bodies_sorted.shape[1] if num_rules > 0 else 1

    if num_rules == 0:
        return (
            torch.full((B, S, 0, Bmax, 3), pad, dtype=torch.long, device=dev),
            torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
            torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            Bmax,
        )

    query_preds = goals[:, :, 0]

    # ---- Segment-based rule lookup ----
    N = B * S
    sorted_pos_flat, sub_rule_mask_flat, _ = rule_index.lookup_by_segments(
        query_preds.reshape(-1), K_r)
    sub_rule_mask = sub_rule_mask_flat.view(B, S, K_r)

    R = rule_index.rules_heads_sorted.shape[0]
    safe_pos = sorted_pos_flat.clamp(0, max(R - 1, 0))

    # Original rule indices (used downstream for grounding output)
    sub_rule_idx = rule_index.rules_idx_sorted[safe_pos].view(B, S, K_r)

    # Gather rule data using sorted positions
    flat_sorted_pos = safe_pos.reshape(-1)  # [B*S*K_r]
    sub_heads = rule_index.rules_heads_sorted[flat_sorted_pos]      # [N_r, 3]
    sub_bodies = rule_index.rules_bodies_sorted[flat_sorted_pos]    # [N_r, Bmax, 3]
    sub_lens_flat = rule_index.rule_lens_sorted[flat_sorted_pos]    # [N_r]

    N_r = B * S * K_r

    # ---- Standardization Apart ----
    # Each (batch, state) pair gets a unique variable namespace.
    nv_exp = next_var_indices.view(B, 1, 1).expand(B, S, K_r)  # [B, S, K_r]
    state_offsets = torch.arange(S, device=dev).view(1, S, 1).expand(1, S, K_r) * V
    rule_var_base = (nv_exp + state_offsets).reshape(N_r)  # [N_r]

    template_start = E

    # Rename head variables
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
    flat_queries = goals.unsqueeze(2).expand(B, S, K_r, 3).reshape(N_r, 3)
    ok_flat, subs_flat = unify_one_to_one(flat_queries, std_heads, c_no, pad)
    rule_success = ok_flat.view(B, S, K_r)
    rule_subs = subs_flat.view(B, S, K_r, 2, 2)

    rule_success = (
        rule_success & sub_rule_mask
        & state_valid.unsqueeze(-1) & active_mask.unsqueeze(-1)
    )

    # ---- Apply substitutions to [body, remaining, grounding_body] ----
    subs_flat_apply = rule_subs.reshape(N_r, 2, 2)
    rem_exp = remaining.unsqueeze(2).expand(B, S, K_r, G, 3).reshape(N_r, G, 3)

    if grounding_body is not None:
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

    # ---- Mask body atoms beyond rule length ----
    sub_lens_v = sub_lens_flat.view(B, S, K_r)
    atom_idx = torch.arange(Bmax, device=dev).view(1, 1, 1, Bmax)
    inactive = atom_idx >= sub_lens_v.unsqueeze(-1)
    rule_body_subst = torch.where(
        inactive.unsqueeze(-1).expand(B, S, K_r, Bmax, 3),
        torch.tensor(pad, dtype=torch.long, device=dev),
        rule_body_subst,
    )

    return rule_body_subst, rule_remaining, rule_gbody_out, rule_success, sub_rule_idx, sub_lens_v, Bmax


# ---------------------------------------------------------------------------
# Internal dispatch for resolve_facts
# ---------------------------------------------------------------------------

def _resolve_facts_argkey(
    goals: Tensor,              # [B, S, 3]
    remaining: Tensor,          # [B, S, G, 3]
    fact_index,                 # ArgKeyFactIndex
    facts_idx: Tensor,          # [F, 3]
    constant_no: int,
    padding_idx: int,
    K_f: int,
    state_valid: Tensor,        # [B, S]
    active_mask: Tensor,        # [B, S]
    grounding_body: Optional[Tensor],
    excluded_queries: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """ArgKey-based fact resolution: targeted_lookup → unify → substitute."""
    B, S, _ = goals.shape
    G = remaining.shape[2]
    M_g = grounding_body.shape[2] if grounding_body is not None else 0
    dev = goals.device
    pad = padding_idx

    N = B * S
    flat_q = goals.reshape(N, 3)
    flat_active = (active_mask & state_valid).reshape(N)

    # Targeted fact lookup
    fact_item_idx, fact_valid = fact_index.targeted_lookup(flat_q, K_f)
    # fact_item_idx: [N, K_f], fact_valid: [N, K_f]

    F = facts_idx.shape[0]
    if F == 0:
        gbody_out = (
            grounding_body.unsqueeze(2).expand(B, S, K_f, M_g, 3).clone()
            if grounding_body is not None
            else torch.zeros(B, S, K_f, max(M_g, 1), 3, dtype=torch.long, device=dev)
        )
        return (
            torch.full((B, S, K_f, G, 3), pad, dtype=torch.long, device=dev),
            gbody_out,
            torch.zeros(B, S, K_f, dtype=torch.bool, device=dev),
        )

    safe_idx = fact_item_idx.clamp(0, max(F - 1, 0))
    fact_atoms = facts_idx[safe_idx.view(-1)].view(N, K_f, 3)

    # Unify goals with gathered facts
    q_exp = flat_q.unsqueeze(1).expand(-1, K_f, -1)  # [N, K_f, 3]
    ok_flat, subs_flat = unify_one_to_one(
        q_exp.reshape(-1, 3), fact_atoms.reshape(-1, 3),
        constant_no, pad)
    ok = ok_flat.view(N, K_f)
    subs = subs_flat.view(N, K_f, 2, 2)

    success = ok & fact_valid & flat_active.unsqueeze(1)

    # Cycle prevention: exclude queries matching excluded_queries
    if excluded_queries is not None and facts_idx.numel() > 0:
        excl_flat = excluded_queries[:, 0, :].unsqueeze(1).expand(B, S, 3).reshape(N, 1, 3)
        fact_atoms_r = facts_idx[safe_idx.view(-1)].view(N, K_f, 3)
        match_excl = (fact_atoms_r == excl_flat).all(dim=-1)  # [N, K_f]
        success = success & ~match_excl

    # Apply substitutions to remaining (+ grounding_body if tracked)
    subs_flat_for_apply = subs.reshape(N * K_f, 2, 2)
    flat_rem = remaining.reshape(N, G, 3)
    rem_exp = flat_rem.unsqueeze(1).expand(-1, K_f, -1, -1).reshape(N * K_f, G, 3)

    if grounding_body is not None:
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
        fact_gbody = torch.zeros(B, S, K_f, max(M_g, 1), 3, dtype=torch.long, device=dev)

    # Mask invalid entries
    pad_t = torch.tensor(pad, dtype=torch.long, device=dev)
    fact_goals = torch.where(success.view(B, S, K_f, 1, 1), fact_goals, pad_t)
    if grounding_body is not None:
        fact_gbody = torch.where(
            success.view(B, S, K_f, 1, 1), fact_gbody,
            torch.tensor(0, dtype=torch.long, device=dev))

    fact_success = success.view(B, S, K_f)
    return fact_goals, fact_gbody, fact_success


def _resolve_facts_enumerate(
    goals: Tensor,              # [B, S, 3]
    remaining: Tensor,          # [B, S, G, 3]
    fact_index,                 # InvertedFactIndex or BlockSparseFactIndex
    constant_no: int,
    padding_idx: int,
    state_valid: Tensor,        # [B, S]
    active_mask: Tensor,        # [B, S]
    grounding_body: Optional[Tensor],
    excluded_queries: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Enumerate-based fact resolution for Inverted/BlockSparse indices."""
    B, S, _ = goals.shape
    G = remaining.shape[2]
    M_g = grounding_body.shape[2] if grounding_body is not None else 0
    dev = goals.device
    pad = padding_idx
    c_no = constant_no

    pred = goals[:, :, 0]
    arg0 = goals[:, :, 1]
    arg1 = goals[:, :, 2]

    arg0_ground = (arg0 <= c_no)
    arg1_ground = (arg1 <= c_no)
    has_ground = arg0_ground | arg1_ground
    both_ground = arg0_ground & arg1_ground

    is_active = state_valid & active_mask

    use_arg0 = arg0_ground
    direction = torch.where(
        use_arg0, torch.zeros_like(arg0), torch.ones_like(arg0))
    bound_arg = torch.where(
        has_ground & is_active,
        torch.where(use_arg0, arg0, arg1),
        torch.zeros_like(arg0))
    safe_pred = torch.where(is_active, pred, torch.zeros_like(pred))
    free_var = torch.where(use_arg0, arg1, arg0)

    # Enumerate candidates
    cands, cand_mask = fact_index.enumerate(
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

    # Both-ground filter: when both args are ground, candidate must match
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

    if grounding_body is not None:
        gbody_exp = grounding_body.unsqueeze(2).expand(
            B, S, K_f, M_g, 3).reshape(N_f, M_g, 3)
        combined = torch.cat([rem_exp, gbody_exp], dim=1)
        combined = apply_substitutions(combined, subs, pad)
        fact_goals = combined[:, :G, :].view(B, S, K_f, G, 3)
        fact_gbody = combined[:, G:, :].view(B, S, K_f, M_g, 3)
    else:
        fact_goals = apply_substitutions(rem_exp, subs, pad).view(B, S, K_f, G, 3)
        fact_gbody = torch.zeros(B, S, K_f, max(M_g, 1), 3, dtype=torch.long, device=dev)

    return fact_goals, fact_gbody, fact_success


# ======================================================================
# init_mgu — shared parameter computation for SLD / RTF
# ======================================================================

def init_mgu(
    resolution: str,
    K_f: int,
    K_r: int,
    rule_index,
    max_total_groundings: int,
    *,
    K_MAX: int = 550,
    max_derived_per_state: Optional[int] = None,
    max_states: Optional[int] = None,
    max_groundings_per_rule: Optional[int] = None,
) -> dict:
    """Compute MGU parameters for SLD or RTF resolution.

    Returns dict with: K, S, K_f, max_vars_per_rule, effective_total_G,
    max_fact_pairs_body.
    """
    K_uncapped = K_f * K_r if resolution == "rtf" else K_f + K_r
    K = min(K_uncapped, K_MAX)
    if max_derived_per_state is not None:
        K = int(max_derived_per_state)
    if K_f > K:
        K_f = K

    S = max_states if max_states is not None else K

    if rule_index.rule_lens_sorted.numel() > 0:
        max_vars_per_rule = int(
            rule_index.rule_lens_sorted.max().item()) + 2
    else:
        max_vars_per_rule = 3

    if max_groundings_per_rule is not None:
        effective_total_G = min(
            max_total_groundings,
            rule_index.R_eff * max(max_groundings_per_rule, 1))
    else:
        effective_total_G = max_total_groundings

    return {
        "K": K, "S": S, "K_f": K_f,
        "max_vars_per_rule": max_vars_per_rule,
        "effective_total_G": effective_total_G,
        "max_fact_pairs_body": K_f,
    }
