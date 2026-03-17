"""RTF resolution — Rule-Then-Fact cascade (K = K_r × K_f).

    resolve_rtf = resolve_rules(goal) → resolve_facts(body[0])

Resolves rules first (head unification), then resolves the first body
atom of each rule child against facts.
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from grounder.resolution.mgu import resolve_facts as mgu_resolve_facts
from grounder.resolution.mgu import resolve_rules as mgu_resolve_rules

if TYPE_CHECKING:
    from grounder.nesy.hooks import ResolutionFactHook, ResolutionRuleHook


def resolve_rtf(
    queries: Tensor,           # [B, S, 3]
    remaining: Tensor,         # [B, S, G, 3]
    grounding_body: Tensor,    # [B, S, M, 3]
    state_valid: Tensor,       # [B, S]
    active_mask: Tensor,       # [B, S]
    *,
    next_var_indices: Tensor,
    fact_index,
    facts_idx: Tensor,
    rule_index,
    constant_no: int,
    padding_idx: int,
    K_f: int,
    K_r: int,
    K: int,
    max_vars_per_rule: int,
    num_rules: int,
    max_fact_pairs_body: int,
    track_grounding_body: bool = True,
    fact_hook: Optional[ResolutionFactHook] = None,
    rule_hook: Optional[ResolutionRuleHook] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor]:
    """RTF resolution: rules first, then facts on first body atom.

    Returns 9-tensor tuple (same format as resolve_sld):
        fact_goals   [B, S, 0, G, 3]     (empty — no standalone facts)
        fact_gbody   [B, S, 0, M, 3]
        fact_success [B, S, 0]
        rule_goals   [B, S, K_rtf, G, 3]
        rule_gbody   [B, S, K_rtf, M, 3]
        rule_success [B, S, K_rtf]
        sub_rule_idx [B, S, K_rtf]
        fact_subs    [B, S, 0, 2, 2]     (empty — no standalone facts)
        rule_subs    [B, S, K_rtf, 2, 2]
    """
    B, S, _ = queries.shape
    G = remaining.shape[2]
    M_g = grounding_body.shape[2]
    dev = queries.device
    pad = padding_idx

    # Empty fact results (RTF resolves facts through rule body)
    fact_goals = torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev)
    fact_gbody = torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev)
    fact_success = torch.zeros(B, S, 0, dtype=torch.bool, device=dev)
    fact_subs = torch.full((B, S, 0, 2, 2), pad, dtype=torch.long, device=dev)

    if num_rules == 0:
        return (
            fact_goals, fact_gbody, fact_success,
            torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
            torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            fact_subs,
            torch.full((B, S, 0, 2, 2), pad, dtype=torch.long, device=dev),
        )

    # All MGU operations are pure index ops — no gradients needed
    with torch.no_grad():
        # Step 1: Rule head unification → K_r children
        # Pass grounding_body=None: accumulated body sync is handled separately
        rule_body_subst, rule_remaining, rule_gbody_l1, rule_success_l1, \
            sub_rule_idx_l1, _, Bmax, rule_subs_l1 = mgu_resolve_rules(
                queries, remaining, rule_index,
                constant_no, padding_idx, K_r,
                max_vars_per_rule, num_rules,
                state_valid, active_mask, next_var_indices,
                grounding_body=None)

        # Step 2: Resolve first body atom against facts → K_f children per rule
        n_body_rem = max(Bmax - 1, 0)
        n_goal_rem = min(G - n_body_rem, G)
        body_rem = torch.full(
            (B, S, K_r, G, 3), pad, dtype=torch.long, device=dev)
        if n_body_rem > 0:
            body_rem[:, :, :, :n_body_rem, :] = rule_body_subst[:, :, :, 1:Bmax, :]
        if n_goal_rem > 0:
            body_rem[:, :, :, n_body_rem:n_body_rem + n_goal_rem, :] = \
                rule_remaining[:, :, :, :n_goal_rem, :]

        # Flatten [B, S] → [N] for fact resolution
        N = B * S
        flat_atoms = rule_body_subst[:, :, :, 0, :].reshape(N, K_r, 3)
        flat_rem = body_rem.reshape(N, K_r, G, 3)
        flat_valid = rule_success_l1.reshape(N, K_r)
        flat_active = torch.ones(N, K_r, dtype=torch.bool, device=dev)

        children, _, success, _ = mgu_resolve_facts(
            flat_atoms, flat_rem, fact_index, facts_idx,
            constant_no, padding_idx, max_fact_pairs_body,
            flat_valid, flat_active, grounding_body=None)

        K_f_actual = children.shape[2]
        K_rtf = K_r * K_f_actual

        # Reshape to [B, S, K_rtf, ...]
        rule_goals = children.reshape(B, S, K_rtf, G, 3)
        rule_success_out = success.reshape(B, S, K_rtf)

        # Propagate gbody and ridx from level-1 rule resolution
        # Use actual M from resolution output (0 when track_grounding_body=False)
        M_g_actual = rule_gbody_l1.shape[3]
        rule_gbody_out = rule_gbody_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual, M_g_actual, 3).reshape(B, S, K_rtf, M_g_actual, 3)
        sub_ridx_out = sub_rule_idx_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual).reshape(B, S, K_rtf)

        # Propagate rule_subs from level-1 (rule head unification subs)
        # Expand to K_rtf: each rule child's K_f_actual fact children share the same subs
        rule_subs_out = rule_subs_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual, 2, 2).reshape(B, S, K_rtf, 2, 2)

    # Hook: may contain learned parameters — outside no_grad
    if rule_hook is not None:
        rule_success_out = rule_hook.filter_rules(
            rule_goals, rule_success_out, queries)

    return (fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody_out, rule_success_out, sub_ridx_out,
            fact_subs, rule_subs_out)
