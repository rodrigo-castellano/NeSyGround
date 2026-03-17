"""SLD resolution — parallel fact + rule resolution (K = K_f + K_r).

    resolve_sld = resolve_facts ∥ resolve_rules

Public API:
  resolve_sld():  core resolution (fact + rule children)
  init_mgu():     compute MGU parameters (K, S, etc.) for SLD or RTF
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from grounder.resolution.mgu import resolve_facts as mgu_resolve_facts
from grounder.resolution.mgu import resolve_rules as mgu_resolve_rules

if TYPE_CHECKING:
    from grounder.nesy.hooks import ResolutionFactHook, ResolutionRuleHook


def resolve_sld(
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
    max_vars_per_rule: int,
    num_rules: int,
    track_grounding_body: bool = True,
    excluded_queries: Optional[Tensor] = None,
    fact_hook: Optional[ResolutionFactHook] = None,
    rule_hook: Optional[ResolutionRuleHook] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor]:
    """SLD resolution: resolve facts and rules in parallel.

    Returns 9-tensor tuple:
        fact_goals   [B, S, K_f, G, 3]
        fact_gbody   [B, S, K_f, M, 3]
        fact_success [B, S, K_f]
        rule_goals   [B, S, K_r, G, 3]
        rule_gbody   [B, S, K_r, M, 3]
        rule_success [B, S, K_r]
        sub_rule_idx [B, S, K_r]
        fact_subs    [B, S, K_f, 2, 2]
        rule_subs    [B, S, K_r, 2, 2]
    """
    B, S, _ = queries.shape
    G = remaining.shape[2]
    M_g = grounding_body.shape[2]
    dev = queries.device
    pad = padding_idx

    # Fact resolution (pure index ops — no gradients needed)
    # Pass grounding_body=None: accumulated body sync is handled separately
    # in bc.py._sync_accumulated, so resolution doesn't need to substitute it.
    with torch.no_grad():
        fact_goals, fact_gbody, fact_success, fact_subs = mgu_resolve_facts(
            queries, remaining, fact_index, facts_idx,
            constant_no, padding_idx, K_f,
            state_valid, active_mask,
            grounding_body=None,
            excluded_queries=excluded_queries)

    # Hook: may contain learned parameters (KGE scorer, etc.)
    if fact_hook is not None:
        fact_success = fact_hook.filter_facts(fact_goals, fact_success, queries)

    # Rule resolution (pure index ops — no gradients needed)
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

    # Pass grounding_body=None: accumulated body sync is handled separately
    with torch.no_grad():
        rule_body_subst, rule_remaining, rule_gbody_out, rule_success, \
            sub_rule_idx, _, Bmax, rule_subs = mgu_resolve_rules(
                queries, remaining, rule_index,
                constant_no, padding_idx, K_r,
                max_vars_per_rule, num_rules,
                state_valid, active_mask, next_var_indices,
                grounding_body=None)

        # Assemble goals = body + remaining
        rule_goals = torch.full(
            (B, S, K_r, G, 3), pad, dtype=torch.long, device=dev)
        rule_goals[:, :, :, :Bmax, :] = rule_body_subst
        n_rem = min(G - Bmax, G)
        if n_rem > 0:
            rule_goals[:, :, :, Bmax:Bmax + n_rem, :] = \
                rule_remaining[:, :, :, :n_rem, :]

    # Hook: may contain learned parameters
    if rule_hook is not None:
        rule_success = rule_hook.filter_rules(rule_goals, rule_success, queries)

    return (fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody_out, rule_success, sub_rule_idx,
            fact_subs, rule_subs)
