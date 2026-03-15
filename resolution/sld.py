"""SLD resolution — parallel fact + rule resolution (K = K_f + K_r).

    resolve_sld = resolve_facts ∥ resolve_rules

Public API:
  resolve_sld():  core resolution (fact + rule children)
  init_mgu():     compute MGU parameters (K, S, etc.) for SLD or RTF
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.resolution.mgu import resolve_facts as mgu_resolve_facts
from grounder.resolution.mgu import resolve_rules as mgu_resolve_rules


@torch.no_grad()
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """SLD resolution: resolve facts and rules in parallel.

    Returns 7-tensor tuple:
        fact_goals   [B, S, K_f, G, 3]
        fact_gbody   [B, S, K_f, M, 3]
        fact_success [B, S, K_f]
        rule_goals   [B, S, K_r, G, 3]
        rule_gbody   [B, S, K_r, M, 3]
        rule_success [B, S, K_r]
        sub_rule_idx [B, S, K_r]
    """
    B, S, _ = queries.shape
    G = remaining.shape[2]
    M_g = grounding_body.shape[2]
    dev = queries.device
    pad = padding_idx

    # Fact resolution
    fact_goals, fact_gbody, fact_success = mgu_resolve_facts(
        queries, remaining, fact_index, facts_idx,
        constant_no, padding_idx, K_f,
        state_valid, active_mask,
        grounding_body if track_grounding_body else None)

    # Rule resolution
    if num_rules == 0:
        return (
            fact_goals, fact_gbody, fact_success,
            torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
            torch.zeros(B, S, 0, dtype=torch.long, device=dev),
        )

    rule_body_subst, rule_remaining, rule_gbody_out, rule_success, \
        sub_rule_idx, _, Bmax = mgu_resolve_rules(
            queries, remaining, rule_index,
            constant_no, padding_idx, K_r,
            max_vars_per_rule, num_rules,
            state_valid, active_mask, next_var_indices,
            grounding_body if track_grounding_body else None)

    # Assemble goals = body + remaining
    rule_goals = torch.full(
        (B, S, K_r, G, 3), pad, dtype=torch.long, device=dev)
    rule_goals[:, :, :, :Bmax, :] = rule_body_subst
    n_rem = min(G - Bmax, G)
    if n_rem > 0:
        rule_goals[:, :, :, Bmax:Bmax + n_rem, :] = \
            rule_remaining[:, :, :, :n_rem, :]

    return (fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody_out, rule_success, sub_rule_idx)


