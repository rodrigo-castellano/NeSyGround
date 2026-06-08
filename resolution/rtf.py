"""RTF resolution — Rule-Then-Fact cascade (K = K_r × K_f).

    resolve_rtf = resolve_rules(goal)  →  resolve_facts(body[0] of each rule child)

Resolves rules first (head unification), then resolves the FIRST body atom of
each rule child against facts. There are no standalone fact children (facts are
reached through the rule body), so the fact slots are empty.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor

from grounder.data.encoding import Encoding
from grounder.execution.capability import Cell, EAGER
from grounder.resolution.mgu import empty_rule_results, resolve_facts, resolve_rules
from grounder.base.types import FlatResolvedChildren, Layout, ResolvedChildren

if TYPE_CHECKING:
    from grounder.nesy.hooks import ResolutionFactHook, ResolutionRuleHook
    from grounder.resolution.api import ResolveRequest


def _rtf_cascade(
    queries: Tensor, remaining: Tensor, goal_valid: Tensor, active_mask: Tensor, *,
    next_var: Tensor, fact_index, facts_idx: Tensor, rule_index, enc: Encoding,
    K_r: int, max_vars_per_rule: int, num_rules: int, max_fact_pairs_body: int,
    rule_hook: Optional["ResolutionRuleHook"],
):
    """Shared RTF L1→L2 cascade (num_rules>0): rule-head unification, then resolve
    each rule child's first body atom against facts → K_rtf=K_r·K_f children.
    Returns (rule_goals, rule_success, sub_rule_idx, rule_subs, gb_l1, K_f, K_rtf);
    dense expands gb_l1 into a grounding body, flat discards it."""
    B, S, _ = queries.shape
    G = remaining.shape[2]
    dev = queries.device
    pad = enc.pad

    with torch.no_grad():
        # L1: rule head unification → K_r children (body at [:Bmax], remaining after)
        rule_goals_l1, gb_l1, rule_success_l1, sub_rule_idx_l1, _, Bmax, rule_subs_l1 = \
            resolve_rules(queries, remaining, rule_index, enc, K_r, max_vars_per_rule,
                          num_rules, goal_valid, active_mask, next_var)

        # L2: extract body[1:] + remaining, resolve first body atom against facts
        n_body_rem = max(Bmax - 1, 0)
        n_avail_rem = G - Bmax
        n_goal_rem = min(G - n_body_rem, n_avail_rem)
        body_rem = torch.full((B, S, K_r, G, 3), pad, dtype=torch.long, device=dev)
        if n_body_rem > 0:
            body_rem[:, :, :, :n_body_rem, :] = rule_goals_l1[:, :, :, 1:Bmax, :]
        if n_goal_rem > 0:
            body_rem[:, :, :, n_body_rem:n_body_rem + n_goal_rem, :] = \
                rule_goals_l1[:, :, :, Bmax:Bmax + n_goal_rem, :]

        N = B * S
        flat_atoms = rule_goals_l1[:, :, :, 0, :].reshape(N, K_r, 3)
        flat_rem = body_rem.reshape(N, K_r, G, 3)
        flat_valid = rule_success_l1.reshape(N, K_r)
        flat_active = torch.ones(N, K_r, dtype=torch.bool, device=dev)

        children, _, success, _ = resolve_facts(
            flat_atoms, flat_rem, fact_index, facts_idx, enc, max_fact_pairs_body,
            flat_valid, flat_active)

        K_f_actual = children.shape[2]
        K_rtf = K_r * K_f_actual
        rule_goals = children.reshape(B, S, K_rtf, G, 3)
        rule_success_out = success.reshape(B, S, K_rtf)
        sub_rule_idx_out = sub_rule_idx_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual).reshape(B, S, K_rtf)
        rule_subs_out = rule_subs_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual, 2, 2).reshape(B, S, K_rtf, 2, 2)

    if rule_hook is not None:
        rule_success_out = rule_hook.filter_rules(rule_goals, rule_success_out, queries)

    return (rule_goals, rule_success_out, sub_rule_idx_out, rule_subs_out,
            gb_l1, K_f_actual, K_rtf)


def resolve_rtf(
    queries: Tensor,           # [B, G, 3]
    remaining: Tensor,         # [B, G, L, 3]
    goal_valid: Tensor,       # [B, G]
    active_mask: Tensor,       # [B, G]
    *,
    next_var: Tensor,  # [B]
    fact_index,
    facts_idx: Tensor,
    rule_index,
    enc: Encoding,
    K_f: int,
    K_r: int,
    max_vars_per_rule: int,
    num_rules: int,
    max_fact_pairs_body: int,
    fact_hook: Optional["ResolutionFactHook"] = None,
    rule_hook: Optional["ResolutionRuleHook"] = None,
) -> ResolvedChildren:
    """RTF: rules first, then facts on each rule child's first body atom."""
    B, S, _ = queries.shape
    G = remaining.shape[2]
    dev = queries.device
    pad = enc.pad

    fact_goals = torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev)
    fact_grounding_body = torch.zeros(B, S, 0, 0, 3, dtype=torch.long, device=dev)
    fact_success = torch.zeros(B, S, 0, dtype=torch.bool, device=dev)
    fact_subs = torch.full((B, S, 0, 2, 2), pad, dtype=torch.long, device=dev)

    if num_rules == 0:
        rg, gb, su, ri, rs = empty_rule_results(B, S, G, 0, pad, dev)
        return ResolvedChildren(Layout.DENSE, fact_goals, fact_grounding_body, fact_success,
                                rg, gb, su, ri, fact_subs, rs)

    rule_goals, rule_success_out, sub_rule_idx_out, rule_subs_out, gb_l1, K_f_actual, K_rtf = \
        _rtf_cascade(
            queries, remaining, goal_valid, active_mask, next_var=next_var,
            fact_index=fact_index, facts_idx=facts_idx, rule_index=rule_index, enc=enc,
            K_r=K_r, max_vars_per_rule=max_vars_per_rule, num_rules=num_rules,
            max_fact_pairs_body=max_fact_pairs_body, rule_hook=rule_hook)
    M_g = gb_l1.shape[3]
    rule_grounding_body_out = gb_l1.unsqueeze(3).expand(
        B, S, K_r, K_f_actual, M_g, 3).reshape(B, S, K_rtf, M_g, 3)
    return ResolvedChildren(Layout.DENSE, fact_goals, fact_grounding_body, fact_success,
                            rule_goals, rule_grounding_body_out, rule_success_out,
                            sub_rule_idx_out, fact_subs, rule_subs_out)


def resolve_rtf_flat(
    queries: Tensor,           # [B, G, 3]
    remaining: Tensor,         # [B, G, L, 3]
    goal_valid: Tensor,       # [B, G]
    active_mask: Tensor,       # [B, G]
    *,
    next_var: Tensor,  # [B]
    fact_index,
    facts_idx: Tensor,
    rule_index,
    enc: Encoding,
    K_f: int,
    K_r: int,
    max_vars_per_rule: int,
    num_rules: int,
    max_fact_pairs_body: int,
    top_rule_idx: Tensor,      # [B, G]
    fact_hook: Optional["ResolutionFactHook"] = None,
    rule_hook: Optional["ResolutionRuleHook"] = None,
) -> FlatResolvedChildren:
    """RTF flat: same L1->L2 cascade as resolve_rtf, flattened (rule children only)."""
    B, S, _ = queries.shape
    G = remaining.shape[2]
    dev = queries.device
    pad = enc.pad

    if num_rules == 0:
        z = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        return FlatResolvedChildren(
            Layout.FLAT, torch.full((0, G, 3), pad, dtype=torch.long, device=dev),
            z(0, 0, 3), z(0), z(0), z(0),
            torch.full((0, 2, 2), pad, dtype=torch.long, device=dev),
            torch.zeros(0, dtype=torch.bool, device=dev), z(0), B, S, False)

    rule_goals, rule_success_out, sub_rule_idx_out, rule_subs_out, _gb, _K_f, K_rtf = \
        _rtf_cascade(
            queries, remaining, goal_valid, active_mask, next_var=next_var,
            fact_index=fact_index, facts_idx=facts_idx, rule_index=rule_index, enc=enc,
            K_r=K_r, max_vars_per_rule=max_vars_per_rule, num_rules=num_rules,
            max_fact_pairs_body=max_fact_pairs_body, rule_hook=rule_hook)

    idx_r = torch.nonzero(rule_success_out.reshape(-1), as_tuple=False).squeeze(1)
    b_r = idx_r // (S * K_rtf)
    s_r = (idx_r // K_rtf) % S
    flat_goals = rule_goals.reshape(B * S * K_rtf, G, 3)[idx_r]
    flat_subs = rule_subs_out.reshape(B * S * K_rtf, 2, 2)[idx_r]
    flat_rule_idx = sub_rule_idx_out.reshape(-1)[idx_r]
    flat_is_fact = torch.zeros(idx_r.numel(), dtype=torch.bool, device=dev)
    flat_top_rule_idx = top_rule_idx[b_r, s_r]
    T = flat_goals.shape[0]
    flat_grounding_body = torch.zeros(T, 0, 3, dtype=torch.long, device=dev)

    return FlatResolvedChildren(
        Layout.FLAT, flat_goals, flat_grounding_body, flat_rule_idx,
        b_r, s_r, flat_subs, flat_is_fact, flat_top_rule_idx, B, S, False)


class RtfResolver:
    """RESOLVERS["rtf"] — wraps resolve_rtf / resolve_rtf_flat (byte-identical dispatch)."""
    name = "rtf"

    def declared_cells(self) -> frozenset:
        return frozenset({Cell(Layout.DENSE, EAGER), Cell(Layout.FLAT, EAGER)})

    def resolve(self, req: "ResolveRequest"):
        plan, fr, kb = req.plan, req.frontier, req.plan.kb
        flat = plan.strategy.layout() is Layout.FLAT
        if flat:
            return resolve_rtf_flat(
                req.queries, req.remaining, req.goal_valid, req.active_mask,
                next_var=fr.next_var,
                fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
                rule_index=kb.rule_index, enc=kb.encoding,
                K_f=kb.K_f, K_r=kb.K_r,
                max_vars_per_rule=plan.max_vars_per_rule, num_rules=kb.num_rules,
                max_fact_pairs_body=plan.max_fact_pairs_body,
                top_rule_idx=fr.top_rule_idx,
                fact_hook=req.fact_hook, rule_hook=req.rule_hook)
        return resolve_rtf(
            req.queries, req.remaining, req.goal_valid, req.active_mask,
            next_var=fr.next_var,
            fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
            rule_index=kb.rule_index, enc=kb.encoding,
            K_f=kb.K_f, K_r=kb.K_r,
            max_vars_per_rule=plan.max_vars_per_rule, num_rules=kb.num_rules,
            max_fact_pairs_body=plan.max_fact_pairs_body,
            fact_hook=req.fact_hook, rule_hook=req.rule_hook)


__all__ = ["resolve_rtf", "resolve_rtf_flat", "RtfResolver"]
