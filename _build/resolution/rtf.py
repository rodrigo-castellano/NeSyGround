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

from grounder._build.data.encoding import Encoding
from grounder._build.execution.capability import Cell, EAGER
from grounder._build.resolution.mgu import empty_rule_results, resolve_facts, resolve_rules
from grounder._build.types import FlatResolvedChildren, Layout, ResolvedChildren

if TYPE_CHECKING:
    from grounder._build.nesy.hooks import ResolutionFactHook, ResolutionRuleHook
    from grounder._build.resolution.api import ResolveRequest


def resolve_rtf(
    queries: Tensor,           # [B, S, 3]
    remaining: Tensor,         # [B, S, G, 3]
    state_valid: Tensor,       # [B, S]
    active_mask: Tensor,       # [B, S]
    *,
    next_var_indices: Tensor,  # [B]
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
    fact_gbody = torch.zeros(B, S, 0, 0, 3, dtype=torch.long, device=dev)
    fact_success = torch.zeros(B, S, 0, dtype=torch.bool, device=dev)
    fact_subs = torch.full((B, S, 0, 2, 2), pad, dtype=torch.long, device=dev)

    if num_rules == 0:
        rg, gb, su, ri, rs = empty_rule_results(B, S, G, 0, pad, dev)
        return ResolvedChildren(Layout.DENSE, fact_goals, fact_gbody, fact_success,
                                rg, gb, su, ri, fact_subs, rs)

    with torch.no_grad():
        # L1: rule head unification → K_r children (body at [:Bmax], remaining after)
        rule_goals_l1, rule_gbody_l1, rule_success_l1, sub_rule_idx_l1, _, Bmax, rule_subs_l1 = \
            resolve_rules(queries, remaining, rule_index, enc, K_r, max_vars_per_rule,
                          num_rules, state_valid, active_mask, next_var_indices)

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
        M_g = rule_gbody_l1.shape[3]
        rule_gbody_out = rule_gbody_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual, M_g, 3).reshape(B, S, K_rtf, M_g, 3)
        sub_ridx_out = sub_rule_idx_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual).reshape(B, S, K_rtf)
        rule_subs_out = rule_subs_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual, 2, 2).reshape(B, S, K_rtf, 2, 2)

    if rule_hook is not None:
        rule_success_out = rule_hook.filter_rules(rule_goals, rule_success_out, queries)

    return ResolvedChildren(Layout.DENSE, fact_goals, fact_gbody, fact_success,
                            rule_goals, rule_gbody_out, rule_success_out,
                            sub_ridx_out, fact_subs, rule_subs_out)


def resolve_rtf_flat(
    queries: Tensor,           # [B, S, 3]
    remaining: Tensor,         # [B, S, G, 3]
    state_valid: Tensor,       # [B, S]
    active_mask: Tensor,       # [B, S]
    *,
    next_var_indices: Tensor,  # [B]
    fact_index,
    facts_idx: Tensor,
    rule_index,
    enc: Encoding,
    K_f: int,
    K_r: int,
    max_vars_per_rule: int,
    num_rules: int,
    max_fact_pairs_body: int,
    top_rule_idx: Tensor,      # [B, S]
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

    with torch.no_grad():
        rule_goals_l1, _gb_l1, rule_success_l1, sub_rule_idx_l1, _, Bmax, rule_subs_l1 = \
            resolve_rules(queries, remaining, rule_index, enc, K_r, max_vars_per_rule,
                          num_rules, state_valid, active_mask, next_var_indices)

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
        sub_ridx_out = sub_rule_idx_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual).reshape(B, S, K_rtf)
        rule_subs_out = rule_subs_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_actual, 2, 2).reshape(B, S, K_rtf, 2, 2)

    if rule_hook is not None:
        rule_success_out = rule_hook.filter_rules(rule_goals, rule_success_out, queries)

    idx_r = torch.nonzero(rule_success_out.reshape(-1), as_tuple=False).squeeze(1)
    b_r = idx_r // (S * K_rtf)
    s_r = (idx_r // K_rtf) % S
    flat_goals = rule_goals.reshape(B * S * K_rtf, G, 3)[idx_r]
    flat_subs = rule_subs_out.reshape(B * S * K_rtf, 2, 2)[idx_r]
    flat_rule_idx = sub_ridx_out.reshape(-1)[idx_r]
    flat_is_fact = torch.zeros(idx_r.numel(), dtype=torch.bool, device=dev)
    flat_top_ridx = top_rule_idx[b_r, s_r]
    T = flat_goals.shape[0]
    flat_grounding_body = torch.zeros(T, 0, 3, dtype=torch.long, device=dev)

    return FlatResolvedChildren(
        Layout.FLAT, flat_goals, flat_grounding_body, flat_rule_idx,
        b_r, s_r, flat_subs, flat_is_fact, flat_top_ridx, B, S, False)


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
                req.queries, req.remaining, req.state_valid, req.active_mask,
                next_var_indices=fr.next_var,
                fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
                rule_index=kb.rule_index, enc=kb.encoding,
                K_f=kb.K_f, K_r=kb.K_r,
                max_vars_per_rule=plan.max_vars_per_rule, num_rules=kb.num_rules,
                max_fact_pairs_body=plan.max_fact_pairs_body,
                top_rule_idx=fr.top_rule_idx,
                fact_hook=req.fact_hook, rule_hook=req.rule_hook)
        return resolve_rtf(
            req.queries, req.remaining, req.state_valid, req.active_mask,
            next_var_indices=fr.next_var,
            fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
            rule_index=kb.rule_index, enc=kb.encoding,
            K_f=kb.K_f, K_r=kb.K_r,
            max_vars_per_rule=plan.max_vars_per_rule, num_rules=kb.num_rules,
            max_fact_pairs_body=plan.max_fact_pairs_body,
            fact_hook=req.fact_hook, rule_hook=req.rule_hook)


__all__ = ["resolve_rtf", "resolve_rtf_flat", "RtfResolver"]
