"""SLD resolution — fact ∥ rule children (K = K_f + K_r).

    resolve_sld = resolve_facts  ∥  resolve_rules

Facts and rules are resolved independently and returned side by side (dense
layout). The engine reconstructs the substituted body from the accumulated body
via the sync pass, so MGU need not track the per-goal working body here.
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


def _resolve_facts_and_rules(
    queries: Tensor, remaining: Tensor, goal_valid: Tensor, active_mask: Tensor, *,
    next_var: Tensor, fact_index, facts_idx: Tensor, rule_index, enc: Encoding,
    K_f: int, K_r: int, max_vars_per_rule: int, num_rules: int,
    excluded_queries: Optional[Tensor],
    fact_hook: Optional["ResolutionFactHook"],
    rule_hook: Optional["ResolutionRuleHook"],
):
    """Shared SLD prelude: resolve facts ∥ rules (+ hooks); dense keeps the
    grounding-body tensors, flat discards them. 9-tuple of fact/rule results."""
    B, S, _ = queries.shape
    G = remaining.shape[2]
    dev = queries.device
    pad = enc.pad

    with torch.no_grad():
        fact_goals, fact_grounding_body, fact_success, fact_subs = resolve_facts(
            queries, remaining, fact_index, facts_idx, enc, K_f,
            goal_valid, active_mask, excluded_queries=excluded_queries)
    if fact_hook is not None:
        fact_success = fact_hook.filter_facts(fact_goals, fact_success, queries)

    if num_rules == 0:
        rule_goals, rule_grounding_body, rule_success, sub_rule_idx, rule_subs = \
            empty_rule_results(B, S, G, 0, pad, dev)
    else:
        with torch.no_grad():
            rule_goals, rule_grounding_body, rule_success, sub_rule_idx, _, _, rule_subs = resolve_rules(
                queries, remaining, rule_index, enc, K_r,
                max_vars_per_rule, num_rules, goal_valid, active_mask, next_var)
        if rule_hook is not None:
            rule_success = rule_hook.filter_rules(rule_goals, rule_success, queries)

    return (fact_goals, fact_grounding_body, fact_success, fact_subs,
            rule_goals, rule_grounding_body, rule_success, sub_rule_idx, rule_subs)


def resolve_sld(
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
    excluded_queries: Optional[Tensor] = None,
    fact_hook: Optional["ResolutionFactHook"] = None,
    rule_hook: Optional["ResolutionRuleHook"] = None,
) -> ResolvedChildren:
    """SLD: resolve facts and rules in parallel → dense ResolvedChildren."""
    (fact_goals, fact_grounding_body, fact_success, fact_subs,
     rule_goals, rule_grounding_body, rule_success, sub_rule_idx, rule_subs) = \
        _resolve_facts_and_rules(
            queries, remaining, goal_valid, active_mask, next_var=next_var,
            fact_index=fact_index, facts_idx=facts_idx, rule_index=rule_index,
            enc=enc, K_f=K_f, K_r=K_r, max_vars_per_rule=max_vars_per_rule,
            num_rules=num_rules, excluded_queries=excluded_queries,
            fact_hook=fact_hook, rule_hook=rule_hook)
    return ResolvedChildren(
        Layout.DENSE, fact_goals, fact_grounding_body, fact_success,
        rule_goals, rule_grounding_body, rule_success, sub_rule_idx, fact_subs, rule_subs)


def resolve_sld_flat(
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
    top_rule_idx: Tensor,      # [B, G]
    body_count: Tensor,        # [B, G, D] or [B, G]
    excluded_queries: Optional[Tensor] = None,
    fact_hook: Optional["ResolutionFactHook"] = None,
    rule_hook: Optional["ResolutionRuleHook"] = None,
    collect_evidence: bool = True,
) -> FlatResolvedChildren:
    """SLD flat: same facts/rules MGU as resolve_sld, flattened facts-then-rules."""
    (fact_goals, _fgb, fact_success, fact_subs,
     rule_goals, _rgb, rule_success, sub_rule_idx, rule_subs) = \
        _resolve_facts_and_rules(
            queries, remaining, goal_valid, active_mask, next_var=next_var,
            fact_index=fact_index, facts_idx=facts_idx, rule_index=rule_index,
            enc=enc, K_f=K_f, K_r=K_r, max_vars_per_rule=max_vars_per_rule,
            num_rules=num_rules, excluded_queries=excluded_queries,
            fact_hook=fact_hook, rule_hook=rule_hook)

    B, S, _ = queries.shape
    G = remaining.shape[2]
    dev = queries.device

    K_f_a = fact_goals.shape[2]
    K_r_a = rule_goals.shape[2]

    # ── FACT children: AND the dense skip_fact guard before flatten ──
    facts_valid = fact_success                                   # [B, G, K_f]
    if collect_evidence:
        uninit = (body_count.sum(-1) == 0) if body_count.dim() == 3 else (body_count == 0)
        skip = uninit & ~(top_rule_idx == -1)                    # [B, G]
        facts_valid = facts_valid & ~skip.unsqueeze(-1)
    idx_f = torch.nonzero(facts_valid.reshape(-1), as_tuple=False).squeeze(1)
    b_f = idx_f // (S * K_f_a)
    s_f = (idx_f // K_f_a) % S
    flat_goals_f = fact_goals.reshape(B * S * K_f_a, G, 3)[idx_f]
    flat_subs_f = fact_subs.reshape(B * S * K_f_a, 2, 2)[idx_f]
    flat_rule_idx_f = torch.full((idx_f.numel(),), -1, dtype=torch.long, device=dev)
    is_fact_f = torch.ones(idx_f.numel(), dtype=torch.bool, device=dev)
    top_f = top_rule_idx[b_f, s_f]

    # ── RULE children ──
    idx_r = torch.nonzero(rule_success.reshape(-1), as_tuple=False).squeeze(1)
    b_r = idx_r // (S * K_r_a)
    s_r = (idx_r // K_r_a) % S
    flat_goals_r = rule_goals.reshape(B * S * K_r_a, G, 3)[idx_r]
    flat_subs_r = rule_subs.reshape(B * S * K_r_a, 2, 2)[idx_r]
    flat_rule_idx_r = sub_rule_idx.reshape(-1)[idx_r]
    is_fact_r = torch.zeros(idx_r.numel(), dtype=torch.bool, device=dev)
    top_r = top_rule_idx[b_r, s_r]

    # ── concat FACTS then RULES (facts-first, matching dense) ──
    flat_goals = torch.cat([flat_goals_f, flat_goals_r], dim=0)
    flat_subs = torch.cat([flat_subs_f, flat_subs_r], dim=0)
    flat_rule_idx = torch.cat([flat_rule_idx_f, flat_rule_idx_r], dim=0)
    flat_is_fact = torch.cat([is_fact_f, is_fact_r], dim=0)
    flat_top_rule_idx = torch.cat([top_f, top_r], dim=0)
    flat_batch_idx = torch.cat([b_f, b_r], dim=0)
    flat_state_idx = torch.cat([s_f, s_r], dim=0)
    T = flat_goals.shape[0]
    flat_grounding_body = torch.zeros(T, 0, 3, dtype=torch.long, device=dev)

    return FlatResolvedChildren(
        Layout.FLAT, flat_goals, flat_grounding_body, flat_rule_idx,
        flat_batch_idx, flat_state_idx, flat_subs, flat_is_fact, flat_top_rule_idx,
        B, S, False)


class SldResolver:
    """RESOLVERS["sld"] — wraps resolve_sld / resolve_sld_flat (byte-identical dispatch)."""
    name = "sld"

    def declared_cells(self) -> frozenset:
        return frozenset({Cell(Layout.DENSE, EAGER), Cell(Layout.FLAT, EAGER)})

    def resolve(self, req: "ResolveRequest"):
        plan, fr, kb = req.plan, req.frontier, req.plan.kb
        flat = plan.strategy.layout() is Layout.FLAT
        if flat:
            return resolve_sld_flat(
                req.queries, req.remaining, req.goal_valid, req.active_mask,
                next_var=fr.next_var,
                fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
                rule_index=kb.rule_index, enc=kb.encoding,
                K_f=kb.K_f, K_r=kb.K_r, max_vars_per_rule=plan.max_vars_per_rule,
                num_rules=kb.num_rules, top_rule_idx=fr.top_rule_idx,
                body_count=fr.body_count, excluded_queries=req.excluded_queries,
                fact_hook=req.fact_hook, rule_hook=req.rule_hook,
                collect_evidence=plan.collect_evidence)
        return resolve_sld(
            req.queries, req.remaining, req.goal_valid, req.active_mask,
            next_var=fr.next_var,
            fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
            rule_index=kb.rule_index, enc=kb.encoding,
            K_f=kb.K_f, K_r=kb.K_r, max_vars_per_rule=plan.max_vars_per_rule,
            num_rules=kb.num_rules, excluded_queries=req.excluded_queries,
            fact_hook=req.fact_hook, rule_hook=req.rule_hook)


__all__ = ["resolve_sld", "resolve_sld_flat", "SldResolver"]
