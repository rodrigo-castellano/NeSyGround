"""SLD resolution — fact ∥ rule children (K = K_f + K_r).

    resolve_sld = resolve_facts  ∥  resolve_rules

Facts and rules are resolved independently and returned side by side (dense
layout). The engine reconstructs the substituted body from the accumulated body
via the sync pass, so MGU need not track the per-state working body here.
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


def resolve_sld(
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
    excluded_queries: Optional[Tensor] = None,
    fact_hook: Optional["ResolutionFactHook"] = None,
    rule_hook: Optional["ResolutionRuleHook"] = None,
) -> ResolvedChildren:
    """SLD: resolve facts and rules in parallel → dense ResolvedChildren."""
    B, S, _ = queries.shape
    G = remaining.shape[2]
    dev = queries.device
    pad = enc.pad

    with torch.no_grad():
        fact_goals, fact_gbody, fact_success, fact_subs = resolve_facts(
            queries, remaining, fact_index, facts_idx, enc, K_f,
            state_valid, active_mask, excluded_queries=excluded_queries)
    if fact_hook is not None:
        fact_success = fact_hook.filter_facts(fact_goals, fact_success, queries)

    if num_rules == 0:
        rule_goals, rule_gbody, rule_success, sub_rule_idx, rule_subs = \
            empty_rule_results(B, S, G, 0, pad, dev)
    else:
        with torch.no_grad():
            rule_goals, rule_gbody, rule_success, sub_rule_idx, _, _, rule_subs = resolve_rules(
                queries, remaining, rule_index, enc, K_r,
                max_vars_per_rule, num_rules, state_valid, active_mask,
                next_var_indices)
        if rule_hook is not None:
            rule_success = rule_hook.filter_rules(rule_goals, rule_success, queries)

    return ResolvedChildren(
        Layout.DENSE, fact_goals, fact_gbody, fact_success,
        rule_goals, rule_gbody, rule_success, sub_rule_idx, fact_subs, rule_subs)


def resolve_sld_flat(
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
    top_rule_idx: Tensor,      # [B, S]
    body_count: Tensor,        # [B, S, D] or [B, S]
    excluded_queries: Optional[Tensor] = None,
    fact_hook: Optional["ResolutionFactHook"] = None,
    rule_hook: Optional["ResolutionRuleHook"] = None,
    collect_evidence: bool = True,
) -> FlatResolvedChildren:
    """SLD flat: same facts/rules MGU as resolve_sld, flattened facts-then-rules."""
    B, S, _ = queries.shape
    G = remaining.shape[2]
    dev = queries.device
    pad = enc.pad

    with torch.no_grad():
        fact_goals, _fact_gbody, fact_success, fact_subs = resolve_facts(
            queries, remaining, fact_index, facts_idx, enc, K_f,
            state_valid, active_mask, excluded_queries=excluded_queries)
    if fact_hook is not None:
        fact_success = fact_hook.filter_facts(fact_goals, fact_success, queries)

    if num_rules == 0:
        rule_goals, _rule_gbody, rule_success, sub_rule_idx, rule_subs = \
            empty_rule_results(B, S, G, 0, pad, dev)
    else:
        with torch.no_grad():
            rule_goals, _rule_gbody, rule_success, sub_rule_idx, _, _, rule_subs = resolve_rules(
                queries, remaining, rule_index, enc, K_r,
                max_vars_per_rule, num_rules, state_valid, active_mask,
                next_var_indices)
        if rule_hook is not None:
            rule_success = rule_hook.filter_rules(rule_goals, rule_success, queries)

    K_f_a = fact_goals.shape[2]
    K_r_a = rule_goals.shape[2]

    # ── FACT children: AND the dense skip_fact guard before flatten ──
    facts_valid = fact_success                                   # [B, S, K_f]
    if collect_evidence:
        uninit = (body_count.sum(-1) == 0) if body_count.dim() == 3 else (body_count == 0)
        skip = uninit & ~(top_rule_idx == -1)                    # [B, S]
        facts_valid = facts_valid & ~skip.unsqueeze(-1)
    idx_f = torch.nonzero(facts_valid.reshape(-1), as_tuple=False).squeeze(1)
    b_f = idx_f // (S * K_f_a)
    s_f = (idx_f // K_f_a) % S
    flat_goals_f = fact_goals.reshape(B * S * K_f_a, G, 3)[idx_f]
    flat_subs_f = fact_subs.reshape(B * S * K_f_a, 2, 2)[idx_f]
    flat_ridx_f = torch.full((idx_f.numel(),), -1, dtype=torch.long, device=dev)
    is_fact_f = torch.ones(idx_f.numel(), dtype=torch.bool, device=dev)
    top_f = top_rule_idx[b_f, s_f]

    # ── RULE children ──
    idx_r = torch.nonzero(rule_success.reshape(-1), as_tuple=False).squeeze(1)
    b_r = idx_r // (S * K_r_a)
    s_r = (idx_r // K_r_a) % S
    flat_goals_r = rule_goals.reshape(B * S * K_r_a, G, 3)[idx_r]
    flat_subs_r = rule_subs.reshape(B * S * K_r_a, 2, 2)[idx_r]
    flat_ridx_r = sub_rule_idx.reshape(-1)[idx_r]
    is_fact_r = torch.zeros(idx_r.numel(), dtype=torch.bool, device=dev)
    top_r = top_rule_idx[b_r, s_r]

    # ── concat FACTS then RULES (facts-first, matching dense) ──
    flat_goals = torch.cat([flat_goals_f, flat_goals_r], dim=0)
    flat_subs = torch.cat([flat_subs_f, flat_subs_r], dim=0)
    flat_rule_idx = torch.cat([flat_ridx_f, flat_ridx_r], dim=0)
    flat_is_fact = torch.cat([is_fact_f, is_fact_r], dim=0)
    flat_top_ridx = torch.cat([top_f, top_r], dim=0)
    flat_batch_idx = torch.cat([b_f, b_r], dim=0)
    flat_state_idx = torch.cat([s_f, s_r], dim=0)
    T = flat_goals.shape[0]
    flat_grounding_body = torch.zeros(T, 0, 3, dtype=torch.long, device=dev)

    return FlatResolvedChildren(
        Layout.FLAT, flat_goals, flat_grounding_body, flat_rule_idx,
        flat_batch_idx, flat_state_idx, flat_subs, flat_is_fact, flat_top_ridx,
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
                req.queries, req.remaining, req.state_valid, req.active_mask,
                next_var_indices=fr.next_var,
                fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
                rule_index=kb.rule_index, enc=kb.encoding,
                K_f=kb.K_f, K_r=kb.K_r, max_vars_per_rule=plan.max_vars_per_rule,
                num_rules=kb.num_rules, top_rule_idx=fr.top_rule_idx,
                body_count=fr.body_count, excluded_queries=req.excluded_queries,
                fact_hook=req.fact_hook, rule_hook=req.rule_hook,
                collect_evidence=plan.collect_evidence)
        return resolve_sld(
            req.queries, req.remaining, req.state_valid, req.active_mask,
            next_var_indices=fr.next_var,
            fact_index=kb.fact_index, facts_idx=kb.fact_index.facts_idx,
            rule_index=kb.rule_index, enc=kb.encoding,
            K_f=kb.K_f, K_r=kb.K_r, max_vars_per_rule=plan.max_vars_per_rule,
            num_rules=kb.num_rules, excluded_queries=req.excluded_queries,
            fact_hook=req.fact_hook, rule_hook=req.rule_hook)


__all__ = ["resolve_sld", "resolve_sld_flat", "SldResolver"]
