"""Per-depth proof step — SELECT → RESOLVE → PACK → POSTPROCESS(+sync).

Ported from OLD ``bc/step.py`` (eager path only — the fingerprint runs CPU/eager).
RESOLVE dispatches via the ``RESOLVERS`` registry keyed on ``plan.resolution``
(sld/rtf/pbc) — a pure lookup returning the same tuples the old if/elif did. The
considered ``capture_step`` runs AFTER resolve and BEFORE pack. Working state is a
frozen ``Frontier`` (+ private ``_Collected``); each phase returns a functional update.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder._build.backward.buffers import _Collected
from grounder._build.backward.considered import capture_step
from grounder._build.backward.pack import compact_atoms, pack_states, pack_states_flat
from grounder._build.backward.postprocess import collect_groundings
from grounder._build.backward.sync import sync_accumulated
from grounder._build.filters.prune_facts import prune_ground_facts
from grounder._build.grounder.registry import RESOLVERS
from grounder._build.resolution.api import ResolveRequest
from grounder._build.backward.state import Frontier
from grounder._build.types import FlatResolvedChildren


def capture_selected_goal(plan, fr: Frontier) -> Frontier:
    """Snapshot the goal resolved at this depth (= head atom) for per-depth heads."""
    if plan.collect_evidence or plan.collect_rule_groundings:
        return fr.replace(selected_goal=fr.proof_goals[:, :, 0, :].clone())
    return fr


def step(plan, fr: Frontier, coll: Optional[_Collected], run, dsel,
         excluded_queries: Optional[Tensor] = None):
    """One proof step: SELECT → RESOLVE → PACK → POSTPROCESS."""
    fr = capture_selected_goal(plan, fr)

    goal_queries, remaining, active_mask = select(plan, fr)

    resolved = resolve(plan, goal_queries, remaining, fr, active_mask, dsel,
                       excluded_queries)

    if plan.collect_rule_groundings:
        run = capture_step(plan, resolved, fr, run, run.chunk_query_offset)

    fr, sync = pack(plan, resolved, fr)
    fr, coll = postprocess(plan, fr, coll, sync, dsel, excluded_queries)
    return fr, coll, run


def select(plan, fr: Frontier) -> Tuple[Tensor, Tensor, Tensor]:
    """Extract first goal from each proof state."""
    proof_goals = fr.proof_goals
    active_mask = proof_goals[:, :, 0, 0] != plan.kb.padding_idx
    queries = proof_goals[:, :, 0, :]
    queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)
    remaining = proof_goals.clone()
    remaining[:, :, 0, :] = plan.kb.padding_idx
    return queries, remaining, active_mask


def resolve(plan, queries, remaining, fr: Frontier, active_mask, dsel,
            excluded_queries):
    """Dispatch via the Resolver registry → ResolvedChildren / FlatResolvedChildren.

    Pure lookup: RESOLVERS[plan.resolution].resolve(req) returns the IDENTICAL
    tuples the old if/elif did. Hooks ride the request (inert for pbc)."""
    req = ResolveRequest(
        plan=plan, queries=queries, remaining=remaining,
        state_valid=fr.state_valid, active_mask=active_mask, frontier=fr,
        depth_selector=dsel, excluded_queries=excluded_queries,
        fact_hook=plan.fact_hook, rule_hook=plan.rule_hook)
    return RESOLVERS[plan.resolution].resolve(req)


def pack(plan, resolved, fr: Frontier) -> Tuple[Frontier, dict]:
    """Flatten S*K children, compact to S; return (Frontier, sync dict)."""
    if isinstance(resolved, FlatResolvedChildren):
        subs_noop = resolved.subs_noop
        packed = pack_states_flat(
            resolved, fr.top_rule_idx, fr.grounding_body,
            fr.body_count, plan.kb.padding_idx,
            collect_evidence=plan.collect_evidence, M_rule=plan.kb.M,
            dedup=(plan.pack_dedup and subs_noop), subs_noop=subs_noop,
            S_cap=(None if subs_noop else plan.S))
    else:
        subs_noop = False
        packed = pack_states(
            resolved, fr.top_rule_idx, fr.grounding_body,
            fr.body_count, plan.S, plan.kb.padding_idx,
            collect_evidence=plan.collect_evidence, M_rule=plan.kb.M)

    # next_var advances by the FIXED dense width on the sld/rtf flat path so the
    # unbound-var labels match dense (enum-flat keeps the dynamic S_out advance).
    S_adv = plan.S if (isinstance(resolved, FlatResolvedChildren)
                       and not subs_noop) else packed.proof_goals.shape[1]
    fr = fr.replace(
        grounding_body=packed.grounding_body,
        proof_goals=packed.proof_goals,
        top_rule_idx=packed.top_ridx,
        state_valid=packed.state_valid,
        next_var=fr.next_var + S_adv * plan.max_vars_per_rule)

    sync = {
        "parent_map": packed.parent_map,
        "winning_subs": packed.winning_subs,
        "has_new_body": packed.has_new_body,
        "parent_bcount": packed.body_count,
        "current_ridx": packed.current_ridx,
        "subs_noop": subs_noop,
    }
    return fr, sync


def postprocess_goals(plan, fr: Frontier, excluded_queries) -> Frontier:
    """Optionally prune ground facts, then compact atoms."""
    if plan.prune_facts:
        proof_goals, _, _ = prune_ground_facts(
            fr.proof_goals, fr.state_valid,
            plan.kb.fact_index.fact_hashes, plan.kb.fact_index.pack_base,
            plan.kb.constant_no, plan.kb.padding_idx,
            excluded_queries=excluded_queries)
        return fr.replace(proof_goals=compact_atoms(proof_goals, plan.kb.padding_idx))
    return fr.replace(proof_goals=compact_atoms(fr.proof_goals, plan.kb.padding_idx))


def collect_groundings_step(plan, fr: Frontier, coll: _Collected) -> Tuple[Frontier, _Collected]:
    """Collect completed groundings into output buffer (coll non-None here)."""
    deactivate = (plan.collect_mode != "grounded")
    cb, cm, cr, sv, c_bc, c_hd = collect_groundings(
        fr.accumulated_body, fr.proof_goals, fr.state_valid,
        fr.ridx_per_depth, coll.collected_body, coll.collected_mask,
        coll.collected_ridx, plan.kb.constant_no, plan.kb.padding_idx,
        plan.C, body_count=fr.body_count,
        collected_bcount=coll.collected_bcount, collect_mode=plan.collect_mode,
        deactivate=deactivate, head_per_depth=fr.head_per_depth,
        collected_head=coll.collected_head,
        variant_to_orig=plan.variant_to_orig)
    fr = fr.replace(state_valid=sv)
    coll = coll._replace(
        collected_body=cb, collected_mask=cm, collected_ridx=cr,
        collected_bcount=c_bc,
        collected_head=(c_hd if c_hd is not None else coll.collected_head))
    return fr, coll


def postprocess(plan, fr: Frontier, coll: Optional[_Collected], sync, dsel,
                excluded_queries) -> Tuple[Frontier, Optional[_Collected]]:
    """Prune goals + sync accumulated + (last-step clear) + collect groundings."""
    fr = postprocess_goals(plan, fr, excluded_queries)
    fr = sync_accumulated(plan, fr, sync, dsel)
    if plan.w_last_depth is not None and plan.w_last_depth > 0:
        if dsel.is_last:
            fr = fr.replace(proof_goals=torch.full_like(fr.proof_goals,
                                                        plan.kb.padding_idx))
    if plan.collect_evidence:
        fr, coll = collect_groundings_step(plan, fr, coll)
    return fr, coll


__all__ = ["step", "select", "resolve", "pack", "postprocess"]
