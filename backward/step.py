"""Per-depth proof step — SELECT → RESOLVE → PACK → POSTPROCESS(+sync).

Eager path (the fingerprint runs CPU/eager). RESOLVE dispatches via the
``RESOLVERS`` registry keyed on ``plan.resolution`` (sld/rtf/pbc) — a pure lookup.
The ``capture_step`` accumulator runs AFTER resolve and BEFORE pack. Working state
is a frozen ``Frontier`` (+ private ``_Collected``); each phase returns a
functional update.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.backward.buffers import _Collected
from grounder.backward.considered import capture_step
from grounder.backward.pack import compact_atoms, pack_states, pack_states_flat
from grounder.backward.postprocess import collect_groundings
from grounder.backward.sync import sync_accumulated
from grounder.filters.prune_facts import prune_ground_facts
from grounder.api.registry import RESOLVERS
from grounder.resolution.api import ResolveRequest
from grounder.backward.state import Frontier
from grounder.base.types import FlatResolvedChildren


def capture_selected_atom(plan, fr: Frontier) -> Frontier:
    """Snapshot the atom resolved at this depth (= head atom) for per-depth heads."""
    if plan.collect_evidence or plan.collect_rule_groundings:
        return fr.replace(selected_atom=fr.goal_atoms[:, :, 0, :].clone())
    return fr


def step(plan, fr: Frontier, coll: Optional[_Collected], run, dsel,
         excluded_queries: Optional[Tensor] = None):
    """One proof step: SELECT → RESOLVE → PACK → POSTPROCESS."""
    fr = capture_selected_atom(plan, fr)

    goal_queries, remaining, active_mask = select(plan, fr)

    resolved = resolve(plan, goal_queries, remaining, fr, active_mask, dsel,
                       excluded_queries)

    if plan.collect_rule_groundings:
        run = capture_step(plan, resolved, fr, run, run.chunk_query_offset)

    fr, sync = pack(plan, resolved, fr)
    fr, coll = postprocess(plan, fr, coll, sync, dsel, excluded_queries)
    return fr, coll, run


def select(plan, fr: Frontier) -> Tuple[Tensor, Tensor, Tensor]:
    """Extract first atom from each goal."""
    goal_atoms = fr.goal_atoms
    active_mask = goal_atoms[:, :, 0, 0] != plan.kb.padding_idx
    queries = goal_atoms[:, :, 0, :]
    queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)
    remaining = goal_atoms.clone()
    remaining[:, :, 0, :] = plan.kb.padding_idx
    return queries, remaining, active_mask


def resolve(plan, queries, remaining, fr: Frontier, active_mask, dsel,
            excluded_queries):
    """Dispatch via the Resolver registry → ResolvedChildren / FlatResolvedChildren.

    Pure lookup: RESOLVERS[plan.resolution].resolve(req) returns the resolved
    children tuples. Hooks ride the request (inert for pbc)."""
    req = ResolveRequest(
        plan=plan, queries=queries, remaining=remaining,
        goal_valid=fr.goal_valid, active_mask=active_mask, frontier=fr,
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
    # unbound-var labels match dense (enum-flat keeps the dynamic G_out advance).
    S_adv = plan.S if (isinstance(resolved, FlatResolvedChildren)
                       and not subs_noop) else packed.goal_atoms.shape[1]
    fr = fr.replace(
        grounding_body=packed.grounding_body,
        goal_atoms=packed.goal_atoms,
        top_rule_idx=packed.top_rule_idx,
        goal_valid=packed.goal_valid,
        next_var=fr.next_var + S_adv * plan.max_vars_per_rule)

    sync = {
        "parent_map": packed.parent_map,
        "winning_subs": packed.winning_subs,
        "has_new_body": packed.has_new_body,
        "parent_body_count": packed.body_count,
        "current_rule_idx": packed.current_rule_idx,
        "subs_noop": subs_noop,
    }
    return fr, sync


def postprocess_goals(plan, fr: Frontier, excluded_queries) -> Frontier:
    """Optionally prune ground facts, then compact atoms."""
    if plan.prune_facts:
        goal_atoms = prune_ground_facts(
            fr.goal_atoms,
            plan.kb.fact_index.fact_hashes, plan.kb.fact_index.pack_base,
            plan.kb.constant_no, plan.kb.padding_idx,
            excluded_queries=excluded_queries)
        return fr.replace(goal_atoms=compact_atoms(goal_atoms, plan.kb.padding_idx))
    return fr.replace(goal_atoms=compact_atoms(fr.goal_atoms, plan.kb.padding_idx))


def collect_groundings_step(plan, fr: Frontier, coll: _Collected) -> Tuple[Frontier, _Collected]:
    """Collect completed groundings into output buffer (coll non-None here)."""
    deactivate = (plan.collect_mode != "grounded")
    cb, cm, cr, sv, c_bc, c_hd = collect_groundings(
        fr.accumulated_body, fr.goal_atoms, fr.goal_valid,
        fr.rule_idx_per_depth, coll.collected_body, coll.collected_mask,
        coll.collected_rule_idx, plan.kb.constant_no, plan.kb.padding_idx,
        plan.Y_q, body_count=fr.body_count,
        collected_body_count=coll.collected_body_count, collect_mode=plan.collect_mode,
        deactivate=deactivate, head_per_depth=fr.head_per_depth,
        collected_head=coll.collected_head,
        variant_to_orig=plan.variant_to_orig)
    fr = fr.replace(goal_valid=sv)
    coll = coll._replace(
        collected_body=cb, collected_mask=cm, collected_rule_idx=cr,
        collected_body_count=c_bc,
        collected_head=(c_hd if c_hd is not None else coll.collected_head))
    return fr, coll


def postprocess(plan, fr: Frontier, coll: Optional[_Collected], sync, dsel,
                excluded_queries) -> Tuple[Frontier, Optional[_Collected]]:
    """Prune goals + sync accumulated + (last-step clear) + collect groundings."""
    fr = postprocess_goals(plan, fr, excluded_queries)
    fr = sync_accumulated(plan, fr, sync, dsel)
    if plan.w_last_depth is not None and plan.w_last_depth > 0 and dsel.is_last:
        fr = fr.replace(goal_atoms=torch.full_like(fr.goal_atoms, plan.kb.padding_idx))
    if plan.collect_evidence:
        fr, coll = collect_groundings_step(plan, fr, coll)
    return fr, coll


__all__ = ["step", "select", "resolve", "pack", "postprocess"]
