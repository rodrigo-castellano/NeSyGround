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
from grounder.backward.considered import extract_rows
from grounder.backward.pack import compact_atoms, pack_states, pack_states_flat
from grounder.backward.postprocess import collect_groundings
from grounder.backward.sync import sync_accumulated
from grounder.filters.prune_facts import prune_ground_facts
from grounder.api.registry import RESOLVERS
from grounder.resolution.api import ResolveRequest
from grounder.backward.state import Frontier
from grounder.base.types import FlatResolvedChildren


def capture_selected_atom(plan, fr: Frontier) -> Frontier:
    """Snapshot the atom resolved at this depth (= head atom) for per-depth heads.

    A view, not a clone: goal_atoms is never mutated in place (select writes
    only its own clone on sld/rtf; pack/postprocess build fresh tensors) and
    every consumer (extract_rows, sync_accumulated) copies via gather/indexing."""
    if plan.collect_evidence or plan.collect_rule_groundings:
        return fr.replace(selected_atom=fr.goal_atoms[:, :, 0, :])
    return fr


def step_core(plan, fr: Frontier, coll: Optional[_Collected], dsel,
              excluded_queries: Optional[Tensor] = None,
              skip_tail: bool = False):
    """Tensor-pure step body: SELECT → RESOLVE → (rows) → PACK → POSTPROCESS.

    This is the COMPILE UNIT (``strategy.wrap_step`` wraps it): every op is a
    tensor op, including the considered-rows extraction (fixed-shape sentinel
    rows on the dense path). The Python-side ``RunState`` append happens in
    the eager rim (``backward.loop``), outside any compiled graph.

    ``skip_tail`` (a Python constant decided by the loop, never a tensor):
    on the FINAL depth of a FIRINGS-only request the packed/postprocessed
    frontier feeds nothing — no next step, no GoalState, no TREES collection —
    so pack+postprocess are skipped. Firings are captured at resolve time,
    before pack, so the rule_groundings output is byte-identical.
    Returns ``(fr, coll, rows|None)``.
    """
    fr = capture_selected_atom(plan, fr)

    goal_queries, remaining, active_mask = select(plan, fr)

    resolved = resolve(plan, goal_queries, remaining, fr, active_mask, dsel,
                       excluded_queries)

    rows = (extract_rows(plan, resolved, fr)
            if plan.collect_rule_groundings else None)

    if skip_tail:
        return fr, coll, rows

    fr, sync = pack(plan, resolved, fr)
    fr, coll = postprocess(plan, fr, coll, sync, dsel, excluded_queries)
    return fr, coll, rows


def step(plan, fr: Frontier, coll: Optional[_Collected], run, dsel,
         excluded_queries: Optional[Tensor] = None,
         skip_tail: bool = False):
    """One proof step (eager composition): core + the RunState firing append."""
    fr, coll, rows = step_core(plan, fr, coll, dsel, excluded_queries, skip_tail)
    if rows is not None:
        from dataclasses import replace as _replace
        from grounder.backward.state import FiringSet
        emission = FiringSet.from_emission(
            rows[0], rows[1], rows[2], rows[3] + run.chunk_query_offset)
        run = _replace(run, firings=run.firings.extend(emission))
    return fr, coll, run


def select(plan, fr: Frontier) -> Tuple[Tensor, Tensor, Tensor]:
    """Extract first atom from each goal."""
    goal_atoms = fr.goal_atoms
    active_mask = goal_atoms[:, :, 0, 0] != plan.kb.padding_idx
    queries = goal_atoms[:, :, 0, :]
    queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)
    if plan.pbc is not None:
        # A PbcPlan is carried exactly by the pbc/join resolver family, whose
        # materializers read only remaining[..., 1:, :] (+ its shape): slot 0
        # is dead there, so skip the full-frontier clone + pad write.
        # (sld/rtf substitute over the whole tensor incl. the padded slot 0.)
        return queries, goal_atoms, active_mask
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
            collect_evidence=plan.collect_evidence, M_rule=plan.kb.M,
            parent_goals=fr.goal_atoms)

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
    """Optionally prune ground facts, then compact atoms (fused: the prune
    returns its keep mask and compact scatters straight from the original
    goals — no intermediate hole-punched tensor)."""
    if plan.prune_facts:
        keep = prune_ground_facts(
            fr.goal_atoms,
            plan.kb.fact_index.fact_hashes, plan.kb.fact_index.pack_base,
            plan.kb.constant_no, plan.kb.padding_idx,
            excluded_queries=excluded_queries, return_keep=True)
        return fr.replace(goal_atoms=compact_atoms(
            fr.goal_atoms, plan.kb.padding_idx, valid=keep))
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
