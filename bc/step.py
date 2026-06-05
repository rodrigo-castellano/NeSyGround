"""Per-depth proof step engine for ``BCGrounder``.

One step has four phases:

* **SELECT** — pick the first goal from each proof state.
* **RESOLVE** — dispatch to ``resolve_sld`` / ``resolve_rtf`` /
  ``resolve_enum_step`` over the selected goal.
* **PACK** — flatten S*K children, propagate grounding body, compact
  back to S via ``pack_states`` / ``pack_states_flat``.
* **POSTPROCESS** — prune ground facts, sync the accumulated body
  through ``apply_substitutions``, and (when ``collect_evidence``)
  collect completed groundings.

All phase functions are free functions with ``grounder`` as first
argument so they can be reused by both the eager loop (:func:`step`)
and the outer-compile / inner-compile paths (``compiled_step``).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from grounder.bc.common import (
    collect_groundings,
    compact_atoms,
    pack_states,
    prune_ground_facts,
)
from grounder.filters.search import filter_prune_dead, filter_width
from grounder.resolution.enum import resolve_enum_step
from grounder.resolution.primitives import apply_substitutions
from grounder.resolution.rtf import resolve_rtf
from grounder.resolution.sld import resolve_sld
from grounder.types import (
    FlatResolvedChildren,
    ResolvedChildren,
    SyncParams,
)


def step(
    grounder, states: Dict[str, Tensor], d: int,
) -> Dict[str, Tensor]:
    """One proof step: SELECT → RESOLVE → PACK → POSTPROCESS."""
    # No rules in the KB → nothing to resolve; the step is a no-op and
    # the incoming states pass through unchanged.
    if grounder.kb.num_rules == 0:
        return states

    # Compiled fast path. The flat path is gated off because its
    # ``torch.nonzero`` produces dynamic shapes incompatible with
    # ``mode='reduce-overhead'``. The dense path compiles every
    # depth, including the last — the ``d == depth-1`` branches in
    # ``resolve_enum_step`` / :func:`postprocess` are static at trace
    # time, so dynamo just specialises an additional graph for them.
    # With ``init_state_shape='full'`` every depth shares the same
    # shape, so the compile cache holds **one** graph regardless of
    # depth.
    if grounder._compiled:
        flat_step = getattr(grounder, "_flat_intermediate", False)
        if not flat_step:
            from grounder.bc.compiled_step import step_compiled
            return step_compiled(grounder, states, d)

    # Capture the goal being resolved at this depth (= head atom)
    if grounder.collect_evidence or grounder._collect_rule_groundings:
        states["_selected_goal"] = states["proof_goals"][:, :, 0, :].clone()

    goal_queries, remaining, active_mask = select(grounder, states)

    resolved = resolve(
        grounder,
        goal_queries, remaining,
        states["grounding_body"], states["state_valid"],
        active_mask, states, d,
    )

    resolved = apply_search_filters(grounder, resolved)
    resolved = grounder._apply_hooks(resolved, states)

    # Capture every candidate firing BEFORE pack/prune drops dead
    # children. This is the "considered" rule_groundings semantics
    # that matches keras-ns ``rule2groundings``: rule applications
    # are recorded regardless of whether their proof tree completes
    # within the depth budget. The end-of-BFS ``prune_rule_groundings``
    # filters down to apps whose body atoms are transitively groundable.
    #
    # SKIP when inside torch.compile tracing — the SLD outer-compile
    # path wraps ``_forward_one_batch_inner`` in ``fullgraph=True``,
    # and the Python ``list.append`` in ``capture_step`` breaks the
    # trace and triggers cudagraph partitioning, hurting throughput.
    # Compiled callers fall back to evidence-derived rule_groundings
    # (only completed proof trees), matching the compiled enum-dense
    # path's behaviour. Eager callers get the full "considered" set.
    if (grounder._collect_rule_groundings
            and not torch.compiler.is_compiling()):
        from grounder.bc.considered import capture_step
        capture_step(grounder, resolved, states, d)

    states, sync = pack(grounder, resolved, states)
    states = postprocess(grounder, states, sync, d)
    return states


def select(
    grounder, states: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Extract first goal from each proof state."""
    proof_goals = states["proof_goals"]
    active_mask = proof_goals[:, :, 0, 0] != grounder.kb.padding_idx
    queries = proof_goals[:, :, 0, :]
    queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)
    remaining = proof_goals.clone()
    remaining[:, :, 0, :] = grounder.kb.padding_idx
    return queries, remaining, active_mask


def resolve(
    grounder,
    queries: Tensor,           # [B, S, 3]
    remaining: Tensor,         # [B, S, G, 3]
    grounding_body: Tensor,    # [B, S, M, 3]
    state_valid: Tensor,       # [B, S]
    active_mask: Tensor,       # [B, S]
    states: Dict[str, Tensor],
    d,                                  # int (eager) or 0-dim Tensor (compiled)
    is_last=None,                       # Optional[Tensor] (compiled path)
    use_hooks: bool = True,
) -> ResolvedChildren:
    """Dispatch to resolution strategy. Returns ResolvedChildren.

    ``d`` is a Python int when called from the eager step loop, and
    a 0-dim long tensor when called from the compiled step. The
    ``resolve_enum_step`` downstream accepts both shapes.
    """
    fh = grounder.fact_hook if use_hooks else None
    rh = grounder.rule_hook if use_hooks else None

    if grounder.resolution == "sld":
        return resolve_sld(
            queries, remaining, grounding_body, state_valid, active_mask,
            next_var_indices=states["next_var_indices"],
            fact_index=grounder.kb.fact_index,
            facts_idx=grounder.kb.fact_index.facts_idx,
            rule_index=grounder.kb.rule_index,
            constant_no=grounder.kb.constant_no,
            padding_idx=grounder.kb.padding_idx,
            K_f=grounder.kb.K_f, K_r=grounder.kb.K_r,
            max_vars_per_rule=grounder.max_vars_per_rule,
            num_rules=grounder.kb.num_rules,
            collect_evidence=grounder.collect_evidence,
            excluded_queries=states.get("excluded_queries"),
            fact_hook=fh, rule_hook=rh,
        )
    elif grounder.resolution == "rtf":
        return resolve_rtf(
            queries, remaining, grounding_body, state_valid, active_mask,
            next_var_indices=states["next_var_indices"],
            fact_index=grounder.kb.fact_index,
            facts_idx=grounder.kb.fact_index.facts_idx,
            rule_index=grounder.kb.rule_index,
            constant_no=grounder.kb.constant_no,
            padding_idx=grounder.kb.padding_idx,
            K_f=grounder.kb.K_f, K_r=grounder.kb.K_r, K=grounder.K,
            max_vars_per_rule=grounder.max_vars_per_rule,
            num_rules=grounder.kb.num_rules,
            max_fact_pairs_body=grounder._max_fact_pairs_body,
            collect_evidence=grounder.collect_evidence,
            fact_hook=fh, rule_hook=rh,
        )
    else:
        return resolve_enum_step(
            queries, remaining, grounding_body, state_valid, active_mask,
            fact_index=grounder.kb.fact_index,
            d=d, depth=grounder.depth, width=grounder.width, is_last=is_last,
            M=grounder.kb.M, padding_idx=grounder.kb.padding_idx,
            G_r=grounder.G_r, K=grounder.K,
            any_dual=grounder.any_dual,
            pred_rule_indices=grounder.pred_rule_indices,
            pred_rule_mask=grounder.pred_rule_mask,
            has_free=grounder.has_free,
            body_preds=grounder.body_preds,
            num_body_atoms=grounder.num_body_atoms,
            enum_pred_a=grounder.enum_pred_a,
            enum_bound_binding_a=grounder.enum_bound_binding_a,
            enum_direction_a=grounder.enum_direction_a,
            check_arg_source_a=grounder.check_arg_source_a,
            head_pred_mask=grounder.head_pred_mask,
            has_dual=getattr(grounder, "has_dual", None),
            enum_pred_b=getattr(grounder, "enum_pred_b", None),
            enum_bound_binding_b=getattr(grounder, "enum_bound_binding_b", None),
            enum_direction_b=getattr(grounder, "enum_direction_b", None),
            check_arg_source_b=getattr(grounder, "check_arg_source_b", None),
            collect_evidence=grounder.collect_evidence,
            cartesian_product=grounder._enum_cartesian,
            E=grounder._E,
            w_last_depth=grounder._w_last_depth,
            fv_enum_pred=getattr(grounder, "fv_enum_pred", None),
            fv_enum_bound_src=getattr(grounder, "fv_enum_bound_src", None),
            fv_enum_direction=getattr(grounder, "fv_enum_direction", None),
            fv_enum_valid=getattr(grounder, "fv_enum_valid", None),
            V=grounder.V,
            K_v=grounder.K_v,
            fv_any_valid=grounder._fv_any_valid,
            arg_source_dep=getattr(grounder, "arg_source_dep", None),
            body_preds_dep=getattr(grounder, "body_preds_dep", None),
            flat_intermediate=getattr(grounder, "_flat_intermediate", False),
            dedup_goals=getattr(grounder, "_resolve_skip_enabled", False),
        )


def apply_search_filters(
    grounder, resolved: ResolvedChildren,
) -> ResolvedChildren:
    """Per-step search filters. No gradients, zero overhead when disabled."""
    if not grounder._step_prune_dead and grounder._step_width is None:
        return resolved

    (fg, fgb, fs, rule_goals, rgb, rule_success, sri,
     f_subs, r_subs) = resolved

    if grounder._step_prune_dead:
        rule_success = filter_prune_dead(
            rule_goals, rule_success,
            head_pred_mask=grounder._step_head_pred_mask,
            fact_index=grounder.kb.fact_index,
            constant_no=grounder.kb.constant_no,
            padding_idx=grounder.kb.padding_idx,
            M=grounder.kb.M,
            a0_lens=grounder._step_a0_lens if grounder._step_has_csr else None,
            a1_lens=grounder._step_a1_lens if grounder._step_has_csr else None,
            p_lens=getattr(grounder, '_step_p_lens', None),
            key_scale=grounder._step_key_scale if grounder._step_has_csr else 0,
        )

    if grounder._step_width is not None:
        rule_success = filter_width(
            rule_goals, rule_success,
            fact_index=grounder.kb.fact_index,
            constant_no=grounder.kb.constant_no,
            padding_idx=grounder.kb.padding_idx,
            M=grounder.kb.M,
            width=grounder._step_width,
        )

    return ResolvedChildren(fg, fgb, fs, rule_goals, rgb, rule_success,
                            sri, f_subs, r_subs)


def pack(
    grounder, resolved, states: Dict[str, Tensor],
) -> Tuple[Dict, SyncParams]:
    """Flatten S*K children, propagate grounding body, compact to S.

    Dispatches to pack_states (dense) or pack_states_flat (flat K).
    Returns (states, sync) — no dict pollution with underscore keys.
    """
    if isinstance(resolved, FlatResolvedChildren):
        from grounder.bc.common import pack_states_flat
        packed = pack_states_flat(
            resolved,
            states["top_ridx"], states["grounding_body"],
            states["body_count"],
            grounder.kb.padding_idx,
            collect_evidence=grounder.collect_evidence,
            M_rule=grounder.kb.M,
            dedup=grounder._pack_dedup,
        )
        # The flat resolve emits all-padding substitutions (``flat_subs`` is
        # ``torch.full(..., pad)`` in every branch), so ``winning_subs`` is
        # guaranteed all-pad. ``apply_substitutions`` is then the identity,
        # letting ``sync_accumulated`` skip 3 no-op substitution passes.
        grounder._winning_subs_noop = True
    else:
        grounder._winning_subs_noop = False
        packed = pack_states(
            *resolved,
            states["top_ridx"], states["grounding_body"],
            states["body_count"],
            grounder.S, grounder.kb.padding_idx,
            collect_evidence=grounder.collect_evidence,
            M_rule=grounder.kb.M,
        )

    states["grounding_body"] = packed.grounding_body
    states["proof_goals"] = packed.proof_goals
    states["top_ridx"] = packed.top_ridx
    states["state_valid"] = packed.state_valid

    sync = SyncParams(
        parent_map=packed.parent_map,
        winning_subs=packed.winning_subs,
        has_new_body=packed.has_new_body,
        parent_bcount=packed.body_count,
        current_ridx=packed.current_ridx,
    )

    S_in = packed.proof_goals.shape[1]  # output S (may differ from input)
    states["next_var_indices"] = (
        states["next_var_indices"] + S_in * grounder.max_vars_per_rule)
    return states, sync


def sync_accumulated(
    grounder, states: Dict[str, Tensor], sync: SyncParams, d: int,
) -> Dict[str, Tensor]:
    """Propagate accumulated_body: gather from parents, apply subs, write at depth d.

    Structured layout: accumulated_body is [B, S, D, M, 3].
    Each depth d writes its body atoms to slot ``[:, :, d, :, :]``.

    Args:
        states: Current states dict with accumulated_body and grounding_body.
        sync: SyncParams with parent_map, winning_subs, has_new_body,
              parent_bcount, current_ridx.
        d: Current depth index.
    """
    parent_map = sync.parent_map
    winning_subs = sync.winning_subs
    has_new_body = sync.has_new_body
    parent_bcount = sync.parent_bcount

    if not grounder.collect_evidence:
        states["body_count"] = parent_bcount
        return states

    B, S_out = parent_map.shape
    D_dim = states["accumulated_body"].shape[2]  # D
    M_acc = states["accumulated_body"].shape[3]   # M
    M_work = states["grounding_body"].shape[2]
    pad = grounder.kb.padding_idx
    dev = parent_map.device

    # The flat resolve always emits all-pad ``winning_subs`` (set in
    # ``pack``), making ``apply_substitutions`` the identity. Skipping the
    # three substitution passes below is byte-identical for that path and
    # removes ~30 no-op tensor dispatches per step on the eager hot path.
    subs_noop = getattr(grounder, "_winning_subs_noop", False)

    # a. Gather accumulated_body [B, S_out, D, M, 3] from parents
    pi = parent_map[:, :, None, None, None].expand(-1, -1, D_dim, M_acc, 3)
    acc = states["accumulated_body"].gather(1, pi)

    # b. Gather ridx_per_depth [B, S_out, D] from parents
    rpi = parent_map[:, :, None].expand(-1, -1, D_dim)
    ridx = states["ridx_per_depth"].gather(1, rpi)

    # c. Gather body_count [B, S_out, D] from parents
    bc = states["body_count"].gather(1, rpi)

    # d. Apply substitutions to entire accumulated body
    if subs_noop:
        subs_flat = winning_subs.reshape(B * S_out, 2, 2)
    else:
        acc_flat = acc.reshape(B * S_out, D_dim * M_acc, 3)
        subs_flat = winning_subs.reshape(B * S_out, 2, 2)
        acc_flat = apply_substitutions(acc_flat, subs_flat, pad)
        acc = acc_flat.reshape(B, S_out, D_dim, M_acc, 3)

    # e. Write new body atoms at depth slot d
    new_atoms = states["grounding_body"]  # [B, S_out, M_work, 3]
    if M_work > M_acc:
        write_atoms = new_atoms[:, :, :M_acc, :]
    elif M_work < M_acc:
        write_atoms = torch.full(
            (B, S_out, M_acc, 3), pad, dtype=torch.long, device=dev)
        write_atoms[:, :, :M_work, :] = new_atoms
    else:
        write_atoms = new_atoms
    write_mask = has_new_body[:, :, None, None]  # [B, S_out, 1, 1]
    # ``d`` may be a Python int (eager) or a 0-dim long tensor
    # (compiled). For the compiled path we use a one-hot
    # broadcast-mask along the D dimension so the write becomes a
    # plain ``torch.where`` (no indexed scatter) — that keeps the
    # graph depth-agnostic and shareable across all d values.
    # Remap variant rule indices to original rule indices when
    # ``all_anchors=True`` is active (each rule is split into one
    # variant per body atom; the dedup hash already collapses them
    # in ``unique_groundings_mask``, but ``ridx_per_depth`` would
    # otherwise hold the variant index — out-of-range w.r.t.
    # ``num_rules`` once consumed by ``evidence_to_rule_groundings``).
    # ``_all_anchors`` is a Python bool — specialised at trace time;
    # the remap pass only enters the graph for enum + all_anchors.
    cur_ridx = sync.current_ridx
    if getattr(grounder, "_all_anchors", False):
        v2o = grounder._variant_to_orig_t
        cur_ridx = torch.where(
            cur_ridx >= 0, v2o[cur_ridx.clamp(min=0)], cur_ridx)

    if isinstance(d, torch.Tensor):
        D_acc = acc.shape[2]
        d_arange = torch.arange(D_acc, device=dev)
        is_slot = (d_arange == d).view(1, 1, D_acc, 1, 1)   # [1,1,D,1,1]
        write_atoms_b = write_atoms.unsqueeze(2)             # [B,S,1,M,3]
        write_mask_b = write_mask.unsqueeze(2)               # [B,S,1,1,1]
        acc = torch.where(
            is_slot & write_mask_b, write_atoms_b, acc)
        # Write ridx and bc at slot d via the same trick:
        is_slot_2d = (d_arange == d).view(1, 1, D_acc)
        ridx = torch.where(
            is_slot_2d & has_new_body.unsqueeze(-1),
            cur_ridx.unsqueeze(-1).expand(-1, -1, D_acc),
            ridx,
        )
        new_active = (write_atoms[:, :, :, 0] != pad)
        new_lens = new_active.long().sum(dim=-1)             # [B, S_out]
        bc = torch.where(
            is_slot_2d & has_new_body.unsqueeze(-1),
            new_lens.unsqueeze(-1).expand(-1, -1, D_acc),
            bc,
        )
    else:
        # Eager fast path — Python int slice.
        acc[:, :, d, :, :] = torch.where(write_mask, write_atoms,
                                         acc[:, :, d, :, :])
        ridx[:, :, d] = torch.where(has_new_body, cur_ridx,
                                    ridx[:, :, d])
        new_active = (write_atoms[:, :, :, 0] != pad)
        new_lens = new_active.long().sum(dim=-1)
        bc[:, :, d] = torch.where(has_new_body, new_lens, bc[:, :, d])

    # h. Gather and write head_per_depth at depth d
    hpi = parent_map[:, :, None, None].expand(-1, -1, D_dim, 3)
    head = states["head_per_depth"].gather(1, hpi)
    if not subs_noop:
        head_flat = head.reshape(B * S_out, D_dim, 3)
        head_flat = apply_substitutions(head_flat, subs_flat, pad)
        head = head_flat.reshape(B, S_out, D_dim, 3)
    if "_selected_goal" in states:
        sel = states["_selected_goal"]  # [B, S_in, 3]
        sel_parent = sel.gather(
            1, parent_map.unsqueeze(-1).expand(-1, -1, 3))
        if not subs_noop:
            sel_flat = sel_parent.reshape(B * S_out, 1, 3)
            sel_flat = apply_substitutions(sel_flat, subs_flat, pad)
            sel_parent = sel_flat.reshape(B, S_out, 3)
        if isinstance(d, torch.Tensor):
            d_arange_h = torch.arange(D_dim, device=dev)
            is_slot_h = (d_arange_h == d).view(1, 1, D_dim, 1)
            head = torch.where(
                is_slot_h & has_new_body.view(B, S_out, 1, 1),
                sel_parent.unsqueeze(2),
                head,
            )
        else:
            head[:, :, d, :] = torch.where(
                has_new_body.unsqueeze(-1), sel_parent, head[:, :, d, :])

    states["accumulated_body"] = acc
    states["body_count"] = bc
    states["ridx_per_depth"] = ridx
    states["head_per_depth"] = head
    return states


def postprocess_goals(grounder, states: Dict) -> Dict[str, Tensor]:
    """Optionally prune ground facts, compact atoms, and standardize.

    When ``prune_facts=True``, known ground facts are removed from
    proof_goals between steps (compressed depth semantics).
    When ``prune_facts=False`` (default), only compaction is applied
    (standard SLD semantics where every resolution costs 1 depth).

    When ``collect_evidence=False`` and standardization is configured,
    output variables are standardized (proof_goals are the final output).

    Safe for torch.compile — ``grounder.prune_facts`` is a static Python bool.
    """
    if grounder.prune_facts:
        proof_goals, _, _ = prune_ground_facts(
            states["proof_goals"], states["state_valid"],
            grounder.kb.fact_index.fact_hashes,
            grounder.kb.fact_index.pack_base,
            grounder.kb.constant_no, grounder.kb.padding_idx,
            excluded_queries=states.get("excluded_queries"),
        )
        states["proof_goals"] = compact_atoms(
            proof_goals, grounder.kb.padding_idx)
    else:
        states["proof_goals"] = compact_atoms(
            states["proof_goals"], grounder.kb.padding_idx)

    # Standardize output variables when proof_goals are the final output
    if not grounder.collect_evidence and grounder._standardize_fn is not None:
        counts = states["state_valid"].long().sum(dim=1)
        nv = states.get("initial_next_var", states["next_var_indices"])
        inp = states.get("initial_goals", states["proof_goals"].new_zeros(0))
        std, std_nv = grounder._standardize_fn(
            states["proof_goals"], counts, nv, inp)
        # Clone to detach from CUDA graph output buffers — prevents
        # "overwritten by a subsequent run" errors when these tensors
        # are consumed by the next compiled step.
        states["proof_goals"] = std.clone()
        states["next_var_indices"] = std_nv.clone()

    return states


def collect_groundings_step(grounder, states: Dict) -> Dict[str, Tensor]:
    """Collect completed groundings into output buffer.

    Uses accumulated_body [B, S, D, M, 3] (structured). Called outside
    the compiled step to keep G_body tensors out of the CUDA graph.
    """
    deactivate = (grounder._collect_mode != "grounded")
    cb, cm, cr, sv, c_bc, c_hd = collect_groundings(
        states["accumulated_body"], states["proof_goals"],
        states["state_valid"], states["ridx_per_depth"],
        states["collected_body"], states["collected_mask"],
        states["collected_ridx"],
        grounder.kb.constant_no, grounder.kb.padding_idx, grounder.C,
        body_count=states["body_count"],
        collected_bcount=states["collected_bcount"],
        collect_mode=grounder._collect_mode,
        deactivate=deactivate,
        head_per_depth=states.get("head_per_depth"),
        collected_head=states.get("collected_head"),
        variant_to_orig=getattr(grounder, "_variant_to_orig_t", None),
    )

    states["collected_body"] = cb
    states["collected_mask"] = cm
    states["collected_ridx"] = cr
    states["state_valid"] = sv
    states["collected_bcount"] = c_bc
    if c_hd is not None:
        states["collected_head"] = c_hd
    return states


def postprocess(
    grounder, states: Dict[str, Tensor], sync: SyncParams,
    d, is_last=None,
) -> Dict[str, Tensor]:
    """Full postprocess: prune goals + sync accumulated + collect groundings.

    ``d`` is a Python int when called from the eager step loop,
    and a 0-dim long tensor when called from the compiled step.
    ``is_last`` is None in eager (computed from ``d`` directly) and
    a 0-dim bool tensor in compiled mode.
    """
    states = postprocess_goals(grounder, states)
    states = sync_accumulated(grounder, states, sync, d)
    # Last step + w_last_depth>0: leftover ground unknowns in
    # proof_goals would block terminal collection. The body atoms
    # are already in accumulated_body; clear proof_goals so the
    # rule application is emitted (matches keras-ns
    # prune_incomplete_proofs=False semantics).
    if grounder._w_last_depth is not None and grounder._w_last_depth > 0:
        pad = grounder.kb.padding_idx
        if is_last is not None:
            cleared = torch.full_like(states["proof_goals"], pad)
            states["proof_goals"] = torch.where(
                is_last, cleared, states["proof_goals"])
        elif d == grounder.depth - 1:
            states["proof_goals"] = torch.full_like(
                states["proof_goals"], pad)
    if grounder.collect_evidence:
        states = collect_groundings_step(grounder, states)
    return states


__all__ = [
    "step", "select", "resolve", "apply_search_filters",
    "pack", "sync_accumulated", "postprocess_goals",
    "collect_groundings_step", "postprocess",
]
