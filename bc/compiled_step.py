"""Compiled per-step path for the multi-step (non-outer-compile) regime.

Used when ``BCGrounder._compiled = True`` — i.e. enum-dense with
``compile_mode`` set on CUDA at depth > 1. SLD/RTF with
``mode='reduce-overhead'`` use the outer-compile path instead
(``_compiled_inner`` wraps ``_forward_one_batch_inner`` directly), so
this module is unused for those configurations.

The three functions mirror the eager step path from ``bc.step``:

* :func:`fn_step_for_depth` — lazy ``torch.compile`` of ``step_impl``
  with a tiny per-shape cache.
* :func:`step_compiled` — dict-of-tensors → raw-tensor adapter; clones
  every output to detach from the CUDA-graph private pool so subsequent
  replays don't overwrite still-referenced tensors.
* :func:`step_impl` — the raw-tensor body (SELECT → RESOLVE →
  PACK → POSTPROCESS). Lives outside the BCGrounder class but takes
  ``grounder`` as first arg for access to ``self.*`` state.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


def fn_step_for_depth(grounder, d: int):
    """Get the (cached, depth-agnostic) compiled step function.

    Static-shape compile: ``dynamic=False``. With
    ``init_state_shape='full'`` every depth has the same input shape,
    and ``d`` enters as a 0-dim tensor (not a Python int) so dynamo
    doesn't specialise the graph on ``d``'s value. One graph covers
    every depth.

    With ``init_state_shape='minimal'`` (DpRL-friendly), d=0 has
    ``S_in=1`` while d≥1 has ``S_in=max_states``; those need separate
    graphs, so the cache holds at most 2 entries.
    """
    if grounder._init_state_shape == "full" or d > 0:
        key = "main"
    else:
        key = "init"
    if key not in grounder._fn_steps_by_depth:
        from grounder.bc.compile import Compiler
        grounder._fn_steps_by_depth[key] = Compiler.wrap(
            lambda *args: step_impl(grounder, *args),
            mode=grounder.compile_mode, fullgraph=True, dynamic=False,
            bump_recompile_limit=True,
        )
    return grounder._fn_steps_by_depth[key]


def step_compiled(grounder, states: Dict, d: int = 0) -> Dict[str, Tensor]:
    """Compiled step: dict <-> raw tensors.

    ``d`` is converted to a 0-dim long tensor and ``is_last`` to a
    0-dim bool tensor before entering the compiled region — this keeps
    the graph structure depth-agnostic, so a single compile replays
    for every depth.
    """
    if grounder._clone_between_steps:
        states = {k: v.clone() if isinstance(v, Tensor) else v
                  for k, v in states.items()}

    fn = fn_step_for_depth(grounder, d)
    dev = states["proof_goals"].device
    d_t = torch.tensor(d, dtype=torch.long, device=dev)
    is_last_t = torch.tensor(
        d == grounder.depth - 1, dtype=torch.bool, device=dev)

    (gb, ab, bc, rpd, hpd, pg, tr, sv, nvi,
     cb, cm, cr, cbc, chh) = fn(
        states["grounding_body"], states["accumulated_body"],
        states["body_count"], states["ridx_per_depth"],
        states["head_per_depth"],
        states["proof_goals"],
        states["top_ridx"], states["state_valid"],
        states["next_var_indices"],
        states["collected_body"], states["collected_mask"],
        states["collected_ridx"],
        states["collected_bcount"],
        states["collected_head"],
        d_t, is_last_t,
    )
    # Clone every output to detach from the CUDA-graph-managed private
    # pool. Without this, the next ``fn`` replay overwrites the buffers
    # while the outer Python still holds references in the ``states``
    # dict, raising "accessing tensor output of CUDAGraphs that has
    # been overwritten by a subsequent run".
    states["grounding_body"]   = gb.clone()
    states["accumulated_body"] = ab.clone()
    states["body_count"]       = bc.clone()
    states["ridx_per_depth"]   = rpd.clone()
    states["head_per_depth"]   = hpd.clone()
    states["proof_goals"]      = pg.clone()
    states["top_ridx"]         = tr.clone()
    states["state_valid"]      = sv.clone()
    states["next_var_indices"] = nvi.clone()
    states["collected_body"]   = cb.clone()
    states["collected_mask"]   = cm.clone()
    states["collected_ridx"]   = cr.clone()
    states["collected_bcount"] = cbc.clone()
    states["collected_head"]   = chh.clone()
    return states


def step_impl(
    grounder,
    grounding_body: Tensor,
    accumulated_body: Tensor,
    body_count: Tensor,
    ridx_per_depth: Tensor,
    head_per_depth: Tensor,
    proof_goals: Tensor,
    top_ridx: Tensor,
    state_valid: Tensor,
    next_var_indices: Tensor,
    collected_body: Tensor,
    collected_mask: Tensor,
    collected_ridx: Tensor,
    collected_bcount: Tensor,
    collected_head: Tensor,
    d_t: Tensor,
    is_last_t: Tensor,
) -> Tuple[Tensor, ...]:
    """Raw-tensor SELECT → RESOLVE → PACK → POSTPROCESS step.

    Returns the 14 state tensors. Same phases as the eager
    ``BCGrounder.step``; differences are pure plumbing (dict <-> raw
    tensors and tensor ``d`` / ``is_last`` so dynamo can share one
    graph across depths).
    """
    states = {
        "grounding_body": grounding_body,
        "accumulated_body": accumulated_body,
        "body_count": body_count,
        "ridx_per_depth": ridx_per_depth,
        "head_per_depth": head_per_depth,
        "proof_goals": proof_goals,
        "top_ridx": top_ridx,
        "state_valid": state_valid,
        "next_var_indices": next_var_indices,
        "collected_body": collected_body,
        "collected_mask": collected_mask,
        "collected_ridx": collected_ridx,
        "collected_bcount": collected_bcount,
        "collected_head": collected_head,
    }
    # Capture the goal being resolved at this depth (= head atom).
    # Same conditional as eager ``step``: ``sync_accumulated`` reads
    # ``states["_selected_goal"]`` to write per-depth heads into
    # ``head_per_depth``. Skipping this caused all rule firings to
    # share the padding head atom, collapsing the rule-path output
    # to a constant pool and zeroing the gradient signal for SBR /
    # DCR / R2N (val MRR stuck at 1/N from epoch 1).
    if grounder.collect_evidence or grounder._collect_rule_groundings:
        states["_selected_goal"] = proof_goals[:, :, 0, :].clone()
    from grounder.bc.step import (
        select, resolve, apply_search_filters, pack, postprocess,
    )
    queries, remaining, active_mask = select(grounder, states)
    resolved = resolve(
        grounder, queries, remaining, grounding_body, state_valid,
        active_mask, states, d=d_t, is_last=is_last_t, use_hooks=False,
    )
    resolved = apply_search_filters(grounder, resolved)
    resolved = grounder._apply_hooks(resolved, states)
    # NOTE: ``bc.considered.capture_step`` is NOT called here — the
    # compiled inner step is fullgraph-traced and Python-side list
    # appends break the trace. The eager enum-flat path captures
    # per-step. enum-dense + compile_mode currently surfaces only
    # evidence-derived rule_groundings; consumers needing the keras-
    # equivalent "considered" count should use enum-flat.
    states, sync = pack(grounder, resolved, states)
    states = postprocess(grounder, states, sync, d_t, is_last_t)
    return (states["grounding_body"], states["accumulated_body"],
            states["body_count"], states["ridx_per_depth"],
            states["head_per_depth"], states["proof_goals"],
            states["top_ridx"], states["state_valid"],
            states["next_var_indices"], states["collected_body"],
            states["collected_mask"], states["collected_ridx"],
            states["collected_bcount"], states["collected_head"])


__all__ = ["fn_step_for_depth", "step_compiled", "step_impl"]
