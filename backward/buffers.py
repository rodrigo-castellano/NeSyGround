"""Per-call working-state allocation for the backward proof loop.

Ported byte-identically from the OLD ``bc/buffers.py`` ``init_states``. Builds the
frozen ``Frontier`` working tape plus a private ``_Collected`` trees buffer once at
the top of each forward (same tensors, same shapes as the old dict).
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from grounder.backward.state import Frontier


class _Collected(NamedTuple):
    """Private engine seam: completed-grounding output buffer (mirrors ``_Packed``)."""
    collected_body: Tensor      # [B, C, D, M, 3]
    collected_mask: Tensor      # [B, C]
    collected_ridx: Tensor      # [B, C, D]
    collected_bcount: Tensor    # [B, C, D]
    collected_head: Tensor      # [B, C, D, 3]


def init_frontier(
    plan,
    queries: Tensor,
    query_mask: Tensor,
    *,
    initial_goals: Optional[Tensor] = None,
    next_var: Optional[Tensor] = None,
    excluded_queries: Optional[Tensor] = None,
) -> Tuple[Frontier, Optional[_Collected]]:
    """Build initial Frontier + (optional) _Collected for the proof loop.

    ``_Collected`` is allocated ONLY when the trees tier is requested
    (``plan.output_spec.trees``); the acc D-structure stub predicate is VERBATIM."""
    B = queries.size(0)
    dev = queries.device
    pad = plan.kb.padding_idx
    G = plan.max_goals
    C = plan.C
    D = plan.depth
    M = plan.kb.M
    M_work = M

    S_init = 1 if plan.init_state_shape == "minimal" else plan.S

    proof_goals = torch.full((B, S_init, G, 3), pad, dtype=torch.long, device=dev)
    if initial_goals is not None:
        M_in = initial_goals.shape[1]
        proof_goals[:, 0, :M_in, :] = initial_goals
    else:
        proof_goals[:, 0, 0, :] = queries
    grounding_body = torch.full((B, S_init, M_work, 3), pad, dtype=torch.long, device=dev)

    skip_acc = (not plan.collect_evidence
                and plan.filter_mode != "fp_global")
    acc_D = 1 if skip_acc else D
    acc_M = 1 if skip_acc else M
    accumulated_body = torch.full(
        (B, S_init, acc_D, acc_M, 3), pad, dtype=torch.long, device=dev)
    body_count = torch.zeros(B, S_init, acc_D, dtype=torch.long, device=dev)
    ridx_per_depth = torch.full((B, S_init, acc_D), -1, dtype=torch.long, device=dev)
    head_per_depth = torch.full((B, S_init, acc_D, 3), pad, dtype=torch.long, device=dev)
    top_rule_idx = torch.full((B, S_init), -1, dtype=torch.long, device=dev)
    if S_init == 1:
        state_valid = query_mask.unsqueeze(1)
    else:
        state_valid = torch.zeros(B, S_init, dtype=torch.bool, device=dev)
        state_valid[:, 0] = query_mask

    if next_var is None:
        E = plan.kb.constant_no + 1
        next_var = torch.full((B,), E, dtype=torch.long, device=dev)

    fr = Frontier(
        proof_goals=proof_goals,
        grounding_body=grounding_body,
        state_valid=state_valid,
        top_rule_idx=top_rule_idx,
        accumulated_body=accumulated_body,
        ridx_per_depth=ridx_per_depth,
        head_per_depth=head_per_depth,
        body_count=body_count,
        next_var=next_var,
        selected_goal=None,
    )
    coll = None
    if plan.output_spec.trees:
        coll = _Collected(
            collected_body=queries.new_zeros(B, C, acc_D, acc_M, 3),
            collected_mask=torch.zeros(B, C, dtype=torch.bool, device=dev),
            collected_ridx=queries.new_full((B, C, acc_D), -1, dtype=torch.long),
            collected_bcount=torch.zeros(B, C, acc_D, dtype=torch.long, device=dev),
            collected_head=torch.full((B, C, acc_D, 3), pad, dtype=torch.long, device=dev),
        )
    return fr, coll


__all__ = ["init_frontier", "_Collected"]
