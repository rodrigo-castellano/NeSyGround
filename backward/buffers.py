"""Per-call working-state allocation for the backward proof loop.

Builds the frozen ``Frontier`` working tape plus a private ``_Collected`` trees
buffer once at the top of each forward.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from grounder.backward.state import Frontier


class _Collected(NamedTuple):
    """Private engine seam: completed-grounding output buffer (mirrors ``_Packed``)."""
    collected_body: Tensor      # [B, Y_q, D, M, 3]
    collected_mask: Tensor      # [B, Y_q]
    collected_rule_idx: Tensor      # [B, Y_q, D]
    collected_body_count: Tensor    # [B, Y_q, D]
    collected_head: Tensor      # [B, Y_q, D, 3]


def init_frontier(
    plan,
    queries: Tensor,
    query_mask: Tensor,
    *,
    initial_goals: Optional[Tensor] = None,
    next_var: Optional[Tensor] = None,
) -> Tuple[Frontier, Optional[_Collected]]:
    """Build initial Frontier + (optional) _Collected for the proof loop.

    ``_Collected`` is allocated ONLY when the trees tier is requested
    (``plan.output_spec.trees``); the acc D-structure stub predicate is VERBATIM."""
    B = queries.size(0)
    dev = queries.device
    pad = plan.kb.padding_idx
    G = plan.max_goals
    Y_q = plan.Y_q
    D = plan.depth
    M = plan.kb.M
    M_work = M

    S_init = 1 if plan.init_state_shape == "minimal" else plan.S

    goal_atoms = torch.full((B, S_init, G, 3), pad, dtype=torch.long, device=dev)
    if initial_goals is not None:
        M_in = initial_goals.shape[1]
        goal_atoms[:, 0, :M_in, :] = initial_goals
    else:
        goal_atoms[:, 0, 0, :] = queries
    grounding_body = torch.full((B, S_init, M_work, 3), pad, dtype=torch.long, device=dev)

    skip_acc = (not plan.collect_evidence
                and plan.filter_mode != "fp_global")
    acc_D = 1 if skip_acc else D
    acc_M = 1 if skip_acc else M
    accumulated_body = torch.full(
        (B, S_init, acc_D, acc_M, 3), pad, dtype=torch.long, device=dev)
    body_count = torch.zeros(B, S_init, acc_D, dtype=torch.long, device=dev)
    rule_idx_per_depth = torch.full((B, S_init, acc_D), -1, dtype=torch.long, device=dev)
    head_per_depth = torch.full((B, S_init, acc_D, 3), pad, dtype=torch.long, device=dev)
    top_rule_idx = torch.full((B, S_init), -1, dtype=torch.long, device=dev)
    if S_init == 1:
        goal_valid = query_mask.unsqueeze(1)
    else:
        goal_valid = torch.zeros(B, S_init, dtype=torch.bool, device=dev)
        goal_valid[:, 0] = query_mask

    if next_var is None:
        E = plan.kb.constant_no + 1
        next_var = torch.full((B,), E, dtype=torch.long, device=dev)

    fr = Frontier(
        goal_atoms=goal_atoms,
        grounding_body=grounding_body,
        goal_valid=goal_valid,
        top_rule_idx=top_rule_idx,
        accumulated_body=accumulated_body,
        rule_idx_per_depth=rule_idx_per_depth,
        head_per_depth=head_per_depth,
        body_count=body_count,
        next_var=next_var,
        selected_atom=None,
        # Terminal standardize (build_proof_state) renames output vars relative to
        # the passed-in next_var / goals — captured here, untouched by step.
        initial_next_var=next_var,
        initial_goals=(initial_goals if initial_goals is not None
                       else goal_atoms.new_zeros(B, 0, 3)),
    )
    coll = None
    if plan.output_spec.trees:
        coll = _Collected(
            collected_body=queries.new_zeros(B, Y_q, acc_D, acc_M, 3),
            collected_mask=torch.zeros(B, Y_q, dtype=torch.bool, device=dev),
            collected_rule_idx=queries.new_full((B, Y_q, acc_D), -1, dtype=torch.long),
            collected_body_count=torch.zeros(B, Y_q, acc_D, dtype=torch.long, device=dev),
            collected_head=torch.full((B, Y_q, acc_D, 3), pad, dtype=torch.long, device=dev),
        )
    return fr, coll


__all__ = ["init_frontier", "_Collected"]
