"""Initial-state builder for ``BCGrounder``'s proof loop.

A single free function — :func:`init_states` — allocates every tensor
the per-depth loop reads/writes and packs them into the canonical
states dict consumed by ``BCGrounder.step``. Lives outside the class
so the shape derivation (``S_init`` / ``M_work`` / accumulator
dimensions) is testable in isolation and the BC orchestration in
``bc.py`` doesn't have to inline all of it.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor


def init_states(
    grounder,
    queries: Tensor,
    query_mask: Tensor,
    *,
    initial_goals: Optional[Tensor] = None,
    next_var_indices: Optional[Tensor] = None,
    excluded_queries: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Build initial states dict for the proof loop.

    Args:
        grounder: ``BCGrounder`` instance — reads ``max_goals`` /
            ``C`` / ``depth`` / ``S`` / ``kb.M`` / ``kb.padding_idx``
            / ``kb.constant_no`` / ``_init_state_shape`` /
            ``collect_evidence`` / ``filter_mode`` from it.
        queries: ``[B, 3]`` query atoms.
        query_mask: ``[B]`` validity mask.
        initial_goals: ``[B, M_in, 3]`` multi-atom goal list to use
            instead of the single query atom (for RL mid-proof entry).
        next_var_indices: ``[B]`` pre-allocated variable counters.
            Defaults to ``constant_no + 1`` (fresh).
        excluded_queries: optional tensor for cycle prevention.
    """
    B = queries.size(0)
    dev = queries.device
    pad = grounder.kb.padding_idx
    G = grounder.max_goals
    C = grounder.C
    D = grounder.depth
    M = grounder.kb.M  # max body atoms in any single rule
    # M_work: working buffer for the current depth's body atoms.
    # Must fit the widest rule body (``M``) — ``pack_states_flat``
    # writes ``new_body = flat_goals[:, :M_rule, :]`` into this
    # buffer (bc/common.py around line 421), and rules with
    # ``M_rule > 1`` overflow the index target if M_work=1. The
    # earlier ``1 if not self.collect_evidence else M`` shortcut
    # saved a few KB on collect_evidence=False but produced
    # ``shape mismatch: value tensor of shape [N, 2, 3] cannot be
    # broadcast to indexing result of shape [N, 1, 3]`` on any
    # KB with ``kb.M > 1`` (every dataset except trivially-1-body
    # ones — ablation_d2, family, wn18rr, fb15k237, …).
    M_work = M

    # ``init_state_shape``:
    #   "minimal" — S_init=1; smallest possible buffer at d=0.
    #   "full"    — S_init=self.S; same shape as d>=1 so a single
    #               compiled graph covers every depth.
    S_init = 1 if grounder._init_state_shape == "minimal" else grounder.S

    proof_goals = torch.full(
        (B, S_init, G, 3), pad, dtype=torch.long, device=dev)
    if initial_goals is not None:
        M_in = initial_goals.shape[1]
        proof_goals[:, 0, :M_in, :] = initial_goals
    else:
        proof_goals[:, 0, 0, :] = queries
    # M-sized working buffer (current depth's rule body atoms)
    grounding_body = torch.full(
        (B, S_init, M_work, 3), pad, dtype=torch.long, device=dev)
    # Structured accumulator: [B, S, D, M, 3] — one slot per depth.
    # ``acc_D``/``acc_M`` collapse to 1 when nothing reads the
    # accumulator, but the ``fp_global`` filter writes witness
    # body atoms into ``accumulated_body`` via
    # ``inject_witnesses_into_evidence`` and needs the full
    # ``M`` slot count even when ``collect_evidence=False``.
    skip_acc = (not grounder.collect_evidence
                and grounder.filter_mode != "fp_global")
    acc_D = 1 if skip_acc else D
    acc_M = 1 if skip_acc else M
    accumulated_body = torch.full(
        (B, S_init, acc_D, acc_M, 3), pad, dtype=torch.long, device=dev)
    body_count = torch.zeros(B, S_init, acc_D, dtype=torch.long, device=dev)
    ridx_per_depth = torch.full(
        (B, S_init, acc_D), -1, dtype=torch.long, device=dev)
    head_per_depth = torch.full(
        (B, S_init, acc_D, 3), pad, dtype=torch.long, device=dev)
    top_ridx = torch.full((B, S_init), -1, dtype=torch.long, device=dev)
    if S_init == 1:
        state_valid = query_mask.unsqueeze(1)              # [B, 1]
    else:
        # Only slot 0 carries the active query; slots 1.. are inactive.
        state_valid = torch.zeros(
            B, S_init, dtype=torch.bool, device=dev)
        state_valid[:, 0] = query_mask

    if next_var_indices is None:
        E = grounder.kb.constant_no + 1
        next_var_indices = torch.full(
            (B,), E, dtype=torch.long, device=dev)

    states = {
        "queries": queries,
        "query_mask": query_mask,
        "proof_goals": proof_goals,
        "grounding_body": grounding_body,
        "accumulated_body": accumulated_body,
        "body_count": body_count,
        "ridx_per_depth": ridx_per_depth,
        "head_per_depth": head_per_depth,
        "top_ridx": top_ridx,
        "state_valid": state_valid,
        "next_var_indices": next_var_indices,
        "initial_next_var": next_var_indices,
        "collected_body": queries.new_zeros(B, C, acc_D, acc_M, 3),
        "collected_mask": torch.zeros(B, C, dtype=torch.bool, device=dev),
        "collected_ridx": queries.new_full((B, C, acc_D), -1,
                                           dtype=torch.long),
        "collected_bcount": torch.zeros(B, C, acc_D, dtype=torch.long,
                                        device=dev),
        "collected_head": torch.full((B, C, acc_D, 3), pad,
                                     dtype=torch.long, device=dev),
    }
    if initial_goals is not None:
        states["initial_goals"] = initial_goals
    if excluded_queries is not None:
        states["excluded_queries"] = excluded_queries
    return states


__all__ = ["init_states"]
