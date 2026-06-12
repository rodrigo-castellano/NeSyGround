"""PBC width filtering — the BC_{w,d,u} soundness/pressure prune.

A candidate grounding survives when: at most ``width`` of its active body atoms
are unknown (not facts); no body atom equals the query (cycle exclusion); and —
when ``width`` is bounded — every unknown body atom carries a head predicate
(provable at a later step). At the last step (``width==0`` / ``is_last``) every
active body atom must be a fact.

BIT-EXACT RECIPES (fingerprint-enforced): the ``num_unknown <= width`` test, the
query-exclusion all-equal, the head-pred prune, and the ``width==0`` strict
all-exist branch — collectively the BC width semantics the keras oracle matches.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def depth_gate(d, depth: int, width, w_last_depth: int, head_pred_mask, is_last):
    """Last-step gating → ``(width_d, head_pred_mask_d)``. Eager (``is_last`` None →
    Python branch): last step uses ``w_last_depth`` and drops the head-pred prune.
    Compiled (``is_last`` 0-dim bool): width via ``torch.where``; prune gated in the
    filter, so head_pred_mask passes through."""
    if is_last is None:
        last = (d == depth - 1)
        width_d = w_last_depth if (last and width is not None) else width
        return width_d, (None if last else head_pred_mask)
    if width is None:
        return None, head_pred_mask
    dev = is_last.device
    width_d = torch.where(is_last, torch.tensor(w_last_depth, dtype=torch.long, device=dev),
                          torch.tensor(width, dtype=torch.long, device=dev))
    return width_d, head_pred_mask


def all_body_active_ok(ok: Tensor, body_active: Tensor) -> Tensor:
    """Reduce a per-atom OK mask over body atoms: survives iff every ACTIVE
    body atom is OK. ``[...,M] -> [...]`` (rank-agnostic: dense & flat)."""
    return (ok | ~body_active).all(dim=-1)


def head_pred_all_ok(body_pred_vals: Tensor, exists: Tensor,
                     body_active: Tensor, head_pred_mask: Tensor) -> Tensor:
    """Every active body atom is a fact OR carries a head predicate (provable
    later). Padded preds clamp into range — ``~body_active`` admits them anyway."""
    P = head_pred_mask.size(0)
    head_pred_ok = head_pred_mask[body_pred_vals.clamp(max=P - 1)]
    return all_body_active_ok(exists | head_pred_ok, body_active)


def filter_dense(has_query_atom: Tensor, exists: Tensor, body_active: Tensor,
                        body_preds: Tensor, active_mask: Tensor, cand_mask: Tensor,
                        width, head_pred_mask: Optional[Tensor],
                        *, is_last: Optional[Tensor] = None) -> Tensor:
    """Dense width + query-exclusion + head-pred prune → ``[B,K_r,Y_r]`` mask.

    Slice-fused: the padded ``[.,Y_r,M,3]`` body tensor never exists.
    ``has_query_atom [B,K_r,Y_r]`` is precomputed by the dense fill (the only
    consumer of full body triples); the head-pred prune reads ``body_preds
    [B,K_r,M]`` directly (per-rule, y-independent). Inactive slots feed raw
    (unpadded) preds — every reduction ORs ``~body_active``, so the result is
    bit-identical to the padded-body recipe.

    ``width``: int (eager) | 0-dim Tensor (compiled) | None (∞).
    ``head_pred_mask=None`` disables the unprovable-unknown prune (last step,
    keras ``prune_incomplete_proofs=False``). In the compiled path the prune /
    strict branch are gated on ``is_last`` via ``torch.where`` (no Python if).
    """
    Y_r = exists.size(2)
    body_active_exp = body_active.expand(-1, -1, Y_r, -1)

    if width is None:
        mask = active_mask.unsqueeze(2) & cand_mask
    else:
        num_unknown = (body_active_exp & ~exists).sum(dim=-1)
        mask = (num_unknown <= width) & active_mask.unsqueeze(2) & cand_mask

    mask = mask & ~has_query_atom

    if width is not None:
        if head_pred_mask is not None:
            P = head_pred_mask.size(0)
            head_pred_ok = head_pred_mask[body_preds.clamp(max=P - 1)]   # [B,K_r,M]
            all_ok = all_body_active_ok(exists | head_pred_ok.unsqueeze(2),
                                        body_active_exp)
            if is_last is not None:
                all_ok = torch.where(is_last, torch.ones_like(all_ok), all_ok)
            mask = mask & all_ok
        if torch.is_tensor(width):
            if is_last is not None:
                strict = mask & all_body_active_ok(exists, body_active_exp)
                mask = torch.where(is_last, strict, mask)
        elif width == 0:
            mask = mask & all_body_active_ok(exists, body_active_exp)
    return mask


def filter_flat(flat_body: Tensor, flat_exists: Tensor, flat_b_idx: Tensor,
                       flat_num_body: Tensor, queries: Tensor, width,
                       head_pred_mask: Optional[Tensor], M: int,
                       body_active: Optional[Tensor] = None) -> Tensor:
    """Flat width + query-exclusion + head-pred prune → ``[T]`` survivor mask."""
    T = flat_body.size(0)
    dev = flat_body.device
    if body_active is None:
        atom_idx = torch.arange(M, device=dev).unsqueeze(0)
        body_active = atom_idx < flat_num_body.unsqueeze(1)

    if width is None:
        mask = torch.ones(T, dtype=torch.bool, device=dev)
    else:
        mask = (body_active & ~flat_exists).sum(dim=-1) <= width

    query_exp = queries[flat_b_idx]
    has_query_atom = ((flat_body == query_exp.unsqueeze(1)).all(dim=-1) & body_active).any(dim=-1)
    mask = mask & ~has_query_atom

    if width is not None:
        if head_pred_mask is not None:
            mask = mask & head_pred_all_ok(flat_body[..., 0], flat_exists, body_active, head_pred_mask)
        if width == 0:
            mask = mask & all_body_active_ok(flat_exists, body_active)
    return mask


__all__ = ["filter_dense", "filter_flat", "all_body_active_ok", "head_pred_all_ok", "depth_gate"]
