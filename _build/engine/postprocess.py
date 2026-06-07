"""Grounding collection + dedup — ported from OLD ``bc/postprocessing.py``.

``collect_groundings`` gathers terminal (all-goals-padding, body-ground) states
into the per-query collected buffer each step; ``_dedup_groundings`` removes
duplicate (rule, body) firings via the 5-prime order-invariant hash.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder._build.engine.pack import _pow_desc


def collect_groundings(
    grounding_body: Tensor,     # [B, S, D, M, 3]
    proof_goals: Tensor,        # [B, S, G, 3]
    state_valid: Tensor,        # [B, S]
    ridx_per_depth: Tensor,     # [B, S, D]
    collected_body: Tensor,     # [B, C, D, M, 3]
    collected_mask: Tensor,     # [B, C]
    collected_ridx: Tensor,     # [B, C, D]
    constant_no: int,
    pad_idx: int,
    C: int,
    body_count: Tensor,          # [B, S, D]
    collected_bcount: Tensor,    # [B, C, D]
    collect_mode: str = "terminal",
    deactivate: bool = True,
    head_per_depth: Optional[Tensor] = None,
    collected_head: Optional[Tensor] = None,
    variant_to_orig: Optional[Tensor] = None,
) -> Tuple:
    """Collect completed groundings into output buffer (structured body)."""
    B, S, D_dim, M_dim, _ = grounding_body.shape
    dev = grounding_body.device
    E = constant_no + 1
    G_body_flat = D_dim * M_dim

    is_padding = (proof_goals[:, :, :, 0] == pad_idx)

    body_flat = grounding_body.reshape(B, S, G_body_flat, 3)
    body_args = body_flat[:, :, :, 1:3]
    body_active = (body_flat[:, :, :, 0] != pad_idx)
    is_ground = ((body_args < E) | ~body_active.unsqueeze(-1)).all(dim=-1).all(dim=-1)

    if collect_mode == "grounded":
        goal_args = proof_goals[:, :, :, 1:3]
        goal_grounded = (goal_args < E).all(dim=-1)
        all_goals_ok = (is_padding | goal_grounded).all(dim=2)
    else:
        all_goals_ok = is_padding.all(dim=2)

    valid_grounding = all_goals_ok & is_ground & state_valid

    has_head = head_per_depth is not None and collected_head is not None

    n_new = S
    n_cat = C + n_new
    cb = torch.cat([collected_body, grounding_body], dim=1)
    cm = torch.cat([collected_mask, valid_grounding], dim=1)
    cr = torch.cat([collected_ridx, ridx_per_depth], dim=1)
    c_bc = torch.cat([collected_bcount, body_count], dim=1)
    if has_head:
        c_hd = torch.cat([collected_head, head_per_depth], dim=1)

    cb_flat = cb.reshape(B, n_cat, G_body_flat, 3)
    cm = _dedup_groundings(cb_flat, cr, cm, G_body_flat, variant_to_orig=variant_to_orig)

    n_k = min(C, n_cat)
    _, ki = cm.to(torch.int8).topk(n_k, dim=1, largest=True, sorted=False)

    ki_body = ki[:, :, None, None, None].expand(-1, -1, D_dim, M_dim, 3)
    ki_ridx = ki[:, :, None].expand(-1, -1, D_dim)
    ki_head = ki[:, :, None, None].expand(-1, -1, D_dim, 3) if has_head else None

    if n_k < C:
        p2 = C - n_k
        out_body = torch.nn.functional.pad(
            cb.gather(1, ki_body), (0, 0, 0, 0, 0, 0, 0, p2))
        out_mask = torch.nn.functional.pad(cm.gather(1, ki), (0, p2))
        out_ridx = torch.nn.functional.pad(cr.gather(1, ki_ridx), (0, 0, 0, p2))
        out_bcount = torch.nn.functional.pad(c_bc.gather(1, ki_ridx), (0, 0, 0, p2))
        out_head = (torch.nn.functional.pad(
            c_hd.gather(1, ki_head), (0, 0, 0, 0, 0, p2)) if has_head else None)
    else:
        out_body = cb.gather(1, ki_body)
        out_mask = cm.gather(1, ki)
        out_ridx = cr.gather(1, ki_ridx)
        out_bcount = c_bc.gather(1, ki_ridx)
        out_head = c_hd.gather(1, ki_head) if has_head else None

    if deactivate:
        state_valid = state_valid & ~valid_grounding

    return out_body, out_mask, out_ridx, state_valid, out_bcount, out_head


def _dedup_groundings(
    body: Tensor,       # [B, N, G_body, 3]
    ridx: Tensor,       # [B, N] or [B, N, D]
    mask: Tensor,       # [B, N]
    G_body: int,
    *,
    variant_to_orig: Optional[Tensor] = None,
) -> Tensor:
    """Remove duplicate groundings based on (ridx, body) hash."""
    B, N = mask.shape
    dev = mask.device
    P1, P2, P3, P4, P5 = 1_000_003, 999_983, 999_979, 999_961, 999_959

    atom_hashes = (body[..., 0].long() * P1
                   + body[..., 1].long() * P2
                   + body[..., 2].long() * P3)
    if ridx.dim() == 3:
        D = ridx.shape[2]
        M = G_body // D
        ah_sorted, _ = atom_hashes.view(B, N, D, M).sort(dim=-1)
        m_powers = _pow_desc(P4, M, dev)
        per_depth_hash = (ah_sorted * m_powers).sum(dim=-1)
        d_powers = _pow_desc(P5, D, dev)
        body_hash = (per_depth_hash * d_powers).sum(dim=-1)
    else:
        powers = _pow_desc(P4, G_body, dev)
        body_hash = (atom_hashes * powers).sum(dim=-1)

    ridx_long = ridx.long()
    if variant_to_orig is not None:
        idx = ridx_long.clamp(min=0)
        remapped = variant_to_orig[idx]
        ridx_eff = torch.where(ridx >= 0, remapped, ridx_long)
    else:
        ridx_eff = ridx_long

    if ridx.dim() == 3:
        D = ridx.shape[2]
        r_powers = _pow_desc(P4, D, dev)
        ridx_hash = (ridx_eff * r_powers).sum(dim=-1)
    else:
        ridx_hash = ridx_eff

    g_hash = ridx_hash * P1 + body_hash

    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    gh = torch.where(mask, g_hash, sentinel.expand(B, N))
    sorted_gh, sort_idx = gh.sort(dim=1)
    prev_gh = torch.nn.functional.pad(sorted_gh[:, :-1], (1, 0), value=-2)
    is_dup = (sorted_gh == prev_gh)
    inv_sort = sort_idx.argsort(dim=1)
    is_dup_orig = is_dup.gather(1, inv_sort)
    return mask & ~is_dup_orig


__all__ = ["collect_groundings", "_dedup_groundings"]
