"""Grounding collection + dedup.

``collect_groundings`` gathers terminal (all-goals-padding, body-ground) states
into the per-query collected buffer each step; ``_dedup_groundings`` removes
duplicate (rule, body) firings via the 5-prime order-invariant hash.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.backward.pack import _pow_desc


def collect_groundings(
    grounding_body: Tensor,     # [B, G, D, M, 3]
    goal_atoms: Tensor,         # [B, G, L, 3]
    goal_valid: Tensor,         # [B, G]
    rule_idx_per_depth: Tensor,     # [B, G, D]
    collected_body: Tensor,     # [B, Y_q, D, M, 3]
    collected_mask: Tensor,     # [B, Y_q]
    collected_rule_idx: Tensor,     # [B, Y_q, D]
    constant_no: int,
    pad_idx: int,
    Y_q: int,
    body_count: Tensor,          # [B, G, D]
    collected_body_count: Tensor,    # [B, Y_q, D]
    collect_mode: str = "terminal",
    deactivate: bool = True,
    head_per_depth: Optional[Tensor] = None,
    collected_head: Optional[Tensor] = None,
    variant_to_orig: Optional[Tensor] = None,
) -> Tuple:
    """Collect completed groundings into output buffer (structured body)."""
    B, S, D_dim, M_dim, _ = grounding_body.shape
    E = constant_no + 1
    G_body_flat = D_dim * M_dim

    is_padding = (goal_atoms[:, :, :, 0] == pad_idx)

    body_flat = grounding_body.reshape(B, S, G_body_flat, 3)
    body_args = body_flat[:, :, :, 1:3]
    body_active = (body_flat[:, :, :, 0] != pad_idx)
    is_ground = ((body_args < E) | ~body_active.unsqueeze(-1)).all(dim=-1).all(dim=-1)

    if collect_mode == "grounded":
        goal_args = goal_atoms[:, :, :, 1:3]
        goal_grounded = (goal_args < E).all(dim=-1)
        all_goals_ok = (is_padding | goal_grounded).all(dim=2)
    else:
        all_goals_ok = is_padding.all(dim=2)

    valid_grounding = all_goals_ok & is_ground & goal_valid

    n_new = S
    n_cat = Y_q + n_new  # > Y_q, so the topk keeps exactly Y_q (no pad branch)
    cb = torch.cat([collected_body, grounding_body], dim=1)
    cm = torch.cat([collected_mask, valid_grounding], dim=1)
    cr = torch.cat([collected_rule_idx, rule_idx_per_depth], dim=1)
    c_bc = torch.cat([collected_body_count, body_count], dim=1)
    c_hd = torch.cat([collected_head, head_per_depth], dim=1)

    cb_flat = cb.reshape(B, n_cat, G_body_flat, 3)
    cm = _dedup_groundings(cb_flat, cr, cm, G_body_flat, variant_to_orig=variant_to_orig)

    _, ki = cm.to(torch.int8).topk(Y_q, dim=1, largest=True, sorted=False)

    ki_body = ki[:, :, None, None, None].expand(-1, -1, D_dim, M_dim, 3)
    ki_rule_idx = ki[:, :, None].expand(-1, -1, D_dim)
    ki_head = ki[:, :, None, None].expand(-1, -1, D_dim, 3)

    out_body = cb.gather(1, ki_body)
    out_mask = cm.gather(1, ki)
    out_rule_idx = cr.gather(1, ki_rule_idx)
    out_body_count = c_bc.gather(1, ki_rule_idx)
    out_head = c_hd.gather(1, ki_head)

    if deactivate:
        goal_valid = goal_valid & ~valid_grounding

    return out_body, out_mask, out_rule_idx, goal_valid, out_body_count, out_head


def _dedup_groundings(
    body: Tensor,       # [B, N, G_body, 3]
    rule_idx: Tensor,       # [B, N, D]
    mask: Tensor,       # [B, N]
    G_body: int,
    *,
    variant_to_orig: Optional[Tensor] = None,
) -> Tensor:
    """Remove duplicate groundings based on (rule_idx, body) hash."""
    B, N = mask.shape
    dev = mask.device
    P1, P2, P3, P4, P5 = 1_000_003, 999_983, 999_979, 999_961, 999_959

    atom_hashes = (body[..., 0].long() * P1
                   + body[..., 1].long() * P2
                   + body[..., 2].long() * P3)
    D = rule_idx.shape[2]  # rule_idx is always 3-D ([B, N, D])
    M = G_body // D
    ah_sorted, _ = atom_hashes.view(B, N, D, M).sort(dim=-1)
    m_powers = _pow_desc(P4, M, dev)
    per_depth_hash = (ah_sorted * m_powers).sum(dim=-1)
    d_powers = _pow_desc(P5, D, dev)
    body_hash = (per_depth_hash * d_powers).sum(dim=-1)

    rule_idx_long = rule_idx.long()
    if variant_to_orig is not None:
        idx = rule_idx_long.clamp(min=0)
        remapped = variant_to_orig[idx]
        rule_idx_eff = torch.where(rule_idx >= 0, remapped, rule_idx_long)
    else:
        rule_idx_eff = rule_idx_long

    r_powers = _pow_desc(P4, D, dev)
    rule_idx_hash = (rule_idx_eff * r_powers).sum(dim=-1)

    g_hash = rule_idx_hash * P1 + body_hash

    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    gh = torch.where(mask, g_hash, sentinel.expand(B, N))
    sorted_gh, sort_idx = gh.sort(dim=1)
    prev_gh = torch.nn.functional.pad(sorted_gh[:, :-1], (1, 0), value=-2)
    is_dup = (sorted_gh == prev_gh)
    inv_sort = sort_idx.argsort(dim=1)
    is_dup_orig = is_dup.gather(1, inv_sort)
    return mask & ~is_dup_orig


__all__ = ["collect_groundings", "_dedup_groundings"]
