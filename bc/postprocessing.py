"""BC grounding collection + deduplication (``collect_groundings`` and the
collision-free injective-int64 ``_dedup_groundings``).

Pure and torch.compile-compatible. Split out of ``bc/common.py``.
"""

from __future__ import annotations
from typing import Dict, Optional, Set, Tuple

import torch
from torch import Tensor

from grounder.bc.packing import _pow_desc


def collect_groundings(
    grounding_body: Tensor,     # [B, S, D, M, 3] structured
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
    head_per_depth: Optional[Tensor] = None,   # [B, S, D, 3]
    collected_head: Optional[Tensor] = None,    # [B, C, D, 3]
    variant_to_orig: Optional[Tensor] = None,   # [num_variants] long
) -> Tuple:
    """Collect completed groundings into output buffer.

    Handles structured body [B, S, D, M, 3] and per-depth rule indices [B, S, D].

    Args:
        deactivate: If True (default), collected states are deactivated so they
            are not explored further. Set to False when collecting intermediate
            (grounded) states that should continue to deeper depths.

    Returns:
        out_body:    [B, C, D, M, 3]
        out_mask:    [B, C]
        out_ridx:    [B, C, D]
        state_valid: [B, S] updated
        out_bcount:  [B, C, D]
    """
    B, S, D_dim, M_dim, _ = grounding_body.shape
    dev = grounding_body.device
    E = constant_no + 1
    G_body_flat = D_dim * M_dim

    is_padding = (proof_goals[:, :, :, 0] == pad_idx)  # [B, S, G]

    # Flatten body for ground-check: [B, S, D*M, 3]
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
    # Cat along dim=1 — inner dims D, M, 3 carried through
    cb = torch.cat([collected_body, grounding_body], dim=1)     # [B, C+S, D, M, 3]
    cm = torch.cat([collected_mask, valid_grounding], dim=1)    # [B, C+S]
    cr = torch.cat([collected_ridx, ridx_per_depth], dim=1)     # [B, C+S, D]
    c_bc = torch.cat([collected_bcount, body_count], dim=1)     # [B, C+S, D]
    if has_head:
        c_hd = torch.cat([collected_head, head_per_depth], dim=1)  # [B, C+S, D, 3]

    # Dedup: hash over flat body + all D rule indices
    cb_flat = cb.reshape(B, n_cat, G_body_flat, 3)
    cm = _dedup_groundings(
        cb_flat, cr, cm, G_body_flat,
        variant_to_orig=variant_to_orig,
    )

    n_k = min(C, n_cat)
    _, ki = cm.to(torch.int8).topk(
        n_k, dim=1, largest=True, sorted=False)

    # Gather with structured dimensions
    ki_body = ki[:, :, None, None, None].expand(-1, -1, D_dim, M_dim, 3)
    ki_ridx = ki[:, :, None].expand(-1, -1, D_dim)
    ki_head = ki[:, :, None, None].expand(-1, -1, D_dim, 3) if has_head else None

    if n_k < C:
        p2 = C - n_k
        out_body = torch.nn.functional.pad(
            cb.gather(1, ki_body), (0, 0, 0, 0, 0, 0, 0, p2))
        out_mask = torch.nn.functional.pad(cm.gather(1, ki), (0, p2))
        out_ridx = torch.nn.functional.pad(cr.gather(1, ki_ridx), (0, 0, 0, p2))
        out_bcount = torch.nn.functional.pad(
            c_bc.gather(1, ki_ridx), (0, 0, 0, p2))
        out_head = (torch.nn.functional.pad(
            c_hd.gather(1, ki_head), (0, 0, 0, 0, 0, p2))
            if has_head else None)
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
    body: Tensor,       # [B, N, G_body, 3] (flat view)
    ridx: Tensor,       # [B, N] or [B, N, D]
    mask: Tensor,       # [B, N]
    G_body: int,
    *,
    variant_to_orig: Optional[Tensor] = None,  # [num_variants] long
) -> Tensor:
    """Remove duplicate groundings based on (ridx, body) hash.

    Hash is invariant to body-atom permutations within each depth's M slot
    (so all_anchors variants of the same logical rule application collapse),
    while preserving depth ordering. When ``variant_to_orig`` is provided,
    variant rule indices are remapped to their original rule index before
    hashing so anchor variants of the same rule share a key.

    Args:
        body: [B, N, G_body, 3] grounding body atoms (flat view)
        ridx: [B, N, D] per-depth rule indices, or [B, N] single rule index
        mask: [B, N] validity mask
        G_body: number of body atom slots in flat view
        variant_to_orig: optional [num_variants] map from variant rule
            index to original rule index (for all_anchors).

    Returns:
        mask: [B, N] updated mask with duplicates removed
    """
    B, N = mask.shape
    dev = mask.device
    P1, P2, P3, P4, P5 = 1_000_003, 999_983, 999_979, 999_961, 999_959

    # Body hash: [B, N], order-invariant within each depth's M slot.
    atom_hashes = (body[..., 0].long() * P1
                   + body[..., 1].long() * P2
                   + body[..., 2].long() * P3)               # [B, N, G_body]
    # ``ridx.dim()`` is a Python int (static under torch.compile), so this
    # branch specializes at trace time. Caller guarantees G_body == D * M
    # in the structured layout (see ``collect_groundings``).
    if ridx.dim() == 3:
        D = ridx.shape[2]
        M = G_body // D
        # Sort atom hashes within each depth's M slot → canonical order.
        ah_sorted, _ = atom_hashes.view(B, N, D, M).sort(dim=-1)
        m_powers = _pow_desc(P4, M, dev)
        per_depth_hash = (ah_sorted * m_powers).sum(dim=-1)   # [B, N, D]
        d_powers = _pow_desc(P5, D, dev)
        body_hash = (per_depth_hash * d_powers).sum(dim=-1)   # [B, N]
    else:
        powers = _pow_desc(P4, G_body, dev)
        body_hash = (atom_hashes * powers).sum(dim=-1)        # [B, N]

    # Remap variant rule indices to originals so anchor variants collapse.
    # ``variant_to_orig is not None`` is a Python None check, specialized
    # at trace time — no data-dependent branching.
    ridx_long = ridx.long()
    if variant_to_orig is not None:
        idx = ridx_long.clamp(min=0)
        remapped = variant_to_orig[idx]
        ridx_eff = torch.where(ridx >= 0, remapped, ridx_long)
    else:
        ridx_eff = ridx_long

    # Rule index hash: include all D dimensions if structured.
    if ridx.dim() == 3:
        D = ridx.shape[2]
        r_powers = _pow_desc(P4, D, dev)
        ridx_hash = (ridx_eff * r_powers).sum(dim=-1)         # [B, N]
    else:
        ridx_hash = ridx_eff                                  # [B, N]

    g_hash = ridx_hash * P1 + body_hash                       # [B, N]

    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    gh = torch.where(mask, g_hash, sentinel.expand(B, N))
    sorted_gh, sort_idx = gh.sort(dim=1)
    prev_gh = torch.nn.functional.pad(
        sorted_gh[:, :-1], (1, 0), value=-2)
    is_dup = (sorted_gh == prev_gh)
    inv_sort = sort_idx.argsort(dim=1)
    is_dup_orig = is_dup.gather(1, inv_sort)
    return mask & ~is_dup_orig


# ---------------------------------------------------------------------------
# rule2groundings pruning + tensor conversion
# ---------------------------------------------------------------------------
