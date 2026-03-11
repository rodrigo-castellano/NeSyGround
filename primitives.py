"""Core unification primitives: pairwise term unification and substitution application.

These are leaf-level building blocks with no internal package dependencies.
All operations are vectorized and torch.compile compatible.

Part of the grounder package.
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor


@torch.no_grad()
def apply_substitutions(goals: Tensor, subs_pairs: Tensor, padding_idx: int) -> Tensor:
    """Apply variable substitutions to goal atoms (optimized for S=2).

    Args:
        goals:      [N, M, 3] goal atoms (pred, arg0, arg1)
        subs_pairs: [N, S, 2] substitution pairs (from, to)
        padding_idx: index used for padding

    Returns:
        [N, M, 3] goals with substitutions applied to argument positions
    """
    if goals.numel() == 0:
        return goals

    N, M = goals.shape[:2]
    S = subs_pairs.shape[1]
    pad = padding_idx

    preds = goals[:, :, 0:1]  # [N, M, 1]
    args = goals[:, :, 1:]    # [N, M, 2]

    # OPTIMIZATION: Loop-unrolled for common S=2 case
    if S == 2:
        frm_0 = subs_pairs[:, 0, 0].view(N, 1, 1)  # [N] → [N, 1, 1] for broadcast vs [N, M, 2]
        to_0 = subs_pairs[:, 0, 1].view(N, 1, 1)
        frm_1 = subs_pairs[:, 1, 0].view(N, 1, 1)
        to_1 = subs_pairs[:, 1, 1].view(N, 1, 1)

        valid_0 = (frm_0 != pad)  # [N, 1, 1]
        valid_1 = (frm_1 != pad)  # [N, 1, 1]

        result_args = torch.where((args == frm_0) & valid_0, to_0, args)  # [N, M, 2]
        result_args = torch.where((result_args == frm_1) & valid_1, to_1, result_args)  # [N, M, 2]

        return torch.cat([preds, result_args], dim=2)  # [N, M, 3]

    # General case for S != 2
    valid = subs_pairs[..., 0] != pad             # [N, S]
    frm = subs_pairs[:, :, 0].view(N, S, 1, 1)   # [N, S, 1, 1] for broadcast vs args
    to_ = subs_pairs[:, :, 1].view(N, S, 1, 1)   # [N, S, 1, 1]
    args_exp = args.view(N, 1, M, 2)              # [N, 1, M, 2] for broadcast vs subs
    valid_exp = valid.view(N, S, 1, 1)             # [N, S, 1, 1]

    match = (args_exp == frm) & valid_exp  # [N, S, M, 2]
    any_match = match.any(dim=1)           # [N, M, 2]
    match_idx = match.long().argmax(dim=1) # [N, M, 2] — index into S dim

    to_flat = subs_pairs[:, :, 1]                          # [N, S]
    match_idx_flat = match_idx.view(N, M * 2)              # [N, M*2]
    to_gathered = to_flat.gather(1, match_idx_flat).view(N, M, 2)  # [N, M*2] → [N, M, 2]

    result_args = torch.where(any_match, to_gathered, args)  # [N, M, 2]
    return torch.cat([preds, result_args], dim=2)  # [N, M, 3]


@torch.no_grad()
def unify_one_to_one(
    queries: Tensor,
    terms: Tensor,
    constant_no: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Perform pairwise unification between queries and terms.

    Args:
        queries:     [L, 3] query atoms
        terms:       [L, 3] target atoms (facts or rule heads)
        constant_no: highest constant index (variables start at constant_no + 1)
        padding_idx: padding index

    Returns:
        mask: [L] bool — whether each pair unified successfully
        subs: [L, 2, 2] substitution pairs (from, to) per position
    """
    device = queries.device
    L = queries.shape[0]

    if L == 0:
        return (torch.empty(0, dtype=torch.bool, device=device),
                torch.full((0, 2, 2), padding_idx, dtype=torch.long, device=device))

    var_start = constant_no + 1
    pad = padding_idx

    # Extract predicates and args
    pred_ok = (queries[:, 0] == terms[:, 0])       # [L]
    q_args, t_args = queries[:, 1:], terms[:, 1:]  # [L, 2], [L, 2]

    # Compute masks once — all [L, 2]
    q_const = (q_args <= constant_no)
    t_const = (t_args <= constant_no)
    qv = (q_args >= var_start) & (q_args != pad)
    tv = (t_args >= var_start) & (t_args != pad)

    # Constant conflict check
    const_conflict = (q_const & t_const & (q_args != t_args)).any(dim=1)  # [L, 2] → [L]
    mask = pred_ok & ~const_conflict  # [L]

    # Compute substitutions in single vectorized pass — all cases are [L, 2]
    case1 = qv & ~tv & (t_args != 0)
    case2 = ~qv & (q_args != 0) & tv
    case3 = qv & tv

    # Nested torch.where: cases are mutually exclusive, so collapse to 2 ops
    pad_t = torch.tensor(pad, dtype=q_args.dtype, device=q_args.device)
    from_val = torch.where(case1, q_args, torch.where(case2 | case3, t_args, pad_t))  # [L, 2]
    to_val = torch.where(case1, t_args, torch.where(case2 | case3, q_args, pad_t))    # [L, 2]

    # Stack into subs: from_val[L,2] + to_val[L,2] → [L, 2, 2] (per-position from/to pairs)
    subs = torch.stack([from_val, to_val], dim=2)  # [L, 2, 2]

    # Consistency check: same var bound to different values
    same_var = (subs[:, 0, 0] == subs[:, 1, 0]) & (subs[:, 0, 0] != pad)  # [L]
    diff_tgt = subs[:, 0, 1] != subs[:, 1, 1]  # [L]
    conflict = same_var & diff_tgt  # [L]
    mask = mask & ~conflict

    # Clear subs for failed unifications
    fail_mask = ~mask
    subs = torch.where(fail_mask.view(L, 1, 1), pad_t, subs)  # [L, 1, 1] broadcast → [L, 2, 2]

    return mask, subs
