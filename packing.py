"""Packing and compaction utilities for unification results.

Provides scatter-based compaction of derived states into fixed-shape output
tensors.  All functions are pure (no class dependencies) and torch.compile
compatible.
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor


def pack_combined(
    states: Tensor,             # [B, K_total, M_var, 3]
    success: Tensor,            # [B, K_total]
    K: int,
    M: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Compact valid entries from a unified states tensor to the front, cap at K.

    Uses scatter-based compaction for torch.compile compatibility (no topk/
    nonzero).

    Args:
        states:  [B, K_total, M_var, 3] derived states (M_var may differ from M)
        success: [B, K_total] validity mask
        K:       maximum output slots
        M:       target atoms dimension
        padding_idx: padding value

    Returns:
        derived: [B, K, M, 3] valid entries compacted to front
        counts:  [B] number of valid entries per batch element
    """
    B = states.shape[0]
    device = states.device
    pad = padding_idx
    M_var = states.shape[2]

    # Normalize M dimension
    if M_var < M:
        states = torch.nn.functional.pad(states, (0, 0, 0, M - M_var), value=pad)
    elif M_var > M:
        states = states[:, :, :M, :]

    # Count valid (capped at K)
    counts = success.sum(dim=1).clamp(max=K)

    # Scatter-based compaction for torch.compile compatibility
    cumsum = success.long().cumsum(dim=1)  # [B, K_total]

    target_idx = torch.where(
        success,
        cumsum - 1,
        K,  # garbage slot
    ).clamp(min=0, max=K)

    # Allocate output with +1 garbage slot
    derived = torch.full((B, K + 1, M, 3), pad, dtype=states.dtype, device=device)

    target_idx_exp = target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3)
    derived.scatter_(1, target_idx_exp, states)

    # Discard garbage slot
    derived = derived[:, :K, :, :].contiguous()

    return derived, counts


def compact_atoms(
    states: Tensor,             # [..., M, 3]
    padding_idx: int,
) -> Tensor:
    """Left-align atoms by removing gaps after pruning.

    Works for any shape [..., M, 3] (BE uses [B, K, M, 3], TS uses [B, S, G, 3]).
    Non-padding atoms are moved to the front within each (*, M) slice.

    Args:
        states: [..., M, 3] tensor with potential gaps (padding) in the M dimension
        padding_idx: padding value

    Returns:
        [..., M, 3] with atoms left-aligned
    """
    if states.numel() == 0:
        return states

    *leading, M, _ = states.shape
    # Flatten leading dims for uniform processing
    flat = states.reshape(-1, M, 3)
    N = flat.shape[0]
    device = states.device
    pad = padding_idx

    valid_atom = (flat[:, :, 0] != pad)  # [N, M]
    pos = torch.cumsum(valid_atom.long(), dim=1) - 1
    M_t = torch.tensor(M, dtype=pos.dtype, device=device)
    sort_key = torch.where(valid_atom, pos, M_t)

    sorted_indices = torch.argsort(sort_key, dim=1, stable=True)
    sorted_indices_exp = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
    result = torch.gather(flat, 1, sorted_indices_exp)

    return result.reshape(*leading, M, 3)


def pack_fact_rule(
    fact_gbody: Tensor,     # [B, N_f, M, 3]
    fact_goals: Tensor,     # [B, N_f, G, 3]
    fact_valid: Tensor,     # [B, N_f]
    fact_ridx: Tensor,      # [B, N_f]
    rule_gbody: Tensor,     # [B, N_r, M, 3]
    rule_goals: Tensor,     # [B, N_r, G, 3]
    rule_valid: Tensor,     # [B, N_r]
    rule_ridx: Tensor,      # [B, N_r]
    S: int,
    pad_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compact fact + rule derived states into [B, S, ...] (TS-style 4-output pack).

    Facts are placed first, then rules.  Uses cumsum + scatter for
    torch.compile compatibility (no topk).

    Args:
        fact_gbody: [B, N_f, M, 3] fact grounding bodies
        fact_goals: [B, N_f, G, 3] fact proof goals
        fact_valid: [B, N_f] fact validity mask
        fact_ridx:  [B, N_f] fact rule indices
        rule_gbody: [B, N_r, M, 3] rule grounding bodies
        rule_goals: [B, N_r, G, 3] rule proof goals
        rule_valid: [B, N_r] rule validity mask
        rule_ridx:  [B, N_r] rule rule indices
        S: maximum output slots
        pad_idx: padding value

    Returns:
        grounding_body: [B, S, M, 3]
        proof_goals:    [B, S, G, 3]
        top_ridx:       [B, S]
        state_valid:    [B, S]
    """
    B = fact_gbody.shape[0]
    M = fact_gbody.shape[2]
    G = fact_goals.shape[2]
    dev = fact_gbody.device

    # Concatenate: facts first, rules second
    all_gbody = torch.cat([fact_gbody, rule_gbody], dim=1)
    all_goals = torch.cat([fact_goals, rule_goals], dim=1)
    all_valid = torch.cat([fact_valid, rule_valid], dim=1)
    all_ridx = torch.cat([fact_ridx, rule_ridx], dim=1)

    # Scatter-based compaction
    cumsum = all_valid.long().cumsum(dim=1)       # [B, N]
    target_idx = torch.where(
        all_valid,
        cumsum - 1,                               # valid -> 0, 1, 2, ...
        torch.tensor(S, dtype=torch.long, device=dev),  # garbage
    ).clamp(min=0, max=S)                         # [B, N]

    # Allocate with garbage slot at S
    out_gbody = torch.zeros(
        B, S + 1, M, 3, dtype=torch.long, device=dev)
    out_goals = torch.full(
        (B, S + 1, G, 3), pad_idx, dtype=torch.long, device=dev)
    out_ridx = torch.zeros(
        B, S + 1, dtype=torch.long, device=dev)

    ti_4d = target_idx.unsqueeze(-1).unsqueeze(-1)
    out_gbody.scatter_(1, ti_4d.expand(-1, -1, M, 3), all_gbody)
    out_goals.scatter_(1, ti_4d.expand(-1, -1, G, 3), all_goals)
    out_ridx.scatter_(1, target_idx, all_ridx)

    # Valid mask from counts
    counts = all_valid.sum(dim=1).clamp(max=S)  # [B]
    arange_k = torch.arange(S, device=dev).unsqueeze(0)
    out_valid = arange_k < counts.unsqueeze(1)  # [B, S]

    # Discard garbage slot
    return (
        out_gbody[:, :S, :, :],
        out_goals[:, :S, :, :],
        out_ridx[:, :S],
        out_valid,
    )
