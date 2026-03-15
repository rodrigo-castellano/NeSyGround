"""Variable standardization for derived proof states.

Three public APIs:
- standardize_vars_offset: fast offset-based renaming (no fullgraph guarantee)
- standardize_vars_canonical: graph-safe canonical renaming (torch.compile compatible)
- build_standardize_fn: factory that returns a compiled standardizer callable

These are pure functions with no internal package dependencies (only torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class StandardizationConfig:
    """Configuration for output variable standardization.

    Bundles the parameters needed by build_standardize_fn into a single
    immutable config object, eliminating scattered parameter passing.
    """
    mode: str                   # 'offset' | 'canonical'
    constant_no: int
    runtime_var_end_index: int
    padding_idx: int
    body_width: int
    compile_mode: str = "reduce-overhead"
    enforce_runtime_range: bool = False


def standardize_vars_offset(
    states: Tensor,          # [B, K, M, 3]
    counts: Tensor,          # [B] (kept for signature compatibility)
    next_var_indices: Tensor,  # [B]
    constant_no: int,
    runtime_var_end_index: Optional[int],
    padding_idx: int,
    input_states: Optional[Tensor] = None,
    extra_new_vars: int = 15,
    enforce_runtime_range: bool = False,
    out_of_place: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Fast offset-based runtime variable standardization.

    Shifts variables in derived ``states`` so they don't collide with existing
    variables tracked by ``next_var_indices``.  The offset is computed from
    the minimum variable in ``input_states`` (the goals that produced these
    derived states).

    Args:
        states: [B, K, M, 3] derived proof-goal states
        counts: [B] valid state count per batch (unused, kept for API compat)
        next_var_indices: [B] next free variable index per batch element
        constant_no: highest constant index (variables start at constant_no + 1)
        runtime_var_end_index: upper bound for variable IDs (optional)
        padding_idx: padding index
        input_states: [B, ?, 3] parent states used to derive ``states``
        extra_new_vars: headroom added to next_var estimate
        enforce_runtime_range: if True, assert output variables are in range
        out_of_place: if True, never modify ``states`` in-place

    Returns:
        standardized: [B, K, M, 3] states with shifted variables
        new_next_var: [B] updated next free variable indices
    """
    device = states.device
    B, K, M, _ = states.shape
    pad = padding_idx

    if B == 0 or states.numel() == 0:
        return states, next_var_indices

    LARGE = 1_000_000
    min_var_in = torch.full((B,), LARGE, dtype=torch.long, device=device)
    max_var_in = torch.zeros(B, dtype=torch.long, device=device)
    has_input_vars = torch.zeros(B, dtype=torch.bool, device=device)

    if input_states is not None and input_states.numel() > 0:
        in_args = input_states[:, :, 1:3]
        is_var_in = (in_args > constant_no) & (in_args != pad)

        large_t = torch.tensor(LARGE, dtype=in_args.dtype, device=in_args.device)
        masked_min = torch.where(is_var_in, in_args, large_t)
        min_var_in = masked_min.min(dim=-1).values.min(dim=-1).values
        masked_max = torch.where(is_var_in, in_args, torch.zeros_like(in_args))
        max_var_in = masked_max.max(dim=-1).values.max(dim=-1).values
        has_input_vars = min_var_in < LARGE

    offset = torch.where(
        has_input_vars,
        next_var_indices - min_var_in,
        torch.zeros_like(next_var_indices),
    )

    args = states[:, :, :, 1:3]
    is_var_out = (args > constant_no) & (args != pad)
    offset_exp = offset.view(B, 1, 1, 1).expand(-1, K, M, 2)
    standardized_args = torch.where(is_var_out, args + offset_exp, args)

    if enforce_runtime_range and runtime_var_end_index is not None:
        lower_ok = ((~is_var_out) | (standardized_args >= (constant_no + 1))).all()
        upper_ok = ((~is_var_out) | (standardized_args <= runtime_var_end_index)).all()
        torch._assert_async(lower_ok, "Variable standardization produced ID below runtime range")
        torch._assert_async(upper_ok, "Variable standardization produced ID above runtime range")

    if out_of_place:
        standardized = states.clone()
        standardized[:, :, :, 1:3] = standardized_args
    else:
        states[:, :, :, 1:3] = standardized_args
        standardized = states

    max_in_shifted = torch.where(
        has_input_vars,
        max_var_in + offset,
        torch.zeros_like(max_var_in),
    )
    max_gen_shifted = next_var_indices + extra_new_vars
    current_max_new = torch.maximum(max_in_shifted, max_gen_shifted)
    new_next_var = current_max_new + 1

    if enforce_runtime_range and runtime_var_end_index is not None:
        next_ok = (new_next_var <= runtime_var_end_index).all()
        torch._assert_async(next_ok, "Next variable pointer exceeded runtime variable range")

    return standardized, new_next_var


def standardize_vars_canonical(
    states: Tensor,          # [B, K, M, 3]
    counts: Tensor,          # [B] (kept for signature compatibility)
    next_var_indices: Tensor,  # [B]
    constant_no: int,
    runtime_var_end_index: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Graph-safe canonical standardization.

    Renumbers variables in derived ``states`` to a contiguous canonical form
    starting at ``next_var_indices[b]`` for each batch element.  Variable
    ordering is determined by first appearance (left-to-right scan).

    This implementation avoids dynamic boolean indexing and data-dependent
    shape ops so it can run under ``torch.compile(..., fullgraph=True)`` with
    ``mode='reduce-overhead'`` and CUDA graph capture.

    Args:
        states: [B, K, M, 3] derived proof-goal states
        counts: [B] valid state count per batch (unused, kept for API compat)
        next_var_indices: [B] next free variable index per batch element
        constant_no: highest constant index (variables start at constant_no + 1)
        runtime_var_end_index: upper bound for variable IDs
        padding_idx: padding index

    Returns:
        standardized: [B, K, M, 3] states with canonically renumbered variables
        new_next_var: [B] updated next free variable indices
    """
    if states.numel() == 0:
        return states, next_var_indices

    device = states.device
    B, K, M, _ = states.shape
    if B == 0 or K == 0:
        return states, next_var_indices

    N = B * K
    P = M * 2
    flat_states = states.view(N, M, 3)

    # Owners map each flattened derived state back to batch owner.
    owners = torch.arange(B, device=device).repeat_interleave(K)
    base = next_var_indices.index_select(0, owners).view(N, 1)

    args = flat_states[:, :, 1:3].reshape(N, P)
    is_var = (args > constant_no) & (args != padding_idx)
    # Match legacy semantics: only renumber variables at/above owner next-var.
    is_new = is_var & (args >= base)

    pos = torch.arange(P, device=device, dtype=torch.long).view(1, P).expand(N, P)

    # Keep non-new positions isolated so only true new variables group together.
    nonnew_key_base = int(runtime_var_end_index) + 1
    key = torch.where(is_new, args, pos + nonnew_key_base)
    sort_key = key * (P + 1) + pos

    order = torch.argsort(sort_key, dim=1, stable=True)
    key_sorted = torch.gather(key, 1, order)
    pos_sorted = torch.gather(pos, 1, order)
    is_new_sorted = torch.gather(is_new, 1, order)
    args_sorted = torch.gather(args, 1, order)

    # Segment groups by grouped key.
    seg_start = torch.cat(
        (
            torch.ones((N, 1), dtype=torch.bool, device=device),
            key_sorted[:, 1:] != key_sorted[:, :-1],
        ),
        dim=1,
    )
    group_id = torch.cumsum(seg_start.long(), dim=1) - 1  # [N, P]

    # Group statistics in fixed-shape [N, P] slot space.
    first_pos_by_group = torch.full((N, P), P, dtype=torch.long, device=device)
    first_pos_by_group.scatter_reduce_(1, group_id, pos_sorted, reduce="amin", include_self=True)
    used_group = first_pos_by_group < P

    group_is_new = torch.zeros((N, P), dtype=torch.long, device=device)
    group_is_new.scatter_reduce_(1, group_id, is_new_sorted.long(), reduce="amax", include_self=True)
    group_is_new = group_is_new.bool() & used_group

    # Rank groups by first appearance position and count only new-variable groups.
    group_order = torch.argsort(first_pos_by_group, dim=1, stable=True)
    group_is_new_in_order = torch.gather(group_is_new, 1, group_order)
    group_rank_in_order = torch.cumsum(group_is_new_in_order.long(), dim=1) - 1

    rank_by_group = torch.zeros((N, P), dtype=torch.long, device=device)
    rank_by_group.scatter_(1, group_order, group_rank_in_order)
    rank_sorted = torch.gather(rank_by_group, 1, group_id)

    new_id_sorted = base.expand(-1, P) + rank_sorted
    args_out_sorted = torch.where(is_new_sorted, new_id_sorted, args_sorted)

    args_out = torch.empty_like(args)
    args_out.scatter_(1, order, args_out_sorted)

    std_states = flat_states.clone()
    std_states[:, :, 1:3] = args_out.view(N, M, 2)
    std_derived = std_states.view(B, K, M, 3)

    vars_per_state = group_is_new.sum(dim=1).long()
    next_end_per_state = base.view(N) + vars_per_state

    next_end_B = next_var_indices.clone()
    next_end_B.scatter_reduce_(0, owners, next_end_per_state, reduce="amax", include_self=True)
    new_next_var = torch.maximum(next_var_indices, next_end_B)

    is_var_out = (args_out > constant_no) & (args_out != padding_idx)
    lower_ok = ((~is_var_out) | (args_out >= (constant_no + 1))).all()
    upper_ok = ((~is_var_out) | (args_out <= runtime_var_end_index)).all()
    next_ok = (new_next_var <= runtime_var_end_index).all()
    torch._assert_async(lower_ok, "Canonical standardization produced ID below runtime range")
    torch._assert_async(upper_ok, "Canonical standardization produced ID above runtime range")
    torch._assert_async(next_ok, "Canonical standardization advanced next-var beyond runtime range")

    return std_derived, new_next_var


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_standardize_fn(
    config: StandardizationConfig,
    device: torch.device,
) -> Callable:
    """Build a (optionally compiled) output-standardization callable.

    Args:
        config: standardization configuration (mode, constants, etc.).
        device: target device (compile only on CUDA).

    Returns:
        Callable ``(states [B,K,M,3], counts [B], nv [B], inp [B,?,3])``
        ``→ (std_states [B,K,M,3], new_nv [B])``.
    """
    rve = int(config.runtime_var_end_index)
    constant_no = config.constant_no
    padding_idx = config.padding_idx

    if config.mode == "canonical":
        def _fn(states: Tensor, counts: Tensor, nv: Tensor, inp: Tensor):
            s, n = standardize_vars_canonical(
                states, counts, nv, constant_no, rve, padding_idx)
            return s.clone(), n.clone()
    elif config.mode == "offset":
        extra = config.body_width + 2
        enforce_runtime_range = config.enforce_runtime_range
        def _fn(states: Tensor, counts: Tensor, nv: Tensor, inp: Tensor):
            s, n = standardize_vars_offset(
                states, counts, nv, constant_no, rve, padding_idx,
                input_states=inp, extra_new_vars=extra,
                enforce_runtime_range=enforce_runtime_range,
                out_of_place=True)
            return s.clone(), n.clone()
    else:
        raise ValueError(
            f"Unknown standardization mode '{config.mode}'. Expected 'offset' or 'canonical'.")

    if device.type == "cuda":
        _fn = torch.compile(_fn, mode=config.compile_mode, fullgraph=True)
    return _fn
