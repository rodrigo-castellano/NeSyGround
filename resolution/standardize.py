"""Rename a derived state's new template vars so they don't collide with vars
already live in the batch. Two algorithms:

  offset    — shift each batch's vars by ``next_var − min_var_in`` (cheap).
  canonical — renumber by first appearance (fullgraph + CUDA-graph safe).

Pure (torch only); compilation is the ExecStrategy's job. Bit-exact (fingerprint-
locked): the offset, the ``id>constant_no & id!=pad`` var test, the canonical
renumbering.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

# Hot-path var-id range guards: each runs a `.all()` reduction + a launched
# assert kernel per grounding step. Debug-only — off by default (set
# GROUNDER_STD_ASSERTS=1 to re-enable). Output is unaffected either way.
_STD_ASSERTS = os.environ.get("GROUNDER_STD_ASSERTS") == "1"


@dataclass(frozen=True)
class StandardizationConfig:
    """Bundled params for ``build_standardize_fn`` (one immutable config)."""
    mode: str                   # 'offset' | 'canonical'
    constant_no: int
    runtime_var_end_index: int
    padding_idx: int
    body_width: int
    enforce_runtime_range: bool = False
    compile_mode: Optional[str] = None   # accepted for consumer compat; no-op (ExecStrategy owns compile)


def standardize_vars_offset(
    states: Tensor,                       # [B, K, M, 3]
    counts: Tensor,                       # [B] (signature compat; unused)
    next_var_indices: Tensor,             # [B]
    constant_no: int,
    runtime_var_end_index: Optional[int],
    padding_idx: int,
    input_states: Optional[Tensor] = None,  # [B, ?, 3] parents
    extra_new_vars: int = 15,
    enforce_runtime_range: bool = False,
    out_of_place: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Offset-shift derived-state variables past the live range.

    The offset is ``next_var − min_var_in`` per batch (0 when the parents have
    no variables). Returns ``(standardized [B,K,M,3], new_next_var [B])``.
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
        large_t = torch.tensor(LARGE, dtype=in_args.dtype, device=device)
        min_var_in = torch.where(is_var_in, in_args, large_t).amin(dim=(-1, -2))
        max_var_in = torch.where(is_var_in, in_args, torch.zeros_like(in_args)).amax(dim=(-1, -2))
        has_input_vars = min_var_in < LARGE

    offset = torch.where(has_input_vars, next_var_indices - min_var_in,
                         torch.zeros_like(next_var_indices))

    args = states[:, :, :, 1:3]
    is_var_out = (args > constant_no) & (args != pad)
    offset_exp = offset.view(B, 1, 1, 1).expand(-1, K, M, 2)
    std_args = torch.where(is_var_out, args + offset_exp, args)

    if _STD_ASSERTS and enforce_runtime_range and runtime_var_end_index is not None:
        lo = ((~is_var_out) | (std_args >= (constant_no + 1))).all()
        hi = ((~is_var_out) | (std_args <= runtime_var_end_index)).all()
        torch._assert_async(lo, "standardize_offset: var id below runtime range")
        torch._assert_async(hi, "standardize_offset: var id above runtime range")

    if out_of_place:
        standardized = states.clone()
        standardized[:, :, :, 1:3] = std_args
    else:
        states[:, :, :, 1:3] = std_args
        standardized = states

    max_in_shifted = torch.where(has_input_vars, max_var_in + offset,
                                 torch.zeros_like(max_var_in))
    max_gen_shifted = next_var_indices + extra_new_vars
    new_next_var = torch.maximum(max_in_shifted, max_gen_shifted) + 1

    if enforce_runtime_range and runtime_var_end_index is not None:
        if _STD_ASSERTS:
            torch._assert_async((new_next_var <= runtime_var_end_index).all(),
                                "standardize_offset: next-var beyond runtime range")
    return standardized, new_next_var


def standardize_vars_canonical(
    states: Tensor,                       # [B, K, M, 3]
    counts: Tensor,                       # [B] (signature compat; unused)
    next_var_indices: Tensor,             # [B]
    constant_no: int,
    runtime_var_end_index: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Graph-safe canonical renumbering by first appearance (fixed-shape ops).

    Renumbers each derived state's new variables (those ``>= owner next_var``) to
    a contiguous block starting at ``next_var``, ordered by first appearance.
    """
    if states.numel() == 0:
        return states, next_var_indices
    device = states.device
    B, K, M, _ = states.shape
    if B == 0 or K == 0:
        return states, next_var_indices

    N, P = B * K, M * 2
    flat = states.view(N, M, 3)
    owners = torch.arange(B, device=device).repeat_interleave(K)
    base = next_var_indices.index_select(0, owners).view(N, 1)

    args = flat[:, :, 1:3].reshape(N, P)
    is_var = (args > constant_no) & (args != padding_idx)
    is_new = is_var & (args >= base)               # only renumber at/above next-var
    pos = torch.arange(P, device=device, dtype=torch.long).view(1, P).expand(N, P)

    # Isolate non-new positions so only true new vars group together.
    nonnew_base = int(runtime_var_end_index) + 1
    key = torch.where(is_new, args, pos + nonnew_base)
    sort_key = key * (P + 1) + pos

    order = torch.argsort(sort_key, dim=1, stable=True)
    key_s = torch.gather(key, 1, order)
    pos_s = torch.gather(pos, 1, order)
    is_new_s = torch.gather(is_new, 1, order)
    args_s = torch.gather(args, 1, order)

    seg_start = torch.cat(
        (torch.ones((N, 1), dtype=torch.bool, device=device),
         key_s[:, 1:] != key_s[:, :-1]), dim=1)
    group_id = torch.cumsum(seg_start.long(), dim=1) - 1

    first_pos = torch.full((N, P), P, dtype=torch.long, device=device)
    first_pos.scatter_reduce_(1, group_id, pos_s, reduce="amin", include_self=True)
    used = first_pos < P
    group_new = torch.zeros((N, P), dtype=torch.long, device=device)
    group_new.scatter_reduce_(1, group_id, is_new_s.long(), reduce="amax", include_self=True)
    group_new = group_new.bool() & used

    group_order = torch.argsort(first_pos, dim=1, stable=True)
    group_new_ord = torch.gather(group_new, 1, group_order)
    rank_ord = torch.cumsum(group_new_ord.long(), dim=1) - 1
    rank_by_group = torch.zeros((N, P), dtype=torch.long, device=device)
    rank_by_group.scatter_(1, group_order, rank_ord)
    rank_s = torch.gather(rank_by_group, 1, group_id)

    new_id_s = base.expand(-1, P) + rank_s
    args_out_s = torch.where(is_new_s, new_id_s, args_s)
    args_out = torch.empty_like(args)
    args_out.scatter_(1, order, args_out_s)

    std = flat.clone()
    std[:, :, 1:3] = args_out.view(N, M, 2)
    std_derived = std.view(B, K, M, 3)

    vars_per_state = group_new.sum(dim=1).long()
    next_end = base.view(N) + vars_per_state
    next_end_B = next_var_indices.clone()
    next_end_B.scatter_reduce_(0, owners, next_end, reduce="amax", include_self=True)
    new_next_var = torch.maximum(next_var_indices, next_end_B)

    if _STD_ASSERTS:
        is_var_out = (args_out > constant_no) & (args_out != padding_idx)
        torch._assert_async(((~is_var_out) | (args_out >= (constant_no + 1))).all(),
                            "standardize_canonical: var id below runtime range")
        torch._assert_async(((~is_var_out) | (args_out <= runtime_var_end_index)).all(),
                            "standardize_canonical: var id above runtime range")
        torch._assert_async((new_next_var <= runtime_var_end_index).all(),
                            "standardize_canonical: next-var beyond runtime range")
    return std_derived, new_next_var


def build_standardize_fn(config: StandardizationConfig) -> Callable:
    """Return the (eager) standardize callable for ``config.mode``.

    ``(states[B,K,M,3], counts[B], nv[B], inp[B,?,3]) -> (std[B,K,M,3], nv[B])``.
    Compilation is the ExecStrategy's job (the single compile site), not here.
    """
    rve = int(config.runtime_var_end_index)
    c_no, pad = config.constant_no, config.padding_idx

    if config.mode == "canonical":
        def _fn(states, counts, nv, inp):
            s, n = standardize_vars_canonical(states, counts, nv, c_no, rve, pad)
            return s.clone(), n.clone()
        return _fn
    if config.mode == "offset":
        extra = config.body_width + 2
        enforce = config.enforce_runtime_range
        def _fn(states, counts, nv, inp):
            s, n = standardize_vars_offset(states, counts, nv, c_no, rve, pad,
                                           input_states=inp, extra_new_vars=extra,
                                           enforce_runtime_range=enforce, out_of_place=True)
            return s.clone(), n.clone()
        return _fn
    raise ValueError(f"Unknown standardization mode {config.mode!r}. Expected 'offset' or 'canonical'.")


__all__ = [
    "StandardizationConfig", "standardize_vars_offset",
    "standardize_vars_canonical", "build_standardize_fn",
]
