"""Static-buffer rule-grounding accumulation — fullgraph-compile-safe.

Default ``BCGrounder`` accumulates per-rule groundings via Python list
``.append`` per step; that path is fast and memory-cheap but
dynamo-untraceable. This module provides the opt-in alternative:
pre-allocated ``[D, max_per_step, ...]`` tensor buffers written by slot
index, plus a shared finalize that takes flat tensors and is reused by
both paths.

Opt in via ``BCGrounder(..., static_buffers=True)``. Refused for
``flat_intermediate=True`` (enum-flat); incompatible with its
``torch.nonzero`` dynamic shapes.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def alloc_static_buffers(
    *, depth: int, max_per_step: int, M: int, pad: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Pre-allocate per-step write buffers ``[D, max_per_step, ...]``.

    ``rule_idx`` defaults to -1 (invalid sentinel); ``head`` / ``body``
    to ``pad``. Re-used across forwards via ``buf[k].fill_(...)``.
    """
    return {
        "rule_idx": torch.full(
            (depth, max_per_step), -1, dtype=torch.long, device=device,
        ),
        "head": torch.full(
            (depth, max_per_step, 3), pad, dtype=torch.long, device=device,
        ),
        "body": torch.full(
            (depth, max_per_step, M, 3), pad, dtype=torch.long, device=device,
        ),
    }


def collect_step(
    buf: Dict[str, Tensor],
    d: int,
    rule_idx: Tensor,        # [T] long — full unfiltered
    head: Tensor,             # [T, 3] long
    body_sorted: Tensor,      # [T, M, 3] long
    valid: Tensor,            # [T] bool — invalid → rule_idx -1 sentinel
    pad: int = 0,
) -> None:
    """Write step ``d`` into the static buffers, padded to full slot.

    Pads input tensors up to ``max_per_step`` so the write is
    full-slot (no data-dependent slicing). Invalid rows are tagged
    ``rule_idx = -1``; finalize remaps them out of range and drops
    them via the per-rule bucket loop.
    """
    max_per_step = buf["rule_idx"].shape[1]
    T = rule_idx.shape[0]
    M = body_sorted.shape[1]
    pad_n = max_per_step - T

    rule_idx_marked = torch.where(
        valid, rule_idx, torch.full_like(rule_idx, -1),
    )
    buf["rule_idx"][d].copy_(torch.cat(
        [rule_idx_marked, rule_idx_marked.new_full((pad_n,), -1)], dim=0))
    buf["head"][d].copy_(torch.cat(
        [head, head.new_full((pad_n, 3), pad)], dim=0))
    buf["body"][d].copy_(torch.cat(
        [body_sorted, body_sorted.new_full((pad_n, M, 3), pad)], dim=0))


def finalize_flat(
    rule_idx: Tensor,         # [T] long — flat per-firing rule indices
    head: Tensor,              # [T, 3] long — flat head atoms
    body: Tensor,              # [T, M, 3] long — flat body atoms (sorted)
    *,
    num_rules: int,
    pad: int,
):
    """Tensor-only dedup → per-rule bucket → ``RuleGroundings``.

    Used by both the static-buffer path (after ``reshape(-1, ...)``
    over the ``[D, max_per_step]`` buffers) and the eager Python-list
    path (after ``torch.cat`` over the per-step lists). No host syncs
    (``bool``, ``.item()``); invalid rows (``rule_idx < 0``) get
    remapped to ``num_rules`` so the per-rule bucket loop discards
    them via ``r in [0, num_rules)``.
    """
    from grounder.types import RuleGroundings

    T = rule_idx.size(0)
    M = body.size(1)

    rule_idx_safe = torch.where(
        rule_idx < 0, torch.full_like(rule_idx, num_rules), rule_idx,
    )
    # Encode each row as a single comparable tensor for unique:
    # [rule, head[3], body[M*3]].
    combined = torch.cat([
        rule_idx_safe.unsqueeze(-1),
        head.long(),
        body.long().reshape(T, M * 3),
    ], dim=-1)
    uniq = torch.unique(combined, dim=0)
    u_rule = uniq[:, 0].long()
    u_head = uniq[:, 1:4].long()
    u_body = uniq[:, 4:].reshape(-1, M, 3).long()

    # Atom table = union of (head, body) atoms (one ``torch.unique``).
    all_atoms = torch.cat([u_head.unsqueeze(1), u_body], dim=1)  # [U, M+1, 3]
    atom_table, inverse = torch.unique(
        all_atoms.reshape(-1, 3), dim=0, return_inverse=True,
    )
    inverse = inverse.reshape(-1, M + 1)
    head_atom_idx = inverse[:, 0]
    body_atom_idx = inverse[:, 1:]

    # Bucket per rule. ``num_rules`` is a Python constant — Dynamo
    # unrolls. We don't compact body width (no per-rule .item() sync);
    # padding atoms map to slot 0 (``populate_query_pool_idx`` reserves
    # it).
    A_in: Dict[int, Tensor] = {}
    A_out: Dict[int, Tensor] = {}
    for r in range(num_rules):
        mask = (u_rule == r)
        A_in[r] = body_atom_idx[mask]
        A_out[r] = head_atom_idx[mask].unsqueeze(-1)

    return RuleGroundings(
        atom_table=atom_table.contiguous(),
        A_in=A_in, A_out=A_out,
        num_atoms=int(atom_table.size(0)),
        num_rules=num_rules,
    )


__all__ = ["alloc_static_buffers", "collect_step", "finalize_flat"]
