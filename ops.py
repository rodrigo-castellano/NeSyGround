"""Operations every grounder shares: atom keys, membership, distinct rows, the least model of a ground program.

A key packs an atom ``(p, a1, ..., a_n)`` — optionally with its pool as the most significant digit — into one int64,
``((pool * base + p) * base + a1) * base + ...``; ``base`` exceeds every id, so keys order atoms lexicographically.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor


def key(atoms: Tensor, base: int, pool: Optional[Tensor] = None) -> Tensor:
    """``[..., W] -> [...]`` int64 keys (``pool [...]``: the most significant digit)."""
    atoms = atoms.long()
    k = atoms[..., 0] if pool is None else pool * base + atoms[..., 0]
    for j in range(1, atoms.shape[-1]):
        k = k * base + atoms[..., j]
    return k


def decode(keys: Tensor, base: int, W: int = 3, pool: bool = False):
    """The atoms ``[..., W]`` of ``keys`` (and their pools, with ``pool``)."""
    cols = []
    for _ in range(W):
        cols.append(keys % base)
        keys = keys // base
    atoms = torch.stack(cols[::-1], -1)
    return (keys, atoms) if pool else atoms


def unique_rows(cols: List[Tensor], radix: List[int]) -> List[Tensor]:
    """The distinct rows of the digit columns ``cols`` (``0 <= cols[i] < radix[i]``), sorted lexicographically: the
    digits packed into as few int64 keys as fit, radix-sorted from the least significant key (stable), and the first
    row of each run kept."""
    keys, k, span = [], None, 1
    for c, r in zip(cols, radix):
        if k is not None and span * r >= 1 << 62:
            keys.append(k)
            k, span = None, 1
        k, span = (c if k is None else k * r + c), span * r
    keys.append(k)
    order = torch.arange(cols[0].shape[0], device=cols[0].device)
    for k in reversed(keys):
        order = order[torch.sort(k[order], stable=True)[1]]
    first = torch.zeros_like(order, dtype=torch.bool)
    first[:1] = True
    for k in keys:
        k = k[order]
        first[1:] |= k[1:] != k[:-1]
    keep = order[first]
    return [c[keep] for c in cols]


__all__ = ["key", "decode", "unique_rows"]


def groups(count: Tensor, limit: int) -> List[int]:
    """The ends of consecutive groups of rows with ``count`` each, a group's rows but its last starting within
    ``limit`` of its first (so at most ``limit`` in all, but for a row with more)."""
    if count.numel() == 0:
        return []
    if int(count.sum()) <= limit:
        return [count.numel()]
    return torch.unique_consecutive((count.cumsum(0) - count) // limit, return_counts=True)[1].cumsum(0).tolist()
