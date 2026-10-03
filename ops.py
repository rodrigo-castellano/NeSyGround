"""Operations every grounder shares: atom keys, membership, distinct rows, the least model of a ground program.

A key packs an atom ``(p, a1, ..., a_n)`` — optionally with its pool as the most significant digit — into one int64,
``((pool * base + p) * base + a1) * base + ...``; ``base`` exceeds every id, so keys order atoms lexicographically.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grounder.types import Groundings


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




def groups(count: Tensor, limit: int) -> List[int]:
    """The ends of consecutive groups of rows with ``count`` each, a group's rows but its last starting within
    ``limit`` of its first (so at most ``limit`` in all, but for a row with more)."""
    if count.numel() == 0:
        return []
    if int(count.sum()) <= limit:
        return [count.numel()]
    return torch.unique_consecutive((count.cumsum(0) - count) // limit, return_counts=True)[1].cumsum(0).tolist()


def canonical(rule: Tensor, head: Tensor, body: Tensor, pool: Tensor, qs: Tensor, qpool: Tensor, S: int, R: int,
              base: int, pad: int) -> Groundings:
    """The canonical ``Groundings`` of ``S`` pools from raw groundings ``(rule [T], head [T, 3], body [T, M, 3],
    pool [T])`` (repeats merge) and the queries ``qs [S * Q, 3]`` of pools ``qpool``: each pool's atoms are its
    groundings' atoms sorted by key, then its queries that are none of them, sorted; its groundings are distinct,
    sorted by rule, then head and body rows."""
    dev, Q, M = qs.device, qs.shape[0] // max(S, 1), body.shape[1]
    # unique atoms (sorted by pool, then key), unique groundings (sorted by pool, rule, atoms)
    akey, ainv = torch.unique(key(torch.cat([head.unsqueeze(1), body], 1), base, pool.unsqueeze(1)),
                              return_inverse=True)                                  # ainv [T, 1 + M]
    apool, table = decode(akey, base, pool=True)
    start = torch.searchsorted(apool, torch.arange(S + 1, device=dev))              # each pool's first atom
    local = ainv - start[pool].unsqueeze(1)
    A = max(int(start.diff().max()), 1) if S > 1 else max(int(akey.shape[0]), 1)     # atoms per pool
    fpool, rule, *local = unique_rows([pool, rule, *local.unbind(1)], [S, R] + [A] * (M + 1))
    idx = torch.stack(local, 1)                                                     # [N, 1 + M] rows local to pools

    # each pool's queries that are none of its atoms, after them
    qk = key(qs, base, qpool)
    at = torch.searchsorted(akey, qk).clamp(max=max(akey.numel() - 1, 0))
    in_pool = (akey[at] == qk) if akey.numel() else torch.zeros_like(qk, dtype=torch.bool)
    nkey = torch.unique(qk[~in_pool])
    npool, ntable = decode(nkey, base, pool=True)
    nstart = torch.searchsorted(npool, torch.arange(S + 1, device=dev))
    n_a, n_n = start.diff(), nstart.diff()
    atom_offsets = torch.zeros(S + 1, dtype=torch.long, device=dev)
    atom_offsets[1:] = (n_a + n_n).cumsum(0)
    arow = torch.arange(akey.shape[0], device=dev)
    nrow = torch.arange(nkey.shape[0], device=dev)
    atoms = torch.empty(akey.shape[0] + nkey.shape[0], 3, dtype=torch.long, device=dev)
    atoms[arow - start[apool] + atom_offsets[apool]] = table
    atoms[n_a[npool] + nrow - nstart[npool] + atom_offsets[npool]] = ntable
    query = torch.where(in_pool, at - start[qpool], n_a[qpool] + torch.searchsorted(nkey, qk) - nstart[qpool])
    query = (query + atom_offsets[qpool]).view(S, Q)
    rows = idx + atom_offsets[fpool].unsqueeze(1)
    offsets = torch.zeros(S, R + 1, dtype=torch.long, device=dev)
    offsets[:, 1:] = torch.bincount(fpool * R + rule, minlength=S * R).view(S, R).cumsum(1)
    offsets = offsets + torch.cat([offsets.new_zeros(1), offsets[:-1, -1].cumsum(0)]).unsqueeze(1)
    return Groundings(atoms, rule, rows[:, 0], rows[:, 1:], table[idx[:, 1:] + start[fpool].unsqueeze(1), 0] != pad,
                      offsets, atom_offsets, query)


def concat(parts: List[Groundings]) -> Groundings:
    """The pools of ``parts``, in order, as one ``Groundings``."""
    atoms_at = torch.cumsum(torch.tensor([0] + [len(p.atoms) for p in parts[:-1]]), 0).tolist()
    rows_at = torch.cumsum(torch.tensor([0] + [len(p.rule) for p in parts[:-1]]), 0).tolist()
    return Groundings(
        torch.cat([p.atoms for p in parts]), torch.cat([p.rule for p in parts]),
        torch.cat([p.head + a for p, a in zip(parts, atoms_at)]), torch.cat([p.body + a for p, a in zip(parts, atoms_at)]),
        torch.cat([p.body_mask for p in parts]), torch.cat([p.offsets + n for p, n in zip(parts, rows_at)]),
        torch.cat([parts[0].atom_offsets[:1]] + [p.atom_offsets[1:] + a for p, a in zip(parts, atoms_at)]),
        torch.cat([p.query + a for p, a in zip(parts, atoms_at)]))


__all__ = ["key", "decode", "unique_rows", "groups", "canonical", "concat"]
