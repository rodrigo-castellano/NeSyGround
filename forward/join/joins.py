"""PS/PO atom index + expansion primitives for the staged ragged join.

Hash encoding throughout: ``h = pred * E^2 + subj * E + obj``. A "PS" index is
keyed by ``pred * E + subj`` (enumerate objects); a "PO" index by
``pred * E + obj`` (enumerate subjects). ``_build_atom_index`` turns a sorted
hash tensor into CSR-style ``(offsets, vals)`` pairs; the ``_*_expand`` helpers
do the per-key ragged lookups that the stage loop joins on.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _build_atom_index(
    hashes: Tensor, E: int, P: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Hash encoding: pred * E^2 + subj * E + obj.
    Returns ps_offsets [P*E+1], ps_vals, po_offsets [P*E+1], po_vals.
    """
    dev = hashes.device
    E2 = E * E
    N = hashes.numel()
    empty_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
    empty_val = torch.zeros(0, dtype=torch.long, device=dev)
    if N == 0:
        return empty_off, empty_val, empty_off.clone(), empty_val.clone()

    preds = hashes // E2
    rem = hashes % E2
    subjs = rem // E
    objs = rem % E
    ones = torch.ones(N, dtype=torch.long, device=dev)

    ps_keys = preds * E + subjs
    ps_sort = torch.argsort(ps_keys, stable=True)
    ps_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
    ps_off.scatter_add_(0, ps_keys[ps_sort] + 1, ones)
    ps_off = torch.cumsum(ps_off, 0)

    po_keys = preds * E + objs
    po_sort = torch.argsort(po_keys, stable=True)
    po_off = torch.zeros(P * E + 1, dtype=torch.long, device=dev)
    po_off.scatter_add_(0, po_keys[po_sort] + 1, ones)
    po_off = torch.cumsum(po_off, 0)

    return ps_off, objs[ps_sort], po_off, subjs[po_sort]


def _pred_pairs_from_ps(
    pred_idx: int, ps_off: Tensor, ps_vals: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """All (subj, obj) pairs for pred_idx from a PS index."""
    dev = ps_off.device
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if ps_vals.numel() == 0:
        return empty, empty
    base = pred_idx * E
    counts = ps_off[base + 1: base + E + 1] - ps_off[base: base + E]
    total = int(counts.sum().item())
    if total == 0:
        return empty, empty
    subjs = torch.repeat_interleave(
        torch.arange(E, dtype=torch.long, device=dev), counts)
    objs = ps_vals[int(ps_off[base].item()): int(ps_off[base + E].item())]
    return subjs, objs


def _ps_expand(
    pred_idx: int, key_vals: Tensor,
    ps_off: Tensor, ps_vals: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """PS lookup: for each subject key, enumerate all objects."""
    dev = ps_off.device
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if ps_vals.numel() == 0 or key_vals.numel() == 0:
        return empty, empty
    N = key_vals.shape[0]
    keys = (pred_idx * E + key_vals).clamp(0, max(ps_off.shape[0] - 2, 0))
    starts = ps_off[keys]
    counts = (ps_off[keys + 1] - starts).clamp(min=0)
    total = int(counts.sum().item())
    if total == 0:
        return empty, empty
    row_ids = torch.repeat_interleave(
        torch.arange(N, dtype=torch.long, device=dev), counts)
    cumcnt = counts.cumsum(0)
    k_idx = torch.arange(total, dtype=torch.long, device=dev) - \
            torch.repeat_interleave(cumcnt - counts, counts)
    val_abs = (starts[row_ids] + k_idx).clamp(0, ps_vals.numel() - 1)
    return row_ids, ps_vals[val_abs]


def _po_expand(
    pred_idx: int, key_vals: Tensor,
    po_off: Tensor, po_vals: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """PO lookup: for each object key, enumerate all subjects."""
    dev = po_off.device
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if po_vals.numel() == 0 or key_vals.numel() == 0:
        return empty, empty
    N = key_vals.shape[0]
    keys = (pred_idx * E + key_vals).clamp(0, max(po_off.shape[0] - 2, 0))
    starts = po_off[keys]
    counts = (po_off[keys + 1] - starts).clamp(min=0)
    total = int(counts.sum().item())
    if total == 0:
        return empty, empty
    row_ids = torch.repeat_interleave(
        torch.arange(N, dtype=torch.long, device=dev), counts)
    cumcnt = counts.cumsum(0)
    k_idx = torch.arange(total, dtype=torch.long, device=dev) - \
            torch.repeat_interleave(cumcnt - counts, counts)
    val_abs = (starts[row_ids] + k_idx).clamp(0, po_vals.numel() - 1)
    return row_ids, po_vals[val_abs]


def _ps_expand_combined(
    pred_idx: int, key_vals: Tensor,
    base_ps_off: Tensor, base_ps_vals: Tensor,
    prov_ps_off: Tensor, prov_ps_vals: Tensor,
    E: int,
) -> Tuple[Tensor, Tensor]:
    """PS lookup in base ∪ provable."""
    dev = base_ps_off.device
    ri_b, ov_b = _ps_expand(pred_idx, key_vals, base_ps_off, base_ps_vals, E)
    ri_p, ov_p = _ps_expand(pred_idx, key_vals, prov_ps_off, prov_ps_vals, E)
    if ri_b.numel() == 0 and ri_p.numel() == 0:
        return (torch.zeros(0, dtype=torch.long, device=dev),
                torch.zeros(0, dtype=torch.long, device=dev))
    if ri_b.numel() == 0:
        return ri_p, ov_p
    if ri_p.numel() == 0:
        return ri_b, ov_b
    return torch.cat([ri_b, ri_p]), torch.cat([ov_b, ov_p])


def _po_expand_combined(
    pred_idx: int, key_vals: Tensor,
    base_po_off: Tensor, base_po_vals: Tensor,
    prov_po_off: Tensor, prov_po_vals: Tensor,
    E: int,
) -> Tuple[Tensor, Tensor]:
    """PO lookup in base ∪ provable."""
    dev = base_po_off.device
    ri_b, sv_b = _po_expand(pred_idx, key_vals, base_po_off, base_po_vals, E)
    ri_p, sv_p = _po_expand(pred_idx, key_vals, prov_po_off, prov_po_vals, E)
    if ri_b.numel() == 0 and ri_p.numel() == 0:
        return (torch.zeros(0, dtype=torch.long, device=dev),
                torch.zeros(0, dtype=torch.long, device=dev))
    if ri_b.numel() == 0:
        return ri_p, sv_p
    if ri_p.numel() == 0:
        return ri_b, sv_b
    return torch.cat([ri_b, ri_p]), torch.cat([sv_b, sv_p])


__all__ = [
    "_build_atom_index", "_pred_pairs_from_ps",
    "_ps_expand", "_po_expand", "_ps_expand_combined", "_po_expand_combined",
]
