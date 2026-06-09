"""Prune known ground facts from proof goals (search-time filter).

Removes atoms from candidate states that are already known facts in the fact index
(compressed-depth semantics when ``prune_facts=True``). Used by the engine
postprocess step.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from grounder.data.fact_index import fact_contains


def prune_ground_facts(
    candidates: Tensor,         # [..., M, 3]
    fact_hashes: Tensor,        # [F]
    pack_base: int,
    constant_no: int,
    padding_idx: int,
    excluded_queries: Optional[Tensor] = None,
) -> Tensor:
    """Remove known ground facts from candidates → pruned states (fixed shape)."""
    B, K, M, _ = candidates.shape
    pad = padding_idx

    preds = candidates[:, :, :, 0]
    args = candidates[:, :, :, 1:3]

    valid_atom = (preds != pad)
    is_ground = (args <= constant_no).all(dim=-1)
    ground_atoms = valid_atom & is_ground

    flat_atoms = candidates.reshape(-1, 3)
    is_fact_flat = fact_contains(flat_atoms, fact_hashes, pack_base)
    is_fact = is_fact_flat.reshape(B, K, M)
    is_fact = is_fact & ground_atoms

    if excluded_queries is not None:
        excl_first = excluded_queries[:, 0, :]
        excl_exp = excl_first.unsqueeze(1).unsqueeze(1)
        is_excluded_atom = (candidates == excl_exp).all(dim=-1) & ground_atoms
        is_fact = is_fact & ~is_excluded_atom

    keep_atom = valid_atom & ~is_fact
    pad_t = torch.tensor(pad, dtype=candidates.dtype, device=candidates.device)
    pruned_states = torch.where(keep_atom.unsqueeze(-1), candidates, pad_t)
    return pruned_states


__all__ = ["prune_ground_facts"]
