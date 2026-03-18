"""Prune known ground facts from proof goals.

Removes atoms from candidates that are already known facts in the fact index.
When all atoms are pruned, the candidate is a complete proof.

Extracted from bc/common.py — can be used as a hook 4 (filter_step) filter.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.data.fact_index import fact_contains


def prune_ground_facts(
    candidates: Tensor,         # [..., M, 3] — any leading dims (e.g. [B,K,M,3])
    valid_mask: Tensor,         # [...] matching leading dims
    fact_hashes: Tensor,        # [F] sorted fact hashes
    pack_base: int,
    constant_no: int,
    padding_idx: int,
    true_pred_idx: Optional[int] = None,
    excluded_queries: Optional[Tensor] = None,  # [B, 1, 3]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Remove known ground facts from candidates (fixed shape, fully vectorized).

    Works on any 4D state tensor [..., M, 3] (e.g. [B,K,M,3] or [B,S,G,3]).
    Also treats True predicate atoms as "facts" (proof indicators).

    Args:
        candidates: [..., M, 3] candidate derived states
        valid_mask: [...] which candidates are valid
        fact_hashes: [F] sorted int64 fact hashes for membership testing
        pack_base: packing base for hash computation
        constant_no: highest constant index
        padding_idx: padding value
        true_pred_idx: predicate index for True atoms (treated as resolved)
        excluded_queries: [B, 1, 3] atoms NOT to prune (cycle prevention)

    Returns:
        pruned_states: [..., M, 3] with facts removed
        pruned_counts: [...] new atom counts per candidate
        is_proof: [...] whether candidate became empty (proof found)
    """
    B, K, M, _ = candidates.shape
    pad = padding_idx

    # Check which atoms are ground facts
    preds = candidates[:, :, :, 0]               # [B, K, M]
    args = candidates[:, :, :, 1:3]              # [B, K, M, 2]

    valid_atom = (preds != pad)                  # [B, K, M]
    is_ground = (args <= constant_no).all(dim=-1)  # [B, K, M]
    ground_atoms = valid_atom & is_ground        # [B, K, M]

    # Check ALL atoms against fact hashes (fully vectorized)
    flat_atoms = candidates.reshape(-1, 3)        # [B*K*M, 3]
    is_fact_flat = fact_contains(flat_atoms, fact_hashes, pack_base)
    is_fact = is_fact_flat.reshape(B, K, M)

    # Only mark as fact if it was actually a ground atom
    is_fact = is_fact & ground_atoms

    # Handle Exclusion: Keep excluded atoms (don't prune them)
    if excluded_queries is not None:
        excl_first = excluded_queries[:, 0, :]  # [B, 3]
        excl_exp = excl_first.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
        is_excluded_atom = (candidates == excl_exp).all(dim=-1) & ground_atoms
        is_fact = is_fact & ~is_excluded_atom

    # Also treat True predicate atoms as "facts" (proof indicators)
    if true_pred_idx is not None:
        is_true_pred = (preds == true_pred_idx)  # [B, K, M]
        is_fact = is_fact | is_true_pred

    # Atoms to keep: valid AND NOT a known fact
    keep_atom = valid_atom & ~is_fact  # [B, K, M]

    # Compute new counts
    pruned_counts = keep_atom.sum(dim=-1)  # [B, K]

    # Detect proofs: candidate with zero remaining atoms
    is_proof = (pruned_counts == 0) & valid_mask

    # Mask out removed atoms (gaps remain, downstream handles via compact_atoms)
    pad_t = torch.tensor(pad, dtype=candidates.dtype, device=candidates.device)
    pruned_states = torch.where(
        keep_atom.unsqueeze(-1),
        candidates,
        pad_t,
    )

    return pruned_states, pruned_counts, is_proof
