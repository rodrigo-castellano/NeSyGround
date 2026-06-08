"""Filters: ``fp_batch`` (Kleene T_P fixed-point — the BC search filter) and
``prune_facts``. ``check_in_fp_global`` is a closure-membership helper for the
soft-scoring path (``nesy/soft.py``): which atoms are in a precomputed FC
provable set (sorted hashes)."""
import torch
from torch import Tensor

__all__ = ["check_in_fp_global"]


def check_in_fp_global(atom_hashes: Tensor, fp_global_hashes: Tensor) -> Tensor:
    """Bool mask (shape of ``atom_hashes``) of membership in the sorted
    ``fp_global_hashes`` via searchsorted."""
    flat = atom_hashes.reshape(-1)
    pos = torch.searchsorted(fp_global_hashes, flat).clamp(
        max=fp_global_hashes.size(0) - 1)
    return (fp_global_hashes[pos] == flat).view(atom_hashes.shape)
