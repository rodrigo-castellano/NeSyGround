"""Terminal soundness filters — verify collected groundings after BC loop.

Submodules:
  fp_batch.py  — cross-query Kleene T_P fixed-point (per-batch)
  fp_global.py — FC provable set check (precomputed global)
"""

from typing import Optional
import torch
from torch import Tensor


def check_in_fp_global(atom_hashes: Tensor, fp_global_hashes: Tensor) -> Tensor:
    """Check which atoms are in the fp_global set (I_D) via searchsorted.

    Args:
        atom_hashes: Arbitrary-shape long tensor of atom hashes.
        fp_global_hashes: Sorted 1-D long tensor of fp_global atom hashes.

    Returns:
        Bool tensor of same shape as atom_hashes.
    """
    flat = atom_hashes.reshape(-1)
    pos = torch.searchsorted(fp_global_hashes, flat)
    pos = pos.clamp(max=fp_global_hashes.size(0) - 1)
    found = fp_global_hashes[pos] == flat
    return found.view(atom_hashes.shape)
