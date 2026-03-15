"""Shared triple hashing utility for filters."""

from torch import Tensor


def hash_atoms(atoms: Tensor, pack_base: int) -> Tensor:
    """Hash [..., 3] triples: pred * base^2 + arg0 * base + arg1.

    Args:
        atoms: [..., 3] tensor of (pred, arg0, arg1) triples.
        pack_base: hash packing base.

    Returns:
        [...] int64 hash values.
    """
    return (atoms[..., 0].long() * (pack_base * pack_base)
            + atoms[..., 1].long() * pack_base
            + atoms[..., 2].long())
