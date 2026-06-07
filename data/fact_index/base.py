"""FactIndex base — fact hashing + membership (``exists``).

The base owns the one thing every index shares: a sorted int64 hash of every
fact, queried by binary search. Subclasses add a *capability* on top:

    ArgKeyFactIndex      targeted_lookup   (sld/rtf — bind the free arg of a goal)
    InvertedFactIndex    enumerate         (pbc — list free-variable bindings)
    BlockSparseFactIndex enumerate+exists  (pbc — dense [P,E,K] blocks)

BIT-EXACT RECIPES reproduced here (the fingerprint enforces them byte-for-byte):
  * ``pack_triples_64`` key order  ((pred*base)+arg0)*base+arg1
  * ``pack_base`` default          max(constant_no, padding_idx) + 2
  * ``exists`` = searchsorted on the sorted hashes + clamped equality
Re-authored from scratch against those recipes — not a copy of the old monolith.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor


@torch.no_grad()
def pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """Pack ``[..,3]`` triples ``[pred, arg0, arg1]`` into int64 keys.

    BIT-EXACT: the key order ``((pred*base)+arg0)*base+arg1`` and int64 dtype
    are the membership contract — any other order reshuffles the sorted hashes
    and breaks ``exists``/the fingerprint. ``base`` must exceed every id.
    """
    if atoms.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=atoms.device)
    a = atoms.long()
    return ((a[:, 0] * base) + a[:, 1]) * base + a[:, 2]


def fact_contains(atoms: Tensor, fact_hashes: Tensor, pack_base: int) -> Tensor:
    """``[N,3] -> [N]`` bool membership via binary search on sorted hashes.

    BIT-EXACT: ``searchsorted`` then clamp the insertion point to the last slot
    and test equality — the clamp keeps out-of-range keys (vars/padding) safe and
    reports them absent, matching the indexing contract the fingerprint locks.
    """
    N = atoms.shape[0]
    if N == 0 or fact_hashes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool, device=atoms.device)
    keys = pack_triples_64(atoms.long(), pack_base)
    F = fact_hashes.shape[0]
    insert = torch.searchsorted(fact_hashes, keys)
    return (insert < F) & (fact_hashes[insert.clamp(max=F - 1)] == keys)


class FactIndex(nn.Module):
    """Sorted-hash fact set with binary-search membership.

    Build a concrete index through ``data.fact_index.create`` (the factory),
    which selects the subclass by name. The base is instantiable on its own
    only for plain membership.
    """

    pack_base: int

    def __init__(
        self,
        facts_idx: Tensor,
        *,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
        order: Literal["original", "shuffle"] = "original",
        order_seed: int = 42,
    ) -> None:
        super().__init__()
        self._constant_no = constant_no
        self._padding_idx = padding_idx
        # BIT-EXACT: base must clear both the largest entity id and the padding
        # sentinel so no two distinct triples collide to the same packed key.
        self.pack_base = (pack_base if pack_base is not None
                          else max(int(constant_no), int(padding_idx)) + 2)

        if facts_idx.numel() == 0:
            raise ValueError("facts_idx is empty — cannot build a fact index without facts")

        facts = facts_idx.long().to(device)
        if order == "shuffle":
            facts = self._shuffle_per_predicate(facts, order_seed)
        hashes = pack_triples_64(facts, self.pack_base)
        sort_order = hashes.argsort()
        self.register_buffer("facts_idx", facts[sort_order])
        self.register_buffer("fact_hashes", hashes[sort_order])

    @staticmethod
    def _shuffle_per_predicate(facts_idx: Tensor, seed: int) -> Tensor:
        """Randomly permute facts *within* each predicate group (vectorized).

        Optional ``order='shuffle'`` knob: when a K_f cap truncates a slot,
        shuffling decorrelates which facts survive. Group boundaries are kept
        intact by offsetting the random key with a per-group integer base.
        """
        device = facts_idx.device
        N = facts_idx.shape[0]
        if N <= 1:
            return facts_idx.clone()
        gen = torch.Generator(device=device).manual_seed(seed)
        preds = facts_idx[:, 0]
        order = torch.argsort(preds, stable=True)
        sorted_facts = facts_idx[order]
        sorted_preds = preds[order]
        _, _, counts = torch.unique_consecutive(
            sorted_preds, return_inverse=True, return_counts=True)
        group_ids = torch.arange(counts.shape[0], device=device).repeat_interleave(counts)
        random_keys = torch.rand(N, device=device, generator=gen)
        # group base * 2.0 > any random key in [0,1) -> groups never interleave
        composite = group_ids.float() * 2.0 + random_keys
        return sorted_facts[torch.argsort(composite, stable=True)]

    @property
    def num_facts(self) -> int:
        return self.facts_idx.shape[0]

    def exists(self, atoms: Tensor) -> Tensor:
        """``[N,3] -> [N]`` bool membership."""
        return fact_contains(atoms, self.fact_hashes, self.pack_base)


__all__ = ["FactIndex", "pack_triples_64", "fact_contains"]
