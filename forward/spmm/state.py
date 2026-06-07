"""Closure state: old / delta sparse mat dicts, hashes, transpose caches.

``ClosureState`` encapsulates the bookkeeping required by the SpMM
forward chaining loop. At the start of step k:

  * ``old_mats[p]``    — atoms of pred ``p`` in S^{k-2} (= base if k==1)
  * ``delta_mats[p]``  — atoms of pred ``p`` added at step k-1
  * ``old_hashes``     — sorted hashes for the "old" set
  * ``delta_hashes``   — sorted hashes added at step k-1
  * ``provable_hashes`` — running union; used for global dedup

End-of-step ``advance(added)`` rotates the state: the current delta
folds into old, and the new ``added`` becomes the next step's delta.

``TransposeCache`` is a lazy proxy that materialises the transpose of
each tracked matrix only on first read in a given step. The cache is
invalidated whenever the underlying source changes.
"""
from __future__ import annotations

from typing import Dict, Set

import torch
from torch import Tensor

from grounder.forward.spmm.matrices import (
    sparse_nnz, transpose_csr, csr_local_hashes,
    build_csr_from_local_hashes, build_csr_and_transpose, sorted_merge,
)


__all__ = ["ClosureState", "TransposeCache"]


class TransposeCache:
    """Dict-like proxy that lazily transposes a source CSR mat per pred.

    Implements ``__contains__`` and ``__getitem__`` so callers (kernels
    in particular) can use it interchangeably with a plain
    ``Dict[int, Tensor]``. Use ``invalidate(p)`` after mutating the
    underlying source mat for that pred.
    """

    def __init__(self, source_mats: Dict[int, Tensor], E: int):
        self._source = source_mats
        self._E = E
        self._cache: Dict[int, Tensor] = {}

    def __contains__(self, p: int) -> bool:
        return p in self._source

    def __getitem__(self, p: int) -> Tensor:
        cached = self._cache.get(p)
        if cached is not None:
            return cached
        t = transpose_csr(self._source[p], self._E)
        self._cache[p] = t
        return t

    def get(self, p: int, default=None):
        if p not in self._source:
            return default
        return self[p]

    def invalidate(self, p: int = None) -> None:
        if p is None:
            self._cache.clear()
        else:
            self._cache.pop(p, None)


class ClosureState:
    def __init__(
        self,
        base_mats: Dict[int, Tensor],
        base_mats_T: Dict[int, Tensor],
        tracked_preds: Set[int],
        needs_T: Set[int],
        fact_hashes: Tensor,
        num_facts: int,
        E: int,
        P: int,
        device: torch.device,
    ) -> None:
        self.E = E
        self.P = P
        self.E2 = E * E
        self.device = device
        self.tracked_preds = tracked_preds
        self.needs_T = needs_T

        self.fact_hashes = fact_hashes
        self.num_facts = num_facts
        self.pred_boundaries = (
            torch.arange(P + 1, dtype=torch.long, device=device) * self.E2)

        self.base_mats = base_mats
        self.base_mats_T = base_mats_T

        # State at start of step 0: nothing is "old", base is "delta".
        self.old_mats: Dict[int, Tensor] = {}
        # Lazy transpose proxy over ``old_mats`` — invalidated per pred
        # whenever ``old_mats[p]`` is rebuilt.
        self.old_mats_T: TransposeCache = TransposeCache(self.old_mats, E)

        self.delta_mats: Dict[int, Tensor] = {
            p: m for p, m in base_mats.items() if p in tracked_preds}
        self.delta_mats_T: TransposeCache = TransposeCache(
            self.delta_mats, E)
        # Pre-warm the transpose cache for preds we know we'll need —
        # this lets the first step skip an extra pass through the COO
        # round-trip path inside the kernel.
        for p in needs_T:
            if p in base_mats_T:
                self.delta_mats_T._cache[p] = base_mats_T[p]

        self.old_hashes: Tensor = torch.zeros(
            0, dtype=torch.long, device=device)
        self.delta_hashes: Tensor = fact_hashes
        self.provable_hashes: Tensor = torch.zeros(
            0, dtype=torch.long, device=device)

        self.step = 0   # number of completed iterations

    @property
    def empty_fact_hashes(self) -> Tensor:
        return torch.zeros(0, dtype=torch.long, device=self.device)

    def total_provable(self) -> int:
        return int(self.provable_hashes.numel())

    def delta_total_nnz(self) -> int:
        return sum(sparse_nnz(m) for m in self.delta_mats.values())

    def old_total_nnz(self) -> int:
        return sum(sparse_nnz(m) for m in self.old_mats.values())

    def advance(self, added: Tensor) -> None:
        """End-of-step rotation: old ← old ∪ delta; delta ← ``added``.

        ``added`` is the sorted set of new global hashes derived this
        step (already deduped against ``provable_hashes``).
        """
        # Provable hash list grows monotonically.
        self.provable_hashes = sorted_merge(self.provable_hashes, added)

        # old ← old ∪ delta (per pred). Invalidate transpose cache
        # for every pred whose old_mats[p] changes.
        new_old_hashes = sorted_merge(self.old_hashes, self.delta_hashes)
        for p, d_csr in list(self.delta_mats.items()):
            o_csr = self.old_mats.get(p)
            if o_csr is None:
                self.old_mats[p] = d_csr
                # Promote any pre-built transpose from delta cache.
                if p in self.delta_mats_T._cache:
                    self.old_mats_T._cache[p] = \
                        self.delta_mats_T._cache[p]
                continue
            o_local = csr_local_hashes(o_csr, self.E)
            d_local = csr_local_hashes(d_csr, self.E)
            merged = torch.cat([o_local, d_local]).sort().values
            if p in self.needs_T:
                self.old_mats[p], t = build_csr_and_transpose(
                    merged, self.E)
                self.old_mats_T._cache[p] = t
            else:
                self.old_mats[p] = build_csr_from_local_hashes(
                    merged, self.E)
                self.old_mats_T.invalidate(p)
        self.old_hashes = new_old_hashes

        # delta ← `added`
        self.delta_mats.clear()
        self.delta_mats_T.invalidate()  # whole cache cleared
        d_starts = torch.searchsorted(added, self.pred_boundaries[:-1])
        d_ends = torch.searchsorted(added, self.pred_boundaries[1:])
        for p in self.tracked_preds:
            ds = int(d_starts[p].item())
            de = int(d_ends[p].item())
            if de <= ds:
                continue
            new_local = added[ds:de] - p * self.E2
            if p in self.needs_T:
                csr, t = build_csr_and_transpose(new_local, self.E)
                self.delta_mats[p] = csr
                self.delta_mats_T._cache[p] = t
            else:
                self.delta_mats[p] = build_csr_from_local_hashes(
                    new_local, self.E)
        self.delta_hashes = added

        self.step += 1
