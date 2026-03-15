"""Fact indexing — hashing, membership, targeted lookup, and enumeration.

FactIndex(nn.Module)
    Base: sort facts by hash, exists() via binary search.

ArgKeyFactIndex(FactIndex)
    MGU: O(1) targeted lookup via (pred, arg) composite-key tables.

InvertedFactIndex(FactIndex)
    Enum: O(1) enumeration via (pred*E + bound) offset tables.

BlockSparseFactIndex(InvertedFactIndex)
    Enum: dense [P, E, K] blocks, falls back to offset when memory exceeds limit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# Module-level helpers
# ======================================================================

@torch.no_grad()
def pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """Pack [pred, arg0, arg1] → int64 keys: ((pred * base) + arg0) * base + arg1."""
    if atoms.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=atoms.device)
    a = atoms.long()
    return ((a[:, 0] * base) + a[:, 1]) * base + a[:, 2]


def fact_contains(atoms: Tensor, fact_hashes: Tensor, pack_base: int) -> Tensor:
    """Check if atoms exist in a fact set via binary search on sorted hashes."""
    N = atoms.shape[0]
    if N == 0 or fact_hashes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool, device=atoms.device)
    keys = pack_triples_64(atoms.long(), pack_base)
    F = fact_hashes.shape[0]
    idx = torch.searchsorted(fact_hashes, keys)
    return (idx < F) & (fact_hashes[idx.clamp(max=F - 1)] == keys)


def _build_segment_index(
    keys: Tensor, max_key: int, device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build (order, starts, lens) segment index for sorted composite keys.

    Returns:
        order:  [N] sort permutation.
        starts: [max_key] start offset per key.
        lens:   [max_key] count per key.
    """
    order = keys.argsort(stable=True)
    sorted_keys = keys[order]
    unique, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    starts = torch.zeros(max_key, dtype=torch.long, device=device)
    lens = torch.zeros(max_key, dtype=torch.long, device=device)
    cum = counts.cumsum(0)
    starts[unique] = cum - counts
    lens[unique] = counts
    return order, starts, lens


def _build_offset_table(
    keys: Tensor, values: Tensor, num_slots: int, device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Build offset table: sorted values + cumulative offset array.

    Returns:
        sorted_values: [N] values sorted by key.
        offsets:       [num_slots + 1] cumulative offsets per key.
    """
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = values[sort_idx]
    offsets = torch.zeros(num_slots + 1, dtype=torch.long, device=device)
    ones = torch.ones(keys.size(0), dtype=torch.long, device=device)
    offsets.scatter_add_(0, sorted_keys + 1, ones)
    offsets = torch.cumsum(offsets, dim=0)
    return sorted_values, offsets


# ======================================================================
# FactIndex — base: sort, hash, exists
# ======================================================================

class FactIndex(nn.Module):
    """Base fact index: sorted facts + binary-search membership.

    Subclasses add targeted lookup (ArgKey) or enumeration (Inverted, BlockSparse).
    """

    pack_base: int

    def __init__(
        self,
        facts_idx: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._constant_no = constant_no
        self._padding_idx = padding_idx
        self.pack_base = (pack_base if pack_base is not None
                          else max(int(constant_no), int(padding_idx)) + 2)

        if facts_idx.numel() == 0:
            self.register_buffer("facts_idx",
                                 torch.zeros(0, 3, dtype=torch.long, device=device))
            self.register_buffer("fact_hashes",
                                 torch.zeros(0, dtype=torch.long, device=device))
            return

        facts = facts_idx.long().to(device)
        hashes = pack_triples_64(facts, self.pack_base)
        order = hashes.argsort()
        self.register_buffer("facts_idx", facts[order])
        self.register_buffer("fact_hashes", hashes[order])

    def exists(self, atoms: Tensor) -> Tensor:
        """[N, 3] → [N] bool via binary search."""
        return fact_contains(atoms, self.fact_hashes, self.pack_base)


# ======================================================================
# ArgKeyFactIndex — MGU: targeted lookup
# ======================================================================

class ArgKeyFactIndex(FactIndex):
    """O(1) targeted fact lookup via (pred, arg) composite-key tables.

    Given a partially-bound atom (pred, ?X, const) or (pred, const, ?Y),
    returns matching fact indices in constant time.
    """

    def __init__(
        self,
        facts_idx: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
    ) -> None:
        super().__init__(facts_idx, constant_no, padding_idx, device,
                         pack_base=pack_base)
        self._build_indices(device)

    def _build_indices(self, device: torch.device) -> None:
        facts = self.facts_idx
        if facts.numel() == 0:
            z = torch.zeros(0, dtype=torch.long, device=device)
            for name in ("_a0_order", "_a0_starts", "_a0_lens",
                         "_a1_order", "_a1_starts", "_a1_lens",
                         "_p_order", "_p_starts", "_p_lens"):
                self.register_buffer(name, z.clone())
            self._key_scale = self._constant_no + 2
            self._max_fact_pairs = 1
            return

        preds = facts[:, 0].long()
        arg0, arg1 = facts[:, 1].long(), facts[:, 2].long()
        ks = max(int(self._constant_no), int(self._padding_idx)) + 2
        self._key_scale = ks

        # (pred, arg0), (pred, arg1), pred-only — same pattern via helper
        o0, s0, l0 = _build_segment_index(preds * ks + arg0,
                                           int((preds * ks + arg0).max()) + 1, device)
        o1, s1, l1 = _build_segment_index(preds * ks + arg1,
                                           int((preds * ks + arg1).max()) + 1, device)
        op, sp, lp = _build_segment_index(preds,
                                           int(preds.max()) + 1, device)

        self.register_buffer("_a0_order", o0)
        self.register_buffer("_a0_starts", s0)
        self.register_buffer("_a0_lens", l0)
        self.register_buffer("_a1_order", o1)
        self.register_buffer("_a1_starts", s1)
        self.register_buffer("_a1_lens", l1)
        self.register_buffer("_p_order", op)
        self.register_buffer("_p_starts", sp)
        self.register_buffer("_p_lens", lp)

        max0 = int(l0.max().item()) if l0.numel() > 0 else 1
        max1 = int(l1.max().item()) if l1.numel() > 0 else 1
        self._max_fact_pairs = max(max0, max1, 1)

    @property
    def max_fact_pairs(self) -> int:
        return self._max_fact_pairs

    def targeted_lookup(
        self, query_atoms: Tensor, max_results: int,
    ) -> Tuple[Tensor, Tensor]:
        """O(1) fact lookup. Preference: (pred,arg0) → (pred,arg1) → pred-only.

        Returns: (fact_idx [B, K], valid [B, K]).
        """
        B = query_atoms.shape[0]
        dev = query_atoms.device
        cno, pad, ks = self._constant_no, self._padding_idx, self._key_scale
        F = self._a0_order.shape[0]
        clamp_max = max(F - 1, 0)

        preds, a0, a1 = query_atoms[:, 0], query_atoms[:, 1], query_atoms[:, 2]
        is_c0 = (a0 <= cno) & (a0 != pad)
        is_c1 = (a1 <= cno) & (a1 != pad)
        offsets = torch.arange(max_results, device=dev).unsqueeze(0)

        def _lookup(order, starts, lens, keys, is_const):
            safe = keys.clamp(0, starts.shape[0] - 1)
            left = starts[safe]
            cnt = lens[safe].clamp(max=max_results)
            idx = (left.unsqueeze(1) + offsets).clamp(0, clamp_max)
            v = (offsets < cnt.unsqueeze(1)) & is_const.unsqueeze(1)
            return order[idx.reshape(-1)].reshape(B, max_results), v

        fi0, v0 = _lookup(self._a0_order, self._a0_starts, self._a0_lens,
                           preds * ks + a0, is_c0)
        fi1, v1 = _lookup(self._a1_order, self._a1_starts, self._a1_lens,
                           preds * ks + a1, is_c1)

        use0 = is_c0.unsqueeze(1)
        fact_idx = torch.where(use0, fi0, fi1)
        valid = torch.where(use0, v0, v1)

        # Pred-only fallback for both-variable queries
        if F > 0 and self._p_starts.numel() > 0:
            both_var = ~is_c0 & ~is_c1 & (preds != pad)
            fip, vp = _lookup(self._p_order, self._p_starts, self._p_lens,
                              preds, both_var)
            bv = both_var.unsqueeze(1)
            fact_idx = torch.where(bv, fip, fact_idx)
            valid = torch.where(bv, vp, valid)

        return fact_idx, valid

    def enumerate(self, preds, bound_args, direction):
        raise NotImplementedError("Use InvertedFactIndex for enumerate().")

    def __repr__(self) -> str:
        return (f"ArgKeyFactIndex(F={self.facts_idx.shape[0]}, "
                f"K={self._max_fact_pairs})")


# ======================================================================
# InvertedFactIndex — Enum: offset-table enumeration
# ======================================================================

class InvertedFactIndex(FactIndex):
    """O(1) free-variable enumeration via (pred*E + bound) offset tables."""

    def __init__(
        self,
        facts_idx: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        num_entities: int,
        num_predicates: int,
        max_facts_per_query: int = 64,
    ) -> None:
        super().__init__(facts_idx, constant_no, padding_idx, device)
        self._num_entities = num_entities
        self._num_predicates = num_predicates
        self._max_facts_per_query = max_facts_per_query
        self._build_offset_tables(device)

    def _build_offset_tables(self, device: torch.device) -> None:
        facts = self.facts_idx
        E, P = self._num_entities, self._num_predicates
        num_slots = P * E

        if facts.shape[0] == 0:
            z = torch.zeros(1, dtype=torch.long, device=device)
            o = torch.zeros(num_slots + 1, dtype=torch.long, device=device)
            self.register_buffer("_ps_values", z)
            self.register_buffer("_ps_offsets", o)
            self.register_buffer("_po_values", z)
            self.register_buffer("_po_offsets", o)
            return

        preds, subjs, objs = facts[:, 0].long(), facts[:, 1].long(), facts[:, 2].long()

        # (pred, subj) → objs
        ps_vals, ps_off = _build_offset_table(preds * E + subjs, objs, num_slots, device)
        self.register_buffer("_ps_values", ps_vals)
        self.register_buffer("_ps_offsets", ps_off)

        # (pred, obj) → subjs
        po_vals, po_off = _build_offset_table(preds * E + objs, subjs, num_slots, device)
        self.register_buffer("_po_values", po_vals)
        self.register_buffer("_po_offsets", po_off)

    # Backward compat aliases
    @property
    def ps_sorted_objs(self): return self._ps_values
    @property
    def ps_offsets(self): return self._ps_offsets
    @property
    def po_sorted_subjs(self): return self._po_values
    @property
    def po_offsets(self): return self._po_offsets

    def enumerate(
        self, preds: Tensor, bound_args: Tensor, direction: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Enumerate free-variable bindings. direction: 0=objects, 1=subjects."""
        N = preds.size(0)
        M = self._max_facts_per_query
        dev = preds.device

        keys = preds * self._num_entities + bound_args
        is_obj = (direction == 0)

        starts_ps = self._ps_offsets[keys]
        counts_ps = (self._ps_offsets[keys + 1] - starts_ps).clamp(0, M)
        starts_po = self._po_offsets[keys]
        counts_po = (self._po_offsets[keys + 1] - starts_po).clamp(0, M)

        starts = torch.where(is_obj, starts_ps, starts_po)
        counts = torch.where(is_obj, counts_ps, counts_po)

        pos = torch.arange(M, device=dev).unsqueeze(0).expand(N, -1)
        valid = pos < counts.unsqueeze(1)
        gi = (starts.unsqueeze(1) + pos).clamp(
            0, max(self._ps_values.size(0), self._po_values.size(0)) - 1)

        gi_ps = gi.clamp(max=self._ps_values.size(0) - 1)
        gi_po = gi.clamp(max=self._po_values.size(0) - 1)
        candidates = torch.where(
            is_obj.unsqueeze(1),
            self._ps_values[gi_ps],
            self._po_values[gi_po])

        return candidates, valid

    def targeted_lookup(self, query_atoms, max_results):
        raise NotImplementedError("Use ArgKeyFactIndex for targeted_lookup().")

    def __repr__(self) -> str:
        return (f"InvertedFactIndex(F={self.facts_idx.shape[0]}, "
                f"P={self._num_predicates}, E={self._num_entities})")


# ======================================================================
# BlockSparseFactIndex — Enum: dense [P, E, K] blocks
# ======================================================================

class BlockSparseFactIndex(InvertedFactIndex):
    """Dense [P, E, K] blocks for O(1) enumerate + exists.

    Falls back to InvertedFactIndex when memory exceeds max_memory_mb.
    """

    def __init__(
        self,
        facts_idx: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        num_entities: int,
        num_predicates: int,
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
    ) -> None:
        super().__init__(facts_idx, constant_no, padding_idx, device,
                         num_entities, num_predicates, max_facts_per_query)
        self._build_dense(device, max_memory_mb)

    def _build_dense(self, device: torch.device, max_memory_mb: int) -> None:
        facts = self.facts_idx
        P, E, M = self._num_predicates, self._num_entities, self._max_facts_per_query

        if facts.shape[0] == 0:
            self._use_dense = False
            return

        preds, subjs, objs = facts[:, 0], facts[:, 1], facts[:, 2]

        _, ps_cnt = torch.unique(preds * E + subjs, return_counts=True)
        _, po_cnt = torch.unique(preds * E + objs, return_counts=True)
        K = min(max(
            int(ps_cnt.max().item()) if ps_cnt.numel() else 0,
            int(po_cnt.max().item()) if po_cnt.numel() else 0), M)

        if 2 * P * E * K * 8 / (1024 ** 2) > max_memory_mb:
            self._use_dense = False
            return

        self._use_dense = True
        self._K = K

        # Build on CPU then move
        ps_blocks = torch.zeros(P, E, K, dtype=torch.long)
        ps_counts = torch.zeros(P, E, dtype=torch.long)
        po_blocks = torch.zeros(P, E, K, dtype=torch.long)
        po_counts = torch.zeros(P, E, dtype=torch.long)

        p_c, s_c, o_c = preds.cpu(), subjs.cpu(), objs.cpu()
        for i in range(facts.shape[0]):
            pi, si, oi = int(p_c[i]), int(s_c[i]), int(o_c[i])
            j = int(ps_counts[pi, si])
            if j < K:
                ps_blocks[pi, si, j] = oi
                ps_counts[pi, si] = j + 1
            j = int(po_counts[pi, oi])
            if j < K:
                po_blocks[pi, oi, j] = si
                po_counts[pi, oi] = j + 1

        self.register_buffer("_ps_blocks", ps_blocks.to(device))
        self.register_buffer("_ps_counts", ps_counts.to(device))
        self.register_buffer("_po_blocks", po_blocks.to(device))
        self.register_buffer("_po_counts", po_counts.to(device))

    @property
    def max_fact_pairs(self) -> int:
        return self._K if self._use_dense else self._max_facts_per_query

    def enumerate(self, preds, bound_args, direction):
        if not self._use_dense:
            return super().enumerate(preds, bound_args, direction)

        K = self._K
        is_obj = (direction == 0)
        cands = torch.where(
            is_obj.unsqueeze(1),
            self._ps_blocks[preds, bound_args],
            self._po_blocks[preds, bound_args])
        counts = torch.where(
            is_obj,
            self._ps_counts[preds, bound_args],
            self._po_counts[preds, bound_args])
        valid = torch.arange(K, device=preds.device).unsqueeze(0) < counts.unsqueeze(1)
        return cands, valid

    def exists(self, atoms: Tensor) -> Tensor:
        if not self._use_dense:
            return super().exists(atoms)
        K = self._K
        block = self._ps_blocks[atoms[:, 0], atoms[:, 1]]
        counts = self._ps_counts[atoms[:, 0], atoms[:, 1]]
        pos = torch.arange(K, device=atoms.device).unsqueeze(0)
        return ((block == atoms[:, 2].unsqueeze(1)) & (pos < counts.unsqueeze(1))).any(1)

    def __repr__(self) -> str:
        d = f"dense=True, K={self._K}" if self._use_dense else "dense=False"
        return (f"BlockSparseFactIndex(F={self.facts_idx.shape[0]}, "
                f"P={self._num_predicates}, E={self._num_entities}, {d})")


# ======================================================================
# Utilities
# ======================================================================

def shuffle_facts_per_predicate(
    facts_idx: Tensor,
    predicate_no: Optional[int] = None,
    seed: int = 42,
) -> Tensor:
    """Randomly shuffle facts within each predicate segment."""
    device = facts_idx.device
    gen = torch.Generator(device=device).manual_seed(seed)
    preds = facts_idx[:, 0]
    num_preds = max(int(preds.max().item()) + 1,
                    (predicate_no + 1) if predicate_no else 1)
    order = torch.argsort(preds, stable=True)
    facts_sorted = facts_idx[order]
    counts = torch.bincount(preds.long(), minlength=num_preds)
    starts = torch.zeros(num_preds + 1, dtype=torch.long, device=device)
    starts[1:] = counts.cumsum(0)

    shuffled = facts_sorted.clone()
    for p in range(num_preds):
        s, e = starts[p].item(), starts[p + 1].item()
        if e - s > 1:
            shuffled[s:e] = facts_sorted[
                s + torch.randperm(e - s, device=device, generator=gen)]
    return shuffled
