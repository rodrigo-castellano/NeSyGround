"""Fact indexing — hashing, membership, targeted lookup, and enumeration.

Hierarchy
---------
FactIndex(nn.Module)          base: sort facts by hash, exists() via binary search
├── ArgKeyFactIndex           MGU: O(1) targeted lookup via (pred, arg) composite keys
├── InvertedFactIndex         Enum: O(1) enumeration via (pred*E + bound) offset tables
└── BlockSparseFactIndex      Enum: dense [P, E, K] blocks, offset fallback

Factory: ``FactIndex.create(facts_idx, type='arg_key', ...)``
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# Hashing helpers (module-level, used across classes)
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


# ======================================================================
# FactIndex — base
# ======================================================================

_FACT_INDEX_TYPES = {}  # filled after subclass definitions


class FactIndex(nn.Module):
    """Base fact index: sorted facts + binary-search membership.

    Subclasses add targeted lookup (ArgKey) or enumeration (Inverted, BlockSparse).
    Use ``FactIndex.create(...)`` to construct the right subclass by name.
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
        **kwargs,
    ) -> None:
        super().__init__()
        self._constant_no = constant_no
        self._padding_idx = padding_idx
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
        """Shuffle facts within each predicate group."""
        device = facts_idx.device
        gen = torch.Generator(device=device).manual_seed(seed)
        preds = facts_idx[:, 0]
        num_preds = int(preds.max().item()) + 1 if preds.numel() > 0 else 1
        sort_order = torch.argsort(preds, stable=True)
        sorted_facts = facts_idx[sort_order]
        counts = torch.bincount(preds.long(), minlength=num_preds)
        starts = torch.zeros(num_preds + 1, dtype=torch.long, device=device)
        starts[1:] = counts.cumsum(0)
        shuffled = sorted_facts.clone()
        for p in range(num_preds):
            s, e = starts[p].item(), starts[p + 1].item()
            if e - s > 1:
                shuffled[s:e] = sorted_facts[
                    s + torch.randperm(e - s, device=device, generator=gen)]
        return shuffled

    @property
    def num_facts(self) -> int:
        return self.facts_idx.shape[0]

    def exists(self, atoms: Tensor) -> Tensor:
        """[N, 3] → [N] bool via binary search."""
        return fact_contains(atoms, self.fact_hashes, self.pack_base)

    @classmethod
    def create(
        cls,
        facts_idx: Tensor,
        *,
        type: Literal["arg_key", "inverted", "block_sparse"] = "arg_key",
        constant_no: int,
        predicate_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
        order: Literal["original", "shuffle"] = "original",
        order_seed: int = 42,
    ) -> "FactIndex":
        """Factory: create a FactIndex subclass by type name.

        Args:
            type: 'arg_key' (MGU lookup), 'inverted' (offset enumeration),
                  or 'block_sparse' (dense blocks with offset fallback).
            order: 'original' (keep input order) or 'shuffle' (random within
                   each predicate group — useful when K_f caps results).
            order_seed: random seed for shuffle reproducibility.
        """
        if type not in _FACT_INDEX_TYPES:
            raise ValueError(
                f"Unknown fact index type: {type!r}. "
                f"Choose from {list(_FACT_INDEX_TYPES)}"
            )
        return _FACT_INDEX_TYPES[type](
            facts_idx,
            constant_no=constant_no, predicate_no=predicate_no,
            padding_idx=padding_idx, device=device, pack_base=pack_base,
            max_facts_per_query=max_facts_per_query,
            max_memory_mb=max_memory_mb,
            order=order, order_seed=order_seed,
        )


# ======================================================================
# ArgKeyFactIndex — MGU: targeted lookup
# ======================================================================

class ArgKeyFactIndex(FactIndex):
    """O(1) targeted fact lookup via (pred, arg) composite-key tables."""

    def __init__(
        self,
        facts_idx: Tensor,
        *,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(facts_idx, constant_no=constant_no,
                         padding_idx=padding_idx, device=device,
                         pack_base=pack_base)
        self._build_indices(device)

    @staticmethod
    def _build_segment_index(
        keys: Tensor, max_key: int, device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build (order, starts, lens) segment index for sorted composite keys."""
        order = keys.argsort(stable=True)
        sorted_keys = keys[order]
        unique, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        starts = torch.zeros(max_key, dtype=torch.long, device=device)
        lens = torch.zeros(max_key, dtype=torch.long, device=device)
        cum = counts.cumsum(0)
        starts[unique] = cum - counts
        lens[unique] = counts
        return order, starts, lens

    def _build_indices(self, device: torch.device) -> None:
        facts = self.facts_idx
        preds = facts[:, 0].long()
        arg0, arg1 = facts[:, 1].long(), facts[:, 2].long()
        ks = max(int(self._constant_no), int(self._padding_idx)) + 2
        self._key_scale = ks
        bsi = self._build_segment_index

        o0, s0, l0 = bsi(preds * ks + arg0,
                          int((preds * ks + arg0).max()) + 1, device)
        o1, s1, l1 = bsi(preds * ks + arg1,
                          int((preds * ks + arg1).max()) + 1, device)
        op, sp, lp = bsi(preds, int(preds.max()) + 1, device)

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
        """O(1) fact lookup. Returns: (fact_idx [B, K], valid [B, K])."""
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
        *,
        constant_no: int,
        predicate_no: int,
        padding_idx: int,
        device: torch.device,
        max_facts_per_query: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(facts_idx, constant_no=constant_no,
                         padding_idx=padding_idx, device=device)
        self._num_entities = constant_no
        self._num_predicates = predicate_no + 1
        self._max_facts_per_query = max_facts_per_query
        self._build_offset_tables(device)

    @staticmethod
    def _build_offset_table(
        keys: Tensor, values: Tensor, num_slots: int, device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Build offset table: sorted values + cumulative offset array."""
        sort_idx = torch.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_values = values[sort_idx]
        offsets = torch.zeros(num_slots + 1, dtype=torch.long, device=device)
        ones = torch.ones(keys.size(0), dtype=torch.long, device=device)
        offsets.scatter_add_(0, sorted_keys + 1, ones)
        offsets = torch.cumsum(offsets, dim=0)
        return sorted_values, offsets

    def _build_offset_tables(self, device: torch.device) -> None:
        facts = self.facts_idx
        E, P = self._num_entities, self._num_predicates
        num_slots = P * E
        preds, subjs, objs = facts[:, 0].long(), facts[:, 1].long(), facts[:, 2].long()
        bot = self._build_offset_table

        ps_vals, ps_off = bot(preds * E + subjs, objs, num_slots, device)
        self.register_buffer("_ps_values", ps_vals)
        self.register_buffer("_ps_offsets", ps_off)

        po_vals, po_off = bot(preds * E + objs, subjs, num_slots, device)
        self.register_buffer("_po_values", po_vals)
        self.register_buffer("_po_offsets", po_off)

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
        *,
        constant_no: int,
        predicate_no: int,
        padding_idx: int,
        device: torch.device,
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(facts_idx, constant_no=constant_no,
                         predicate_no=predicate_no, padding_idx=padding_idx,
                         device=device, max_facts_per_query=max_facts_per_query)
        self._build_dense(device, max_memory_mb)

    def _build_dense(self, device: torch.device, max_memory_mb: int) -> None:
        facts = self.facts_idx
        P, E, M = self._num_predicates, self._num_entities, self._max_facts_per_query
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
# Registry
# ======================================================================

_FACT_INDEX_TYPES.update({
    "arg_key": ArgKeyFactIndex,
    "inverted": InvertedFactIndex,
    "block_sparse": BlockSparseFactIndex,
})


