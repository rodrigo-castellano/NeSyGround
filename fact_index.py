"""Unified fact index: hashing, membership, targeted lookup, and enumeration.

Three concrete implementations behind a common Protocol:

- **ArgKeyFactIndex** — BE-style O(1) targeted lookup via (pred, arg) composite-key
  tables.  Best for unification engines that query "which facts match this
  partially-bound atom?".
- **InvertedFactIndex** — TS-style O(1) enumeration via (pred*E+bound) offset
  tables.  Best for grounding engines that enumerate free-variable bindings.
- **BlockSparseFactIndex** — Dense [P, E, K] blocks for O(1) enumerate + exists.
  Subclass of InvertedFactIndex; falls back to offset-based when memory exceeds
  a configurable limit.

Module-level helpers ``pack_triples_64`` and ``fact_contains`` are used by
postprocessing and other downstream code.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """Pack [pred, arg0, arg1] triples into single int64 keys.

    Formula: ``((pred * base) + arg0) * base + arg1``

    Args:
        atoms: [N, 3] integer tensor.
        base:  Multiplier (must exceed max entity/constant index).

    Returns:
        [N] int64 keys.
    """
    if atoms.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=atoms.device)
    atoms_l = atoms.long()
    p, a, b = atoms_l[:, 0], atoms_l[:, 1], atoms_l[:, 2]
    return ((p * base) + a) * base + b


def fact_contains(
    atoms: Tensor,          # [N, 3]
    fact_hashes: Tensor,    # [F] sorted int64 hashes
    pack_base: int,
) -> Tensor:
    """Check if atoms exist in a fact set via binary search.

    Args:
        atoms:       [N, 3] (pred, arg0, arg1).
        fact_hashes: [F] **sorted** int64 keys from ``pack_triples_64``.
        pack_base:   Same base used when building *fact_hashes*.

    Returns:
        [N] bool — True where the atom is a known fact.
    """
    N = atoms.shape[0]
    if N == 0 or fact_hashes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool, device=atoms.device)
    keys = pack_triples_64(atoms.long(), pack_base)
    F = fact_hashes.shape[0]
    idx = torch.searchsorted(fact_hashes, keys)
    return (idx < F) & (fact_hashes[idx.clamp(max=F - 1)] == keys)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FactIndex(Protocol):
    """Common interface for all fact-index implementations."""

    facts_idx: Tensor       # [F, 3] stored facts
    fact_hashes: Tensor     # [F] sorted int64 hashes
    pack_base: int          # multiplier used by pack_triples_64

    def targeted_lookup(
        self,
        query_atoms: Tensor,    # [B, 3]
        max_results: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return indices into ``facts_idx`` that match each query atom.

        Returns:
            fact_idx: [B, max_results] indices into facts_idx.
            valid:    [B, max_results] boolean mask.
        """
        ...

    def enumerate(
        self,
        preds: Tensor,          # [N]
        bound_args: Tensor,     # [N]
        direction: Tensor,      # [N] int  0=objects, 1=subjects
    ) -> Tuple[Tensor, Tensor]:
        """Enumerate free-variable bindings.

        Returns:
            candidates: [N, K] entity indices.
            valid:      [N, K] boolean mask.
        """
        ...

    def exists(self, atoms: Tensor) -> Tensor:
        """Membership test.

        Args:
            atoms: [N, 3] (pred, arg0, arg1).

        Returns:
            [N] bool mask.
        """
        ...


# ---------------------------------------------------------------------------
# Shared base — hashing + exists
# ---------------------------------------------------------------------------

class _FactIndexBase(nn.Module):
    """Shared construction: sort facts by hash, register core buffers."""

    pack_base: int

    def __init__(
        self,
        facts_idx: Tensor,      # [F, 3]  int (pred, arg0, arg1)
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._constant_no = constant_no
        self._padding_idx = padding_idx
        if pack_base is not None:
            self.pack_base = int(pack_base)
        else:
            self.pack_base = max(int(constant_no), int(padding_idx)) + 2

        if facts_idx.numel() == 0:
            z3 = torch.zeros(0, 3, dtype=torch.long, device=device)
            z1 = torch.zeros(0, dtype=torch.long, device=device)
            self.register_buffer("facts_idx", z3)
            self.register_buffer("fact_hashes", z1)
            return

        facts_long = facts_idx.long().to(device)
        hashes = pack_triples_64(facts_long, self.pack_base)
        order = hashes.argsort()

        self.register_buffer("facts_idx", facts_long[order])
        self.register_buffer("fact_hashes", hashes[order])

    # -- exists (shared, O(log F) via searchsorted) --------------------------

    def exists(self, atoms: Tensor) -> Tensor:
        """[N, 3] -> [N] bool via binary search on sorted fact_hashes."""
        return fact_contains(atoms, self.fact_hashes, self.pack_base)


# ---------------------------------------------------------------------------
# ArgKeyFactIndex  (BE-style targeted lookup)
# ---------------------------------------------------------------------------

class ArgKeyFactIndex(_FactIndexBase):
    """O(1) targeted fact lookup via (pred, arg) composite-key index tables.

    Best for unification engines: given a partially-bound query atom
    ``(pred, ?X, const)`` or ``(pred, const, ?Y)``, returns the set of
    matching fact indices in constant time.

    ``enumerate()`` is **not supported** — use ``InvertedFactIndex`` for that.
    """

    def __init__(
        self,
        facts_idx: Tensor,      # [F, 3]
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
    ) -> None:
        super().__init__(facts_idx, constant_no, padding_idx, device, pack_base=pack_base)
        self._build_arg_lookup_indices(device)

    # -- Index construction ---------------------------------------------------

    def _build_arg_lookup_indices(self, device: torch.device) -> None:
        """Build (pred, arg0), (pred, arg1), and pred-only direct-index tables."""
        facts = self.facts_idx  # already sorted by hash
        if facts.numel() == 0:
            z = torch.zeros(0, dtype=torch.long, device=device)
            for name in (
                "_arg0_order", "_arg0_starts", "_arg0_lens",
                "_arg1_order", "_arg1_starts", "_arg1_lens",
                "_pred_order", "_pred_starts", "_pred_lens",
            ):
                self.register_buffer(name, z.clone())
            self._key_scale: int = self._constant_no + 2
            self._max_fact_pairs: int = 1
            return

        preds = facts[:, 0].long()
        arg0 = facts[:, 1].long()
        arg1 = facts[:, 2].long()
        key_scale = max(int(self._constant_no), int(self._padding_idx)) + 2
        self._key_scale = key_scale

        # --- arg0 index ---
        key0 = preds * key_scale + arg0
        order0 = key0.argsort(stable=True)
        sorted0 = key0[order0]
        max_key0 = int(sorted0[-1].item()) + 1
        unique0, counts0 = torch.unique_consecutive(sorted0, return_counts=True)
        starts0 = torch.zeros(max_key0, dtype=torch.long, device=device)
        lens0 = torch.zeros(max_key0, dtype=torch.long, device=device)
        cum0 = counts0.cumsum(0)
        starts0[unique0] = cum0 - counts0
        lens0[unique0] = counts0

        self.register_buffer("_arg0_order", order0)
        self.register_buffer("_arg0_starts", starts0)
        self.register_buffer("_arg0_lens", lens0)

        # --- arg1 index ---
        key1 = preds * key_scale + arg1
        order1 = key1.argsort(stable=True)
        sorted1 = key1[order1]
        max_key1 = int(sorted1[-1].item()) + 1
        unique1, counts1 = torch.unique_consecutive(sorted1, return_counts=True)
        starts1 = torch.zeros(max_key1, dtype=torch.long, device=device)
        lens1 = torch.zeros(max_key1, dtype=torch.long, device=device)
        cum1 = counts1.cumsum(0)
        starts1[unique1] = cum1 - counts1
        lens1[unique1] = counts1

        self.register_buffer("_arg1_order", order1)
        self.register_buffer("_arg1_starts", starts1)
        self.register_buffer("_arg1_lens", lens1)

        # --- predicate-only index (both-variable fallback) ---
        order_p = preds.argsort(stable=True)
        sorted_p = preds[order_p]
        max_pred = int(sorted_p[-1].item()) + 1
        unique_p, counts_p = torch.unique_consecutive(sorted_p, return_counts=True)
        starts_p = torch.zeros(max_pred, dtype=torch.long, device=device)
        lens_p = torch.zeros(max_pred, dtype=torch.long, device=device)
        cum_p = counts_p.cumsum(0)
        starts_p[unique_p] = cum_p - counts_p
        lens_p[unique_p] = counts_p

        self.register_buffer("_pred_order", order_p)
        self.register_buffer("_pred_starts", starts_p)
        self.register_buffer("_pred_lens", lens_p)

        max_pairs_0 = int(counts0.max().item()) if counts0.numel() > 0 else 1
        max_pairs_1 = int(counts1.max().item()) if counts1.numel() > 0 else 1
        self._max_fact_pairs = max(max_pairs_0, max_pairs_1, 1)

    # -- Public API -----------------------------------------------------------

    @property
    def max_fact_pairs(self) -> int:
        return self._max_fact_pairs

    def targeted_lookup(
        self,
        query_atoms: Tensor,    # [B, 3]
        max_results: int,
    ) -> Tuple[Tensor, Tensor]:
        """O(1) fact lookup via direct-indexed (pred, arg) composite-key tables.

        Preference order: (pred, arg0) → (pred, arg1) → pred-only.

        Returns:
            fact_idx: [B, max_results] indices into self.facts_idx.
            valid:    [B, max_results] boolean mask.
        """
        B = query_atoms.shape[0]
        device = query_atoms.device
        pad = self._padding_idx
        cno = self._constant_no
        F = self._arg0_order.shape[0]
        ks = self._key_scale

        preds = query_atoms[:, 0]
        a0 = query_atoms[:, 1]
        a1 = query_atoms[:, 2]

        is_const0 = (a0 <= cno) & (a0 != pad)
        is_const1 = (a1 <= cno) & (a1 != pad)

        offsets = torch.arange(max_results, device=device).unsqueeze(0)  # [1, K]
        clamp_max = max(F - 1, 0)

        # --- arg0 lookup ---
        key0 = preds * ks + a0
        safe_key0 = key0.clamp(0, self._arg0_starts.shape[0] - 1)
        left0 = self._arg0_starts[safe_key0]
        count0 = self._arg0_lens[safe_key0].clamp(max=max_results)
        sorted_idx0 = (left0.unsqueeze(1) + offsets).clamp(min=0, max=clamp_max)
        valid0 = (offsets < count0.unsqueeze(1)) & is_const0.unsqueeze(1)
        orig_idx0 = self._arg0_order[sorted_idx0.reshape(-1)].reshape(B, max_results)

        # --- arg1 lookup ---
        key1 = preds * ks + a1
        safe_key1 = key1.clamp(0, self._arg1_starts.shape[0] - 1)
        left1 = self._arg1_starts[safe_key1]
        count1 = self._arg1_lens[safe_key1].clamp(max=max_results)
        sorted_idx1 = (left1.unsqueeze(1) + offsets).clamp(min=0, max=clamp_max)
        valid1 = (offsets < count1.unsqueeze(1)) & is_const1.unsqueeze(1)
        orig_idx1 = self._arg1_order[sorted_idx1.reshape(-1)].reshape(B, max_results)

        # --- select: prefer arg0, fallback arg1 ---
        use_arg0 = is_const0.unsqueeze(1)
        fact_idx = torch.where(use_arg0, orig_idx0, orig_idx1)
        valid = torch.where(use_arg0, valid0, valid1)

        # --- pred-only fallback for both-variable queries ---
        if F > 0 and self._pred_starts.numel() > 0:
            both_var = ~is_const0 & ~is_const1 & (preds != pad)
            safe_pred = preds.clamp(0, self._pred_starts.shape[0] - 1)
            left_p = self._pred_starts[safe_pred]
            count_p = self._pred_lens[safe_pred].clamp(max=max_results)
            sorted_idx_p = (left_p.unsqueeze(1) + offsets).clamp(min=0, max=clamp_max)
            valid_p = (offsets < count_p.unsqueeze(1)) & both_var.unsqueeze(1)
            orig_idx_p = self._pred_order[sorted_idx_p.reshape(-1)].reshape(B, max_results)

            both_var_exp = both_var.unsqueeze(1)
            fact_idx = torch.where(both_var_exp, orig_idx_p, fact_idx)
            valid = torch.where(both_var_exp, valid_p, valid)

        return fact_idx, valid

    def enumerate(
        self,
        preds: Tensor,
        bound_args: Tensor,
        direction: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "ArgKeyFactIndex does not support enumerate(). "
            "Use InvertedFactIndex or BlockSparseFactIndex instead."
        )

    def __repr__(self) -> str:
        F = self.facts_idx.shape[0]
        return (
            f"ArgKeyFactIndex(num_facts={F}, "
            f"pack_base={self.pack_base}, "
            f"max_fact_pairs={self._max_fact_pairs})"
        )


# ---------------------------------------------------------------------------
# InvertedFactIndex  (TS-style offset-table enumeration)
# ---------------------------------------------------------------------------

class InvertedFactIndex(_FactIndexBase):
    """O(1) free-variable enumeration via (pred*E + bound) offset tables.

    Best for grounding engines: given a predicate and a bound argument,
    enumerate all entities that complete a valid fact.

    ``targeted_lookup()`` is **not supported** — use ``ArgKeyFactIndex``.
    """

    def __init__(
        self,
        facts_idx: Tensor,          # [F, 3]
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
        self._build_inverted_indices(device)

    # -- Index construction ---------------------------------------------------

    def _build_inverted_indices(self, device: torch.device) -> None:
        """Build (pred, subj)->objs and (pred, obj)->subjs offset tables."""
        facts = self.facts_idx
        E = self._num_entities
        P = self._num_predicates
        num_facts = facts.shape[0]

        if num_facts == 0:
            self.register_buffer(
                "ps_sorted_objs",
                torch.zeros(1, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "ps_offsets",
                torch.zeros(P * E + 1, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "po_sorted_subjs",
                torch.zeros(1, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "po_offsets",
                torch.zeros(P * E + 1, dtype=torch.long, device=device),
            )
            return

        preds = facts[:, 0].long()
        subjs = facts[:, 1].long()
        objs = facts[:, 2].long()

        # --- (pred, subj) -> objs ---
        ps_keys = preds * E + subjs
        ps_sort_idx = torch.argsort(ps_keys)
        ps_sorted_keys = ps_keys[ps_sort_idx]
        ps_sorted_objs = objs[ps_sort_idx]

        ps_offsets = torch.zeros(P * E + 1, dtype=torch.long, device=device)
        ones = torch.ones(num_facts, dtype=torch.long, device=device)
        ps_offsets.scatter_add_(0, ps_sorted_keys + 1, ones)
        ps_offsets = torch.cumsum(ps_offsets, dim=0)

        self.register_buffer("ps_sorted_objs", ps_sorted_objs)
        self.register_buffer("ps_offsets", ps_offsets)

        # --- (pred, obj) -> subjs ---
        po_keys = preds * E + objs
        po_sort_idx = torch.argsort(po_keys)
        po_sorted_keys = po_keys[po_sort_idx]
        po_sorted_subjs = subjs[po_sort_idx]

        po_offsets = torch.zeros(P * E + 1, dtype=torch.long, device=device)
        po_offsets.scatter_add_(0, po_sorted_keys + 1, ones)
        po_offsets = torch.cumsum(po_offsets, dim=0)

        self.register_buffer("po_sorted_subjs", po_sorted_subjs)
        self.register_buffer("po_offsets", po_offsets)

    # -- Public API -----------------------------------------------------------

    def enumerate(
        self,
        preds: Tensor,          # [N]
        bound_args: Tensor,     # [N]
        direction: Tensor,      # [N] int  0=objects, 1=subjects
    ) -> Tuple[Tensor, Tensor]:
        """O(1) enumeration via offset tables.

        Args:
            preds:      [N] predicate indices.
            bound_args: [N] the bound argument value.
            direction:  [N] int — 0 = enumerate objects (bound = subject),
                        1 = enumerate subjects (bound = object).

        Returns:
            candidates: [N, M] entity indices.
            valid:      [N, M] boolean mask.
        """
        N = preds.size(0)
        E = self._num_entities
        M = self._max_facts_per_query
        dev = preds.device

        keys = preds * E + bound_args

        # Look up in BOTH directions
        starts_ps = self.ps_offsets[keys]
        counts_ps = torch.clamp(self.ps_offsets[keys + 1] - starts_ps, 0, M)

        starts_po = self.po_offsets[keys]
        counts_po = torch.clamp(self.po_offsets[keys + 1] - starts_po, 0, M)

        # Select based on direction
        is_obj = (direction == 0)
        starts = torch.where(is_obj, starts_ps, starts_po)
        counts = torch.where(is_obj, counts_ps, counts_po)

        # Build output
        pos_indices = torch.arange(M, device=dev).unsqueeze(0).expand(N, -1)
        valid = pos_indices < counts.unsqueeze(1)
        global_indices = torch.clamp(
            starts.unsqueeze(1) + pos_indices,
            0,
            max(self.ps_sorted_objs.size(0), self.po_sorted_subjs.size(0)) - 1,
        )

        # Gather from both arrays, select via torch.where
        gi_ps = torch.clamp(global_indices, 0, self.ps_sorted_objs.size(0) - 1)
        gi_po = torch.clamp(global_indices, 0, self.po_sorted_subjs.size(0) - 1)
        vals_obj = self.ps_sorted_objs[gi_ps]
        vals_subj = self.po_sorted_subjs[gi_po]
        candidates = torch.where(is_obj.unsqueeze(1), vals_obj, vals_subj)

        return candidates, valid

    def targeted_lookup(
        self,
        query_atoms: Tensor,
        max_results: int,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "InvertedFactIndex does not support targeted_lookup(). "
            "Use ArgKeyFactIndex instead."
        )

    def __repr__(self) -> str:
        F = self.facts_idx.shape[0]
        return (
            f"InvertedFactIndex(num_facts={F}, "
            f"num_predicates={self._num_predicates}, "
            f"num_entities={self._num_entities}, "
            f"max_facts_per_query={self._max_facts_per_query})"
        )


# ---------------------------------------------------------------------------
# BlockSparseFactIndex  (dense [P, E, K] blocks)
# ---------------------------------------------------------------------------

class BlockSparseFactIndex(InvertedFactIndex):
    """Dense [P, E, K] blocks for O(1) enumerate and O(K) exists.

    Builds dense 3-D tensors when memory fits within *max_memory_mb*.
    Falls back to parent's offset-based implementation otherwise.
    """

    def __init__(
        self,
        facts_idx: Tensor,          # [F, 3]
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        num_entities: int,
        num_predicates: int,
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
    ) -> None:
        super().__init__(
            facts_idx, constant_no, padding_idx, device,
            num_entities, num_predicates, max_facts_per_query,
        )
        self._max_memory_mb = max_memory_mb
        self._build_dense_blocks(device)

    # -- Dense block construction ---------------------------------------------

    def _build_dense_blocks(self, device: torch.device) -> None:
        facts = self.facts_idx
        P = self._num_predicates
        E = self._num_entities
        M = self._max_facts_per_query
        num_facts = facts.shape[0]

        if num_facts == 0:
            self._use_dense = False
            return

        preds = facts[:, 0]
        subjs = facts[:, 1]
        objs = facts[:, 2]

        # Compute actual max K across all (pred, entity) slots
        ps_keys = preds * E + subjs
        po_keys = preds * E + objs

        _, ps_counts = torch.unique(ps_keys, return_counts=True)
        _, po_counts = torch.unique(po_keys, return_counts=True)

        max_ps = int(ps_counts.max().item()) if ps_counts.numel() > 0 else 0
        max_po = int(po_counts.max().item()) if po_counts.numel() > 0 else 0
        K = min(max(max_ps, max_po), M)

        # Memory check: 2 directions x P x E x K x 8 bytes
        mem_bytes = 2 * P * E * K * 8
        mem_mb = mem_bytes / (1024 ** 2)

        if mem_mb > self._max_memory_mb:
            self._use_dense = False
            return

        self._use_dense = True
        self._K = K
        self._pad_size = M - K

        # Build blocks on CPU then move (avoids large GPU scatter)
        ps_blocks = torch.zeros(P, E, K, dtype=torch.long)
        ps_block_counts = torch.zeros(P, E, dtype=torch.long)
        po_blocks = torch.zeros(P, E, K, dtype=torch.long)
        po_block_counts = torch.zeros(P, E, dtype=torch.long)

        # Vectorised fill via scatter
        p_cpu = preds.cpu()
        s_cpu = subjs.cpu()
        o_cpu = objs.cpu()

        for i in range(num_facts):
            p_i = int(p_cpu[i].item())
            s_i = int(s_cpu[i].item())
            o_i = int(o_cpu[i].item())

            idx_ps = int(ps_block_counts[p_i, s_i].item())
            if idx_ps < K:
                ps_blocks[p_i, s_i, idx_ps] = o_i
                ps_block_counts[p_i, s_i] = idx_ps + 1

            idx_po = int(po_block_counts[p_i, o_i].item())
            if idx_po < K:
                po_blocks[p_i, o_i, idx_po] = s_i
                po_block_counts[p_i, o_i] = idx_po + 1

        self.register_buffer("_ps_blocks", ps_blocks.to(device))
        self.register_buffer("_ps_block_counts", ps_block_counts.to(device))
        self.register_buffer("_po_blocks", po_blocks.to(device))
        self.register_buffer("_po_block_counts", po_block_counts.to(device))

    # -- Overrides ------------------------------------------------------------

    def enumerate(
        self,
        preds: Tensor,
        bound_args: Tensor,
        direction: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """O(1) dense enumerate — coalesced 3D read."""
        if not self._use_dense:
            return super().enumerate(preds, bound_args, direction)

        K = self._K
        dev = preds.device
        is_obj = (direction == 0)

        # Coalesced 3D reads
        cands_ps = self._ps_blocks[preds, bound_args]           # [N, K]
        counts_ps = self._ps_block_counts[preds, bound_args]    # [N]
        cands_po = self._po_blocks[preds, bound_args]           # [N, K]
        counts_po = self._po_block_counts[preds, bound_args]    # [N]

        candidates = torch.where(is_obj.unsqueeze(1), cands_ps, cands_po)
        counts = torch.where(is_obj, counts_ps, counts_po)

        pos_indices = torch.arange(K, device=dev).unsqueeze(0)
        valid = pos_indices < counts.unsqueeze(1)

        # Pad to max_facts_per_query (static size for torch.compile)
        if self._pad_size > 0:
            candidates = torch.nn.functional.pad(candidates, (0, self._pad_size))
            valid = torch.nn.functional.pad(valid, (0, self._pad_size))

        return candidates, valid

    def exists(self, atoms: Tensor) -> Tensor:
        """O(K) dense exists — membership check in block."""
        if not self._use_dense:
            return super().exists(atoms)

        K = self._K
        preds = atoms[:, 0]
        subjs = atoms[:, 1]
        objs = atoms[:, 2]

        block = self._ps_blocks[preds, subjs]              # [N, K]
        counts = self._ps_block_counts[preds, subjs]       # [N]

        pos_indices = torch.arange(K, device=atoms.device).unsqueeze(0)
        slot_valid = pos_indices < counts.unsqueeze(1)

        return ((block == objs.unsqueeze(1)) & slot_valid).any(dim=1)

    def __repr__(self) -> str:
        F = self.facts_idx.shape[0]
        if self._use_dense:
            dense_str = f", dense=True, K={self._K}"
        else:
            dense_str = ", dense=False (offset fallback)"
        return (
            f"BlockSparseFactIndex(num_facts={F}, "
            f"num_predicates={self._num_predicates}, "
            f"num_entities={self._num_entities}{dense_str})"
        )
