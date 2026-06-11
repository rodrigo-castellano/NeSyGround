"""InvertedFactIndex — free-variable enumeration for pbc resolution.

For a (predicate, bound argument) and a direction, list the bound values of the
other argument. Backed by two offset tables: ``ps`` keyed by (pred, subject)
listing objects, ``po`` keyed by (pred, object) listing subjects.

The tables are sized on the *actual* data range (max id in ``facts_idx`` + 1),
not the loader's inflated ``predicate_no``/``constant_no``: the loader pads those
so downstream embeddings fit, which on wn18rr would blow ``P*E`` up to ~1.6e9
slots for 11 real predicates. Out-of-range inputs (padded body atoms) are masked
to an empty result instead.

BIT-EXACT RECIPES (fingerprint-enforced):
  * num_predicates/num_entities = per-column max(facts_idx) + 1
  * offset table = scatter_add ones at ``sorted_key+1`` then cumsum
  * ``max_facts_per_query`` auto-clamps DOWN to the true max slot occupancy
  * enumerate: direction 0 reads ps (objects), direction 1 reads po (subjects)
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from grounder.data.fact_index.base import FactIndex


class InvertedFactIndex(FactIndex):
    """O(1) free-variable enumeration via (pred, bound-arg) offset tables."""

    def __init__(
        self,
        facts_idx: Tensor,
        *,
        constant_no: int,
        predicate_no: int,
        padding_idx: int,
        device: torch.device,
        max_facts_per_query: int = 64,
        **_: object,
    ) -> None:
        super().__init__(facts_idx, constant_no=constant_no,
                         padding_idx=padding_idx, device=device)
        # Size on the real data range, not the inflated loader padding.
        self._num_predicates = int(facts_idx[:, 0].max().item()) + 1
        self._num_entities = max(int(facts_idx[:, 1].max().item()),
                                 int(facts_idx[:, 2].max().item())) + 1
        self._max_facts_per_query = max_facts_per_query
        self._build_offset_tables(device)
        # Clamp the per-query budget DOWN to the true max occupancy: enumerate
        # allocates [N, M] every call, so an oversized M wastes memory once N
        # grows with state fan-out. Never truncates a real slot.
        ps_span = self._ps_offsets[1:] - self._ps_offsets[:-1]
        po_span = self._po_offsets[1:] - self._po_offsets[:-1]
        actual_max = int(torch.maximum(ps_span.max(), po_span.max()).item())
        if actual_max < self._max_facts_per_query:
            self._max_facts_per_query = max(actual_max, 1)

    @staticmethod
    def _offset_table(keys: Tensor, values: Tensor, num_slots: int,
                      device: torch.device) -> Tuple[Tensor, Tensor]:
        """Sorted values + CSR offsets keyed by ``keys``.

        BIT-EXACT: argsort the keys, scatter-add a 1 at ``sorted_key+1`` for each
        fact, then cumsum -> a monotone offset array where empty keys span 0.
        """
        order = torch.argsort(keys)
        sorted_keys, sorted_vals = keys[order], values[order]
        offsets = torch.zeros(num_slots + 1, dtype=torch.long, device=device)
        ones = torch.ones(keys.size(0), dtype=torch.long, device=device)
        offsets.scatter_add_(0, sorted_keys + 1, ones)
        offsets = offsets.cumsum(0)
        return sorted_vals, offsets

    def _build_offset_tables(self, device: torch.device) -> None:
        facts = self.facts_idx
        E = self._num_entities
        preds, subjs, objs = facts[:, 0].long(), facts[:, 1].long(), facts[:, 2].long()
        num_slots = self._num_predicates * E

        ps_vals, ps_off = self._offset_table(preds * E + subjs, objs, num_slots, device)
        self.register_buffer("_ps_values", ps_vals)
        self.register_buffer("_ps_offsets", ps_off)

        po_vals, po_off = self._offset_table(preds * E + objs, subjs, num_slots, device)
        self.register_buffer("_po_values", po_vals)
        self.register_buffer("_po_offsets", po_off)

    def enumerate(self, preds: Tensor, bound_args: Tensor,
                  direction: Tensor) -> Tuple[Tensor, Tensor]:
        """``[N] preds, [N] bound_args, [N] direction -> (cands[N,M], valid[N,M])``.

        direction: 0 = bound subject, enumerate objects (ps); 1 = bound object,
        enumerate subjects (po). Out-of-range (padded) inputs return empty.
        """
        N = preds.size(0)
        M = self._max_facts_per_query
        dev = preds.device

        valid_input = (preds < self._num_predicates) & (bound_args < self._num_entities)
        safe_preds = torch.where(valid_input, preds, torch.zeros_like(preds))
        safe_args = torch.where(valid_input, bound_args, torch.zeros_like(bound_args))
        keys = safe_preds * self._num_entities + safe_args
        is_obj = (direction == 0)

        starts_ps = self._ps_offsets[keys]
        counts_ps = (self._ps_offsets[keys + 1] - starts_ps).clamp(0, M)
        starts_po = self._po_offsets[keys]
        counts_po = (self._po_offsets[keys + 1] - starts_po).clamp(0, M)

        starts = torch.where(is_obj, starts_ps, starts_po)
        counts = torch.where(is_obj, counts_ps, counts_po)
        counts = torch.where(valid_input, counts, torch.zeros_like(counts))

        # Memory-lean candidate gather. Both value arrays hold exactly one
        # entry per fact (``_offset_table`` returns ``values[order]``), so
        # their sizes are equal and the historical per-array re-clamps were
        # no-ops — dropped. ``[N, M]`` int64 transients are the fill-stage
        # peak at production shapes (wn18rr: 7 simultaneous [502k, 64]
        # tensors); the broadcast add + in-place clamp + ONE stacked-table
        # gather leaves exactly ``gi`` + ``cands``, and the eager path
        # row-chunks the gather (per-row pure) so the ``gi`` transient stays
        # ~128 MB next to the unavoidable [N, M] output.
        F = self._ps_values.size(0)
        pos = torch.arange(M, device=dev).unsqueeze(0)            # [1, M] (broadcast)
        valid = pos < counts.unsqueeze(1)
        # vals2 is [2, F] (tiny); row 0 = ps (direction 0), row 1 = po.
        vals2 = torch.stack([self._ps_values, self._po_values])
        d = (~is_obj).long()
        chunk = max(1, 64_000_000 // (8 * max(M, 1)))              # 64 MB gi budget
        if torch.compiler.is_compiling() or N <= chunk:
            gi = (starts.unsqueeze(1) + pos).clamp_(0, F - 1)      # [N, M]
            cands = vals2[d.unsqueeze(1), gi]                      # [N, M]
            return cands, valid
        cands = torch.empty((N, M), dtype=vals2.dtype, device=dev)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            gi = (starts[s:e].unsqueeze(1) + pos).clamp_(0, F - 1)
            cands[s:e] = vals2[d[s:e].unsqueeze(1), gi]
        return cands, valid

    def __repr__(self) -> str:
        return (f"InvertedFactIndex(F={self.num_facts}, "
                f"P={self._num_predicates}, E={self._num_entities})")


__all__ = ["InvertedFactIndex"]
