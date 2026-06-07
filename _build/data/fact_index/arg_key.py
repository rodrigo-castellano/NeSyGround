"""ArgKeyFactIndex — targeted lookup for sld/rtf resolution.

Given a goal with one bound argument, return the facts that match it so the free
argument can be bound. Backed by three CSR segment tables keyed by
``pred*key_scale + arg`` (one for arg0, one for arg1) plus a predicate-only
table for the all-variable case.

BIT-EXACT RECIPES (fingerprint-enforced):
  * ``key_scale`` = max(constant_no, padding_idx) + 2   (same base family as pack)
  * CSR build: ``argsort(stable)`` -> ``unique_consecutive`` counts ->
    scatter cumulative counts at ``key+1`` -> ``cummax`` forward-fill of gaps
  * lookup branch order: arg0-const wins over arg1-const wins over both-var
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder._build.data.fact_index.base import FactIndex


class ArgKeyFactIndex(FactIndex):
    """O(1) targeted fact lookup via (pred, arg) composite-key CSR tables."""

    def __init__(
        self,
        facts_idx: Tensor,
        *,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        pack_base: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__(facts_idx, constant_no=constant_no,
                         padding_idx=padding_idx, device=device, pack_base=pack_base)
        self._build_tables(device)

    @staticmethod
    def _segment_table(keys: Tensor, num_slots: int,
                       device: torch.device) -> Tuple[Tensor, Tensor]:
        """CSR segment index over composite keys.

        Returns ``(order, offsets)`` where ``order`` is the argsort permutation
        and ``offsets[k]..offsets[k+1]`` is key ``k``'s span in ``order``.

        BIT-EXACT: stable argsort, then place each unique key's cumulative count
        at ``offsets[key+1]`` and ``cummax`` to forward-fill empty keys — this
        exact gap-fill makes ``offsets`` monotone so empty keys yield count 0.
        """
        order = keys.argsort(stable=True)
        unique, counts = torch.unique_consecutive(keys[order], return_counts=True)
        offsets = torch.zeros(num_slots + 1, dtype=torch.long, device=device)
        offsets[unique + 1] = counts.cumsum(0)
        offsets = offsets.cummax(0).values
        return order, offsets

    def _build_tables(self, device: torch.device) -> None:
        facts = self.facts_idx
        preds, arg0, arg1 = (facts[:, 0].long(), facts[:, 1].long(), facts[:, 2].long())
        ks = max(int(self._constant_no), int(self._padding_idx)) + 2
        self._key_scale = ks

        a0_order, a0_off = self._segment_table(
            preds * ks + arg0, int((preds * ks + arg0).max()) + 2, device)
        a1_order, a1_off = self._segment_table(
            preds * ks + arg1, int((preds * ks + arg1).max()) + 2, device)
        p_order, p_off = self._segment_table(preds, int(preds.max()) + 2, device)

        self.register_buffer("_a0_order", a0_order)
        self.register_buffer("_a0_offsets", a0_off)
        self.register_buffer("_a1_order", a1_order)
        self.register_buffer("_a1_offsets", a1_off)
        self.register_buffer("_p_order", p_order)
        self.register_buffer("_p_offsets", p_off)

        def _max_span(offsets: Tensor) -> int:
            if offsets.numel() < 2:
                return 1
            return max(int((offsets[1:] - offsets[:-1]).max().item()), 1)

        self._max_fact_pairs = max(_max_span(a0_off), _max_span(a1_off), 1)

    @property
    def max_fact_pairs(self) -> int:
        return self._max_fact_pairs

    def targeted_lookup(self, query_atoms: Tensor,
                        max_results: int) -> Tuple[Tensor, Tensor]:
        """Bind the free argument of each goal. ``[B,3] -> (fact_idx[B,K], valid[B,K])``.

        BIT-EXACT: branch precedence is arg0-const, then arg1-const, then
        both-variable (predicate-only). ``valid`` requires both an in-range slot
        position AND the corresponding argument actually being a constant.
        """
        B = query_atoms.shape[0]
        dev = query_atoms.device
        cno, pad, ks = self._constant_no, self._padding_idx, self._key_scale
        F = self._a0_order.shape[0]
        clamp_max = max(F - 1, 0)

        preds, a0, a1 = query_atoms[:, 0], query_atoms[:, 1], query_atoms[:, 2]
        is_c0 = (a0 <= cno) & (a0 != pad)
        is_c1 = (a1 <= cno) & (a1 != pad)
        pos = torch.arange(max_results, device=dev).unsqueeze(0)

        def _lookup(order, offsets, keys, is_const):
            safe = keys.clamp(0, offsets.shape[0] - 2)
            left = offsets[safe]
            cnt = (offsets[safe + 1] - left).clamp(max=max_results)
            gi = (left.unsqueeze(1) + pos).clamp(0, clamp_max)
            valid = (pos < cnt.unsqueeze(1)) & is_const.unsqueeze(1)
            return order[gi.reshape(-1)].reshape(B, max_results), valid

        fi0, v0 = _lookup(self._a0_order, self._a0_offsets, preds * ks + a0, is_c0)
        fi1, v1 = _lookup(self._a1_order, self._a1_offsets, preds * ks + a1, is_c1)

        use0 = is_c0.unsqueeze(1)
        fact_idx = torch.where(use0, fi0, fi1)
        valid = torch.where(use0, v0, v1)

        if F > 0 and self._p_offsets.numel() > 0:
            both_var = ~is_c0 & ~is_c1 & (preds != pad)
            fip, vp = _lookup(self._p_order, self._p_offsets, preds, both_var)
            bv = both_var.unsqueeze(1)
            fact_idx = torch.where(bv, fip, fact_idx)
            valid = torch.where(bv, vp, valid)

        return fact_idx, valid

    def __repr__(self) -> str:
        return f"ArgKeyFactIndex(F={self.num_facts}, K={self._max_fact_pairs})"


__all__ = ["ArgKeyFactIndex"]
