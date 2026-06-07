"""RuleIndex base — sorted rule storage + predicate→rule segment lookup.

Sorts rules by head predicate and builds CSR segment offsets so a batch of query
predicates maps to its candidate rules in one gather. This is everything sld/rtf
resolution needs; ``pbc`` adds binding analysis on top (see ``pbc.py``).

BIT-EXACT RECIPES (fingerprint-enforced):
  * stable argsort on the head-predicate column (ties keep input order)
  * CSR offsets: place each predicate's cumulative count at ``offsets[pred+1]``,
    then ``cummax`` to forward-fill absent predicates (monotone -> count 0)
  * ``max_rule_pairs`` (K_r) = max rules sharing a head predicate
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def build_csr_offsets(uniq: Tensor, cnts: Tensor, size: int, device) -> Tensor:
    """CSR segment offsets over ``size`` predicates.

    BIT-EXACT: ``offsets[p+1]-offsets[p]`` is predicate ``p``'s rule count (0 for
    absent predicates via the trailing ``cummax``). ``uniq``/``cnts`` come from a
    unique over the sorted head-predicate column.
    """
    seg = torch.zeros(size + 1, dtype=torch.long, device=device)
    cum = cnts.cumsum(0)
    mask = uniq < size
    seg[uniq[mask] + 1] = cum[mask]
    return seg.cummax(0).values


class RuleIndex(nn.Module):
    """Sorted rules with predicate→rule segment lookup (sld/rtf)."""

    def __init__(
        self,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        *,
        predicate_no: Optional[int] = None,
        padding_idx: int = 0,
        device: torch.device,
        order: Literal["original", "shuffle"] = "original",
        order_seed: int = 42,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        R = rules_heads_idx.shape[0]
        if R == 0:
            raise ValueError("rules_heads_idx is empty — cannot build a rule index without rules")

        if order == "shuffle":
            perm = self._shuffle_within_head_pred(rules_heads_idx, R, device, order_seed)
            rules_heads_idx = rules_heads_idx[perm]
            rules_bodies_idx = rules_bodies_idx[perm]
            rule_lens = rule_lens[perm]

        sort_perm = torch.argsort(rules_heads_idx[:, 0], stable=True)
        heads = rules_heads_idx.index_select(0, sort_perm).to(device)
        bodies = rules_bodies_idx.index_select(0, sort_perm).to(device)
        lens = rule_lens.index_select(0, sort_perm).to(device)

        preds = heads[:, 0]
        uniq, cnts = torch.unique_consecutive(preds, return_counts=True)
        num_pred = (predicate_no + 1 if predicate_no is not None
                    else int(preds.max().item()) + 2)
        seg_offsets = build_csr_offsets(uniq, cnts, num_pred, device)
        self._max_rule_pairs = int(cnts.max().item())

        self.register_buffer("rules_heads_sorted", heads)
        self.register_buffer("rules_bodies_sorted", bodies)
        self.register_buffer("rules_idx_sorted", sort_perm.to(device))
        self.register_buffer("rule_lens_sorted", lens)
        self.register_buffer("_seg_offsets", seg_offsets)

    # ── sizes / accessors ──
    @property
    def num_rules(self) -> int:
        return self.rules_heads_sorted.shape[0]

    @property
    def max_rule_pairs(self) -> int:
        return self._max_rule_pairs

    @property
    def K_r(self) -> int:
        """Rules per predicate."""
        return self._max_rule_pairs

    @property
    def rules_heads(self) -> Tensor:
        return self.rules_heads_sorted

    @property
    def rules_bodies(self) -> Tensor:
        return self.rules_bodies_sorted

    @property
    def rule_lens(self) -> Tensor:
        return self.rule_lens_sorted

    @torch.no_grad()
    def lookup(self, query_preds: Tensor,
               max_pairs: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Predicate→rule segment lookup.

        ``[B] -> (item_idx[B,K], valid[B,K], query_idx[B,K])`` where item_idx are
        rows of the sorted rule arrays and valid masks the live slots.
        """
        B = query_preds.shape[0]
        dev = query_preds.device
        if B == 0:
            z = torch.zeros((0, max_pairs), dtype=torch.long, device=dev)
            return z, z.bool(), z
        qp = query_preds.long().clamp(0, self._seg_offsets.shape[0] - 2)
        starts = self._seg_offsets[qp]
        lens = (self._seg_offsets[qp + 1] - starts).clamp(max=max_pairs)
        pos = torch.arange(max_pairs, device=dev).unsqueeze(0)
        return (starts.unsqueeze(1) + pos,
                pos < lens.unsqueeze(1),
                torch.arange(B, device=dev).unsqueeze(1).expand(-1, max_pairs))

    @staticmethod
    def _shuffle_within_head_pred(rules_heads_idx: Tensor, R: int,
                                  device: torch.device, seed: int) -> Tensor:
        """Permutation that shuffles rules within each head-predicate group."""
        gen = torch.Generator(device=rules_heads_idx.device).manual_seed(seed)
        head_preds = rules_heads_idx[:, 0]
        sort_order = torch.argsort(head_preds, stable=True)
        sorted_preds = head_preds[sort_order]
        num_preds = int(sorted_preds.max().item()) + 1 if R > 0 else 1
        counts = torch.bincount(sorted_preds.long(), minlength=num_preds)
        starts = torch.zeros(num_preds + 1, dtype=torch.long, device=rules_heads_idx.device)
        starts[1:] = counts.cumsum(0)
        perm = sort_order.clone()
        for p in range(num_preds):
            s, e = starts[p].item(), starts[p + 1].item()
            if e - s > 1:
                perm[s:e] = sort_order[s + torch.randperm(
                    e - s, device=rules_heads_idx.device, generator=gen)]
        return perm


__all__ = ["RuleIndex", "build_csr_offsets"]
