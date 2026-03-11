"""Rule index: segment-based and table-based predicate-to-rule lookup.

Sorts rules by head predicate and provides two lookup strategies:
- Segment-based (BE style): fixed-width offset lookup via pairs_via_predicate_ranges
- Table-based (TS style): direct [P, R_eff] index table with mask

All tensors are registered as buffers for device transfer and state_dict support.

Part of the grounder package.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RuleIndex(nn.Module):
    """Predicate-to-rule lookup index with segment and table strategies.

    Args:
        rules_heads_idx: [R, 3] rule head atoms (pred, arg0, arg1)
        rules_bodies_idx: [R, Bmax, 3] rule body atoms
        rule_lens: [R] number of body atoms per rule
        device: target device
        predicate_no: total number of predicates (exclusive upper bound);
            if None, inferred from rules_heads_idx
        padding_idx: index used for padding
    """

    def __init__(
        self,
        rules_heads_idx: Tensor,  # [R, 3]
        rules_bodies_idx: Tensor,  # [R, Bmax, 3]
        rule_lens: Tensor,  # [R]
        device: torch.device,
        predicate_no: Optional[int] = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        R = rules_heads_idx.shape[0]

        if R > 0:
            # Sort rules by head predicate
            order = torch.argsort(rules_heads_idx[:, 0], stable=True)
            heads_sorted = rules_heads_idx.index_select(0, order).to(device)
            bodies_sorted = rules_bodies_idx.index_select(0, order).to(device)
            idx_sorted = order.to(device)
            lens_sorted = rule_lens.index_select(0, order).to(device)

            # --- Segment-based index (BE style) ---
            preds = heads_sorted[:, 0]
            uniq, cnts = torch.unique_consecutive(preds, return_counts=True)
            seg_input = torch.zeros(cnts.shape[0], dtype=torch.long, device=device)
            seg_input[1:] = cnts[:-1]
            seg_starts = torch.cumsum(seg_input, dim=0)

            num_pred = (predicate_no + 1) if predicate_no is not None else int(preds.max().item()) + 2
            rule_seg_starts = torch.zeros(num_pred, dtype=torch.long, device=device)
            rule_seg_lens = torch.zeros(num_pred, dtype=torch.long, device=device)
            mask = uniq < num_pred
            rule_seg_starts[uniq[mask]] = seg_starts[mask]
            rule_seg_lens[uniq[mask]] = cnts[mask]

            _max_rule_pairs = int(cnts.max().item())

            # --- Table-based index (TS style) ---
            r_eff = _max_rule_pairs
            pred_rule_indices = torch.zeros(num_pred, r_eff, dtype=torch.long, device=device)
            pred_rule_mask = torch.zeros(num_pred, r_eff, dtype=torch.bool, device=device)
            for p_idx in range(num_pred):
                start = int(rule_seg_starts[p_idx].item())
                length = int(rule_seg_lens[p_idx].item())
                if length > 0:
                    pred_rule_indices[p_idx, :length] = idx_sorted[start:start + length]
                    pred_rule_mask[p_idx, :length] = True
        else:
            heads_sorted = rules_heads_idx.to(device)
            bodies_sorted = rules_bodies_idx.to(device)
            idx_sorted = torch.zeros(0, dtype=torch.long, device=device)
            lens_sorted = rule_lens.to(device)
            rule_seg_starts = torch.zeros(1, dtype=torch.long, device=device)
            rule_seg_lens = torch.zeros(1, dtype=torch.long, device=device)
            _max_rule_pairs = 0
            r_eff = 0
            num_pred = 1
            pred_rule_indices = torch.zeros(1, 0, dtype=torch.long, device=device)
            pred_rule_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)

        # Register all tensors as buffers
        self.register_buffer("rules_heads_sorted", heads_sorted)
        self.register_buffer("rules_bodies_sorted", bodies_sorted)
        self.register_buffer("rules_idx_sorted", idx_sorted)
        self.register_buffer("rule_lens_sorted", lens_sorted)
        self.register_buffer("rule_seg_starts", rule_seg_starts)
        self.register_buffer("rule_seg_lens", rule_seg_lens)
        self.register_buffer("pred_rule_indices", pred_rule_indices)
        self.register_buffer("pred_rule_mask", pred_rule_mask)

        self._max_rule_pairs = _max_rule_pairs
        self._R_eff = r_eff

    # --- Properties ---

    @property
    def max_rule_pairs(self) -> int:
        """Maximum number of rules sharing a single head predicate."""
        return self._max_rule_pairs

    @property
    def R_eff(self) -> int:
        """Width of the table-based index (= max_rule_pairs)."""
        return self._R_eff

    # Convenience aliases used by operations.py
    @property
    def rules_heads(self) -> Tensor:
        return self.rules_heads_sorted

    @property
    def rules_bodies(self) -> Tensor:
        return self.rules_bodies_sorted

    @property
    def rule_lens(self) -> Tensor:
        return self.rule_lens_sorted

    # --- Segment-based lookup (BE style) ---

    @torch.no_grad()
    def lookup_by_segments(
        self,
        query_preds: Tensor,  # [B]
        max_pairs: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Segment-based predicate lookup with fixed output shape.

        Args:
            query_preds: [B] predicate ID per query
            max_pairs: max pairs per query (output width)

        Returns:
            item_idx:   [B, max_pairs] indices into sorted rule arrays
            valid_mask: [B, max_pairs] which pairs are valid
            query_idx:  [B, max_pairs] repeated query indices
        """
        B = query_preds.shape[0]
        device = query_preds.device

        if B == 0:
            return (
                torch.zeros((0, max_pairs), dtype=torch.long, device=device),
                torch.zeros((0, max_pairs), dtype=torch.bool, device=device),
                torch.zeros((0, max_pairs), dtype=torch.long, device=device),
            )

        lens = self.rule_seg_lens[query_preds.long()]      # [B]
        starts = self.rule_seg_starts[query_preds.long()]   # [B]

        offsets = torch.arange(max_pairs, device=device, dtype=torch.long).unsqueeze(0)  # [1, max_pairs]

        item_idx = starts.unsqueeze(1) + offsets     # [B, max_pairs]
        valid_mask = offsets < lens.unsqueeze(1)      # [B, max_pairs]
        query_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(-1, max_pairs)

        return item_idx, valid_mask, query_idx

    # --- Table-based lookup (TS style) ---

    @torch.no_grad()
    def lookup_by_table(
        self,
        query_preds: Tensor,  # [N]
    ) -> Tuple[Tensor, Tensor]:
        """Table-based predicate lookup returning original rule indices + mask.

        Args:
            query_preds: [N] predicate IDs

        Returns:
            rule_idx: [N, R_eff] original rule indices per query predicate
            mask:     [N, R_eff] validity mask
        """
        safe_preds = query_preds.long().clamp(0, self.pred_rule_indices.shape[0] - 1)
        rule_idx = self.pred_rule_indices[safe_preds]   # [N, R_eff]
        mask = self.pred_rule_mask[safe_preds]           # [N, R_eff]
        return rule_idx, mask
