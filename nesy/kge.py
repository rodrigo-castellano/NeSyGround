"""KGE min-conjunction scorer — PostResolutionHook.

Scores groundings by min(KGE body atom scores), selects top-k.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class KGEScorer(nn.Module):
    """Score groundings by KGE min-conjunction, select top-k.

    kge_model interface:
        kge_model.score_atoms(preds, subjs, objs) -> Tensor of scalar scores.

    Args:
        kge_model:     nn.Module with score_atoms().
        output_budget: number of groundings to keep.
        padding_idx:   padding value (for body_active detection).
    """

    def __init__(
        self,
        kge_model: nn.Module,
        output_budget: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        self._kge_ref: list = [kge_model]
        self._output_tG = output_budget
        self._padding_idx = padding_idx

    def apply(
        self,
        body: Tensor,       # [B, tG, M, 3]
        mask: Tensor,       # [B, tG]
        rule_idx: Tensor,   # [B, tG]
    ) -> tuple:
        B, tG_in, M, _ = body.shape
        dev = body.device
        kge = self._kge_ref[0]

        # Body-active mask
        body_active = body[..., 0] != self._padding_idx  # [B, tG, M]

        # KGE atom scores
        atom_scores = kge.score_atoms(
            body[..., 0].reshape(-1),
            body[..., 1].reshape(-1),
            body[..., 2].reshape(-1),
        ).view(B, tG_in, M)

        # Mask inactive → large value so min ignores them
        atom_scores = torch.where(body_active, atom_scores,
                                  torch.tensor(1e9, device=dev))

        # Min-conjunction score, mask invalid groundings
        scores = atom_scores.min(dim=-1).values
        scores = torch.where(mask, scores, torch.tensor(-1e9, device=dev))

        # Top-k
        from grounder.nesy import _topk_select
        return _topk_select(body, mask, rule_idx, scores, self._output_tG)

    def __repr__(self) -> str:
        return f"KGEScorer(output_budget={self._output_tG})"
