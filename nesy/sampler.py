"""Random sampler — PostResolutionHook.

Selects a random subset of valid groundings (train) or valid-first (eval).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RandomSampler(nn.Module):
    """Random subsampling of groundings.

    Train: random scores → topk. Eval: deterministic valid-first.

    Args:
        output_budget: number of groundings to keep.
    """

    def __init__(self, output_budget: int) -> None:
        super().__init__()
        self._output_tG = output_budget

    def apply(
        self,
        body: Tensor,       # [B, tG, M, 3]
        mask: Tensor,       # [B, tG]
        rule_idx: Tensor,   # [B, tG]
    ) -> tuple:
        B, tG_in = mask.shape
        dev = mask.device

        if self.training:
            scores = torch.rand(B, tG_in, device=dev) * mask.float()
        else:
            scores = mask.float()

        from grounder.nesy import _topk_select
        return _topk_select(body, mask, rule_idx, scores, self._output_tG)

    def __repr__(self) -> str:
        return f"RandomSampler(output_budget={self._output_tG})"
