"""Neural attention scorer — PostResolutionHook.

Scores groundings by learned attention MLP over body embeddings, selects top-k.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GroundingAttention(nn.Module):
    """Learned attention over body atom embeddings. MLP: input_size → hidden → 1."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 1),
        )

    def forward(self, body_emb_flat: Tensor) -> Tensor:
        return self.net(body_emb_flat)


class NeuralScorer(nn.Module):
    """Score groundings by learned attention, select top-k.

    kge_model interface:
        kge_model.embed_atoms(subjs, objs) -> [N, E] embeddings.
        kge_model.atom_embedding_size -> int.

    Args:
        kge_model:     nn.Module with embed_atoms() and atom_embedding_size.
        output_budget: number of groundings to keep.
        padding_idx:   padding value.
        max_body:      M (max body atoms per rule).
    """

    def __init__(
        self,
        kge_model: nn.Module,
        output_budget: int,
        padding_idx: int,
        max_body: int,
    ) -> None:
        super().__init__()
        self._kge_ref: list = [kge_model]
        self._output_tG = output_budget
        self._padding_idx = padding_idx
        E = kge_model.atom_embedding_size
        self._attention = GroundingAttention(max_body * E)
        self._atom_emb_size = E

    def apply(
        self,
        body: Tensor,       # [B, tG, M, 3]
        mask: Tensor,       # [B, tG]
        rule_idx: Tensor,   # [B, tG]
    ) -> tuple:
        B, tG_in, M, _ = body.shape
        E = self._atom_emb_size
        kge = self._kge_ref[0]

        body_active = body[..., 0] != self._padding_idx
        emb = kge.embed_atoms(
            body[..., 1].reshape(-1),
            body[..., 2].reshape(-1),
        ).view(B, tG_in, M, E)
        emb = emb * body_active.unsqueeze(-1).float()

        scores = self._attention(emb.reshape(B * tG_in, M * E)).view(B, tG_in)
        scores = scores + (mask.float() - 1.0) * 1e9

        from grounder.nesy import _topk_select
        return _topk_select(body, mask, rule_idx, scores, self._output_tG)

    def __repr__(self) -> str:
        return f"NeuralScorer(output_budget={self._output_tG})"
