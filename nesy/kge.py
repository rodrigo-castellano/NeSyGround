"""KGE-scored and neural attention grounders.

KGEGrounder: Score groundings by min(KGE body atom scores), top-k selection.
NeuralGrounder: Score groundings by learned attention MLP, top-k selection.
Both inherit from ParametrizedBCGrounder with 2x over-provisioning.

Compatible with torch.compile(fullgraph=True, mode='reduce-overhead').
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.grounders.parametrized import ParametrizedBCGrounder
from grounder.types import ForwardResult


# ======================================================================
# Attention MLP
# ======================================================================


class GroundingAttention(nn.Module):
    """Learned attention over body atom embeddings.

    MLP: input_size -> hidden -> 1

    Args:
        input_size: M * E where M = max_body_atoms, E = atom_embedding_size.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        hidden = input_size // 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, body_emb_flat: Tensor) -> Tensor:
        """Score grounding from flattened body embeddings.

        Args:
            body_emb_flat: [N, input_size] flattened body embeddings per grounding.

        Returns:
            scores: [N, 1]
        """
        return self.net(body_emb_flat)


# ======================================================================
# KGEGrounder
# ======================================================================


class KGEGrounder(ParametrizedBCGrounder):
    """KGE min-conjunction scored grounder.

    Inherits from ParametrizedBCGrounder with a 2x over-provisioned inner
    budget.  Scores resulting groundings by min(KGE body atom scores), then
    selects the top-k groundings to fit the output budget.

    The ``kge_model`` is stored as ``self._kge_ref = [kge_model]`` to avoid
    ``nn.Module`` auto-registration (the KGE model is owned elsewhere).

    Constructor takes the same raw tensor args as ParametrizedBCGrounder,
    plus ``kge_model`` and ``output_budget``.

    kge_model interface:
        - ``kge_model.score_atoms(preds, subjs, objs) -> Tensor`` scalar scores.

    Args:
        facts_idx:           [F, 3] fact triples.
        rules_heads_idx:     [R, 3] rule head atoms.
        rules_bodies_idx:    [R, Bmax, 3] rule body atoms (padded).
        rule_lens:           [R] body lengths.
        constant_no:         highest constant index.
        padding_idx:         padding value.
        device:              target device.
        kge_model:           nn.Module with ``score_atoms(preds, subjs, objs)``.
        output_budget:       output grounding budget (defaults to max_total_groundings).
        max_total_groundings: base total groundings (used as output_budget default).
        **kwargs:            forwarded to ParametrizedBCGrounder.
    """

    def __init__(
        self,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        *,
        kge_model: nn.Module,
        output_budget: Optional[int] = None,
        max_total_groundings: int = 64,
        **kwargs,
    ) -> None:
        if output_budget is None:
            output_budget = max_total_groundings

        # Inner grounder gets 2x the output budget for over-provisioning
        inner_budget = output_budget * 2

        super().__init__(
            facts_idx,
            rules_heads_idx,
            rules_bodies_idx,
            rule_lens,
            constant_no,
            padding_idx,
            device,
            max_total_groundings=inner_budget,
            **kwargs,
        )

        # Store KGE model ref without nn.Module auto-registration
        self._kge_ref: list = [kge_model]

        # Chunk size for score_atoms to avoid OOM on large eval batches.
        # 0 = no chunking (default, safe for torch.compile).
        # Set to e.g. 100_000 externally for benchmark runs (no_compile mode).
        self._score_chunk_size: int = 0

        # Save inner capacity before clamping
        self._inner_tG: int = self.effective_total_G

        # Output budget -- clamped to inner capacity so topk never overflows
        self._output_tG: int = min(output_budget, self._inner_tG)
        self.effective_total_G = self._output_tG

    def forward(
        self,
        queries: Tensor,       # [B, 3]
        query_mask: Tensor,    # [B]
    ) -> ForwardResult:
        """Ground rules and select top groundings by KGE min-conjunction score.

        Args:
            queries:    [B, 3] [pred_idx, subj_idx, obj_idx].
            query_mask: [B] valid queries.

        Returns:
            ForwardResult with selected groundings.
        """
        result = super().forward(queries, query_mask)
        body = result.collected_body    # [B, tG_in, M, 3]
        mask = result.collected_mask    # [B, tG_in]
        ridx = result.collected_ridx   # [B, tG_in]

        B, tG_in, M, _ = body.shape
        dev = body.device

        # Get num_body_atoms per grounding for body-active masking
        num_body_q = self.num_body_atoms[ridx]  # [B, tG_in]
        atom_idx = torch.arange(M, device=dev).view(1, 1, M)
        body_active = atom_idx < num_body_q.unsqueeze(-1)  # [B, tG_in, M]

        # KGE min-conjunction scoring
        preds_flat = body[..., 0].reshape(-1)
        subjs_flat = body[..., 1].reshape(-1)
        objs_flat = body[..., 2].reshape(-1)
        kge = self._kge_ref[0]

        if self._score_chunk_size > 0:
            cs = self._score_chunk_size
            n = preds_flat.size(0)
            chunks = [
                kge.score_atoms(
                    preds_flat[s : s + cs],
                    subjs_flat[s : s + cs],
                    objs_flat[s : s + cs],
                )
                for s in range(0, n, cs)
            ]
            atom_scores = torch.cat(chunks, dim=0)
        else:
            atom_scores = kge.score_atoms(preds_flat, subjs_flat, objs_flat)

        atom_scores = atom_scores.view(B, tG_in, M)

        # Mask inactive body atoms with large value so min ignores them
        atom_scores = torch.where(
            body_active,
            atom_scores,
            torch.tensor(1e9, device=dev),
        )

        # Min conjunction: grounding score = min over active body atoms
        scores = atom_scores.min(dim=-1).values  # [B, tG_in]

        # Mask invalid groundings with -inf
        scores = torch.where(
            mask,
            scores,
            torch.tensor(-1e9, device=dev),
        )

        # Top-k selection
        out_tG = self._output_tG
        _, top_idx = scores.topk(out_tG, dim=1, largest=True, sorted=False)

        # Gather selected groundings
        idx_body = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3)
        out_body = body.gather(1, idx_body)
        out_mask = mask.gather(1, top_idx)
        out_ridx = ridx.gather(1, top_idx)
        out_count = out_mask.sum(dim=1)

        return ForwardResult(
            collected_body=out_body,
            collected_mask=out_mask,
            collected_count=out_count,
            collected_ridx=out_ridx,
        )

    def __repr__(self) -> str:
        return (
            f"KGEGrounder(output_budget={self._output_tG}, "
            f"inner_tG={self._inner_tG}, "
            f"depth={self.depth}, width={self.width})"
        )


# ======================================================================
# NeuralGrounder
# ======================================================================


class NeuralGrounder(ParametrizedBCGrounder):
    """Neural attention scored grounder.

    Inherits from ParametrizedBCGrounder with a 2x over-provisioned inner
    budget.  Scores resulting groundings by a learned attention MLP over
    body atom embeddings, then selects the top-k groundings.

    The ``kge_model`` must provide ``embed_atoms(subjs, objs) -> [N, E]``
    and ``atom_embedding_size`` (int property).

    Constructor takes the same raw tensor args as ParametrizedBCGrounder,
    plus ``kge_model`` and ``output_budget``.

    Args:
        facts_idx:           [F, 3] fact triples.
        rules_heads_idx:     [R, 3] rule head atoms.
        rules_bodies_idx:    [R, Bmax, 3] rule body atoms (padded).
        rule_lens:           [R] body lengths.
        constant_no:         highest constant index.
        padding_idx:         padding value.
        device:              target device.
        kge_model:           nn.Module with ``embed_atoms(subjs, objs)`` and
                             ``atom_embedding_size``.
        output_budget:       output grounding budget (defaults to max_total_groundings).
        max_total_groundings: base total groundings (used as output_budget default).
        **kwargs:            forwarded to ParametrizedBCGrounder.
    """

    def __init__(
        self,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        *,
        kge_model: nn.Module,
        output_budget: Optional[int] = None,
        max_total_groundings: int = 64,
        **kwargs,
    ) -> None:
        if output_budget is None:
            output_budget = max_total_groundings

        # Inner grounder gets 2x the output budget for over-provisioning
        inner_budget = output_budget * 2

        super().__init__(
            facts_idx,
            rules_heads_idx,
            rules_bodies_idx,
            rule_lens,
            constant_no,
            padding_idx,
            device,
            max_total_groundings=inner_budget,
            **kwargs,
        )

        # Store KGE model ref without nn.Module auto-registration
        self._kge_ref: list = [kge_model]

        # Save inner capacity before clamping
        self._inner_tG: int = self.effective_total_G

        # Output budget -- clamped to inner capacity so topk never overflows
        self._output_tG: int = min(output_budget, self._inner_tG)
        self.effective_total_G = self._output_tG

        # Learned attention over body embeddings
        E = kge_model.atom_embedding_size
        self._attention = GroundingAttention(self.M * E)
        self._atom_emb_size: int = E

    def forward(
        self,
        queries: Tensor,       # [B, 3]
        query_mask: Tensor,    # [B]
    ) -> ForwardResult:
        """Ground rules and select top groundings by neural attention score.

        Args:
            queries:    [B, 3] [pred_idx, subj_idx, obj_idx].
            query_mask: [B] valid queries.

        Returns:
            ForwardResult with selected groundings.
        """
        result = super().forward(queries, query_mask)
        body = result.collected_body    # [B, tG_in, M, 3]
        mask = result.collected_mask    # [B, tG_in]
        ridx = result.collected_ridx   # [B, tG_in]

        B, tG_in, M, _ = body.shape
        E = self._atom_emb_size
        dev = body.device

        # Get num_body_atoms per grounding for body-active masking
        num_body_q = self.num_body_atoms[ridx]  # [B, tG_in]
        atom_idx = torch.arange(M, device=dev).view(1, 1, M)
        body_active = atom_idx < num_body_q.unsqueeze(-1)  # [B, tG_in, M]

        # Neural attention scoring
        subjs_flat = body[..., 1].reshape(-1)
        objs_flat = body[..., 2].reshape(-1)
        kge = self._kge_ref[0]
        emb = kge.embed_atoms(subjs_flat, objs_flat)  # [B*tG_in*M, E]
        emb = emb.view(B, tG_in, M, E)

        # Zero out inactive body atoms
        emb = emb * body_active.unsqueeze(-1).float()

        # Flatten body dim and score
        emb_flat = emb.reshape(B * tG_in, M * E)
        scores = self._attention(emb_flat).view(B, tG_in)

        # Mask invalid groundings with -inf
        scores = scores + (mask.float() - 1.0) * 1e9

        # Top-k selection
        out_tG = self._output_tG
        _, top_idx = scores.topk(out_tG, dim=1, largest=True, sorted=False)

        # Gather selected groundings
        idx_body = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3)
        out_body = body.gather(1, idx_body)
        out_mask = mask.gather(1, top_idx)
        out_ridx = ridx.gather(1, top_idx)
        out_count = out_mask.sum(dim=1)

        return ForwardResult(
            collected_body=out_body,
            collected_mask=out_mask,
            collected_count=out_count,
            collected_ridx=out_ridx,
        )

    def __repr__(self) -> str:
        return (
            f"NeuralGrounder(output_budget={self._output_tG}, "
            f"inner_tG={self._inner_tG}, "
            f"depth={self.depth}, width={self.width})"
        )
