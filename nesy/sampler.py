"""Uniform random sampling grounder -- wraps ParametrizedBCGrounder.

Oversamples candidates (4x budget), then selects a random subset via topk
on randomized scores. During eval: deterministic (valid-first by mask).
Compatible with torch.compile(fullgraph=True, mode='reduce-overhead').
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.grounders.parametrized import ParametrizedBCGrounder
from grounder.types import ForwardResult


class SamplerGrounder(ParametrizedBCGrounder):
    """Random sampling wrapper: 4x oversample, then random top-k (train) or valid-first (eval).

    Inherits from ParametrizedBCGrounder with a 4x-inflated inner budget.
    The ``forward`` method calls the parent to collect oversampled candidates,
    then selects a random (train) or deterministic (eval) subset.

    Constructor takes the same raw tensor args as ParametrizedBCGrounder,
    plus ``max_sample`` controlling the output budget.

    Args:
        facts_idx:           [F, 3] fact triples (pred, arg0, arg1).
        rules_heads_idx:     [R, 3] rule head atoms.
        rules_bodies_idx:    [R, Bmax, 3] rule body atoms (padded).
        rule_lens:           [R] number of body atoms per rule.
        constant_no:         highest constant index.
        padding_idx:         padding value.
        device:              target device.
        max_sample:          output budget (effective_total_G for this grounder).
        depth:               backward chaining depth.
        width:               max unproven body atoms per grounding.
        max_groundings_per_query: G budget per rule per query.
        max_total_groundings:  ignored (overridden by max_sample * 4 internally).
        prune_incomplete_proofs: whether to prune groundings with unprovable atoms.
        fc_method:           forward-chaining method for provable set.
        fc_depth:            forward-chaining depth.
        predicate_no:        total number of predicates (exclusive).
        num_entities:        total number of entities.
        max_facts_per_query: K_f for inverted/block_sparse index.
        fact_index_type:     'block_sparse' (default) or 'inverted'.
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
        depth: int = 2,
        width: Optional[int] = 1,
        max_groundings_per_query: int = 32,
        max_total_groundings: int = 64,
        max_sample: int = 64,
        prune_incomplete_proofs: bool = True,
        fc_method: str = "join",
        fc_depth: int = 10,
        predicate_no: Optional[int] = None,
        num_entities: Optional[int] = None,
        max_facts_per_query: int = 64,
        fact_index_type: str = "block_sparse",
        **kwargs,
    ) -> None:
        # Inner grounder gets 4x the output budget
        inner_total = max_sample * 4

        super().__init__(
            facts_idx,
            rules_heads_idx,
            rules_bodies_idx,
            rule_lens,
            constant_no,
            padding_idx,
            device,
            depth=depth,
            width=width,
            max_groundings_per_query=max_groundings_per_query,
            max_total_groundings=inner_total,
            prune_incomplete_proofs=prune_incomplete_proofs,
            fc_method=fc_method,
            fc_depth=fc_depth,
            predicate_no=predicate_no,
            num_entities=num_entities,
            max_facts_per_query=max_facts_per_query,
            fact_index_type=fact_index_type,
            **kwargs,
        )

        # Save the inner grounder's effective_total_G before clamping
        self._inner_tG: int = self.effective_total_G
        # Clamp output budget to inner capacity (inner may be smaller when
        # R_eff * max_groundings_per_query < max_sample)
        self._output_tG: int = min(max_sample, self._inner_tG)
        # Override effective_total_G to the output budget
        self.effective_total_G = self._output_tG

    @torch.no_grad()
    def forward(
        self,
        queries: Tensor,       # [B, 3]
        query_mask: Tensor,    # [B]
    ) -> ForwardResult:
        """Ground rules then sample a random subset of valid groundings.

        Args:
            queries:    [B, 3] - [pred_idx, subj_idx, obj_idx].
            query_mask: [B] - valid queries.

        Returns:
            ForwardResult with sampled groundings.
        """
        # 1. Get oversampled candidates from parent
        result = super().forward(queries, query_mask)
        body = result.collected_body    # [B, inner_tG, M, 3]
        mask = result.collected_mask    # [B, inner_tG]
        ridx = result.collected_ridx   # [B, inner_tG]

        B, tG_in, M, _ = body.shape
        tG = self._output_tG
        dev = body.device

        # 2. Compute selection scores -- branch on training mode
        #    (safe for torch.compile: train/eval use separate CUDA graphs)
        if self.training:
            # Random scores: valid groundings get rand in (0,1], invalid get 0
            scores = torch.rand(B, tG_in, device=dev) * mask.float()
        else:
            # Deterministic: valid first (score=1), invalid=0
            scores = mask.float()

        # 3. Select top-k groundings by score (tG <= inner_tG by construction)
        _, top_idx = scores.topk(tG, dim=1, largest=True, sorted=False)

        # 4. Gather body atoms, mask, and rule indices at selected positions
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
            f"SamplerGrounder(max_sample={self._output_tG}, "
            f"inner_tG={self._inner_tG}, "
            f"depth={self.depth}, width={self.width})"
        )
