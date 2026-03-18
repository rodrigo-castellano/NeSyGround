"""Soft provability scorer — PostResolutionHook.

Known atoms (facts or provable) → score 1.0.
Unknown atoms → sigmoid(KGE score) or learned MLP.
Grounding confidence = product of atom scores.
Selects top-k by confidence.

Two modes: 'kge' (sigmoid) or 'neural' (learned MLP).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data.rule_index import compile_rules
from grounder.filters import check_in_fp_global
from grounder.fc.fc import run_forward_chaining


class ProvabilityMLP(nn.Module):
    """Learned soft provability: Linear → ReLU → Linear → Sigmoid."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class SoftScorer(nn.Module):
    """Soft provability scoring + top-k selection.

    Computes its own provable set at init via forward chaining.

    Args:
        kge_model:       nn.Module with score_atoms() and optionally
                         embed_atoms() + atom_embedding_size.
        mode:            'kge' for sigmoid(KGE), 'neural' for learned MLP.
        output_budget:   number of groundings to keep.
        padding_idx:     padding value.
        fact_index:      FactIndex for fact existence checks.
        rules_heads_idx: [R, 3] for provable set computation.
        rules_bodies_idx: [R, M, 3] for provable set computation.
        rule_lens:       [R] for provable set computation.
        constant_no:     highest constant index.
        predicate_no:    number of predicates.
        num_entities:    number of entities.
        device:          target device.
    """

    def __init__(
        self,
        kge_model: nn.Module,
        mode: str,
        output_budget: int,
        padding_idx: int,
        fact_index,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        constant_no: int,
        predicate_no: int,
        num_entities: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._kge_ref: list = [kge_model]
        self._is_neural = (mode == "neural")
        self._output_tG = output_budget
        self._padding_idx = padding_idx
        self._fact_index = fact_index
        self._E = num_entities

        # Provability MLP (neural mode)
        if self._is_neural and kge_model is not None:
            self._provability_mlp: Optional[ProvabilityMLP] = ProvabilityMLP(
                kge_model.atom_embedding_size)
        else:
            self._provability_mlp = None

        # Compute fp_global set (I_D) via forward chaining
        compiled = compile_rules(
            rules_heads_idx, rules_bodies_idx, rule_lens, constant_no)
        fp_global, n = run_forward_chaining(
            compiled_rules=compiled, facts_idx=facts_idx,
            num_entities=num_entities, num_predicates=predicate_no,
            depth=10, device="cpu")
        self.register_buffer("_fp_global_hashes", fp_global.to(device))
        self._has_fp_global = n > 0

    def apply(
        self,
        body: Tensor,       # [B, tG, M, 3]
        mask: Tensor,       # [B, tG]
        rule_idx: Tensor,   # [B, tG]
    ) -> tuple:
        B, tG_in, M, _ = body.shape
        dev = body.device
        kge = self._kge_ref[0]

        body_active = body[..., 0] != self._padding_idx

        # Fact + provability check
        flat = body.reshape(-1, 3)
        is_fact = self._fact_index.exists(flat).view(B, tG_in, M)
        if self._has_fp_global:
            E = self._E
            h = flat[:, 0] * (E * E) + flat[:, 1] * E + flat[:, 2]
            in_fp_global = check_in_fp_global(
                h, self._fp_global_hashes).view(B, tG_in, M)
        else:
            in_fp_global = torch.zeros_like(is_fact)
        is_known = is_fact | in_fp_global

        # Soft scores for unknown atoms
        if self._is_neural:
            emb = kge.embed_atoms(
                body[..., 1].reshape(-1),
                body[..., 2].reshape(-1))
            soft = self._provability_mlp(emb).view(B, tG_in, M)
        else:
            soft = torch.sigmoid(kge.score_atoms(
                body[..., 0].reshape(-1),
                body[..., 1].reshape(-1),
                body[..., 2].reshape(-1))).view(B, tG_in, M)

        # Per-atom: known=1, unknown=soft, inactive=1
        atom_scores = torch.where(is_known, torch.ones_like(soft), soft)
        atom_scores = torch.where(body_active, atom_scores, torch.ones_like(atom_scores))

        # Grounding confidence = product, masked
        conf = atom_scores.prod(dim=-1) * mask.float()

        from grounder.nesy import _topk_select
        return _topk_select(body, mask, rule_idx, conf, self._output_tG)

    def __repr__(self) -> str:
        mode = "neural" if self._is_neural else "kge"
        return f"SoftScorer(mode={mode}, output_budget={self._output_tG})"
