"""Soft provability grounder -- known atoms=1.0, unknown=sigmoid(KGE) or MLP.

Groundings ranked by product of atom scores (soft conjunction).
Renamed from UGrounder.  Extends ParametrizedBCGrounder by replacing hard
pruning with soft scoring:
- Known atoms (facts or provable) get score 1.0
- Unknown atoms get a soft score via sigmoid(KGE) or learned MLP
- Groundings are ranked by product of atom scores (soft conjunction)
- Top-k groundings are selected by confidence

Two modes:
- 'kge'    : sigmoid(KGE score) for soft provability
- 'neural' : learned ProvabilityMLP for soft provability

Compatible with torch.compile(fullgraph=True, mode='reduce-overhead').
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.grounders.parametrized import ParametrizedBCGrounder
from grounder.forward_chaining import run_forward_chaining
from grounder.compilation import compile_rules, check_in_provable
from grounder.types import ForwardResult


# ======================================================================
# ProvabilityMLP
# ======================================================================


class ProvabilityMLP(nn.Module):
    """Learned soft provability from entity embeddings.

    Architecture: Linear -> ReLU -> Linear -> Sigmoid

    Args:
        input_size: Atom embedding dimensionality (from kge_model).
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        hidden = input_size // 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute soft provability scores.

        Args:
            x: [N, input_size] entity-pair embeddings.

        Returns:
            scores: [N] soft provability in [0, 1].
        """
        return self.net(x).squeeze(-1)


# ======================================================================
# SoftGrounder
# ======================================================================


class SoftGrounder(ParametrizedBCGrounder):
    """Hybrid grounder with soft provability for unproved body atoms.

    Inherits from ParametrizedBCGrounder (with ``prune_incomplete_proofs=False``)
    and replaces hard pruning with soft confidence scoring + top-k selection.

    Runs forward chaining at init to compute the provable set, which is used
    to distinguish known atoms (score 1.0) from unknown atoms (soft score).

    Constructor takes the same raw tensor args as ParametrizedBCGrounder,
    plus ``kge_model``, ``mode``, and ``output_budget``.

    kge_model interface:
        - ``kge_model.score_atoms(preds, subjs, objs) -> Tensor`` scalar scores.
        - ``kge_model.embed_atoms(subjs, objs) -> Tensor`` embeddings [N, E].
        - ``kge_model.atom_embedding_size`` int property.

    Args:
        facts_idx:           [F, 3] fact triples.
        rules_heads_idx:     [R, 3] rule head atoms.
        rules_bodies_idx:    [R, Bmax, 3] rule body atoms (padded).
        rule_lens:           [R] body lengths.
        constant_no:         highest constant index.
        padding_idx:         padding value.
        device:              target device.
        kge_model:           nn.Module with ``score_atoms`` and optionally
                             ``embed_atoms`` + ``atom_embedding_size``.
        mode:                'kge' for sigmoid(KGE), 'neural' for learned MLP.
        output_budget:       output grounding budget (defaults to inner size).
        predicate_no:        total number of predicates (exclusive upper bound).
        num_entities:        total entity count.
        max_total_groundings: base total groundings passed to parent.
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
        kge_model: nn.Module = None,
        mode: str = "kge",
        output_budget: Optional[int] = None,
        predicate_no: int,
        num_entities: int,
        max_total_groundings: int = 64,
        **kwargs,
    ) -> None:
        # Create inner grounder with prune_incomplete_proofs=False
        # (we do soft pruning instead of hard pruning)
        super().__init__(
            facts_idx,
            rules_heads_idx,
            rules_bodies_idx,
            rule_lens,
            constant_no,
            padding_idx,
            device,
            prune_incomplete_proofs=False,
            max_total_groundings=max_total_groundings,
            predicate_no=predicate_no,
            num_entities=num_entities,
            **kwargs,
        )

        self._is_neural: bool = (mode == "neural")

        # Chunk size for score_atoms to avoid OOM on large eval batches.
        # 0 = no chunking (default, safe for torch.compile).
        self._score_chunk_size: int = 0

        # Store kge_model as plain list to avoid nn.Module auto-registration
        self._kge_ref: list = [kge_model]

        # Save the inner effective_total_G before possibly clamping to output_budget
        self._inner_tG: int = self.effective_total_G

        # Output budget (may be smaller than inner grounder's output)
        if output_budget is not None:
            self._output_tG: int = output_budget
        else:
            self._output_tG = self._inner_tG
        self.effective_total_G = self._output_tG

        # For neural mode: create provability MLP
        if self._is_neural and kge_model is not None:
            self._provability_mlp: Optional[ProvabilityMLP] = ProvabilityMLP(
                kge_model.atom_embedding_size
            )
        else:
            self._provability_mlp = None

        # Compile rules from raw tensors for forward chaining
        compiled_rules_fc = compile_rules(
            rules_heads_idx, rules_bodies_idx, rule_lens, constant_no,
        )

        # Compute provable set via forward chaining and store as buffer.
        # Uses a separate name to avoid collision with the parent's
        # provable_hashes buffer (parent has prune=False so its buffer is
        # a dummy zeros(1) tensor).
        provable, n = run_forward_chaining(
            compiled_rules=compiled_rules_fc,
            facts_idx=facts_idx,
            num_entities=num_entities,
            num_predicates=predicate_no,
            depth=10,
            device="cpu",
        )
        self.register_buffer(
            "soft_provable_hashes", provable.to(device)
        )
        self.register_buffer(
            "soft_num_provable",
            torch.tensor(n, dtype=torch.long, device=device),
        )

        # Store metadata for entity-based hashing
        self._num_entities_soft: int = num_entities

        prefix = "uneural_" if self._is_neural else "u_"
        print(
            f"  {prefix}grounder: inner={self._inner_tG} -> "
            f"output={self._output_tG}, provable={n}"
        )

    def _check_soft_provable(
        self,
        preds: Tensor,  # [N]
        subjs: Tensor,  # [N]
        objs: Tensor,   # [N]
    ) -> Tensor:
        """Check if atoms are in precomputed provable set.

        Uses ``check_in_provable`` (binary search on sorted hash tensor).
        Fully compatible with torch.compile(fullgraph=True).

        Args:
            preds: [N] predicate indices.
            subjs: [N] subject indices.
            objs:  [N] object indices.

        Returns:
            provable: [N] boolean tensor.
        """
        E = self._num_entities_soft
        query_hashes = preds * (E * E) + subjs * E + objs
        return check_in_provable(query_hashes, self.soft_provable_hashes)

    def forward(
        self,
        queries: Tensor,       # [B, 3]
        query_mask: Tensor,    # [B]
    ) -> ForwardResult:
        """Ground rules with soft provability scoring and top-k selection.

        Steps:
            1. Get inner result (no hard pruning).
            2. Check facts and provability for all body atoms.
            3. Compute soft scores for unknown atoms (KGE sigmoid or MLP).
            4. Per-atom: known=1.0, unknown=soft, inactive=1.0.
            5. Grounding confidence = product of atom scores.
            6. topk by confidence -> ForwardResult.

        Args:
            queries:    [B, 3] - [pred_idx, subj_idx, obj_idx].
            query_mask: [B] - valid queries.

        Returns:
            ForwardResult with soft-scored groundings.
        """
        # 1. Run parent forward (no hard pruning)
        result = super().forward(queries, query_mask)
        body = result.collected_body    # [B, tG_in, M, 3]
        mask = result.collected_mask    # [B, tG_in]
        ridx = result.collected_ridx   # [B, tG_in]

        B, tG_in, M, _ = body.shape
        dev = body.device

        # 2. Check facts and provability for all body atoms
        preds_flat = body[..., 0].reshape(-1)
        subjs_flat = body[..., 1].reshape(-1)
        objs_flat = body[..., 2].reshape(-1)

        # Fact existence check via the fact index
        atoms_flat = torch.stack([preds_flat, subjs_flat, objs_flat], dim=-1)
        is_fact = self.fact_index.exists(atoms_flat)
        is_fact = is_fact.view(B, tG_in, M)

        is_provable = self._check_soft_provable(preds_flat, subjs_flat, objs_flat)
        is_provable = is_provable.view(B, tG_in, M)

        is_known = is_fact | is_provable

        # Body-active mask: which atom positions are actually used per rule
        num_body_q = self.num_body_atoms[ridx]  # [B, tG_in]
        atom_idx = torch.arange(M, device=dev).view(1, 1, M)
        body_active = atom_idx < num_body_q.unsqueeze(-1)  # [B, tG_in, M]

        # 3. Soft scores for unknown atoms
        if self._is_neural:
            kge = self._kge_ref[0]
            emb = kge.embed_atoms(subjs_flat, objs_flat)  # [N, emb_size]
            soft_scores = self._provability_mlp(emb).view(B, tG_in, M)
        else:
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
                raw_scores = torch.cat(chunks, dim=0)
            else:
                raw_scores = kge.score_atoms(
                    preds_flat, subjs_flat, objs_flat
                )
            soft_scores = torch.sigmoid(raw_scores).view(B, tG_in, M)

        # 4. Per-atom score: known=1.0, unknown=soft_score, inactive=1.0
        atom_scores = torch.where(
            is_known, torch.ones_like(soft_scores), soft_scores
        )
        atom_scores = torch.where(
            body_active, atom_scores, torch.ones_like(atom_scores)
        )

        # 5. Grounding confidence = product of atom scores (soft conjunction)
        grounding_conf = atom_scores.prod(dim=-1)  # [B, tG_in]
        grounding_conf = grounding_conf * mask.float()  # zero out invalid

        # 6. Top-k selection by confidence
        out_tG = self._output_tG

        if out_tG >= tG_in:
            # No selection needed -- pad if necessary
            pad = out_tG - tG_in
            if pad > 0:
                out_body = torch.nn.functional.pad(body, (0, 0, 0, 0, 0, pad))
                out_mask = torch.nn.functional.pad(mask, (0, pad))
                out_ridx = torch.nn.functional.pad(ridx, (0, pad))
            else:
                out_body = body
                out_mask = mask
                out_ridx = ridx
            out_count = out_mask.sum(dim=1)
            return ForwardResult(
                collected_body=out_body,
                collected_mask=out_mask,
                collected_count=out_count,
                collected_ridx=out_ridx,
            )

        _, top_idx = grounding_conf.topk(
            out_tG, dim=1, largest=True, sorted=False
        )

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
        mode = "neural" if self._is_neural else "kge"
        return (
            f"SoftGrounder(mode={mode}, "
            f"output_tG={self._output_tG}, "
            f"inner_tG={self._inner_tG}, "
            f"depth={self.depth}, width={self.width}, "
            f"provable={self.soft_num_provable})"
        )
