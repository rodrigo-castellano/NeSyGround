"""KGE-based hooks for resolution filtering and grounding scoring.

KGEScorer
    GroundingHook — min-conjunction of body atom KGE scores + top-k.

KGEFactFilter
    ResolutionFactHook — scores matched fact triples with KGE, keeps top-k.

KGERuleFilter
    ResolutionRuleHook — scores first body atom of rule children, keeps top-k.

KGEStepFilter
    StepHook — scores collected groundings between BFS depths, keeps top-k.
    Supports ground + partial scoring via precomputed [P, E] tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, Tuple

from grounder.nesy.scoring import kge_score_triples, score_partial_atoms


class KGEScorer(nn.Module):
    """Score groundings by KGE min-conjunction, select top-k.

    GroundingHook: applied after grounding in BCGrounder.forward().

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

        # KGE atom scores via unified scoring primitive
        atom_scores = kge_score_triples(
            kge,
            body[..., 1].reshape(-1),  # h (subj)
            body[..., 0].reshape(-1),  # r (pred)
            body[..., 2].reshape(-1),  # t (obj)
        ).view(B, tG_in, M)

        # Mask inactive -> large value so min ignores them
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


class KGEFactFilter(nn.Module):
    """Score matched fact triples with KGE, keep top-k.

    ResolutionFactHook: applied inside resolve_sld/rtf after mgu_resolve_facts.

    Re-looks up fact candidates via fact_index to obtain ground triples,
    scores them with kge_score_triples, and zeros out low-scoring entries
    in fact_success.

    Args:
        kge_model:    nn.Module with score_atoms().
        fact_index:   ArgKeyFactIndex (or compatible) with targeted_lookup().
        facts_idx:    [F, 3] tensor of all facts.
        top_k:        max fact candidates to keep per (batch, state).
        padding_idx:  padding value.
    """

    def __init__(
        self,
        kge_model: nn.Module,
        fact_index,
        facts_idx: Tensor,
        top_k: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        self._kge_ref: list = [kge_model]
        self._fact_index = fact_index
        self.register_buffer("_facts_idx", facts_idx)
        self._top_k = top_k
        self._padding_idx = padding_idx

    def filter_facts(
        self,
        fact_goals: Tensor,      # [B, S, K_f, G, 3]
        fact_success: Tensor,    # [B, S, K_f]
        queries: Tensor,         # [B, S, 3]
    ) -> Tensor:
        B, S, K_f = fact_success.shape
        if K_f == 0 or self._top_k >= K_f:
            return fact_success

        dev = fact_success.device
        kge = self._kge_ref[0]
        N = B * S

        # Re-lookup matched fact triples
        flat_q = queries.reshape(N, 3)
        fact_item_idx, _ = self._fact_index.targeted_lookup(flat_q, K_f)
        F = self._facts_idx.shape[0]
        safe_idx = fact_item_idx.clamp(0, max(F - 1, 0))
        fact_triples = self._facts_idx[safe_idx.view(-1)].view(N, K_f, 3)

        # Score: triples are [pred, subj, obj]
        scores = kge_score_triples(
            kge,
            fact_triples[..., 1].reshape(-1),  # h (subj)
            fact_triples[..., 0].reshape(-1),  # r (pred)
            fact_triples[..., 2].reshape(-1),  # t (obj)
        ).view(B, S, K_f)

        # Mask invalid candidates to -inf
        scores = torch.where(
            fact_success, scores,
            torch.tensor(-1e9, device=dev, dtype=scores.dtype))

        # Top-k selection: zero out entries below top-k
        _, top_idx = scores.view(N, K_f).topk(
            min(self._top_k, K_f), dim=1, largest=True, sorted=False)
        keep = torch.zeros(N, K_f, dtype=torch.bool, device=dev)
        keep.scatter_(1, top_idx, True)
        return fact_success & keep.view(B, S, K_f)

    def __repr__(self) -> str:
        return f"KGEFactFilter(top_k={self._top_k})"


class KGERuleFilter(nn.Module):
    """Score rule children's first body atom with KGE, keep top-k.

    ResolutionRuleHook: applied inside resolve_sld/rtf after mgu_resolve_rules.

    For each rule child, scores the first body atom (rule_goals[:,:,:,0,:]).
    Only ground atoms (all args <= constant_no) are scored; non-ground atoms
    get a neutral score of 0.

    Args:
        kge_model:    nn.Module with score_atoms().
        top_k:        max rule candidates to keep per (batch, state).
        constant_no:  highest constant index (for ground detection).
        padding_idx:  padding value.
    """

    def __init__(
        self,
        kge_model: nn.Module,
        top_k: int,
        constant_no: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        self._kge_ref: list = [kge_model]
        self._top_k = top_k
        self._constant_no = constant_no
        self._padding_idx = padding_idx

    def filter_rules(
        self,
        rule_goals: Tensor,      # [B, S, K_r, G, 3]
        rule_success: Tensor,    # [B, S, K_r]
        queries: Tensor,         # [B, S, 3]
    ) -> Tensor:
        B, S, K_r = rule_success.shape
        if K_r == 0 or self._top_k >= K_r:
            return rule_success

        dev = rule_success.device
        kge = self._kge_ref[0]
        c_no = self._constant_no

        # First body atom of each rule child
        first_atoms = rule_goals[:, :, :, 0, :]  # [B, S, K_r, 3]
        p = first_atoms[..., 0]   # pred
        a1 = first_atoms[..., 1]  # subj
        a2 = first_atoms[..., 2]  # obj

        # Ground detection: both args are constants and pred is not padding
        is_ground = (a1 <= c_no) & (a2 <= c_no) & (p != self._padding_idx)

        # Safe indices for embedding lookup (clamp variables to 0)
        safe_p = torch.where(is_ground, p, torch.zeros_like(p))
        safe_a1 = torch.where(is_ground, a1, torch.zeros_like(a1))
        safe_a2 = torch.where(is_ground, a2, torch.zeros_like(a2))

        scores = kge_score_triples(
            kge,
            safe_a1.reshape(-1),  # h
            safe_p.reshape(-1),   # r
            safe_a2.reshape(-1),  # t
        ).view(B, S, K_r)

        # Non-ground -> 0 (neutral), invalid -> -inf
        scores = torch.where(is_ground, scores, torch.zeros_like(scores))
        scores = torch.where(
            rule_success, scores,
            torch.tensor(-1e9, device=dev, dtype=scores.dtype))

        # Top-k selection
        N = B * S
        _, top_idx = scores.view(N, K_r).topk(
            min(self._top_k, K_r), dim=1, largest=True, sorted=False)
        keep = torch.zeros(N, K_r, dtype=torch.bool, device=dev)
        keep.scatter_(1, top_idx, True)
        return rule_success & keep.view(B, S, K_r)

    def __repr__(self) -> str:
        return (f"KGERuleFilter(top_k={self._top_k}, "
                f"constant_no={self._constant_no})")


class KGEStepFilter(nn.Module):
    """StepHook: score collected groundings between BFS depths, keep top-k.

    Scores the first body atom of each collected grounding:
    - Ground atoms: scored via kge_score_triples.
    - Partial atoms: scored via precomputed max_tail_score / max_head_score.
    - Fully unbound atoms (both args variable): score 0 (kept unconditionally
      unless displaced by higher-scoring entries).

    Pure tensor ops — CUDA graph compatible when shapes are static.

    Scoring modes:
        ground_only:  score ground atoms; non-ground atoms unconditionally kept
        partial_only: score partial atoms; ground atoms unconditionally kept
        both:         score ground + partial; only fully-unbound kept unconditionally

    Args:
        kge_model:       nn.Module with score_triples() or score_atoms().
        top_k:           max groundings to keep per query.
        constant_no:     highest constant index (for ground detection).
        padding_idx:     padding value.
        max_tail_score:  [P, E] precomputed (from precompute_partial_scores).
        max_head_score:  [P, E] precomputed (from precompute_partial_scores).
        scoring_mode:    'ground_only' | 'partial_only' | 'both'.
    """

    def __init__(
        self,
        kge_model: nn.Module,
        top_k: int,
        constant_no: int,
        padding_idx: int,
        max_tail_score: Optional[Tensor] = None,
        max_head_score: Optional[Tensor] = None,
        scoring_mode: str = "ground_only",
    ) -> None:
        super().__init__()
        self._kge_ref: list = [kge_model]
        self._top_k = top_k
        self._constant_no = constant_no
        self._padding_idx = padding_idx
        self._scoring_mode = scoring_mode
        # Register as buffers so .to(device) moves them
        if max_tail_score is not None:
            self.register_buffer("_max_tail_score", max_tail_score)
        else:
            self._max_tail_score = None
        if max_head_score is not None:
            self.register_buffer("_max_head_score", max_head_score)
        else:
            self._max_head_score = None

    def on_step(
        self,
        body: Tensor,        # [B, tG, M, 3]
        mask: Tensor,        # [B, tG]
        rule_idx: Tensor,    # [B, tG]
        d: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Score first atom of each collected grounding, keep top-k per query.

        Branchless: no .item()/.any() calls, no dynamic-shape indexing.
        All ops run unconditionally on the full [B, tG] tensors using
        torch.where for conditional scoring. Safe to call from a Python
        loop between compiled steps (no GPU→CPU sync).
        """
        B, tG, M, _ = body.shape
        dev = body.device
        kge = self._kge_ref[0]
        c_no = self._constant_no
        pad = self._padding_idx
        mode = self._scoring_mode
        top_k = self._top_k

        # First body atom of each grounding — [B, tG, 3]
        first = body[:, :, 0, :]
        p = first[..., 0]                  # [B, tG]
        a1 = first[..., 1]
        a2 = first[..., 2]

        # Classify atoms (all branchless tensor ops)
        is_ground = mask & (a1 <= c_no) & (a2 <= c_no) & (p != pad)
        is_partial = mask & ~is_ground & (p != pad) & (
            ((a1 > 0) & (a1 <= c_no) & (a2 > c_no)) |
            ((a1 > c_no) & (a2 > 0) & (a2 <= c_no)))

        # Start with -inf for invalid, 0 for valid-but-unscored
        scores = torch.where(mask, torch.zeros(1, device=dev),
                             torch.full((1,), -1e9, device=dev))  # [B, tG]
        scored = torch.zeros(B, tG, dtype=torch.bool, device=dev)

        # --- Score ground atoms (unconditional — safe_* clamp handles empty) ---
        if mode in ("ground_only", "both"):
            safe_p = torch.where(is_ground, p, torch.zeros_like(p))
            safe_a1 = torch.where(is_ground, a1, torch.zeros_like(a1))
            safe_a2 = torch.where(is_ground, a2, torch.zeros_like(a2))
            g_scores = kge_score_triples(
                kge,
                safe_a1.reshape(-1),
                safe_p.reshape(-1),
                safe_a2.reshape(-1),
            ).view(B, tG)
            scores = torch.where(is_ground, g_scores, scores)
            scored = scored | is_ground

        # --- Score partial atoms (unconditional — score_partial_atoms handles
        #     empty via torch.where internally, returns 0 for non-partial) ---
        if mode in ("partial_only", "both"):
            if self._max_tail_score is not None and self._max_head_score is not None:
                p_scores = score_partial_atoms(
                    p.reshape(-1), a1.reshape(-1), a2.reshape(-1),
                    c_no, self._max_tail_score, self._max_head_score,
                ).view(B, tG)
                scores = torch.where(is_partial, p_scores, scores)
                scored = scored | (is_partial & (p_scores > 0.0))

        # Unconditional entries: always kept regardless of score
        if mode == "ground_only":
            unconditional = mask & ~is_ground
        elif mode == "partial_only":
            unconditional = mask & ~is_partial
        else:  # both
            unconditional = mask & ~is_ground & ~is_partial

        # --- Branchless top-k over ALL rows ---
        # Set unscored entries to -inf so topk ignores them
        topk_scores = torch.where(scored, scores,
                                  torch.full((1,), -float('inf'), device=dev))
        k = min(top_k, tG)
        _, topk_idx = topk_scores.topk(k, dim=1)           # [B, k]

        # Build keep mask: top-k scored + all unconditional
        keep = torch.zeros(B, tG, dtype=torch.bool, device=dev)
        keep.scatter_(1, topk_idx, True)
        keep = keep | unconditional

        new_mask = mask & keep
        return body, new_mask, rule_idx

    def __repr__(self) -> str:
        return (f"KGEStepFilter(top_k={self._top_k}, "
                f"scoring_mode='{self._scoring_mode}')")
