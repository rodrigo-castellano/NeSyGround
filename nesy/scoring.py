"""Unified KGE scoring primitives.

Three core functions that work with any KGE model implementing either:

* **torch-ns interface** (``experiments/model.py``)::

      model.score_atoms(preds, subjs, objs) -> Tensor

* **kge_module interface** (``kge_module/models/architectures.py``)::

      model.score_triples(h, r, t) -> Tensor
      model.score_all_tails_batch(h, r) -> Tensor   # [B, E]
      model.score_all_heads_batch(r, t) -> Tensor   # [B, E]

The primitives auto-detect the available methods and dispatch accordingly.

Also provides precomputed partial scoring:
  precompute_partial_scores() — build [P, E] lookup tables (once at init)
  score_partial_atoms() — pure tensor indexing (CUDA graph safe)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def kge_score_triples(
    model: nn.Module,
    h: Tensor,       # [N] head entity indices
    r: Tensor,       # [N] relation indices
    t: Tensor,       # [N] tail entity indices
) -> Tensor:         # [N] scores (sigmoid-normalized)
    """Score ground triples.

    Uses ``model.score_triples(h, r, t)`` if available,
    otherwise falls back to ``model.score_atoms(preds=r, subjs=h, objs=t)``.

    Returns sigmoid-normalized scores.
    """
    if hasattr(model, 'score_triples'):
        raw = model.score_triples(h, r, t)
    else:
        raw = model.score_atoms(r, h, t)
    return torch.sigmoid(raw)


def kge_score_all_tails(
    model: nn.Module,
    h: Tensor,       # [B] head entity indices
    r: Tensor,       # [B] or scalar — relation indices
) -> Tensor:         # [B, num_entities] scores
    """Score all possible tails for given (h, r) pairs.

    Uses ``model.score_all_tails_batch(h, r)`` if available,
    otherwise expands (h, r) against every entity and calls score_triples/score_atoms.
    """
    if hasattr(model, 'score_all_tails_batch'):
        return torch.sigmoid(model.score_all_tails_batch(h, r))

    # Fallback: expand and score
    B = h.shape[0]
    E: int = model.num_constants
    dev = h.device
    h_exp = h.unsqueeze(1).expand(B, E).reshape(-1)
    r_exp = r.unsqueeze(1).expand(B, E).reshape(-1) if r.dim() > 0 else r.expand(B * E)
    t_all = torch.arange(E, device=dev).unsqueeze(0).expand(B, E).reshape(-1)
    return kge_score_triples(model, h_exp, r_exp, t_all).view(B, E)


def kge_score_all_heads(
    model: nn.Module,
    r: Tensor,       # [B] or scalar — relation indices
    t: Tensor,       # [B] tail entity indices
) -> Tensor:         # [B, num_entities] scores
    """Score all possible heads for given (r, t) pairs.

    Uses ``model.score_all_heads_batch(r, t)`` if available,
    otherwise expands (r, t) against every entity and calls score_triples/score_atoms.
    """
    if hasattr(model, 'score_all_heads_batch'):
        return torch.sigmoid(model.score_all_heads_batch(r, t))

    # Fallback: expand and score
    B = t.shape[0]
    E: int = model.num_constants
    dev = t.device
    t_exp = t.unsqueeze(1).expand(B, E).reshape(-1)
    r_exp = r.unsqueeze(1).expand(B, E).reshape(-1) if r.dim() > 0 else r.expand(B * E)
    h_all = torch.arange(E, device=dev).unsqueeze(0).expand(B, E).reshape(-1)
    return kge_score_triples(model, h_all, r_exp, t_exp).view(B, E)


# ======================================================================
# Precomputed partial scoring
# ======================================================================

@torch.no_grad()
def precompute_partial_scores(
    kge_model: nn.Module,
    pred_remap: Tensor,    # [P_im] → KGE relation id (-1 if unmapped)
    const_remap: Tensor,   # [E_im] → KGE entity id (-1 if unmapped)
    batch_chunk: int = 0,  # 0 = auto based on GPU memory
    entity_chunk: int = 2048,
) -> Tuple[Tensor, Tensor]:
    """Precompute max_tail_score[P, E] and max_head_score[P, E].

    For each (pred, entity) pair:
      max_tail_score[p, h] = max_t sigmoid(score(h, p, t)) — best tail for pred(h, ?)
      max_head_score[p, t] = max_h sigmoid(score(h, p, t)) — best head for pred(?, t)

    Uses kge_score_all_tails / kge_score_all_heads with chunking.
    One call per unique relation. ~1-10s depending on dataset.

    Memory: 2 * P * E * 4 bytes (family=284KB, fb15k237=28MB, yago=37MB).

    Returns:
        max_tail_score: [P_im, E_im] float tensor
        max_head_score: [P_im, E_im] float tensor
    """
    device = const_remap.device
    P_im = pred_remap.shape[0]
    E_im = const_remap.shape[0]

    max_tail_score = torch.zeros(P_im, E_im, dtype=torch.float32, device=device)
    max_head_score = torch.zeros(P_im, E_im, dtype=torch.float32, device=device)

    valid_preds = (pred_remap >= 0).nonzero(as_tuple=True)[0]
    valid_ents = (const_remap >= 0).nonzero(as_tuple=True)[0]
    n_ents = valid_ents.shape[0]

    if n_ents == 0 or valid_preds.shape[0] == 0:
        return max_tail_score, max_head_score

    # Auto batch chunk: target ~2 GB peak
    if batch_chunk <= 0:
        dim = 512
        if hasattr(kge_model, 'ent_re'):
            dim = kge_model.ent_re.weight.shape[1]
        elif hasattr(kge_model, 'embedding_dim'):
            dim = kge_model.embedding_dim
        bytes_per_elem = entity_chunk * dim * 4 * 2
        batch_chunk = max(8, min(512, int(2e9 / bytes_per_elem)))

    kge_ents = const_remap[valid_ents]  # [n_ents] KGE entity ids

    for im_pred in valid_preds:
        kge_rel = pred_remap[im_pred]

        # max_tail_score: for each entity as head, max over all tails
        tail_scores = _partial_score_chunked(
            kge_model, kge_ents, kge_rel, role=0, batch_chunk=batch_chunk)
        max_tail_score[im_pred, valid_ents] = tail_scores

        # max_head_score: for each entity as tail, max over all heads
        head_scores = _partial_score_chunked(
            kge_model, kge_ents, kge_rel, role=1, batch_chunk=batch_chunk)
        max_head_score[im_pred, valid_ents] = head_scores

    return max_tail_score, max_head_score


def _partial_score_chunked(
    kge_model: nn.Module,
    kge_ents: Tensor,    # [E] KGE entity ids
    kge_rel: Tensor,     # scalar KGE relation id
    role: int,           # 0 = entities as heads, 1 = entities as tails
    batch_chunk: int = 64,
) -> Tensor:
    """Compute max-over-completions for each entity, chunked to avoid OOM."""
    E = kge_ents.shape[0]
    device = kge_ents.device
    result = torch.empty(E, dtype=torch.float32, device=device)

    for start in range(0, E, batch_chunk):
        end = min(start + batch_chunk, E)
        chunk = kge_ents[start:end]
        B = chunk.shape[0]
        rel_exp = kge_rel.expand(B)
        if role == 0:
            raw = kge_score_all_tails(kge_model, chunk, rel_exp)  # [B, E_kge]
        else:
            raw = kge_score_all_heads(kge_model, rel_exp, chunk)  # [B, E_kge]
        result[start:end] = raw.max(dim=1).values

    return result


def score_partial_atoms(
    preds: Tensor,              # [N] predicate indices (IM space)
    args1: Tensor,              # [N] first arg indices (IM space)
    args2: Tensor,              # [N] second arg indices (IM space)
    constant_no: int,
    max_tail_score: Tensor,     # [P, E]
    max_head_score: Tensor,     # [P, E]
) -> Tensor:                    # [N] scores
    """Score partial atoms via precomputed lookup. Pure tensor indexing.

    pred(const, ?): score = max_tail_score[pred, const]
    pred(?, const): score = max_head_score[pred, const]
    pred(?, ?): score = 0

    Branchless — no .any()/.item(), no dynamic-shape indexing.
    Safe for CUDA graphs and torch.compile.
    """
    N = preds.shape[0]
    device = preds.device
    scores = torch.zeros(N, dtype=torch.float32, device=device)
    if N == 0:
        return scores

    a1 = args1.long()
    a2 = args2.long()
    p = preds.long()
    C = constant_no
    P, E = max_tail_score.shape

    # Clamp indices to valid range so lookups never OOB.
    # Wrong entries are zeroed out by torch.where below.
    safe_p = p.clamp(0, P - 1)
    safe_a1 = a1.clamp(0, E - 1)
    safe_a2 = a2.clamp(0, E - 1)

    # pred(const, ?): head is ground, tail is variable
    tail_var = (a1 > 0) & (a1 <= C) & (a2 > C)
    tail_scores = max_tail_score[safe_p, safe_a1]           # [N]
    scores = torch.where(tail_var, tail_scores, scores)

    # pred(?, const): head is variable, tail is ground
    head_var = (a1 > C) & (a2 > 0) & (a2 <= C)
    head_scores = max_head_score[safe_p, safe_a2]           # [N]
    scores = torch.where(head_var, head_scores, scores)

    return scores
