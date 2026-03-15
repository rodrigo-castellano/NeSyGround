"""Unified KGE scoring primitives.

Three functions that work with any KGE model implementing either:

* **torch-ns interface** (``experiments/model.py``)::

      model.score_atoms(preds, subjs, objs) -> Tensor

* **kge_module interface** (``kge_module/models/architectures.py``)::

      model.score_triples(h, r, t) -> Tensor
      model.score_all_tails_batch(h, r) -> Tensor   # [B, E]
      model.score_all_heads_batch(r, t) -> Tensor   # [B, E]

The primitives auto-detect the available methods and dispatch accordingly.
"""

from __future__ import annotations

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
