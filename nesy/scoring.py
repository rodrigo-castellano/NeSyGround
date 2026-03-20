"""Shared grounder adapter around the torch-kge-kernels scoring kernels."""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from kge_kernels.partial import precompute_partial_scores as _precompute_partial_scores
from kge_kernels.partial import score_partial_atoms
from kge_kernels.scoring import score as _score
from kge_kernels.types import KGEBackend


def _unwrap_model(model: nn.Module) -> nn.Module:
    actual = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(actual, "_orig_mod"):
        actual = actual._orig_mod
    return actual


def _num_entities(model: nn.Module) -> int:
    actual = _unwrap_model(model)
    if hasattr(actual, "num_constants"):
        return int(actual.num_constants)
    if hasattr(actual, "num_entities"):
        return int(actual.num_entities)
    raise AttributeError("Model adapter requires num_constants or num_entities for fallback batched scoring")


def _score_triples_sigmoid(model: nn.Module, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    actual = _unwrap_model(model)
    if hasattr(actual, "score_triples"):
        raw = actual.score_triples(h, r, t)
    elif hasattr(actual, "score_atoms"):
        raw = actual.score_atoms(r, h, t)
    else:
        raise AttributeError("Model adapter requires score_triples(...) or score_atoms(preds, subjs, objs)")
    return torch.sigmoid(raw)


def _score_all_tails_sigmoid(model: nn.Module, h: Tensor, r: Tensor) -> Tensor:
    actual = _unwrap_model(model)
    if hasattr(actual, "score_all_tails_batch"):
        return torch.sigmoid(actual.score_all_tails_batch(h, r))

    batch_size = h.shape[0]
    num_entities = _num_entities(actual)
    device = h.device
    all_tails = torch.arange(num_entities, device=device).unsqueeze(0).expand(batch_size, -1)
    h_exp = h.unsqueeze(1).expand_as(all_tails).reshape(-1)
    if r.dim() == 0:
        r_exp = r.expand(batch_size * num_entities)
    else:
        r_exp = r.unsqueeze(1).expand_as(all_tails).reshape(-1)
    t_exp = all_tails.reshape(-1)
    return _score_triples_sigmoid(actual, h_exp, r_exp, t_exp).view(batch_size, num_entities)


def _score_all_heads_sigmoid(model: nn.Module, r: Tensor, t: Tensor) -> Tensor:
    actual = _unwrap_model(model)
    if hasattr(actual, "score_all_heads_batch"):
        return torch.sigmoid(actual.score_all_heads_batch(r, t))

    batch_size = t.shape[0]
    num_entities = _num_entities(actual)
    device = t.device
    all_heads = torch.arange(num_entities, device=device).unsqueeze(0).expand(batch_size, -1)
    h_exp = all_heads.reshape(-1)
    if r.dim() == 0:
        r_exp = r.expand(batch_size * num_entities)
    else:
        r_exp = r.unsqueeze(1).expand_as(all_heads).reshape(-1)
    t_exp = t.unsqueeze(1).expand_as(all_heads).reshape(-1)
    return _score_triples_sigmoid(actual, h_exp, r_exp, t_exp).view(batch_size, num_entities)


def _build_backend(model: nn.Module) -> KGEBackend:
    return KGEBackend(
        score_triples=lambda h, r, t: _score_triples_sigmoid(model, h, r, t),
        score_all_tails=lambda h, r: _score_all_tails_sigmoid(model, h, r),
        score_all_heads=lambda r, t: _score_all_heads_sigmoid(model, r, t),
    )


def _stack_rht(r: Tensor, h: Tensor, t: Tensor) -> Tensor:
    if r.dim() == 0:
        r = r.expand(h.shape[0])
    return torch.stack([r, h, t], dim=1)


def _apply_masks(
    scores: Tensor,
    idx1: Tensor,
    idx2: Tensor,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]],
    true_entities: Optional[Tensor],
    domain: Optional[Set[int]],
) -> None:
    batch_size, num_entities = scores.shape
    device = scores.device

    if domain is not None:
        domain_mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
        domain_mask[torch.tensor(sorted(domain), dtype=torch.long, device=device)] = True
        scores[:, ~domain_mask] = float("-inf")

    if filter_map is not None and true_entities is not None:
        idx1_list = idx1.tolist() if idx1.dim() > 0 else [int(idx1.item())] * batch_size
        idx2_list = idx2.tolist() if idx2.dim() > 0 else [int(idx2.item())] * batch_size
        for row in range(batch_size):
            key = (int(idx1_list[row]), int(idx2_list[row]))
            known = filter_map.get(key)
            if known:
                true_ent = int(true_entities[row].item())
                filtered = [ent for ent in known if ent != true_ent]
                if filtered:
                    scores[row, torch.tensor(filtered, dtype=torch.long, device=device)] = float("-inf")


def kge_score_triples(model: nn.Module, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    triples = _stack_rht(r, h, t)
    return _score(_build_backend(model), triples, mode="triples").scores


def kge_score_all_tails(
    model: nn.Module,
    h: Tensor,
    r: Tensor,
    *,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]] = None,
    true_tails: Optional[Tensor] = None,
    domain: Optional[Set[int]] = None,
) -> Tensor:
    anchor_t = true_tails if true_tails is not None else torch.zeros_like(h)
    triples = _stack_rht(r, h, anchor_t)
    scores = _score(_build_backend(model), triples, mode="tail").scores
    _apply_masks(scores, h, r, filter_map, true_tails, domain)
    return scores


def kge_score_all_heads(
    model: nn.Module,
    r: Tensor,
    t: Tensor,
    *,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]] = None,
    true_heads: Optional[Tensor] = None,
    domain: Optional[Set[int]] = None,
) -> Tensor:
    anchor_h = true_heads if true_heads is not None else torch.zeros_like(t)
    triples = _stack_rht(r, anchor_h, t)
    scores = _score(_build_backend(model), triples, mode="head").scores
    _apply_masks(scores, r, t, filter_map, true_heads, domain)
    return scores


def kge_score_k_tails(
    model: nn.Module,
    h: Tensor,
    r: Tensor,
    t: Tensor,
    sampler: object,
    num_corruptions: int,
) -> Tuple[Tensor, Tensor]:
    triples = _stack_rht(r, h, t)
    out = _score(_build_backend(model), triples, mode="tail", num_corruptions=num_corruptions, sampler=sampler)
    assert out.valid_mask is not None
    return out.scores, out.valid_mask


def kge_score_k_heads(
    model: nn.Module,
    h: Tensor,
    r: Tensor,
    t: Tensor,
    sampler: object,
    num_corruptions: int,
) -> Tuple[Tensor, Tensor]:
    triples = _stack_rht(r, h, t)
    out = _score(_build_backend(model), triples, mode="head", num_corruptions=num_corruptions, sampler=sampler)
    assert out.valid_mask is not None
    return out.scores, out.valid_mask


def precompute_partial_scores(
    kge_model: nn.Module,
    pred_remap: Tensor,
    const_remap: Tensor,
    batch_chunk: int = 64,
    entity_chunk: int = 2048,
) -> Tuple[Tensor, Tensor]:
    del entity_chunk
    return _precompute_partial_scores(_build_backend(kge_model), pred_remap, const_remap, batch_chunk=batch_chunk)


__all__ = [
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_k_heads",
    "kge_score_k_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "score_partial_atoms",
]
