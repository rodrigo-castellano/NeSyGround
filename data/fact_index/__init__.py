"""Fact index package — membership + capability indices, built by ``create``.

    create(facts_idx, type="arg_key", ...)   -> ArgKeyFactIndex      (targeted_lookup)
    create(facts_idx, type="inverted", ...)  -> InvertedFactIndex    (enumerate)
    create(facts_idx, type="block_sparse",.) -> BlockSparseFactIndex (enumerate+exists)

The registry is the single name->class chokepoint; the factory is the only public
constructor (subclass ``__init__`` signatures differ, the factory smooths that).
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from grounder.data.fact_index.arg_key import ArgKeyFactIndex
from grounder.data.fact_index.base import (
    FactIndex, fact_contains, pack_triples_64,
)
from grounder.data.fact_index.block_sparse import BlockSparseFactIndex
from grounder.data.fact_index.inverted import InvertedFactIndex

FactIndexType = Literal["arg_key", "inverted", "block_sparse"]

_REGISTRY = {
    "arg_key": ArgKeyFactIndex,
    "inverted": InvertedFactIndex,
    "block_sparse": BlockSparseFactIndex,
}


def create(
    facts_idx: Tensor,
    *,
    type: FactIndexType = "arg_key",
    constant_no: int,
    predicate_no: int,
    padding_idx: int,
    device: torch.device,
    pack_base: Optional[int] = None,
    max_facts_per_query: int = 64,
    max_memory_mb: int = 256,
    order: Literal["original", "shuffle"] = "original",
    order_seed: int = 42,
) -> FactIndex:
    """Build the fact index named by ``type`` (the only public constructor)."""
    if type not in _REGISTRY:
        raise ValueError(f"Unknown fact index type: {type!r}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[type](
        facts_idx, constant_no=constant_no, predicate_no=predicate_no,
        padding_idx=padding_idx, device=device, pack_base=pack_base,
        max_facts_per_query=max_facts_per_query, max_memory_mb=max_memory_mb,
        order=order, order_seed=order_seed,
    )


__all__ = [
    "create", "FactIndexType",
    "FactIndex", "ArgKeyFactIndex", "InvertedFactIndex", "BlockSparseFactIndex",
    "pack_triples_64", "fact_contains",
]
