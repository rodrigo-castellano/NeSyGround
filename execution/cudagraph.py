"""CUDA-graph bookkeeping — the single clone-to-detach seam + step marking.

``detach_from_pool`` is the ONLY place outputs are deep-cloned out of the
CUDA-graph private pool (so a later replay can't overwrite a still-referenced
tensor). It is a structural deep-clone over an arbitrary pytree (tensors inside
tuples / NamedTuples / dataclasses / lists / dicts), so it covers both the
step-output tuple and the nested ``BackwardResult`` tree.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Tuple, TypeVar

import torch
from torch import Tensor

T = TypeVar("T")


def mark_step_begin() -> None:
    """Tell CUDA-graph trees a new iteration starts (safe to reuse static-address
    output buffers). Skipped automatically inside compiled tracing by the caller."""
    torch.compiler.cudagraph_mark_step_begin()


def detach_from_pool(obj: T) -> T:
    """Structural deep-clone: clone every tensor, rebuild containers. THE ONLY
    clone-to-detach-from-CUDA-graph-pool seam."""
    if torch.is_tensor(obj):
        return obj.clone()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return type(obj)(**{f.name: detach_from_pool(getattr(obj, f.name))
                            for f in dataclasses.fields(obj)})
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):     # NamedTuple
        return type(obj)(*(detach_from_pool(x) for x in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(detach_from_pool(x) for x in obj)
    if isinstance(obj, dict):
        return {k: detach_from_pool(v) for k, v in obj.items()}
    return obj


def pad_chunk(queries: Tensor, mask: Tensor, to: int) -> Tuple[Tensor, Tensor]:
    """Pad a short final chunk up to ``to`` rows (padded rows have mask=False) so a
    captured CUDA graph always sees the same static batch shape."""
    n = queries.shape[0]
    if n == to:
        return queries, mask
    q = queries.new_zeros((to,) + tuple(queries.shape[1:]))
    m = torch.zeros(to, dtype=torch.bool, device=mask.device)
    q[:n] = queries
    m[:n] = mask
    return q, m


__all__ = ["mark_step_begin", "detach_from_pool", "pad_chunk"]
