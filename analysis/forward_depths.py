"""Forward-chaining proof-depth analysis — the DYNAMIC depth tool.

``run_forward_depths`` computes, per query, the minimum number of
forward-chaining rule-application iterations at which the query atom first
enters the closure (``-1`` if not provable within ``max_depth``). This is the
bottom-up / semi-naive notion of proof depth: a fixpoint closure where each
iteration is one round of rule firing, so a query provable by ``k`` chained
rule applications has depth ``k``.

It is the single shared owner of proof-depth files for the consumers (torch-ns,
DpRL): both build a :class:`grounder.data.KB` and call this. It drives the
public forward engine (:class:`grounder.api.forward.ForwardGrounder`,
``method='spmm'``) via depth truncation — provability is monotone in depth, so
the min depth is the smallest ``d`` whose depth-``d`` closure contains the
query. Cheap (FC is fast); memory grows with the closure, so for very large KBs
build the KB on CPU to reach higher depths (the closure for fb15k237 fits GPU
to depth ~2, CPU to ~3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import Tensor

from grounder.api.forward import ForwardGrounder
from grounder.core import GroundRequest
from grounder.data import KB


@dataclass
class ForwardDepthStats:
    """Aggregate of a :func:`run_forward_depths` pass."""
    total_queries: int
    total_proven: int
    prove_rate: float
    depth_distribution: Dict[int, int] = field(default_factory=dict)  # depth -> count (-1 = unproven)
    max_depth: int = 0
    method: str = "spmm"


@torch.no_grad()
def run_forward_depths(
    kb: KB,
    queries: Tensor,                 # [N, 3] (pred, subj, obj), kb id space, on kb.device_
    max_depth: int = 6,
    *,
    method: str = "spmm",
) -> tuple[Tensor, ForwardDepthStats]:
    """Per-query min forward-chaining proof depth.

    Returns ``(depths, stats)`` where ``depths`` is ``[N]`` long: the smallest
    ``d in 1..max_depth`` whose depth-``d`` closure contains the query, else
    ``-1``. Truncation re-grounds the closure at each depth and stops early once
    every query is covered.
    """
    queries = queries.to(device=kb.device_, dtype=torch.long)
    n = int(queries.shape[0])
    depths = torch.full((n,), -1, dtype=torch.long, device=queries.device)
    fg = ForwardGrounder(kb, method=method, depth=max_depth)
    for d in range(1, max_depth + 1):
        closure = fg.ground(GroundRequest(closure_depth=d))
        newly = closure.contains(queries) & (depths < 0)
        depths[newly] = d
        if int((depths >= 0).sum()) == n:
            break

    proven = int((depths >= 0).sum())
    dist: Dict[int, int] = {}
    for v in depths.tolist():
        dist[v] = dist.get(v, 0) + 1
    stats = ForwardDepthStats(
        total_queries=n, total_proven=proven,
        prove_rate=(proven / n if n else 0.0),
        depth_distribution=dict(sorted(dist.items())),
        max_depth=max_depth, method=method,
    )
    return depths, stats


__all__ = ["run_forward_depths", "ForwardDepthStats"]
