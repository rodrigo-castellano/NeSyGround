"""BFS proof-depth analysis over a ``BackwardGrounder``.

``run_bfs`` computes, per query, the *minimum* proof depth (smallest d at which
the query first gets a grounding, ``-1`` if never proved within ``grounder.depth``)
plus per-query ``BFSResult`` records and an aggregate ``BatchBFSStats``.

Ported from the OLD ``analysis/generate_depths_static.py``. The old code drove the
grounder's internal step loop (``init_states``/``step``) and inspected
``collected_mask`` between depths — that loop is now private to ``backward.loop``.
Against the redesign's public API the equivalent (and byte-equivalent for the
no-hook case) is *depth truncation*: provability is monotonic in depth, so the min
proof depth is the smallest d for which a depth-d grounder proves the query.

The old between-depth ``step_hook`` (KGEStepFilter) has no redesign analog — KGE
filtering is now a ctor ``fact_hook``/``rule_hook`` applied at resolution time, not
a step filter applied between depths. ``run_bfs`` raises if a ``step_hook`` is set
rather than silently returning unfiltered numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from grounder.grounder.backward import BackwardGrounder


@dataclass
class BFSResult:
    """Result of BFS for a single query."""
    depth: int              # min proof depth, -1 if not proved
    predicate_idx: int      # query predicate index (for per-pred stats)
    branching_per_depth: Dict[int, List[int]] = field(default_factory=dict)
    frontier_sizes: List[int] = field(default_factory=list)


@dataclass
class BatchBFSStats:
    """Aggregate stats from batched BFS (most fields empty for static BFS)."""
    branching_per_depth: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    post_filter_branching: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    ground_children_per_depth: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    nonground_children_per_depth: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    frontier_per_depth: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    unique_states_per_depth: Dict[int, int] = field(default_factory=dict)
    total_states_per_depth: Dict[int, int] = field(default_factory=dict)
    avg_max_frontier: float = 0.0


def _truncate(grounder: BackwardGrounder, depth: int) -> BackwardGrounder:
    """Rebuild a plain BackwardGrounder over the same KB with ``depth`` and
    evidence on (needed for per-query provability). ``_ctor_kwargs`` is captured
    by the base ctor for every grounder, so fact/rule hooks + standardization are
    preserved; only Python-level subclass method overrides are dropped (no
    run_bfs caller relies on those — RLGrounder is depth-1 RL-only)."""
    kwargs = dict(getattr(grounder, "_ctor_kwargs", {}))
    kwargs.update(depth=depth, collect_evidence=True)
    return BackwardGrounder(grounder.kb, **kwargs)


@torch.no_grad()
def run_bfs(
    grounder: BackwardGrounder,
    queries_idx: Tensor,                         # [N, 3]
    batch_size: int = 256,
    excluded_queries: Optional[Tensor] = None,   # [N, 1, 3] or None
) -> Tuple[Tensor, List[BFSResult], BatchBFSStats]:
    """Run BFS over ``queries_idx``, returning per-query minimum proof depth.

    Args:
        grounder: a BackwardGrounder configured with the desired max depth.
        queries_idx: [N, 3] query tensor on the grounder's device.
        batch_size: queries per batch (B dimension).
        excluded_queries: optional [N, 1, 3] excluded queries (cycle prevention).

    Returns:
        depths: [N] long tensor of proof depths (-1 if not proved).
        results: per-query BFSResult.
        stats: BatchBFSStats (aggregate; per-depth fields empty for static BFS).
    """
    if getattr(grounder, "step_hook", None) is not None:
        raise NotImplementedError(
            "run_bfs no longer supports a between-depth step_hook: the redesign "
            "grounder applies KGE filtering as a ctor fact_hook/rule_hook at "
            "resolution time, not as a StepHook between depths. Rebuild the "
            "grounder with the hook wired in instead of setting .step_hook.")

    N = queries_idx.shape[0]
    dev = queries_idx.device
    D = grounder.depth
    depths = torch.full((N,), -1, dtype=torch.long, device=dev)
    t0 = time.time()

    # One grounder per truncated depth (shared KB), reusing the passed grounder
    # at d==D when it already carries evidence.
    per_depth = {
        d: (grounder if (d == D and grounder.collect_evidence) else _truncate(grounder, d))
        for d in range(1, D + 1)
    }

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        bq = queries_idx[batch_start:batch_end]
        B = bq.shape[0]
        qmask = torch.ones(B, dtype=torch.bool, device=dev)
        init_kw: dict = {}
        if excluded_queries is not None:
            init_kw["excluded_queries"] = excluded_queries[batch_start:batch_end]
        batch_depths = torch.full((B,), -1, dtype=torch.long, device=dev)

        for d in range(1, D + 1):
            if bool((batch_depths >= 0).all()):
                break
            out = per_depth[d](bq, qmask, **init_kw)
            ev = out.evidence
            proved = (ev.count > 0) if ev is not None else torch.zeros(B, dtype=torch.bool, device=dev)
            newly = proved & (batch_depths < 0)
            batch_depths[newly] = d

        depths[batch_start:batch_end] = batch_depths
        n_proven = int((depths[:batch_end] >= 0).sum())
        print(f"\r  BFS: {batch_end}/{N} queries, proved={n_proven}", end="", flush=True)
    print()

    depths_cpu = depths.cpu().tolist()
    pred_indices = queries_idx[:, 0].cpu().tolist()
    results = [BFSResult(depth=depths_cpu[i], predicate_idx=pred_indices[i])
               for i in range(N)]
    stats = BatchBFSStats(avg_max_frontier=0.0)

    n_proven = sum(1 for d in depths_cpu if d >= 0)
    print(f"  BFS complete: {n_proven}/{N} proved in {time.time() - t0:.1f}s")
    return depths, results, stats


__all__ = ["BFSResult", "BatchBFSStats", "run_bfs"]
