"""Backward proof-search driver — ``run_backward``.

Ported from OLD ``bc/forward.py:forward_one_batch``, now chunk-aware: snapshots the
grounder into a frozen ``RunPlan`` ONCE, then drives ``plan.strategy.iter_chunks``.
Each chunk gets a rebatched ``plan.for_chunk(B_i)`` + its own ``Frontier`` (+ private
``_Collected``) and depth loop; firings accumulate into the run-scoped ``RunState``
(query_idx lifted by the global chunk offset at capture), trees fold in one
``ProofTrees`` piece per chunk via ``with_chunk``. ``strategy.merge`` then stitches
the per-chunk proof-state parts + threaded RunState via ``merge_finalize``.
The per-step fn routes through the ONE compile seam (``strategy.wrap_step``);
identity on eager cells, so default ChunkPolicy=one-chunk stays byte-identical.
"""
from __future__ import annotations

from typing import Optional

from torch import Tensor

from grounder._build.engine.buffers import init_frontier
from grounder._build.engine.finalize import build_proof_state, merge_finalize
from grounder._build.engine.step import step
from grounder._build.plan import RunPlan
from grounder._build.state import ProofTrees, RunState
from grounder._build.types import GrounderOutput


def run_backward(grounder, queries: Tensor, query_mask: Tensor,
                 *, excluded_queries: Optional[Tensor] = None,
                 **init_kwargs) -> GrounderOutput:
    """snapshot plan → per chunk (init_frontier → depth loop → emit pieces) → merge."""
    plan = RunPlan.snapshot(grounder)
    run = RunState.init(plan.output_spec)

    parts = []
    for cq, cm, csh, _offset in plan.strategy.iter_chunks(
            queries, query_mask, plan.shapes):
        cplan = plan.for_chunk(cq.shape[0])
        fr, coll = init_frontier(cplan, cq, cm,
                                 excluded_queries=excluded_queries, **init_kwargs)
        step_fn = _make_step(cplan, excluded_queries)  # single compile seam (eager=identity)
        for d in range(cplan.depth):
            dsel = cplan.strategy.depth_selector(d, cplan.depth)
            fr, coll, run = step_fn(fr, coll, run, dsel)
        parts.append(build_proof_state(cplan, fr))
        run = emit_trees(run, coll, cq.shape[0])  # one ProofTrees piece + advance offset

    return plan.strategy.merge(parts, run, plan.shapes, merge_finalize(plan))


def _make_step(plan, excluded_queries):
    """Bind the per-step fn to the chunk plan and route it through the ONE compile
    seam (``strategy.wrap_step``); identity on eager cells (all fingerprint cells)."""
    def _step(fr, coll, run, dsel):
        return step(plan, fr, coll, run, dsel, excluded_queries)
    return plan.strategy.wrap_step(_step, plan.shapes)


def emit_trees(run: RunState, coll, n_chunk: int) -> RunState:
    """Fold the chunk's _Collected into run.trees as one ProofTrees piece and
    advance the global query offset (firings were already lifted at capture)."""
    if run.trees is None or coll is None:
        return run.with_chunk(n_chunk=n_chunk)
    piece = ProofTrees.from_chunk(
        coll.collected_body, coll.collected_ridx, coll.collected_head,
        coll.collected_bcount, coll.collected_mask)
    return run.with_chunk(trees=piece, n_chunk=n_chunk)


__all__ = ["run_backward"]
