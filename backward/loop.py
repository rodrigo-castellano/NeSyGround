"""Backward proof-search driver — ``run_backward``.

Chunk-aware: snapshots the grounder into a frozen ``RunPlan`` ONCE, then drives
``plan.strategy.iter_chunks``. Each chunk gets a rebatched ``plan.for_chunk(B_i)``
+ its own ``Frontier`` (+ private ``_Collected``) and depth loop; firings
accumulate into the run-scoped ``RunState`` (query_idx lifted by the global chunk
offset at capture), trees fold in one ``ProofTrees`` piece per chunk via
``with_chunk``. ``strategy.merge`` then stitches the per-chunk proof-state parts +
threaded RunState via ``merge_finalize``. The per-step fn routes through the ONE
compile seam (``strategy.wrap_step``); identity on eager cells, so the default
ChunkPolicy=one-chunk runs the whole batch in a single pass.
"""
from __future__ import annotations

from typing import Optional

from torch import Tensor

from dataclasses import replace as _replace

from grounder.backward.buffers import init_frontier
from grounder.backward.finalize import build_proof_state, merge_finalize
from grounder.backward.step import step_core
from grounder.backward.plan import RunPlan
from grounder.backward.state import FiringSet, ProofTrees, RunState
from grounder.core import BackwardResult


def run_backward(grounder, queries: Tensor, query_mask: Tensor,
                 *, excluded_queries: Optional[Tensor] = None,
                 **init_kwargs) -> BackwardResult:
    """snapshot plan → per chunk (init_frontier → depth loop → emit pieces) → merge."""
    plan = RunPlan.snapshot(grounder)
    run = RunState.init(plan.output_spec)

    # FIRINGS-only request: the final step's packed frontier feeds nothing
    # (no next step, no GoalState tier, no TREES collection) — skip its
    # pack/postprocess and the per-chunk proof-state build entirely.
    want_proof_state = plan.output_spec.groundings
    skip_final_tail = not want_proof_state and not plan.collect_evidence

    parts = []
    for cq, cm in plan.strategy.iter_chunks(queries, query_mask):
        cplan = plan.for_chunk(cq.shape[0])
        fr, coll = init_frontier(cplan, cq, cm, **init_kwargs)
        step_fn = _make_step(cplan, excluded_queries)  # single compile seam (eager=identity)
        for d in range(cplan.depth):
            dsel = cplan.strategy.depth_selector(d, cplan.depth)
            skip_tail = skip_final_tail and (d == cplan.depth - 1)
            fr, coll, run = step_fn(fr, coll, run, dsel, skip_tail)
        if want_proof_state:
            parts.append(build_proof_state(cplan, fr))
        run = emit_trees(run, coll, cq.shape[0])  # one ProofTrees piece + advance offset

    return plan.strategy.merge(parts, run, merge_finalize(plan))


def _make_step(plan, excluded_queries):
    """Bind the per-step fn to the chunk plan: the tensor-pure ``step_core``
    routes through the ONE compile seam (``strategy.wrap_step``; identity on
    eager cells), while the ``RunState`` firing append stays in this eager rim
    — Python tuple appends cannot live inside a fullgraph compile, and the
    dense core emits fixed-shape sentinel rows precisely so it doesn't have to.
    """
    def _core(fr, coll, dsel, skip_tail):
        return step_core(plan, fr, coll, dsel, excluded_queries, skip_tail)
    core = plan.strategy.wrap_step(_core, plan.shapes)

    def _step(fr, coll, run, dsel, skip_tail=False):
        fr, coll, rows = core(fr, coll, dsel, skip_tail)
        if rows is not None:
            emission = FiringSet.from_emission(
                rows[0], rows[1], rows[2], rows[3] + run.chunk_query_offset)
            run = _replace(run, firings=run.firings.extend(emission))
        return fr, coll, run
    return _step


def emit_trees(run: RunState, coll, n_chunk: int) -> RunState:
    """Fold the chunk's _Collected into run.trees as one ProofTrees piece and
    advance the global query offset (firings were already lifted at capture)."""
    if run.trees is None or coll is None:
        return run.with_chunk(n_chunk=n_chunk)
    piece = ProofTrees.from_chunk(
        coll.collected_body, coll.collected_rule_idx, coll.collected_head,
        coll.collected_body_count, coll.collected_mask)
    return run.with_chunk(trees=piece, n_chunk=n_chunk)


__all__ = ["run_backward"]
