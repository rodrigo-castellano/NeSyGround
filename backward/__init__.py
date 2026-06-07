"""Backward — the backward proof-search loop (port of OLD ``bc/``).

    run_backward = init_frontier → for d in range(depth): step → finalize

``step`` = SELECT → RESOLVE (dispatch sld/rtf/pbc) → PACK → POSTPROCESS(+sync).
The considered accumulator (PRIMARY for RuleGroundings) captures every firing
between resolve and pack; fp_batch prunes after the loop.
"""
from grounder.backward.buffers import init_frontier
from grounder.backward.finalize import (
    build_proof_state, finalize_evidence, finalize_rule_groundings,
)
from grounder.backward.loop import run_backward
from grounder.backward.step import step

__all__ = [
    "init_frontier", "step", "run_backward",
    "build_proof_state", "finalize_evidence", "finalize_rule_groundings",
]
