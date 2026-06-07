"""Backward — the backward proof-search loop (port of OLD ``bc/``).

    run_backward = init_frontier → for d in range(depth): step → finalize

``step`` = SELECT → RESOLVE (dispatch sld/rtf/pbc) → PACK → POSTPROCESS(+sync).
The considered accumulator (PRIMARY for RuleGroundings) captures every firing
between resolve and pack; fp_batch prunes after the loop.
"""
from grounder._build.backward.buffers import init_frontier
from grounder._build.backward.finalize import (
    build_proof_state, finalize_evidence, finalize_rule_groundings,
)
from grounder._build.backward.loop import run_backward
from grounder._build.backward.step import step

__all__ = [
    "init_frontier", "step", "run_backward",
    "build_proof_state", "finalize_evidence", "finalize_rule_groundings",
]
