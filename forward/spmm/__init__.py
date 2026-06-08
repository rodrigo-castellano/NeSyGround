"""SpMM forward chaining: sparse-matrix-multiplication closure.

Computes the FC closure (provable atoms) using sparse matrix
representations per predicate. For each predicate ``p``, build a
``[E, E]`` sparse boolean CSR matrix where ``(s, o) = 1`` iff
``p(s, o)`` is a fact. A 2-body chain rule ``p(X,Y), q(Y,Z) → r(X,Z)``
collapses to a sparse matmul ``M_p @ M_q``.

Module layout (each file owns one responsibility):

  classify.py    — ``SpMMOp``, ``SpMMRuleDesc``, ``classify_rule``
  matrices.py    — sparse helpers (build, transpose, merge, hash IO)
  apply.py       — ``apply_spmm_rule_slots`` per-rule fire kernel
  state.py       — ``ClosureState``, ``TransposeCache`` (lazy)
  strategies.py  — ``IterationStrategy``: Seminaive / TrueIStar / Hybrid
  engine.py      — ``run_forward_chaining_spmm`` semi-naive fixpoint loop

Public API (re-exported here):

  ``run_forward_chaining_spmm`` — the main entrypoint.
  ``classify_rule``, ``SpMMOp``, ``SpMMRuleDesc`` — for callers that
    want to introspect classification.
  ``build_pred_sparse_matrices``, ``apply_spmm_rule``,
    ``apply_spmm_rule_slots`` — low-level building blocks.

All functions accept a ``device`` argument and run unchanged on
``cuda`` — sparse CSR matmul, transpose, and the hash-set ops all
honour the input tensor's device.
"""
from grounder.forward.spmm.apply import apply_spmm_rule, apply_spmm_rule_slots
from grounder.forward.spmm.matrices import build_pred_sparse_matrices
from grounder.forward.spmm.classify import SpMMOp, SpMMRuleDesc, classify_rule
from grounder.forward.spmm.engine import run_forward_chaining_spmm
from grounder.forward.spmm.state import ClosureState, TransposeCache
from grounder.forward.spmm.strategies import (
    HybridStrategy, IterationStrategy, SeminaiveStrategy, TrueIStarStrategy,
)


__all__ = [
    "SpMMOp",
    "SpMMRuleDesc",
    "classify_rule",
    "build_pred_sparse_matrices",
    "apply_spmm_rule",
    "apply_spmm_rule_slots",
    "run_forward_chaining_spmm",
    "ClosureState",
    "TransposeCache",
    "IterationStrategy",
    "SeminaiveStrategy",
    "TrueIStarStrategy",
    "HybridStrategy",
]
