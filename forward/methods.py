"""AXIS 3 — the ForwardMethod seam (the closure ENGINE {spmm, staged}).

A ``ForwardMethod`` is the forward-chaining closure engine: ``SpmmMethod``
(semi-naive sparse-matmul) and ``StagedMethod`` (the staged ragged join,
FCDynamic). ``run_forward_chaining`` (fc.py) is the per-rule ROUTER over these:
the spmm WHOLE-SET fast path is gated by ``SpmmMethod.supports(rules)`` (all
``classify_rule != UNSUPPORTED``); on any UNSUPPORTED rule it falls through to
``StagedMethod``.

The JoinAlgo sub-axis ``{staged, chunked}`` lives INSIDE ``StagedMethod`` (the
FCDynamic ``join_algo`` knob); ``leapfrog`` is a documented placeholder, NOT a
registered extension point. The spmm IterationStrategy stays a SUB-strategy
inside SpmmMethod (the ``mode`` knob on ``run_forward_chaining_spmm``).
"""
from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from torch import Tensor

from grounder.data.rule_index import RulePattern
from grounder.forward.api import Closure


@runtime_checkable
class ForwardMethod(Protocol):
    name: str                                       # "spmm" | "staged"

    def supports(self, rules: List[RulePattern]) -> bool: ...   # whole-set fast-path gate

    def run(self, rules: List[RulePattern], facts_idx: Tensor,
            num_entities: int, num_predicates: int, *, depth: int, device: str,
            join_algo: str, join_chunk_size: int) -> Closure: ...


class SpmmMethod:
    """Semi-naive sparse-matmul FC. Supports the whole set iff every rule
    classifies to a non-UNSUPPORTED SpMM op (1/2/3-body of a handled shape)."""

    name = "spmm"

    def supports(self, rules: List[RulePattern]) -> bool:
        from grounder.forward.spmm.ops import SpMMOp, classify_rule
        return all(classify_rule(cr).op != SpMMOp.UNSUPPORTED for cr in rules)

    def run(self, rules: List[RulePattern], facts_idx: Tensor,
            num_entities: int, num_predicates: int, *, depth: int, device: str,
            join_algo: str, join_chunk_size: int) -> Closure:
        from grounder.forward.spmm import run_forward_chaining_spmm
        hashes, n = run_forward_chaining_spmm(
            rules, facts_idx, num_entities, num_predicates,
            depth=depth, device=device)
        return Closure(hashes=hashes, n_provable=n, num_entities=num_entities)


class StagedMethod:
    """Staged ragged join (FCDynamic). Handles the full rule mix; the
    JoinAlgo sub-axis {staged, chunked} is its ``join_algo`` knob."""

    name = "staged"

    def supports(self, rules: List[RulePattern]) -> bool:
        return True

    def run(self, rules: List[RulePattern], facts_idx: Tensor,
            num_entities: int, num_predicates: int, *, depth: int, device: str,
            join_algo: str, join_chunk_size: int) -> Closure:
        from grounder.forward.fc import FCDynamic
        fc = FCDynamic(rules, facts_idx, num_entities, num_predicates, device,
                       join_algo=join_algo, join_chunk_size=join_chunk_size)
        hashes, n = fc.run(depth)
        return Closure(hashes=hashes, n_provable=n, num_entities=num_entities)


FORWARD_METHODS: Dict[str, ForwardMethod] = {
    "spmm": SpmmMethod(),
    "staged": StagedMethod(),
}

__all__ = ["ForwardMethod", "SpmmMethod", "StagedMethod", "FORWARD_METHODS"]
