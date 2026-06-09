"""Per-rule forward-chaining router — ``run_forward_chaining``.

Routes a rule set over ``FORWARD_METHODS`` (AXIS 3): the spmm whole-set fast path
is taken iff ``method='spmm'`` and ``SpmmMethod.supports(rules)`` (every rule
SpMM-classifiable); otherwise it falls through to ``StagedMethod`` (FCDynamic),
which handles the full rule mix.
"""
from __future__ import annotations

from typing import List, Tuple

from torch import Tensor

from grounder.data.rule_index import RulePattern


def run_forward_chaining(
    compiled_rules: List[RulePattern],
    facts_idx: Tensor,
    num_entities: int,
    num_predicates: int,
    depth: int = 10,
    device: str = "cpu",
    *,
    method: str = "spmm",
    join_algo: str = "staged",
    join_chunk_size: int = 0,
) -> Tuple[Tensor, int]:
    """Run forward chaining and return (sorted_hashes, n_provable).

    Args:
        compiled_rules: List of RulePattern from grounder/compilation.py.
        facts_idx: [F, 3] raw fact triples.
        num_entities: Total entity count.
        num_predicates: Total predicate count.
        depth: Max FC iterations.
        device: Target device.
        method: Forward-chaining strategy.
            * ``'spmm'`` (default) — semi-naive sparse-matrix-multiplication
              FC. Per-predicate ``[E, E]`` sparse CSR matrices; rules
              compile to MATMUL / ELEM_AND / CASE_A / EXIST_AND ops on
              those matrices with delta-tracked semi-naive iteration.
              Scales to 100M+ atoms on commodity hardware (validated
              on fb15k237). Falls back to ``'staged'`` automatically
              when the rule set has any 3-body rule (SpMM doesn't
              support those).
            * ``'staged'`` — staged ragged join (FCDynamic). Slower per
              atom and bounded by Python dispatch overhead; use for rule
              sets with 3+ body atoms or when SpMM classifies a rule as
              ``UNSUPPORTED``.
        join_algo: ``staged``-only knob. Per-rule join strategy:
            ``'staged'`` (default) or ``'chunked'``.
        join_chunk_size: Rows per chunk in the ``staged`` chunked
            path. ``0`` = the FCDynamic default (100k rows).

    Returns:
        sorted_hashes: 1-D sorted tensor of provable atom hashes.
        n_provable: Number of provable atoms (0 if none).
    """
    # Per-rule ROUTER over FORWARD_METHODS (AXIS 3). The spmm WHOLE-SET fast
    # path is gated by SpmmMethod.supports(): SpMM handles 1/2/3-body rules
    # whose binding shape matches the ops table (COPY, TRANSPOSE, MATMUL,
    # ELEM_AND, CASE_A, EXIST_AND, MATMUL3). Anything else — 4+ body rules or
    # 1/2/3-body rules classified UNSUPPORTED (e.g. compositional rules wrapped
    # with a disequality-guard body) — falls through to StagedMethod (FCDynamic)
    # so it can handle the full mix. Reproduces the historical fallback exactly.
    from grounder.forward.methods import FORWARD_METHODS
    spmm = FORWARD_METHODS["spmm"]
    chosen = spmm if (method == "spmm" and spmm.supports(compiled_rules)) \
        else FORWARD_METHODS["staged"]
    closure = chosen.run(
        compiled_rules, facts_idx, num_entities, num_predicates,
        depth=depth, device=device, join_algo=join_algo,
        join_chunk_size=join_chunk_size)
    return closure.hashes, closure.n_provable


__all__ = ["run_forward_chaining"]
