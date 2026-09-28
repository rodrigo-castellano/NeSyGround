"""``run_forward_chaining``: the closure engine for a rule set — spmm when every rule has a sparse-matrix form, else
the staged join."""
from __future__ import annotations

from typing import List, Tuple

from torch import Tensor

from grounder.kb import RulePattern


def run_forward_chaining(compiled_rules: List[RulePattern], facts_idx: Tensor, num_entities: int, num_predicates: int,
                         depth: int = 10, device: str = "cpu", *, method: str = "spmm") -> Tuple[Tensor, int]:
    """``(hashes, n)``: the atoms the rules derive from ``facts_idx [F, 3]`` within ``depth`` steps, as sorted hashes
    ``p * E**2 + s * E + o`` (``E = num_entities``, above every entity id; ``hashes[:n]``; a single 0 when ``n == 0``).
    ``method="spmm"`` takes the sparse-matmul engine when ``spmm.classify`` gives every rule a matrix form (1-body copies
    and transposes, 2-body chains and joins, 3-body chains), else the staged join (any rule shape, constants)."""
    from grounder.forward.join.engine import FCDynamic
    from grounder.forward.spmm import SpMMOp, classify_rule, run_forward_chaining_spmm
    if method == "spmm" and all(classify_rule(cr).op != SpMMOp.UNSUPPORTED for cr in compiled_rules):
        return run_forward_chaining_spmm(compiled_rules, facts_idx, num_entities, num_predicates, depth=depth,
                                         device=device, verbose=False)
    return FCDynamic(compiled_rules, facts_idx, num_entities, num_predicates, device).run(depth)


__all__ = ["run_forward_chaining"]
