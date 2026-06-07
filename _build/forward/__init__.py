"""Forward chaining — FC as a first-class grounder (not a filter).

Faithful port of the working old ``fc/`` engine (the from-scratch rewrite hit a
torch.sparse SpGEMM col-ordering crash; this intricate sparse-kernel leaf is
ported verbatim with repointed imports and gated by an exact closure-set A/B).

  fc.py         — run_forward_chaining (entry) + FCDynamic (staged/leapfrog triejoin)
  spmm/         — semi-naive sparse-matmul FC (the default ``method='spmm'``)
  closure.py    — closure-set builder + KB augmentation (the fp_global use-case)
  grounder.py   — ForwardGrounder (the consumer-facing FC grounder)
"""
from grounder._build.forward.fc import FCDynamic, run_forward_chaining
from grounder._build.forward.grounder import ForwardGrounder
from grounder._build.forward.spmm import (
    SpMMOp, classify_rule, run_forward_chaining_spmm,
)

__all__ = [
    "ForwardGrounder", "run_forward_chaining", "FCDynamic",
    "run_forward_chaining_spmm", "classify_rule", "SpMMOp",
]
