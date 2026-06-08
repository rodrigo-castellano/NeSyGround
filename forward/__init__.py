"""Forward chaining — FC as a first-class grounder (not a filter).

Faithful port of the working old ``fc/`` engine (the from-scratch rewrite hit a
torch.sparse SpGEMM col-ordering crash; this intricate sparse-kernel leaf is
ported verbatim with repointed imports and gated by ``tests/fc_fingerprint.py``).

  api.py        — Closure (FC's GroundResult type)
  methods.py    — ForwardMethod seam (AXIS 3): SpmmMethod/StagedMethod + FORWARD_METHODS
  fc.py         — run_forward_chaining (per-rule router) + FCDynamic (staged/leapfrog triejoin)
  spmm/         — semi-naive sparse-matmul FC (the default ``method='spmm'``)

The consumer-facing ``ForwardGrounder`` shell lives in ``grounder.api.forward``
(sibling of ``grounder.api.backward``); the engine here is its impl.
"""
from grounder.forward.api import Closure
from grounder.forward.fc import FCDynamic, run_forward_chaining
from grounder.forward.methods import (
    FORWARD_METHODS, ForwardMethod, SpmmMethod, StagedMethod,
)
from grounder.forward.spmm import (
    SpMMOp, classify_rule, run_forward_chaining_spmm,
)

__all__ = [
    "Closure", "run_forward_chaining", "FCDynamic",
    "run_forward_chaining_spmm", "classify_rule", "SpMMOp",
    "ForwardMethod", "SpmmMethod", "StagedMethod", "FORWARD_METHODS",
]
