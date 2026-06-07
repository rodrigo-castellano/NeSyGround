"""Forward chaining — FC as a first-class grounder (not a filter).

Faithful port of the working old ``fc/`` engine (the from-scratch rewrite hit a
torch.sparse SpGEMM col-ordering crash; this intricate sparse-kernel leaf is
ported verbatim with repointed imports and gated by ``tests/fc_fingerprint.py``).

  api.py        — Closure (FC's GroundResult type)
  methods.py    — ForwardMethod seam (AXIS 3): SpmmMethod/StagedMethod + FORWARD_METHODS
  fc.py         — run_forward_chaining (per-rule router) + FCDynamic (staged/leapfrog triejoin)
  spmm/         — semi-naive sparse-matmul FC (the default ``method='spmm'``)
  grounder.py   — ForwardGrounder (the consumer-facing FC grounder)
"""
from grounder._build.forward.api import Closure
from grounder._build.forward.fc import FCDynamic, run_forward_chaining
from grounder._build.forward.grounder import ForwardGrounder
from grounder._build.forward.methods import (
    FORWARD_METHODS, ForwardMethod, SpmmMethod, StagedMethod,
)
from grounder._build.forward.spmm import (
    SpMMOp, classify_rule, run_forward_chaining_spmm,
)

__all__ = [
    "Closure", "ForwardGrounder", "run_forward_chaining", "FCDynamic",
    "run_forward_chaining_spmm", "classify_rule", "SpMMOp",
    "ForwardMethod", "SpmmMethod", "StagedMethod", "FORWARD_METHODS",
]
