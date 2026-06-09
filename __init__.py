"""grounder — public API.

The top-level ``grounder.*`` names consumers import:

    from grounder import make_grounder, BackwardGrounder, KB
    from grounder import Backward, Forward, PBC, SLD, RTF
    g = make_grounder(kb, Backward(PBC(depth=2, width=1)))
"""
from grounder.api.config import Backward, Forward, PBC, RTF, SLD
from grounder.data import KB, KGDataset, Encoding
from grounder.api.factory import make_grounder
from grounder.api import BackwardGrounder
from grounder.core import BackwardResult
from grounder.base.types import CompletedTreeFirings, GoalState, RuleGroundings

__all__ = [
    "make_grounder",
    "BackwardGrounder",
    "Backward", "Forward", "PBC", "SLD", "RTF",
    "KB", "KGDataset", "Encoding",
    "BackwardResult", "CompletedTreeFirings", "GoalState", "RuleGroundings",
]


def __getattr__(name):
    if name == "ForwardGrounder":
        from grounder.api.forward import ForwardGrounder
        return ForwardGrounder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
