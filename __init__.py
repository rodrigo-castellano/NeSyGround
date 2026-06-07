"""grounder (redesign, staged under ``_build``) — public API.

After promotion (``_build/*`` → package root) these are the top-level
``grounder.*`` names consumers import. Today: ``grounder.*``.

    from grounder import make_grounder, BackwardGrounder, create_grounder, make_bcwd, KB
"""
from grounder.config import FCConfig, PBCConfig, RTFConfig, SLDConfig
from grounder.data import KB, KGDataset, Encoding
from grounder.factory import (
    create_grounder, make_bcwd, make_grounder, parse_grounder_type,
)
from grounder.grounder import BackwardGrounder
from grounder.types import (
    BackwardResult, CompletedTreeFirings, GrounderOutput, ProofEvidence,
    ProofState, RuleGroundings,
)

__all__ = [
    "make_grounder", "create_grounder", "make_bcwd", "parse_grounder_type",
    "BackwardGrounder",
    "SLDConfig", "RTFConfig", "PBCConfig", "FCConfig",
    "KB", "KGDataset", "Encoding",
    "BackwardResult", "CompletedTreeFirings", "ProofState", "RuleGroundings",
    # one-window back-compat aliases
    "GrounderOutput", "ProofEvidence",
]


def __getattr__(name):
    if name == "ForwardGrounder":
        from grounder.forward.grounder import ForwardGrounder
        return ForwardGrounder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
