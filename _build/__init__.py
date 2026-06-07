"""grounder (redesign, staged under ``_build``) — public API.

After promotion (``_build/*`` → package root) these are the top-level
``grounder.*`` names consumers import. Today: ``grounder._build.*``.

    from grounder._build import BackwardGrounder, create_grounder, make_bcwd, KB, KGDataset
"""
from grounder._build.data import KB, KGDataset, Encoding
from grounder._build.factory import create_grounder, make_bcwd, parse_grounder_type
from grounder._build.grounder import BackwardGrounder
from grounder._build.types import (
    GrounderOutput, ProofState, ProofEvidence, RuleGroundings,
)

__all__ = [
    "BackwardGrounder", "create_grounder", "make_bcwd", "parse_grounder_type",
    "KB", "KGDataset", "Encoding",
    "GrounderOutput", "ProofState", "ProofEvidence", "RuleGroundings",
]


def __getattr__(name):
    if name == "ForwardGrounder":
        from grounder._build.forward.grounder import ForwardGrounder
        return ForwardGrounder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
