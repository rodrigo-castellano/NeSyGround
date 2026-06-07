"""Grounder shells — the consumer-facing grounder classes.

    BackwardGrounder — query-directed proof search (sld / rtf / pbc).
    ForwardGrounder  — forward chaining (lazy import; built in forward/).
"""
from grounder._build.grounder.backward import BackwardGrounder

__all__ = ["BackwardGrounder", "ForwardGrounder"]


def __getattr__(name):
    # Lazy so importing the backward shell doesn't require forward/ to exist.
    if name == "ForwardGrounder":
        from grounder._build.forward.grounder import ForwardGrounder
        return ForwardGrounder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
