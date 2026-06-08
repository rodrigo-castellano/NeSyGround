"""Grounder shells — the consumer-facing grounder classes.

    BackwardGrounder — query-directed proof search (sld / rtf / pbc).
    ForwardGrounder  — forward chaining (lazy import; sibling forward.py).
"""
from grounder.api.backward import BackwardGrounder

__all__ = ["BackwardGrounder", "ForwardGrounder"]


def __getattr__(name):
    # Lazy so importing the backward shell doesn't require the FC engine to exist.
    if name == "ForwardGrounder":
        from grounder.api.forward import ForwardGrounder
        return ForwardGrounder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
