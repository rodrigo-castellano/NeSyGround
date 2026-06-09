"""AXIS 4 — the ProgramTransform seam contract + the IdentityTransform.

A ``ProgramTransform`` is a pure, order-significant ``(kb, req) -> (kb', req')``
rewrite of the grounding problem (e.g. magic-sets). It is composed by
``make_grounder`` into a ``core.Pipeline``: per call the pipeline threads kb/req
through each transform, then runs ``base.rebound(kb').ground(req')``.

``IdentityTransform`` is the no-op (returns kb, req unchanged) — the second real
impl that keeps the seam honest and provides the byte-identity reference: a
pipeline of identities must reproduce the bare grounder exactly.
"""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from grounder.core import GroundRequest
from grounder.data.kb import KB


@runtime_checkable
class ProgramTransform(Protocol):
    name: str

    def apply(self, kb: KB, req: GroundRequest) -> Tuple[KB, GroundRequest]:
        """Pure rewrite (order-significant); never mutates ``kb``/``req``."""
        ...


class IdentityTransform:
    """The no-op transform (returns kb, req unchanged)."""

    name = "identity"

    def apply(self, kb: KB, req: GroundRequest) -> Tuple[KB, GroundRequest]:
        return kb, req


__all__ = ["ProgramTransform", "IdentityTransform"]
