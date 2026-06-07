"""``Pipeline`` — Transform∘Grounder is itself a ``Grounder``.

Applies an ordered sequence of pure ``(kb, req) -> (kb', req')`` program
transforms, then re-snapshots the base grounder over the rewritten KB and grounds.
ADDITIVE (Phase A): the ``ProgramTransform`` seam (``transform/api.py``) lands in
Step 6; until then this composes with an empty transform list (identity).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from grounder.core.grounder import Grounder
from grounder.core.request import GroundRequest
from grounder.core.result import GroundResult

if TYPE_CHECKING:
    from grounder.transform.api import ProgramTransform


class Pipeline:
    """A grounder that pre-stages program transforms (order-significant)."""

    def __init__(self, transforms: "Sequence[ProgramTransform]", base: Grounder) -> None:
        self.transforms = tuple(transforms)
        self.base = base

    @property
    def kb(self):
        return self.base.kb

    def ground(self, req: GroundRequest) -> GroundResult:
        kb = self.base.kb
        for t in self.transforms:
            kb, req = t.apply(kb, req)
        return self.base.rebound(kb).ground(req)

    def capability_row(self):
        return self.base.capability_row()

    def producible_tiers(self):
        return self.base.producible_tiers()

    def rebound(self, kb) -> "Pipeline":
        return Pipeline(self.transforms, self.base.rebound(kb))


__all__ = ["Pipeline"]
