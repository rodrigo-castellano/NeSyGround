"""FilterHook protocol — interface for logical and nesy filters.

All filters at each of the 5 hook points implement this protocol.
A filter receives candidates and a mask, and returns an updated mask.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class FilterHook(Protocol):
    """Protocol for filter hooks at any of the 5 BCGrounder hook points.

    Filters receive candidates and a validity mask, and return an updated mask.
    True = keep, False = reject.

    Args:
        candidates: Tensor of candidates (shape depends on hook point).
        mask: Bool tensor — current validity mask.
        **context: Additional context (e.g. depth, fact_index, padding_idx).

    Returns:
        Updated mask tensor (same shape as input mask).
    """

    def __call__(
        self,
        candidates: Tensor,
        mask: Tensor,
        **context,
    ) -> Tensor:
        ...
