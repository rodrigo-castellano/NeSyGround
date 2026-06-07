"""DepthSelector — the ONE owner of the depth ``d`` / ``is_last`` representation duality.

Eager path: ``d`` is a Python int, ``is_last`` a bool.
Compiled path: both are 0-dim tensors (so dynamo shares one graph across depths).
Every phase that consumes depth (PBC width gate, sync write-slot) takes a
DepthSelector — so ``isinstance(d, Tensor)`` appears NOWHERE else (enforced by
``contracts/test_execution.py::test_no_isinstance_d_tensor``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class DepthSelector:
    d: Union[int, Tensor]              # int (eager) | [] long (compiled)
    is_last: Union[bool, Tensor]       # bool (eager) | [] bool (compiled)
    compiled: bool

    @staticmethod
    def make(d: int, depth: int, *, compiled: bool,
             device: Optional[torch.device] = None) -> "DepthSelector":
        last = (d == depth - 1)
        if compiled:
            return DepthSelector(
                torch.tensor(d, dtype=torch.long, device=device),
                torch.tensor(last, dtype=torch.bool, device=device), True)
        return DepthSelector(d, last, False)

    def width_for(self, w: int, u: int) -> Union[int, Tensor]:
        """Per-step unknown-fact cap: ``u`` at the last step, else ``w``.
        Branch (eager) vs ``torch.where`` (compiled) — the duality lives here."""
        if self.compiled:
            dev = self.is_last.device
            return torch.where(self.is_last,
                               torch.tensor(u, dtype=torch.long, device=dev),
                               torch.tensor(w, dtype=torch.long, device=dev))
        return u if self.is_last else w

    def write_slot(self, accumulated: Tensor, value: Tensor) -> Tensor:
        """Write ``value`` into ``accumulated`` at depth slot ``d`` along dim 2
        (``[B, S, D, ...]``). Index-write (eager) vs one-hot scatter (compiled).
        Returns the updated tensor (functional; no in-place on the input)."""
        D = accumulated.shape[2]
        if self.compiled:
            onehot = torch.arange(D, device=accumulated.device) == self.d   # [D] bool
            view = [1, 1, D] + [1] * (accumulated.dim() - 3)
            return torch.where(onehot.view(*view), value.unsqueeze(2), accumulated)
        out = accumulated.clone()
        out[:, :, self.d] = value
        return out


__all__ = ["DepthSelector"]
