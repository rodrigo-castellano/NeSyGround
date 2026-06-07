"""AXIS 3 — the forward result type ``Closure`` (and, later, the ForwardMethod seam).

``Closure`` is the FC family's honest ``GroundResult``: provable triple hashes
(``h = pred*E^2 + subj*E + obj``) + ``n_provable`` + ``num_entities`` (E, for
decode). It is set-valued, so it carries NO provenance — ``as_rule_groundings``
returns ``None`` (verified: FC has no firings today). ``facts()`` decodes to
``[N, 3]`` triples EXACTLY as the old ``ForwardGrounder.closure_facts()`` did
(gated byte-for-byte by ``tests/fc_fingerprint.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class Closure:
    """FC's result — provable-atom hashes + count + entity range (for decode)."""

    hashes: Tensor          # [N] sorted, h = pred*E^2 + subj*E + obj
    n_provable: int
    num_entities: int       # E, for decode
    kind: str = "closure"

    def facts(self) -> Tensor:
        """Decode to ``[N, 3]`` (pred, subj, obj) — == old ``closure_facts()``."""
        if self.n_provable == 0:
            return torch.empty(0, 3, dtype=torch.long, device=self.hashes.device)
        E = self.num_entities
        E2 = E * E
        pred = self.hashes // E2
        rem = self.hashes % E2
        return torch.stack([pred, rem // E, rem % E], dim=1)

    def contains(self, atoms: Tensor) -> Tensor:
        """Membership mask over ``[N, 3]`` query atoms (pred, subj, obj)."""
        E = self.num_entities
        h = atoms[:, 0] * (E * E) + atoms[:, 1] * E + atoms[:, 2]
        return torch.isin(h, self.hashes)

    def as_rule_groundings(self) -> Optional[object]:
        """FC has NO provenance today (set-valued closure) -> None."""
        return None


__all__ = ["Closure"]
