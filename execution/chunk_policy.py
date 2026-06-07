"""ChunkPolicy — a pure value object: how to split queries into padded chunks.

Static memory math only (no data-dependent probe). ``ExecStrategy`` consumes this
to drive ``iter_chunks``/``merge``; this object owns no mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from grounder.shapes import Shapes


@dataclass(frozen=True)
class ChunkPolicy:
    batch_size: int = 0          # 0 = whole input in one chunk (no chunking)

    @staticmethod
    def auto(*, peak_budget_bytes: int, shapes: Shapes) -> "ChunkPolicy":
        """Pick a chunk size so the dominant dense intermediate
        ``[B*S, K_r, G_r, M, 3]`` (long = 8 bytes) stays under the budget."""
        per_query = shapes.S * shapes.K_r * shapes.G_r * shapes.M * 3 * 8
        b = max(1, peak_budget_bytes // max(per_query, 1))
        return ChunkPolicy(batch_size=int(b))

    def slices(self, n: int) -> List[slice]:
        """Contiguous chunk slices covering ``n`` queries."""
        bs = self.batch_size or n
        bs = max(1, bs)
        return [slice(i, min(i + bs, n)) for i in range(0, n, bs)]


__all__ = ["ChunkPolicy"]
