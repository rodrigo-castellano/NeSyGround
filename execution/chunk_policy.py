"""ChunkPolicy — a pure value object: how to split queries into padded chunks.

Static memory math only (no data-dependent probe). ``ExecStrategy`` consumes this
to drive ``iter_chunks``/``merge``; this object owns no mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from grounder.vocab.shapes import Shapes


@dataclass(frozen=True)
class ChunkPolicy:
    batch_size: int = 0          # 0 = whole input in one chunk (no chunking)

    @staticmethod
    def auto(*, peak_budget_bytes: int, shapes: Shapes) -> "ChunkPolicy":
        """Pick a chunk size so the dominant dense intermediate
        ``[B*G, K_r, Y_r, M, 3]`` (long = 8 bytes) stays under the budget."""
        per_query = shapes.G * shapes.K_r * shapes.Y_r * shapes.M * 3 * 8
        b = max(1, peak_budget_bytes // max(per_query, 1))
        return ChunkPolicy(batch_size=int(b))

    def slices(self, n: int) -> List[slice]:
        """Contiguous chunk slices covering ``n`` queries."""
        bs = self.batch_size or n
        bs = max(1, bs)
        return [slice(i, min(i + bs, n)) for i in range(0, n, bs)]


__all__ = ["ChunkPolicy"]
