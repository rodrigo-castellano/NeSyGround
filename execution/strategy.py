"""ExecStrategy — the resolved execution policy for one run.

Owns EVERY compile/layout/chunk/cudagraph decision and all chunk mechanics; the
engine names none of torch.compile / CUDA graphs / a layout directly. ``merge``
stitches chunks and threads ``RunState`` but DELEGATES rule-grounding assembly to
an engine-provided ``finalize_fn`` — so this module never imports grounding
semantics (no convert/dedup/types-output imports).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

from torch import Tensor

from grounder.execution.auto_select import auto_select
from grounder.execution.capability import (
    CapabilityRow, Cell, Integration, validate,
)
from grounder.execution.chunk_policy import ChunkPolicy
from grounder.execution.compiler import Compiler
from grounder.execution.depth import DepthSelector
from grounder.vocab.shapes import Shapes
from grounder.base.types import Layout

# Conservative default budget for the dense/flat enumeration peak when no explicit
# chunk_size is given. Deliberately small: the static estimate under-counts the
# V≥2 Cartesian (K_v^V) blowup, so a larger auto budget OOMs high-fanout cells.
# The robust path is an explicit per-dataset chunk_size (see GrounderAdapter /
# torch-ns DATASET_GROUNDER_CFG); this is only the unconfigured fallback.
_PEAK_BUDGET_BYTES = 1_000_000_000   # ~1 GB


@dataclass(frozen=True)
class ExecStrategy:
    cell: Cell
    chunk: ChunkPolicy

    # ── construction ──
    @staticmethod
    def default_chunk(shapes: Shapes) -> ChunkPolicy:
        """The auto chunk policy (one owner; plan.py needs no private budget)."""
        return ChunkPolicy.auto(peak_budget_bytes=_PEAK_BUDGET_BYTES, shapes=shapes)

    @staticmethod
    def auto(row: CapabilityRow, *, shapes: Shapes) -> "ExecStrategy":
        cell = auto_select(row, K_f=shapes.K_f, K_r=shapes.K_r, Y_r=shapes.Y_r,
                           M=shapes.M, B=shapes.B)
        return ExecStrategy(cell, ExecStrategy.default_chunk(shapes))

    @staticmethod
    def explicit(row: CapabilityRow, cell: Cell, chunk: ChunkPolicy) -> "ExecStrategy":
        validate(row, cell)
        return ExecStrategy(cell, chunk)

    # ── policy ──
    def layout(self) -> Layout:
        return self.cell.layout

    def clones_between_steps(self) -> bool:
        """CUDA-graph paths must clone step outputs out of the private pool."""
        return self.cell.compile.uses_cudagraphs()

    def depth_selector(self, d: int, depth: int,
                       device: Optional[Any] = None) -> DepthSelector:
        return DepthSelector.make(d, depth,
                                  compiled=not self.cell.compile.eager, device=device)

    # ── compile (routes through the single Compiler.wrap site) ──
    def wrap_step(self, step_fn: Callable, shapes: Shapes) -> Callable:
        """Per-step inner compile (compiled spec with integration=STEP); else unchanged."""
        spec = self.cell.compile
        if (not spec.eager) and spec.integration is Integration.STEP:
            return Compiler.wrap(step_fn, spec=spec, shapes=shapes)
        return step_fn

    def wrap_batch(self, batch_fn: Callable, shapes: Shapes) -> Callable:
        """Whole-batch outer compile (compiled spec with integration=OUTER); else unchanged."""
        spec = self.cell.compile
        if (not spec.eager) and spec.integration is Integration.OUTER:
            return Compiler.wrap(batch_fn, spec=spec, shapes=shapes)
        return batch_fn

    # ── chunk mechanics ──
    def iter_chunks(self, queries: Tensor, mask: Tensor
                    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """Yield (chunk_queries, chunk_mask) per chunk (the loop rebatches its plan
        via ``for_chunk`` and tracks the query offset in ``RunState``)."""
        for sl in self.chunk.slices(queries.shape[0]):
            yield queries[sl], mask[sl]

    def merge(self, parts: Sequence[Any], run_state: Any,
              finalize_fn: Callable[[Sequence[Any], Any], Any]) -> Any:
        """Stitch chunk parts + threaded RunState into the final output. Delegates
        the grounding-semantics build to the engine's ``finalize_fn`` (this module
        imports no grounding semantics)."""
        return finalize_fn(parts, run_state)


__all__ = ["ExecStrategy"]
