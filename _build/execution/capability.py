"""Capability matrix — what (layout, compile-spec) cells a grounder supports.

Each grounder DECLARES a SPARSE set of ``Cell``s (``CapabilityRow``); ``validate``
rejects globally-illegal combos and any undeclared cell.

Compile config is GRANULAR: ``CompileSpec`` exposes the orthogonal axes
(eager · integration step/outer · backend · dynamic · fullgraph). The proven
combinations are named presets (``EAGER`` / ``COMPILED_STEP`` /
``COMPILED_DYNAMIC`` / ``OUTER_REDUCE_OVERHEAD``) — the default surface used by auto_select and
declare_strategies — while experiments (e.g. the L3-join follow-up) may build a
custom spec. ``CompileSpec`` self-validates illegal axis combos at construction;
``validate`` adds the layout-vs-compile and membership rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import FrozenSet, Optional

from grounder._build.errors import StrategyError
from grounder._build.types import Layout


class Integration(StrEnum):
    STEP = "step"        # per-step inner compile
    OUTER = "outer"      # whole-batch outer compile


class CompileBackend(StrEnum):
    REDUCE_OVERHEAD = "reduce_overhead"   # inductor + CUDA graphs
    DEFAULT = "default"                   # inductor, no CUDA graphs
    MAX_AUTOTUNE = "max_autotune"         # autotuned, no CUDA graphs

    def torch_mode(self) -> str:
        return {"reduce_overhead": "reduce-overhead", "default": "default",
                "max_autotune": "max-autotune-no-cudagraphs"}[self.value]


_CUDAGRAPH_BACKENDS = (CompileBackend.REDUCE_OVERHEAD, CompileBackend.MAX_AUTOTUNE)


@dataclass(frozen=True)
class CompileSpec:
    """The orthogonal compile axes. Self-validates at construction; presets below
    cover the proven combinations. When ``eager`` is True the other axes are inert."""

    eager: bool = True
    integration: Integration = Integration.STEP
    backend: CompileBackend = CompileBackend.REDUCE_OVERHEAD
    dynamic: Optional[bool] = False        # False | None(auto) | True
    fullgraph: bool = True

    def __post_init__(self) -> None:
        if self.eager:
            return
        if self.backend in _CUDAGRAPH_BACKENDS and self.dynamic is True:
            raise StrategyError(
                f"{self.backend} uses CUDA graphs — incompatible with dynamic=True")

    def uses_cudagraphs(self) -> bool:
        return (not self.eager) and self.backend in _CUDAGRAPH_BACKENDS


# ── Named presets: the proven combinations (the default surface) ──
EAGER = CompileSpec(eager=True)
COMPILED_STEP = CompileSpec(eager=False, integration=Integration.STEP,
                            backend=CompileBackend.REDUCE_OVERHEAD,
                            dynamic=False, fullgraph=True)
COMPILED_DYNAMIC = CompileSpec(eager=False, integration=Integration.STEP,
                               backend=CompileBackend.DEFAULT,
                               dynamic=True, fullgraph=True)
OUTER_REDUCE_OVERHEAD = CompileSpec(eager=False, integration=Integration.OUTER,
                                    backend=CompileBackend.REDUCE_OVERHEAD,
                                    dynamic=None, fullgraph=True)


@dataclass(frozen=True)
class Cell:
    """One supported execution point: a layout + a compile spec."""

    layout: Layout
    compile: CompileSpec


@dataclass(frozen=True)
class CapabilityRow:
    """The SPARSE set of cells one grounder declares it supports."""

    cells: FrozenSet[Cell]

    def supports(self, cell: Cell) -> bool:
        return cell in self.cells


def validate(row: CapabilityRow, cell: Cell) -> None:
    """Raise ``StrategyError`` on a globally-illegal combo or an undeclared cell.

    Globally illegal (independent of grounder):
      - flat/sparse layout + any compile : flat is data-dependent (nonzero/.item)
        and sparse is forward-chaining ⇒ both run eager only.
    (Internal axis consistency, e.g. reduce-overhead ⊥ dynamic=True, is already
    guaranteed by ``CompileSpec.__post_init__``. Grounder-specific illegality is
    encoded by the grounder NOT declaring the cell — caught by membership.)
    """
    if cell.layout in (Layout.FLAT, Layout.SPARSE) and not cell.compile.eager:
        raise StrategyError(f"{cell.layout} layout runs eager only (got non-eager compile)")
    if not row.supports(cell):
        raise StrategyError(f"cell {cell} not in the grounder's declared row {set(row.cells)}")


__all__ = [
    "Integration", "CompileBackend", "CompileSpec",
    "EAGER", "COMPILED_STEP", "COMPILED_DYNAMIC", "OUTER_REDUCE_OVERHEAD",
    "Cell", "CapabilityRow", "validate",
]
