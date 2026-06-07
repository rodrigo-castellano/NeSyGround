"""auto_select — pick a Cell within a grounder's declared row, keyed on the STATIC
budget (fan-out K_f, etc.). Never probes data-dependent per-query statistics,
never returns a dynamic mode or max-autotune (both dominated in the study).

Policy (verified by the execution study):
  - forward                 -> (sparse, eager)            [only cell]
  - sld / rtf               -> (dense, eager)             [dodge the 20-33s warmup;
                               outer reduce-overhead is opt-in via explicit()]
  - pbc, low fan-out        -> (flat, eager)              [~6x faster, light]
  - pbc, high fan-out       -> (dense, compiled_step)     [the only BOUNDED path]
"""
from __future__ import annotations

from grounder.execution.capability import (
    CapabilityRow, Cell, COMPILED_STEP, EAGER, validate,
)
from grounder.errors import StrategyError
from grounder.types import Layout

# fan-out (K_f) at/below which flat-eager is preferred for PBC; above it, flat's
# data-dependent survivor set risks the per-query OOM cliff -> bounded compiled-dense.
FANOUT_FLAT_MAX = 256


def auto_select(row: CapabilityRow, *, K_f: int, K_r: int, G_r: int,
                M: int, B: int) -> Cell:
    """Return a declared cell best for this static budget."""
    sparse = Cell(Layout.SPARSE, EAGER)
    flat = Cell(Layout.FLAT, EAGER)
    dense_eager = Cell(Layout.DENSE, EAGER)
    dense_compiled = Cell(Layout.DENSE, COMPILED_STEP)

    # forward grounder
    if row.supports(sparse):
        return _ok(row, sparse)
    # PBC: fan-out keyed
    if K_f <= FANOUT_FLAT_MAX and row.supports(flat):
        return _ok(row, flat)
    if row.supports(dense_compiled):
        return _ok(row, dense_compiled)
    # sld/rtf (or PBC fallback): plain eager dense dodges the warmup tax
    for fallback in (flat, dense_eager):
        if row.supports(fallback):
            return _ok(row, fallback)
    raise StrategyError(f"auto_select: no usable cell in declared row {set(row.cells)}")


def _ok(row: CapabilityRow, cell: Cell) -> Cell:
    validate(row, cell)
    return cell


__all__ = ["auto_select", "FANOUT_FLAT_MAX"]
