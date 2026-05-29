"""Back-compat shim — ``init_states`` now lives in :mod:`grounder.bc.buffers`.

Phase 2c relocated the state-buffer allocator into ``bc/buffers.py`` (the
single buffer-allocation owner). This module re-exports it so existing
imports (``from grounder.bc.state import init_states``) keep working.
"""
from __future__ import annotations

from grounder.bc.buffers import init_states

__all__ = ["init_states"]
