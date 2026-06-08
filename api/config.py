"""Typed grounder configs — the config TYPE selects the grounder family.

Two top-level configs, each dispatched by ``make_grounder`` on its TYPE:

  - ``Backward(resolution, …)`` — backward chaining; ``resolution`` is one of the
    typed resolutions ``PBC`` / ``SLD`` / ``RTF`` (depth lives ON the resolution).
  - ``Forward(depth, …)``       — forward chaining (closure to fixpoint).

There is no Resolution enum, no string grammar, and no per-leaf ``filter()``: the
filter is derived ONCE from the resolution (``PBC`` with ``u==0`` → fp_batch, else
none) at construction. PBC-specific knobs (width/u/flat_intermediate/…) live on
``PBC``; shared backward knobs live on ``Backward``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union

from grounder.base.errors import ConfigError


# ── resolutions (the WHAT axis of Backward; depth lives here) ──
@dataclass(frozen=True, kw_only=True)
class SLD:
    """Plain backward chaining, SLD resolution (fact + rule)."""

    depth: int

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ConfigError(f"depth must be >= 1, got {self.depth}")


@dataclass(frozen=True, kw_only=True)
class RTF:
    """Plain backward chaining, Rule-Then-Fact resolution."""

    depth: int

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ConfigError(f"depth must be >= 1, got {self.depth}")


@dataclass(frozen=True, kw_only=True)
class PBC:
    """Parametrized Backward Chaining — the IJCAI ``BC_{w,d,u}`` grounder.

    ``depth`` is ``d``; ``width`` is the intermediate-step unknown-fact cap (``w``);
    ``u`` is the last-step cap (paper convention 0). Always uses the fp_batch filter
    when ``u==0``. ``max_groundings_per_rule`` caps Y_r (per-rule groundings).
    """

    depth: int
    width: int = 1
    u: int = 0
    flat_intermediate: bool = True
    max_groundings_per_rule: int = 32      # Y_r cap (per rule, per query)
    cartesian_product: bool = False        # all-entities ablation
    materialization: str = "cartesian"     # pbc layout: "cartesian" | "join" (L3)

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ConfigError(f"depth must be >= 1, got {self.depth}")
        if self.materialization not in ("cartesian", "join"):
            raise ConfigError(
                f"materialization must be cartesian|join, got {self.materialization!r}")


Resolution = Union[SLD, RTF, PBC]


@dataclass(frozen=True)
class Backward:
    """Backward-chaining grounder config — ``resolution`` selects sld/rtf/pbc.

    Carries the shared backward knobs; PBC-specific knobs live on ``PBC``. The
    soundness ``filter`` is derived from the resolution unless given explicitly.
    ``resolution`` is positional; every tuning knob is keyword-only.
    """

    resolution: Resolution
    _: dataclasses.KW_ONLY
    max_groundings_per_query: int = 64          # Y_q: collected groundings budget
    pack_dedup: bool = True
    prune_facts: bool = False
    bump_s_to_k: bool = True
    init_state_shape: str = "minimal"
    max_goals: Optional[int] = None             # caps L
    max_states: Optional[int] = None            # caps G (default 256)
    max_children: Optional[int] = None           # caps K; None → 550 (sld/rtf) | max_groundings_per_query (pbc)
    standardization: object = None
    filter: Optional[str] = None                # None → derived from resolution
    hooks: Optional[List] = None
    fact_hook: object = None
    rule_hook: object = None

    @property
    def depth(self) -> int:
        return self.resolution.depth


@dataclass(frozen=True, kw_only=True)
class Forward:
    """Forward chaining grounder (no soundness filter — computes a closure)."""

    depth: int
    method: str = "spmm"            # spmm | staged (ForwardMethod axis)
    join_algo: str = "staged"       # staged | chunked (StagedMethod sub-axis)

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ConfigError(f"depth must be >= 1, got {self.depth}")
        if self.method not in ("spmm", "staged"):
            raise ConfigError(f"method must be 'spmm' or 'staged', got {self.method!r}")
        if self.join_algo not in ("staged", "chunked"):
            raise ConfigError(f"join_algo must be 'staged' or 'chunked', got {self.join_algo!r}")


__all__ = ["SLD", "RTF", "PBC", "Resolution", "Backward", "Forward"]
