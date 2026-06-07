"""Per-grounder configs — the config TYPE selects the grounder.

There is no Resolution enum and no GrounderConfig wrapper: each grounder family
carries exactly its own parameters, and the factory dispatches on config type.

  - ``SLDConfig`` / ``RTFConfig`` — plain backward chaining (params: depth)
  - ``PBCConfig``                 — Parametrized Backward Chaining, the IJCAI
                                    ``BC_{w,d,u}`` grounder (params: depth, w, u);
                                    always uses the fp_batch filter
  - ``FCConfig``                  — forward chaining (params: depth, method, join_algo)

Each backward config knows its own ``filter()`` (PBC→fp_batch, sld/rtf→none), so
there is no separate filter-derivation site. ``all_anchors`` is a PBC invariant
forced in the resolver, not exposed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from grounder._build.errors import ConfigError


class FilterMode(StrEnum):
    NONE = "none"
    FP_BATCH = "fp_batch"   # the ONLY filter; always used by PBC


@dataclass(frozen=True, kw_only=True)
class BackwardConfig:
    """Shared base for backward-chaining grounder configs (SLD/RTF/PBC)."""

    depth: int                              # D
    max_total_groundings: int               # C
    collect_rule_groundings: bool = False   # matches BackwardGrounder ctor default
    standardize: bool = False

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ConfigError(f"depth must be >= 1, got {self.depth}")

    def filter(self) -> FilterMode:
        raise NotImplementedError       # each leaf config declares its filter


@dataclass(frozen=True, kw_only=True)
class SLDConfig(BackwardConfig):
    """Plain backward chaining, SLD resolution (fact + rule)."""

    def filter(self) -> FilterMode:
        return FilterMode.NONE


@dataclass(frozen=True, kw_only=True)
class RTFConfig(BackwardConfig):
    """Plain backward chaining, Rule-Then-Fact resolution."""

    def filter(self) -> FilterMode:
        return FilterMode.NONE


@dataclass(frozen=True, kw_only=True)
class PBCConfig(BackwardConfig):
    """Parametrized Backward Chaining — the IJCAI ``BC_{w,d,u}`` grounder.

    ``depth`` is ``d``; ``w`` is the intermediate-step unknown-fact cap; ``u`` is
    the last-step cap (paper convention 0). Always uses the fp_batch filter.
    """

    w: int
    u: int = 0
    flat_intermediate: bool = True
    max_groundings_per_query: Optional[int] = None  # G_r cap; DEFAULT UNCHANGED → baselines hold
    strict_cap: bool = False                          # opt-in: raise CapMissError on overflow

    def filter(self) -> FilterMode:
        return FilterMode.FP_BATCH


@dataclass(frozen=True, kw_only=True)
class FCConfig:
    """Forward chaining grounder (no soundness filter — computes a closure)."""

    depth: int
    method: str = "spmm"            # spmm | staged (AXIS 3 ForwardMethod)
    join_algo: str = "staged"       # staged | chunked (StagedMethod sub-axis)

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ConfigError(f"depth must be >= 1, got {self.depth}")
        if self.method not in ("spmm", "staged"):
            raise ConfigError(f"method must be 'spmm' or 'staged', got {self.method!r}")
        if self.join_algo not in ("staged", "chunked"):
            raise ConfigError(f"join_algo must be 'staged' or 'chunked', got {self.join_algo!r}")


__all__ = [
    "FilterMode", "BackwardConfig", "SLDConfig", "RTFConfig", "PBCConfig", "FCConfig",
]
