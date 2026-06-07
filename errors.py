"""Typed exceptions for the grounder library.

Lib-wide Technical Rule: bugs propagate as exceptions — never a silent fallback,
clamp, or hidden slow path. Each failure mode is raised by exactly one owner.
"""
from __future__ import annotations


class GrounderError(Exception):
    """Base class for all grounder errors."""


class ConfigError(GrounderError):
    """Invalid/contradictory GrounderConfig (raised in config)."""


class StrategyError(GrounderError):
    """Illegal grounder x execution-strategy combo / undeclared cell
    (raised in execution/capability.validate)."""


class CapMissError(GrounderError):
    """G_r cap exceeded while PBCConfig.strict_cap is set. Default mode
    flags+clamps instead of raising; this is opt-in strict mode only."""


class ShapeContractError(GrounderError):
    """A phase produced a tensor violating its declared shape contract
    (debug aid; not raised in hot paths)."""


__all__ = [
    "GrounderError",
    "ConfigError",
    "StrategyError",
    "CapMissError",
    "ShapeContractError",
]
