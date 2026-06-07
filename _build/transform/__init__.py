"""AXIS 4 — the ProgramTransform seam (shared; composes with any family).

A ``ProgramTransform`` is a pure ``(kb, req) -> (kb', req')`` rewrite composed by
``make_grounder`` (it wraps the base grounder in a ``core.Pipeline``). Two impls:
``IdentityTransform`` (api.py) and ``MagicSetTransform`` (magic_set.py).
"""
from grounder._build.transform.api import IdentityTransform, ProgramTransform
from grounder._build.transform.magic_set import MagicSetTransform

__all__ = ["ProgramTransform", "IdentityTransform", "MagicSetTransform"]
