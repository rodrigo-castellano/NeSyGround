"""Resolution layer — unification primitives + the three resolution algorithms.

  primitives  — unify_one_to_one, apply_substitutions (leaf MGU)
  standardize — derived-state variable renaming (offset | canonical)
  mgu         — resolve_facts / resolve_rules (shared by sld/rtf)
  sld         — resolve_sld  (fact ∥ rule)
  rtf         — resolve_rtf  (rule → fact cascade)
  pbc         — Parametrized Backward Chaining candidate generation (see pbc/)

The Resolver classes (the engine↔resolution seam: ResolveRequest → ResolvedChildren)
live with the engine, which owns the working state these functions consume.
"""
from __future__ import annotations

from grounder._build.resolution import pbc
from grounder._build.resolution.mgu import (
    empty_rule_results, init_mgu, resolve_facts, resolve_rules,
)
from grounder._build.resolution.pbc import build_plan, init_enum, resolve_step
from grounder._build.resolution.primitives import apply_substitutions, unify_one_to_one
from grounder._build.resolution.rtf import resolve_rtf
from grounder._build.resolution.sld import resolve_sld
from grounder._build.resolution.standardize import (
    StandardizationConfig, build_standardize_fn,
    standardize_vars_canonical, standardize_vars_offset,
)

__all__ = [
    "unify_one_to_one", "apply_substitutions",
    "resolve_facts", "resolve_rules", "empty_rule_results", "init_mgu",
    "resolve_sld", "resolve_rtf",
    "pbc", "init_enum", "build_plan", "resolve_step",
    "StandardizationConfig", "build_standardize_fn",
    "standardize_vars_offset", "standardize_vars_canonical",
]
