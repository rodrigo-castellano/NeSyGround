"""Resolution layer — unification primitives + the three resolution algorithms.

  primitives  — unify_one_to_one, apply_substitutions (leaf MGU)
  standardize — derived-state variable renaming (offset | canonical)
  mgu         — resolve_facts / resolve_rules (shared by sld/rtf)
  sld         — resolve_sld  (fact ∥ rule)
  rtf         — resolve_rtf  (rule → fact cascade)
  pbc         — Parametrized Backward Chaining candidate generation (see pbc/)

The AXIS-1 seam (Resolver Protocol + ResolveRequest) lives in ``api.py``; the
concrete SldResolver/RtfResolver/PbcResolver wrap the function pairs in their own
module. The registry table (RESOLVERS) lives with the family shells (grounder/).
"""
from __future__ import annotations

from grounder.resolution import pbc
from grounder.resolution.api import ResolveRequest, Resolver
from grounder.resolution.mgu import (
    empty_rule_results, init_mgu, resolve_facts, resolve_rules,
)
from grounder.resolution.pbc import PbcResolver, build_plan, init_enum, resolve_step
from grounder.resolution.primitives import apply_substitutions, unify_one_to_one
from grounder.resolution.rtf import RtfResolver, resolve_rtf
from grounder.resolution.sld import SldResolver, resolve_sld
from grounder.resolution.standardize import (
    StandardizationConfig, build_standardize_fn,
    standardize_vars_canonical, standardize_vars_offset,
)

__all__ = [
    "unify_one_to_one", "apply_substitutions",
    "resolve_facts", "resolve_rules", "empty_rule_results", "init_mgu",
    "resolve_sld", "resolve_rtf",
    "pbc", "init_enum", "build_plan", "resolve_step",
    "Resolver", "ResolveRequest", "SldResolver", "RtfResolver", "PbcResolver",
    "StandardizationConfig", "build_standardize_fn",
    "standardize_vars_offset", "standardize_vars_canonical",
]
