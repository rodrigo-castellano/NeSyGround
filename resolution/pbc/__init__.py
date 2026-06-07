"""PBC (Parametrized Backward Chaining) candidate generation.

  compile    — init_enum (build binding tables + budgets) → PbcPlan via build_plan
  candidates — cluster + enumerate (single|cartesian × dense|flat) + fill
  width      — the BC_{w,d,u} width filter + depth_gate
  resolve    — resolve_step driver + Dense/Flat/JoinMaterializer + PbcResolver/JoinResolver

L3 join (RESOLVERS["join"]): a SEMANTICS-PRESERVING sibling of pbc that pushes the
width predicate into the join as a branch pruner — set-equality gated (join_ab.py),
NOT byte-identity. pbc's cartesian path is untouched.
"""
from __future__ import annotations

from grounder.resolution.pbc.compile import PbcPlan, PbcTables, build_plan, init_enum
from grounder.resolution.pbc.resolve import (
    DenseMaterializer, FlatMaterializer, JoinMaterializer, JoinResolver, PbcResolver,
    StepInputs, resolve_step, resolve_step_join,
)

__all__ = [
    "init_enum", "build_plan", "PbcPlan", "PbcTables",
    "resolve_step", "resolve_step_join", "StepInputs",
    "DenseMaterializer", "FlatMaterializer", "JoinMaterializer",
    "PbcResolver", "JoinResolver",
]
