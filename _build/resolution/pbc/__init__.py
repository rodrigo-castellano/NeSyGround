"""PBC (Parametrized Backward Chaining) candidate generation.

  compile    — init_enum (build binding tables + budgets) → PbcPlan via build_plan
  candidates — cluster + enumerate (single|cartesian × dense|flat) + fill
  width      — the BC_{w,d,u} width filter + depth_gate
  resolve    — resolve_step driver + Dense/FlatMaterializer (one pipeline, two layouts)

NO L3 join (the lossless+scalable rewrite) — that is a fingerprint-gated follow-up.
"""
from __future__ import annotations

from grounder._build.resolution.pbc.compile import PbcPlan, PbcTables, build_plan, init_enum
from grounder._build.resolution.pbc.resolve import (
    DenseMaterializer, FlatMaterializer, StepInputs, resolve_step,
)

__all__ = [
    "init_enum", "build_plan", "PbcPlan", "PbcTables",
    "resolve_step", "StepInputs", "DenseMaterializer", "FlatMaterializer",
]
